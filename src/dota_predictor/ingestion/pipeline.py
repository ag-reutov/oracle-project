"""Historical STRATZ league ingestion orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.exc import SQLAlchemyError

from dota_predictor.data.canonical_schema import CanonicalMatchError
from dota_predictor.data.stratz_mapping import (
    CANONICAL_MAPPER_VERSION,
    canonical_match_from_stratz,
)
from dota_predictor.ingestion.client import LeagueMatchesFetcher, MatchByIdFetcher
from dota_predictor.ingestion.config import DEFAULT_PAGE_SIZE
from dota_predictor.ingestion.cursor import MATCH_IDS_CURSOR_MODE, CursorState
from dota_predictor.ingestion.discovery import (
    dedupe_match_ids,
    discover_league_match_ids,
)
from dota_predictor.ingestion.errors import (
    LeagueFetchModeError,
    LeagueNotAllowlistedError,
    PageValidationError,
    PaginationDriftError,
    StratzClientError,
)
from dota_predictor.ingestion.page_validation import (
    validate_match_belongs_to_league,
    validate_match_page,
)
from dota_predictor.ingestion.pagination import (
    advance_cursor_after_page,
    empty_page_is_terminal,
    verify_resume_anchor,
)
from dota_predictor.storage.ingestion_writer import (
    ensure_league_allowlisted,
    ensure_league_ingestion_state,
    get_canonical_mapper_version,
    get_league_fetch_mode,
    get_matches_seen_count,
    get_persisted_match_ids_for_league,
    insert_match_ingestion_error,
    is_league_allowlisted,
    league_canonicalization_complete,
    list_allowlisted_league_ids,
    list_raw_matches_for_league,
    load_cursor_state,
    persist_raw_page,
    update_league_ingestion_state,
)
from dota_predictor.storage.schema import (
    LEAGUE_FETCH_MODE_MATCH_IDS,
    LEAGUE_INGESTION_STATE,
    MATCHES,
)
from dota_predictor.storage.writer import write_canonical_match

__all__ = [
    "IngestionResult",
    "MatchIdIngestionResult",
    "ingest_league",
    "ingest_leagues",
    "ingest_matches_by_id",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    league_id: int
    status: str
    fetch_complete: bool
    matches_seen_count: int
    canonicalization_failures: int
    message: str | None = None


@dataclass(frozen=True)
class MatchIdIngestionResult:
    """Outcome of one `ingest_matches_by_id` run."""

    league_id: int
    status: str
    match_ids_unique: int
    fetch_successes: int
    fetch_failures: int
    league_id_mismatches: int
    skipped_already_raw: int
    raw_row_count: int
    raw_rows_before: int
    canonical_row_count: int
    canonical_already_current: int
    canonicalization_failures: int
    fetch_complete: bool
    min_start_time: datetime | None
    max_start_time: datetime | None
    all_canonical_league_ids_match: bool
    message: str | None = None


def ingest_leagues(
    engine: Engine,
    fetcher: LeagueMatchesFetcher,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[IngestionResult]:
    with engine.connect() as conn:
        league_ids = list_allowlisted_league_ids(conn)
    return [
        ingest_league(engine, fetcher, league_id, page_size=page_size)
        for league_id in league_ids
    ]


def ingest_league(
    engine: Engine,
    fetcher: LeagueMatchesFetcher,
    league_id: int,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    match_ids: Sequence[int] | None = None,
) -> IngestionResult:
    with engine.begin() as conn:
        if not is_league_allowlisted(conn, league_id):
            raise LeagueNotAllowlistedError(
                f"League {league_id} is not in ingestion_leagues"
            )
        ensure_league_ingestion_state(conn, league_id)
        fetch_mode = get_league_fetch_mode(conn, league_id)

    # Registry fetch_mode is authoritative. Do not wait for a prior
    # match-id run to stamp cursor.mode — `--all` on a fresh state must
    # not call `league(id)` for catalog-null leagues.
    if fetch_mode == LEAGUE_FETCH_MODE_MATCH_IDS:
        return _ingest_league_match_ids(engine, fetcher, league_id, match_ids)

    cursor = _load_cursor(engine, league_id)
    matches_seen_count = _load_matches_seen_count(engine, league_id)

    # Defense in depth: a leftover match_ids cursor still skips league(id).
    if cursor.mode != MATCH_IDS_CURSOR_MODE and not cursor.fetch_complete:
        try:
            cursor, matches_seen_count = _acquire_raw_pages(
                engine,
                fetcher,
                league_id,
                cursor,
                matches_seen_count,
                page_size=page_size,
            )
        except PaginationDriftError as exc:
            cursor, matches_seen_count = _reload_persisted_pagination_state(
                engine, league_id
            )
            _mark_league_error(engine, league_id, cursor, matches_seen_count, str(exc))
            return IngestionResult(
                league_id=league_id,
                status="ERROR",
                fetch_complete=cursor.fetch_complete,
                matches_seen_count=matches_seen_count,
                canonicalization_failures=0,
                message=str(exc),
            )
        except (PageValidationError, StratzClientError, SQLAlchemyError) as exc:
            cursor, matches_seen_count = _reload_persisted_pagination_state(
                engine, league_id
            )
            _mark_league_error(engine, league_id, cursor, matches_seen_count, str(exc))
            return IngestionResult(
                league_id=league_id,
                status="ERROR",
                fetch_complete=cursor.fetch_complete,
                matches_seen_count=matches_seen_count,
                canonicalization_failures=0,
                message=str(exc),
            )

    canonical_failures = _canonicalize_league_raw_matches(engine, league_id)
    final_status = _finalize_league_status(engine, league_id, canonical_failures)

    with engine.connect() as conn:
        row = conn.execute(
            select(
                LEAGUE_INGESTION_STATE.c.status,
                LEAGUE_INGESTION_STATE.c.matches_seen_count,
                LEAGUE_INGESTION_STATE.c.cursor_state,
            ).where(LEAGUE_INGESTION_STATE.c.league_id == league_id)
        ).one()
        cursor = load_cursor_state(conn, league_id)

    return IngestionResult(
        league_id=league_id,
        status=str(row.status),
        fetch_complete=bool(cursor.fetch_complete),
        matches_seen_count=int(row.matches_seen_count),
        canonicalization_failures=canonical_failures,
        message=None if final_status == "COMPLETE" else "canonicalization incomplete",
    )


def _require_match_by_id_fetcher(fetcher: object, league_id: int) -> MatchByIdFetcher:
    fetch_match = getattr(fetcher, "fetch_match", None)
    if not callable(fetch_match):
        raise LeagueFetchModeError(
            f"League {league_id} has fetch_mode=match_ids; refusing to call "
            "league(id). The fetcher must implement fetch_match."
        )
    return fetcher  # type: ignore[return-value]


def _ingest_league_match_ids(
    engine: Engine,
    fetcher: object,
    league_id: int,
    match_ids: Sequence[int] | None,
) -> IngestionResult:
    match_fetcher = _require_match_by_id_fetcher(fetcher, league_id)
    if match_ids is None:
        discovery = discover_league_match_ids(league_id, skip_team_walk=True)
        for note in discovery.notes:
            logger.info("league %s ID discovery: %s", league_id, note)
        match_ids = discovery.match_ids
    result = ingest_matches_by_id(engine, match_fetcher, league_id, match_ids)
    return IngestionResult(
        league_id=result.league_id,
        status=result.status,
        fetch_complete=result.fetch_complete,
        matches_seen_count=result.match_ids_unique,
        canonicalization_failures=result.canonicalization_failures,
        message=result.message,
    )


def ingest_matches_by_id(
    engine: Engine,
    fetcher: MatchByIdFetcher,
    league_id: int,
    match_ids: Sequence[int],
) -> MatchIdIngestionResult:
    """Fetch STRATZ `match(id)` payloads and reuse raw/canonical writers.

    `match_ids` may contain duplicates; they are dropped while preserving
    order. Already-persisted raw rows are not re-fetched. Nested `league`
    metadata may be null. A returned `leagueId` that is missing or not
    equal to `league_id` is rejected and recorded as a FETCH error.
    """
    unique_ids = dedupe_match_ids(match_ids)

    with engine.begin() as conn:
        ensure_league_allowlisted(conn, league_id)
        ensure_league_ingestion_state(conn, league_id)
        cursor = load_cursor_state(conn, league_id)
        cursor = CursorState(
            next_skip=cursor.next_skip,
            take=cursor.take or DEFAULT_PAGE_SIZE,
            last_match_id=cursor.last_match_id,
            last_start_date_time=cursor.last_start_date_time,
            fetch_complete=False,
            mode=MATCH_IDS_CURSOR_MODE,
        )
        update_league_ingestion_state(
            conn,
            league_id=league_id,
            status="IN_PROGRESS",
            cursor=cursor,
            matches_seen_count=len(unique_ids),
        )

    with engine.connect() as conn:
        persisted = get_persisted_match_ids_for_league(conn, league_id)
    raw_rows_before = len(persisted)

    fetch_successes = 0
    fetch_failures = 0
    league_id_mismatches = 0
    skipped_already_raw = 0

    for match_id in unique_ids:
        if match_id in persisted:
            skipped_already_raw += 1
            continue
        try:
            payload = fetcher.fetch_match(match_id)
        except StratzClientError as exc:
            fetch_failures += 1
            _record_fetch_error(engine, match_id, league_id, str(exc))
            continue
        if payload is None:
            fetch_failures += 1
            _record_fetch_error(
                engine,
                match_id,
                league_id,
                f"STRATZ match({match_id}) returned null",
            )
            continue
        try:
            validate_match_belongs_to_league(payload, league_id)
        except PageValidationError as exc:
            league_id_mismatches += 1
            fetch_failures += 1
            _record_fetch_error(engine, match_id, league_id, str(exc), payload)
            continue

        fetched_at = datetime.now(UTC)
        try:
            with engine.begin() as conn:
                persist_raw_page(
                    conn,
                    league_id=league_id,
                    matches=[payload],
                    fetched_at=fetched_at,
                )
        except SQLAlchemyError as exc:
            fetch_failures += 1
            with engine.begin() as conn:
                insert_match_ingestion_error(
                    conn,
                    match_id=match_id,
                    league_id=league_id,
                    stage="WRITE",
                    error_message=str(exc),
                    raw_payload_snapshot=payload,
                )
            continue
        persisted.add(match_id)
        fetch_successes += 1

    with engine.connect() as conn:
        raw_ids = get_persisted_match_ids_for_league(conn, league_id)
        canonical_already_current = 0
        for match_id in unique_ids:
            version = get_canonical_mapper_version(conn, match_id)
            if version is not None and version >= CANONICAL_MAPPER_VERSION:
                canonical_already_current += 1

    fetch_complete = set(unique_ids) <= raw_ids
    last_match_id = unique_ids[-1] if unique_ids else None
    cursor = CursorState(
        next_skip=0,
        take=DEFAULT_PAGE_SIZE,
        last_match_id=last_match_id,
        last_start_date_time=None,
        fetch_complete=fetch_complete,
        mode=MATCH_IDS_CURSOR_MODE,
    )
    with engine.begin() as conn:
        update_league_ingestion_state(
            conn,
            league_id=league_id,
            status="IN_PROGRESS",
            cursor=cursor,
            matches_seen_count=len(unique_ids),
        )

    canonical_failures = _canonicalize_league_raw_matches(engine, league_id)
    final_status = _finalize_league_status(engine, league_id, canonical_failures)

    with engine.connect() as conn:
        raw_row_count = len(get_persisted_match_ids_for_league(conn, league_id))
        canonical_row_count = int(
            conn.execute(
                select(func.count()).select_from(MATCHES).where(
                    MATCHES.c.league_id == league_id
                )
            ).scalar_one()
        )
        min_start, max_start = conn.execute(
            select(func.min(MATCHES.c.start_time), func.max(MATCHES.c.start_time)).where(
                MATCHES.c.league_id == league_id
            )
        ).one()
        if unique_ids:
            league_ids_for_ids = conn.execute(
                select(MATCHES.c.league_id).where(MATCHES.c.match_id.in_(unique_ids))
            ).all()
            all_match = all(
                int(row.league_id) == league_id for row in league_ids_for_ids
            )
        else:
            all_match = True

    message = None
    if final_status != "COMPLETE":
        message = "canonicalization incomplete"
        if not fetch_complete:
            message = "fetch incomplete"
            if canonical_failures:
                message = "fetch incomplete; canonicalization incomplete"

    return MatchIdIngestionResult(
        league_id=league_id,
        status=str(final_status),
        match_ids_unique=len(unique_ids),
        fetch_successes=fetch_successes,
        fetch_failures=fetch_failures,
        league_id_mismatches=league_id_mismatches,
        skipped_already_raw=skipped_already_raw,
        raw_row_count=raw_row_count,
        raw_rows_before=raw_rows_before,
        canonical_row_count=canonical_row_count,
        canonical_already_current=canonical_already_current,
        canonicalization_failures=canonical_failures,
        fetch_complete=fetch_complete,
        min_start_time=min_start,
        max_start_time=max_start,
        all_canonical_league_ids_match=bool(all_match),
        message=message,
    )


def _record_fetch_error(
    engine: Engine,
    match_id: int,
    league_id: int,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    with engine.begin() as conn:
        insert_match_ingestion_error(
            conn,
            match_id=match_id,
            league_id=league_id,
            stage="FETCH",
            error_message=message,
            raw_payload_snapshot=payload,
        )


def _load_cursor(engine: Engine, league_id: int) -> CursorState:
    with engine.connect() as conn:
        return load_cursor_state(conn, league_id)


def _load_matches_seen_count(engine: Engine, league_id: int) -> int:
    with engine.connect() as conn:
        return get_matches_seen_count(conn, league_id)


def _reload_persisted_pagination_state(
    engine: Engine, league_id: int
) -> tuple[CursorState, int]:
    """Re-read the cursor/count checkpoint most recently committed to
    `league_ingestion_state`.

    `_acquire_raw_pages` commits its cursor/count checkpoint together with
    each page's raw rows (see its per-page `engine.begin()` blocks), so
    that persisted state is authoritative the moment it lands -- including
    when a later page's fetch/validation fails. The pre-call `cursor`/
    `matches_seen_count` values still held by `ingest_league` at that point
    are stale (the `_acquire_raw_pages(...)` call raised before its
    `cursor, matches_seen_count = ...` assignment could complete), so the
    error path must not pass those stale values into `_mark_league_error`
    -- doing so would roll back a checkpoint that was already durably
    committed for one or more successfully processed pages.
    """
    return _load_cursor(engine, league_id), _load_matches_seen_count(engine, league_id)


def _mark_league_error(
    engine: Engine,
    league_id: int,
    cursor: CursorState,
    matches_seen_count: int,
    message: str,
) -> None:
    with engine.begin() as conn:
        update_league_ingestion_state(
            conn,
            league_id=league_id,
            status="ERROR",
            cursor=cursor,
            matches_seen_count=matches_seen_count,
            last_error=message,
            increment_error_count=1,
        )


def _verify_resume_anchor_fetch(
    fetcher: LeagueMatchesFetcher,
    league_id: int,
    cursor: CursorState,
) -> None:
    if cursor.next_skip <= 0 or cursor.fetch_complete:
        return
    anchor_page = fetcher.fetch_league_matches_page(
        league_id,
        skip=cursor.next_skip - 1,
        take=1,
    )
    anchor_match = anchor_page[0] if anchor_page else None
    verify_resume_anchor(anchor_match, cursor)


def _acquire_raw_pages(
    engine: Engine,
    fetcher: LeagueMatchesFetcher,
    league_id: int,
    cursor: CursorState,
    matches_seen_count: int,
    *,
    page_size: int,
) -> tuple[CursorState, int]:
    _verify_resume_anchor_fetch(fetcher, league_id, cursor)

    with engine.connect() as conn:
        persisted_ids = get_persisted_match_ids_for_league(conn, league_id)

    with engine.begin() as conn:
        update_league_ingestion_state(
            conn,
            league_id=league_id,
            status="IN_PROGRESS",
            cursor=cursor,
            matches_seen_count=matches_seen_count,
        )

    while not cursor.fetch_complete:
        matches = fetcher.fetch_league_matches_page(
            league_id,
            skip=cursor.next_skip,
            take=page_size,
        )

        if empty_page_is_terminal(len(matches)):
            with engine.begin() as conn:
                cursor = CursorState(
                    next_skip=cursor.next_skip,
                    take=page_size,
                    last_match_id=cursor.last_match_id,
                    last_start_date_time=cursor.last_start_date_time,
                    fetch_complete=True,
                )
                update_league_ingestion_state(
                    conn,
                    league_id=league_id,
                    status="IN_PROGRESS",
                    cursor=cursor,
                    matches_seen_count=matches_seen_count,
                )
            break

        validate_match_page(
            matches,
            league_id=league_id,
            persisted_match_ids=persisted_ids,
        )

        fetched_at = datetime.now(UTC)
        cursor = advance_cursor_after_page(cursor, matches, page_size=page_size)
        matches_seen_count += len(matches)
        persisted_ids.update(int(m["id"]) for m in matches)

        with engine.begin() as conn:
            persist_raw_page(
                conn,
                league_id=league_id,
                matches=matches,
                fetched_at=fetched_at,
            )
            update_league_ingestion_state(
                conn,
                league_id=league_id,
                status="IN_PROGRESS",
                cursor=cursor,
                matches_seen_count=matches_seen_count,
            )

    return cursor, matches_seen_count


def _canonicalize_league_raw_matches(engine: Engine, league_id: int) -> int:
    with engine.connect() as conn:
        raw_rows = list_raw_matches_for_league(conn, league_id)

    failures = 0
    for row in raw_rows:
        match_id = int(row["match_id"])
        payload: dict[str, Any] = row["payload"]

        with engine.connect() as conn:
            current_version = get_canonical_mapper_version(conn, match_id)
        if current_version is not None and current_version >= CANONICAL_MAPPER_VERSION:
            continue

        try:
            canonical = canonical_match_from_stratz(payload)
        except CanonicalMatchError as exc:
            failures += 1
            with engine.begin() as conn:
                insert_match_ingestion_error(
                    conn,
                    match_id=match_id,
                    league_id=league_id,
                    stage="MAP",
                    error_message=str(exc),
                    raw_payload_snapshot=payload,
                )
            continue

        try:
            write_canonical_match(engine, canonical)
        except Exception as exc:  # noqa: BLE001 - surface DB failures as ingestion errors
            failures += 1
            with engine.begin() as conn:
                insert_match_ingestion_error(
                    conn,
                    match_id=match_id,
                    league_id=league_id,
                    stage="WRITE",
                    error_message=str(exc),
                    raw_payload_snapshot=payload,
                )

    return failures


def _finalize_league_status(
    engine: Engine,
    league_id: int,
    canonical_failures: int,
) -> str:
    with engine.begin() as conn:
        cursor = load_cursor_state(conn, league_id)
        matches_seen_count = get_matches_seen_count(conn, league_id)
        complete = league_canonicalization_complete(conn, league_id)
        if cursor.fetch_complete and complete and canonical_failures == 0:
            status = "COMPLETE"
            last_error = None
        else:
            status = "ERROR"
            last_error = (
                f"canonicalization failures in last run: {canonical_failures}"
                if canonical_failures
                else "canonicalization incomplete"
            )
        update_league_ingestion_state(
            conn,
            league_id=league_id,
            status=status,
            cursor=cursor,
            matches_seen_count=matches_seen_count,
            last_error=last_error,
        )
    return status
