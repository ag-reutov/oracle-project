"""Historical STRATZ league ingestion orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError

from dota_predictor.data.canonical_schema import CanonicalMatchError
from dota_predictor.data.stratz_mapping import (
    CANONICAL_MAPPER_VERSION,
    canonical_match_from_stratz,
)
from dota_predictor.ingestion.client import LeagueMatchesFetcher
from dota_predictor.ingestion.config import DEFAULT_PAGE_SIZE
from dota_predictor.ingestion.cursor import CursorState
from dota_predictor.ingestion.errors import (
    LeagueNotAllowlistedError,
    PageValidationError,
    PaginationDriftError,
    StratzClientError,
)
from dota_predictor.ingestion.page_validation import validate_match_page
from dota_predictor.ingestion.pagination import (
    advance_cursor_after_page,
    empty_page_is_terminal,
    verify_resume_anchor,
)
from dota_predictor.storage.ingestion_writer import (
    ensure_league_ingestion_state,
    get_canonical_mapper_version,
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
from dota_predictor.storage.schema import LEAGUE_INGESTION_STATE
from dota_predictor.storage.writer import write_canonical_match

__all__ = ["IngestionResult", "ingest_league", "ingest_leagues"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    league_id: int
    status: str
    fetch_complete: bool
    matches_seen_count: int
    canonicalization_failures: int
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
) -> IngestionResult:
    with engine.begin() as conn:
        if not is_league_allowlisted(conn, league_id):
            raise LeagueNotAllowlistedError(
                f"League {league_id} is not in ingestion_leagues"
            )
        ensure_league_ingestion_state(conn, league_id)

    cursor = _load_cursor(engine, league_id)
    matches_seen_count = _load_matches_seen_count(engine, league_id)

    if not cursor.fetch_complete:
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
