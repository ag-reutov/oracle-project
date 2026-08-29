"""Pipeline tests with mocked STRATZ fetcher and optional Postgres integration."""

from __future__ import annotations

from typing import Any

import pytest
from ingestion_helpers import (
    build_stratz_match,
    requires_test_database,
    seed_ingestion_league,
)
from sqlalchemy import select

from dota_predictor.data.stratz_mapping import CANONICAL_MAPPER_VERSION
from dota_predictor.ingestion.errors import LeagueNotAllowlistedError, StratzClientError
from dota_predictor.ingestion.pipeline import ingest_league
from dota_predictor.storage.ingestion_writer import load_cursor_state
from dota_predictor.storage.schema import (
    LEAGUE_INGESTION_STATE,
    MATCH_INGESTION_ERRORS,
    MATCHES,
    STRATZ_RAW_MATCHES,
)

LEAGUE_ID = 16935


class MockFetcher:
    def __init__(self, pages: dict[tuple[int, int], list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.calls: list[tuple[int, int, int]] = []

    def fetch_league_matches_page(
        self,
        league_id: int,
        *,
        skip: int,
        take: int,
    ) -> list[dict[str, Any]]:
        self.calls.append((league_id, skip, take))
        return list(self.pages.get((skip, take), []))


def _match(match_id: int, start: int) -> dict[str, Any]:
    return build_stratz_match(
        match_id=match_id,
        league_id=LEAGUE_ID,
        start_date_time=start,
    )


@requires_test_database
def test_page_advancement_and_short_page_termination(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    pages = {
        (0, 100): [_match(1000 - i, 1000 - i) for i in range(50)],
    }
    fetcher = MockFetcher(pages)
    result = ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)

    assert result.status == "COMPLETE"
    assert result.fetch_complete is True
    assert result.matches_seen_count == 50

    with engine.connect() as conn:
        cursor = load_cursor_state(conn, LEAGUE_ID)
        assert cursor.next_skip == 50
        assert cursor.fetch_complete is True
        raw_count = conn.execute(
            select(STRATZ_RAW_MATCHES.c.match_id).where(
                STRATZ_RAW_MATCHES.c.league_id == LEAGUE_ID
            )
        ).all()
    assert len(raw_count) == 50


@requires_test_database
def test_exact_multiple_of_100_requires_empty_terminal_page(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    pages = {
        (0, 100): [_match(200 - i, 200 - i) for i in range(100)],
        (100, 100): [_match(100 - i, 100 - i) for i in range(100)],
        (200, 100): [],
    }
    fetcher = MockFetcher(pages)
    result = ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)

    assert result.status == "COMPLETE"
    assert result.matches_seen_count == 200
    assert (200, 100) in {(c[1], c[2]) for c in fetcher.calls}


@requires_test_database
def test_checkpoint_not_advanced_on_fetch_failure(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    first_page = [_match(i, 1000 - i) for i in range(1, 101)]

    class FailingFetcher:
        def fetch_league_matches_page(
            self, league_id: int, *, skip: int, take: int
        ) -> list[dict[str, Any]]:
            if skip == 0:
                return first_page
            raise StratzClientError("simulated fetch failure")

    fetcher = FailingFetcher()
    result = ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)

    assert result.status == "ERROR"
    # The checkpoint for the successfully persisted first page (100 matches,
    # next_skip 0 -> 100) must survive the second page's fetch failure --
    # not be rolled back to the pre-acquisition state (next_skip=0).
    assert result.matches_seen_count == 100
    with engine.connect() as conn:
        cursor = load_cursor_state(conn, LEAGUE_ID)
        assert cursor.next_skip == 100
        assert cursor.fetch_complete is False
        row = conn.execute(
            select(LEAGUE_INGESTION_STATE.c.matches_seen_count).where(
                LEAGUE_INGESTION_STATE.c.league_id == LEAGUE_ID
            )
        ).one()
        assert int(row.matches_seen_count) == 100
        raw_ids = conn.execute(
            select(STRATZ_RAW_MATCHES.c.match_id).where(
                STRATZ_RAW_MATCHES.c.league_id == LEAGUE_ID
            )
        ).all()
        assert len(raw_ids) == 100

    # A resume attempt should therefore start from the committed checkpoint
    # (skip=100), not re-fetch the already-persisted first page. The resume
    # anchor check (skip=99, take=1) must see the same last match persisted
    # by the successful first page.
    resume_fetcher = MockFetcher({(99, 1): [first_page[-1]], (100, 100): []})
    resumed = ingest_league(engine, resume_fetcher, LEAGUE_ID, page_size=100)
    assert resumed.status == "COMPLETE"
    assert resumed.matches_seen_count == 100
    assert (100, 100) in {(c[1], c[2]) for c in resume_fetcher.calls}
    assert (0, 100) not in {(c[1], c[2]) for c in resume_fetcher.calls}


@requires_test_database
def test_checkpoint_not_advanced_when_raw_write_fails(engine, monkeypatch) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    pages = {(0, 100): [_match(1, 100)]}
    fetcher = MockFetcher(pages)

    def boom(*args: object, **kwargs: object) -> None:
        from sqlalchemy.exc import SQLAlchemyError

        raise SQLAlchemyError("db write failed")

    monkeypatch.setattr(
        "dota_predictor.ingestion.pipeline.persist_raw_page",
        boom,
    )
    result = ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)
    assert result.status == "ERROR"

    with engine.connect() as conn:
        cursor = load_cursor_state(conn, LEAGUE_ID)
        assert cursor.next_skip == 0
        assert conn.execute(select(STRATZ_RAW_MATCHES)).first() is None


@requires_test_database
def test_resume_anchor_verification(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    page1 = [_match(i, 1000 - i) for i in range(1, 6)]
    fetcher = MockFetcher({(0, 100): page1})
    ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)

    with engine.begin() as conn:
        from dota_predictor.ingestion.cursor import CursorState
        from dota_predictor.storage.ingestion_writer import (
            update_league_ingestion_state,
        )

        cursor = CursorState(
            next_skip=5,
            take=100,
            last_match_id=5,
            last_start_date_time=996,
            fetch_complete=False,
        )
        update_league_ingestion_state(
            conn,
            league_id=LEAGUE_ID,
            status="IN_PROGRESS",
            cursor=cursor,
            matches_seen_count=5,
        )

    resume_fetcher = MockFetcher({(4, 1): [_match(5, 996)], (5, 100): [_match(4, 997)]})
    result = ingest_league(engine, resume_fetcher, LEAGUE_ID, page_size=100)
    assert result.status == "COMPLETE"
    assert (4, 1) in {(c[1], c[2]) for c in resume_fetcher.calls}


@requires_test_database
def test_anchor_mismatch_marks_pagination_drift(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    with engine.begin() as conn:
        from dota_predictor.ingestion.cursor import CursorState
        from dota_predictor.storage.ingestion_writer import (
            ensure_league_ingestion_state,
            update_league_ingestion_state,
        )

        ensure_league_ingestion_state(conn, LEAGUE_ID)
        cursor = CursorState(
            next_skip=10,
            take=100,
            last_match_id=999,
            fetch_complete=False,
        )
        update_league_ingestion_state(
            conn,
            league_id=LEAGUE_ID,
            status="IN_PROGRESS",
            cursor=cursor,
            matches_seen_count=10,
        )

    fetcher = MockFetcher({(9, 1): [_match(1, 100)]})
    result = ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)
    assert result.status == "ERROR"
    assert result.message is not None
    assert "drift" in result.message.lower()

    with engine.connect() as conn:
        cursor = load_cursor_state(conn, LEAGUE_ID)
        assert cursor.next_skip == 10


@requires_test_database
def test_mapping_failure_preserves_raw_and_continues(engine, monkeypatch) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    good = _match(1, 100)
    bad = _match(2, 90)
    fetcher = MockFetcher({(0, 100): [good, bad]})

    def flaky_map(raw: dict[str, Any]) -> Any:
        if int(raw["id"]) == 2:
            from dota_predictor.data.canonical_schema import CanonicalMatchError

            raise CanonicalMatchError("bad payload")
        from dota_predictor.data.stratz_mapping import canonical_match_from_stratz

        return canonical_match_from_stratz(raw)

    monkeypatch.setattr(
        "dota_predictor.ingestion.pipeline.canonical_match_from_stratz",
        flaky_map,
    )
    result = ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)

    assert result.status == "ERROR"
    assert result.fetch_complete is True
    with engine.connect() as conn:
        raw_rows = conn.execute(
            select(STRATZ_RAW_MATCHES.c.match_id).where(
                STRATZ_RAW_MATCHES.c.league_id == LEAGUE_ID
            )
        ).all()
        assert len(raw_rows) == 2
        errors = conn.execute(
            select(MATCH_INGESTION_ERRORS.c.match_id, MATCH_INGESTION_ERRORS.c.stage)
        ).all()
        assert (2, "MAP") in {(int(r.match_id), r.stage) for r in errors}
        canonical = conn.execute(select(MATCHES.c.match_id)).all()
        assert len(canonical) == 1


@requires_test_database
def test_rerun_completed_league_is_idempotent(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    pages = {(0, 100): [_match(1, 100)]}
    fetcher = MockFetcher(pages)
    first = ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)
    assert first.status == "COMPLETE"

    second = ingest_league(engine, MockFetcher({}), LEAGUE_ID, page_size=100)
    assert second.status == "COMPLETE"
    assert second.matches_seen_count == 1


@requires_test_database
def test_recanonicalize_older_mapper_version(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    raw = _match(1, 100)
    fetcher = MockFetcher({(0, 100): [raw]})
    ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)

    with engine.begin() as conn:
        row = conn.execute(
            select(MATCHES.c.mapper_version).where(MATCHES.c.match_id == 1)
        ).one()
        assert int(row.mapper_version) == CANONICAL_MAPPER_VERSION

        conn.execute(
            MATCHES.update()
            .where(MATCHES.c.match_id == 1)
            .values(mapper_version=CANONICAL_MAPPER_VERSION - 1)
        )

    result = ingest_league(engine, MockFetcher({}), LEAGUE_ID, page_size=100)
    assert result.status == "COMPLETE"
    with engine.connect() as conn:
        row = conn.execute(
            select(MATCHES.c.mapper_version).where(MATCHES.c.match_id == 1)
        ).one()
        assert int(row.mapper_version) == CANONICAL_MAPPER_VERSION


@requires_test_database
def test_refuses_non_allowlisted_league(engine) -> None:
    fetcher = MockFetcher({})
    with pytest.raises(LeagueNotAllowlistedError):
        ingest_league(engine, fetcher, 99999, page_size=100)
