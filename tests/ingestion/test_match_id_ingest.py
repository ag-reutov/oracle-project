"""Tests for match-id STRATZ ingest (reuses raw/canonical writers)."""

from __future__ import annotations

from typing import Any

import pytest
from ingestion_helpers import (
    build_stratz_match,
    requires_test_database,
    seed_ingestion_league,
)
from sqlalchemy import select

from dota_predictor.ingestion.cursor import MATCH_IDS_CURSOR_MODE
from dota_predictor.ingestion.errors import LeagueNotRegisteredError
from dota_predictor.ingestion.pipeline import ingest_league, ingest_matches_by_id
from dota_predictor.storage.ingestion_writer import load_cursor_state
from dota_predictor.storage.schema import (
    LEAGUES,
    MATCH_INGESTION_ERRORS,
    MATCHES,
    MATCH_PLAYERS,
    DRAFT_EVENTS,
    STRATZ_RAW_MATCHES,
)

LEAGUE_ID = 17419


class MockMatchFetcher:
    def __init__(self, matches: dict[int, dict[str, Any] | None]) -> None:
        self.matches = matches
        self.calls: list[int] = []

    def fetch_match(self, match_id: int) -> dict[str, Any] | None:
        self.calls.append(match_id)
        if match_id not in self.matches:
            return None
        payload = self.matches[match_id]
        if payload is None:
            return None
        return dict(payload)


def _match(match_id: int, start: int, *, league: dict | None | bool = True) -> dict[str, Any]:
    raw = build_stratz_match(
        match_id=match_id,
        league_id=LEAGUE_ID,
        start_date_time=start,
    )
    if league is True:
        return raw
    raw["league"] = None if league is False or league is None else league
    return raw


@requires_test_database
def test_null_nested_league_is_accepted_when_league_id_present(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID, name="SLAM IV")

    payload = _match(100, 1_760_000_000, league=None)
    assert payload["league"] is None
    fetcher = MockMatchFetcher({100: payload})
    result = ingest_matches_by_id(engine, fetcher, LEAGUE_ID, [100])

    assert result.status == "COMPLETE"
    assert result.fetch_successes == 1
    with engine.connect() as conn:
        row = conn.execute(
            select(MATCHES.c.league_id, MATCHES.c.league_name).where(
                MATCHES.c.match_id == 100
            )
        ).one()
        assert int(row.league_id) == LEAGUE_ID
        assert row.league_name is None


@requires_test_database
def test_match_id_ingest_uses_existing_raw_and_canonical_tables(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID, name="SLAM IV")

    payload = _match(200, 1_760_000_100)
    result = ingest_matches_by_id(
        engine, MockMatchFetcher({200: payload}), LEAGUE_ID, [200]
    )
    assert result.status == "COMPLETE"
    assert result.raw_row_count == 1
    assert result.canonical_row_count == 1
    assert result.all_canonical_league_ids_match is True

    with engine.connect() as conn:
        raw = conn.execute(
            select(STRATZ_RAW_MATCHES.c.match_id, STRATZ_RAW_MATCHES.c.league_id)
        ).one()
        assert int(raw.match_id) == 200
        assert int(raw.league_id) == LEAGUE_ID
        players = conn.execute(
            select(MATCH_PLAYERS.c.player_id).where(MATCH_PLAYERS.c.match_id == 200)
        ).all()
        drafts = conn.execute(
            select(DRAFT_EVENTS.c.sequence).where(DRAFT_EVENTS.c.match_id == 200)
        ).all()
        assert len(players) == 10
        assert len(drafts) == 17


@requires_test_database
def test_duplicate_input_ids_are_fetched_once(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID, name="SLAM IV")

    payload = _match(300, 1_760_000_200)
    fetcher = MockMatchFetcher({300: payload})
    result = ingest_matches_by_id(
        engine, fetcher, LEAGUE_ID, [300, 300, 300]
    )
    assert result.match_ids_unique == 1
    assert fetcher.calls == [300]
    assert result.raw_row_count == 1


@requires_test_database
def test_rerun_is_idempotent(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID, name="SLAM IV")

    payload = _match(400, 1_760_000_300)
    first = ingest_matches_by_id(
        engine, MockMatchFetcher({400: payload}), LEAGUE_ID, [400]
    )
    assert first.status == "COMPLETE"
    assert first.fetch_successes == 1
    assert first.skipped_already_raw == 0

    second_fetcher = MockMatchFetcher({400: payload})
    second = ingest_matches_by_id(engine, second_fetcher, LEAGUE_ID, [400])
    assert second.status == "COMPLETE"
    assert second.fetch_successes == 0
    assert second.skipped_already_raw == 1
    assert second.raw_row_count == first.raw_row_count == 1
    assert second.canonical_row_count == first.canonical_row_count == 1
    assert second.canonical_already_current == 1
    assert second_fetcher.calls == []

    with engine.connect() as conn:
        raw_n = conn.execute(select(STRATZ_RAW_MATCHES.c.match_id)).all()
        can_n = conn.execute(select(MATCHES.c.match_id)).all()
    assert len(raw_n) == 1
    assert len(can_n) == 1


@requires_test_database
def test_mismatched_league_id_is_rejected(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID, name="SLAM IV")

    payload = _match(500, 1_760_000_400)
    payload["leagueId"] = 18324
    result = ingest_matches_by_id(
        engine, MockMatchFetcher({500: payload}), LEAGUE_ID, [500]
    )
    assert result.status == "ERROR"
    assert result.league_id_mismatches == 1
    assert result.fetch_failures == 1
    assert result.raw_row_count == 0
    assert result.canonical_row_count == 0

    with engine.connect() as conn:
        assert conn.execute(select(STRATZ_RAW_MATCHES)).first() is None
        err = conn.execute(
            select(MATCH_INGESTION_ERRORS.c.stage, MATCH_INGESTION_ERRORS.c.match_id)
        ).one()
        assert err.stage == "FETCH"
        assert int(err.match_id) == 500


@requires_test_database
def test_ingest_league_does_not_page_match_id_mode(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID, name="SLAM IV")

    payload = _match(600, 1_760_000_500)
    ingest_matches_by_id(engine, MockMatchFetcher({600: payload}), LEAGUE_ID, [600])

    class BoomFetcher:
        def fetch_league_matches_page(self, *args: object, **kwargs: object) -> list:
            raise AssertionError("league(id) pagination must not run")

    result = ingest_league(engine, BoomFetcher(), LEAGUE_ID, page_size=100)
    assert result.status == "COMPLETE"
    with engine.connect() as conn:
        cursor = load_cursor_state(conn, LEAGUE_ID)
        assert cursor.mode == MATCH_IDS_CURSOR_MODE


@requires_test_database
def test_allowlist_requires_leagues_registry_row(engine) -> None:
    payload = _match(700, 1_760_000_600)
    with pytest.raises(LeagueNotRegisteredError):
        ingest_matches_by_id(
            engine, MockMatchFetcher({700: payload}), LEAGUE_ID, [700]
        )


@requires_test_database
def test_allowlist_inserts_ingestion_leagues_when_registered(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            LEAGUES.insert().values(
                league_id=LEAGUE_ID,
                name="SLAM IV",
                liquipedia_tier="T1",
                in_scope=False,
            )
        )

    payload = _match(800, 1_760_000_700)
    result = ingest_matches_by_id(
        engine, MockMatchFetcher({800: payload}), LEAGUE_ID, [800]
    )
    assert result.status == "COMPLETE"
    assert result.canonical_row_count == 1
