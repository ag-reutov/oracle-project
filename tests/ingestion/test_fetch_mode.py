"""Registry fetch_mode dispatch for match-id vs league(id) ingestion."""

from __future__ import annotations

from typing import Any

import pytest
from ingestion_helpers import (
    build_stratz_match,
    requires_test_database,
    seed_ingestion_league,
)
from sqlalchemy import select

from dota_predictor.ingestion.cursor import MATCH_IDS_CURSOR_MODE, CursorState
from dota_predictor.ingestion.errors import LeagueFetchModeError
from dota_predictor.ingestion.pipeline import ingest_league
from dota_predictor.storage.ingestion_writer import (
    load_cursor_state,
    update_league_ingestion_state,
)
from dota_predictor.storage.schema import (
    LEAGUE_FETCH_MODE_MATCH_IDS,
    LEAGUES,
    MATCHES,
    STRATZ_RAW_MATCHES,
)

LEAGUE_ID = 17419


class LeaguePageBoom:
    """Fetcher that must never be asked to page `league(id)`."""

    def fetch_league_matches_page(self, *args: object, **kwargs: object) -> list:
        raise AssertionError("league(id) pagination must not run")


class MatchIdFetcher(LeaguePageBoom):
    def __init__(self, matches: dict[int, dict[str, Any]]) -> None:
        self.matches = matches
        self.match_calls: list[int] = []

    def fetch_match(self, match_id: int) -> dict[str, Any] | None:
        self.match_calls.append(match_id)
        payload = self.matches.get(match_id)
        return None if payload is None else dict(payload)


def _match(match_id: int, start: int) -> dict[str, Any]:
    return build_stratz_match(
        match_id=match_id,
        league_id=LEAGUE_ID,
        start_date_time=start,
    )


@requires_test_database
def test_match_ids_fetch_mode_uses_match_id_path_on_fresh_state(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(
            conn, LEAGUE_ID, name="SLAM IV", fetch_mode=LEAGUE_FETCH_MODE_MATCH_IDS
        )
        cursor = load_cursor_state(conn, LEAGUE_ID)
        assert cursor.mode is None
        assert cursor.fetch_complete is False

    payload = _match(100, 1_760_000_000)
    fetcher = MatchIdFetcher({100: payload})
    result = ingest_league(engine, fetcher, LEAGUE_ID, match_ids=[100])

    assert result.status == "COMPLETE"
    assert result.fetch_complete is True
    assert fetcher.match_calls == [100]
    with engine.connect() as conn:
        assert conn.execute(
            select(STRATZ_RAW_MATCHES.c.match_id)
        ).scalar_one() == 100
        assert conn.execute(select(MATCHES.c.match_id)).scalar_one() == 100
        cursor = load_cursor_state(conn, LEAGUE_ID)
        assert cursor.mode == MATCH_IDS_CURSOR_MODE
        assert cursor.fetch_complete is True


@requires_test_database
def test_match_ids_fetch_mode_never_calls_league_id(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(
            conn, LEAGUE_ID, fetch_mode=LEAGUE_FETCH_MODE_MATCH_IDS
        )

    fetcher = MatchIdFetcher({200: _match(200, 1_760_000_100)})
    ingest_league(engine, fetcher, LEAGUE_ID, match_ids=[200])
    assert fetcher.match_calls == [200]


@requires_test_database
def test_match_ids_mode_without_match_fetcher_refuses_league_path(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(
            conn, LEAGUE_ID, fetch_mode=LEAGUE_FETCH_MODE_MATCH_IDS
        )

    with pytest.raises(LeagueFetchModeError, match="refusing to call"):
        ingest_league(engine, LeaguePageBoom(), LEAGUE_ID, match_ids=[1])


@requires_test_database
def test_default_fetch_mode_still_uses_league_path(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, LEAGUE_ID)

    class LeagueFetcher:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def fetch_league_matches_page(
            self, league_id: int, *, skip: int, take: int
        ) -> list[dict[str, Any]]:
            self.calls.append((skip, take))
            if skip == 0:
                return [_match(300, 1_760_000_200)]
            return []

        def fetch_match(self, match_id: int) -> dict[str, Any] | None:
            raise AssertionError("default fetch_mode must not call match(id)")

    fetcher = LeagueFetcher()
    result = ingest_league(engine, fetcher, LEAGUE_ID, page_size=100)
    assert result.status == "COMPLETE"
    assert fetcher.calls[0] == (0, 100)
    with engine.connect() as conn:
        cursor = load_cursor_state(conn, LEAGUE_ID)
        assert cursor.mode != MATCH_IDS_CURSOR_MODE


@requires_test_database
def test_fetch_mode_survives_cursor_reset(engine) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(
            conn, LEAGUE_ID, fetch_mode=LEAGUE_FETCH_MODE_MATCH_IDS
        )

    payload = _match(400, 1_760_000_300)
    first = MatchIdFetcher({400: payload})
    ingest_league(engine, first, LEAGUE_ID, match_ids=[400])

    with engine.begin() as conn:
        update_league_ingestion_state(
            conn,
            league_id=LEAGUE_ID,
            status="PENDING",
            cursor=CursorState(),
            matches_seen_count=0,
        )
        conn.execute(
            LEAGUES.update()
            .where(LEAGUES.c.league_id == LEAGUE_ID)
            .values(fetch_mode=LEAGUE_FETCH_MODE_MATCH_IDS)
        )

    second = MatchIdFetcher({400: payload})
    result = ingest_league(engine, second, LEAGUE_ID, match_ids=[400])
    assert result.status == "COMPLETE"
    assert second.match_calls == []  # already raw; skipped, not league(id)
    with engine.connect() as conn:
        cursor = load_cursor_state(conn, LEAGUE_ID)
        assert cursor.mode == MATCH_IDS_CURSOR_MODE
