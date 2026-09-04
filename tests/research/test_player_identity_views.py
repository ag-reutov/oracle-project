"""Semantic tests for the `research.players` player-universe view (Slice 2).

Seeds canonical relational rows and asserts the player-identity view
behavior: one row per canonical `player_id`, correct first/last seen and
match counts, a deterministic (NULL today) display name, no mutable
team/position/current-state columns, and orphan registry ids excluded.
The view is created from the same SQL the Alembic migration applies
(`dota_predictor.research.views` / `dota_predictor.data.player_identity`).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from research_helpers import (
    DIRE_PLAYER_BASE_IDS,
    PLAYER_BASE_IDS,
    seed_league,
    seed_match,
)
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set; skipping DB-touching test",
)

UTC2024 = datetime(2024, 1, 1, tzinfo=UTC)
UTC2025 = datetime(2025, 6, 1, tzinfo=UTC)

_ALL_PLAYER_IDS = sorted(PLAYER_BASE_IDS + DIRE_PLAYER_BASE_IDS)


def _rows(engine, sql: str) -> list[sa.Row]:
    with engine.connect() as conn:
        return conn.execute(sa.text(sql)).all()


def test_one_row_per_player_with_correct_summary(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 1001, name="Test League", tier="T1")
        seed_match(conn, match_id=1, league_id=1001, start_time=UTC2024)
        seed_match(conn, match_id=2, league_id=1001, start_time=UTC2025)

    rows = _rows(engine, "SELECT * FROM research.players ORDER BY player_id")
    by_id = {int(r.player_id): r for r in rows}
    assert len(by_id) == 10
    assert set(by_id.keys()) == set(_ALL_PLAYER_IDS)
    for player_id in _ALL_PLAYER_IDS:
        r = by_id[player_id]
        assert r.first_seen_at == UTC2024
        assert r.last_seen_at == UTC2025
        assert r.match_count == 2


def test_display_name_is_null_and_no_mutable_state_columns(engine) -> None:
    """The identity view exposes only identity summary; no current team,
    current position, rating, or other time-varying state can leak in."""
    with engine.begin() as conn:
        seed_league(conn, 1002, name="Test League", tier="T1")
        seed_match(conn, match_id=3, league_id=1002, start_time=UTC2024)

    column_sql = (
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'research' AND table_name = 'players'"
    )
    columns = {r[0] for r in _rows(engine, column_sql)}
    assert columns == {
        "player_id",
        "display_name",
        "first_seen_at",
        "last_seen_at",
        "match_count",
    }

    for forbidden in (
        "team_id",
        "position",
        "lane",
        "role",
        "rating",
        "elo",
        "hero_pool",
    ):
        assert forbidden not in columns

    display = {
        int(r[0])
        for r in _rows(engine, "SELECT player_id, display_name FROM research.players")
    }
    nulls = _rows(
        engine, "SELECT count(*) FROM research.players WHERE display_name IS NULL"
    )[0][0]
    assert nulls == len(display)


def test_match_counts_aggregate_across_matches(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 1003, name="Test League", tier="T1")
        for match_id in (11, 12, 13):
            seed_match(conn, match_id=match_id, league_id=1003, start_time=UTC2024)

    rows = _rows(engine, "SELECT DISTINCT match_count FROM research.players")
    assert [int(r[0]) for r in rows] == [3]


def test_repeated_appearances_yield_one_identity(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 1004, name="Test League", tier="T1")
        seed_match(conn, match_id=21, league_id=1004, start_time=UTC2024)
        seed_match(conn, match_id=22, league_id=1004, start_time=UTC2025)

    count = _rows(
        engine, "SELECT count(*), count(DISTINCT player_id) FROM research.players"
    )[0]
    assert count[0] == count[1] == 10


def test_orphan_registry_ids_excluded_from_universe(engine) -> None:
    """Registry ids with no observed match are not part of the professional
    player universe (they are reported as orphans by the audit instead)."""
    with engine.begin() as conn:
        seed_league(conn, 1005, name="Test League", tier="T1")
        seed_match(conn, match_id=31, league_id=1005, start_time=UTC2024)
        conn.execute(sa.text("INSERT INTO players (player_id) VALUES (777777777)"))

    rows = _rows(engine, "SELECT player_id FROM research.players")
    player_ids = {int(r[0]) for r in rows}
    assert 777777777 not in player_ids
    assert player_ids == set(_ALL_PLAYER_IDS)


def test_first_last_seen_correct_across_time(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 1006, name="Test League", tier="T1")
        seed_match(conn, match_id=41, league_id=1006, start_time=UTC2024)
        seed_match(
            conn,
            match_id=42,
            league_id=1006,
            start_time=datetime(2024, 12, 31, tzinfo=UTC),
        )

    rows = _rows(
        engine, "SELECT DISTINCT first_seen_at, last_seen_at FROM research.players"
    )
    assert len(rows) == 1
    assert rows[0][0] == UTC2024
    assert rows[0][1] == datetime(2024, 12, 31, tzinfo=UTC)


def test_null_player_id_rejected_by_schema(engine) -> None:
    """Unknown/null player identities are handled intentionally: the schema
    forbids a NULL `match_players.player_id` (canonical identity must always
    resolve), so no null identity can silently enter the universe."""
    with engine.begin() as conn:
        seed_league(conn, 1007, name="Test League", tier="T1")
        seed_match(conn, match_id=51, league_id=1007, start_time=UTC2024)
        with pytest.raises(IntegrityError):
            conn.execute(
                sa.text(
                    "INSERT INTO match_players "
                    "(match_id, side, slot_in_side, player_id, hero_id) "
                    "VALUES (:match_id, 'RADIANT', 0, NULL, 1)"
                ),
                {"match_id": 51},
            )
