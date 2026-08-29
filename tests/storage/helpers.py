"""Shared test helpers for storage tests (not a test module itself)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from itertools import cycle

import pytest
from sqlalchemy import Connection

from dota_predictor.data.canonical_schema import (
    CanonicalMatch,
    DraftAction,
    DraftEvent,
    Side,
)
from dota_predictor.storage.schema import INGESTION_LEAGUES, LEAGUES

requires_test_database = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set; skipping DB-touching test",
)


def seed_ingestion_league(conn: Connection, league_id: int, *, name: str = "Test League") -> None:
    """Insert a league directly into `leagues` + `ingestion_leagues`.

    Bypasses `scripts/load_league_registry.py` on purpose -- tests exercise
    the schema/writer, not the loader (which has its own tests).
    """
    conn.execute(
        LEAGUES.insert().values(
            league_id=league_id,
            name=name,
            liquipedia_tier="T1",
            in_scope=True,
        )
    )
    conn.execute(INGESTION_LEAGUES.insert().values(league_id=league_id))


def build_canonical_match(
    *,
    match_id: int,
    league_id: int,
    num_bans: int = 4,
    radiant_win: bool = True,
) -> CanonicalMatch:
    """Build a structurally valid `CanonicalMatch` with `num_bans` bans.

    Draft events: `num_bans` alternating-side bans, then 5 radiant picks,
    then 5 dire picks -- every hero id is unique across the whole draft so
    `CanonicalMatch`'s hero-uniqueness invariant always holds regardless
    of `num_bans`.
    """
    events: list[DraftEvent] = []
    hero_id = 1
    sequence = 0
    sides = cycle([Side.RADIANT, Side.DIRE])
    for _ in range(num_bans):
        events.append(
            DraftEvent(
                sequence=sequence,
                action=DraftAction.BAN,
                side=next(sides),
                hero_id=hero_id,
            )
        )
        hero_id += 1
        sequence += 1
    for side in (Side.RADIANT, Side.DIRE):
        for _ in range(5):
            events.append(
                DraftEvent(
                    sequence=sequence,
                    action=DraftAction.PICK,
                    side=side,
                    hero_id=hero_id,
                )
            )
            hero_id += 1
            sequence += 1

    return CanonicalMatch(
        match_id=match_id,
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        league_id=league_id,
        league_name="Test League",
        radiant_team_id=100,
        radiant_team_name_observed="Radiant Team",
        radiant_player_ids=(1, 2, 3, 4, 5),
        dire_team_id=200,
        dire_team_name_observed="Dire Team",
        dire_player_ids=(6, 7, 8, 9, 10),
        draft_events=tuple(events),
        radiant_win=radiant_win,
        duration_seconds=1800,
    )
