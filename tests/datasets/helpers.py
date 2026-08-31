"""Shared test helpers for dataset-export tests (not a test module itself)."""

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


def seed_ingestion_league(
    conn: Connection, league_id: int, *, name: str = "Test League"
) -> None:
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
    radiant_team_id: int = 100,
    dire_team_id: int = 200,
    radiant_player_ids: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5),
    dire_player_ids: tuple[int, int, int, int, int] = (6, 7, 8, 9, 10),
) -> CanonicalMatch:
    """Build a structurally valid `CanonicalMatch` with `num_bans` bans.

    `num_bans=0` produces a 10-event, all-pick, zero-ban draft -- the same
    shape observed on 13 real matches in the live canonical dataset (see
    the pre-implementation audit), used here to exercise that this is a
    genuinely supported draft length, not an edge case invented for the
    test suite.
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

    draft_events = tuple(events)
    return CanonicalMatch(
        match_id=match_id,
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        league_id=league_id,
        league_name="Test League",
        radiant_team_id=radiant_team_id,
        radiant_team_name_observed="Radiant Team",
        radiant_player_ids=radiant_player_ids,
        radiant_hero_ids=tuple(
            event.hero_id
            for event in draft_events
            if event.action is DraftAction.PICK and event.side is Side.RADIANT
        ),
        dire_team_id=dire_team_id,
        dire_team_name_observed="Dire Team",
        dire_player_ids=dire_player_ids,
        dire_hero_ids=tuple(
            event.hero_id
            for event in draft_events
            if event.action is DraftAction.PICK and event.side is Side.DIRE
        ),
        draft_events=draft_events,
        radiant_win=radiant_win,
        duration_seconds=1800,
    )
