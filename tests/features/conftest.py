"""Shared fixtures for feature-layer (Step 3A) tests.

Builds a small, deterministic canonical Parquet fixture directly via the
real Step 2 transform functions (`datasets.canonical_export`), never
touching PostgreSQL, so these tests exercise the actual Step 2 Parquet
contract rather than a hand-rolled approximation of it.

The two fixture matches are deliberately assigned `match_id`/`start_time`
pairs that go in *opposite* order (`MATCH_EARLY_ID > MATCH_LATE_ID`
numerically, despite `MATCH_EARLY_ID` having the earlier `start_time`),
so any test that accidentally sorts/filters by `match_id` instead of
`start_time` fails loudly instead of coincidentally passing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from dota_predictor.datasets.canonical_export import (
    DRAFT_EVENTS_FILENAME,
    MATCH_PLAYERS_FILENAME,
    MATCHES_FILENAME,
    build_draft_events_table,
    build_match_players_table,
    build_matches_table,
    write_canonical_dataset,
)
from dota_predictor.features.config import FeatureStoreConfig

EARLY_START_TIME = datetime(2024, 1, 1, tzinfo=UTC)
LATE_START_TIME = datetime(2024, 6, 1, tzinfo=UTC)

# Numerically "backwards" relative to start_time on purpose -- see module
# docstring.
MATCH_EARLY_ID = 2002
MATCH_LATE_ID = 1001

MATCH_EARLY_RADIANT_TEAM_ID = 100
MATCH_EARLY_DIRE_TEAM_ID = 200
MATCH_EARLY_RADIANT_PLAYER_IDS = (11, 12, 13, 14, 15)
MATCH_EARLY_DIRE_PLAYER_IDS = (21, 22, 23, 24, 25)
MATCH_EARLY_NUM_BANS = 4  # 14-event draft

MATCH_LATE_RADIANT_TEAM_ID = 300
MATCH_LATE_DIRE_TEAM_ID = 400
MATCH_LATE_RADIANT_PLAYER_IDS = (31, 32, 33, 34, 35)
MATCH_LATE_DIRE_PLAYER_IDS = (41, 42, 43, 44, 45)
MATCH_LATE_NUM_BANS = 0  # real, observed 10-event zero-ban draft shape

# Deterministic positive hero ids per lobby slot. `slot_in_side` is lobby
# order, not Dota position 1-5 -- these values are fixture identity only.
MATCH_EARLY_RADIANT_HERO_IDS = (1, 2, 3, 4, 5)
MATCH_EARLY_DIRE_HERO_IDS = (6, 7, 8, 9, 10)
MATCH_LATE_RADIANT_HERO_IDS = (11, 12, 13, 14, 15)
MATCH_LATE_DIRE_HERO_IDS = (16, 17, 18, 19, 20)


def _match_row(
    match_id: int,
    *,
    start_time: datetime,
    radiant_team_id: int,
    dire_team_id: int,
    radiant_win: bool,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "league_id": 1,
        "start_time": start_time,
        "league_name": "Test League",
        "series_id": 10,
        "series_type": "BEST_OF_THREE",
        "game_number_in_series": None,
        "game_version_id": 176,
        "radiant_team_id": radiant_team_id,
        "radiant_team_name_observed": "Radiant Team",
        "dire_team_id": dire_team_id,
        "dire_team_name_observed": "Dire Team",
        "radiant_win": radiant_win,
        "duration_seconds": 1800,
        "mapper_version": 1,
        "canonicalized_at": start_time,
    }


def _player_rows(
    match_id: int,
    *,
    radiant_ids: tuple[int, ...],
    dire_ids: tuple[int, ...],
    radiant_hero_ids: tuple[int, ...],
    dire_hero_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot, (player_id, hero_id) in enumerate(
        zip(radiant_ids, radiant_hero_ids, strict=True)
    ):
        rows.append(
            {
                "match_id": match_id,
                "side": "RADIANT",
                "slot_in_side": slot,
                "player_id": player_id,
                "hero_id": hero_id,
            }
        )
    for slot, (player_id, hero_id) in enumerate(
        zip(dire_ids, dire_hero_ids, strict=True)
    ):
        rows.append(
            {
                "match_id": match_id,
                "side": "DIRE",
                "slot_in_side": slot,
                "player_id": player_id,
                "hero_id": hero_id,
            }
        )
    return rows


def _draft_rows(match_id: int, *, num_bans: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sequence = 0
    for i in range(num_bans):
        rows.append(
            {
                "match_id": match_id,
                "sequence": sequence,
                "action": "BAN",
                "side": "RADIANT" if i % 2 == 0 else "DIRE",
                "hero_id": 1000 + sequence,
                "was_successful": True,
            }
        )
        sequence += 1
    for side in ("RADIANT", "DIRE"):
        for _ in range(5):
            rows.append(
                {
                    "match_id": match_id,
                    "sequence": sequence,
                    "action": "PICK",
                    "side": side,
                    "hero_id": 1000 + sequence,
                    "was_successful": None,
                }
            )
            sequence += 1
    return rows


@pytest.fixture
def feature_store_config(tmp_path: Path) -> FeatureStoreConfig:
    """A `FeatureStoreConfig` pointing at a freshly built, real Step 2
    analytical schema-v2 Parquet triple under `tmp_path`."""
    match_rows = [
        _match_row(
            MATCH_EARLY_ID,
            start_time=EARLY_START_TIME,
            radiant_team_id=MATCH_EARLY_RADIANT_TEAM_ID,
            dire_team_id=MATCH_EARLY_DIRE_TEAM_ID,
            radiant_win=True,
        ),
        _match_row(
            MATCH_LATE_ID,
            start_time=LATE_START_TIME,
            radiant_team_id=MATCH_LATE_RADIANT_TEAM_ID,
            dire_team_id=MATCH_LATE_DIRE_TEAM_ID,
            radiant_win=False,
        ),
    ]
    player_rows = _player_rows(
        MATCH_EARLY_ID,
        radiant_ids=MATCH_EARLY_RADIANT_PLAYER_IDS,
        dire_ids=MATCH_EARLY_DIRE_PLAYER_IDS,
        radiant_hero_ids=MATCH_EARLY_RADIANT_HERO_IDS,
        dire_hero_ids=MATCH_EARLY_DIRE_HERO_IDS,
    ) + _player_rows(
        MATCH_LATE_ID,
        radiant_ids=MATCH_LATE_RADIANT_PLAYER_IDS,
        dire_ids=MATCH_LATE_DIRE_PLAYER_IDS,
        radiant_hero_ids=MATCH_LATE_RADIANT_HERO_IDS,
        dire_hero_ids=MATCH_LATE_DIRE_HERO_IDS,
    )
    draft_rows = _draft_rows(
        MATCH_EARLY_ID, num_bans=MATCH_EARLY_NUM_BANS
    ) + _draft_rows(MATCH_LATE_ID, num_bans=MATCH_LATE_NUM_BANS)

    matches_table = build_matches_table(match_rows, player_rows)
    match_players_table = build_match_players_table(match_rows, player_rows)
    draft_events_table = build_draft_events_table(draft_rows)

    write_canonical_dataset(
        tmp_path,
        matches_table=matches_table,
        draft_events_table=draft_events_table,
        match_players_table=match_players_table,
    )

    return FeatureStoreConfig(
        matches_path=tmp_path / MATCHES_FILENAME,
        match_players_path=tmp_path / MATCH_PLAYERS_FILENAME,
        draft_events_path=tmp_path / DRAFT_EVENTS_FILENAME,
    )
