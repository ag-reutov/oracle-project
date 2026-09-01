"""Fixture builders for `test_player_position.py`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from player_match_helpers import draft_and_player_rows, write_hero_meta_store

from dota_predictor.features.config import FeatureStoreConfig
from dota_predictor.features.duckdb_layer import connect
from dota_predictor.features.player_position import build_player_position_state

__all__ = [
    "assign_positions",
    "draft_and_player_rows",
    "player_position_state_frame",
    "write_hero_meta_store",
]


def assign_positions(
    players: list[dict[str, Any]],
    assignments: dict[tuple[int, int], str | None],
) -> list[dict[str, Any]]:
    """Set observed `position` on selected `(match_id, player_id)` rows.

    Unmentioned rows keep whatever the fixture already had (usually NULL).
    Does not infer or fill missing positions.
    """
    for row in players:
        key = (int(row["match_id"]), int(row["player_id"]))
        if key in assignments:
            row["position"] = assignments[key]
    return players


def player_position_state_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    match_id: int | None = None,
) -> pd.DataFrame:
    """Materialized player × position state for a fixture dataset."""
    feature_config, _reference_config = write_hero_meta_store(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=None,
        game_versions=None,
    )
    return _frame_from_store(feature_config, match_id=match_id)


def _frame_from_store(
    feature_config: FeatureStoreConfig, *, match_id: int | None
) -> pd.DataFrame:
    with connect(feature_config) as store:
        return build_player_position_state(store, match_id=match_id).to_frame()
