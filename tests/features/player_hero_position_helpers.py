"""Fixture builders for Player × Hero × expected-position tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from player_position_helpers import (
    assign_positions,
    draft_and_player_rows,
    write_hero_meta_store,
)

from dota_predictor.features.config import FeatureStoreConfig
from dota_predictor.features.duckdb_layer import connect
from dota_predictor.features.player_hero_position import build_player_hero_position

__all__ = [
    "assign_positions",
    "draft_and_player_rows",
    "player_hero_position_frame",
]


def player_hero_position_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    method: str,
    match_id: int | None = None,
) -> pd.DataFrame:
    feature_config, _reference_config = write_hero_meta_store(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=None,
        game_versions=None,
    )
    return _frame_from_store(
        feature_config, method=method, match_id=match_id
    )


def _frame_from_store(
    feature_config: FeatureStoreConfig,
    *,
    method: str,
    match_id: int | None,
) -> pd.DataFrame:
    with connect(feature_config) as store:
        return build_player_hero_position(
            store, method=method, match_id=match_id
        ).to_frame()
