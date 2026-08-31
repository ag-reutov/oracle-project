"""Fixture builders for `test_team_hero.py`.

Reuses the hero-meta Parquet write path and the player×hero draft/player
row builder so team×hero fixtures still go through the real Step 2
transform functions. Player ids are explicit so roster-change tests can
swap a team's five players without changing `team_id`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from hero_meta_helpers import write_hero_meta_store
from player_hero_helpers import draft_and_player_rows, match_row

from dota_predictor.features.config import FeatureStoreConfig, ReferenceStoreConfig
from dota_predictor.features.duckdb_layer import connect, register_reference_views
from dota_predictor.features.team_hero import build_team_hero

__all__ = [
    "draft_and_player_rows",
    "match_row",
    "team_hero_frame",
    "write_hero_meta_store",
]


def team_hero_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
    match_id: int | None = None,
) -> pd.DataFrame:
    """Materialized team×hero DataFrame for a fixture dataset."""
    feature_config, reference_config = write_hero_meta_store(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        game_versions=game_versions,
    )
    return _frame_from_store(
        feature_config,
        reference_config=reference_config,
        match_id=match_id,
    )


def _frame_from_store(
    feature_config: FeatureStoreConfig,
    *,
    reference_config: ReferenceStoreConfig | None,
    match_id: int | None,
) -> pd.DataFrame:
    with connect(feature_config) as store:
        if reference_config is not None:
            register_reference_views(store, reference_config)
        return build_team_hero(store, match_id=match_id).to_frame()
