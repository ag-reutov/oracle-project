"""Fixture builders for Slice 10 Elo-adjusted Player × Hero tests.

Reuses the hero-meta Parquet write path so fixtures still go through the
real Step 2 transform. History follows ``player_id`` × ``hero_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from player_hero_helpers import draft_and_player_rows, match_row, write_hero_meta_store

from dota_predictor.features.config import FeatureStoreConfig, ReferenceStoreConfig
from dota_predictor.features.duckdb_layer import connect, register_reference_views
from dota_predictor.features.player_hero_elo import build_player_hero_elo

__all__ = [
    "draft_and_player_rows",
    "match_row",
    "player_hero_elo_frame",
    "write_hero_meta_store",
]


def player_hero_elo_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
    match_id: int | None = None,
    shrinkage_k: float | None = None,
) -> pd.DataFrame:
    """Materialized Slice 10 DataFrame for a fixture dataset."""
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
        shrinkage_k=shrinkage_k,
    )


def _frame_from_store(
    feature_config: FeatureStoreConfig,
    *,
    reference_config: ReferenceStoreConfig | None,
    match_id: int | None,
    shrinkage_k: float | None,
) -> pd.DataFrame:
    kwargs: dict[str, Any] = {"match_id": match_id}
    if shrinkage_k is not None:
        kwargs["shrinkage_k"] = shrinkage_k
    with connect(feature_config) as store:
        if reference_config is not None:
            register_reference_views(store, reference_config)
        return build_player_hero_elo(store, **kwargs).to_frame()
