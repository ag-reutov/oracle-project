"""Fixture builders for Slice 6 Player × Hero meta-relevance tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from player_position_helpers import (
    assign_positions,
    draft_and_player_rows,
    write_hero_meta_store,
)

from dota_predictor.features.config import FeatureStoreConfig, ReferenceStoreConfig
from dota_predictor.features.duckdb_layer import connect, register_reference_views
from dota_predictor.features.player_hero_meta import build_player_hero_meta

__all__ = [
    "assign_positions",
    "draft_and_player_rows",
    "player_hero_meta_frame",
    "write_hero_meta_store",
]


def player_hero_meta_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    method: str,
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
    match_id: int | None = None,
) -> pd.DataFrame:
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
        method=method,
        match_id=match_id,
    )


def _frame_from_store(
    feature_config: FeatureStoreConfig,
    *,
    reference_config: ReferenceStoreConfig | None,
    method: str,
    match_id: int | None,
) -> pd.DataFrame:
    with connect(feature_config) as store:
        if reference_config is not None:
            register_reference_views(store, reference_config)
        return build_player_hero_meta(
            store, method=method, match_id=match_id
        ).to_frame()
