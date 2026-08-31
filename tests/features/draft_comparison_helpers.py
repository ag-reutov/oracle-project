"""Fixture builders for `test_draft_comparison.py`.

Reuses the draft-profile Parquet write path so comparison fixtures still
go through the real Step 2 transform functions and the real side-level
Draft Profile. The comparison layer must not recompute history itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from draft_profile_helpers import write_hero_meta_store

from dota_predictor.features.config import FeatureStoreConfig, ReferenceStoreConfig
from dota_predictor.features.draft_comparison import build_draft_comparison
from dota_predictor.features.draft_profile import build_draft_profile
from dota_predictor.features.duckdb_layer import connect, register_reference_views

__all__ = [
    "draft_comparison_frame",
    "draft_comparison_layers",
]


def draft_comparison_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
    match_id: int | None = None,
) -> pd.DataFrame:
    """Materialized one-row-per-match comparison DataFrame."""
    comparison, _ = draft_comparison_layers(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        game_versions=game_versions,
        match_id=match_id,
    )
    return comparison


def draft_comparison_layers(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
    match_id: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comparison plus the source `(match_id, side)` profile from one store."""
    feature_config, reference_config = write_hero_meta_store(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        game_versions=game_versions,
    )
    return _frames_from_store(
        feature_config,
        reference_config=reference_config,
        match_id=match_id,
    )


def _frames_from_store(
    feature_config: FeatureStoreConfig,
    *,
    reference_config: ReferenceStoreConfig | None,
    match_id: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with connect(feature_config) as store:
        if reference_config is not None:
            register_reference_views(store, reference_config)
        comparison = build_draft_comparison(store, match_id=match_id).to_frame()
        profile = build_draft_profile(store, match_id=match_id).to_frame()
    return comparison, profile
