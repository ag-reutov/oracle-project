"""Fixture builders for `test_draft_profile.py`.

Reuses the player×hero draft/player row builder and the hero-meta Parquet
write path so draft-profile fixtures still go through the real Step 2
transform functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from hero_meta_helpers import write_hero_meta_store
from player_hero_helpers import draft_and_player_rows, match_row

from dota_predictor.features.config import FeatureStoreConfig, ReferenceStoreConfig
from dota_predictor.features.draft_profile import build_draft_profile
from dota_predictor.features.duckdb_layer import connect, register_reference_views
from dota_predictor.features.hero_meta import build_hero_meta
from dota_predictor.features.player_hero import build_player_hero
from dota_predictor.features.team_hero import build_team_hero

__all__ = [
    "draft_and_player_rows",
    "draft_profile_frame",
    "draft_profile_layers",
    "match_row",
    "write_hero_meta_store",
]


def draft_profile_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
    match_id: int | None = None,
) -> pd.DataFrame:
    """Materialized `(match_id, side)` draft-profile DataFrame."""
    profile, _, _, _ = draft_profile_layers(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        game_versions=game_versions,
        match_id=match_id,
    )
    return profile


def draft_profile_layers(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
    match_id: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Profile plus the three underlying layer frames from one store."""
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with connect(feature_config) as store:
        if reference_config is not None:
            register_reference_views(store, reference_config)
        profile = build_draft_profile(store, match_id=match_id).to_frame()
        player_hero = build_player_hero(store, match_id=match_id).to_frame()
        team_hero = build_team_hero(store, match_id=match_id).to_frame()
        hero_meta = build_hero_meta(store, match_id=match_id).to_frame()
    return profile, player_hero, team_hero, hero_meta
