"""Fixture builders for `test_player_match.py`.

Reuses the hero-meta Parquet write path so player-match fixtures still go
through the real Step 2 transform functions. `slot_in_side` is lobby order
only and is never treated as Dota position 1-5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from player_hero_helpers import draft_and_player_rows, write_hero_meta_store

from dota_predictor.features.config import FeatureStoreConfig
from dota_predictor.features.duckdb_layer import connect
from dota_predictor.features.player_match import build_player_match, build_player_state

__all__ = [
    "draft_and_player_rows",
    "player_match_frame",
    "player_state_frame",
    "write_hero_meta_store",
]


def player_match_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
) -> pd.DataFrame:
    """Materialized player-match fact DataFrame for a fixture dataset."""
    feature_config, _reference_config = write_hero_meta_store(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=None,
        game_versions=None,
    )
    return _frame_from_store(feature_config, state=False)


def player_state_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    match_id: int | None = None,
) -> pd.DataFrame:
    """Materialized player-state DataFrame for a fixture dataset."""
    feature_config, _reference_config = write_hero_meta_store(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=None,
        game_versions=None,
    )
    return _frame_from_store(feature_config, state=True, match_id=match_id)


def _frame_from_store(
    feature_config: FeatureStoreConfig,
    *,
    state: bool,
    match_id: int | None = None,
) -> pd.DataFrame:
    with connect(feature_config) as store:
        if state:
            return build_player_state(store, match_id=match_id).to_frame()
        return build_player_match(store).to_frame()
