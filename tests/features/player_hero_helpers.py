"""Fixture builders for `test_player_hero.py`.

Reuses the hero-meta Parquet write path so player×hero fixtures still go
through the real Step 2 transform functions. Player ids are explicit
(history follows `player_id` across matches and teams). `slot_in_side`
is lobby order only and is never treated as Dota position 1-5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from hero_meta_helpers import SideBan, match_row, write_hero_meta_store

from dota_predictor.features.config import FeatureStoreConfig, ReferenceStoreConfig
from dota_predictor.features.duckdb_layer import connect, register_reference_views
from dota_predictor.features.player_hero import build_player_hero

__all__ = [
    "draft_and_player_rows",
    "match_row",
    "player_hero_frame",
    "write_hero_meta_store",
]


def draft_and_player_rows(
    match_id: int,
    *,
    radiant_player_ids: tuple[int, int, int, int, int],
    dire_player_ids: tuple[int, int, int, int, int],
    radiant_hero_ids: tuple[int, int, int, int, int],
    dire_hero_ids: tuple[int, int, int, int, int],
    successful_bans: tuple[SideBan, ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aligned draft events and match_players rows for one match.

    Hero sets must match successful PICK sets (canonical contract).
    `slot_in_side` is 0-4 lobby order per side, not position 1-5.
    """
    draft_rows: list[dict[str, Any]] = []
    sequence = 0
    for side, hero_id in successful_bans:
        draft_rows.append(
            {
                "match_id": match_id,
                "sequence": sequence,
                "action": "BAN",
                "side": side,
                "hero_id": hero_id,
                "was_successful": True,
            }
        )
        sequence += 1
    for side, picks in (("RADIANT", radiant_hero_ids), ("DIRE", dire_hero_ids)):
        for hero_id in picks:
            draft_rows.append(
                {
                    "match_id": match_id,
                    "sequence": sequence,
                    "action": "PICK",
                    "side": side,
                    "hero_id": hero_id,
                    "was_successful": None,
                }
            )
            sequence += 1

    player_rows: list[dict[str, Any]] = []
    for slot, (player_id, hero_id) in enumerate(
        zip(radiant_player_ids, radiant_hero_ids, strict=True)
    ):
        player_rows.append(
            {
                "match_id": match_id,
                "side": "RADIANT",
                "slot_in_side": slot,
                "player_id": player_id,
                "hero_id": hero_id,
            }
        )
    for slot, (player_id, hero_id) in enumerate(
        zip(dire_player_ids, dire_hero_ids, strict=True)
    ):
        player_rows.append(
            {
                "match_id": match_id,
                "side": "DIRE",
                "slot_in_side": slot,
                "player_id": player_id,
                "hero_id": hero_id,
            }
        )
    return draft_rows, player_rows


def player_hero_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
    match_id: int | None = None,
) -> pd.DataFrame:
    """Materialized player×hero DataFrame for a fixture dataset."""
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
        return build_player_hero(store, match_id=match_id).to_frame()
