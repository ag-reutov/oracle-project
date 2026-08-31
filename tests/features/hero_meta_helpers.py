"""Fixture builders for `test_hero_meta.py`.

Builds real canonical Parquet files (via the Step 2 transform functions)
with aligned `draft_events` / `match_players.hero_id` sets, plus an
optional STRATZ reference catalog. PRE_DRAFT snapshot tests keep using
`pre_draft_helpers` (empty drafts); this helper exists so hero-meta
fixtures can control pick/ban/success semantics without changing that
contract.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dota_predictor.datasets.canonical_export import (
    DRAFT_EVENTS_FILENAME,
    MATCH_PLAYERS_FILENAME,
    MATCHES_FILENAME,
    build_draft_events_table,
    build_match_players_table,
    build_matches_table,
    write_canonical_dataset,
)
from dota_predictor.datasets.reference_export import build_reference_dataset
from dota_predictor.features.config import FeatureStoreConfig, ReferenceStoreConfig
from dota_predictor.features.duckdb_layer import connect, register_reference_views
from dota_predictor.features.hero_meta import build_hero_meta

SideBan = tuple[str, int]


def match_row(
    match_id: int,
    *,
    start_time: datetime,
    radiant_win: bool,
    game_version_id: int,
    radiant_team_id: int = 100,
    dire_team_id: int = 200,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "league_id": 1,
        "start_time": start_time,
        "league_name": "Test League",
        "series_id": 10,
        "series_type": "BEST_OF_THREE",
        "game_number_in_series": None,
        "game_version_id": game_version_id,
        "radiant_team_id": radiant_team_id,
        "radiant_team_name_observed": "Radiant Team",
        "dire_team_id": dire_team_id,
        "dire_team_name_observed": "Dire Team",
        "radiant_win": radiant_win,
        "duration_seconds": 1800,
        "mapper_version": 1,
        "canonicalized_at": start_time,
    }


def draft_and_player_rows(
    match_id: int,
    *,
    radiant_picks: tuple[int, int, int, int, int],
    dire_picks: tuple[int, int, int, int, int],
    successful_bans: tuple[SideBan, ...] = (),
    unsuccessful_bans: tuple[SideBan, ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aligned draft events and match_players rows for one match.

    Unsuccessful bans are emitted first so a failed ban of a later-picked
    hero is a legal canonical sequence (failed bans are not actual
    actions). Successful bans follow, then Radiant picks, then Dire picks.
    """
    draft_rows: list[dict[str, Any]] = []
    sequence = 0
    for side, hero_id in unsuccessful_bans:
        draft_rows.append(
            {
                "match_id": match_id,
                "sequence": sequence,
                "action": "BAN",
                "side": side,
                "hero_id": hero_id,
                "was_successful": False,
            }
        )
        sequence += 1
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
    for side, picks in (("RADIANT", radiant_picks), ("DIRE", dire_picks)):
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
    for slot, hero_id in enumerate(radiant_picks):
        player_rows.append(
            {
                "match_id": match_id,
                "side": "RADIANT",
                "slot_in_side": slot,
                "player_id": match_id * 100 + slot + 1,
                "hero_id": hero_id,
            }
        )
    for slot, hero_id in enumerate(dire_picks):
        player_rows.append(
            {
                "match_id": match_id,
                "side": "DIRE",
                "slot_in_side": slot,
                "player_id": match_id * 100 + slot + 6,
                "hero_id": hero_id,
            }
        )
    return draft_rows, player_rows


def write_hero_meta_store(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
) -> tuple[FeatureStoreConfig, ReferenceStoreConfig | None]:
    """Write fact Parquet (and optional reference catalog) under `tmp_path`."""
    matches_table = build_matches_table(matches, players)
    match_players_table = build_match_players_table(matches, players)
    draft_events_table = build_draft_events_table(drafts)
    write_canonical_dataset(
        tmp_path,
        matches_table=matches_table,
        draft_events_table=draft_events_table,
        match_players_table=match_players_table,
    )
    feature_config = FeatureStoreConfig(
        matches_path=tmp_path / MATCHES_FILENAME,
        match_players_path=tmp_path / MATCH_PLAYERS_FILENAME,
        draft_events_path=tmp_path / DRAFT_EVENTS_FILENAME,
    )
    reference_config: ReferenceStoreConfig | None = None
    if heroes is not None:
        versions = game_versions or [
            {"id": 1, "name": "7.00", "asOfDateTime": 1_700_000_000}
        ]
        build_reference_dataset(
            tmp_path, heroes=heroes, game_versions=versions
        )
        reference_config = ReferenceStoreConfig(
            heroes_path=tmp_path / "heroes.parquet",
            game_versions_path=tmp_path / "game_versions.parquet",
        )
    return feature_config, reference_config


def hero_meta_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    heroes: list[dict[str, Any]] | None = None,
    game_versions: list[dict[str, Any]] | None = None,
    match_id: int | None = None,
) -> pd.DataFrame:
    """Materialized hero-meta DataFrame for a fixture dataset."""
    feature_config, reference_config = write_hero_meta_store(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        game_versions=game_versions,
    )
    with connect(feature_config) as store:
        if reference_config is not None:
            register_reference_views(store, reference_config)
        return build_hero_meta(store, match_id=match_id).to_frame()
