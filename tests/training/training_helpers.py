"""Shared fixture builders for Step 4A (`training/`) tests.

Deliberately duplicates the small canonical-Parquet-fixture pattern
already used by `tests/features/pre_draft_helpers.py` and
`tests/datasets/helpers.py`, rather than cross-importing across test
packages -- see those modules for the same convention.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from dota_predictor.datasets.canonical_export import (
    DRAFT_EVENTS_FILENAME,
    MATCH_PLAYERS_FILENAME,
    MATCHES_FILENAME,
    build_draft_events_table,
    build_match_players_table,
    build_matches_table,
    write_canonical_dataset,
)
from dota_predictor.features.config import FeatureStoreConfig
from dota_predictor.features.duckdb_layer import FeatureDuckDBConnection, connect
from dota_predictor.features.pre_draft_snapshot import (
    PreDraftSnapshot,
    build_pre_draft_snapshot,
)
from dota_predictor.training.dataset import ModelReadyDataset, build_model_ready_dataset


def match_row(
    match_id: int,
    *,
    start_time: datetime,
    radiant_team_id: int,
    dire_team_id: int,
    radiant_win: bool,
    league_id: int = 1,
    series_id: int | None = 10,
    series_type: str | None = "BEST_OF_THREE",
    game_version_id: int | None = 176,
) -> dict[str, Any]:
    """One `matches` row spec, defaulting every field Step 4A does not
    exercise so call sites only need to state what varies per test."""
    return {
        "match_id": match_id,
        "league_id": league_id,
        "start_time": start_time,
        "league_name": "Test League",
        "series_id": series_id,
        "series_type": series_type,
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


def player_rows(
    match_id: int,
    *,
    radiant_ids: tuple[int, int, int, int, int],
    dire_ids: tuple[int, int, int, int, int],
) -> list[dict[str, Any]]:
    """The 10 `match_players` rows for one match, 5 per side, in slot order.

    `hero_id` is a deterministic positive fixture value keyed by lobby
    slot (0-4), not Dota position 1-5. Training assembly does not read it.
    """
    rows: list[dict[str, Any]] = []
    for slot, player_id in enumerate(radiant_ids):
        rows.append(
            {
                "match_id": match_id,
                "side": "RADIANT",
                "slot_in_side": slot,
                "player_id": player_id,
                "hero_id": slot + 1,
            }
        )
    for slot, player_id in enumerate(dire_ids):
        rows.append(
            {
                "match_id": match_id,
                "side": "DIRE",
                "slot_in_side": slot,
                "player_id": player_id,
                "hero_id": slot + 6,
            }
        )
    return rows


def build_feature_store_config(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
) -> FeatureStoreConfig:
    """Write the analytical schema-v2 Parquet triple for `matches`/`players`
    under `tmp_path` (empty `draft_events`, exactly as
    `tests/features/pre_draft_helpers.py` does -- Step 4A never reads it
    either)."""
    matches_table = build_matches_table(matches, players)
    match_players_table = build_match_players_table(matches, players)
    draft_events_table = build_draft_events_table([])

    write_canonical_dataset(
        tmp_path,
        matches_table=matches_table,
        draft_events_table=draft_events_table,
        match_players_table=match_players_table,
    )

    return FeatureStoreConfig(
        matches_path=tmp_path / MATCHES_FILENAME,
        match_players_path=tmp_path / MATCH_PLAYERS_FILENAME,
        draft_events_path=tmp_path / DRAFT_EVENTS_FILENAME,
    )


def build_snapshot_store(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
) -> FeatureDuckDBConnection:
    """Open a `FeatureDuckDBConnection` over a fresh fixture. Caller
    must use this as a context manager (or close it) -- the returned
    connection must stay open for the lifetime of any
    `build_pre_draft_snapshot(store).to_frame()`-based call."""
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    return connect(config)


def build_dataset(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
) -> ModelReadyDataset:
    """End-to-end: fixture matches/players -> real Parquet -> DuckDB ->
    `PreDraftSnapshot` -> `ModelReadyDataset`, exercising the real
    Step 3/4A pipeline exactly as production code would."""
    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        snapshot: PreDraftSnapshot = build_pre_draft_snapshot(store)
        return build_model_ready_dataset(snapshot)
