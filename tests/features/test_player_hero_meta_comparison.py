"""Tests for Slice 6 → match-grain aggregation used by Slice 7.

Aggregation mirrors career Player × Hero (mean / min / zero-count for
counts; NULL-skipping mean for rates; Radiant − Dire). Not a production
feature.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from dota_predictor.features.player_hero_meta_comparison import (
    SLICE7_COMPARISON_COLUMNS,
    player_hero_meta_comparison_from_players,
    player_hero_meta_side_profile,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _player(
    match_id: int,
    player_id: int,
    *,
    side: str,
    team_id: int,
    start_time: datetime,
    same_version_matches: int,
    same_version_win_rate: float | None,
    recent_20_matches: int = 0,
    recent_20_win_rate: float | None = None,
    compatibility: float | None = None,
    player_share: float | None = None,
    hero_share: float | None = None,
    game_version_id: int = 176,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "start_time": start_time,
        "game_version_id": game_version_id,
        "side": side,
        "team_id": team_id,
        "player_hero_same_version_matches": same_version_matches,
        "player_hero_same_version_win_rate": same_version_win_rate,
        "player_hero_recent_20_matches": recent_20_matches,
        "player_hero_recent_20_win_rate": recent_20_win_rate,
        "player_hero_recent_role_compatibility": compatibility,
        "player_hero_share_at_expected_position": player_share,
        "hero_meta_share_at_expected_position": hero_share,
    }


def _side_players(
    match_id: int,
    *,
    side: str,
    team_id: int,
    start_time: datetime,
    player_ids: range,
    same_version_matches: list[int],
    same_version_win_rate: list[float | None],
) -> list[dict[str, object]]:
    return [
        _player(
            match_id,
            player_id,
            side=side,
            team_id=team_id,
            start_time=start_time,
            same_version_matches=matches,
            same_version_win_rate=rate,
        )
        for player_id, matches, rate in zip(
            player_ids, same_version_matches, same_version_win_rate, strict=True
        )
    ]


def test_slice7_comparison_columns_are_not_production_features() -> None:
    for column in SLICE7_COMPARISON_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS


def test_zero_same_version_matches_is_not_zero_win_rate() -> None:
    start = T0
    radiant = _side_players(
        1,
        side="RADIANT",
        team_id=1,
        start_time=start,
        player_ids=range(1, 6),
        same_version_matches=[0, 0, 0, 0, 0],
        same_version_win_rate=[None, None, None, None, None],
    )
    dire = _side_players(
        1,
        side="DIRE",
        team_id=2,
        start_time=start,
        player_ids=range(6, 11),
        same_version_matches=[2, 2, 2, 2, 2],
        same_version_win_rate=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    frame = pd.DataFrame(radiant + dire)
    side = player_hero_meta_side_profile(frame)
    radiant_row = side.loc[side["side"] == "RADIANT"].iloc[0]
    dire_row = side.loc[side["side"] == "DIRE"].iloc[0]
    assert radiant_row["mean_player_hero_same_version_matches"] == 0.0
    assert radiant_row["players_with_zero_player_hero_same_version_matches"] == 5
    assert pd.isna(radiant_row["mean_player_hero_same_version_win_rate"])
    assert dire_row["mean_player_hero_same_version_matches"] == 2.0
    assert dire_row["players_with_zero_player_hero_same_version_matches"] == 0
    assert dire_row["mean_player_hero_same_version_win_rate"] == 0.0

    comparison = player_hero_meta_comparison_from_players(frame)
    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["mean_player_hero_same_version_matches_diff"] == -2.0
    assert pd.isna(row["mean_player_hero_same_version_win_rate_diff"])


def test_comparison_uses_only_the_current_match_player_rows() -> None:
    early = _side_players(
        1,
        side="RADIANT",
        team_id=1,
        start_time=T0,
        player_ids=range(1, 6),
        same_version_matches=[0, 0, 0, 0, 0],
        same_version_win_rate=[None] * 5,
    ) + _side_players(
        1,
        side="DIRE",
        team_id=2,
        start_time=T0,
        player_ids=range(6, 11),
        same_version_matches=[0, 0, 0, 0, 0],
        same_version_win_rate=[None] * 5,
    )
    later = _side_players(
        2,
        side="RADIANT",
        team_id=1,
        start_time=T0 + timedelta(days=1),
        player_ids=range(1, 6),
        same_version_matches=[9, 9, 9, 9, 9],
        same_version_win_rate=[1.0] * 5,
    ) + _side_players(
        2,
        side="DIRE",
        team_id=2,
        start_time=T0 + timedelta(days=1),
        player_ids=range(6, 11),
        same_version_matches=[1, 1, 1, 1, 1],
        same_version_win_rate=[0.0] * 5,
    )
    comparison = player_hero_meta_comparison_from_players(pd.DataFrame(early + later))
    first = comparison.loc[comparison["match_id"] == 1].iloc[0]
    second = comparison.loc[comparison["match_id"] == 2].iloc[0]
    assert first["mean_player_hero_same_version_matches_diff"] == 0.0
    assert pd.isna(first["mean_player_hero_same_version_win_rate_diff"])
    assert second["mean_player_hero_same_version_matches_diff"] == 8.0
    assert second["mean_player_hero_same_version_win_rate_diff"] == 1.0


def test_rate_mean_skips_null_and_keeps_observed_zero() -> None:
    start = T0
    radiant = _side_players(
        1,
        side="RADIANT",
        team_id=1,
        start_time=start,
        player_ids=range(1, 6),
        same_version_matches=[2, 0, 2, 0, 2],
        same_version_win_rate=[0.0, None, 0.5, None, 1.0],
    )
    dire = _side_players(
        1,
        side="DIRE",
        team_id=2,
        start_time=start,
        player_ids=range(6, 11),
        same_version_matches=[0, 0, 0, 0, 0],
        same_version_win_rate=[None] * 5,
    )
    side = player_hero_meta_side_profile(pd.DataFrame(radiant + dire))
    radiant_row = side.loc[side["side"] == "RADIANT"].iloc[0]
    assert radiant_row["mean_player_hero_same_version_matches"] == pytest.approx(1.2)
    assert radiant_row["min_player_hero_same_version_matches"] == 0
    assert radiant_row["players_with_zero_player_hero_same_version_matches"] == 2
    assert radiant_row["mean_player_hero_same_version_win_rate"] == pytest.approx(0.5)
