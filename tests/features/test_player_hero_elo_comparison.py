"""Tests for Slice 10 side profile and Radiant − Dire comparison.

Not a production feature. Volume is evidence; residual is strength.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from dota_predictor.features.player_hero_elo import (
    DEFAULT_SHRINKAGE_K,
    PLAYER_HERO_ELO_COLUMNS,
    PLAYER_HERO_ELO_IDENTITY_COLUMNS,
)
from dota_predictor.features.player_hero_elo_comparison import (
    PLAYER_HERO_ELO_COMPARISON_COLUMNS,
    PLAYER_HERO_ELO_COMPARISON_METRIC_COLUMNS,
    player_hero_elo_comparison_from_players,
    player_hero_elo_side_profile,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    PLAYER_HERO_COMPARISON_COLUMNS,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_FROZEN_SPECS,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _player(
    match_id: int,
    player_id: int,
    *,
    side: str,
    team_id: int,
    hero_id: int,
    prior_games: int,
    prior_wins: float,
    prior_expected: float,
    start_time: datetime = T0,
    game_version_id: int = 176,
) -> dict[str, object]:
    residual = (
        (prior_wins - prior_expected) / prior_games if prior_games > 0 else None
    )
    weight = (
        prior_games / (prior_games + DEFAULT_SHRINKAGE_K) if prior_games > 0 else 0.0
    )
    shrunk = 0.0 if residual is None else weight * residual
    return {
        "match_id": match_id,
        "start_time": start_time,
        "game_version_id": game_version_id,
        "player_id": player_id,
        "hero_id": hero_id,
        "hero_name": f"Hero {hero_id}",
        "side": side,
        "team_id": team_id,
        "slot_in_side": (player_id - 1) % 5,
        "prior_games_on_hero": prior_games,
        "prior_wins_on_hero": prior_wins,
        "prior_elo_expected_wins_on_hero": prior_expected,
        "prior_wins_minus_expected_on_hero": prior_wins - prior_expected,
        "mean_outcome_residual_on_hero": residual,
        "shrunk_outcome_residual_on_hero": shrunk,
        "shrinkage_weight_on_hero": weight,
    }


def _side(
    match_id: int,
    *,
    side: str,
    team_id: int,
    player_ids: range,
    hero_ids: list[int],
    games: list[int],
    wins: list[float],
    expected: list[float],
) -> list[dict[str, object]]:
    return [
        _player(
            match_id,
            player_id,
            side=side,
            team_id=team_id,
            hero_id=hero_id,
            prior_games=n,
            prior_wins=w,
            prior_expected=e,
        )
        for player_id, hero_id, n, w, e in zip(
            player_ids, hero_ids, games, wins, expected, strict=True
        )
    ]


def test_slice10_comparison_columns_are_not_production_features() -> None:
    for column in PLAYER_HERO_ELO_COMPARISON_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
        assert column not in ELO_PLUS_PLAYER_HERO_COLUMNS
        assert column not in PLAYER_HERO_COMPARISON_COLUMNS
    for spec in SLICE9_FROZEN_SPECS:
        assert set(PLAYER_HERO_ELO_COMPARISON_METRIC_COLUMNS).isdisjoint(
            spec.feature_columns
        )
    assert set(PLAYER_HERO_ELO_COMPARISON_METRIC_COLUMNS).isdisjoint(
        SLICE9_CANDIDATE_SPEC.feature_columns
    )


def test_identity_columns_cover_required_player_fields() -> None:
    assert "hero_id" in PLAYER_HERO_ELO_IDENTITY_COLUMNS
    assert "player_id" in PLAYER_HERO_ELO_IDENTITY_COLUMNS
    assert list(PLAYER_HERO_ELO_COLUMNS)[: len(PLAYER_HERO_ELO_IDENTITY_COLUMNS)] == list(
        PLAYER_HERO_ELO_IDENTITY_COLUMNS
    )


def test_five_player_side_means_and_radiant_minus_dire() -> None:
    radiant = _side(
        1,
        side="RADIANT",
        team_id=1,
        player_ids=range(1, 6),
        hero_ids=[1, 2, 3, 4, 5],
        games=[8, 8, 8, 8, 8],
        wins=[5.0, 5.0, 5.0, 5.0, 5.0],
        expected=[4.0, 4.0, 4.0, 4.0, 4.0],
    )
    dire = _side(
        1,
        side="DIRE",
        team_id=2,
        player_ids=range(6, 11),
        hero_ids=[6, 7, 8, 9, 10],
        games=[2, 2, 2, 2, 2],
        wins=[1.0, 1.0, 1.0, 1.0, 1.0],
        expected=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    frame = pd.DataFrame(radiant + dire)
    side = player_hero_elo_side_profile(frame)
    radiant_row = side.loc[side["side"] == "RADIANT"].iloc[0]
    dire_row = side.loc[side["side"] == "DIRE"].iloc[0]
    assert radiant_row["mean_player_hero_prior_games"] == 8.0
    assert radiant_row["min_player_hero_prior_games"] == 8.0
    assert radiant_row["players_with_zero_player_hero_prior_games"] == 0
    assert radiant_row["mean_player_hero_outcome_residual"] == pytest.approx(0.125)
    assert dire_row["mean_player_hero_prior_games"] == 2.0
    assert dire_row["mean_player_hero_outcome_residual"] == pytest.approx(0.0)

    comparison = player_hero_elo_comparison_from_players(frame)
    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["mean_player_hero_prior_games_diff"] == pytest.approx(6.0)
    assert row["mean_player_hero_outcome_residual_diff"] == pytest.approx(0.125)
    assert row["players_with_zero_player_hero_prior_games_diff"] == 0


def test_all_cold_start_side_has_null_raw_mean_and_zero_shrunk() -> None:
    radiant = _side(
        1,
        side="RADIANT",
        team_id=1,
        player_ids=range(1, 6),
        hero_ids=[1, 2, 3, 4, 5],
        games=[0, 0, 0, 0, 0],
        wins=[0.0, 0.0, 0.0, 0.0, 0.0],
        expected=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    dire = _side(
        1,
        side="DIRE",
        team_id=2,
        player_ids=range(6, 11),
        hero_ids=[6, 7, 8, 9, 10],
        games=[4, 4, 4, 4, 4],
        wins=[3.0, 3.0, 3.0, 3.0, 3.0],
        expected=[2.0, 2.0, 2.0, 2.0, 2.0],
    )
    frame = pd.DataFrame(radiant + dire)
    side = player_hero_elo_side_profile(frame)
    radiant_row = side.loc[side["side"] == "RADIANT"].iloc[0]
    assert radiant_row["mean_player_hero_prior_games"] == 0.0
    assert radiant_row["players_with_zero_player_hero_prior_games"] == 5
    assert pd.isna(radiant_row["mean_player_hero_outcome_residual"])
    assert radiant_row["mean_player_hero_shrunk_residual"] == 0.0
    assert radiant_row["mean_player_hero_shrinkage_weight"] == 0.0

    comparison = player_hero_elo_comparison_from_players(frame)
    row = comparison.iloc[0]
    assert pd.isna(row["mean_player_hero_outcome_residual_diff"])
    assert row["mean_player_hero_shrunk_residual_diff"] == pytest.approx(
        0.0
        - (4.0 / (4.0 + DEFAULT_SHRINKAGE_K)) * 0.25
    )
