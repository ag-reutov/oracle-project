"""Tests for Step 4B feature-column ablation subsets."""

from __future__ import annotations

from dota_predictor.features.draft_comparison import DRAFT_COMPARISON_METRIC_COLUMNS
from dota_predictor.features.draft_profile import DRAFT_PROFILE_METRIC_COLUMNS
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PLAYER_HISTORY_FEATURE_COLUMNS,
    ROSTER_CONTINUITY_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEAM_HISTORY_FEATURE_COLUMNS,
)
from dota_predictor.features.team_elo import TEAM_ELO_FEATURE_COLUMNS
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    DRAFT_COMPARISON_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_DRAFT_COMPARISON_COLUMNS,
    HISTORICAL_WITHOUT_ELO_COLUMNS,
)


def test_all_feature_columns_matches_snapshot_contract() -> None:
    assert ALL_FEATURE_COLUMNS == FEATURE_COLUMNS


def test_elo_only_feature_columns_match_team_elo_contract() -> None:
    assert ELO_ONLY_FEATURE_COLUMNS == TEAM_ELO_FEATURE_COLUMNS


def test_historical_without_elo_columns_match_snapshot_history_groups() -> None:
    assert HISTORICAL_WITHOUT_ELO_COLUMNS == (
        TEAM_HISTORY_FEATURE_COLUMNS
        + PLAYER_HISTORY_FEATURE_COLUMNS
        + ROSTER_CONTINUITY_FEATURE_COLUMNS
    )


def test_feature_sets_partition_all_snapshot_features_without_overlap() -> None:
    elo = set(ELO_ONLY_FEATURE_COLUMNS)
    historical = set(HISTORICAL_WITHOUT_ELO_COLUMNS)
    assert elo.isdisjoint(historical)
    assert elo | historical == set(FEATURE_COLUMNS)


def test_draft_comparison_features_are_the_full_metric_set() -> None:
    assert DRAFT_COMPARISON_FEATURE_COLUMNS == DRAFT_COMPARISON_METRIC_COLUMNS
    assert DRAFT_COMPARISON_FEATURE_COLUMNS == tuple(
        f"{metric}_diff" for metric in DRAFT_PROFILE_METRIC_COLUMNS
    )


def test_elo_plus_draft_comparison_is_elo_then_every_diff() -> None:
    assert ELO_PLUS_DRAFT_COMPARISON_COLUMNS == (
        TEAM_ELO_FEATURE_COLUMNS + DRAFT_COMPARISON_METRIC_COLUMNS
    )


def test_draft_comparison_features_are_not_in_pre_draft_snapshot() -> None:
    assert set(DRAFT_COMPARISON_FEATURE_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(DRAFT_COMPARISON_FEATURE_COLUMNS).isdisjoint(ALL_FEATURE_COLUMNS)
    assert TARGET_COLUMN not in DRAFT_COMPARISON_FEATURE_COLUMNS
