"""Tests for Slice 15 side profile and Radiant − Dire farming comparison.

PRE_DRAFT: rostered players only. Volume is evidence; shrunk B is strength.
Not a production FEATURE_COLUMNS column.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from dota_predictor.data.canonical_schema import InformationAvailability
from dota_predictor.features.availability import (
    PLAYER_FARMING_COMPARISON_COLUMN_AVAILABILITY,
    PLAYER_FARMING_STATE_COLUMN_AVAILABILITY,
    SnapshotStage,
    columns_allowed_for_stage,
)
from dota_predictor.features.player_farming_comparison import (
    FARMING_CAUSAL_B_COLUMN,
    PLAYER_FARMING_COMPARISON_COLUMNS,
    PLAYER_FARMING_COMPARISON_METRIC_COLUMNS,
    PLAYER_FARMING_FEATURE_COLUMNS,
    PLAYER_FARMING_REQUIRED_COLUMNS,
    PLAYER_FARMING_STATE_FEATURE_COLUMNS,
    merge_player_farming_comparison,
    player_farming_comparison_from_players,
    player_farming_side_profile,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_PLUS_PLAYER_FARMING_COLUMNS,
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    PLAYER_HERO_COMPARISON_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_FROZEN_SPECS,
    SLICE15_CANDIDATE_SPEC,
    SLICE15_FROZEN_SPECS,
)
from dota_predictor.training.feature_sets import (
    PLAYER_FARMING_COMPARISON_COLUMNS as FARMING_SPEC_COLUMNS,
)
from dota_predictor.training.player_farming_state import FROZEN_SHRINKAGE_K

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _player(
    match_id: int,
    player_id: int,
    *,
    side: str,
    team_id: int,
    prior_n: int,
    shrunk: float,
    raw_mean: float | None = None,
    weight: float | None = None,
    start_time: datetime = T0,
    game_version_id: int = 176,
    slot_in_side: int | None = None,
) -> dict[str, object]:
    if weight is None:
        weight = prior_n / (prior_n + FROZEN_SHRINKAGE_K) if prior_n > 0 else 0.0
    if raw_mean is None:
        raw_mean = None if prior_n == 0 else shrunk / weight if weight else None
    return {
        "match_id": match_id,
        "start_time": start_time,
        "game_version_id": game_version_id,
        "player_id": player_id,
        "hero_id": player_id,
        "side": side,
        "team_id": team_id,
        "slot_in_side": (
            slot_in_side if slot_in_side is not None else (player_id - 1) % 5
        ),
        "farming_prior_n": prior_n,
        "farming_prior_mean_b": raw_mean,
        "farming_shrinkage_weight": weight,
        "farming_shrunk_b": shrunk,
        FARMING_CAUSAL_B_COLUMN: 99.0,
        "num_last_hits": 10_000,
        "duration_seconds": 1000,
        "radiant_win": True,
    }


def _side(
    match_id: int,
    *,
    side: str,
    team_id: int,
    player_ids: range,
    prior_n: list[int],
    shrunk: list[float],
) -> list[dict[str, object]]:
    return [
        _player(
            match_id,
            player_id,
            side=side,
            team_id=team_id,
            prior_n=n,
            shrunk=s,
            slot_in_side=slot,
        )
        for slot, (player_id, n, s) in enumerate(
            zip(player_ids, prior_n, shrunk, strict=True)
        )
    ]


def test_slice15_comparison_columns_are_not_production_features() -> None:
    for column in PLAYER_FARMING_COMPARISON_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
        assert column not in ELO_PLUS_PLAYER_HERO_COLUMNS
        assert column not in PLAYER_HERO_COMPARISON_COLUMNS
    for spec in SLICE9_FROZEN_SPECS:
        assert set(PLAYER_FARMING_COMPARISON_METRIC_COLUMNS).isdisjoint(
            spec.feature_columns
        )
    assert set(PLAYER_FARMING_COMPARISON_METRIC_COLUMNS).isdisjoint(
        SLICE9_CANDIDATE_SPEC.feature_columns
    )
    for spec in POST_DRAFT_BLOCK_ABLATION_SPECS:
        assert set(PLAYER_FARMING_COMPARISON_METRIC_COLUMNS).isdisjoint(
            spec.feature_columns
        )
    assert FARMING_CAUSAL_B_COLUMN not in PLAYER_FARMING_REQUIRED_COLUMNS
    assert FARMING_CAUSAL_B_COLUMN not in PLAYER_FARMING_STATE_FEATURE_COLUMNS
    assert "hero_id" not in PLAYER_FARMING_REQUIRED_COLUMNS
    assert PLAYER_FARMING_FEATURE_COLUMNS == ("mean_farming_shrunk_b_diff",)
    assert PLAYER_FARMING_FEATURE_COLUMNS[0] in PLAYER_FARMING_COMPARISON_METRIC_COLUMNS
    assert FARMING_SPEC_COLUMNS == PLAYER_FARMING_COMPARISON_METRIC_COLUMNS


def test_slice15_spec_is_elo_plus_frozen_farming_feature() -> None:
    assert SLICE15_CANDIDATE_SPEC.feature_columns == ELO_PLUS_PLAYER_FARMING_COLUMNS
    assert ELO_PLUS_PLAYER_FARMING_COLUMNS == (
        SLICE9_FROZEN_SPECS[0].feature_columns + PLAYER_FARMING_FEATURE_COLUMNS
    )
    assert tuple(spec.name for spec in SLICE15_FROZEN_SPECS) == (
        "logistic_elo_only",
        "logistic_elo_plus_player_farming",
    )
    extra = set(SLICE15_CANDIDATE_SPEC.feature_columns) - set(
        SLICE9_FROZEN_SPECS[0].feature_columns
    )
    assert extra == set(PLAYER_FARMING_FEATURE_COLUMNS)
    assert extra.isdisjoint(FEATURE_COLUMNS)
    assert extra.isdisjoint(ALL_FEATURE_COLUMNS)


def test_five_player_side_means_and_radiant_minus_dire() -> None:
    radiant = _side(
        1,
        side="RADIANT",
        team_id=1,
        player_ids=range(1, 6),
        prior_n=[8, 8, 8, 8, 8],
        shrunk=[0.4, 0.4, 0.4, 0.4, 0.4],
    )
    dire = _side(
        1,
        side="DIRE",
        team_id=2,
        player_ids=range(6, 11),
        prior_n=[2, 2, 2, 2, 2],
        shrunk=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    frame = pd.DataFrame(radiant + dire)
    side = player_farming_side_profile(frame)
    radiant_row = side.loc[side["side"] == "RADIANT"].iloc[0]
    dire_row = side.loc[side["side"] == "DIRE"].iloc[0]
    assert radiant_row["mean_farming_prior_n"] == 8.0
    assert radiant_row["min_farming_prior_n"] == 8.0
    assert radiant_row["players_with_zero_farming_prior_n"] == 0
    assert radiant_row["mean_farming_shrunk_b"] == pytest.approx(0.4)
    assert dire_row["mean_farming_prior_n"] == 2.0
    assert dire_row["mean_farming_shrunk_b"] == pytest.approx(0.0)

    comparison = player_farming_comparison_from_players(frame)
    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["mean_farming_prior_n_diff"] == pytest.approx(6.0)
    assert row["mean_farming_shrunk_b_diff"] == pytest.approx(0.4)
    assert row["players_with_zero_farming_prior_n_diff"] == 0
    assert row["radiant_team_id"] == 1
    assert row["dire_team_id"] == 2


def test_all_cold_start_side_has_null_raw_mean_and_zero_shrunk() -> None:
    radiant = _side(
        1,
        side="RADIANT",
        team_id=1,
        player_ids=range(1, 6),
        prior_n=[0, 0, 0, 0, 0],
        shrunk=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    dire = _side(
        1,
        side="DIRE",
        team_id=2,
        player_ids=range(6, 11),
        prior_n=[4, 4, 4, 4, 4],
        shrunk=[0.2, 0.2, 0.2, 0.2, 0.2],
    )
    frame = pd.DataFrame(radiant + dire)
    side = player_farming_side_profile(frame)
    radiant_row = side.loc[side["side"] == "RADIANT"].iloc[0]
    assert radiant_row["mean_farming_prior_n"] == 0.0
    assert radiant_row["players_with_zero_farming_prior_n"] == 5
    assert pd.isna(radiant_row["mean_farming_prior_mean_b"])
    assert radiant_row["mean_farming_shrunk_b"] == 0.0
    assert radiant_row["mean_farming_shrinkage_weight"] == 0.0

    comparison = player_farming_comparison_from_players(frame)
    row = comparison.iloc[0]
    assert pd.isna(row["mean_farming_prior_mean_b_diff"])
    assert row["mean_farming_shrunk_b_diff"] == pytest.approx(-0.2)


def test_comparison_does_not_require_hero_or_current_box_score() -> None:
    frame = pd.DataFrame(
        _side(
            1,
            side="RADIANT",
            team_id=1,
            player_ids=range(1, 6),
            prior_n=[5, 5, 5, 5, 5],
            shrunk=[1.0, 1.0, 1.0, 1.0, 1.0],
        )
        + _side(
            1,
            side="DIRE",
            team_id=2,
            player_ids=range(6, 11),
            prior_n=[5, 5, 5, 5, 5],
            shrunk=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
    )
    original = player_farming_comparison_from_players(frame)
    stripped = frame.drop(
        columns=[
            "hero_id",
            FARMING_CAUSAL_B_COLUMN,
            "num_last_hits",
            "duration_seconds",
            "radiant_win",
        ]
    )
    rerun = player_farming_comparison_from_players(stripped)
    pd.testing.assert_frame_equal(original, rerun)


def test_current_causal_b_cannot_change_comparison() -> None:
    frame = pd.DataFrame(
        _side(
            1,
            side="RADIANT",
            team_id=1,
            player_ids=range(1, 6),
            prior_n=[10, 10, 10, 10, 10],
            shrunk=[0.5, 0.5, 0.5, 0.5, 0.5],
        )
        + _side(
            1,
            side="DIRE",
            team_id=2,
            player_ids=range(6, 11),
            prior_n=[10, 10, 10, 10, 10],
            shrunk=[-0.5, -0.5, -0.5, -0.5, -0.5],
        )
    )
    original = player_farming_comparison_from_players(frame)
    mutated = frame.copy()
    mutated[FARMING_CAUSAL_B_COLUMN] = 1_000.0
    rerun = player_farming_comparison_from_players(mutated)
    pd.testing.assert_frame_equal(original, rerun)
    assert original.iloc[0]["mean_farming_shrunk_b_diff"] == pytest.approx(1.0)


def test_merge_joins_feature_without_touching_identity() -> None:
    matches = pd.DataFrame(
        {"match_id": [1], "radiant_win": [True], "radiant_team_elo": [1500.0]}
    )
    comparison = pd.DataFrame(
        _side(
            1,
            side="RADIANT",
            team_id=1,
            player_ids=range(1, 6),
            prior_n=[1, 1, 1, 1, 1],
            shrunk=[0.2, 0.2, 0.2, 0.2, 0.2],
        )
        + _side(
            1,
            side="DIRE",
            team_id=2,
            player_ids=range(6, 11),
            prior_n=[1, 1, 1, 1, 1],
            shrunk=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
    )
    diffs = player_farming_comparison_from_players(comparison)
    merged = merge_player_farming_comparison(matches, diffs)
    assert "radiant_win" in merged.columns
    assert "mean_farming_shrunk_b_diff" in merged.columns
    assert merged.iloc[0]["mean_farming_shrunk_b_diff"] == pytest.approx(0.2)


def test_farming_state_metrics_are_pre_draft_causal_b_is_not() -> None:
    assert (
        PLAYER_FARMING_STATE_COLUMN_AVAILABILITY["farming_shrunk_b"]
        == InformationAvailability.PRE_DRAFT
    )
    assert (
        PLAYER_FARMING_STATE_COLUMN_AVAILABILITY["farming_prior_n"]
        == InformationAvailability.PRE_DRAFT
    )
    assert (
        PLAYER_FARMING_STATE_COLUMN_AVAILABILITY[FARMING_CAUSAL_B_COLUMN]
        == InformationAvailability.POST_MATCH
    )
    assert (
        PLAYER_FARMING_STATE_COLUMN_AVAILABILITY["hero_id"]
        == InformationAvailability.DRAFT
    )
    assert (
        PLAYER_FARMING_STATE_COLUMN_AVAILABILITY["player_id"]
        == InformationAvailability.PRE_DRAFT
    )
    pre_draft = columns_allowed_for_stage(
        "player_farming_state", SnapshotStage.PRE_DRAFT
    )
    post_draft = columns_allowed_for_stage(
        "player_farming_state", SnapshotStage.POST_DRAFT
    )
    assert "farming_shrunk_b" in pre_draft
    assert "player_id" in pre_draft
    assert FARMING_CAUSAL_B_COLUMN not in pre_draft
    assert "hero_id" not in pre_draft
    assert "hero_id" in post_draft
    assert FARMING_CAUSAL_B_COLUMN not in post_draft


def test_farming_comparison_metrics_are_pre_draft() -> None:
    for column in PLAYER_FARMING_COMPARISON_METRIC_COLUMNS:
        assert (
            PLAYER_FARMING_COMPARISON_COLUMN_AVAILABILITY[column]
            == InformationAvailability.PRE_DRAFT
        )
    pre_draft = columns_allowed_for_stage(
        "player_farming_comparison", SnapshotStage.PRE_DRAFT
    )
    assert set(PLAYER_FARMING_FEATURE_COLUMNS).issubset(pre_draft)
    assert "hero_id" not in pre_draft
    assert FARMING_CAUSAL_B_COLUMN not in pre_draft
    assert "mean_farming_shrunk_b_diff" in pre_draft
