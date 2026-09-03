"""Tests for Slice 19 side profile and Radiant − Dire combat comparison.

PRE_DRAFT: rostered players only. Volume is evidence; shrunk C is strength.
Not a production FEATURE_COLUMNS column.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from dota_predictor.data.canonical_schema import InformationAvailability
from dota_predictor.features.availability import (
    PLAYER_COMBAT_COMPARISON_COLUMN_AVAILABILITY,
    PLAYER_COMBAT_STATE_COLUMN_AVAILABILITY,
    SnapshotStage,
    columns_allowed_for_stage,
)
from dota_predictor.features.player_combat_comparison import (
    COMBAT_CAUSAL_C_COLUMN,
    PLAYER_COMBAT_COMPARISON_COLUMNS,
    PLAYER_COMBAT_COMPARISON_EVIDENCE_COLUMNS,
    PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS,
    PLAYER_COMBAT_FEATURE_COLUMNS,
    PLAYER_COMBAT_REQUIRED_COLUMNS,
    PLAYER_COMBAT_STATE_FEATURE_COLUMNS,
    CombatRosterError,
    merge_player_combat_comparison,
    player_combat_comparison_from_players,
    player_combat_side_profile,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_PLUS_PLAYER_COMBAT_COLUMNS,
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    PLAYER_HERO_COMPARISON_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_FROZEN_SPECS,
    SLICE19_CANDIDATE_SPEC,
    SLICE19_FROZEN_SPECS,
)
from dota_predictor.training.feature_sets import (
    PLAYER_COMBAT_COMPARISON_COLUMNS as COMBAT_SPEC_COLUMNS,
)
from dota_predictor.training.player_combat_state import FROZEN_COMBAT_SHRINKAGE_K
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
    start_time: datetime = T0,
    game_version_id: int = 176,
    slot_in_side: int | None = None,
) -> dict[str, object]:
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
        "position": f"POSITION_{(slot_in_side or (player_id - 1) % 5) + 1}",
        "combat_prior_n": prior_n,
        "combat_shrunk_c": shrunk,
        COMBAT_CAUSAL_C_COLUMN: 99.0,
        "hero_damage": 10_000,
        "kills": 10,
        "assists": 10,
        "deaths": 10,
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


def test_slice19_comparison_columns_are_not_production_features() -> None:
    for column in PLAYER_COMBAT_COMPARISON_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
        assert column not in ELO_PLUS_PLAYER_HERO_COLUMNS
        assert column not in PLAYER_HERO_COMPARISON_COLUMNS
    for spec in SLICE9_FROZEN_SPECS:
        assert set(PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS).isdisjoint(
            spec.feature_columns
        )
    assert set(PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS).isdisjoint(
        SLICE9_CANDIDATE_SPEC.feature_columns
    )
    for spec in POST_DRAFT_BLOCK_ABLATION_SPECS:
        assert set(PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS).isdisjoint(
            spec.feature_columns
        )
    assert COMBAT_CAUSAL_C_COLUMN not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert COMBAT_CAUSAL_C_COLUMN not in PLAYER_COMBAT_STATE_FEATURE_COLUMNS
    assert "hero_id" not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert "position" not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert PLAYER_COMBAT_FEATURE_COLUMNS == ("mean_combat_shrunk_c_diff",)
    assert PLAYER_COMBAT_FEATURE_COLUMNS[0] in PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS
    assert COMBAT_SPEC_COLUMNS == PLAYER_COMBAT_FEATURE_COLUMNS


def test_slice19_spec_is_elo_plus_frozen_combat_feature() -> None:
    assert SLICE19_CANDIDATE_SPEC.feature_columns == ELO_PLUS_PLAYER_COMBAT_COLUMNS
    assert ELO_PLUS_PLAYER_COMBAT_COLUMNS == (
        SLICE9_FROZEN_SPECS[0].feature_columns + PLAYER_COMBAT_FEATURE_COLUMNS
    )
    assert tuple(spec.name for spec in SLICE19_FROZEN_SPECS) == (
        "logistic_elo_only",
        "logistic_elo_plus_player_combat",
    )
    extra = set(SLICE19_CANDIDATE_SPEC.feature_columns) - set(
        SLICE9_FROZEN_SPECS[0].feature_columns
    )
    assert extra == set(PLAYER_COMBAT_FEATURE_COLUMNS)
    assert extra.isdisjoint(FEATURE_COLUMNS)
    assert extra.isdisjoint(ALL_FEATURE_COLUMNS)
    assert extra.isdisjoint(PLAYER_COMBAT_COMPARISON_EVIDENCE_COLUMNS)
    assert FROZEN_COMBAT_SHRINKAGE_K == 20.0
    assert FROZEN_SHRINKAGE_K == 5.0
    assert len(FEATURE_COLUMNS) == 33


def test_five_player_side_means_and_radiant_minus_dire() -> None:
    radiant = _side(
        1,
        side="RADIANT",
        team_id=1,
        player_ids=range(1, 6),
        prior_n=[8, 8, 8, 8, 8],
        shrunk=[0.10, 0.20, 0.30, 0.40, 0.50],
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
    side = player_combat_side_profile(frame)
    radiant_row = side.loc[side["side"] == "RADIANT"].iloc[0]
    dire_row = side.loc[side["side"] == "DIRE"].iloc[0]
    assert radiant_row["mean_combat_shrunk_c"] == pytest.approx(0.30)
    assert radiant_row["combat_prior_n_sum"] == 40.0
    assert radiant_row["combat_cold_start_count"] == 0
    assert dire_row["mean_combat_shrunk_c"] == pytest.approx(0.0)
    assert dire_row["combat_prior_n_sum"] == 10.0

    comparison = player_combat_comparison_from_players(frame)
    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["radiant_mean_combat_shrunk_c"] == pytest.approx(0.30)
    assert row["dire_mean_combat_shrunk_c"] == pytest.approx(0.0)
    assert row["mean_combat_shrunk_c_diff"] == pytest.approx(0.30)
    assert row["radiant_team_id"] == 1
    assert row["dire_team_id"] == 2


def test_cold_start_contributes_zero_and_is_not_renormalized() -> None:
    radiant = _side(
        1,
        side="RADIANT",
        team_id=1,
        player_ids=range(1, 6),
        prior_n=[0, 8, 8, 8, 8],
        shrunk=[0.0, 0.40, 0.40, 0.40, 0.40],
    )
    dire = _side(
        1,
        side="DIRE",
        team_id=2,
        player_ids=range(6, 11),
        prior_n=[4, 4, 4, 4, 4],
        shrunk=[0.20, 0.20, 0.20, 0.20, 0.20],
    )
    frame = pd.DataFrame(radiant + dire)
    side = player_combat_side_profile(frame)
    radiant_row = side.loc[side["side"] == "RADIANT"].iloc[0]
    assert radiant_row["combat_cold_start_count"] == 1
    assert radiant_row["mean_combat_shrunk_c"] == pytest.approx(0.32)
    assert radiant_row["mean_combat_shrunk_c"] != pytest.approx(0.40)

    comparison = player_combat_comparison_from_players(frame)
    row = comparison.iloc[0]
    assert row["mean_combat_shrunk_c_diff"] == pytest.approx(0.32 - 0.20)
    assert row["radiant_combat_cold_start_count"] == 1


def test_swapping_sides_negates_the_diff_exactly() -> None:
    frame = pd.DataFrame(
        _side(
            1,
            side="RADIANT",
            team_id=1,
            player_ids=range(1, 6),
            prior_n=[5, 5, 5, 5, 5],
            shrunk=[0.4, 0.4, 0.4, 0.4, 0.4],
        )
        + _side(
            1,
            side="DIRE",
            team_id=2,
            player_ids=range(6, 11),
            prior_n=[5, 5, 5, 5, 5],
            shrunk=[-0.2, -0.2, -0.2, -0.2, -0.2],
        )
    )
    original = player_combat_comparison_from_players(frame).iloc[0]
    swapped = frame.copy()
    swapped["side"] = swapped["side"].map({"RADIANT": "DIRE", "DIRE": "RADIANT"})
    rerun = player_combat_comparison_from_players(swapped).iloc[0]
    assert rerun["radiant_mean_combat_shrunk_c"] == pytest.approx(
        original["dire_mean_combat_shrunk_c"]
    )
    assert rerun["dire_mean_combat_shrunk_c"] == pytest.approx(
        original["radiant_mean_combat_shrunk_c"]
    )
    assert rerun["mean_combat_shrunk_c_diff"] == pytest.approx(
        -original["mean_combat_shrunk_c_diff"]
    )


def test_comparison_does_not_require_hero_position_or_current_box_score() -> None:
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
    original = player_combat_comparison_from_players(frame)
    stripped = frame.drop(
        columns=[
            "hero_id",
            "position",
            COMBAT_CAUSAL_C_COLUMN,
            "hero_damage",
            "kills",
            "assists",
            "deaths",
            "radiant_win",
        ]
    )
    rerun = player_combat_comparison_from_players(stripped)
    pd.testing.assert_frame_equal(original, rerun)


def test_current_causal_c_cannot_change_comparison() -> None:
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
    original = player_combat_comparison_from_players(frame)
    mutated = frame.copy()
    mutated[COMBAT_CAUSAL_C_COLUMN] = 1_000.0
    rerun = player_combat_comparison_from_players(mutated)
    pd.testing.assert_frame_equal(original, rerun)
    assert original.iloc[0]["mean_combat_shrunk_c_diff"] == pytest.approx(1.0)


def test_incomplete_roster_does_not_silently_average() -> None:
    frame = pd.DataFrame(
        _side(
            1,
            side="RADIANT",
            team_id=1,
            player_ids=range(1, 5),
            prior_n=[8, 8, 8, 8],
            shrunk=[0.4, 0.4, 0.4, 0.4],
        )
        + _side(
            1,
            side="DIRE",
            team_id=2,
            player_ids=range(6, 11),
            prior_n=[4, 4, 4, 4, 4],
            shrunk=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
    )
    with pytest.raises(CombatRosterError, match="incomplete or malformed"):
        player_combat_comparison_from_players(frame)


def test_duplicate_and_cross_side_player_ids_are_rejected() -> None:
    duplicate_side = pd.DataFrame(
        _side(
            1,
            side="RADIANT",
            team_id=1,
            player_ids=range(1, 6),
            prior_n=[1, 1, 1, 1, 1],
            shrunk=[0.1, 0.1, 0.1, 0.1, 0.1],
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
    duplicate_side.loc[0, "player_id"] = duplicate_side.loc[1, "player_id"]
    with pytest.raises(CombatRosterError):
        player_combat_comparison_from_players(duplicate_side)

    both_sides = pd.DataFrame(
        _side(
            2,
            side="RADIANT",
            team_id=1,
            player_ids=range(1, 6),
            prior_n=[1, 1, 1, 1, 1],
            shrunk=[0.1, 0.1, 0.1, 0.1, 0.1],
        )
        + _side(
            2,
            side="DIRE",
            team_id=2,
            player_ids=range(6, 11),
            prior_n=[1, 1, 1, 1, 1],
            shrunk=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
    )
    both_sides.loc[both_sides["side"] == "DIRE", "player_id"] = [1, 7, 8, 9, 10]
    with pytest.raises(CombatRosterError):
        player_combat_comparison_from_players(both_sides)


def test_missing_combat_state_is_not_a_valid_five_player_mean() -> None:
    frame = pd.DataFrame(
        _side(
            1,
            side="RADIANT",
            team_id=1,
            player_ids=range(1, 6),
            prior_n=[1, 1, 1, 1, 1],
            shrunk=[0.1, 0.1, 0.1, 0.1, 0.1],
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
    frame.loc[0, "combat_shrunk_c"] = float("nan")
    with pytest.raises(CombatRosterError):
        player_combat_comparison_from_players(frame)


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
    diffs = player_combat_comparison_from_players(comparison)
    merged = merge_player_combat_comparison(matches, diffs)
    assert "radiant_win" in merged.columns
    assert "mean_combat_shrunk_c_diff" in merged.columns
    assert merged.iloc[0]["mean_combat_shrunk_c_diff"] == pytest.approx(0.2)


def test_combat_state_metrics_are_pre_draft_causal_c_is_not() -> None:
    assert (
        PLAYER_COMBAT_STATE_COLUMN_AVAILABILITY["combat_shrunk_c"]
        == InformationAvailability.PRE_DRAFT
    )
    assert (
        PLAYER_COMBAT_STATE_COLUMN_AVAILABILITY["combat_prior_n"]
        == InformationAvailability.PRE_DRAFT
    )
    assert (
        PLAYER_COMBAT_STATE_COLUMN_AVAILABILITY[COMBAT_CAUSAL_C_COLUMN]
        == InformationAvailability.POST_MATCH
    )
    assert (
        PLAYER_COMBAT_STATE_COLUMN_AVAILABILITY["hero_id"]
        == InformationAvailability.DRAFT
    )
    assert (
        PLAYER_COMBAT_STATE_COLUMN_AVAILABILITY["player_id"]
        == InformationAvailability.PRE_DRAFT
    )
    pre_draft = columns_allowed_for_stage(
        "player_combat_state", SnapshotStage.PRE_DRAFT
    )
    post_draft = columns_allowed_for_stage(
        "player_combat_state", SnapshotStage.POST_DRAFT
    )
    assert "combat_shrunk_c" in pre_draft
    assert "player_id" in pre_draft
    assert COMBAT_CAUSAL_C_COLUMN not in pre_draft
    assert "hero_id" not in pre_draft
    assert "hero_id" in post_draft
    assert COMBAT_CAUSAL_C_COLUMN not in post_draft


def test_combat_comparison_metrics_are_pre_draft() -> None:
    for column in PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS:
        assert (
            PLAYER_COMBAT_COMPARISON_COLUMN_AVAILABILITY[column]
            == InformationAvailability.PRE_DRAFT
        )
    pre_draft = columns_allowed_for_stage(
        "player_combat_comparison", SnapshotStage.PRE_DRAFT
    )
    assert set(PLAYER_COMBAT_FEATURE_COLUMNS).issubset(pre_draft)
    assert "hero_id" not in pre_draft
    assert COMBAT_CAUSAL_C_COLUMN not in pre_draft
    assert "mean_combat_shrunk_c_diff" in pre_draft
