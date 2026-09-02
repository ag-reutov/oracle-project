"""Slice 13 farming-target diagnostics: no rating, no features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import (
    build_feature_store_config,
    build_snapshot_store,
    match_row,
    player_rows,
)

from dota_predictor.data.canonical_schema import MATCH_PLAYER_BOX_SCORE_COLUMNS
from dota_predictor.features.duckdb_layer import MATCH_PLAYERS_VIEW, connect
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.farming_performance_target import (
    CANDIDATE_A,
    CANDIDATE_B,
    CANDIDATE_C,
    CANDIDATE_D,
    FARMING_CANDIDATE_COLUMN_NAMES,
    SPARSE_GROUP_MIN_N,
    attach_farming_candidates,
    build_team_spells,
    hero_excluded_prior_history,
    pooled_group_labels,
    position_duration_group_residual,
    prior_history_excluding,
    run_farming_performance_target_diagnostics,
    team_period_centered,
    team_switcher_table,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.player_performance_target import (
    CANDIDATE_COLUMN_NAMES,
    attach_candidate_targets,
    build_player_performance_frame,
    explicit_position_mask,
    ols_residual,
    per_minute,
    position_duration_residual,
    position_standardized,
    prior_player_history,
    restrict_development,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END

RADIANT_IDS = (11, 12, 13, 14, 15)
DIRE_IDS = (21, 22, 23, 24, 25)
POSITIONS = (
    "POSITION_1",
    "POSITION_2",
    "POSITION_3",
    "POSITION_4",
    "POSITION_5",
)


def _box_for_slot(slot: int) -> dict[str, int]:
    return {
        "kills": 10 - slot,
        "deaths": slot,
        "assists": 5 + slot,
        "gold_per_minute": 600 - slot * 80,
        "experience_per_minute": 550 - slot * 70,
        "num_last_hits": 300 - slot * 60,
        "num_denies": 20 - slot * 4,
        "networth": 20000 - slot * 3000,
        "hero_damage": 25000 - slot * 4000,
        "tower_damage": 3000 - slot * 500,
        "hero_healing": slot * 400,
        "level": 25 - slot,
    }


def _annotate_players(
    rows: list[dict[str, object]],
    *,
    extra_by_player: dict[int, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        slot = int(item["slot_in_side"])
        item["position"] = POSITIONS[slot]
        item.update(_box_for_slot(slot))
        player_id = int(item["player_id"])
        if extra_by_player and player_id in extra_by_player:
            item.update(extra_by_player[player_id])
        annotated.append(item)
    return annotated


def _frame_from_store(
    tmp_path: Path,
    *,
    matches: list[dict[str, object]],
    players: list[dict[str, object]],
) -> pd.DataFrame:
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        return build_player_performance_frame(store)


def _farming_frame() -> pd.DataFrame:
    """Hand-checkable last-hit residuals: one position, two heroes."""
    frame = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4, 5, 6],
            "player_id": [11, 11, 12, 12, 13, 13],
            "hero_id": [1, 1, 2, 2, 1, 2],
            "team_id": [100, 100, 100, 200, 200, 200],
            "side": ["RADIANT"] * 6,
            "position": ["POSITION_1"] * 6,
            "position_number": [1.0] * 6,
            "team_won": [1, 0, 1, 0, 1, 0],
            "elo_expected_win": [0.5] * 6,
            "game_version_id": [176, 176, 176, 177, 177, 177],
            "start_time": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 3, 1, tzinfo=UTC),
                datetime(2026, 4, 1, tzinfo=UTC),
            ],
            "duration_seconds": [1000.0, 2000.0, 1000.0, 2000.0, 1000.0, 2000.0],
            "num_last_hits": [100.0, 200.0, 150.0, 300.0, 100.0, 300.0],
            "kills": [5] * 6,
            "deaths": [2] * 6,
            "assists": [8] * 6,
            "gold_per_minute": [500] * 6,
            "experience_per_minute": [450] * 6,
            "num_denies": [10] * 6,
            "networth": [15000] * 6,
            "hero_damage": [12000] * 6,
            "tower_damage": [1000] * 6,
            "hero_healing": [0] * 6,
            "level": [20] * 6,
        }
    )
    return frame


def test_candidate_a_matches_unchanged_slice12_baseline() -> None:
    frame = _farming_frame()
    slice12 = attach_candidate_targets(frame)
    slice13 = attach_farming_candidates(frame, sparse_min_n=2)
    left = slice12[CANDIDATE_A].to_numpy(dtype=float)
    right = slice13[CANDIDATE_A].to_numpy(dtype=float)
    np.testing.assert_allclose(left, right, equal_nan=True)
    independent = position_standardized(
        frame.assign(
            last_hits_per_minute=per_minute(
                frame["num_last_hits"], frame["duration_seconds"]
            )
        ),
        "last_hits_per_minute",
    )
    np.testing.assert_allclose(
        right, independent.to_numpy(dtype=float), equal_nan=True
    )


def test_position_duration_residual_matches_hand_computed_line() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [1.0, 1.0, 1.0],
            "duration_seconds": [1000.0, 2000.0, 3000.0],
            "last_hits_per_minute": [2.0, 3.0, 4.0],
            "num_last_hits": [100.0 / 3.0, 100.0, 200.0],
            "hero_id": [1, 1, 1],
            "player_id": [11, 11, 11],
            "match_id": [1, 2, 3],
            "team_id": [100, 100, 100],
            "team_won": [1, 0, 1],
            "elo_expected_win": [0.5, 0.5, 0.5],
            "game_version_id": [176, 176, 176],
            "start_time": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 3, 1, tzinfo=UTC),
            ],
            "position": ["POSITION_1", "POSITION_1", "POSITION_1"],
        }
    )
    residual = position_duration_residual(frame, "last_hits_per_minute")
    assert residual.to_numpy() == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    attached = attach_farming_candidates(frame, sparse_min_n=1)
    assert attached["_lhpm_position_duration_residual"].to_numpy() == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-9
    )
    assert attached[CANDIDATE_B].to_numpy() == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_position_duration_residual_nonzero_matches_lstsq() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [1.0, 1.0, 1.0],
            "duration_seconds": [1000.0, 2000.0, 3000.0],
            "last_hits_per_minute": [2.0, 3.0, 5.0],
        }
    )
    residual = position_duration_residual(frame, "last_hits_per_minute")
    design = np.column_stack(
        [np.ones(3), frame["duration_seconds"].to_numpy(dtype=float)]
    )
    y = frame["last_hits_per_minute"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    expected = y - design @ coef
    np.testing.assert_allclose(residual.to_numpy(dtype=float), expected, atol=1e-9)
    assert abs(float(np.corrcoef(residual, frame["duration_seconds"])[0, 1])) < 1e-9


def test_hero_adjustment_removes_additive_hero_effect() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [1.0, 1.0, 1.0, 1.0],
            "duration_seconds": [1000.0, 2000.0, 1000.0, 2000.0],
            "last_hits_per_minute": [2.0, 4.0, 10.0, 12.0],
            "num_last_hits": [
                2.0 * 1000.0 / 60.0,
                4.0 * 2000.0 / 60.0,
                10.0 * 1000.0 / 60.0,
                12.0 * 2000.0 / 60.0,
            ],
            "hero_id": [1, 1, 2, 2],
            "player_id": [11, 11, 12, 12],
            "match_id": [1, 2, 3, 4],
            "team_id": [100, 100, 200, 200],
            "team_won": [1, 0, 1, 0],
            "elo_expected_win": [0.5] * 4,
            "game_version_id": [176] * 4,
            "start_time": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, tzinfo=UTC),
            ],
            "position": ["POSITION_1"] * 4,
        }
    )
    labels, summary = pooled_group_labels(
        frame, ("hero_id",), min_n=2, eligible=explicit_position_mask(frame)
    )
    assert int(summary.iloc[0]["n_groups_own_fe"]) == 2
    assert int(summary.iloc[0]["n_rows_pooled"]) == 0
    residual, _coef = position_duration_group_residual(
        frame, "last_hits_per_minute", labels
    )
    assert residual.to_numpy() == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-8)
    attached = attach_farming_candidates(frame, sparse_min_n=2)
    assert attached[CANDIDATE_C].to_numpy() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0], abs=1e-8
    )


def test_hero_within_position_adjustment_uses_cells() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [1.0, 1.0, 5.0, 5.0],
            "duration_seconds": [1800.0, 1800.0, 1800.0, 1800.0],
            "num_last_hits": [300.0, 330.0, 30.0, 60.0],
            "hero_id": [1, 1, 1, 1],
            "player_id": [11, 12, 13, 14],
            "match_id": [1, 2, 3, 4],
            "team_id": [100, 100, 200, 200],
            "team_won": [1, 0, 1, 0],
            "elo_expected_win": [0.5] * 4,
            "game_version_id": [176] * 4,
            "start_time": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 3, 1, tzinfo=UTC),
                datetime(2026, 4, 1, tzinfo=UTC),
            ],
            "position": ["POSITION_1", "POSITION_1", "POSITION_5", "POSITION_5"],
        }
    )
    attached = attach_farming_candidates(frame, sparse_min_n=2)
    labels, summary = pooled_group_labels(
        attached,
        ("hero_id", "position_number"),
        min_n=2,
        eligible=explicit_position_mask(attached),
    )
    assert int(summary.iloc[0]["n_groups_own_fe"]) == 2
    assert int(summary.iloc[0]["n_rows_pooled"]) == 0
    assert set(labels.dropna().unique()) == {"1|1.0", "1|5.0"}
    cell_means = attached.groupby(labels)[CANDIDATE_D].mean()
    for value in cell_means:
        assert value == pytest.approx(0.0, abs=1e-8)


def test_sparse_heroes_are_pooled_not_dropped() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [1.0, 1.0, 1.0, 1.0, 1.0],
            "hero_id": [1, 1, 1, 2, 3],
            "duration_seconds": [1800.0] * 5,
            "num_last_hits": [100.0, 110.0, 120.0, 50.0, 60.0],
            "player_id": [11, 12, 13, 14, 15],
            "match_id": [1, 2, 3, 4, 5],
            "team_id": [100] * 5,
            "team_won": [1] * 5,
            "elo_expected_win": [0.5] * 5,
            "game_version_id": [176] * 5,
            "start_time": [datetime(2026, 1, i + 1, tzinfo=UTC) for i in range(5)],
            "position": ["POSITION_1"] * 5,
        }
    )
    attached = attach_farming_candidates(frame, sparse_min_n=3)
    assert attached[CANDIDATE_C].notna().all()
    labels, summary = pooled_group_labels(
        attached, ("hero_id",), min_n=3, eligible=explicit_position_mask(attached)
    )
    assert int(summary.iloc[0]["n_rows_labeled"]) == 5
    assert int(summary.iloc[0]["n_rows_pooled"]) == 2
    assert int(summary.iloc[0]["n_groups_pooled"]) == 2
    assert int(summary.iloc[0]["n_groups_own_fe"]) == 1
    assert labels.notna().all()


def test_null_position_is_excluded_from_farming_candidates() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [1.0, 1.0, np.nan],
            "duration_seconds": [1800.0, 1800.0, 1800.0],
            "num_last_hits": [100.0, 200.0, 10_000.0],
            "hero_id": [1, 1, 1],
            "player_id": [11, 12, 13],
            "match_id": [1, 1, 1],
            "team_id": [100, 100, 100],
            "team_won": [1, 1, 1],
            "elo_expected_win": [0.5, 0.5, 0.5],
            "game_version_id": [176, 176, 176],
            "start_time": [datetime(2026, 1, 1, tzinfo=UTC)] * 3,
            "position": ["POSITION_1", "POSITION_1", None],
        }
    )
    attached = attach_farming_candidates(frame, sparse_min_n=1)
    for column in FARMING_CANDIDATE_COLUMN_NAMES:
        assert np.isnan(attached.loc[2, column])
        assert attached.loc[[0, 1], column].notna().all()
    a_vals = attached.loc[[0, 1], CANDIDATE_A].to_numpy(dtype=float)
    assert a_vals[0] == pytest.approx(-1.0)
    assert a_vals[1] == pytest.approx(1.0)


def test_zero_last_hits_are_preserved_distinct_from_null() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [5.0, 5.0],
            "duration_seconds": [1800.0, 1800.0],
            "num_last_hits": [0.0, 0.0],
            "hero_id": [1, 1],
            "player_id": [11, 12],
            "match_id": [1, 2],
            "team_id": [100, 100],
            "team_won": [1, 0],
            "elo_expected_win": [0.5, 0.5],
            "game_version_id": [176, 176],
            "start_time": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, tzinfo=UTC),
            ],
            "position": ["POSITION_5", "POSITION_5"],
        }
    )
    attached = attach_farming_candidates(frame, sparse_min_n=1)
    assert attached["last_hits_per_minute"].tolist() == [0.0, 0.0]
    assert attached[CANDIDATE_A].tolist() == [0.0, 0.0]
    assert attached[CANDIDATE_B].notna().all()


def test_development_cutoff_includes_boundary_and_drops_later(
    tmp_path: Path,
) -> None:
    boundary = FROZEN_DEVELOPMENT_END
    later = boundary + timedelta(days=1)
    earlier = boundary - timedelta(days=1)
    matches = [
        match_row(
            1,
            start_time=earlier,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
        match_row(
            2,
            start_time=boundary,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=False,
        ),
        match_row(
            3,
            start_time=later,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
    ]
    players = _annotate_players(
        player_rows(1, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(2, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(3, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
    )
    frame = _frame_from_store(tmp_path, matches=matches, players=players)
    development = restrict_development(frame)
    assert set(development["match_id"].unique()) == {1, 2}
    assert 3 not in set(development["match_id"].unique())
    assert pd.Timestamp(development["start_time"].max()) <= pd.Timestamp(boundary)


def test_canonical_team_id_joins_from_match_side(tmp_path: Path) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    matches = [
        match_row(
            1,
            start_time=t0,
            radiant_team_id=111,
            dire_team_id=222,
            radiant_win=True,
        )
    ]
    players = _annotate_players(
        player_rows(1, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
    )
    frame = _frame_from_store(tmp_path, matches=matches, players=players)
    radiant = frame.loc[frame["side"] == "RADIANT", "team_id"].unique().tolist()
    dire = frame.loc[frame["side"] == "DIRE", "team_id"].unique().tolist()
    assert radiant == [111]
    assert dire == [222]


def test_prior_history_is_strictly_earlier_and_same_timestamp_blind() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    t2 = datetime(2026, 3, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "player_id": [11, 11, 11, 11],
            "match_id": [1, 2, 3, 4],
            "start_time": [t0, t1, t1, t2],
            "hero_id": [1, 2, 3, 2],
            CANDIDATE_B: [10.0, 20.0, 30.0, 40.0],
        }
    )
    prior_mean, prior_n = prior_player_history(frame, CANDIDATE_B)
    assert prior_n.tolist() == [0, 1, 1, 3]
    assert np.isnan(prior_mean.iloc[0])
    assert prior_mean.iloc[1] == pytest.approx(10.0)
    assert prior_mean.iloc[2] == pytest.approx(10.0)
    assert prior_mean.iloc[3] == pytest.approx(20.0)
    hero_mean, hero_n = hero_excluded_prior_history(frame, CANDIDATE_B)
    assert hero_n.tolist() == [0, 1, 1, 2]
    assert hero_mean.iloc[1] == pytest.approx(10.0)
    assert hero_mean.iloc[2] == pytest.approx(10.0)
    assert hero_mean.iloc[3] == pytest.approx(20.0)


def test_hero_excluded_prior_mean_drops_same_hero_history() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    t2 = datetime(2026, 3, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "player_id": [11, 11, 11],
            "match_id": [1, 2, 3],
            "start_time": [t0, t1, t2],
            "hero_id": [1, 1, 2],
            CANDIDATE_B: [10.0, 30.0, 50.0],
        }
    )
    prior_mean, prior_n = hero_excluded_prior_history(frame, CANDIDATE_B)
    assert prior_n.tolist() == [0, 0, 2]
    assert np.isnan(prior_mean.iloc[0])
    assert np.isnan(prior_mean.iloc[1])
    assert prior_mean.iloc[2] == pytest.approx(20.0)
    ordinary_mean, ordinary_n = prior_player_history(frame, CANDIDATE_B)
    assert ordinary_n.tolist() == [0, 1, 2]
    assert ordinary_mean.iloc[1] == pytest.approx(10.0)


def test_cross_version_prior_is_same_timestamp_blind() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "player_id": [11, 11, 11],
            "match_id": [1, 2, 3],
            "start_time": [t0, t1, t1],
            "game_version_id": [176, 177, 176],
            CANDIDATE_B: [4.0, 8.0, 12.0],
        }
    )
    prior_mean, prior_n = prior_history_excluding(
        frame, CANDIDATE_B, "game_version_id"
    )
    assert prior_n.tolist() == [0, 1, 0]
    assert prior_mean.iloc[1] == pytest.approx(4.0)
    assert np.isnan(prior_mean.iloc[2])


def test_team_spells_do_not_use_future_spell_rows() -> None:
    times = [datetime(2026, 1, day, tzinfo=UTC) for day in range(1, 11)]
    frame = pd.DataFrame(
        {
            "player_id": [11] * 10,
            "team_id": [100] * 5 + [200] * 5,
            "match_id": list(range(1, 11)),
            "start_time": times,
            CANDIDATE_B: [1.0] * 5 + [9.0] * 5,
        }
    )
    spells = build_team_spells(frame, CANDIDATE_B, min_appearances=5)
    assert list(spells["team_id"]) == [100, 200]
    assert spells.iloc[0]["mean_value"] == pytest.approx(1.0)
    assert spells.iloc[1]["mean_value"] == pytest.approx(9.0)
    assert pd.Timestamp(spells.iloc[0]["start_time_last"]) < pd.Timestamp(
        spells.iloc[1]["start_time_first"]
    )
    summary, pairs = team_switcher_table(frame, CANDIDATE_B, min_appearances=(5,))
    assert int(summary.iloc[0]["n_qualifying_players"]) == 1
    assert int(summary.iloc[0]["n_team_transitions"]) == 1
    assert pairs.iloc[0]["old_mean"] == pytest.approx(1.0)
    assert pairs.iloc[0]["new_mean"] == pytest.approx(9.0)
    assert pairs.iloc[0]["old_mean"] != pytest.approx(5.0)


def test_stand_in_spell_is_not_treated_as_a_team_switch() -> None:
    times = [datetime(2026, 1, day, tzinfo=UTC) for day in range(1, 12)]
    frame = pd.DataFrame(
        {
            "player_id": [11] * 11,
            "team_id": [100] * 5 + [999] + [100] * 5,
            "match_id": list(range(1, 12)),
            "start_time": times,
            CANDIDATE_B: [1.0] * 5 + [50.0] + [1.0] * 5,
        }
    )
    spells = build_team_spells(frame, CANDIDATE_B, min_appearances=5)
    assert set(spells["team_id"].tolist()) == {100}
    summary, pairs = team_switcher_table(frame, CANDIDATE_B, min_appearances=(5,))
    assert int(summary.iloc[0]["n_team_transitions"]) == 0
    assert pairs.empty or len(pairs) == 0


def test_team_period_centered_is_leave_one_player_out() -> None:
    frame = pd.DataFrame(
        {
            "player_id": [11, 11, 12, 12],
            "team_id": [100, 100, 100, 100],
            "game_version_id": [176, 176, 176, 176],
            CANDIDATE_B: [10.0, 12.0, 20.0, 22.0],
        }
    )
    centered = team_period_centered(frame, CANDIDATE_B)
    # Player 11 vs player 12 mean 21: 10-21 = -11, 12-21 = -9
    assert centered.iloc[0] == pytest.approx(-11.0)
    assert centered.iloc[1] == pytest.approx(-9.0)
    assert centered.iloc[2] == pytest.approx(9.0)
    assert centered.iloc[3] == pytest.approx(11.0)


def test_farming_residuals_are_deterministic() -> None:
    frame = _farming_frame()
    first = attach_farming_candidates(frame, sparse_min_n=2)
    second = attach_farming_candidates(frame, sparse_min_n=2)
    for column in FARMING_CANDIDATE_COLUMN_NAMES:
        np.testing.assert_allclose(
            first[column].to_numpy(dtype=float),
            second[column].to_numpy(dtype=float),
            equal_nan=True,
        )
        assert first[column].notna().all()


def test_candidates_do_not_enter_feature_columns_or_specs() -> None:
    for name in FARMING_CANDIDATE_COLUMN_NAMES:
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
    for name in CANDIDATE_COLUMN_NAMES:
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 33
    assert tuple(spec.name for spec in SLICE9_FROZEN_SPECS) == (
        SLICE9_REFERENCE_SPEC_NAME,
        SLICE9_CANDIDATE_SPEC_NAME,
    )
    assert [spec.name for spec in POST_DRAFT_BLOCK_ABLATION_SPECS] == [
        "logistic_elo_only",
        "logistic_elo_plus_player_hero",
        "logistic_elo_plus_team_hero",
        "logistic_elo_plus_hero_meta",
        "logistic_elo_plus_player_and_team_hero",
        "logistic_elo_plus_all_three",
    ]
    assert SPARSE_GROUP_MIN_N == 5


def test_ols_residual_and_full_diagnostic_run(tmp_path: Path) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    matches = [
        match_row(
            1,
            start_time=t0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
            game_version_id=176,
        ),
        match_row(
            2,
            start_time=t1,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=False,
            game_version_id=177,
        ),
    ]
    players = _annotate_players(
        player_rows(1, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(2, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
    )
    y = pd.Series([1.0, 3.0, 5.0])
    x = pd.DataFrame({"intercept": [1.0, 1.0, 1.0], "x": [0.0, 1.0, 2.0]})
    residual, coef = ols_residual(y, x)
    assert residual.to_numpy() == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert coef[1] == pytest.approx(2.0)

    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        report = run_farming_performance_target_diagnostics(store, sparse_min_n=1)
        view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    assert report.n_development_matches == 2
    assert report.n_development_player_rows == 20
    assert report.integrity["candidate_in_feature_columns"] is False
    assert report.integrity["player_rating_persisted"] is False
    assert report.integrity["player_farming_state_persisted"] is False
    assert report.integrity["model_trained"] is False
    assert report.integrity["stratz_called"] is False
    assert report.integrity["elo_modified"] is False
    assert report.integrity["shrinkage_introduced"] is False
    assert report.integrity["candidate_a_is_slice12_baseline"] is True
    assert report.integrity["box_scores_in_feature_match_players_view"] is False
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in view_columns
    assert report.n_holdout_excluded == 0
    assert CANDIDATE_A in set(report.candidate_comparison["candidate"])
    assert CANDIDATE_B in set(report.candidate_comparison["candidate"])
    assert CANDIDATE_C in set(report.candidate_comparison["candidate"])
    assert CANDIDATE_D in set(report.candidate_comparison["candidate"])
    assert report.classification.iloc[0]["classification"] in {"A", "B", "C"}
    a_means = report.candidate_position_means.loc[
        report.candidate_position_means["candidate"] == CANDIDATE_A
    ].iloc[0]
    for number in (1, 2, 3, 4, 5):
        assert a_means[f"pos{number}_mean"] == pytest.approx(0.0, abs=1e-9)
