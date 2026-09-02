"""Slice 14 leakage-safe player farming state: causal B, shrinkage, no features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import (
    build_snapshot_store,
    match_row,
    player_rows,
)

from dota_predictor.data.canonical_schema import MATCH_PLAYER_BOX_SCORE_COLUMNS
from dota_predictor.features.duckdb_layer import MATCH_PLAYERS_VIEW
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.farming_performance_target import (
    CANDIDATE_B,
    attach_farming_candidates,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.player_farming_state import (
    CAUSAL_B_COLUMN,
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    RESIDUALIZER_N_PARAMS,
    SHRINKAGE_GRID,
    SLICE14_STATE_COLUMNS,
    attach_causal_candidate_b,
    attach_player_farming_state,
    causal_position_duration_design,
    development_tune_end,
    farming_shrinkage_weight,
    farming_shrunk_b,
    fit_causal_residualizer,
    prior_farming_history,
    run_player_farming_state_diagnostics,
    select_shrinkage_k,
)
from dota_predictor.training.player_performance_target import CANDIDATE_COLUMN_NAMES
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
T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = datetime(2026, 2, 1, tzinfo=UTC)
T2 = datetime(2026, 3, 1, tzinfo=UTC)
T3 = datetime(2026, 4, 1, tzinfo=UTC)
T4 = datetime(2026, 5, 1, tzinfo=UTC)


def _appearance(
    *,
    match_id: int,
    player_id: int,
    start_time: datetime,
    position: int,
    duration: float,
    last_hits: float,
    hero_id: int = 1,
    team_id: int = 100,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "hero_id": hero_id,
        "team_id": team_id,
        "side": "RADIANT",
        "position": f"POSITION_{position}",
        "position_number": float(position),
        "start_time": start_time,
        "duration_seconds": float(duration),
        "num_last_hits": float(last_hits),
        "team_won": 1,
        "elo_expected_win": 0.5,
        "game_version_id": 176,
        "kills": 1,
        "deaths": 1,
        "assists": 1,
        "gold_per_minute": 400,
        "experience_per_minute": 400,
        "num_denies": 0,
        "networth": 10000,
        "hero_damage": 10000,
        "tower_damage": 1000,
        "hero_healing": 0,
        "level": 20,
    }


def _stamp_rows(
    match_id: int,
    start_time: datetime,
    duration: float,
    last_hits: tuple[float, float, float, float, float],
    *,
    player_base: int = 10,
) -> list[dict[str, object]]:
    return [
        _appearance(
            match_id=match_id,
            player_id=player_base + position,
            start_time=start_time,
            position=position,
            duration=duration,
            last_hits=last_hits[position - 1],
        )
        for position in range(1, 6)
    ]


def _causal_ready_frame(
    extra: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    rows = (
        _stamp_rows(1, T0, 1000.0, (50.0, 40.0, 30.0, 20.0, 10.0))
        + _stamp_rows(2, T1, 2000.0, (120.0, 100.0, 80.0, 50.0, 20.0))
        + _stamp_rows(3, T2, 1500.0, (90.0, 70.0, 55.0, 35.0, 15.0))
    )
    if extra:
        rows.extend(extra)
    return pd.DataFrame(rows)


def _state_frame() -> pd.DataFrame:
    """Known causal B values; residualizer is not used."""
    return pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4, 5, 6],
            "player_id": [11, 11, 11, 12, 12, 12],
            "start_time": [T0, T1, T2, T0, T1, T2],
            CAUSAL_B_COLUMN: [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
            "last_hits_per_minute": [6.0] * 6,
            "position_number": [1.0] * 6,
            "position": ["POSITION_1"] * 6,
            "num_last_hits": [100.0] * 6,
            "duration_seconds": [1000.0] * 6,
        }
    )


def test_frozen_candidate_is_slice13_b() -> None:
    assert FROZEN_CANDIDATE_B == CANDIDATE_B
    assert CANDIDATE_B == "last_hits_per_min_position_duration_residual_z"
    assert SHRINKAGE_GRID == (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    assert FROZEN_SHRINKAGE_K in SHRINKAGE_GRID
    assert FROZEN_SHRINKAGE_K != 0.0


def test_insufficient_residualizer_history_is_null_not_global_fit() -> None:
    rows = _stamp_rows(1, T0, 1000.0, (50.0, 40.0, 30.0, 20.0, 10.0))
    rows += _stamp_rows(2, T1, 2000.0, (120.0, 100.0, 80.0, 50.0, 20.0))
    frame = pd.DataFrame(rows)
    causal = attach_causal_candidate_b(frame)
    assert causal[CAUSAL_B_COLUMN].isna().all()
    global_fit = attach_farming_candidates(frame, sparse_min_n=1)
    assert global_fit[CANDIDATE_B].notna().all()


def test_causal_b_matches_prior_only_lstsq() -> None:
    frame = _causal_ready_frame()
    attached = attach_causal_candidate_b(frame)
    assert attached.loc[attached["start_time"] == T0, CAUSAL_B_COLUMN].isna().all()
    assert attached.loc[attached["start_time"] == T1, CAUSAL_B_COLUMN].isna().all()
    later = attached.loc[attached["start_time"] == T2]
    assert later[CAUSAL_B_COLUMN].notna().all()

    prior = frame.loc[frame["start_time"] < T2].copy()
    prior["last_hits_per_minute"] = attached.loc[prior.index, "last_hits_per_minute"]
    fitted = fit_causal_residualizer(
        causal_position_duration_design(prior),
        prior["last_hits_per_minute"],
    )
    assert fitted is not None
    coef, mu, sigma = fitted
    current = attached.loc[attached["start_time"] == T2]
    design = causal_position_duration_design(current).to_numpy(dtype=float)
    y = current["last_hits_per_minute"].to_numpy(dtype=float)
    expected = ((y - design @ coef) - mu) / sigma
    np.testing.assert_allclose(
        current[CAUSAL_B_COLUMN].to_numpy(dtype=float), expected, atol=1e-9
    )
    assert int(current["farming_residualizer_n"].iloc[0]) == 10


def test_future_rows_cannot_change_earlier_causal_b() -> None:
    base = _causal_ready_frame()
    future = _stamp_rows(
        4, T3, 3000.0, (500.0, 10.0, 10.0, 10.0, 10.0), player_base=40
    )
    without_future = attach_causal_candidate_b(base)
    with_future = attach_causal_candidate_b(pd.concat([base, pd.DataFrame(future)]))
    earlier = without_future["start_time"] <= T2
    left = without_future.loc[earlier, ["player_id", "start_time", CAUSAL_B_COLUMN]]
    right = with_future.loc[
        with_future["start_time"] <= T2, ["player_id", "start_time", CAUSAL_B_COLUMN]
    ]
    merged = left.merge(right, on=["player_id", "start_time"], suffixes=("_a", "_b"))
    np.testing.assert_allclose(
        merged[f"{CAUSAL_B_COLUMN}_a"].to_numpy(dtype=float),
        merged[f"{CAUSAL_B_COLUMN}_b"].to_numpy(dtype=float),
        equal_nan=True,
        atol=1e-12,
    )


def test_future_residuals_cannot_change_z_scale_at_t() -> None:
    base = _causal_ready_frame()
    mild = _stamp_rows(4, T3, 1600.0, (80.0, 70.0, 60.0, 40.0, 20.0), player_base=50)
    wild = _stamp_rows(4, T3, 1600.0, (800.0, 0.0, 0.0, 0.0, 0.0), player_base=50)
    mild_b = attach_causal_candidate_b(pd.concat([base, pd.DataFrame(mild)]))
    wild_b = attach_causal_candidate_b(pd.concat([base, pd.DataFrame(wild)]))
    left = mild_b.loc[mild_b["start_time"] == T2, CAUSAL_B_COLUMN].to_numpy(dtype=float)
    right = wild_b.loc[wild_b["start_time"] == T2, CAUSAL_B_COLUMN].to_numpy(dtype=float)
    np.testing.assert_allclose(left, right, atol=1e-12)
    global_mild = attach_farming_candidates(
        pd.concat([base, pd.DataFrame(mild)]), sparse_min_n=1
    )
    global_wild = attach_farming_candidates(
        pd.concat([base, pd.DataFrame(wild)]), sparse_min_n=1
    )
    g_left = global_mild.loc[global_mild["start_time"] == T2, CANDIDATE_B]
    g_right = global_wild.loc[global_wild["start_time"] == T2, CANDIDATE_B]
    assert not np.allclose(
        g_left.to_numpy(dtype=float), g_right.to_numpy(dtype=float), atol=1e-6
    )


def test_same_timestamp_rows_are_mutually_blind_in_residualizer() -> None:
    extra_a = _stamp_rows(
        30, T2, 1500.0, (200.0, 10.0, 10.0, 10.0, 10.0), player_base=30
    )
    extra_b = _stamp_rows(
        30, T2, 1500.0, (10.0, 10.0, 10.0, 10.0, 10.0), player_base=30
    )
    base = _causal_ready_frame()
    left = attach_causal_candidate_b(pd.concat([base, pd.DataFrame(extra_a)]))
    right = attach_causal_candidate_b(pd.concat([base, pd.DataFrame(extra_b)]))
    core_left = left.loc[
        (left["start_time"] == T2) & (left["match_id"] == 3), CAUSAL_B_COLUMN
    ].to_numpy(dtype=float)
    core_right = right.loc[
        (right["start_time"] == T2) & (right["match_id"] == 3), CAUSAL_B_COLUMN
    ].to_numpy(dtype=float)
    np.testing.assert_allclose(core_left, core_right, atol=1e-12)


def test_modifying_current_b_cannot_alter_state_at_m() -> None:
    frame = _state_frame()
    original = attach_player_farming_state(frame, k=10.0)
    mutated = original.copy()
    mutated.loc[mutated["start_time"] == T1, CAUSAL_B_COLUMN] = 99.0
    rerun = attach_player_farming_state(mutated, k=10.0)
    at_t1 = rerun["start_time"] == T1
    np.testing.assert_allclose(
        original.loc[at_t1, "farming_prior_n"].to_numpy(),
        rerun.loc[at_t1, "farming_prior_n"].to_numpy(),
    )
    np.testing.assert_allclose(
        original.loc[at_t1, "farming_prior_mean_b"].to_numpy(dtype=float),
        rerun.loc[at_t1, "farming_prior_mean_b"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        original.loc[at_t1, "farming_shrunk_b"].to_numpy(dtype=float),
        rerun.loc[at_t1, "farming_shrunk_b"].to_numpy(dtype=float),
    )
    later = rerun["start_time"] == T2
    assert not np.allclose(
        original.loc[later, "farming_prior_mean_b"].to_numpy(dtype=float),
        rerun.loc[later, "farming_prior_mean_b"].to_numpy(dtype=float),
    )


def test_same_timestamp_state_is_mutually_blind() -> None:
    frame = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "player_id": [11, 11, 11, 11],
            "start_time": [T0, T1, T1, T2],
            CAUSAL_B_COLUMN: [1.0, 10.0, 30.0, 0.0],
            "last_hits_per_minute": [1.0] * 4,
            "position_number": [1.0] * 4,
            "position": ["POSITION_1"] * 4,
            "num_last_hits": [10.0] * 4,
            "duration_seconds": [1000.0] * 4,
        }
    )
    state = attach_player_farming_state(frame, k=5.0)
    at_t1 = state["start_time"] == T1
    assert state.loc[at_t1, "farming_prior_n"].tolist() == [1, 1]
    assert state.loc[at_t1, "farming_prior_mean_b"].to_numpy() == pytest.approx(
        [1.0, 1.0]
    )
    at_t2 = state["start_time"] == T2
    assert int(state.loc[at_t2, "farming_prior_n"].iloc[0]) == 3
    assert float(state.loc[at_t2, "farming_prior_mean_b"].iloc[0]) == pytest.approx(
        41.0 / 3.0
    )


def test_future_matches_cannot_change_earlier_player_state() -> None:
    frame = _state_frame()
    original = attach_player_farming_state(frame, k=10.0)
    future = pd.DataFrame(
        {
            "match_id": [99],
            "player_id": [11],
            "start_time": [T4],
            CAUSAL_B_COLUMN: [50.0],
            "last_hits_per_minute": [1.0],
            "position_number": [1.0],
            "position": ["POSITION_1"],
            "num_last_hits": [10.0],
            "duration_seconds": [1000.0],
        }
    )
    combined = attach_player_farming_state(
        pd.concat([frame, future], ignore_index=True), k=10.0
    )
    earlier = original.loc[original["player_id"] == 11].copy()
    later = combined.loc[
        (combined["player_id"] == 11) & (combined["start_time"] <= T2)
    ]
    for column in (
        "farming_prior_n",
        "farming_prior_sum_b",
        "farming_prior_mean_b",
        "farming_shrunk_b",
    ):
        np.testing.assert_allclose(
            earlier[column].to_numpy(dtype=float),
            later[column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_other_player_history_cannot_enter_current_state() -> None:
    frame = _state_frame()
    original = attach_player_farming_state(frame, k=10.0)
    mutated = frame.copy()
    mutated.loc[mutated["player_id"] == 12, CAUSAL_B_COLUMN] = 99.0
    rerun = attach_player_farming_state(mutated, k=10.0)
    left = original.loc[original["player_id"] == 11]
    right = rerun.loc[rerun["player_id"] == 11]
    np.testing.assert_allclose(
        left["farming_prior_mean_b"].to_numpy(dtype=float),
        right["farming_prior_mean_b"].to_numpy(dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        left["farming_shrunk_b"].to_numpy(dtype=float),
        right["farming_shrunk_b"].to_numpy(dtype=float),
    )


def test_zero_history_mean_is_null_and_shrunk_is_zero() -> None:
    frame = _state_frame()
    state = attach_player_farming_state(frame, k=10.0)
    first = state["start_time"] == T0
    assert (state.loc[first, "farming_prior_n"] == 0).all()
    assert state.loc[first, "farming_prior_mean_b"].isna().all()
    assert (state.loc[first, "farming_shrunk_b"] == 0.0).all()
    assert (state.loc[first, "farming_shrinkage_weight"] == 0.0).all()
    assert (state.loc[first, "farming_prior_sum_b"] == 0.0).all()


def test_shrinkage_formula_known_examples() -> None:
    assert farming_shrunk_b(2.0, 0, k=10.0) == 0.0
    assert farming_shrunk_b(None, 5, k=10.0) == 0.0
    assert farming_shrunk_b(2.0, 10, k=10.0) == pytest.approx(1.0)
    assert farming_shrunk_b(2.0, 5, k=5.0) == pytest.approx(1.0)
    assert farming_shrunk_b(4.0, 3, k=0.0) == pytest.approx(4.0)
    assert farming_shrinkage_weight(0, k=10.0) == 0.0
    assert farming_shrinkage_weight(10, k=10.0) == pytest.approx(0.5)
    assert farming_shrinkage_weight(5, k=0.0) == 1.0


def test_monotonic_shrinkage_toward_raw_mean() -> None:
    mean = 2.0
    k = 10.0
    previous = abs(farming_shrunk_b(mean, 1, k=k) - mean)
    for n in (2, 5, 10, 20, 40, 80):
        gap = abs(farming_shrunk_b(mean, n, k=k) - mean)
        assert gap < previous
        previous = gap
    assert abs(farming_shrunk_b(mean, 80, k=k)) > abs(farming_shrunk_b(mean, 5, k=k))


def test_null_position_never_enters_residualizer_or_state() -> None:
    rows = _stamp_rows(1, T0, 1000.0, (50.0, 40.0, 30.0, 20.0, 10.0))
    rows += _stamp_rows(2, T1, 2000.0, (120.0, 100.0, 80.0, 50.0, 20.0))
    rows.append(
        _appearance(
            match_id=3,
            player_id=99,
            start_time=T1,
            position=1,
            duration=2000.0,
            last_hits=10_000.0,
        )
    )
    frame = pd.DataFrame(rows)
    frame.loc[frame["player_id"] == 99, "position"] = None
    frame.loc[frame["player_id"] == 99, "position_number"] = np.nan
    attached = attach_causal_candidate_b(frame)
    assert np.isnan(attached.loc[attached["player_id"] == 99, CAUSAL_B_COLUMN].iloc[0])
    at_t1 = attached["start_time"] == T1
    assert (
        attached.loc[at_t1 & (attached["player_id"] != 99), CAUSAL_B_COLUMN].isna()
    ).all()


def test_select_shrinkage_k_prefers_stronger_equivalent_shrinkage() -> None:
    grid = pd.DataFrame(
        [
            {
                "k": k,
                "rmse": 1.0 - 0.01 * (1 if k in {10.0, 20.0} else 0),
                "low_n_rmse": 1.20 if k == 0.0 else 1.00,
                "rmse_n_gt_20": 0.80,
            }
            for k in SHRINKAGE_GRID
        ]
    )
    selected, justification = select_shrinkage_k(grid)
    assert selected == 80.0
    assert "strongest shrinkage" in justification or "equivalent" in justification


def test_development_cutoff_excludes_later_rows_from_diagnostics(
    tmp_path: Path,
) -> None:
    boundary = FROZEN_DEVELOPMENT_END
    later = boundary + timedelta(days=1)
    earlier = boundary - timedelta(days=20)
    mid = boundary - timedelta(days=10)
    matches = []
    players: list[dict[str, object]] = []
    for match_id, stamp, duration in (
        (1, earlier, 1000),
        (2, mid, 2000),
        (3, boundary, 1500),
        (4, later, 1800),
    ):
        row = match_row(
            match_id,
            start_time=stamp,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        )
        row["duration_seconds"] = duration
        matches.append(row)
        batch = player_rows(
            match_id, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS
        )
        for item in batch:
            slot = int(item["slot_in_side"])
            item["position"] = POSITIONS[slot]
            item["num_last_hits"] = 300 - slot * 40
            item["kills"] = 1
            item["deaths"] = 1
            item["assists"] = 1
            item["gold_per_minute"] = 400
            item["experience_per_minute"] = 400
            item["num_denies"] = 0
            item["networth"] = 10000
            item["hero_damage"] = 10000
            item["tower_damage"] = 1000
            item["hero_healing"] = 0
            item["level"] = 20
        players.extend(batch)

    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        report = run_player_farming_state_diagnostics(store)
        view_columns = store.relation(MATCH_PLAYERS_VIEW).columns

    assert report.n_development_matches == 3
    assert report.n_development_player_rows == 30
    assert report.n_holdout_excluded == 10
    assert report.tune_end <= FROZEN_DEVELOPMENT_END
    assert report.development_end == FROZEN_DEVELOPMENT_END
    assert pd.Timestamp(report.split.iloc[0]["validation_max_start_time"]) <= (
        pd.Timestamp(FROZEN_DEVELOPMENT_END)
    )
    assert report.integrity["holdout_used_for_k"] is False
    assert report.integrity["ti2026_used_for_k"] is False
    assert report.integrity["holdout_rows_in_development"] is False
    assert report.integrity["stratz_called"] is False
    assert report.integrity["model_trained"] is False
    assert report.integrity["win_model_benchmarked"] is False
    assert report.integrity["shrinkage_chosen_from_outcomes"] is False
    assert report.integrity["global_residualizer_fallback_used"] is False
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["state_in_feature_columns"] is False
    assert report.selected_k in SHRINKAGE_GRID
    assert report.classification.iloc[0]["classification"] in {"A", "B", "C"}
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in view_columns


def test_tune_split_keeps_same_timestamp_groups_together() -> None:
    times = pd.Series([T0] * 5 + [T1] * 5 + [T2] * 5 + [T3] * 5)
    tune_end = development_tune_end(times, development_end=T3)
    assert tune_end in {T0, T1, T2}
    assigned = {
        "tune" if stamp <= pd.Timestamp(tune_end) else "val" for stamp in times
    }
    assert assigned == {"tune", "val"}
    t1_parts = {"tune" if T1 <= tune_end else "val"}
    assert len(t1_parts) == 1


def test_state_columns_do_not_enter_feature_columns_or_specs() -> None:
    for name in SLICE14_STATE_COLUMNS:
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
    for name in CANDIDATE_COLUMN_NAMES:
        assert name not in FEATURE_COLUMNS
    assert CANDIDATE_B not in FEATURE_COLUMNS
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
    assert RESIDUALIZER_N_PARAMS == 6


def test_prior_sum_is_n_times_mean() -> None:
    frame = _state_frame()
    counts, sums, means = prior_farming_history(frame)
    state = attach_player_farming_state(frame, k=0.0)
    np.testing.assert_array_equal(state["farming_prior_n"].to_numpy(), counts)
    np.testing.assert_allclose(state["farming_prior_sum_b"].to_numpy(), sums)
    at_t2 = (state["player_id"] == 11) & (state["start_time"] == T2)
    row = state.loc[at_t2].iloc[0]
    assert int(row["farming_prior_n"]) == 2
    assert float(row["farming_prior_sum_b"]) == pytest.approx(3.0)
    assert float(row["farming_prior_mean_b"]) == pytest.approx(1.5)
    assert float(row["farming_shrunk_b"]) == pytest.approx(1.5)
    assert float(means.loc[row.name]) == pytest.approx(1.5)
