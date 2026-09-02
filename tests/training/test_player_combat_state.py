"""Slice 18 leakage-safe player combat state: causal C, shrinkage, no features."""

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
from dota_predictor.training.combat_performance_target import (
    COMBAT_C,
    COMBAT_C_POSITION,
    FROZEN_COMBAT_CANDIDATE,
    attach_combat_candidates,
    hero_damage_share,
)
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.player_combat_state import (
    CAUSAL_C_COLUMN,
    COMBAT_SHRINKAGE_GRID,
    FROZEN_COMBAT_SHRINKAGE_K,
    SLICE18_STATE_COLUMNS,
    attach_causal_candidate_c,
    attach_player_combat_state,
    combat_shrinkage_weight,
    combat_shrunk_c,
    prior_combat_history,
    run_player_combat_state_diagnostics,
    select_combat_shrinkage_k,
)
from dota_predictor.training.player_farming_state import (
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    SHRINKAGE_GRID,
    development_tune_end,
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
    hero_damage: float,
    side: str = "RADIANT",
    hero_id: int = 1,
    team_id: int = 100,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "hero_id": hero_id,
        "team_id": team_id,
        "side": side,
        "position": f"POSITION_{position}",
        "position_number": float(position),
        "start_time": start_time,
        "duration_seconds": 1800.0,
        "num_last_hits": 100.0,
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
        "hero_damage": float(hero_damage),
        "tower_damage": 1000,
        "hero_healing": 0,
        "level": 20,
    }


def _stamp_rows(
    match_id: int,
    start_time: datetime,
    damages: tuple[float, float, float, float, float],
    *,
    player_base: int = 10,
    side: str = "RADIANT",
) -> list[dict[str, object]]:
    return [
        _appearance(
            match_id=match_id,
            player_id=player_base + position,
            start_time=start_time,
            position=position,
            hero_damage=damages[position - 1],
            side=side,
            team_id=100 if side == "RADIANT" else 200,
        )
        for position in range(1, 6)
    ]


def _causal_ready_frame(
    extra: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    rows = (
        _stamp_rows(1, T0, (10.0, 8.0, 6.0, 4.0, 2.0))
        + _stamp_rows(2, T1, (20.0, 10.0, 5.0, 3.0, 2.0))
        + _stamp_rows(3, T2, (12.0, 9.0, 6.0, 3.0, 0.0))
    )
    if extra:
        rows.extend(extra)
    return pd.DataFrame(rows)


def _state_frame() -> pd.DataFrame:
    """Known causal C values; position baseline is not used."""
    return pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4, 5, 6],
            "player_id": [11, 11, 11, 12, 12, 12],
            "start_time": [T0, T1, T2, T0, T1, T2],
            CAUSAL_C_COLUMN: [0.10, 0.20, 0.30, -0.10, -0.20, -0.30],
            COMBAT_C: [0.30, 0.40, 0.50, 0.10, 0.05, 0.00],
            "position_number": [1.0] * 6,
            "position": ["POSITION_1"] * 6,
            "hero_damage": [10000.0] * 6,
            "duration_seconds": [1800.0] * 6,
            "side": ["RADIANT"] * 6,
        }
    )


def test_frozen_invariants() -> None:
    assert FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION
    assert FROZEN_COMBAT_CANDIDATE == "hero_damage_share_position_adj"
    assert FROZEN_CANDIDATE_B == CANDIDATE_B
    assert FROZEN_SHRINKAGE_K == 5.0
    assert COMBAT_SHRINKAGE_GRID == SHRINKAGE_GRID
    assert COMBAT_SHRINKAGE_GRID == (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    assert FROZEN_COMBAT_SHRINKAGE_K in COMBAT_SHRINKAGE_GRID
    assert FROZEN_COMBAT_SHRINKAGE_K != 0.0
    assert FROZEN_COMBAT_SHRINKAGE_K != FROZEN_SHRINKAGE_K
    assert FROZEN_COMBAT_SHRINKAGE_K == 20.0
    assert len(FEATURE_COLUMNS) == 33


def test_warmup_is_null_not_global_or_uniform_fallback() -> None:
    rows = _stamp_rows(1, T0, (10.0, 8.0, 6.0, 4.0, 2.0))
    frame = pd.DataFrame(rows)
    causal = attach_causal_candidate_c(frame)
    assert causal[CAUSAL_C_COLUMN].isna().all()
    assert (causal["combat_position_baseline_n"] == 0).all()
    global_fit = attach_combat_candidates(frame)
    assert global_fit[COMBAT_C_POSITION].notna().all()
    assert global_fit[COMBAT_C].notna().all()


def test_causal_c_matches_prior_only_position_mean() -> None:
    frame = _causal_ready_frame()
    attached = attach_causal_candidate_c(frame)
    assert attached.loc[attached["start_time"] == T0, CAUSAL_C_COLUMN].isna().all()
    later = attached.loc[attached["start_time"] == T1]
    assert later[CAUSAL_C_COLUMN].notna().all()

    t0 = attached.loc[attached["start_time"] == T0]
    t1 = attached.loc[attached["start_time"] == T1]
    for position in range(1, 6):
        prior_share = float(
            t0.loc[t0["position_number"] == float(position), COMBAT_C].iloc[0]
        )
        current = t1.loc[t1["position_number"] == float(position)].iloc[0]
        expected = float(current[COMBAT_C]) - prior_share
        assert float(current[CAUSAL_C_COLUMN]) == pytest.approx(expected)
        assert float(current["combat_position_baseline"]) == pytest.approx(prior_share)
        assert int(current["combat_position_baseline_n"]) == 1

    t2 = attached.loc[attached["start_time"] == T2]
    prior = attached.loc[attached["start_time"] < T2]
    for position in range(1, 6):
        prior_mean = float(
            prior.loc[prior["position_number"] == float(position), COMBAT_C].mean()
        )
        current = t2.loc[t2["position_number"] == float(position)].iloc[0]
        assert float(current["combat_position_baseline"]) == pytest.approx(prior_mean)
        assert float(current[CAUSAL_C_COLUMN]) == pytest.approx(
            float(current[COMBAT_C]) - prior_mean
        )
        assert int(current["combat_position_baseline_n"]) == 2


def test_future_share_cannot_change_earlier_causal_c() -> None:
    base = _causal_ready_frame()
    future = _stamp_rows(4, T3, (100.0, 0.0, 0.0, 0.0, 0.0), player_base=40)
    without_future = attach_causal_candidate_c(base)
    with_future = attach_causal_candidate_c(pd.concat([base, pd.DataFrame(future)]))
    earlier = without_future["start_time"] <= T2
    left = without_future.loc[earlier, ["player_id", "start_time", CAUSAL_C_COLUMN]]
    right = with_future.loc[
        with_future["start_time"] <= T2, ["player_id", "start_time", CAUSAL_C_COLUMN]
    ]
    merged = left.merge(right, on=["player_id", "start_time"], suffixes=("_a", "_b"))
    np.testing.assert_allclose(
        merged[f"{CAUSAL_C_COLUMN}_a"].to_numpy(dtype=float),
        merged[f"{CAUSAL_C_COLUMN}_b"].to_numpy(dtype=float),
        equal_nan=True,
        atol=1e-12,
    )
    global_mild = attach_combat_candidates(base)
    global_wild = attach_combat_candidates(pd.concat([base, pd.DataFrame(future)]))
    g_left = global_mild.loc[global_mild["start_time"] == T2, COMBAT_C_POSITION]
    g_right = global_wild.loc[global_wild["start_time"] == T2, COMBAT_C_POSITION]
    assert not np.allclose(
        g_left.to_numpy(dtype=float), g_right.to_numpy(dtype=float), atol=1e-6
    )


def test_same_timestamp_rows_are_mutually_blind_in_position_baseline() -> None:
    extra_a = _stamp_rows(30, T1, (200.0, 1.0, 1.0, 1.0, 1.0), player_base=30)
    extra_b = _stamp_rows(30, T1, (1.0, 1.0, 1.0, 1.0, 1.0), player_base=30)
    base = _causal_ready_frame()
    left = attach_causal_candidate_c(pd.concat([base, pd.DataFrame(extra_a)]))
    right = attach_causal_candidate_c(pd.concat([base, pd.DataFrame(extra_b)]))
    core_left = left.loc[
        (left["start_time"] == T1) & (left["match_id"] == 2), CAUSAL_C_COLUMN
    ].to_numpy(dtype=float)
    core_right = right.loc[
        (right["start_time"] == T1) & (right["match_id"] == 2), CAUSAL_C_COLUMN
    ].to_numpy(dtype=float)
    np.testing.assert_allclose(core_left, core_right, atol=1e-12)
    left_n = left.loc[
        (left["start_time"] == T1) & (left["match_id"] == 2),
        "combat_position_baseline_n",
    ]
    assert (left_n == 1).all()


def test_same_timestamp_share_the_same_prior_baseline() -> None:
    extra = _stamp_rows(30, T1, (50.0, 40.0, 30.0, 20.0, 10.0), player_base=30)
    frame = _causal_ready_frame(extra)
    attached = attach_causal_candidate_c(frame)
    at_t1 = attached["start_time"] == T1
    for position in range(1, 6):
        means = attached.loc[
            at_t1 & (attached["position_number"] == float(position)),
            "combat_position_baseline",
        ]
        assert means.nunique() == 1


def test_modifying_current_c_cannot_alter_state_at_m() -> None:
    frame = _state_frame()
    original = attach_player_combat_state(frame, k=10.0)
    mutated = original.copy()
    mutated.loc[mutated["start_time"] == T1, CAUSAL_C_COLUMN] = 99.0
    rerun = attach_player_combat_state(mutated, k=10.0)
    at_t1 = rerun["start_time"] == T1
    np.testing.assert_allclose(
        original.loc[at_t1, "combat_prior_n"].to_numpy(),
        rerun.loc[at_t1, "combat_prior_n"].to_numpy(),
    )
    np.testing.assert_allclose(
        original.loc[at_t1, "combat_prior_mean_c"].to_numpy(dtype=float),
        rerun.loc[at_t1, "combat_prior_mean_c"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        original.loc[at_t1, "combat_shrunk_c"].to_numpy(dtype=float),
        rerun.loc[at_t1, "combat_shrunk_c"].to_numpy(dtype=float),
    )
    later = rerun["start_time"] == T2
    assert not np.allclose(
        original.loc[later, "combat_prior_mean_c"].to_numpy(dtype=float),
        rerun.loc[later, "combat_prior_mean_c"].to_numpy(dtype=float),
    )


def test_same_timestamp_state_is_mutually_blind() -> None:
    frame = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "player_id": [11, 11, 11, 11],
            "start_time": [T0, T1, T1, T2],
            CAUSAL_C_COLUMN: [0.10, 1.0, 3.0, 0.0],
            "position_number": [1.0] * 4,
            "position": ["POSITION_1"] * 4,
            "hero_damage": [10.0] * 4,
            "duration_seconds": [1800.0] * 4,
            "side": ["RADIANT"] * 4,
        }
    )
    state = attach_player_combat_state(frame, k=5.0)
    at_t1 = state["start_time"] == T1
    assert state.loc[at_t1, "combat_prior_n"].tolist() == [1, 1]
    assert state.loc[at_t1, "combat_prior_mean_c"].to_numpy() == pytest.approx(
        [0.10, 0.10]
    )
    at_t2 = state["start_time"] == T2
    assert int(state.loc[at_t2, "combat_prior_n"].iloc[0]) == 3
    assert float(state.loc[at_t2, "combat_prior_mean_c"].iloc[0]) == pytest.approx(
        4.10 / 3.0
    )


def test_future_matches_cannot_change_earlier_player_state() -> None:
    frame = _state_frame()
    original = attach_player_combat_state(frame, k=10.0)
    future = pd.DataFrame(
        {
            "match_id": [99],
            "player_id": [11],
            "start_time": [T4],
            CAUSAL_C_COLUMN: [50.0],
            "position_number": [1.0],
            "position": ["POSITION_1"],
            "hero_damage": [10.0],
            "duration_seconds": [1800.0],
            "side": ["RADIANT"],
        }
    )
    combined = attach_player_combat_state(
        pd.concat([frame, future], ignore_index=True), k=10.0
    )
    earlier = original.loc[original["player_id"] == 11].copy()
    later = combined.loc[
        (combined["player_id"] == 11) & (combined["start_time"] <= T2)
    ]
    for column in (
        "combat_prior_n",
        "combat_prior_sum_c",
        "combat_prior_mean_c",
        "combat_shrunk_c",
    ):
        np.testing.assert_allclose(
            earlier[column].to_numpy(dtype=float),
            later[column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_other_player_history_cannot_enter_current_state() -> None:
    frame = _state_frame()
    original = attach_player_combat_state(frame, k=10.0)
    mutated = frame.copy()
    mutated.loc[mutated["player_id"] == 12, CAUSAL_C_COLUMN] = 99.0
    rerun = attach_player_combat_state(mutated, k=10.0)
    left = original.loc[original["player_id"] == 11]
    right = rerun.loc[rerun["player_id"] == 11]
    np.testing.assert_allclose(
        left["combat_prior_mean_c"].to_numpy(dtype=float),
        right["combat_prior_mean_c"].to_numpy(dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        left["combat_shrunk_c"].to_numpy(dtype=float),
        right["combat_shrunk_c"].to_numpy(dtype=float),
    )


def test_null_position_never_enters_baseline_or_target() -> None:
    rows = _stamp_rows(1, T0, (10.0, 8.0, 6.0, 4.0, 2.0))
    rows += _stamp_rows(2, T1, (20.0, 10.0, 5.0, 3.0, 2.0))
    rows.append(
        _appearance(
            match_id=3,
            player_id=99,
            start_time=T0,
            position=1,
            hero_damage=10_000.0,
        )
    )
    frame = pd.DataFrame(rows)
    frame.loc[frame["player_id"] == 99, "position"] = None
    frame.loc[frame["player_id"] == 99, "position_number"] = np.nan
    attached = attach_causal_candidate_c(frame)
    assert np.isnan(attached.loc[attached["player_id"] == 99, CAUSAL_C_COLUMN].iloc[0])
    pos1_t1 = attached.loc[
        (attached["start_time"] == T1) & (attached["position_number"] == 1.0)
    ].iloc[0]
    t0_pos1 = attached.loc[
        (attached["start_time"] == T0)
        & (attached["position_number"] == 1.0)
        & (attached["player_id"] != 99),
        COMBAT_C,
    ]
    assert int(pos1_t1["combat_position_baseline_n"]) == 1
    assert float(pos1_t1["combat_position_baseline"]) == pytest.approx(
        float(t0_pos1.iloc[0])
    )


def test_incomplete_team_damage_keeps_c_null() -> None:
    frame = pd.DataFrame(_stamp_rows(1, T0, (10.0, 8.0, 6.0, 4.0, 2.0)))
    frame.loc[2, "hero_damage"] = np.nan
    attached = attach_causal_candidate_c(frame)
    assert attached[COMBAT_C].isna().all()
    assert attached[CAUSAL_C_COLUMN].isna().all()
    later = _causal_ready_frame()
    later.loc[later["match_id"] == 3, "hero_damage"] = [
        12.0,
        9.0,
        np.nan,
        3.0,
        0.0,
    ]
    attached_later = attach_causal_candidate_c(later)
    t2 = attached_later.loc[attached_later["start_time"] == T2]
    assert t2[CAUSAL_C_COLUMN].isna().all()
    assert t2[COMBAT_C].isna().all()
    t1 = attached_later.loc[attached_later["start_time"] == T1]
    assert t1[CAUSAL_C_COLUMN].notna().all()


def test_zero_team_damage_keeps_c_null() -> None:
    rows = _stamp_rows(1, T0, (10.0, 8.0, 6.0, 4.0, 2.0))
    rows += _stamp_rows(2, T1, (0.0, 0.0, 0.0, 0.0, 0.0))
    attached = attach_causal_candidate_c(pd.DataFrame(rows))
    t1 = attached.loc[attached["start_time"] == T1]
    assert t1[COMBAT_C].isna().all()
    assert t1[CAUSAL_C_COLUMN].isna().all()
    assert hero_damage_share(t1).isna().all()


def test_warmup_does_not_use_another_position_mean() -> None:
    frame = pd.DataFrame(
        _stamp_rows(1, T0, (10.0, 8.0, 6.0, 4.0, 2.0))
        + _stamp_rows(2, T1, (20.0, 10.0, 5.0, 3.0, 2.0))
    )
    frame.loc[frame["start_time"] == T0, "position"] = [
        "POSITION_1",
        "POSITION_2",
        "POSITION_3",
        "POSITION_4",
        "POSITION_4",
    ]
    frame.loc[frame["start_time"] == T0, "position_number"] = [
        1.0,
        2.0,
        3.0,
        4.0,
        4.0,
    ]
    attached = attach_causal_candidate_c(frame)
    t1_pos5 = attached.loc[
        (attached["start_time"] == T1) & (attached["position_number"] == 5.0)
    ]
    assert t1_pos5[CAUSAL_C_COLUMN].isna().all()
    assert int(t1_pos5["combat_position_baseline_n"].iloc[0]) == 0
    t1_pos1 = attached.loc[
        (attached["start_time"] == T1) & (attached["position_number"] == 1.0)
    ]
    assert t1_pos1[CAUSAL_C_COLUMN].notna().all()


def test_zero_history_mean_is_null_and_shrunk_is_zero() -> None:
    frame = _state_frame()
    state = attach_player_combat_state(frame, k=10.0)
    first = state["start_time"] == T0
    assert (state.loc[first, "combat_prior_n"] == 0).all()
    assert state.loc[first, "combat_prior_mean_c"].isna().all()
    assert (state.loc[first, "combat_shrunk_c"] == 0.0).all()
    assert (state.loc[first, "combat_shrinkage_weight"] == 0.0).all()
    assert (state.loc[first, "combat_prior_sum_c"] == 0.0).all()


def test_shrinkage_formula_known_examples() -> None:
    assert combat_shrunk_c(0.20, 0, k=10.0) == 0.0
    assert combat_shrunk_c(None, 5, k=10.0) == 0.0
    assert combat_shrunk_c(0.20, 10, k=10.0) == pytest.approx(0.10)
    assert combat_shrunk_c(0.20, 5, k=5.0) == pytest.approx(0.10)
    assert combat_shrunk_c(0.40, 3, k=0.0) == pytest.approx(0.40)
    assert combat_shrinkage_weight(0, k=10.0) == 0.0
    assert combat_shrinkage_weight(10, k=10.0) == pytest.approx(0.5)
    assert combat_shrinkage_weight(5, k=0.0) == 1.0


def test_select_combat_k_picks_central_plateau_not_strongest() -> None:
    grid = pd.DataFrame(
        [
            {
                "k": k,
                "rmse": 1.0,
                "low_n_rmse": 1.20 if k == 0.0 else 1.00,
                "rmse_n_gt_20": 0.80,
            }
            for k in COMBAT_SHRINKAGE_GRID
        ]
    )
    selected, justification = select_combat_shrinkage_k(grid)
    assert selected == 20.0
    assert "central plateau" in justification


def test_monotonic_shrinkage_toward_raw_mean() -> None:
    mean = 0.20
    k = 10.0
    previous = abs(combat_shrunk_c(mean, 1, k=k) - mean)
    for n in (2, 5, 10, 20, 40, 80):
        gap = abs(combat_shrunk_c(mean, n, k=k) - mean)
        assert gap < previous
        previous = gap
    assert abs(combat_shrunk_c(mean, 80, k=k)) > abs(combat_shrunk_c(mean, 5, k=k))


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
        batch = player_rows(match_id, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
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
            item["hero_damage"] = 25000 - slot * 4000
            item["tower_damage"] = 1000
            item["hero_healing"] = 0
            item["level"] = 20
        players.extend(batch)

    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        report = run_player_combat_state_diagnostics(store)
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
    assert report.integrity["holdout_used_for_validation"] is False
    assert report.integrity["holdout_used_for_eb"] is False
    assert report.integrity["ti2026_used_for_k"] is False
    assert report.integrity["holdout_rows_in_development"] is False
    assert report.integrity["stratz_called"] is False
    assert report.integrity["model_trained"] is False
    assert report.integrity["win_model_benchmarked"] is False
    assert report.integrity["shrinkage_chosen_from_outcomes"] is False
    assert report.integrity["global_position_mean_fallback_used"] is False
    assert report.integrity["team_combat_feature_created"] is False
    assert report.integrity["farming_k_is_5"] is True
    assert report.integrity["slice17_candidate_unchanged"] is True
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["state_in_feature_columns"] is False
    assert report.selected_k in COMBAT_SHRINKAGE_GRID
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
    for name in SLICE18_STATE_COLUMNS:
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
    for name in CANDIDATE_COLUMN_NAMES:
        assert name not in FEATURE_COLUMNS
    assert COMBAT_C_POSITION not in FEATURE_COLUMNS
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


def test_prior_sum_is_n_times_mean() -> None:
    frame = _state_frame()
    counts, sums, means = prior_combat_history(frame)
    state = attach_player_combat_state(frame, k=0.0)
    np.testing.assert_array_equal(state["combat_prior_n"].to_numpy(), counts)
    np.testing.assert_allclose(state["combat_prior_sum_c"].to_numpy(), sums)
    at_t2 = (state["player_id"] == 11) & (state["start_time"] == T2)
    row = state.loc[at_t2].iloc[0]
    assert int(row["combat_prior_n"]) == 2
    assert float(row["combat_prior_sum_c"]) == pytest.approx(0.30)
    assert float(row["combat_prior_mean_c"]) == pytest.approx(0.15)
    assert float(row["combat_shrunk_c"]) == pytest.approx(0.15)
    assert float(means.loc[row.name]) == pytest.approx(0.15)


def test_slice17_share_formula_is_reused_not_redefined() -> None:
    frame = _causal_ready_frame()
    attached = attach_causal_candidate_c(frame)
    expected = hero_damage_share(frame)
    np.testing.assert_allclose(
        attached[COMBAT_C].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        equal_nan=True,
    )
    t1 = attached.loc[attached["start_time"] == T1].iloc[0]
    team = 20.0 + 10.0 + 5.0 + 3.0 + 2.0
    assert float(t1[COMBAT_C]) == pytest.approx(20.0 / team)
