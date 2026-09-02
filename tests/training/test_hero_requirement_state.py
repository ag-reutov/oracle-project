"""Slice 22 leakage-safe hero×position requirement states.

Leave-current-player-out history, shrinkage, freeze boundaries, no fit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import (
    build_feature_store_config,
    match_row,
    player_rows,
)

from dota_predictor.data.canonical_schema import MATCH_PLAYER_BOX_SCORE_COLUMNS
from dota_predictor.features.availability import (
    SnapshotStage,
    columns_allowed_for_stage,
)
from dota_predictor.features.duckdb_layer import MATCH_PLAYERS_VIEW, connect
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.training.combat_performance_target import (
    COMBAT_C,
    COMBAT_C_POSITION,
    FROZEN_COMBAT_CANDIDATE,
)
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.hero_performance_profile import (
    HERO_COMBAT_PROFILE_KEY,
    HERO_COMBAT_PROFILE_TARGET,
    HERO_FARMING_PROFILE_KEY,
    HERO_FARMING_PROFILE_TARGET,
    PLAYER_X_HERO_FIT_NAMES,
    PROFILE_SPECS,
)
from dota_predictor.training.hero_requirement_state import (
    FROZEN_HERO_COMBAT_SHRINKAGE_K,
    FROZEN_HERO_FARM_SHRINKAGE_K,
    HERO_COMBAT_SHRINKAGE_GRID,
    HERO_FARM_SHRINKAGE_GRID,
    SLICE22_STATE_COLUMNS,
    attach_hero_requirement_state,
    hero_requirement_shrinkage_weight,
    hero_requirement_shrunk,
    prior_hero_position_history,
    run_hero_requirement_state_diagnostics,
    select_hero_combat_shrinkage_k,
    select_hero_farm_shrinkage_k,
)
from dota_predictor.training.player_combat_state import (
    CAUSAL_C_COLUMN,
    FROZEN_COMBAT_SHRINKAGE_K,
)
from dota_predictor.training.player_farming_state import (
    CAUSAL_B_COLUMN,
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    development_tune_end,
)
from dota_predictor.training.player_performance_target import restrict_development
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


def _row(
    *,
    match_id: int,
    player_id: int,
    hero_id: int,
    position: int,
    start_time: datetime,
    farming: float,
    combat: float,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "hero_id": hero_id,
        "position_number": float(position),
        "position": f"POSITION_{position}",
        "start_time": start_time,
        CAUSAL_B_COLUMN: farming,
        CAUSAL_C_COLUMN: combat,
        "game_version_id": 176,
        "team_won": 1,
        "elo_expected_win": 0.5,
    }


def _state_frame() -> pd.DataFrame:
    """Known causal B/C. Residualizer / position baseline are not used."""
    return pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                farming=1.0,
                combat=0.10,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                farming=3.0,
                combat=0.30,
            ),
            _row(
                match_id=3,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T2,
                farming=5.0,
                combat=0.50,
            ),
            _row(
                match_id=4,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T3,
                farming=7.0,
                combat=0.70,
            ),
        ]
    )


def _specialist_frame() -> pd.DataFrame:
    """Player 11 owns almost all of hero 1 × position 1."""
    rows = [
        _row(
            match_id=i,
            player_id=11,
            hero_id=1,
            position=1,
            start_time=T0 + timedelta(days=i),
            farming=10.0,
            combat=0.20,
        )
        for i in range(1, 6)
    ]
    rows.append(
        _row(
            match_id=20,
            player_id=99,
            hero_id=1,
            position=1,
            start_time=T0 + timedelta(days=10),
            farming=0.0,
            combat=0.0,
        )
    )
    rows.append(
        _row(
            match_id=21,
            player_id=11,
            hero_id=1,
            position=1,
            start_time=T0 + timedelta(days=20),
            farming=10.0,
            combat=0.20,
        )
    )
    rows.append(
        _row(
            match_id=22,
            player_id=99,
            hero_id=1,
            position=1,
            start_time=T0 + timedelta(days=21),
            farming=0.0,
            combat=0.0,
        )
    )
    return pd.DataFrame(rows)


def _annotate_players(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        slot = int(item["slot_in_side"])
        item["position"] = POSITIONS[slot]
        item["num_last_hits"] = 300 - slot * 40
        item["kills"] = 1
        item["deaths"] = 1
        item["assists"] = 1
        item["gold_per_minute"] = 400
        item["experience_per_minute"] = 400
        item["num_denies"] = 0
        item["networth"] = 10_000
        item["hero_damage"] = 12_000 - slot * 1_500
        item["tower_damage"] = 1_000
        item["hero_healing"] = 0
        item["level"] = 20
        annotated.append(item)
    return annotated


def test_frozen_slice21_targets_and_player_k_unchanged() -> None:
    assert HERO_FARMING_PROFILE_TARGET == CAUSAL_B_COLUMN
    assert HERO_COMBAT_PROFILE_TARGET == CAUSAL_C_COLUMN
    assert HERO_FARMING_PROFILE_KEY == "hero_id × position"
    assert HERO_COMBAT_PROFILE_KEY == "hero_id × position"
    assert FROZEN_CANDIDATE_B == CANDIDATE_B
    assert FROZEN_SHRINKAGE_K == 5.0
    assert FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION
    assert FROZEN_COMBAT_SHRINKAGE_K == 20.0
    assert HERO_FARM_SHRINKAGE_GRID == (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    assert HERO_COMBAT_SHRINKAGE_GRID == HERO_FARM_SHRINKAGE_GRID
    assert FROZEN_HERO_FARM_SHRINKAGE_K in HERO_FARM_SHRINKAGE_GRID
    assert FROZEN_HERO_COMBAT_SHRINKAGE_K in HERO_COMBAT_SHRINKAGE_GRID
    assert FROZEN_HERO_FARM_SHRINKAGE_K == 2.0
    assert FROZEN_HERO_COMBAT_SHRINKAGE_K == 2.0
    assert FROZEN_HERO_FARM_SHRINKAGE_K != FROZEN_SHRINKAGE_K
    assert FROZEN_HERO_COMBAT_SHRINKAGE_K != FROZEN_COMBAT_SHRINKAGE_K
    names = {spec.name for spec in PROFILE_SPECS}
    assert "F1" in names and "F2" in names and "C1" in names and "C2" in names


def test_n0_raw_mean_null_shrunk_zero() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                farming=1.0,
                combat=0.10,
            )
        ]
    )
    state = attach_hero_requirement_state(frame, k_farm=10.0, k_combat=20.0)
    assert int(state["hero_farming_prior_n"].iloc[0]) == 0
    assert int(state["hero_combat_prior_n"].iloc[0]) == 0
    assert np.isnan(state["hero_farming_prior_mean_b"].iloc[0])
    assert np.isnan(state["hero_combat_prior_mean_c"].iloc[0])
    assert float(state["hero_farming_shrunk_b"].iloc[0]) == 0.0
    assert float(state["hero_combat_shrunk_c"].iloc[0]) == 0.0


def test_shrinkage_formula_and_monotonicity() -> None:
    mean = 2.0
    n = 5.0
    assert hero_requirement_shrinkage_weight(0.0, k=10.0) == 0.0
    assert hero_requirement_shrunk(mean, 0.0, k=10.0) == 0.0
    assert hero_requirement_shrinkage_weight(n, k=0.0) == 1.0
    assert hero_requirement_shrunk(mean, n, k=0.0) == pytest.approx(mean)
    expected = n / (n + 10.0) * mean
    assert hero_requirement_shrunk(mean, n, k=10.0) == pytest.approx(expected)
    previous = abs(hero_requirement_shrunk(mean, n, k=0.0))
    for k in HERO_FARM_SHRINKAGE_GRID[1:]:
        gap = abs(hero_requirement_shrunk(mean, n, k=k))
        assert gap < previous
        previous = gap


def test_leave_current_player_out_excludes_own_history() -> None:
    state = attach_hero_requirement_state(_state_frame(), k_farm=0.0, k_combat=0.0)
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    # Player 11 at T2: only player 12's T1 observation (3.0), not own T0 (1.0).
    assert int(t2["hero_farming_prior_n"]) == 1
    assert float(t2["hero_farming_prior_mean_b"]) == pytest.approx(3.0)
    assert int(t2["hero_farming_current_player_prior_n"]) == 1
    assert float(t2["hero_farming_inclusive_prior_mean_b"]) == pytest.approx(2.0)
    t3 = state.loc[state["start_time"] == T3].iloc[0]
    # Player 12 at T3: player 11's T0 and T2 (1 and 5), not own T1 (3).
    assert int(t3["hero_farming_prior_n"]) == 2
    assert float(t3["hero_farming_prior_mean_b"]) == pytest.approx(3.0)
    assert float(t3["hero_farming_inclusive_prior_mean_b"]) == pytest.approx(
        (1.0 + 3.0 + 5.0) / 3.0
    )


def test_other_player_observations_enter_lpo_state() -> None:
    state = attach_hero_requirement_state(_state_frame(), k_farm=0.0, k_combat=0.0)
    t2 = state.loc[state["player_id"] == 11].iloc[-1]
    assert float(t2["hero_farming_prior_sum_b"]) == pytest.approx(3.0)
    assert float(t2["hero_combat_prior_mean_c"]) == pytest.approx(0.30)


def test_inclusive_and_lpo_differ_on_specialist_case() -> None:
    state = attach_hero_requirement_state(_specialist_frame(), k_farm=0.0, k_combat=0.0)
    last_p11 = state.loc[state["match_id"] == 21].iloc[0]
    last_p99 = state.loc[state["match_id"] == 22].iloc[0]
    assert int(last_p11["hero_farming_inclusive_prior_n"]) == 6
    assert int(last_p11["hero_farming_prior_n"]) == 1
    assert float(last_p11["hero_farming_prior_mean_b"]) == pytest.approx(0.0)
    assert float(last_p11["hero_farming_inclusive_prior_mean_b"]) == pytest.approx(
        50.0 / 6.0
    )
    assert int(last_p99["hero_farming_prior_n"]) == 6
    assert float(last_p99["hero_farming_prior_mean_b"]) == pytest.approx(10.0)
    assert int(last_p99["hero_farming_inclusive_prior_n"]) == 7


def test_future_observations_cannot_change_earlier_state() -> None:
    base = _state_frame()
    original = attach_hero_requirement_state(base, k_farm=10.0, k_combat=20.0)
    future = pd.DataFrame(
        [
            _row(
                match_id=99,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T4,
                farming=99.0,
                combat=9.0,
            )
        ]
    )
    combined = attach_hero_requirement_state(
        pd.concat([base, future], ignore_index=True), k_farm=10.0, k_combat=20.0
    )
    earlier = original.loc[original["start_time"] <= T3]
    later = combined.loc[combined["start_time"] <= T3]
    for column in SLICE22_STATE_COLUMNS:
        np.testing.assert_allclose(
            earlier[column].to_numpy(dtype=float),
            later[column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_same_timestamp_rows_are_mutually_blind() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                farming=1.0,
                combat=0.10,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                farming=10.0,
                combat=1.0,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T1,
                farming=30.0,
                combat=3.0,
            ),
            _row(
                match_id=4,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T2,
                farming=0.0,
                combat=0.0,
            ),
        ]
    )
    state = attach_hero_requirement_state(frame, k_farm=0.0, k_combat=0.0)
    at_t1 = state["start_time"] == T1
    assert state.loc[at_t1, "hero_farming_prior_n"].tolist() == [1, 1]
    np.testing.assert_allclose(
        state.loc[at_t1, "hero_farming_prior_mean_b"].to_numpy(dtype=float),
        [1.0, 1.0],
    )
    at_t2 = state.loc[state["start_time"] == T2].iloc[0]
    # LPO for player 12 at T2: T0 (player 11) + T1 player 13, not own T1.
    assert int(at_t2["hero_farming_prior_n"]) == 2
    assert float(at_t2["hero_farming_prior_mean_b"]) == pytest.approx(15.5)


def test_hero_isolation() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                farming=1.0,
                combat=0.10,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=2,
                position=1,
                start_time=T1,
                farming=9.0,
                combat=0.90,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T2,
                farming=2.0,
                combat=0.20,
            ),
        ]
    )
    state = attach_hero_requirement_state(frame, k_farm=0.0, k_combat=0.0)
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    assert int(t2["hero_farming_prior_n"]) == 1
    assert float(t2["hero_farming_prior_mean_b"]) == pytest.approx(1.0)


def test_position_isolation() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                farming=1.0,
                combat=0.10,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=2,
                start_time=T1,
                farming=9.0,
                combat=0.90,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T2,
                farming=2.0,
                combat=0.20,
            ),
        ]
    )
    state = attach_hero_requirement_state(frame, k_farm=0.0, k_combat=0.0)
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    assert int(t2["hero_farming_prior_n"]) == 1
    assert float(t2["hero_farming_prior_mean_b"]) == pytest.approx(1.0)
    t1_pos2 = state.loc[state["position_number"] == 2.0].iloc[0]
    assert int(t1_pos2["hero_farming_prior_n"]) == 0


def test_explicit_positions_one_to_five_only() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                farming=1.0,
                combat=0.10,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                farming=9.0,
                combat=0.90,
            ),
        ]
    )
    frame.loc[frame["match_id"] == 2, "position"] = "UNKNOWN"
    frame.loc[frame["match_id"] == 2, "position_number"] = np.nan
    extra = _row(
        match_id=3,
        player_id=13,
        hero_id=1,
        position=1,
        start_time=T2,
        farming=2.0,
        combat=0.20,
    )
    frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
    state = attach_hero_requirement_state(frame, k_farm=0.0, k_combat=0.0)
    unknown = state.loc[state["match_id"] == 2].iloc[0]
    assert int(unknown["hero_farming_prior_n"]) == 0
    assert np.isnan(unknown["hero_farming_prior_mean_b"])
    t2 = state.loc[state["match_id"] == 3].iloc[0]
    assert int(t2["hero_farming_prior_n"]) == 1
    assert float(t2["hero_farming_prior_mean_b"]) == pytest.approx(1.0)


def test_non_finite_target_does_not_increment_n() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                farming=1.0,
                combat=0.10,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                farming=float("nan"),
                combat=float("nan"),
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T2,
                farming=2.0,
                combat=0.20,
            ),
        ]
    )
    state = attach_hero_requirement_state(frame, k_farm=0.0, k_combat=0.0)
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    assert int(t2["hero_farming_prior_n"]) == 1
    assert float(t2["hero_farming_prior_mean_b"]) == pytest.approx(1.0)


def test_select_farm_k_prefers_stronger_equivalent_shrinkage() -> None:
    grid = pd.DataFrame(
        [
            {
                "k": k,
                "rmse": 1.0,
                "low_n_rmse": 1.20 if k == 0.0 else 1.00,
                "rmse_n_gt_20": 0.80,
            }
            for k in HERO_FARM_SHRINKAGE_GRID
        ]
    )
    selected, justification = select_hero_farm_shrinkage_k(grid)
    assert selected == 80.0
    assert "strongest" in justification or "equivalent" in justification


def test_select_combat_k_picks_central_plateau() -> None:
    grid = pd.DataFrame(
        [
            {
                "k": k,
                "rmse": 1.0,
                "low_n_rmse": 1.20 if k == 0.0 else 1.00,
                "rmse_n_gt_20": 0.80,
            }
            for k in HERO_COMBAT_SHRINKAGE_GRID
        ]
    )
    selected, justification = select_hero_combat_shrinkage_k(grid)
    assert selected in HERO_COMBAT_SHRINKAGE_GRID
    assert selected != 80.0
    assert "central" in justification or "equivalent" in justification


def test_feature_columns_remain_thirty_three_and_no_fit() -> None:
    assert len(FEATURE_COLUMNS) == 33
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)
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
    for name in (*SLICE22_STATE_COLUMNS, *PLAYER_X_HERO_FIT_NAMES):
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
        assert name not in SNAPSHOT_COLUMNS
        assert name not in PRE_DRAFT_SNAPSHOT_SQL
    for name in (
        CAUSAL_B_COLUMN,
        CAUSAL_C_COLUMN,
        COMBAT_C,
        "hero_farming_profile",
        "hero_combat_profile",
        "player_hero_fit",
    ):
        assert name not in FEATURE_COLUMNS
        assert name not in PRE_DRAFT_SNAPSHOT_SQL
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in FEATURE_COLUMNS
        pre_draft = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
        assert column not in pre_draft


def test_development_cutoff_and_holdout_exclusion(tmp_path: Path) -> None:
    later = FROZEN_DEVELOPMENT_END + timedelta(days=1)
    matches = [
        match_row(
            1,
            start_time=T0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
            game_version_id=176,
        ),
        match_row(
            2,
            start_time=T1,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=False,
            game_version_id=177,
        ),
        match_row(
            3,
            start_time=later,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
            game_version_id=177,
        ),
    ]
    players = _annotate_players(
        player_rows(1, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(2, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(3, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
    )
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
        for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
            assert column not in view_columns
        report = run_hero_requirement_state_diagnostics(store)
    assert report.n_development_matches == 2
    assert report.n_development_player_rows == 20
    assert report.n_holdout_excluded == 10
    assert report.integrity["holdout_used_for_k_selection"] is False
    assert report.integrity["holdout_used_for_validation"] is False
    assert report.integrity["holdout_used_for_eb"] is False
    assert report.development_end == FROZEN_DEVELOPMENT_END
    holdout = pd.DataFrame(
        {"start_time": [later], "hero_id": [1], "player_id": [11]}
    )
    restricted = restrict_development(holdout)
    assert restricted.empty


def test_integrity_flags_on_full_run(tmp_path: Path) -> None:
    matches = [
        match_row(
            i,
            start_time=T0 + timedelta(days=i),
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=bool(i % 2),
            game_version_id=176 + (i // 2),
        )
        for i in range(1, 8)
    ]
    players = _annotate_players(
        [
            row
            for i in range(1, 8)
            for row in player_rows(i, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        ]
    )
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        report = run_hero_requirement_state_diagnostics(store)
    assert report.integrity["stratz_called"] is False
    assert report.integrity["ingestion_modified"] is False
    assert report.integrity["schema_modified"] is False
    assert report.integrity["slice21_farming_target_unchanged"] is True
    assert report.integrity["slice21_combat_target_unchanged"] is True
    assert report.integrity["farming_candidate_b_unchanged"] is True
    assert report.integrity["farming_player_k_is_5"] is True
    assert report.integrity["combat_candidate_c_unchanged"] is True
    assert report.integrity["combat_player_k_is_20"] is True
    assert report.integrity["player_hero_fit_created"] is False
    assert report.integrity["current_position_resolved"] is False
    assert report.integrity["team_feature_created"] is False
    assert report.integrity["win_model_run"] is False
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["full_development_mean_fallback"] is False
    assert not report.classification.empty
    assert report.tune_end <= report.development_end
    times = pd.to_datetime([T0, T1, T2], utc=True)
    tune_end = development_tune_end(pd.Series(times), development_end=T2)
    assert tune_end in {T0, T1, T2}


def test_prior_history_helper_matches_attach() -> None:
    frame = _state_frame()
    n, _total, mean, unique, top = prior_hero_position_history(
        frame, CAUSAL_B_COLUMN, leave_player_out=True
    )
    state = attach_hero_requirement_state(frame, k_farm=0.0, k_combat=0.0)
    np.testing.assert_array_equal(n.to_numpy(), state["hero_farming_prior_n"].to_numpy())
    np.testing.assert_allclose(
        mean.to_numpy(dtype=float),
        state["hero_farming_prior_mean_b"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert unique.iloc[-1] >= 1
    assert np.isnan(top.iloc[0])
