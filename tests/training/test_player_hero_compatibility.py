"""Slice 23 player × hero behavioral-compatibility diagnostics.

Interaction-form discovery only. No production fit score, no win model.
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
from dota_predictor.training.hero_performance_profile import PLAYER_X_HERO_FIT_NAMES
from dota_predictor.training.hero_requirement_state import (
    FROZEN_HERO_COMBAT_SHRINKAGE_K,
    FROZEN_HERO_FARM_SHRINKAGE_K,
    SLICE22_STATE_COLUMNS,
    attach_hero_requirement_state,
)
from dota_predictor.training.player_combat_state import (
    CAUSAL_C_COLUMN,
    FROZEN_COMBAT_SHRINKAGE_K,
    attach_player_combat_state,
)
from dota_predictor.training.player_farming_state import (
    CAUSAL_B_COLUMN,
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    attach_player_farming_state,
    development_tune_end,
)
from dota_predictor.training.player_hero_compatibility import (
    COMBAT_SPEC,
    COMPATIBILITY_TERM_NAMES,
    FARMING_SPEC,
    MODEL_SPECS,
    SLICE23_DIAGNOSTIC_COLUMNS,
    attach_compatibility_terms,
    attach_player_hero_compatibility_terms,
    compatibility_terms,
    eligibility_mask,
    permute_hero_requirement,
    run_compatibility_diagnostics_on_frame,
    run_player_hero_compatibility_diagnostics,
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
    farming_state: float | None = None,
    combat_state: float | None = None,
    farming_n: int = 3,
    combat_n: int = 3,
    hero_farm: float | None = None,
    hero_combat: float | None = None,
    hero_n: int = 3,
    version: int = 176,
) -> dict[str, object]:
    p_farm = farming if farming_state is None else farming_state
    p_combat = combat if combat_state is None else combat_state
    h_farm = farming if hero_farm is None else hero_farm
    h_combat = combat if hero_combat is None else hero_combat
    return {
        "match_id": match_id,
        "player_id": player_id,
        "hero_id": hero_id,
        "position_number": float(position),
        "position": f"POSITION_{position}",
        "start_time": start_time,
        CAUSAL_B_COLUMN: farming,
        CAUSAL_C_COLUMN: combat,
        "farming_shrunk_b": p_farm,
        "combat_shrunk_c": p_combat,
        "farming_prior_n": farming_n,
        "combat_prior_n": combat_n,
        "hero_farming_shrunk_b": h_farm,
        "hero_combat_shrunk_c": h_combat,
        "hero_farming_prior_n": hero_n,
        "hero_combat_prior_n": hero_n,
        "hero_farming_unique_prior_players": max(hero_n, 1),
        "hero_combat_unique_prior_players": max(hero_n, 1),
        "hero_farming_top_player_share": 0.2,
        "hero_combat_top_player_share": 0.2,
        "hero_farming_inclusive_prior_mean_b": 99.0,
        "hero_combat_inclusive_prior_mean_c": 9.0,
        "hero_farming_inclusive_prior_n": 99,
        "hero_combat_inclusive_prior_n": 99,
        "game_version_id": version,
        "team_won": 1,
        "radiant_win": True,
        "elo_expected_win": 0.5,
    }


def _state_history_frame() -> pd.DataFrame:
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
                farming_state=0.0,
                combat_state=0.0,
                farming_n=0,
                combat_n=0,
                hero_n=0,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                farming=3.0,
                combat=0.30,
                farming_state=0.4,
                combat_state=0.04,
                farming_n=1,
                combat_n=1,
                hero_farm=1.0,
                hero_combat=0.10,
                hero_n=1,
            ),
            _row(
                match_id=3,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T2,
                farming=5.0,
                combat=0.50,
                farming_state=1.0,
                combat_state=0.10,
                farming_n=1,
                combat_n=1,
                hero_farm=3.0,
                hero_combat=0.30,
                hero_n=1,
            ),
        ]
    )


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


def _synthetic_development(n_times: int = 12, n_per_time: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    match_id = 1
    for t in range(n_times):
        start = T0 + timedelta(days=t * 15)
        version = 176 + (t // 4)
        for i in range(n_per_time):
            position = (i % 5) + 1
            p_farm = float(rng.normal(0.0, 1.0))
            h_farm = float(rng.normal(0.0, 1.0))
            p_combat = float(rng.normal(0.0, 0.2))
            h_combat = float(rng.normal(0.0, 0.2))
            farming = p_farm + 0.4 * h_farm + 0.8 * (p_farm - h_farm) ** 2
            combat = p_combat + 0.4 * h_combat
            rows.append(
                _row(
                    match_id=match_id,
                    player_id=100 + i,
                    hero_id=1 + (i % 7),
                    position=position,
                    start_time=start,
                    farming=farming,
                    combat=combat,
                    farming_state=p_farm,
                    combat_state=p_combat,
                    hero_farm=h_farm,
                    hero_combat=h_combat,
                    farming_n=5 + i,
                    combat_n=5 + i,
                    hero_n=3 + (i % 6),
                    version=version,
                )
            )
            match_id += 1
    return pd.DataFrame(rows)


def test_frozen_upstream_constants_unchanged() -> None:
    assert FROZEN_CANDIDATE_B == CANDIDATE_B
    assert FROZEN_SHRINKAGE_K == 5.0
    assert FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION
    assert FROZEN_COMBAT_SHRINKAGE_K == 20.0
    assert FROZEN_HERO_FARM_SHRINKAGE_K == 2.0
    assert FROZEN_HERO_COMBAT_SHRINKAGE_K == 2.0
    assert FARMING_SPEC.player_state == "farming_shrunk_b"
    assert COMBAT_SPEC.player_state == "combat_shrunk_c"
    assert FARMING_SPEC.hero_requirement == "hero_farming_shrunk_b"
    assert COMBAT_SPEC.hero_requirement == "hero_combat_shrunk_c"
    assert FARMING_SPEC.target == CAUSAL_B_COLUMN
    assert COMBAT_SPEC.target == CAUSAL_C_COLUMN


def test_compatibility_term_formulas() -> None:
    player = pd.Series([2.0, 0.0, -1.0])
    hero = pd.Series([0.5, 0.0, 2.0])
    terms = compatibility_terms(player, hero)
    np.testing.assert_allclose(terms["signed_gap"], [1.5, 0.0, -3.0])
    np.testing.assert_allclose(terms["abs_gap"], [1.5, 0.0, 3.0])
    np.testing.assert_allclose(terms["sq_gap"], [2.25, 0.0, 9.0])
    np.testing.assert_allclose(terms["interaction"], [1.0, 0.0, -2.0])
    np.testing.assert_allclose(terms["player_below_requirement"], [0.0, 0.0, 3.0])
    np.testing.assert_allclose(terms["player_above_requirement"], [1.5, 0.0, 0.0])
    np.testing.assert_allclose(
        terms["player_below_requirement"] + terms["player_above_requirement"],
        terms["abs_gap"],
    )
    np.testing.assert_allclose(
        terms["player_above_requirement"] - terms["player_below_requirement"],
        terms["signed_gap"],
    )


def test_uses_frozen_slice14_and_slice18_player_states() -> None:
    history = pd.DataFrame(
        [
            {
                "match_id": 1,
                "player_id": 11,
                "hero_id": 1,
                "position_number": 1.0,
                "position": "POSITION_1",
                "start_time": T0,
                CAUSAL_B_COLUMN: 2.0,
                CAUSAL_C_COLUMN: 0.20,
            },
            {
                "match_id": 2,
                "player_id": 11,
                "hero_id": 1,
                "position_number": 1.0,
                "position": "POSITION_1",
                "start_time": T1,
                CAUSAL_B_COLUMN: 4.0,
                CAUSAL_C_COLUMN: 0.40,
            },
        ]
    )
    farming = attach_player_farming_state(history, k=FROZEN_SHRINKAGE_K)
    combat = attach_player_combat_state(farming, k=FROZEN_COMBAT_SHRINKAGE_K)
    later = combat.loc[combat["start_time"] == T1].iloc[0]
    expected_farm = FROZEN_SHRINKAGE_K and (1.0 / (1.0 + 5.0) * 2.0)
    expected_combat = 1.0 / (1.0 + 20.0) * 0.20
    assert float(later["farming_shrunk_b"]) == pytest.approx(expected_farm)
    assert float(later["combat_shrunk_c"]) == pytest.approx(expected_combat)


def test_uses_frozen_slice22_lpo_hero_requirements() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_id": 1,
                "player_id": 11,
                "hero_id": 1,
                "position_number": 1.0,
                "position": "POSITION_1",
                "start_time": T0,
                CAUSAL_B_COLUMN: 1.0,
                CAUSAL_C_COLUMN: 0.10,
                "farming_shrunk_b": 0.0,
                "combat_shrunk_c": 0.0,
            },
            {
                "match_id": 2,
                "player_id": 12,
                "hero_id": 1,
                "position_number": 1.0,
                "position": "POSITION_1",
                "start_time": T1,
                CAUSAL_B_COLUMN: 3.0,
                CAUSAL_C_COLUMN: 0.30,
                "farming_shrunk_b": 0.2,
                "combat_shrunk_c": 0.02,
            },
        ]
    )
    state = attach_hero_requirement_state(
        frame,
        k_farm=FROZEN_HERO_FARM_SHRINKAGE_K,
        k_combat=FROZEN_HERO_COMBAT_SHRINKAGE_K,
    )
    later = state.loc[state["start_time"] == T1].iloc[0]
    assert int(later["hero_farming_prior_n"]) == 1
    expected = 1.0 / (1.0 + 2.0) * 1.0
    assert float(later["hero_farming_shrunk_b"]) == pytest.approx(expected)
    terms = attach_compatibility_terms(state, FARMING_SPEC)
    signed = float(terms.loc[terms["start_time"] == T1, "farming_signed_gap"].iloc[0])
    assert signed == pytest.approx(float(later["farming_shrunk_b"]) - expected)


def test_does_not_substitute_inclusive_hero_state() -> None:
    frame = _state_history_frame()
    original = attach_player_hero_compatibility_terms(frame)
    mutated = frame.copy()
    mutated["hero_farming_inclusive_prior_mean_b"] = 999.0
    mutated["hero_farming_inclusive_prior_n"] = 999
    mutated["hero_combat_inclusive_prior_mean_c"] = 99.0
    mutated["hero_combat_inclusive_prior_n"] = 999
    rerun = attach_player_hero_compatibility_terms(mutated)
    for name in COMPATIBILITY_TERM_NAMES:
        np.testing.assert_allclose(
            original[f"farming_{name}"].to_numpy(dtype=float),
            rerun[f"farming_{name}"].to_numpy(dtype=float),
        )
    np.testing.assert_array_equal(
        eligibility_mask(frame, FARMING_SPEC).to_numpy(),
        eligibility_mask(mutated, FARMING_SPEC).to_numpy(),
    )


def test_missing_player_state_and_hero_n0_are_ineligible() -> None:
    frame = _state_history_frame()
    mask = eligibility_mask(frame, FARMING_SPEC)
    assert bool(mask.iloc[0]) is False
    assert int(frame.iloc[0]["hero_farming_prior_n"]) == 0
    missing = frame.copy()
    missing.loc[1, "farming_shrunk_b"] = np.nan
    missing_mask = eligibility_mask(missing, FARMING_SPEC)
    assert bool(missing_mask.iloc[1]) is False
    assert bool(mask.iloc[1]) is True


def test_positions_limited_to_one_through_five() -> None:
    frame = _state_history_frame()
    frame.loc[1, "position"] = "UNKNOWN"
    frame.loc[1, "position_number"] = np.nan
    mask = eligibility_mask(frame, FARMING_SPEC)
    assert bool(mask.iloc[1]) is False
    frame.loc[2, "position_number"] = 6.0
    frame.loc[2, "position"] = "POSITION_6"
    mask = eligibility_mask(frame, FARMING_SPEC)
    assert bool(mask.iloc[2]) is False


def test_farming_and_combat_kept_independent() -> None:
    frame = _state_history_frame()
    terms = attach_player_hero_compatibility_terms(frame)
    mutated = frame.copy()
    mutated["hero_combat_shrunk_c"] = 50.0
    mutated["combat_shrunk_c"] = -50.0
    rerun = attach_player_hero_compatibility_terms(mutated)
    np.testing.assert_allclose(
        terms["farming_signed_gap"].to_numpy(dtype=float),
        rerun["farming_signed_gap"].to_numpy(dtype=float),
    )
    assert not np.allclose(
        terms["combat_signed_gap"].to_numpy(dtype=float),
        rerun["combat_signed_gap"].to_numpy(dtype=float),
    )


def test_future_and_same_timestamp_blindness_inherited() -> None:
    history = pd.DataFrame(
        [
            {
                "match_id": 1,
                "player_id": 11,
                "hero_id": 1,
                "position_number": 1.0,
                "position": "POSITION_1",
                "start_time": T0,
                CAUSAL_B_COLUMN: 1.0,
                CAUSAL_C_COLUMN: 0.10,
            },
            {
                "match_id": 2,
                "player_id": 12,
                "hero_id": 1,
                "position_number": 1.0,
                "position": "POSITION_1",
                "start_time": T1,
                CAUSAL_B_COLUMN: 3.0,
                CAUSAL_C_COLUMN: 0.30,
            },
            {
                "match_id": 3,
                "player_id": 13,
                "hero_id": 1,
                "position_number": 1.0,
                "position": "POSITION_1",
                "start_time": T1,
                CAUSAL_B_COLUMN: 9.0,
                CAUSAL_C_COLUMN: 0.90,
            },
        ]
    )
    original = attach_hero_requirement_state(
        attach_player_combat_state(
            attach_player_farming_state(history, k=FROZEN_SHRINKAGE_K),
            k=FROZEN_COMBAT_SHRINKAGE_K,
        ),
        k_farm=FROZEN_HERO_FARM_SHRINKAGE_K,
        k_combat=FROZEN_HERO_COMBAT_SHRINKAGE_K,
    )
    original = attach_player_hero_compatibility_terms(original)
    future = pd.DataFrame(
        [
            {
                "match_id": 99,
                "player_id": 12,
                "hero_id": 1,
                "position_number": 1.0,
                "position": "POSITION_1",
                "start_time": T4,
                CAUSAL_B_COLUMN: 99.0,
                CAUSAL_C_COLUMN: 9.0,
            }
        ]
    )
    combined = attach_hero_requirement_state(
        attach_player_combat_state(
            attach_player_farming_state(
                pd.concat([history, future], ignore_index=True), k=FROZEN_SHRINKAGE_K
            ),
            k=FROZEN_COMBAT_SHRINKAGE_K,
        ),
        k_farm=FROZEN_HERO_FARM_SHRINKAGE_K,
        k_combat=FROZEN_HERO_COMBAT_SHRINKAGE_K,
    )
    combined = attach_player_hero_compatibility_terms(combined)
    earlier = original.loc[original["start_time"] <= T1]
    later = combined.loc[combined["start_time"] <= T1]
    np.testing.assert_allclose(
        earlier["farming_signed_gap"].to_numpy(dtype=float),
        later["farming_signed_gap"].to_numpy(dtype=float),
        equal_nan=True,
    )
    at_t1 = original.loc[original["start_time"] == T1]
    assert at_t1["hero_farming_prior_n"].tolist() == [1, 1]


def test_permutation_preserves_position_and_version() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=i,
                player_id=10 + i,
                hero_id=1,
                position=1 if i <= 4 else 2,
                start_time=T0 + timedelta(days=i),
                farming=float(i),
                combat=0.1 * i,
                hero_farm=float(i * 10),
                hero_combat=float(i),
                version=176 if i <= 4 else 177,
            )
            for i in range(1, 9)
        ]
    )
    rng = np.random.default_rng(23)
    shuffled = permute_hero_requirement(
        frame, "hero_farming_shrunk_b", rng=rng
    )
    np.testing.assert_allclose(
        frame["farming_shrunk_b"].to_numpy(dtype=float),
        shuffled["farming_shrunk_b"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        frame[CAUSAL_B_COLUMN].to_numpy(dtype=float),
        shuffled[CAUSAL_B_COLUMN].to_numpy(dtype=float),
    )
    for (position, version), group in frame.groupby(
        ["position_number", "game_version_id"], sort=False
    ):
        original = np.sort(group["hero_farming_shrunk_b"].to_numpy(dtype=float))
        rerun = np.sort(
            shuffled.loc[group.index, "hero_farming_shrunk_b"].to_numpy(dtype=float)
        )
        np.testing.assert_allclose(original, rerun)
        _ = (position, version)
    assert not np.allclose(
        frame["hero_farming_shrunk_b"].to_numpy(dtype=float),
        shuffled["hero_farming_shrunk_b"].to_numpy(dtype=float),
    )


def test_tune_only_scaling_and_validation_does_not_fit() -> None:
    frame = _synthetic_development()
    report = run_compatibility_diagnostics_on_frame(
        frame,
        development_end=T4,
        n_permutations=6,
        n_bootstrap=8,
        rng_seed=23,
    )
    mutated = frame.copy()
    val_times = pd.to_datetime(mutated["start_time"], utc=True) > pd.Timestamp(
        report.tune_end
    )
    mutated.loc[val_times, CAUSAL_B_COLUMN] = 1_000.0
    mutated.loc[val_times, CAUSAL_C_COLUMN] = 1_000.0
    rerun = run_compatibility_diagnostics_on_frame(
        mutated,
        development_end=T4,
        n_permutations=6,
        n_bootstrap=8,
        rng_seed=23,
    )
    assert report.integrity["validation_used_for_form_fitting"] is False
    assert report.integrity["validation_used_for_scaling"] is False
    farm = report.farming_comparison
    farm_r = rerun.farming_comparison
    original_strongest = report.classification.iloc[0]["farming_strongest"]
    rerun_strongest = rerun.classification.iloc[0]["farming_strongest"]
    assert original_strongest == rerun_strongest
    orig_tune = farm.loc[(farm["split"] == "tune") & (farm["model"] == original_strongest)]
    rerun_tune = farm_r.loc[
        (farm_r["split"] == "tune") & (farm_r["model"] == rerun_strongest)
    ]
    np.testing.assert_allclose(
        orig_tune["rmse"].to_numpy(dtype=float),
        rerun_tune["rmse"].to_numpy(dtype=float),
    )


def test_signed_gap_is_redundant_with_additive_baseline() -> None:
    frame = _synthetic_development()
    report = run_compatibility_diagnostics_on_frame(
        frame,
        development_end=T4,
        n_permutations=0,
        n_bootstrap=0,
        rng_seed=23,
    )
    m4a = report.farming_comparison.loc[report.farming_comparison["model"] == "M4a"]
    assert bool(m4a["algebraically_redundant"].all())
    diffs = m4a["max_abs_pred_diff_vs_m3"].to_numpy(dtype=float)
    assert np.nanmax(diffs) < 1e-8
    collinear = report.farming_collinearity.set_index("model")
    assert bool(collinear.loc["M4a", "rank_deficient"])
    assert int(collinear.loc["M4a", "rank_unscaled"]) == 3
    assert bool(collinear.loc["M4e", "rank_deficient"])


def test_no_win_result_or_team_aggregation() -> None:
    predictor_names = {name for spec in MODEL_SPECS for name in spec.predictors}
    for banned in (
        "team_won",
        "radiant_win",
        "match_id",
        "radiant_mean",
        "dire_mean",
        "lineup",
        "synergy",
        "counter",
    ):
        assert banned not in predictor_names
    frame = _synthetic_development()
    report = run_compatibility_diagnostics_on_frame(
        frame,
        development_end=T4,
        n_permutations=0,
        n_bootstrap=0,
    )
    assert report.integrity["win_model_run"] is False
    assert report.integrity["win_labels_used_in_predictors"] is False
    assert report.integrity["team_aggregation"] is False
    assert report.integrity["team_feature_created"] is False
    assert report.farming_semantics["win_labels_used"] is False
    assert report.combat_semantics["team_aggregated"] is False


def test_feature_columns_remain_thirty_three_and_no_fit_registered() -> None:
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
    for name in (
        *SLICE23_DIAGNOSTIC_COLUMNS,
        *SLICE22_STATE_COLUMNS,
        *PLAYER_X_HERO_FIT_NAMES,
        CAUSAL_B_COLUMN,
        CAUSAL_C_COLUMN,
        COMBAT_C,
        "player_hero_fit",
        "farming_fit",
        "combat_fit",
    ):
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
        assert name not in SNAPSHOT_COLUMNS
        assert name not in PRE_DRAFT_SNAPSHOT_SQL
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in FEATURE_COLUMNS
        pre_draft = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
        assert column not in pre_draft


def test_holdout_excluded_and_integrity_flags(tmp_path: Path) -> None:
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
        report = run_player_hero_compatibility_diagnostics(
            store, n_permutations=4, n_bootstrap=4
        )
    assert report.n_development_matches == 2
    assert report.n_development_player_rows == 20
    assert report.n_holdout_excluded == 10
    assert report.integrity["holdout_used_for_fitting"] is False
    assert report.integrity["holdout_used_for_form_selection"] is False
    assert report.integrity["holdout_used_for_scaling"] is False
    assert report.development_end == FROZEN_DEVELOPMENT_END
    assert report.integrity["farming_player_k_is_5"] is True
    assert report.integrity["combat_player_k_is_20"] is True
    assert report.integrity["hero_farm_k_is_2"] is True
    assert report.integrity["hero_combat_k_is_2"] is True
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["player_hero_fit_created"] is False
    assert report.integrity["fit_score_frozen"] is False
    assert report.integrity["inclusive_hero_state_used"] is False
    holdout = pd.DataFrame({"start_time": [later], "hero_id": [1], "player_id": [11]})
    assert restrict_development(holdout).empty
    times = pd.to_datetime([T0, T1, T2], utc=True)
    tune_end = development_tune_end(pd.Series(times), development_end=T2)
    assert tune_end in {T0, T1, T2}


def test_synthetic_quadratic_mismatch_is_detectable_on_farming() -> None:
    frame = _synthetic_development()
    report = run_compatibility_diagnostics_on_frame(
        frame,
        development_end=T4,
        n_permutations=12,
        n_bootstrap=20,
        rng_seed=23,
    )
    farm = report.farming_comparison
    m3 = farm.loc[(farm["split"] == "validation") & (farm["model"] == "M3")].iloc[0]
    m4c = farm.loc[(farm["split"] == "validation") & (farm["model"] == "M4c")].iloc[0]
    assert float(m4c["rmse"]) < float(m3["rmse"])
    combat = report.combat_comparison
    c4c = combat.loc[(combat["split"] == "validation") & (combat["model"] == "M4c")].iloc[0]
    assert abs(float(c4c["delta_rmse"])) < abs(float(m4c["delta_rmse"]))
    assert not report.classification.empty
    assert report.integrity["same_timestamp_groups_disjoint"] is True
