"""Slice 24 current-meta Hero × Position state diagnostics.

Temporal leakage, same-timestamp blindness, version/recent semantics,
cold starts. Research state only; not a production feature.
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
from dota_predictor.features.hero_meta import RECENT_WINDOW_DAYS
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
)
from dota_predictor.training.hero_position_meta_state import (
    ELO_RESIDUAL_COLUMN,
    RECENT_WINDOW_DAYS_ALT,
    RECENT_WINDOW_DAYS_PRIMARY,
    SLICE24_DIAGNOSTIC_COLUMNS,
    SLICE24_DIAGNOSTIC_ONLY,
    SLICE24_FROZEN_COMPONENTS,
    SLICE24_RESEARCH_CLASSIFICATION,
    SLICE24_RESIDUAL_SHRINKAGE_K_FROZEN,
    SLICE24_STATE_COLUMNS,
    WINDOW_SPECS,
    attach_elo_residual,
    attach_hero_position_meta_state,
    causal_previous_version_id,
    classify_slice24,
    run_hero_position_meta_diagnostics,
    slice24_report_to_jsonable,
)
from dota_predictor.training.hero_requirement_state import (
    FROZEN_HERO_COMBAT_SHRINKAGE_K,
    FROZEN_HERO_FARM_SHRINKAGE_K,
    SLICE22_STATE_COLUMNS,
    prior_hero_position_history,
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
from dota_predictor.training.player_hero_compatibility import (
    SLICE23_DIAGNOSTIC_ONLY,
    SLICE23_FIT_SCORE_FROZEN,
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
    residual: float,
    farming: float = 0.0,
    combat: float = 0.0,
    game_version_id: int = 176,
    team_won: int = 1,
    elo_expected_win: float = 0.5,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "hero_id": hero_id,
        "position_number": float(position),
        "position": f"POSITION_{position}",
        "start_time": start_time,
        ELO_RESIDUAL_COLUMN: residual,
        CAUSAL_B_COLUMN: farming,
        CAUSAL_C_COLUMN: combat,
        "game_version_id": game_version_id,
        "team_won": team_won,
        "elo_expected_win": elo_expected_win,
    }


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


def test_windows_follow_existing_conventions() -> None:
    names = tuple(spec.name for spec in WINDOW_SPECS)
    assert names == (
        "expanding",
        "recent_90d",
        "recent_180d",
        "current_version",
        "current_plus_previous",
    )
    assert RECENT_WINDOW_DAYS_PRIMARY == RECENT_WINDOW_DAYS == 90
    assert RECENT_WINDOW_DAYS_ALT == 180
    by_name = {spec.name: spec for spec in WINDOW_SPECS}
    assert by_name["recent_90d"].window_days == 90
    assert by_name["recent_180d"].window_days == 180
    assert by_name["expanding"].window_days is None
    assert by_name["current_version"].version_mode == "current"
    assert by_name["current_plus_previous"].version_mode == "current_plus_previous"


def test_frozen_prior_slices_unchanged() -> None:
    assert HERO_FARMING_PROFILE_TARGET == CAUSAL_B_COLUMN
    assert HERO_COMBAT_PROFILE_TARGET == CAUSAL_C_COLUMN
    assert HERO_FARMING_PROFILE_KEY == "hero_id × position"
    assert HERO_COMBAT_PROFILE_KEY == "hero_id × position"
    assert FROZEN_CANDIDATE_B == CANDIDATE_B
    assert FROZEN_SHRINKAGE_K == 5.0
    assert FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION
    assert FROZEN_COMBAT_SHRINKAGE_K == 20.0
    assert FROZEN_HERO_FARM_SHRINKAGE_K == 2.0
    assert FROZEN_HERO_COMBAT_SHRINKAGE_K == 2.0
    assert SLICE23_FIT_SCORE_FROZEN is False
    assert SLICE23_DIAGNOSTIC_ONLY is True
    assert SLICE24_RESIDUAL_SHRINKAGE_K_FROZEN is False
    assert SLICE24_DIAGNOSTIC_ONLY is True
    assert SLICE24_FROZEN_COMPONENTS == ()
    assert SLICE24_RESEARCH_CLASSIFICATION == "C"


def test_elo_residual_is_result_minus_pre_match_expectation() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                residual=np.nan,
                team_won=1,
                elo_expected_win=0.4,
            )
        ]
    )
    del frame[ELO_RESIDUAL_COLUMN]
    attached = attach_elo_residual(frame)
    assert float(attached.iloc[0][ELO_RESIDUAL_COLUMN]) == pytest.approx(0.6)


def test_future_rows_do_not_enter_earlier_state() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                residual=0.20,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                residual=0.40,
            ),
        ]
    )
    base = attach_hero_position_meta_state(frame)
    extra = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    _row(
                        match_id=3,
                        player_id=13,
                        hero_id=1,
                        position=1,
                        start_time=T2,
                        residual=0.90,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    later = attach_hero_position_meta_state(extra)
    t1 = base.loc[base["start_time"] == T1].iloc[0]
    t1_later = later.loc[later["start_time"] == T1].iloc[0]
    assert int(t1["hp_expanding_n"]) == int(t1_later["hp_expanding_n"]) == 1
    assert float(t1["hp_expanding_elo_residual_mean"]) == pytest.approx(0.20)
    assert float(t1_later["hp_expanding_elo_residual_mean"]) == pytest.approx(0.20)


def test_same_timestamp_rows_are_mutually_blind() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                residual=0.10,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                residual=0.50,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T1,
                residual=0.90,
            ),
            _row(
                match_id=4,
                player_id=14,
                hero_id=1,
                position=1,
                start_time=T2,
                residual=0.00,
            ),
        ]
    )
    state = attach_hero_position_meta_state(frame)
    at_t1 = state.loc[state["start_time"] == T1]
    assert at_t1["hp_expanding_n"].tolist() == [1, 1]
    np.testing.assert_allclose(
        at_t1["hp_expanding_elo_residual_mean"].to_numpy(dtype=float),
        [0.10, 0.10],
    )
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    assert int(t2["hp_expanding_n"]) == 3
    assert float(t2["hp_expanding_elo_residual_mean"]) == pytest.approx(
        (0.10 + 0.50 + 0.90) / 3.0
    )


def test_expanding_residual_matches_slice22_inclusive_history() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=i,
                player_id=10 + i,
                hero_id=1,
                position=1,
                start_time=T0 + timedelta(days=i),
                residual=0.1 * i,
            )
            for i in range(1, 5)
        ]
    )
    state = attach_hero_position_meta_state(frame)
    n, _total, mean, _unique, _top = prior_hero_position_history(
        frame, ELO_RESIDUAL_COLUMN, leave_player_out=False
    )
    np.testing.assert_array_equal(
        n.to_numpy(), state["hp_expanding_elo_residual_n"].to_numpy()
    )
    np.testing.assert_allclose(
        mean.to_numpy(dtype=float),
        state["hp_expanding_elo_residual_mean"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_hero_and_position_isolation() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                residual=0.40,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=2,
                position=1,
                start_time=T1,
                residual=0.80,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=2,
                start_time=T1,
                residual=0.90,
            ),
            _row(
                match_id=4,
                player_id=14,
                hero_id=1,
                position=1,
                start_time=T2,
                residual=0.00,
            ),
        ]
    )
    state = attach_hero_position_meta_state(frame)
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    assert int(t2["hp_expanding_n"]) == 1
    assert float(t2["hp_expanding_elo_residual_mean"]) == pytest.approx(0.40)


def test_recent_window_excludes_older_than_cutoff() -> None:
    old = T2 - timedelta(days=91)
    inside = T2 - timedelta(days=30)
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=old,
                residual=0.80,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=inside,
                residual=0.20,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T2,
                residual=0.00,
            ),
        ]
    )
    state = attach_hero_position_meta_state(frame)
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    assert int(t2["hp_expanding_n"]) == 2
    assert int(t2["hp_recent_90d_n"]) == 1
    assert float(t2["hp_recent_90d_elo_residual_mean"]) == pytest.approx(0.20)
    assert int(t2["hp_recent_180d_n"]) == 2


def test_current_version_is_blind_to_other_patches() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                residual=0.30,
                game_version_id=176,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                residual=0.90,
                game_version_id=177,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T2,
                residual=0.00,
                game_version_id=177,
            ),
        ]
    )
    state = attach_hero_position_meta_state(frame)
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    assert int(t2["hp_expanding_n"]) == 2
    assert int(t2["hp_current_version_n"]) == 1
    assert float(t2["hp_current_version_elo_residual_mean"]) == pytest.approx(0.90)
    assert int(t2["hp_current_plus_previous_n"]) == 2


def test_previous_version_is_causal() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                residual=0.10,
                game_version_id=176,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                residual=0.20,
                game_version_id=177,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T2,
                residual=0.30,
                game_version_id=178,
            ),
        ]
    )
    previous = causal_previous_version_id(frame)
    assert pd.isna(previous.iloc[0])
    assert int(previous.iloc[1]) == 176
    assert int(previous.iloc[2]) == 177
    extra = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    _row(
                        match_id=4,
                        player_id=14,
                        hero_id=1,
                        position=1,
                        start_time=T3,
                        residual=0.99,
                        game_version_id=179,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    later = causal_previous_version_id(extra)
    np.testing.assert_array_equal(previous.to_numpy(), later.iloc[:3].to_numpy())


def test_cold_start_does_not_invent_means() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                residual=0.25,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=9,
                position=1,
                start_time=T1,
                residual=0.40,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T1,
                residual=0.10,
                game_version_id=177,
            ),
        ]
    )
    state = attach_hero_position_meta_state(frame)
    unseen = state.loc[state["hero_id"] == 9].iloc[0]
    assert int(unseen["hp_expanding_n"]) == 0
    assert np.isnan(unseen["hp_expanding_elo_residual_mean"])
    assert float(unseen["hp_expanding_hero_share_at_position"]) == pytest.approx(0.0)
    first = state.loc[state["match_id"] == 1].iloc[0]
    assert int(first["hp_expanding_n"]) == 0
    assert np.isnan(first["hp_expanding_elo_residual_mean"])
    new_version = state.loc[state["match_id"] == 3].iloc[0]
    assert int(new_version["hp_expanding_n"]) == 1
    assert int(new_version["hp_current_version_n"]) == 0
    assert np.isnan(new_version["hp_current_version_elo_residual_mean"])


def test_non_explicit_position_is_not_a_contributor() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_id": 1,
                "player_id": 11,
                "hero_id": 1,
                "position_number": np.nan,
                "position": None,
                "start_time": T0,
                ELO_RESIDUAL_COLUMN: 0.99,
                CAUSAL_B_COLUMN: 3.0,
                CAUSAL_C_COLUMN: 0.4,
                "game_version_id": 176,
                "team_won": 1,
                "elo_expected_win": 0.5,
            },
            _row(
                match_id=2,
                player_id=12,
                hero_id=1,
                position=1,
                start_time=T1,
                residual=0.10,
            ),
        ]
    )
    state = attach_hero_position_meta_state(frame)
    t1 = state.loc[state["start_time"] == T1].iloc[0]
    assert int(t1["hp_expanding_n"]) == 0
    assert np.isnan(t1["hp_expanding_elo_residual_mean"])


def test_usage_shares_are_role_conditioned() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                residual=0.10,
            ),
            _row(
                match_id=2,
                player_id=12,
                hero_id=2,
                position=1,
                start_time=T1,
                residual=0.10,
            ),
            _row(
                match_id=3,
                player_id=13,
                hero_id=1,
                position=1,
                start_time=T2,
                residual=0.10,
            ),
        ]
    )
    state = attach_hero_position_meta_state(frame)
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    assert int(t2["hp_expanding_n"]) == 1
    assert int(t2["hp_expanding_pos_n"]) == 2
    assert float(t2["hp_expanding_hero_share_at_position"]) == pytest.approx(0.5)
    assert float(t2["hp_expanding_position_share_of_hero"]) == pytest.approx(1.0)


def test_classify_a_requires_variation_and_persistence() -> None:
    usage = {
        "grade": "A",
        "rationale": "usage persists",
        "candidate": "recent_90d",
    }
    residual = {
        "grade": "A",
        "rationale": "residual persists",
        "candidate": "recent_90d",
    }
    drift = {
        "grade": "A",
        "rationale": "requirements shift",
        "candidate": "recent_90d",
    }
    table = classify_slice24(
        usage_gate=usage,
        residual_gate=residual,
        drift_gate=drift,
        selected_recent_window="recent_90d",
    )
    assert table.iloc[0]["classification"] == "A"
    assert table.iloc[0]["frozen_components"] == (
        "usage",
        "elo_residual",
        "requirement_drift",
    )


def test_classify_c_freezes_nothing() -> None:
    gate = {"grade": "C", "rationale": "noise", "candidate": "recent_90d"}
    table = classify_slice24(
        usage_gate=gate,
        residual_gate=gate,
        drift_gate=gate,
        selected_recent_window="recent_90d",
    )
    assert table.iloc[0]["classification"] == "C"
    assert table.iloc[0]["frozen_components"] == ()
    assert bool(table.iloc[0]["fallback_hierarchy_frozen"]) is False
    assert bool(table.iloc[0]["composite_meta_score"]) is False


def test_classify_b_freezes_only_supported_families() -> None:
    table = classify_slice24(
        usage_gate={"grade": "A", "rationale": "usage", "candidate": "recent_90d"},
        residual_gate={"grade": "C", "rationale": "noise", "candidate": "recent_90d"},
        drift_gate={"grade": "C", "rationale": "stable", "candidate": "recent_90d"},
        selected_recent_window="recent_90d",
    )
    assert table.iloc[0]["classification"] == "B"
    assert table.iloc[0]["frozen_components"] == ("usage",)


def test_feature_columns_remain_thirty_three() -> None:
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
        *SLICE24_STATE_COLUMNS,
        *SLICE24_DIAGNOSTIC_COLUMNS,
        *SLICE22_STATE_COLUMNS,
        *PLAYER_X_HERO_FIT_NAMES,
        ELO_RESIDUAL_COLUMN,
        "player_hero_fit",
        "meta_strength",
    ):
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
        assert name not in SNAPSHOT_COLUMNS
        assert name not in PRE_DRAFT_SNAPSHOT_SQL
    for name in (CAUSAL_B_COLUMN, CAUSAL_C_COLUMN, COMBAT_C):
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
        report = run_hero_position_meta_diagnostics(store)
    assert report.n_development_matches == 2
    assert report.n_development_player_rows == 20
    assert report.n_holdout_excluded == 10
    assert report.integrity["holdout_used_for_window_selection"] is False
    assert report.integrity["holdout_used_for_validation"] is False
    assert report.integrity["holdout_used_for_shrinkage"] is False
    assert report.integrity["win_model_run"] is False
    assert report.integrity["elo_changed"] is False
    assert report.development_end == FROZEN_DEVELOPMENT_END
    payload = slice24_report_to_jsonable(report)
    assert payload["n_holdout_excluded"] == 10
    assert "classification" in payload
    holdout = pd.DataFrame({"start_time": [later], "hero_id": [1], "player_id": [11]})
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
        report = run_hero_position_meta_diagnostics(store)
    assert report.integrity["stratz_called"] is False
    assert report.integrity["slice21_farming_target_unchanged"] is True
    assert report.integrity["slice21_combat_target_unchanged"] is True
    assert report.integrity["farming_candidate_b_unchanged"] is True
    assert report.integrity["farming_player_k_is_5"] is True
    assert report.integrity["combat_candidate_c_unchanged"] is True
    assert report.integrity["combat_player_k_is_20"] is True
    assert report.integrity["hero_farm_k_is_2"] is True
    assert report.integrity["hero_combat_k_is_2"] is True
    assert report.integrity["player_hero_fit_created"] is False
    assert report.integrity["compatibility_score_revived"] is False
    assert report.integrity["synergy_or_counter_created"] is False
    assert report.integrity["win_model_run"] is False
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["full_development_mean_fallback"] is False
    assert report.integrity["fallback_hierarchy_frozen"] is False
    assert report.integrity["composite_meta_score"] is False
    assert report.semantics["slice22_overwritten"] is False
    assert not report.classification.empty
    assert report.tune_end <= report.development_end
    times = pd.to_datetime([T0, T1, T2], utc=True)
    tune_end = development_tune_end(pd.Series(times), development_end=T2)
    assert tune_end in {T0, T1, T2}
