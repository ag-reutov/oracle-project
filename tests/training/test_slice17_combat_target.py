"""Slice 17 combat-target diagnostics: no rating, no features, no farming edits."""

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
    COMBAT_A,
    COMBAT_B,
    COMBAT_C,
    COMBAT_C_DURATION,
    COMBAT_C_POSITION,
    COMBAT_C_POSITION_DURATION,
    COMBAT_CANDIDATE_COLUMN_NAMES,
    COMBAT_D,
    FROZEN_COMBAT_CANDIDATE,
    FROZEN_FARMING_B_COLUMN,
    REQUIRED_TEAM_SIZE,
    attach_combat_candidates,
    attach_frozen_farming_b,
    complete_side_mask,
    consecutive_persistence,
    deaths_per_30,
    frozen_farming_b_values,
    hero_damage_per_min,
    hero_damage_share,
    kill_participation,
    player_variance_decomposition,
    restrict_development,
    run_combat_performance_target_diagnostics,
    team_sum,
)
from dota_predictor.training.farming_performance_target import (
    CANDIDATE_B,
    FARMING_CANDIDATE_COLUMN_NAMES,
    attach_farming_candidates,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.player_farming_state import FROZEN_SHRINKAGE_K
from dota_predictor.training.player_performance_target import (
    build_player_performance_frame,
    first_half_second_half_correlation,
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


def _one_side_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [1] * 5,
            "side": ["RADIANT"] * 5,
            "player_id": [11, 12, 13, 14, 15],
            "position_number": [1.0, 2.0, 3.0, 4.0, 5.0],
            "duration_seconds": [1800.0] * 5,
            "kills": [8.0, 4.0, 3.0, 1.0, 0.0],
            "deaths": [2.0, 3.0, 4.0, 5.0, 6.0],
            "assists": [2.0, 6.0, 8.0, 10.0, 12.0],
            "hero_damage": [10000.0, 8000.0, 6000.0, 4000.0, 2000.0],
            "num_last_hits": [300.0, 200.0, 150.0, 40.0, 20.0],
            "hero_id": [1, 2, 3, 4, 5],
            "start_time": [datetime(2026, 1, 1, tzinfo=UTC)] * 5,
            "team_won": [1, 1, 1, 1, 1],
        }
    )


def test_hero_damage_per_min_formula() -> None:
    frame = _one_side_frame()
    rates = hero_damage_per_min(frame)
    assert rates.iloc[0] == pytest.approx(10000.0 / 30.0)
    assert rates.iloc[4] == pytest.approx(2000.0 / 30.0)


def test_hero_damage_share_sums_to_one_on_complete_side() -> None:
    frame = _one_side_frame()
    share = hero_damage_share(frame)
    assert share.sum() == pytest.approx(1.0)
    assert share.iloc[0] == pytest.approx(10000.0 / 30000.0)
    assert share.iloc[4] == pytest.approx(2000.0 / 30000.0)


def test_kill_participation_formula_and_zero_team_kills() -> None:
    frame = _one_side_frame()
    kp = kill_participation(frame)
    team_kills = 8 + 4 + 3 + 1 + 0
    assert kp.iloc[0] == pytest.approx((8 + 2) / team_kills)
    assert kp.iloc[4] == pytest.approx((0 + 12) / team_kills)
    zeros = frame.copy()
    zeros["kills"] = 0.0
    empty = kill_participation(zeros)
    assert empty.isna().all()


def test_deaths_per_30_scales_with_duration() -> None:
    frame = _one_side_frame()
    rates = deaths_per_30(frame)
    assert rates.iloc[0] == pytest.approx(2.0)
    long = frame.copy()
    long["duration_seconds"] = 3600.0
    long_rates = deaths_per_30(long)
    assert long_rates.iloc[0] == pytest.approx(1.0)
    long["duration_seconds"] = 0.0
    assert deaths_per_30(long).isna().all()


def test_null_player_nulls_team_share_and_participation() -> None:
    frame = _one_side_frame()
    frame.loc[2, "hero_damage"] = np.nan
    share = hero_damage_share(frame)
    assert share.isna().all()
    frame.loc[2, "kills"] = np.nan
    kp = kill_participation(frame)
    assert kp.isna().all()
    assert not complete_side_mask(frame, "hero_damage").any()
    assert team_sum(frame, "hero_damage").isna().all()


def test_zero_is_preserved_distinct_from_null() -> None:
    frame = _one_side_frame()
    frame["hero_damage"] = [0.0, 0.0, 0.0, 0.0, 10.0]
    share = hero_damage_share(frame)
    assert share.iloc[0] == pytest.approx(0.0)
    assert share.iloc[4] == pytest.approx(1.0)
    frame["hero_damage"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    assert hero_damage_share(frame).isna().all()


def test_unknown_position_excluded_from_position_adjustment() -> None:
    frame = _one_side_frame()
    frame.loc[4, "position_number"] = np.nan
    attached = attach_combat_candidates(frame)
    assert pd.isna(attached.loc[4, COMBAT_C_POSITION])
    assert pd.isna(attached.loc[4, COMBAT_C_POSITION_DURATION])
    assert pd.notna(attached.loc[0, COMBAT_C])


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
    report_matches = matches
    config = build_feature_store_config(
        tmp_path / "store", matches=report_matches, players=players
    )
    with connect(config) as store:
        report = run_combat_performance_target_diagnostics(store)
    assert report.n_holdout_excluded == 10
    assert set(restrict_development(frame)["match_id"].unique()).isdisjoint({3})
    assert report.integrity["holdout_used_for_selection"] is False
    assert report.integrity["ti2026_used_for_target_definition"] is False


def test_split_half_is_chronological_and_deterministic() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for player_id, offset in ((11, 0.0), (12, 10.0), (13, 20.0)):
        for i in range(8):
            rows.append(
                {
                    "player_id": player_id,
                    "start_time": t0 + timedelta(days=i),
                    COMBAT_C: offset + float(i),
                }
            )
    shuffled = pd.DataFrame(rows[::-1])
    first = first_half_second_half_correlation(shuffled, COMBAT_C, min_each=3)
    second = first_half_second_half_correlation(shuffled, COMBAT_C, min_each=3)
    assert first["n_paired_players"] == 3
    assert first["pearson"] == pytest.approx(1.0)
    assert first["pearson"] == second["pearson"]
    consecutive = consecutive_persistence(pd.DataFrame(rows), COMBAT_C)
    assert consecutive["n_pairs"] == 21
    assert consecutive["pearson"] == pytest.approx(1.0)


def test_variance_decomposition_tolerates_singleton_players() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [{"player_id": 99, "start_time": t0, COMBAT_C: 0.5}]
    for player_id in range(10):
        for i in range(12):
            rows.append(
                {
                    "player_id": player_id,
                    "start_time": t0 + timedelta(days=i),
                    COMBAT_C: float(player_id) + 0.01 * i,
                }
            )
    stats = player_variance_decomposition(pd.DataFrame(rows), COMBAT_C, min_player_n=10)
    assert stats["n_players"] == 10
    assert np.isfinite(stats["icc"])


def test_combat_attach_does_not_mutate_farming_candidate_b() -> None:
    frame = _one_side_frame()
    frame["position"] = [
        "POSITION_1",
        "POSITION_2",
        "POSITION_3",
        "POSITION_4",
        "POSITION_5",
    ]
    farming = attach_farming_candidates(frame)
    before = farming[CANDIDATE_B].to_numpy(dtype=float).copy()
    combat = attach_combat_candidates(farming)
    np.testing.assert_allclose(
        combat[CANDIDATE_B].to_numpy(dtype=float), before, equal_nan=True
    )
    recomputed = frozen_farming_b_values(frame)
    np.testing.assert_allclose(
        recomputed.to_numpy(dtype=float), before, equal_nan=True, atol=1e-10
    )
    joined = attach_frozen_farming_b(farming)
    np.testing.assert_allclose(
        joined[CANDIDATE_B].to_numpy(dtype=float), before, equal_nan=True
    )
    assert FROZEN_SHRINKAGE_K == 5.0
    assert FROZEN_FARMING_B_COLUMN == CANDIDATE_B


def test_candidates_do_not_enter_feature_columns_or_pre_draft() -> None:
    for name in COMBAT_CANDIDATE_COLUMN_NAMES:
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
        assert name not in SNAPSHOT_COLUMNS
        assert name not in PRE_DRAFT_SNAPSHOT_SQL
    for name in FARMING_CANDIDATE_COLUMN_NAMES:
        assert name not in FEATURE_COLUMNS
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in PRE_DRAFT_SNAPSHOT_SQL
        pre_draft = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
        assert column not in pre_draft
    assert "hero_damage_share" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "kill_participation" not in PRE_DRAFT_SNAPSHOT_SQL
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
    assert REQUIRED_TEAM_SIZE == 5
    assert FROZEN_SHRINKAGE_K == 5.0
    assert FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION
    assert FROZEN_COMBAT_CANDIDATE not in FEATURE_COLUMNS


def test_full_diagnostic_run_excludes_holdout_and_box_scores_from_view(
    tmp_path: Path,
) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    later = FROZEN_DEVELOPMENT_END + timedelta(days=1)
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
        report = run_combat_performance_target_diagnostics(store)
    assert report.n_development_matches == 2
    assert report.n_development_player_rows == 20
    assert report.n_holdout_excluded == 10
    assert report.integrity["stratz_called"] is False
    assert report.integrity["model_trained"] is False
    assert report.integrity["historical_state_built"] is False
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["frozen_k_is_5"] is True
    assert report.integrity["box_scores_in_feature_match_players_view"] is False
    assert COMBAT_A in report.candidate_comparison["candidate"].tolist()
    assert COMBAT_B in report.candidate_comparison["candidate"].tolist()
    assert COMBAT_C in report.candidate_comparison["candidate"].tolist()
    assert COMBAT_C_POSITION in report.candidate_comparison["candidate"].tolist()
    assert COMBAT_C_DURATION in report.candidate_comparison["candidate"].tolist()
    assert (
        COMBAT_C_POSITION_DURATION in report.candidate_comparison["candidate"].tolist()
    )
    assert COMBAT_D in report.candidate_comparison["candidate"].tolist()
    assert not report.classification.empty
    assert report.integrity["farming_code_modified"] is False
    share = attach_combat_candidates(_one_side_frame())
    assert share[COMBAT_C].sum() == pytest.approx(1.0)
