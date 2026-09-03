"""Slice 20 walk-forward combat vs Elo: frozen spec, leakage, holdout."""

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
from dota_predictor.features.player_combat_comparison import (
    COMBAT_CAUSAL_C_COLUMN,
    COMBAT_ROSTER_SIDE_SIZE,
    PLAYER_COMBAT_FEATURE_COLUMNS,
    PLAYER_COMBAT_REQUIRED_COLUMNS,
    player_combat_comparison_from_players,
    player_combat_side_profile,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    build_pre_draft_snapshot,
)
from dota_predictor.features.team_elo import (
    TEAM_ELO_DELTA_COLUMN,
    TEAM_ELO_FEATURE_COLUMNS,
)
from dota_predictor.training.combat_performance_target import (
    COMBAT_C_POSITION,
    FROZEN_COMBAT_CANDIDATE,
)
from dota_predictor.training.dataset import build_model_ready_dataset
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_PLAYER_COMBAT_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
    SLICE15_CANDIDATE_SPEC,
    SLICE19_CANDIDATE_SPEC,
    SLICE19_FROZEN_SPECS,
)
from dota_predictor.training.player_combat_benchmark import (
    CLASSIFICATION_A,
    CLASSIFICATION_B,
    CLASSIFICATION_C,
    COMBAT_FEATURE_COLUMN,
    FARMING_FEATURE_COLUMN,
    HOLDOUT_POLICY,
    SLICE20_CANDIDATE_SPEC,
    SLICE20_CANDIDATE_SPEC_NAME,
    SLICE20_FROZEN_SPECS,
    SLICE20_REFERENCE_SPEC,
    SLICE20_REFERENCE_SPEC_NAME,
    assign_abs_quantile_bucket,
    assign_signed_combat_bucket,
    build_slice20_model_ready_dataset,
    classify_slice20,
    run_slice20_player_combat_benchmark,
)
from dota_predictor.training.player_combat_state import FROZEN_COMBAT_SHRINKAGE_K
from dota_predictor.training.player_farming_state import (
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END
from dota_predictor.training.walk_forward import (
    ELO_BLOCK_SPEC_NAME,
    WalkForwardConfig,
    resolve_walk_forward_folds,
)

RADIANT_IDS = (11, 12, 13, 14, 15)
DIRE_IDS = (21, 22, 23, 24, 25)
POSITIONS = (
    "POSITION_1",
    "POSITION_2",
    "POSITION_3",
    "POSITION_4",
    "POSITION_5",
)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _fill_box_scores(
    players: list[dict[str, object]],
    *,
    hero_damage: tuple[int, ...] | None = None,
    last_hits: tuple[int, ...] | None = None,
    mutate_post_match: bool = False,
) -> list[dict[str, object]]:
    for item in players:
        slot = int(item["slot_in_side"])
        damage = hero_damage[slot] if hero_damage is not None else 20000 - slot * 3000
        hits = last_hits[slot] if last_hits is not None else 300 - slot * 40
        item["position"] = (
            POSITIONS[(slot + 1) % 5] if mutate_post_match else POSITIONS[slot]
        )
        item["hero_id"] = 99 if mutate_post_match else item["hero_id"]
        item["num_last_hits"] = int(hits)
        item["kills"] = 99 if mutate_post_match else 1
        item["deaths"] = 99 if mutate_post_match else 1
        item["assists"] = 99 if mutate_post_match else 1
        item["gold_per_minute"] = 400
        item["experience_per_minute"] = 400
        item["num_denies"] = 0
        item["networth"] = 10000
        item["hero_damage"] = int(damage)
        item["tower_damage"] = 1000
        item["hero_healing"] = 0
        item["level"] = 20
    return players


def _radiant_damage(
    match_index: int, *, bump_match: int | None = None, bump: int = 0
) -> tuple[int, ...]:
    base = (22000 + match_index * 80, 16000, 11000, 6000, 2500)
    if bump_match is not None and match_index == bump_match:
        return (base[0] + bump, *base[1:])
    return base


def _dire_damage(match_index: int) -> tuple[int, ...]:
    return (18000 + match_index * 30, 14000, 10000, 5500, 2200)


def _radiant_hits(match_index: int) -> tuple[int, ...]:
    return (420 + match_index * 8, 280, 190, 90, 40)


def _dire_hits(match_index: int) -> tuple[int, ...]:
    return (300 + match_index * 3, 240, 170, 80, 35)


def _sequential_fixture(
    _tmp_path: Path,
    n: int,
    *,
    holdout_n: int = 0,
    bump_match: int | None = None,
    bump: int = 0,
    mutate_post_match: bool = False,
    extra_same_time_after: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matches: list[dict[str, object]] = []
    players: list[dict[str, object]] = []
    total = n + holdout_n
    extra_added = False
    for i in range(total):
        match_id = 1000 + i
        if i < n:
            start_time = T0 + timedelta(days=i)
        else:
            start_time = FROZEN_DEVELOPMENT_END + timedelta(days=1 + (i - n))
        duration = 1100 + i * 70
        if mutate_post_match and bump_match is not None and i == bump_match:
            duration = duration + 4000
        row = match_row(
            match_id,
            start_time=start_time,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=(i % 2 == 0),
        )
        if mutate_post_match and bump_match is not None and i == bump_match:
            row["radiant_win"] = not bool(row["radiant_win"])
        row["duration_seconds"] = duration
        matches.append(row)
        batch = player_rows(match_id, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        radiant = _radiant_damage(i, bump_match=bump_match, bump=bump)
        dire = _dire_damage(i)
        mutate = bool(mutate_post_match and bump_match is not None and i == bump_match)
        for item in batch:
            if item["side"] == "RADIANT":
                _fill_box_scores(
                    [item],
                    hero_damage=radiant,
                    last_hits=_radiant_hits(i),
                    mutate_post_match=mutate,
                )
            else:
                _fill_box_scores(
                    [item],
                    hero_damage=dire,
                    last_hits=_dire_hits(i),
                    mutate_post_match=mutate,
                )
        players.extend(batch)
        if (
            extra_same_time_after is not None
            and i == extra_same_time_after
            and not extra_added
        ):
            extra_id = 9000
            extra = match_row(
                extra_id,
                start_time=start_time,
                radiant_team_id=300,
                dire_team_id=400,
                radiant_win=False,
            )
            extra["duration_seconds"] = duration + 400
            matches.append(extra)
            extra_players = player_rows(
                extra_id,
                radiant_ids=(31, 32, 33, 34, 35),
                dire_ids=(41, 42, 43, 44, 45),
            )
            _fill_box_scores(
                extra_players[:5],
                hero_damage=(90000, 80000, 70000, 60000, 50000),
                last_hits=(900, 800, 700, 600, 500),
            )
            _fill_box_scores(
                extra_players[5:],
                hero_damage=(500, 400, 300, 200, 100),
                last_hits=(50, 40, 30, 20, 10),
            )
            players.extend(extra_players)
            extra_added = True
    return matches, players


def _assembly(tmp_path: Path, n: int = 18, **kwargs: object):
    matches, players = _sequential_fixture(tmp_path, n, **kwargs)
    return build_snapshot_store(tmp_path, matches=matches, players=players)


def test_reference_is_elo_only_and_candidate_is_elo_plus_combat() -> None:
    assert SLICE20_REFERENCE_SPEC_NAME == "logistic_elo_only"
    assert SLICE20_REFERENCE_SPEC_NAME == ELO_BLOCK_SPEC_NAME
    assert SLICE20_CANDIDATE_SPEC_NAME == "logistic_elo_plus_player_combat"
    assert SLICE20_REFERENCE_SPEC.feature_columns == ELO_ONLY_FEATURE_COLUMNS
    assert SLICE20_REFERENCE_SPEC.feature_columns == TEAM_ELO_FEATURE_COLUMNS
    assert SLICE20_CANDIDATE_SPEC.feature_columns == ELO_PLUS_PLAYER_COMBAT_COLUMNS
    assert SLICE20_CANDIDATE_SPEC.feature_columns == (
        ELO_ONLY_FEATURE_COLUMNS + PLAYER_COMBAT_FEATURE_COLUMNS
    )
    assert SLICE20_CANDIDATE_SPEC.feature_columns == (
        TEAM_ELO_FEATURE_COLUMNS + ("mean_combat_shrunk_c_diff",)
    )
    assert SLICE20_FROZEN_SPECS is SLICE19_FROZEN_SPECS
    assert SLICE20_CANDIDATE_SPEC is SLICE19_CANDIDATE_SPEC
    extra = set(SLICE20_CANDIDATE_SPEC.feature_columns) - set(
        SLICE20_REFERENCE_SPEC.feature_columns
    )
    assert extra == {"mean_combat_shrunk_c_diff"}
    assert FARMING_FEATURE_COLUMN not in SLICE20_CANDIDATE_SPEC.feature_columns
    assert extra.isdisjoint(SLICE15_CANDIDATE_SPEC.feature_columns)


def test_no_prior_n_hero_position_farming_or_current_match_inputs_in_model() -> None:
    forbidden = (
        "combat_prior_n",
        "radiant_combat_prior_n_sum",
        "dire_combat_cold_start_count",
        "radiant_mean_combat_shrunk_c",
        FARMING_FEATURE_COLUMN,
        "hero_id",
        "position",
        "position_number",
        "hero_damage",
        "kills",
        "assists",
        "deaths",
        "num_last_hits",
        "duration_seconds",
        COMBAT_CAUSAL_C_COLUMN,
        "radiant_win",
    )
    for name in forbidden:
        assert name not in SLICE20_CANDIDATE_SPEC.feature_columns
        assert name not in SLICE20_REFERENCE_SPEC.feature_columns
    assert "hero_id" not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert "position" not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert "hero_damage" not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert "kills" not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert "assists" not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert "deaths" not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert "duration_seconds" not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert COMBAT_CAUSAL_C_COLUMN not in PLAYER_COMBAT_REQUIRED_COLUMNS
    assert FROZEN_COMBAT_SHRINKAGE_K == 20.0
    assert FROZEN_SHRINKAGE_K == 5.0
    assert FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION
    assert FROZEN_COMBAT_CANDIDATE == "hero_damage_share_position_adj"
    assert FROZEN_CANDIDATE_B == CANDIDATE_B


def test_feature_columns_and_slice9_remain_unchanged() -> None:
    assert len(FEATURE_COLUMNS) == 33
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)
    assert COMBAT_FEATURE_COLUMN not in FEATURE_COLUMNS
    assert FARMING_FEATURE_COLUMN not in FEATURE_COLUMNS
    assert tuple(spec.name for spec in SLICE9_FROZEN_SPECS) == (
        SLICE9_REFERENCE_SPEC_NAME,
        SLICE9_CANDIDATE_SPEC_NAME,
    )
    assert SLICE9_CANDIDATE_SPEC_NAME == "logistic_elo_plus_player_hero"
    assert COMBAT_FEATURE_COLUMN not in SLICE9_CANDIDATE_SPEC.feature_columns
    assert [spec.name for spec in POST_DRAFT_BLOCK_ABLATION_SPECS] == [
        "logistic_elo_only",
        "logistic_elo_plus_player_hero",
        "logistic_elo_plus_team_hero",
        "logistic_elo_plus_hero_meta",
        "logistic_elo_plus_player_and_team_hero",
        "logistic_elo_plus_all_three",
    ]
    assert "development_oos_only" in HOLDOUT_POLICY


def test_team_comparison_remains_five_player_arithmetic_mean() -> None:
    assert COMBAT_ROSTER_SIDE_SIZE == 5
    frame = pd.DataFrame(
        {
            "match_id": [1] * 10,
            "player_id": list(RADIANT_IDS) + list(DIRE_IDS),
            "start_time": [T0] * 10,
            "game_version_id": [176] * 10,
            "team_id": [100] * 5 + [200] * 5,
            "side": ["RADIANT"] * 5 + ["DIRE"] * 5,
            "slot_in_side": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            "combat_causal_c": [1.0, 0.5, 0.0, -0.5, -1.0] + [0.0] * 5,
            "combat_prior_n": [5] * 5 + [0] * 5,
            "combat_prior_mean_c": [1.0, 0.5, 0.0, -0.5, -1.0] + [None] * 5,
            "combat_shrinkage_weight": [0.5] * 5 + [0.0] * 5,
            "combat_shrunk_c": [0.5, 0.25, 0.0, -0.25, -0.5] + [0.0] * 5,
        }
    )
    side = player_combat_side_profile(frame)
    radiant = side.loc[side["side"] == "RADIANT"].iloc[0]
    assert radiant["mean_combat_shrunk_c"] == pytest.approx(0.0)
    comparison = player_combat_comparison_from_players(frame)
    assert comparison.iloc[0]["mean_combat_shrunk_c_diff"] == pytest.approx(0.0)


def test_assembly_uses_only_pre_draft_combat_and_excludes_holdout(
    tmp_path: Path,
) -> None:
    with _assembly(tmp_path, n=12, holdout_n=2) as store:
        assembly = build_slice20_model_ready_dataset(store)
        view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    assert list(assembly.dataset.X.columns) == list(ELO_PLUS_PLAYER_COMBAT_COLUMNS)
    assert COMBAT_FEATURE_COLUMN in assembly.dataset.X.columns
    assert FARMING_FEATURE_COLUMN not in assembly.dataset.X.columns
    assert "combat_prior_n" not in assembly.dataset.X.columns
    assert "hero_id" not in assembly.dataset.X.columns
    assert "position" not in assembly.dataset.X.columns
    assert "hero_damage" not in assembly.dataset.X.columns
    assert "kills" not in assembly.dataset.X.columns
    assert "assists" not in assembly.dataset.X.columns
    assert "deaths" not in assembly.dataset.X.columns
    assert "duration_seconds" not in assembly.dataset.X.columns
    assert COMBAT_CAUSAL_C_COLUMN not in assembly.dataset.X.columns
    assert assembly.dataset.y.name == "radiant_win"
    assert assembly.n_holdout_excluded == 2
    assert len(assembly.dataset) == 12
    latest = assembly.dataset.context["start_time"].max()
    assert latest <= FROZEN_DEVELOPMENT_END
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in view_columns
    assert FARMING_FEATURE_COLUMN in assembly.farming_by_match.columns


def test_current_match_combat_evidence_does_not_change_that_match_feature(
    tmp_path: Path,
) -> None:
    with _assembly(tmp_path, n=12) as store:
        original = build_slice20_model_ready_dataset(store).dataset
    with _assembly(
        tmp_path,
        n=12,
        bump_match=7,
        bump=500_000,
        mutate_post_match=True,
    ) as store:
        mutated = build_slice20_model_ready_dataset(store).dataset
    left = original.context["match_id"].to_numpy()
    right = mutated.context["match_id"].to_numpy()
    assert np.array_equal(left, right)
    orig_f = original.X[COMBAT_FEATURE_COLUMN].to_numpy(dtype=float)
    mut_f = mutated.X[COMBAT_FEATURE_COLUMN].to_numpy(dtype=float)
    target = original.context["match_id"] == 1007
    assert orig_f[target.to_numpy()] == pytest.approx(mut_f[target.to_numpy()])
    # Accumulation into later matches is covered by Slice 19; this slice
    # only needs current-match PRE_DRAFT invariance on the assembled row.


def test_future_match_cannot_change_earlier_feature_row(tmp_path: Path) -> None:
    with _assembly(tmp_path, n=8) as store:
        original = build_slice20_model_ready_dataset(store).dataset
    with _assembly(tmp_path, n=9) as store:
        extended = build_slice20_model_ready_dataset(store).dataset
    orig_ids = set(original.context["match_id"])
    shared = extended.context["match_id"].isin(orig_ids)
    left = (
        original.X[COMBAT_FEATURE_COLUMN].reset_index(drop=True).to_numpy(dtype=float)
    )
    right = (
        extended.X.loc[shared, COMBAT_FEATURE_COLUMN]
        .reset_index(drop=True)
        .to_numpy(dtype=float)
    )
    assert left == pytest.approx(right)


def test_same_timestamp_extra_match_does_not_contaminate_existing_row(
    tmp_path: Path,
) -> None:
    with _assembly(tmp_path, n=8) as store:
        original = build_slice20_model_ready_dataset(store).dataset
    with _assembly(tmp_path, n=8, extra_same_time_after=5) as store:
        with_extra = build_slice20_model_ready_dataset(store).dataset
    target_id = 1005
    orig = float(
        original.X.loc[
            original.context["match_id"] == target_id, COMBAT_FEATURE_COLUMN
        ].iloc[0]
    )
    rerun = float(
        with_extra.X.loc[
            with_extra.context["match_id"] == target_id, COMBAT_FEATURE_COLUMN
        ].iloc[0]
    )
    assert orig == pytest.approx(rerun)


def test_walk_forward_reuses_existing_fold_boundaries_and_same_oos_ids(
    tmp_path: Path,
) -> None:
    matches, players = _sequential_fixture(tmp_path, 18)
    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        assembly = build_slice20_model_ready_dataset(store)
        pre_draft = build_model_ready_dataset(build_pre_draft_snapshot(store))
        report = run_slice20_player_combat_benchmark(
            store, config=WalkForwardConfig(n_blocks=3)
        )
    pre_folds = resolve_walk_forward_folds(
        pre_draft, config=WalkForwardConfig(n_blocks=3)
    )
    combat_folds = resolve_walk_forward_folds(
        assembly.dataset, config=WalkForwardConfig(n_blocks=3)
    )
    assert len(pre_folds) == len(combat_folds)
    for left, right in zip(pre_folds, combat_folds, strict=True):
        assert list(left.test.context["match_id"]) == list(
            right.test.context["match_id"]
        )
        assert left.train_end == right.train_end
        assert left.validation_end == right.validation_end
        assert left.test_end == right.test_end
    oos = report.walk_forward.oos_predictions
    ref_ids = list(oos.loc[oos["model"] == SLICE20_REFERENCE_SPEC_NAME, "match_id"])
    cand_ids = list(oos.loc[oos["model"] == SLICE20_CANDIDATE_SPEC_NAME, "match_id"])
    assert ref_ids == cand_ids
    assert report.n_oos == len(ref_ids)
    assert report.frozen_combat_k == 20.0
    assert report.frozen_farming_k == 5.0
    assert report.integrity["candidate_excludes_prior_n"] is True
    assert report.integrity["farming_absent_from_x"] is True
    assert report.integrity["farming_in_candidate"] is False
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["identical_oos_match_ids"] is True
    assert report.integrity["k_re_searched"] is False
    assert report.integrity["alternative_combat_features_searched"] is False
    assert report.integrity["holdout_scored"] is False
    assert report.integrity["stratz_called"] is False
    assert report.integrity["combat_k_is_20"] is True
    assert report.integrity["farming_k_is_5"] is True
    assert report.classification in {
        CLASSIFICATION_A,
        CLASSIFICATION_B,
        CLASSIFICATION_C,
    }
    assert TEAM_ELO_DELTA_COLUMN in assembly.dataset.X.columns
    assert FARMING_FEATURE_COLUMN not in assembly.dataset.X.columns
    assert report.slice16_comparison["models_combined"] is False
    assert report.slice16_comparison["promotion_contest"] is False


def test_holdout_matches_are_not_scored(tmp_path: Path) -> None:
    with _assembly(tmp_path, n=18, holdout_n=3) as store:
        report = run_slice20_player_combat_benchmark(
            store, config=WalkForwardConfig(n_blocks=3)
        )
    oos_ids = set(report.walk_forward.oos_predictions["match_id"])
    assert report.n_holdout_excluded == 3
    assert report.n_development_matches == 18
    assert max(oos_ids) < 1000 + 18
    assert report.integrity["holdout_used_for_c"] is False
    assert report.integrity["holdout_scored"] is False


def test_abs_quantile_buckets_are_not_performance_selected() -> None:
    edges = np.array([0.0, 0.1, 0.4, 1.0], dtype=float)
    assert assign_abs_quantile_bucket(0.0, edges) == "Q1"
    assert assign_abs_quantile_bucket(0.2, edges) == "Q2"
    assert assign_abs_quantile_bucket(0.9, edges) == "Q3"
    assert assign_abs_quantile_bucket(float("nan"), edges) == "NULL"


def test_signed_combat_buckets_are_fixed_sign_splits() -> None:
    assert assign_signed_combat_bucket(-0.02) == "combat_diff_lt_0"
    assert assign_signed_combat_bucket(0.0) == "combat_diff_eq_0"
    assert assign_signed_combat_bucket(0.02) == "combat_diff_gt_0"
    assert assign_signed_combat_bucket(float("nan")) == "NULL"


def test_classify_slice20_rules() -> None:
    a_label, a_why = classify_slice20(
        pooled_delta=-0.004,
        ci_low=-0.006,
        ci_high=-0.001,
        frac_delta_negative=0.99,
        n_folds_delta_negative=4,
        n_folds_delta_positive=0,
        n_folds=4,
        coefficient_sign_stable=True,
        reference_brier=0.22,
        candidate_brier=0.219,
        reference_ece=0.03,
        candidate_ece=0.03,
        mean_abs_prediction_delta=0.02,
    )
    assert a_label == CLASSIFICATION_A
    assert "negative" in a_why
    b_label, _ = classify_slice20(
        pooled_delta=-0.0004,
        ci_low=-0.002,
        ci_high=0.001,
        frac_delta_negative=0.7,
        n_folds_delta_negative=2,
        n_folds_delta_positive=2,
        n_folds=4,
        coefficient_sign_stable=False,
        reference_brier=0.22,
        candidate_brier=0.221,
        reference_ece=0.03,
        candidate_ece=0.04,
        mean_abs_prediction_delta=0.004,
    )
    assert b_label == CLASSIFICATION_B
    c_label, _ = classify_slice20(
        pooled_delta=0.003,
        ci_low=0.001,
        ci_high=0.005,
        frac_delta_negative=0.02,
        n_folds_delta_negative=0,
        n_folds_delta_positive=4,
        n_folds=4,
        coefficient_sign_stable=True,
        reference_brier=0.22,
        candidate_brier=0.23,
        reference_ece=0.03,
        candidate_ece=0.06,
        mean_abs_prediction_delta=0.001,
    )
    assert c_label == CLASSIFICATION_C
