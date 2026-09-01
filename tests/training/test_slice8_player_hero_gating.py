"""Tests for Slice 8 Career Player × Hero gating (exploratory).

Does not retune folds, Elo, or production FEATURE_COLUMNS. Slice 7
specs stay the named evaluation blocks they already are.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import build_snapshot_store, match_row, player_rows

from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    PLAYER_HERO_COMPARISON_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE7_META_PLAYER_HERO_SPECS,
    SLICE8_CONTEXT_COLUMNS,
    SLICE8_GATE_SPEC_NAME,
    SLICE8_INTERACTION_COLUMNS,
    SLICE8_MATCH_MEAN_CAREER_GAMES,
    SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY,
    SLICE8_META_PLAYER_HERO_SPECS,
    SLICE8_STATIC_SPECS,
    slice8_interaction_column,
)
from dota_predictor.training.slice7_meta_player_hero import (
    build_slice7_model_ready_dataset,
)
from dota_predictor.training.slice8_player_hero_gating import (
    GATE_CANDIDATE_ORDER,
    GateKind,
    add_slice8_interaction_columns,
    apply_career_gate,
    assert_slice7_slice8_identity,
    assign_train_tertile_bin,
    build_slice8_model_ready_dataset,
    career_gate_weights,
    gates_from_train,
    run_slice8_player_hero_gating_benchmark,
    select_career_gate,
    train_tertile_edges,
)
from dota_predictor.training.split import DatasetPartition
from dota_predictor.training.walk_forward import (
    WalkForwardConfig,
    resolve_walk_forward_folds,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _sequential_store(tmp_path: Path, n: int = 24):
    matches = []
    players = []
    player_counter = 1
    for i in range(n):
        match_id = 1000 + i
        start_time = T0 + timedelta(days=i)
        radiant_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        dire_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        if i >= 5:
            radiant_ids = (1, 2, 3, 4, 5)
            dire_ids = (6, 7, 8, 9, 10)
        matches.append(
            match_row(
                match_id,
                start_time=start_time,
                radiant_team_id=2 * i + 1,
                dire_team_id=2 * i + 2,
                radiant_win=(i % 2 == 0),
                game_version_id=170 + (i // 8),
            )
        )
        players.extend(
            player_rows(match_id, radiant_ids=radiant_ids, dire_ids=dire_ids)
        )
    return build_snapshot_store(tmp_path, matches=matches, players=players)


def _partition(X: pd.DataFrame, y: pd.Series | None = None) -> DatasetPartition:
    n = len(X)
    if y is None:
        y = pd.Series([i % 2 for i in range(n)], dtype="int64")
    context = pd.DataFrame(
        {
            "match_id": np.arange(n),
            "start_time": pd.date_range("2024-01-01", periods=n, tz="UTC"),
            "game_version_id": np.full(n, 176),
        }
    )
    return DatasetPartition(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        context=context,
    )


def test_slice8_not_added_to_production_or_existing_ablation() -> None:
    extra = set(SLICE8_CONTEXT_COLUMNS) | set(SLICE8_INTERACTION_COLUMNS)
    for column in extra:
        assert column not in FEATURE_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
        assert column not in PRE_DRAFT_SNAPSHOT_SQL
        for spec in POST_DRAFT_BLOCK_ABLATION_SPECS:
            assert column not in spec.feature_columns
        for spec in SLICE7_META_PLAYER_HERO_SPECS:
            assert column not in spec.feature_columns


def test_slice7_and_production_specs_are_unchanged() -> None:
    assert [spec.name for spec in POST_DRAFT_BLOCK_ABLATION_SPECS] == [
        "logistic_elo_only",
        "logistic_elo_plus_player_hero",
        "logistic_elo_plus_team_hero",
        "logistic_elo_plus_hero_meta",
        "logistic_elo_plus_player_and_team_hero",
        "logistic_elo_plus_all_three",
    ]
    assert [spec.name for spec in SLICE7_META_PLAYER_HERO_SPECS] == [
        "logistic_elo_only",
        "logistic_elo_plus_player_hero",
        "logistic_elo_plus_same_version_volume",
        "logistic_elo_plus_same_version_volume_performance",
        "logistic_elo_plus_recent20_volume",
        "logistic_elo_plus_recent20_volume_performance",
        "logistic_elo_plus_role_meta",
        "logistic_elo_plus_same_version_role",
        "logistic_elo_plus_recent20_role",
        "logistic_elo_plus_career_role",
    ]
    career = {
        spec.name: spec for spec in SLICE8_META_PLAYER_HERO_SPECS
    }["logistic_elo_plus_player_hero"]
    assert career.feature_columns == ELO_PLUS_PLAYER_HERO_COLUMNS
    assert career.feature_columns == (
        ELO_ONLY_FEATURE_COLUMNS + PLAYER_HERO_COMPARISON_COLUMNS
    )


def test_slice8_specs_exclude_win_rates_and_keep_career_block() -> None:
    by_name = {spec.name: spec for spec in SLICE8_META_PLAYER_HERO_SPECS}
    assert list(by_name) == [
        "logistic_elo_only",
        "logistic_elo_plus_player_hero",
        "logistic_elo_plus_career_evidence_interaction",
        "logistic_elo_plus_career_role_interaction",
        "logistic_elo_plus_career_patch_interaction",
        "logistic_elo_plus_career_full_gating",
        SLICE8_GATE_SPEC_NAME,
    ]
    for spec in SLICE8_META_PLAYER_HERO_SPECS:
        assert not any("win_rate" in column for column in spec.feature_columns)
        extra = set(spec.feature_columns) - set(ELO_ONLY_FEATURE_COLUMNS)
        assert extra.isdisjoint(FEATURE_COLUMNS)
        assert extra.isdisjoint(ALL_FEATURE_COLUMNS)
    assert by_name[SLICE8_GATE_SPEC_NAME].feature_columns == ELO_PLUS_PLAYER_HERO_COLUMNS
    assert SLICE8_STATIC_SPECS == SLICE8_META_PLAYER_HERO_SPECS[:-1]


def test_slice8_reuses_slice7_oos_fold_identity(tmp_path: Path) -> None:
    with _sequential_store(tmp_path, n=24) as store:
        slice7 = build_slice7_model_ready_dataset(store)
        slice8 = build_slice8_model_ready_dataset(store)
    config = WalkForwardConfig(n_blocks=3)
    assert_slice7_slice8_identity(slice7, slice8, config=config)
    assert list(slice7.dataset.context["match_id"]) == list(
        slice8.dataset.context["match_id"]
    )
    pd.testing.assert_frame_equal(
        slice7.dataset.X[list(ELO_PLUS_PLAYER_HERO_COLUMNS)].reset_index(
            drop=True
        ),
        slice8.dataset.X[list(ELO_PLUS_PLAYER_HERO_COLUMNS)].reset_index(
            drop=True
        ),
    )
    assert set(SLICE8_CONTEXT_COLUMNS).issubset(slice8.dataset.X.columns)
    assert set(SLICE8_INTERACTION_COLUMNS).issubset(slice8.dataset.X.columns)


def test_interactions_are_rowwise_products_and_preserve_null_rates() -> None:
    signal = PLAYER_HERO_COMPARISON_COLUMNS[0]
    context = SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY
    X = pd.DataFrame(
        {
            signal: [1.0, 2.0, np.nan],
            context: [0.5, np.nan, 0.8],
            "slice8_log1p_match_mean_career_games": [0.0, 1.0, 2.0],
            "slice8_log1p_match_mean_same_version_games": [0.0, 0.0, 0.0],
        }
    )
    for column in PLAYER_HERO_COMPARISON_COLUMNS[1:]:
        X[column] = 0.0
    out = add_slice8_interaction_columns(X)
    product = slice8_interaction_column(signal, context)
    assert out.loc[0, product] == pytest.approx(0.5)
    assert pd.isna(out.loc[1, product])
    assert pd.isna(out.loc[2, product])
    log_product = slice8_interaction_column(
        signal, "slice8_log1p_match_mean_career_games"
    )
    assert out.loc[0, log_product] == pytest.approx(0.0)
    assert pd.isna(out.loc[2, log_product])


def test_zero_career_count_is_real_zero_not_missing() -> None:
    values = pd.Series([0.0, 3.0])
    assert float(np.log1p(values.iloc[0])) == pytest.approx(0.0)
    assert not pd.isna(np.log1p(values.iloc[0]))


def test_train_tertiles_ignore_held_out_rows() -> None:
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    test = pd.Series([100.0, 200.0, 300.0])
    q_low, q_high = train_tertile_edges(train)
    mixed_low, mixed_high = train_tertile_edges(pd.concat([train, test]))
    assert q_low != mixed_low
    assert q_high != mixed_high
    assert assign_train_tertile_bin(1.0, q_low=q_low, q_high=q_high) == "LOW"
    assert assign_train_tertile_bin(6.0, q_low=q_low, q_high=q_high) == "HIGH"
    assert assign_train_tertile_bin(None, q_low=q_low, q_high=q_high) == "NULL"
    assert "y" not in assign_train_tertile_bin.__code__.co_varnames
    assert "radiant_win" not in train_tertile_edges.__code__.co_varnames


def test_gate_thresholds_come_from_train_quantiles_only() -> None:
    train = pd.DataFrame(
        {
            SLICE8_MATCH_MEAN_CAREER_GAMES: [1.0, 2.0, 3.0, 4.0],
            SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY: [0.2, 0.4, 0.6, 0.8],
        }
    )
    test = pd.DataFrame(
        {
            SLICE8_MATCH_MEAN_CAREER_GAMES: [50.0, 80.0],
            SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY: [0.05, 0.99],
        }
    )
    candidates = gates_from_train(train)
    assert tuple(candidate.name for candidate in candidates) == GATE_CANDIDATE_ORDER
    q50 = next(c for c in candidates if c.name == "career_above_q50")
    assert q50.career_threshold == pytest.approx(float(train.iloc[:, 0].quantile(0.50)))
    mixed = pd.concat([train, test], ignore_index=True)
    mixed_q50 = float(mixed[SLICE8_MATCH_MEAN_CAREER_GAMES].quantile(0.50))
    assert q50.career_threshold != pytest.approx(mixed_q50)
    below = next(c for c in candidates if c.name == "compat_below_q25")
    weights = career_gate_weights(
        pd.DataFrame(
            {
                SLICE8_MATCH_MEAN_CAREER_GAMES: [2.0, 2.0],
                SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY: [
                    below.compatibility_threshold - 0.01,
                    np.nan,
                ],
            }
        ),
        below,
    )
    assert weights.iloc[0] == pytest.approx(0.0)
    assert weights.iloc[1] == pytest.approx(1.0)


def test_select_career_gate_uses_validation_never_test() -> None:
    names = select_career_gate.__code__.co_varnames
    assert "train" in names
    assert "validation" in names
    assert "test" not in names
    n = 20
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            **{column: rng.normal(size=n) for column in ELO_PLUS_PLAYER_HERO_COLUMNS},
            SLICE8_MATCH_MEAN_CAREER_GAMES: np.linspace(0.0, 20.0, n),
            SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY: np.linspace(0.1, 0.9, n),
        }
    )
    y = pd.Series((rng.random(n) > 0.5).astype(int))
    train = _partition(X.iloc[:12].reset_index(drop=True), y.iloc[:12].reset_index(drop=True))
    validation = _partition(
        X.iloc[12:16].reset_index(drop=True), y.iloc[12:16].reset_index(drop=True)
    )
    test = _partition(
        X.iloc[16:].reset_index(drop=True), y.iloc[16:].reset_index(drop=True)
    )
    selected = select_career_gate(train, validation)
    mutated_test = apply_career_gate(test, selected)
    assert selected.name in GATE_CANDIDATE_ORDER
    scrambled = test.y.replace({0: 1, 1: 0})
    again = select_career_gate(train, validation)
    assert again.name == selected.name
    assert again.career_threshold == selected.career_threshold
    assert again.compatibility_threshold == selected.compatibility_threshold
    assert list(mutated_test.context["match_id"]) == list(test.context["match_id"])
    del scrambled


def test_apply_career_gate_preserves_missing_career_rates() -> None:
    signal = PLAYER_HERO_COMPARISON_COLUMNS[0]
    X = pd.DataFrame(
        {
            **{column: [1.0, np.nan] for column in PLAYER_HERO_COMPARISON_COLUMNS},
            SLICE8_MATCH_MEAN_CAREER_GAMES: [10.0, 10.0],
            SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY: [0.9, 0.9],
        }
    )
    for column in ELO_ONLY_FEATURE_COLUMNS:
        if column not in X.columns:
            X[column] = 0.0
    candidate = next(
        c for c in gates_from_train(X) if c.kind is GateKind.IDENTITY
    )
    gated = apply_career_gate(_partition(X), candidate)
    assert gated.X.loc[0, signal] == pytest.approx(1.0)
    assert pd.isna(gated.X.loc[1, signal])


def test_slice8_benchmark_is_deterministic_and_keeps_oos_identity(
    tmp_path: Path,
) -> None:
    config = WalkForwardConfig(n_blocks=3)
    with _sequential_store(tmp_path, n=24) as store:
        first = run_slice8_player_hero_gating_benchmark(store, config=config)
        second = run_slice8_player_hero_gating_benchmark(store, config=config)
        slice7 = build_slice7_model_ready_dataset(store)
    assert first.exploratory is True
    elo_ids = first.oos_predictions.loc[
        first.oos_predictions["model"] == "logistic_elo_only", "match_id"
    ]
    assert_slice7_slice8_identity(slice7, first.assembly, config=config)
    slice7_folds = resolve_walk_forward_folds(slice7.dataset, config=config)
    slice7_test_ids = [
        match_id
        for fold in slice7_folds
        for match_id in fold.test.context["match_id"].tolist()
    ]
    assert list(elo_ids) == slice7_test_ids
    pd.testing.assert_series_equal(
        first.oos_predictions["p_spec"],
        second.oos_predictions["p_spec"],
        check_names=False,
    )
    pd.testing.assert_frame_equal(first.selected_gates, second.selected_gates)
    assert set(first.selected_gates["gate_name"]).issubset(GATE_CANDIDATE_ORDER)
    assert first.n_oos == len(elo_ids)
    for spec in SLICE8_META_PLAYER_HERO_SPECS:
        n = int((first.oos_predictions["model"] == spec.name).sum())
        assert n == first.n_oos
