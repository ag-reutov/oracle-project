"""Tests for post-draft Elo vs Elo + draft-comparison assembly.

Does not change PRE_DRAFT `FEATURE_COLUMNS`. Uses the full comparison
metric set (no correlation-based subset). `slot_in_side` is lobby order
only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from training_helpers import (
    build_snapshot_store,
    match_row,
    player_rows,
)

from dota_predictor.features.draft_comparison import DRAFT_COMPARISON_METRIC_COLUMNS
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
)
from dota_predictor.training.evaluation import REGULARIZATION_CANDIDATES
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_DRAFT_COMPARISON_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
)
from dota_predictor.training.post_draft import (
    build_post_draft_model_ready_dataset,
    run_post_draft_benchmark,
    run_post_draft_block_ablation,
)
from dota_predictor.training.split import SplitBoundaries, chronological_split

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _synthetic_store(tmp_path: Path, n: int = 40):
    matches = []
    players = []
    player_counter = 1
    for i in range(n):
        match_id = 1000 + i
        start_time = T0 + timedelta(days=i)
        r_team = 2 * i + 1
        d_team = 2 * i + 2
        radiant_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        dire_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        matches.append(
            match_row(
                match_id,
                start_time=start_time,
                radiant_team_id=r_team,
                dire_team_id=d_team,
                radiant_win=(i % 2 == 0),
            )
        )
        players.extend(
            player_rows(match_id, radiant_ids=radiant_ids, dire_ids=dire_ids)
        )
    return build_snapshot_store(tmp_path, matches=matches, players=players)


@pytest.fixture
def post_draft_split(tmp_path: Path):
    with _synthetic_store(tmp_path, n=40) as store:
        dataset = build_post_draft_model_ready_dataset(store)
    boundaries = SplitBoundaries(
        train_end=T0 + timedelta(days=27),
        validation_end=T0 + timedelta(days=33),
    )
    return chronological_split(dataset, boundaries=boundaries)


def test_post_draft_x_is_elo_plus_every_draft_diff(tmp_path: Path) -> None:
    with _synthetic_store(tmp_path, n=12) as store:
        dataset = build_post_draft_model_ready_dataset(store)
    assert list(dataset.X.columns) == list(ELO_PLUS_DRAFT_COMPARISON_COLUMNS)
    assert list(dataset.X.columns[-len(DRAFT_COMPARISON_METRIC_COLUMNS) :]) == list(
        DRAFT_COMPARISON_METRIC_COLUMNS
    )
    assert TARGET_COLUMN not in dataset.X.columns
    assert "radiant_win" not in dataset.X.columns
    assert list(dataset.context.columns) == list(IDENTITY_COLUMNS)
    assert dataset.y.name == TARGET_COLUMN
    assert len(dataset) == 12
    assert dataset.X["radiant_team_elo"].notna().all()


def test_post_draft_does_not_change_pre_draft_feature_contract() -> None:
    assert ALL_FEATURE_COLUMNS == FEATURE_COLUMNS
    assert set(DRAFT_COMPARISON_METRIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(ELO_ONLY_FEATURE_COLUMNS).issubset(FEATURE_COLUMNS)


def test_post_draft_split_is_chronological(post_draft_split) -> None:
    train_times = post_draft_split.train.context["start_time"]
    val_times = post_draft_split.validation.context["start_time"]
    test_times = post_draft_split.test.context["start_time"]
    assert train_times.max() <= post_draft_split.boundaries.train_end
    assert val_times.min() > post_draft_split.boundaries.train_end
    assert val_times.max() <= post_draft_split.boundaries.validation_end
    assert test_times.min() > post_draft_split.boundaries.validation_end
    assert train_times.is_monotonic_increasing
    assert val_times.is_monotonic_increasing
    assert test_times.is_monotonic_increasing


def test_post_draft_benchmark_uses_full_draft_set_and_selects_c_on_validation(
    post_draft_split,
) -> None:
    report = run_post_draft_benchmark(
        post_draft_split, include_test_evaluation=True
    )
    assert report.n_draft_comparison_features == len(DRAFT_COMPARISON_METRIC_COLUMNS)
    assert set(report.regularization_comparison["C"]) <= set(REGULARIZATION_CANDIDATES)
    assert report.elo_logistic_C in REGULARIZATION_CANDIDATES
    assert report.elo_plus_draft_C in REGULARIZATION_CANDIDATES
    assert set(report.validation_evaluations) == {
        "constant_0.5",
        "empirical_train_rate",
        "elo_only",
        "logistic_elo_only",
        "logistic_elo_plus_draft_comparison",
    }
    assert set(report.test_evaluations) == set(report.validation_evaluations)
    elo_cols = set(ELO_ONLY_FEATURE_COLUMNS)
    draft_cols = set(DRAFT_COMPARISON_METRIC_COLUMNS)
    used = set(report.coefficients["feature"])
    assert elo_cols.issubset(
        {name.removesuffix("__was_missing") for name in used}
    )
    assert draft_cols.issubset(
        {name.removesuffix("__was_missing") for name in used}
    )
    assert TARGET_COLUMN not in used
    assert "radiant_win" not in used


def test_post_draft_benchmark_is_deterministic(post_draft_split) -> None:
    first = run_post_draft_benchmark(
        post_draft_split, include_test_evaluation=False
    )
    second = run_post_draft_benchmark(
        post_draft_split, include_test_evaluation=False
    )
    assert first.elo_logistic_C == second.elo_logistic_C
    assert first.elo_plus_draft_C == second.elo_plus_draft_C
    pd.testing.assert_frame_equal(
        first.regularization_comparison, second.regularization_comparison
    )
    assert (
        first.validation_evaluations[
            "logistic_elo_plus_draft_comparison"
        ].metrics.log_loss
        == second.validation_evaluations[
            "logistic_elo_plus_draft_comparison"
        ].metrics.log_loss
    )


def test_block_ablation_covers_predefined_specs_on_the_same_split(
    post_draft_split,
) -> None:
    report = run_post_draft_block_ablation(
        post_draft_split, include_test_evaluation=True
    )
    expected_names = [spec.name for spec in POST_DRAFT_BLOCK_ABLATION_SPECS]
    assert [spec.name for spec in report.specs] == expected_names
    assert list(report.validation_evaluations) == expected_names
    assert list(report.test_evaluations) == expected_names
    assert list(report.selected_C) == expected_names
    assert set(report.selected_C.values()) <= set(REGULARIZATION_CANDIDATES)
    assert set(report.regularization_comparison["model"]) == set(expected_names)
    assert report.n_features["logistic_elo_only"] == len(ELO_ONLY_FEATURE_COLUMNS)
    assert report.n_features["logistic_elo_plus_all_three"] == len(
        ELO_PLUS_DRAFT_COMPARISON_COLUMNS
    )
    for spec in POST_DRAFT_BLOCK_ABLATION_SPECS:
        assert set(spec.feature_columns).issubset(post_draft_split.train.X.columns)


def test_block_ablation_is_deterministic(post_draft_split) -> None:
    first = run_post_draft_block_ablation(
        post_draft_split, include_test_evaluation=False
    )
    second = run_post_draft_block_ablation(
        post_draft_split, include_test_evaluation=False
    )
    assert first.selected_C == second.selected_C
    pd.testing.assert_frame_equal(
        first.regularization_comparison, second.regularization_comparison
    )
    for name in first.validation_evaluations:
        assert (
            first.validation_evaluations[name].metrics.log_loss
            == second.validation_evaluations[name].metrics.log_loss
        )
