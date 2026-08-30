"""Tests for Step 4B evaluation orchestration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import build_dataset, match_row, player_rows

from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training import (
    chronological_split,
    run_step4b_benchmark,
)
from dota_predictor.training.baselines import ConstantProbabilityBaseline
from dota_predictor.training.evaluation import (
    REGULARIZATION_CANDIDATES,
    evaluate_predictor,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    HISTORICAL_WITHOUT_ELO_COLUMNS,
)
from dota_predictor.training.logistic_model import (
    LogisticRegressionConfig,
    LogisticRegressionPredictor,
)
from dota_predictor.training.preprocessing import build_preprocessing_pipeline
from dota_predictor.training.split import SplitBoundaries

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _synthetic_dataset(tmp_path: Path, n: int = 40):
    """Build a real-pipeline dataset large enough for train/val/test."""
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
    return build_dataset(tmp_path, matches=matches, players=players)


@pytest.fixture
def split(tmp_path):
    dataset = _synthetic_dataset(tmp_path, n=40)
    boundaries = SplitBoundaries(
        train_end=T0 + timedelta(days=27),
        validation_end=T0 + timedelta(days=33),
    )
    return chronological_split(dataset, boundaries=boundaries)


def test_predictions_remain_row_aligned_with_context(split) -> None:
    model = ConstantProbabilityBaseline(probability=0.5)
    model.fit(split.train.X, split.train.y)
    evaluation = evaluate_predictor("constant", split.validation, model)
    assert len(evaluation.predictions.context) == len(evaluation.predictions.y_true)
    assert len(evaluation.predictions.p_radiant_win) == len(
        evaluation.predictions.context
    )
    assert "match_id" in evaluation.predictions.context.columns
    assert "start_time" in evaluation.predictions.context.columns


def test_preprocessing_pipeline_is_not_refit_on_validation(split) -> None:
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(split.train.X[list(FEATURE_COLUMNS[:6])])
    train_medians = pipeline.named_steps["missingness_impute"].medians_.copy()
    pipeline.transform(split.validation.X[list(FEATURE_COLUMNS[:6])])
    assert pipeline.named_steps["missingness_impute"].medians_.equals(train_medians)


def test_run_step4b_benchmark_produces_validation_and_optional_test(split) -> None:
    report = run_step4b_benchmark(split, include_test_evaluation=True)
    assert "constant_0.5" in report.validation_evaluations
    assert "elo_only" in report.validation_evaluations
    assert "logistic_regression_all_features" in report.validation_evaluations
    assert len(report.ablation_validation) == 4
    assert not report.coefficients.empty
    assert report.test_evaluations


def test_run_step4b_benchmark_can_skip_test_evaluation(split) -> None:
    report = run_step4b_benchmark(split, include_test_evaluation=False)
    assert report.validation_evaluations
    assert not report.test_evaluations


def test_logistic_model_target_and_identity_not_in_feature_columns(split) -> None:
    assert "match_id" not in split.train.X.columns
    assert "radiant_win" not in split.train.X.columns
    model = LogisticRegressionPredictor(feature_columns=FEATURE_COLUMNS)
    model.fit(split.train.X, split.train.y)
    p = model.predict_radiant_win_proba(split.validation.X)
    assert len(p) == len(split.validation.y)


def test_regularization_selection_uses_only_validation_candidates(split) -> None:
    report = run_step4b_benchmark(split, include_test_evaluation=False)
    comparison = report.regularization_comparison

    assert set(comparison["C"]) == set(REGULARIZATION_CANDIDATES)
    assert report.selected_regularization_C in REGULARIZATION_CANDIDATES
    assert report.selected_regularization_C == comparison.loc[
        comparison["validation_log_loss"].idxmin(), "C"
    ]


def test_ablation_validation_covers_required_model_families(split) -> None:
    report = run_step4b_benchmark(split, include_test_evaluation=False)

    assert set(report.ablation_validation) == {
        "elo_only",
        "logistic_elo_only",
        "logistic_historical_without_elo",
        "logistic_all_features",
    }
    assert (
        report.ablation_validation["logistic_all_features"].metrics.log_loss
        == report.validation_evaluations[
            "logistic_regression_all_features"
        ].metrics.log_loss
    )


def test_selected_regularization_c_is_applied_to_final_logistic_model(split) -> None:
    report = run_step4b_benchmark(split, include_test_evaluation=False)
    model = LogisticRegressionPredictor(
        feature_columns=ALL_FEATURE_COLUMNS,
        config=LogisticRegressionConfig(
            C=report.selected_regularization_C,
            preprocessing=report.preprocessing_spec,
        ),
    )
    model.fit(split.train.X, split.train.y)
    expected = evaluate_predictor(
        "logistic_regression_all_features",
        split.validation,
        model,
    ).metrics.log_loss
    actual = report.validation_evaluations[
        "logistic_regression_all_features"
    ].metrics.log_loss
    assert expected == pytest.approx(actual)


def test_ablation_logistic_models_use_expected_feature_subsets(split) -> None:
    report = run_step4b_benchmark(split, include_test_evaluation=False)
    config = LogisticRegressionConfig(
        C=report.selected_regularization_C,
        preprocessing=report.preprocessing_spec,
    )

    elo_model = LogisticRegressionPredictor(
        feature_columns=ELO_ONLY_FEATURE_COLUMNS,
        config=config,
    ).fit(split.train.X, split.train.y)
    historical_model = LogisticRegressionPredictor(
        feature_columns=HISTORICAL_WITHOUT_ELO_COLUMNS,
        config=config,
    ).fit(split.train.X, split.train.y)

    assert elo_model.feature_columns == ELO_ONLY_FEATURE_COLUMNS
    assert historical_model.feature_columns == HISTORICAL_WITHOUT_ELO_COLUMNS
    assert (
        report.ablation_validation["logistic_elo_only"].metrics.log_loss
        == evaluate_predictor(
            "logistic_elo_only", split.validation, elo_model
        ).metrics.log_loss
    )
    assert (
        report.ablation_validation["logistic_historical_without_elo"].metrics.log_loss
        == evaluate_predictor(
            "logistic_historical_without_elo", split.validation, historical_model
        ).metrics.log_loss
    )


def test_benchmark_report_is_deterministic_for_fixed_split(split) -> None:
    first = run_step4b_benchmark(split, include_test_evaluation=False)
    second = run_step4b_benchmark(split, include_test_evaluation=False)

    assert first.selected_regularization_C == second.selected_regularization_C
    pd.testing.assert_frame_equal(
        first.regularization_comparison,
        second.regularization_comparison,
    )
