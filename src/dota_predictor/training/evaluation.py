"""Step 4B: baseline-model evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dota_predictor.training.baselines import (
    ConstantProbabilityBaseline,
    EloOnlyProbabilityBaseline,
    EmpiricalRateBaseline,
    ProbabilityPredictor,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    HISTORICAL_WITHOUT_ELO_COLUMNS,
)
from dota_predictor.training.logistic_model import (
    LogisticRegressionConfig,
    LogisticRegressionPredictor,
    standardized_coefficients,
)
from dota_predictor.training.metrics import EvaluationMetrics, evaluate_probabilities
from dota_predictor.training.split import ChronologicalSplit, DatasetPartition

__all__ = [
    "BenchmarkReport",
    "ModelEvaluation",
    "PredictionResult",
    "evaluate_predictor",
    "run_step4b_benchmark",
]

REGULARIZATION_CANDIDATES: tuple[float, ...] = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class PredictionResult:
    """Aligned predictions with audit context."""

    context: pd.DataFrame
    y_true: pd.Series
    p_radiant_win: pd.Series


@dataclass(frozen=True)
class ModelEvaluation:
    name: str
    metrics: EvaluationMetrics
    predictions: PredictionResult


@dataclass
class BenchmarkReport:
    """Full Step 4B output for one chronological split."""

    preprocessing_spec: object
    selected_regularization_C: float
    regularization_comparison: pd.DataFrame
    validation_evaluations: dict[str, ModelEvaluation]
    ablation_validation: dict[str, ModelEvaluation]
    coefficients: pd.DataFrame
    test_evaluations: dict[str, ModelEvaluation] = field(default_factory=dict)


def _predictions(
    partition: DatasetPartition, predictor: ProbabilityPredictor
) -> PredictionResult:
    p = pd.Series(
        predictor.predict_radiant_win_proba(partition.X),
        index=partition.y.index,
        name="p_radiant_win",
    )
    return PredictionResult(
        context=partition.context.copy(),
        y_true=partition.y.copy(),
        p_radiant_win=p,
    )


def evaluate_predictor(
    name: str,
    partition: DatasetPartition,
    predictor: ProbabilityPredictor,
) -> ModelEvaluation:
    preds = _predictions(partition, predictor)
    metrics = evaluate_probabilities(preds.y_true, preds.p_radiant_win)
    return ModelEvaluation(name=name, metrics=metrics, predictions=preds)


def _fit_logistic(
    train: DatasetPartition,
    feature_columns: tuple[str, ...],
    *,
    config: LogisticRegressionConfig,
) -> LogisticRegressionPredictor:
    model = LogisticRegressionPredictor(
        feature_columns=feature_columns, config=config
    )
    model.fit(train.X, train.y)
    return model


def _select_regularization(
    train: DatasetPartition,
    validation: DatasetPartition,
    feature_columns: tuple[str, ...],
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    best_c = REGULARIZATION_CANDIDATES[0]
    best_log_loss = float("inf")

    for c in REGULARIZATION_CANDIDATES:
        model = _fit_logistic(
            train,
            feature_columns,
            config=LogisticRegressionConfig(C=c),
        )
        metrics = evaluate_predictor(
            f"logistic_regression_C={c}",
            validation,
            model,
        ).metrics
        rows.append({"C": c, "validation_log_loss": metrics.log_loss})
        if metrics.log_loss < best_log_loss:
            best_log_loss = metrics.log_loss
            best_c = c

    return best_c, pd.DataFrame(rows)


def run_step4b_benchmark(
    split: ChronologicalSplit,
    *,
    include_test_evaluation: bool = True,
) -> BenchmarkReport:
    """Run the full Step 4B benchmark on a fixed chronological split.

    Fitting/preprocessing uses TRAIN only. Model/specification decisions
    (regularization ``C``) use VALIDATION only. TEST is evaluated once
    at the end with the frozen specification, if ``include_test_evaluation``.
    """
    from dota_predictor.training.preprocessing import PreprocessingSpec

    preprocessing_spec = PreprocessingSpec()

    # --- baselines on validation ---
    constant = ConstantProbabilityBaseline(probability=0.5)
    constant.fit(split.train.X, split.train.y)

    empirical = EmpiricalRateBaseline()
    empirical.fit(split.train.X, split.train.y)

    elo_only = EloOnlyProbabilityBaseline()
    elo_only.fit(split.train.X, split.train.y)

    validation_evaluations = {
        "constant_0.5": evaluate_predictor(
            "constant_0.5", split.validation, constant
        ),
        "empirical_train_rate": evaluate_predictor(
            "empirical_train_rate", split.validation, empirical
        ),
        "elo_only": evaluate_predictor("elo_only", split.validation, elo_only),
    }

    # --- bounded regularization selection on validation ---
    selected_c, regularization_comparison = _select_regularization(
        split.train, split.validation, ALL_FEATURE_COLUMNS
    )
    logistic_config = LogisticRegressionConfig(
        C=selected_c, preprocessing=preprocessing_spec
    )
    logistic_all = _fit_logistic(split.train, ALL_FEATURE_COLUMNS, config=logistic_config)
    validation_evaluations["logistic_regression_all_features"] = evaluate_predictor(
        "logistic_regression_all_features",
        split.validation,
        logistic_all,
    )

    # --- ablation on validation (same selected C, frozen preprocessing spec) ---
    ablation_validation = {
        "elo_only": validation_evaluations["elo_only"],
        "logistic_elo_only": evaluate_predictor(
            "logistic_elo_only",
            split.validation,
            _fit_logistic(
                split.train,
                ELO_ONLY_FEATURE_COLUMNS,
                config=logistic_config,
            ),
        ),
        "logistic_historical_without_elo": evaluate_predictor(
            "logistic_historical_without_elo",
            split.validation,
            _fit_logistic(
                split.train,
                HISTORICAL_WITHOUT_ELO_COLUMNS,
                config=logistic_config,
            ),
        ),
        "logistic_all_features": validation_evaluations[
            "logistic_regression_all_features"
        ],
    }

    coefficients = standardized_coefficients(logistic_all)

    test_evaluations: dict[str, ModelEvaluation] = {}
    if include_test_evaluation:
        test_evaluations = {
            "constant_0.5": evaluate_predictor("constant_0.5", split.test, constant),
            "empirical_train_rate": evaluate_predictor(
                "empirical_train_rate", split.test, empirical
            ),
            "elo_only": evaluate_predictor("elo_only", split.test, elo_only),
            "logistic_regression_all_features": evaluate_predictor(
                "logistic_regression_all_features",
                split.test,
                logistic_all,
            ),
        }

    return BenchmarkReport(
        preprocessing_spec=preprocessing_spec,
        selected_regularization_C=selected_c,
        regularization_comparison=regularization_comparison,
        validation_evaluations=validation_evaluations,
        ablation_validation=ablation_validation,
        coefficients=coefficients,
        test_evaluations=test_evaluations,
    )
