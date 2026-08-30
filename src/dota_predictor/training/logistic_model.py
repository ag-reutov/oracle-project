"""Step 4B: L2-regularized logistic regression on PRE_DRAFT features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dota_predictor.training.baselines import ProbabilityPredictor
from dota_predictor.training.preprocessing import (
    PreprocessingSpec,
    build_preprocessing_pipeline,
)

__all__ = [
    "DEFAULT_LOGISTIC_CONFIG",
    "LogisticRegressionConfig",
    "LogisticRegressionPredictor",
    "build_logistic_pipeline",
    "standardized_coefficients",
]

DEFAULT_LOGISTIC_CONFIG = None  # set after class definition


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Explicit, reproducible logistic-regression hyperparameters."""

    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 0
    preprocessing: PreprocessingSpec | None = None


DEFAULT_LOGISTIC_CONFIG = LogisticRegressionConfig()


def build_logistic_pipeline(
    config: LogisticRegressionConfig | None = None,
) -> Pipeline:
    """Unfitted ``preprocessing + LogisticRegression(l2)`` pipeline."""
    resolved = config if config is not None else DEFAULT_LOGISTIC_CONFIG
    preprocessing = build_preprocessing_pipeline(resolved.preprocessing)
    classifier = LogisticRegression(
        penalty="l2",
        C=resolved.C,
        solver="lbfgs",
        max_iter=resolved.max_iter,
        random_state=resolved.random_state,
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            ("classifier", classifier),
        ]
    )


@dataclass
class LogisticRegressionPredictor(ProbabilityPredictor):
    """Thin wrapper keeping the sklearn ``Pipeline`` and selected feature
    columns together."""

    feature_columns: tuple[str, ...]
    config: LogisticRegressionConfig = DEFAULT_LOGISTIC_CONFIG
    pipeline_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegressionPredictor:
        selected = X[list(self.feature_columns)]
        pipeline = build_logistic_pipeline(self.config)
        pipeline.fit(selected, y.astype(int))
        self.pipeline_ = pipeline
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("LogisticRegressionPredictor must be fit before predict")
        selected = X[list(self.feature_columns)]
        return self.pipeline_.predict_proba(selected)


def standardized_coefficients(
    predictor: LogisticRegressionPredictor,
) -> pd.DataFrame:
    """Coefficient table for the fitted logistic model.

    Because preprocessing ends with ``StandardScaler``, these coefficients
    are already on the standardized (and missingness-augmented) feature
    scale -- suitable for rough relative-influence inspection, not causal
    interpretation.
    """
    if predictor.pipeline_ is None:
        raise RuntimeError("predictor must be fit before extracting coefficients")

    preprocess = predictor.pipeline_.named_steps["preprocess"]
    classifier: LogisticRegression = predictor.pipeline_.named_steps["classifier"]
    feature_names = preprocess.get_feature_names_out()
    coef = classifier.coef_[0]
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coef})
        .sort_values("coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )
