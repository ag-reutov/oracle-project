"""Step 4B: sklearn preprocessing for logistic regression.

Handles the semantic NULLs produced by Step 3B/3C inside a fitted
Pipeline -- Step 4A and feature generation are never modified.

Strategy (fitted on TRAIN only):
1. For every input column, add a binary ``{column}__was_missing``
   indicator preserving the information that a value was absent.
2. Impute remaining NULLs with the TRAIN-set column median (never
   literal zero -- semantic NULL means "no observed history", not 0).
3. Standardize all augmented columns with ``StandardScaler``.

The imputer/scaler statistics are learned exclusively from whatever
partition is passed to ``Pipeline.fit``; callers must fit on TRAIN
and only ``transform`` validation/test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "MISSINGNESS_INDICATOR_SUFFIX",
    "PreprocessingSpec",
    "build_preprocessing_pipeline",
    "expanded_feature_names",
]

MISSINGNESS_INDICATOR_SUFFIX = "__was_missing"


@dataclass(frozen=True)
class PreprocessingSpec:
    """Explicit, reproducible preprocessing configuration."""

    impute_strategy: str = "median"
    add_missingness_indicators: bool = True
    scale: bool = True


class _MissingnessImputeTransformer(BaseEstimator, TransformerMixin):
    """Add per-column missingness indicators, then median-impute."""

    def __init__(self, *, add_missingness_indicators: bool = True) -> None:
        self.add_missingness_indicators = add_missingness_indicators

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: object | None = None
    ) -> _MissingnessImputeTransformer:
        frame = self._as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        self.medians_ = frame.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = self._as_frame(X)
        imputed = frame.fillna(self.medians_)
        if not self.add_missingness_indicators:
            return imputed.to_numpy()
        indicators = frame.isna().astype(float)
        indicators.columns = [
            f"{column}{MISSINGNESS_INDICATOR_SUFFIX}" for column in indicators.columns
        ]
        augmented = pd.concat([imputed, indicators], axis=1)
        return augmented.to_numpy()

    def get_feature_names_out(
        self, input_features: object | None = None
    ) -> np.ndarray:
        names = list(self.feature_names_in_)
        if self.add_missingness_indicators:
            names.extend(f"{column}{MISSINGNESS_INDICATOR_SUFFIX}" for column in names)
        return np.asarray(names, dtype=object)

    def _as_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if not hasattr(self, "feature_names_in_"):
            raise ValueError(
                "numpy array input requires a prior fit with named columns"
            )
        return pd.DataFrame(X, columns=self.feature_names_in_)


def expanded_feature_names(feature_columns: tuple[str, ...]) -> tuple[str, ...]:
    """The column names after missingness indicators are appended."""
    indicators = tuple(
        f"{column}{MISSINGNESS_INDICATOR_SUFFIX}" for column in feature_columns
    )
    return feature_columns + indicators


def build_preprocessing_pipeline(
    spec: PreprocessingSpec | None = None,
) -> Pipeline:
    """Return an unfitted preprocessing ``Pipeline`` for logistic regression."""
    resolved = spec if spec is not None else PreprocessingSpec()
    steps: list[tuple[str, object]] = [
        (
            "missingness_impute",
            _MissingnessImputeTransformer(
                add_missingness_indicators=resolved.add_missingness_indicators
            ),
        )
    ]
    if resolved.scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)
