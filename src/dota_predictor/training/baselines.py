"""Step 4B: simple probability baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dota_predictor.features.team_elo import (
    DIRE_TEAM_ELO_COLUMN,
    RADIANT_TEAM_ELO_COLUMN,
    expected_score,
)

__all__ = [
    "ConstantProbabilityBaseline",
    "EloOnlyProbabilityBaseline",
    "EmpiricalRateBaseline",
    "ProbabilityPredictor",
]


class ProbabilityPredictor:
    """Minimal sklearn-like contract shared by baselines and pipelines."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ProbabilityPredictor:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_radiant_win_proba(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.predict_proba(X)[:, 1], index=X.index)


@dataclass
class ConstantProbabilityBaseline(ProbabilityPredictor):
    """Always predict ``probability`` for Radiant win (default 0.5)."""

    probability: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ConstantProbabilityBaseline:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class EmpiricalRateBaseline(ProbabilityPredictor):
    """Predict the TRAIN-set empirical Radiant win rate for every row."""

    probability: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EmpiricalRateBaseline:
        self.probability = float(y.astype(int).mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.probability is None:
            raise RuntimeError("EmpiricalRateBaseline must be fit before predict")
        p = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class EloOnlyProbabilityBaseline(ProbabilityPredictor):
    """Convert pre-match team Elo ratings to Radiant win probability via
    the standard expected-score formula (``features.team_elo``)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EloOnlyProbabilityBaseline:
        for column in (RADIANT_TEAM_ELO_COLUMN, DIRE_TEAM_ELO_COLUMN):
            if column not in X.columns:
                raise ValueError(
                    f"EloOnlyProbabilityBaseline requires column {column!r}"
                )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        radiant = X[RADIANT_TEAM_ELO_COLUMN].to_numpy(dtype=float)
        dire = X[DIRE_TEAM_ELO_COLUMN].to_numpy(dtype=float)
        p = np.vectorize(expected_score)(radiant, dire)
        return np.column_stack([1.0 - p, p])
