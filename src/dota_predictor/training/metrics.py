"""Step 4B: probability-focused evaluation metrics and calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

__all__ = [
    "CalibrationBin",
    "EvaluationMetrics",
    "accuracy_at_threshold",
    "bootstrap_mean_ci",
    "calibration_bins",
    "evaluate_probabilities",
    "expected_calibration_error",
    "per_sample_brier",
    "per_sample_log_loss",
]


@dataclass(frozen=True)
class CalibrationBin:
    bin_lower: float
    bin_upper: float
    predicted_mean: float
    observed_frequency: float
    count: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Probability-quality metrics for one model on one partition."""

    log_loss: float
    brier_score: float
    accuracy_at_0_5: float
    roc_auc: float
    expected_calibration_error: float
    calibration_table: pd.DataFrame
    n_samples: int


def accuracy_at_threshold(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, *, threshold: float = 0.5
) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    return float(accuracy_score(y, p >= threshold))


def calibration_bins(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Equal-width bins over [0, 1] with predicted mean, observed frequency,
    and sample count per bin."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(p, edges, right=True) - 1, 0, n_bins - 1)

    rows: list[dict[str, object]] = []
    for idx in range(n_bins):
        mask = bin_index == idx
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_lower": edges[idx],
                    "bin_upper": edges[idx + 1],
                    "predicted_mean": np.nan,
                    "observed_frequency": np.nan,
                    "count": 0,
                }
            )
            continue
        rows.append(
            {
                "bin_lower": edges[idx],
                "bin_upper": edges[idx + 1],
                "predicted_mean": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    """Weighted mean absolute difference between bin predicted mean and
    observed frequency (ECE)."""
    table = calibration_bins(y_true, y_prob, n_bins=n_bins)
    total = table["count"].sum()
    if total == 0:
        return float("nan")
    valid = table["count"] > 0
    abs_diff = (
        table.loc[valid, "predicted_mean"] - table.loc[valid, "observed_frequency"]
    ).abs()
    weights = table.loc[valid, "count"] / total
    return float((abs_diff * weights).sum())


def per_sample_log_loss(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series
) -> np.ndarray:
    """Binary log loss for each row. Mean of this equals ``log_loss``.

    Clips probabilities away from {0, 1} with the same epsilon as
    ``evaluate_probabilities``. Used for paired deltas on a shared
    match set (spec minus Elo; negative means the spec is better).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1.0 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1.0 - p))


def per_sample_brier(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series
) -> np.ndarray:
    """Binary Brier score for each row. Mean of this equals ``brier_score``."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1.0 - 1e-15)
    return (p - y) ** 2


def bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling rows with replacement.

    Match-level paired deltas use this so the interval is over matches,
    not over an assumed parametric SE.
    """
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        return float("nan"), float("nan")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(random_state)
    draws = rng.choice(sample, size=(n_resamples, sample.size), replace=True)
    means = draws.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def evaluate_probabilities(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    n_calibration_bins: int = 10,
) -> EvaluationMetrics:
    """Compute the full Step 4B metric bundle for Radiant-win probabilities."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1.0 - 1e-15)

    cal_table = calibration_bins(y, p, n_bins=n_calibration_bins)
    return EvaluationMetrics(
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier_score=float(brier_score_loss(y, p)),
        accuracy_at_0_5=accuracy_at_threshold(y, p, threshold=0.5),
        roc_auc=float(roc_auc_score(y, p)),
        expected_calibration_error=expected_calibration_error(
            y, p, n_bins=n_calibration_bins
        ),
        calibration_table=cal_table,
        n_samples=len(y),
    )
