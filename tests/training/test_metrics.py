"""Tests for Step 4B evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from dota_predictor.training.metrics import (
    accuracy_at_threshold,
    calibration_bins,
    evaluate_probabilities,
    expected_calibration_error,
)


def test_perfect_predictions_have_best_possible_scores() -> None:
    y = np.array([0, 1, 1, 0])
    p = np.array([0.01, 0.99, 0.95, 0.05])
    metrics = evaluate_probabilities(y, p)
    assert metrics.accuracy_at_0_5 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.brier_score < 0.01
    assert metrics.log_loss < 0.1


def test_constant_half_probability_metrics_hand_computed() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    metrics = evaluate_probabilities(y, p)
    assert metrics.accuracy_at_0_5 == 0.5
    assert metrics.brier_score == pytest.approx(0.25)
    assert metrics.log_loss == pytest.approx(-np.log(0.5))
    assert metrics.roc_auc == pytest.approx(0.5)


def test_calibration_bins_report_counts_and_frequencies() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    table = calibration_bins(y, p, n_bins=2)
    assert table["count"].sum() == 4
    assert table.loc[table["count"] > 0, "observed_frequency"].between(0, 1).all()


def test_expected_calibration_error_is_zero_for_perfect_calibration() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.9, 0.9])
    ece = expected_calibration_error(y, p, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-12)


def test_accuracy_at_threshold() -> None:
    y = np.array([0, 1, 1, 0])
    p = np.array([0.4, 0.6, 0.8, 0.2])
    assert accuracy_at_threshold(y, p, threshold=0.5) == 1.0
