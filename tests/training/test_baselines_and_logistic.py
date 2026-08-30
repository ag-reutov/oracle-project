"""Tests for Step 4B baselines and logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dota_predictor.features.team_elo import (
    DIRE_TEAM_ELO_COLUMN,
    RADIANT_TEAM_ELO_COLUMN,
    expected_score,
)
from dota_predictor.training.baselines import (
    ConstantProbabilityBaseline,
    EloOnlyProbabilityBaseline,
    EmpiricalRateBaseline,
)
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS
from dota_predictor.training.logistic_model import (
    LogisticRegressionPredictor,
    standardized_coefficients,
)


def _elo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            RADIANT_TEAM_ELO_COLUMN: [1500.0, 1600.0],
            DIRE_TEAM_ELO_COLUMN: [1500.0, 1400.0],
        }
    )


def test_constant_baseline_predictions_are_in_unit_interval() -> None:
    model = ConstantProbabilityBaseline(probability=0.5)
    model.fit(_elo_frame(), pd.Series([True, False]))
    p = model.predict_radiant_win_proba(_elo_frame())
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.allclose(p, 0.5)


def test_empirical_baseline_uses_train_rate_only() -> None:
    train_y = pd.Series([True, True, False, False])
    model = EmpiricalRateBaseline()
    model.fit(_elo_frame(), train_y)
    p = model.predict_radiant_win_proba(_elo_frame())
    assert np.allclose(p, 0.5)


def test_elo_only_matches_expected_score_formula() -> None:
    frame = _elo_frame()
    model = EloOnlyProbabilityBaseline()
    model.fit(frame, pd.Series([True, False]))
    p = model.predict_radiant_win_proba(frame)
    assert p.iloc[0] == pytest.approx(expected_score(1500.0, 1500.0))
    assert p.iloc[1] == pytest.approx(expected_score(1600.0, 1400.0))


def test_logistic_regression_predictions_are_deterministic() -> None:
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n))
    cols = ("f1", "f2", "f3")
    model_a = LogisticRegressionPredictor(feature_columns=cols).fit(X, y)
    model_b = LogisticRegressionPredictor(feature_columns=cols).fit(X, y)
    assert np.allclose(
        model_a.predict_radiant_win_proba(X), model_b.predict_radiant_win_proba(X)
    )


def test_logistic_regression_handles_nulls_via_pipeline() -> None:
    X = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0, np.nan, 2.0, 5.0, 6.0],
            "b": [0.0, 1.0, np.nan, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    model = LogisticRegressionPredictor(feature_columns=("a", "b")).fit(X, y)
    p = model.predict_radiant_win_proba(X)
    assert len(p) == len(X)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_logistic_regression_never_sees_target_or_identity_columns() -> None:
    rng = np.random.default_rng(1)
    n = 40
    feature_data = {col: rng.normal(size=n) for col in ALL_FEATURE_COLUMNS[:5]}
    X = pd.DataFrame(feature_data)
    X["match_id"] = np.arange(n)
    X["start_time"] = pd.date_range("2024-01-01", periods=n, freq="D")
    y = pd.Series(rng.integers(0, 2, size=n))
    cols = tuple(ALL_FEATURE_COLUMNS[:5])
    model = LogisticRegressionPredictor(feature_columns=cols).fit(X, y)
    p = model.predict_radiant_win_proba(X)
    assert len(p) == n


def test_empirical_baseline_predict_before_fit_raises() -> None:
    model = EmpiricalRateBaseline()
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict_proba(_elo_frame())


def test_elo_only_baseline_missing_columns_raises() -> None:
    model = EloOnlyProbabilityBaseline()
    with pytest.raises(ValueError, match="requires column"):
        model.fit(pd.DataFrame({"x": [1.0]}), pd.Series([True]))


def test_logistic_predict_before_fit_raises() -> None:
    model = LogisticRegressionPredictor(feature_columns=("a",))
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict_proba(pd.DataFrame({"a": [1.0]}))


def test_standardized_coefficients_before_fit_raises() -> None:
    model = LogisticRegressionPredictor(feature_columns=("a",))
    with pytest.raises(RuntimeError, match="must be fit"):
        standardized_coefficients(model)
