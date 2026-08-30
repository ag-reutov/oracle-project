"""Tests for Step 4B preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS
from dota_predictor.training.preprocessing import (
    MISSINGNESS_INDICATOR_SUFFIX,
    build_preprocessing_pipeline,
)


def _frame_with_nulls() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [10.0, 20.0, np.nan],
        }
    )


def test_missingness_indicators_are_added_and_nulls_are_median_imputed() -> None:
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 20.0, 30.0]})
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(train)
    out = pipeline.named_steps["missingness_impute"].transform(
        pd.DataFrame({"a": [np.nan, 2.0], "b": [10.0, np.nan]})
    )
    feature_names = pipeline.named_steps["missingness_impute"].get_feature_names_out()
    frame = pd.DataFrame(out, columns=feature_names)
    assert f"a{MISSINGNESS_INDICATOR_SUFFIX}" in frame.columns
    assert frame.loc[0, f"a{MISSINGNESS_INDICATOR_SUFFIX}"] == 1.0
    assert frame.loc[0, "a"] == pytest.approx(2.0)  # train median of a is 2.0
    assert frame.loc[1, f"b{MISSINGNESS_INDICATOR_SUFFIX}"] == 1.0


def test_scaler_statistics_use_train_only() -> None:
    train = pd.DataFrame({"x": [0.0, 0.0, 10.0]})
    val = pd.DataFrame({"x": [1000.0]})
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(train)
    val_scaled = pipeline.transform(val)
    # If scaler leaked validation, the value would be ~0; with train-only
    # stats it should be a large z-score.
    assert val_scaled[0, 0] > 5.0


def test_pipeline_does_not_use_literal_zero_imputation_for_semantic_nulls() -> None:
    train = pd.DataFrame({"history_rate": [0.5, np.nan, 1.0]})
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(train)
    transformed = pipeline.named_steps["missingness_impute"].transform(
        pd.DataFrame({"history_rate": [np.nan]})
    )
    names = pipeline.named_steps["missingness_impute"].get_feature_names_out()
    value = pd.DataFrame([transformed[0]], columns=names)["history_rate"].iloc[0]
    assert value == pytest.approx(0.75)
    assert value != 0.0


def test_pipeline_transform_before_fit_raises() -> None:
    pipeline = build_preprocessing_pipeline()
    with pytest.raises(NotFittedError):
        pipeline.transform(pd.DataFrame({"a": [1.0]}))


def test_get_feature_names_out_doubles_input_columns_in_transform_order() -> None:
    """Regression: indicator names must be derived from the original
    ``feature_names_in_``, not from the list being extended (that
    generator never terminates)."""
    assert len(ALL_FEATURE_COLUMNS) == 33
    train = pd.DataFrame(
        {column: np.arange(4, dtype=float) for column in ALL_FEATURE_COLUMNS}
    )
    train.iloc[0, 0] = np.nan

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(train)
    imputer = pipeline.named_steps["missingness_impute"]
    names = imputer.get_feature_names_out()
    transformed = imputer.transform(train)

    assert len(names) == 66
    assert list(names[:33]) == list(ALL_FEATURE_COLUMNS)
    assert list(names[33:]) == [
        f"{column}{MISSINGNESS_INDICATOR_SUFFIX}" for column in ALL_FEATURE_COLUMNS
    ]
    assert transformed.shape == (len(train), 66)
    labeled = pd.DataFrame(transformed, columns=names)
    first_feature = ALL_FEATURE_COLUMNS[0]
    first_indicator = f"{first_feature}{MISSINGNESS_INDICATOR_SUFFIX}"
    assert labeled.columns.tolist() == list(names)
    assert labeled.loc[0, first_indicator] == 1.0
    assert labeled.loc[1, first_indicator] == 0.0
    assert transformed[0, 33] == labeled.loc[0, first_indicator]
