"""Tests for the Step 4A missing-value reporting utilities
(`training.diagnostics`).

Pure, in-memory tests against small hand-built `DataFrame`s -- these
utilities operate on any frame (not just `ModelReadyDataset.X`), so
there is no need to go through the real Parquet/DuckDB pipeline here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dota_predictor.training.diagnostics import (
    missing_value_report,
    rows_with_any_missing,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [np.nan, np.nan, 3.0, 4.0],
            "c": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_missing_value_report_counts_and_percentages() -> None:
    report = missing_value_report(_frame())

    assert report.loc["a", "null_count"] == 2
    assert report.loc["a", "null_percentage"] == 50.0
    assert report.loc["b", "null_count"] == 2
    assert report.loc["b", "null_percentage"] == 50.0
    assert report.loc["c", "null_count"] == 0
    assert report.loc["c", "null_percentage"] == 0.0


def test_missing_value_report_respects_column_subset() -> None:
    report = missing_value_report(_frame(), columns=["b"])
    assert list(report.index) == ["b"]


def test_missing_value_report_on_empty_frame_does_not_divide_by_zero() -> None:
    empty = pd.DataFrame({"a": pd.Series(dtype=float)})
    report = missing_value_report(empty)
    assert report.loc["a", "null_count"] == 0
    assert report.loc["a", "null_percentage"] == 0.0
    assert not math.isnan(report.loc["a", "null_percentage"])


def test_rows_with_any_missing_flags_correct_rows() -> None:
    mask = rows_with_any_missing(_frame())
    # row0: a=1.0,b=NaN -> missing; row1: a=NaN,b=NaN -> missing;
    # row2: a=3.0,b=3.0 -> not missing; row3: a=NaN,b=4.0 -> missing.
    assert list(mask) == [True, True, False, True]


def test_rows_with_any_missing_respects_column_subset() -> None:
    mask = rows_with_any_missing(_frame(), columns=["c"])
    assert list(mask) == [False, False, False, False]


def test_reporting_functions_do_not_mutate_the_input_frame() -> None:
    frame = _frame()
    before = frame.copy()
    missing_value_report(frame)
    rows_with_any_missing(frame)
    pd.testing.assert_frame_equal(frame, before)
