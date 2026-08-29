"""Internal row-alignment guard shared by `training.dataset` and
`training.split`.

Not part of the public `training` API (no `__all__`, not re-exported
from `training/__init__.py`) -- it exists only so both
`ModelReadyDataset` and `DatasetPartition` enforce the exact same
"X/y/context are the same rows, in the same order" invariant without
duplicating the check.
"""

from __future__ import annotations

import pandas as pd


def assert_row_aligned(X: pd.DataFrame, y: pd.Series, context: pd.DataFrame) -> None:
    """Raise `ValueError` unless `X`, `y`, and `context` have identical
    length and index -- i.e. row `i` of each refers to the same match."""
    if not (len(X) == len(y) == len(context)):
        raise ValueError("X, y, and context must have the same row count")
    if not (X.index.equals(y.index) and X.index.equals(context.index)):
        raise ValueError("X, y, and context must share the same row index")
