"""Step 4A: missing-value reporting (no imputation).

`.cursor/rules/ml.mdc` calls for making unavailable information
explicit rather than silently filling it in. This module only
*reports* what is missing; it never fills, drops, or otherwise changes
a single value in any frame passed to it.

A NULL in a `FEATURE_COLUMNS` column produced by
`features.pre_draft_snapshot`/`features.team_elo` means "this
team/player had no observed history before this match" -- it is real
historical information (see those modules' docstrings for exactly
which columns can be NULL and why), not malformed data. This module
does not, and cannot, silently reclassify that; the distinction is
made once, at the point the NULL is produced (Step 3B/3C), and this
module just surfaces the resulting counts so that distinction stays
visible instead of disappearing into an imputation step.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

__all__ = ["missing_value_report", "rows_with_any_missing"]


def missing_value_report(
    frame: pd.DataFrame, *, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """One row per column in `columns` (defaulting to every column of
    `frame`), indexed by column name, with `null_count` and
    `null_percentage` (0-100, of `len(frame)`).

    `null_percentage` is `0.0` for every column when `frame` is empty
    (by convention: zero rows means zero missing values to report, not
    a division-by-zero error).
    """
    selected = list(columns) if columns is not None else list(frame.columns)
    null_counts = frame[selected].isna().sum()
    total = len(frame)
    null_percentage = (null_counts / total * 100.0) if total > 0 else null_counts * 0.0
    return pd.DataFrame(
        {"null_count": null_counts, "null_percentage": null_percentage}
    ).loc[selected]


def rows_with_any_missing(
    frame: pd.DataFrame, *, columns: Sequence[str] | None = None
) -> pd.Series:
    """Boolean mask, aligned to `frame`'s index: `True` for a row with
    at least one NULL among `columns` (defaulting to every column of
    `frame`)."""
    selected = list(columns) if columns is not None else list(frame.columns)
    return frame[selected].isna().any(axis=1)
