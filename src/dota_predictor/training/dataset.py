"""Step 4A: model-ready training dataset assembly.

Converts a `PreDraftSnapshot` (Step 3B/3C) into a `ModelReadyDataset`:
`X` (features only), `y` (target only), and `context` (identity
columns for auditing/splitting) -- three row-aligned, independently
owned frames.

This module is deliberately separate from `features.pre_draft_snapshot`:
feature generation continues to describe what was knowable at
prediction time (Step 3A/3B/3C); this module describes which of those
already-computed rows/columns are usable for ML, and in what
deterministic order. It never computes, alters, or reinterprets any
feature or the target.

Eligibility rule (v1)
----------------------
Every row of `PreDraftSnapshot.to_frame()` is eligible. No match is
excluded here for having a NULL feature value: a NULL in
`FEATURE_COLUMNS` means "this team/player had no observed history
before this match" (see `features.pre_draft_snapshot` and
`features.team_elo` for exactly which columns can be NULL and why) --
that is real historical information, not malformed data, and dropping
those rows would silently discard the coldest-start portion of the
dataset without ever being asked to. If a future version needs an
actual exclusion rule (e.g. "require >= N prior matches per team"), it
must be added here explicitly, named, and documented -- never implied
by a NaN check.

Ordering, not filtering, is this module's only structural change: rows
are sorted by (`start_time`, `match_id`) for a fully deterministic row
order. `match_id` is used strictly as a *presentation* tie-breaker for
rows that already share the exact same `start_time` -- it is never
used to decide feature eligibility or values (that invariant belongs
to `features.pre_draft_snapshot`/`features.team_elo`, and is unaffected
by this module).

Missing values are never imputed here -- see `training.diagnostics` for
a reusable, read-only way to inspect/report them instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
    PreDraftSnapshot,
)
from dota_predictor.training._alignment import assert_row_aligned

__all__ = [
    "ModelReadyDataset",
    "TrainingDatasetError",
    "build_model_ready_dataset",
]


class TrainingDatasetError(ValueError):
    """Raised when a `ModelReadyDataset`'s X/y/context contract is
    violated (wrong columns, misaligned rows, ...)."""


@dataclass(frozen=True)
class ModelReadyDataset:
    """`X`/`y`/`context`, row-aligned (same index), one row per eligible
    canonical match, sorted by (`start_time`, `match_id`).

    * `X`: exactly `feature_columns` (defaults to `FEATURE_COLUMNS`) --
      never identity columns, never the target.
    * `y`: exactly `target_column` (defaults to `TARGET_COLUMN`,
      `radiant_win`), aligned index-for-index with `X`.
    * `context`: exactly `identity_columns` (defaults to
      `IDENTITY_COLUMNS`, includes `match_id`/`start_time`) for
      auditing/splitting -- never fed to a model.

    Each frame/series is an independent copy: mutating one, or the
    `PreDraftSnapshot` this was built from, cannot affect the others.
    """

    X: pd.DataFrame
    y: pd.Series
    context: pd.DataFrame
    feature_columns: tuple[str, ...] = field(default=FEATURE_COLUMNS)
    target_column: str = field(default=TARGET_COLUMN)
    identity_columns: tuple[str, ...] = field(default=IDENTITY_COLUMNS)

    def __post_init__(self) -> None:
        if list(self.X.columns) != list(self.feature_columns):
            raise TrainingDatasetError(
                "X must contain exactly feature_columns, in order, and nothing else"
            )
        if self.y.name != self.target_column:
            raise TrainingDatasetError(f"y must be named {self.target_column!r}")
        if list(self.context.columns) != list(self.identity_columns):
            raise TrainingDatasetError(
                "context must contain exactly identity_columns, in order, "
                "and nothing else"
            )
        try:
            assert_row_aligned(self.X, self.y, self.context)
        except ValueError as exc:
            raise TrainingDatasetError(str(exc)) from exc

    def __len__(self) -> int:
        return len(self.X)


def build_model_ready_dataset(snapshot: PreDraftSnapshot) -> ModelReadyDataset:
    """Build a `ModelReadyDataset` from `snapshot`.

    Materializes `snapshot.to_frame()` exactly once, sorts it
    deterministically by (`start_time`, `match_id`), and splits it into
    `X`/`y`/`context` copies sharing a fresh `0..n-1` `RangeIndex`.

    Does not mutate `snapshot`: `PreDraftSnapshot.to_frame()` already
    returns a freshly materialized `DataFrame` on every call (see that
    module), and every frame/series returned here is an explicit
    `.copy()` of a slice of that local frame -- never a view back into
    `snapshot`.
    """
    full = snapshot.to_frame()
    ordered = full.sort_values(["start_time", "match_id"], kind="stable").reset_index(
        drop=True
    )

    feature_columns = snapshot.feature_columns
    target_column = snapshot.target_column
    identity_columns = snapshot.identity_columns

    X = ordered[list(feature_columns)].copy()
    y = ordered[target_column].copy()
    context = ordered[list(identity_columns)].copy()

    return ModelReadyDataset(
        X=X,
        y=y,
        context=context,
        feature_columns=feature_columns,
        target_column=target_column,
        identity_columns=identity_columns,
    )
