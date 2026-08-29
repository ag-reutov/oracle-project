"""Step 4A: chronological train/validation/test split.

Splits a `ModelReadyDataset` into three time-ordered, non-overlapping
partitions using `context.start_time` -- never `match_id`, never a
random shuffle (`.cursor/rules/ml.mdc`: "validate with chronological /
walk-forward splits", the same rule `features.temporal` already
enforces for feature computation).

Equal-`start_time` matches are a single indivisible unit: every match
sharing an exact `start_time` value lands in the same partition, even
if that makes the realized split proportions deviate slightly from the
requested `ChronologicalSplitConfig` fractions -- see
`resolve_split_boundaries`.

This is intentionally a simple holdout split (train/validation/test),
not walk-forward cross-validation; that can be layered on top of
`resolve_split_boundaries`/`ChronologicalSplit` later without changing
this module's contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from dota_predictor.training._alignment import assert_row_aligned
from dota_predictor.training.dataset import ModelReadyDataset

__all__ = [
    "DEFAULT_SPLIT_CONFIG",
    "ChronologicalSplit",
    "ChronologicalSplitConfig",
    "ChronologicalSplitError",
    "DatasetPartition",
    "SplitBoundaries",
    "chronological_split",
    "resolve_split_boundaries",
]


class ChronologicalSplitError(ValueError):
    """Raised when a chronological split cannot be resolved/validated
    (invalid config, degenerate/empty partition, overlap, ...)."""


@dataclass(frozen=True)
class ChronologicalSplitConfig:
    """Target row-count fractions for train/validation (test is the
    remainder, `1 - train_fraction - validation_fraction`).

    Purely a *target*: see `resolve_split_boundaries` for why the
    realized split can deviate slightly (equal-`start_time` groups are
    never broken apart).
    """

    train_fraction: float = 0.70
    validation_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not (0.0 < self.train_fraction < 1.0):
            raise ChronologicalSplitError("train_fraction must be in (0, 1)")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ChronologicalSplitError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ChronologicalSplitError(
                "train_fraction + validation_fraction must leave a "
                "positive fraction for test"
            )


DEFAULT_SPLIT_CONFIG = ChronologicalSplitConfig()


@dataclass(frozen=True)
class SplitBoundaries:
    """Explicit partition cutoffs, both inclusive of their own partition:

    * `train`: `start_time <= train_end`
    * `validation`: `train_end < start_time <= validation_end`
    * `test`: `start_time > validation_end`

    Can be constructed directly (hand-picked dates) instead of going
    through `resolve_split_boundaries`/`ChronologicalSplitConfig` --
    this is the "date boundaries" configurability the module docstring
    refers to.
    """

    train_end: datetime
    validation_end: datetime

    def __post_init__(self) -> None:
        if not self.train_end < self.validation_end:
            raise ChronologicalSplitError(
                "train_end must be strictly before validation_end"
            )


@dataclass(frozen=True)
class DatasetPartition:
    """One named slice of a `ModelReadyDataset`: the same `X`/`y`/
    `context` contract, row-aligned, with its own fresh `RangeIndex`."""

    X: pd.DataFrame
    y: pd.Series
    context: pd.DataFrame

    def __post_init__(self) -> None:
        try:
            assert_row_aligned(self.X, self.y, self.context)
        except ValueError as exc:
            raise ChronologicalSplitError(str(exc)) from exc

    def __len__(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class ChronologicalSplit:
    """The three partitions plus the exact `SplitBoundaries` that
    produced them (retained for auditing -- e.g. logging alongside
    model hyperparameters, per `.cursor/rules/ml.mdc`)."""

    train: DatasetPartition
    validation: DatasetPartition
    test: DatasetPartition
    boundaries: SplitBoundaries


def resolve_split_boundaries(
    context: pd.DataFrame, config: ChronologicalSplitConfig
) -> SplitBoundaries:
    """Compute `SplitBoundaries` from `config`'s row-count fractions,
    over the actual `start_time` distribution in `context`.

    A boundary always lands exactly on a distinct `start_time` value
    that is present in `context` -- never interpolated between two
    timestamps -- so every match sharing that value is unambiguously
    inside the partition it closes. Concretely: `train_end` is the
    earliest distinct `start_time` whose cumulative row count is >=
    `train_fraction * len(context)`; `validation_end` is the earliest
    distinct `start_time` at or after `train_end` whose cumulative row
    count is >= `(train_fraction + validation_fraction) * len(context)`
    (bumped to the next distinct timestamp if that lands on the exact
    same timestamp as `train_end`, so validation is never empty).

    Raises `ChronologicalSplitError` if no non-degenerate boundary pair
    exists for this `context`/`config` combination (e.g. the dataset's
    `start_time` grouping is too coarse relative to the requested
    fractions).
    """
    counts = context["start_time"].value_counts().sort_index()
    cumulative = counts.cumsum()
    total = int(cumulative.iloc[-1])
    cumulative_values = cumulative.to_numpy()

    train_target = config.train_fraction * total
    train_idx = int(np.searchsorted(cumulative_values, train_target, side="left"))
    train_idx = min(train_idx, len(cumulative) - 1)

    combined_target = (config.train_fraction + config.validation_fraction) * total
    validation_idx = int(
        np.searchsorted(cumulative_values, combined_target, side="left")
    )
    validation_idx = max(validation_idx, train_idx)
    if validation_idx == train_idx:
        validation_idx = min(train_idx + 1, len(cumulative) - 1)

    train_end = cumulative.index[train_idx]
    validation_end = cumulative.index[validation_idx]

    if not train_end < validation_end:
        raise ChronologicalSplitError(
            "cannot resolve a non-degenerate chronological split for this "
            "config: the dataset's start_time grouping is too coarse for "
            f"train_fraction={config.train_fraction}, "
            f"validation_fraction={config.validation_fraction}, n={total}"
        )

    return SplitBoundaries(train_end=train_end, validation_end=validation_end)


def _partition(dataset: ModelReadyDataset, mask: pd.Series) -> DatasetPartition:
    return DatasetPartition(
        X=dataset.X.loc[mask].reset_index(drop=True),
        y=dataset.y.loc[mask].reset_index(drop=True),
        context=dataset.context.loc[mask].reset_index(drop=True),
    )


def chronological_split(
    dataset: ModelReadyDataset,
    *,
    config: ChronologicalSplitConfig | None = None,
    boundaries: SplitBoundaries | None = None,
) -> ChronologicalSplit:
    """Split `dataset` into train/validation/test by `context.start_time`.

    Give at most one of `config` (row-count fractions, resolved against
    `dataset` via `resolve_split_boundaries`) or `boundaries` (explicit
    cutoff timestamps). Defaults to `DEFAULT_SPLIT_CONFIG` (70%/15%,
    test is the 15% remainder) if neither is given.

    Raises `ChronologicalSplitError` if both are given, if any resulting
    partition would be empty, or if the no-temporal-overlap invariant
    (`max(train.start_time) < min(validation.start_time) <=
    max(validation.start_time) < min(test.start_time)`) does not hold --
    this is checked directly here, not merely assumed from the boundary
    arithmetic.
    """
    if config is not None and boundaries is not None:
        raise ChronologicalSplitError(
            "give at most one of config or boundaries, not both"
        )

    start_time = dataset.context["start_time"]

    resolved_boundaries = (
        boundaries
        if boundaries is not None
        else resolve_split_boundaries(
            dataset.context, config if config is not None else DEFAULT_SPLIT_CONFIG
        )
    )

    train_mask = start_time <= resolved_boundaries.train_end
    validation_mask = (start_time > resolved_boundaries.train_end) & (
        start_time <= resolved_boundaries.validation_end
    )
    test_mask = start_time > resolved_boundaries.validation_end

    train = _partition(dataset, train_mask)
    validation = _partition(dataset, validation_mask)
    test = _partition(dataset, test_mask)

    for name, partition in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if len(partition) == 0:
            raise ChronologicalSplitError(
                f"resolved chronological split produced an empty {name} partition"
            )

    if not train.context["start_time"].max() < validation.context["start_time"].min():
        raise ChronologicalSplitError(
            "train/validation partitions are not strictly time-ordered"
        )
    if not (validation.context["start_time"].max() < test.context["start_time"].min()):
        raise ChronologicalSplitError(
            "validation/test partitions are not strictly time-ordered"
        )

    return ChronologicalSplit(
        train=train, validation=validation, test=test, boundaries=resolved_boundaries
    )
