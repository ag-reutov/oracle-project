"""Predefined training-memory policies for walk-forward diagnostics.

These filters choose which *already assembled* historical rows may be
used to fit a fold. They do not recompute features, change OOS test
windows, or peek at evaluation-period version membership.

Policies
--------
``EXPANDING``
    Every eligible row with ``start_time <= past_end`` (existing
    walk-forward past).
``LAST_365D`` / ``LAST_180D``
    Expanding past further restricted to
    ``start_time >= past_end - N days`` (inclusive lower bound).
``CURRENT_PLUS_PREVIOUS_VERSION``
    Expanding past further restricted to the current version and the
    immediately preceding *represented* version.

Current version is the ``game_version_id`` of the latest past match
(``start_time <= past_end``). It is **not** read from the evaluation
window. If that window later spans a newer version, training does not
add it. Previous is the represented version immediately before current
in first-seen order among past rows only. Future rows, including later
matches of the current version, are never eligible.

Equal ``start_time`` groups are kept together because the filter is
exactly on ``start_time`` / ``game_version_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from dota_predictor.training.dataset import ModelReadyDataset
from dota_predictor.training.split import ChronologicalSplitError, DatasetPartition
from dota_predictor.training.walk_forward import (
    WalkForwardFold,
    _partition,
    _train_end_within_past,
)

__all__ = [
    "MEMORY_POLICIES",
    "POLICY_CURRENT_PLUS_PREVIOUS_VERSION",
    "POLICY_EXPANDING",
    "POLICY_LAST_180D",
    "POLICY_LAST_365D",
    "MemoryRestrictedFold",
    "calendar_cutoff",
    "current_and_previous_version_ids",
    "eligible_past_mask",
    "restrict_fold_to_memory",
]

POLICY_EXPANDING = "EXPANDING"
POLICY_LAST_365D = "LAST_365D"
POLICY_LAST_180D = "LAST_180D"
POLICY_CURRENT_PLUS_PREVIOUS_VERSION = "CURRENT_PLUS_PREVIOUS_VERSION"

MEMORY_POLICIES: tuple[str, ...] = (
    POLICY_EXPANDING,
    POLICY_LAST_365D,
    POLICY_LAST_180D,
    POLICY_CURRENT_PLUS_PREVIOUS_VERSION,
)

_CALENDAR_DAYS: dict[str, int] = {
    POLICY_LAST_365D: 365,
    POLICY_LAST_180D: 180,
}


def calendar_cutoff(past_end: datetime, *, days: int) -> pd.Timestamp:
    """Inclusive lower bound for a trailing calendar-day window."""
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    return pd.Timestamp(past_end) - pd.Timedelta(days=days)


def current_and_previous_version_ids(
    context: pd.DataFrame, *, past_end: datetime
) -> tuple[int | None, int | None]:
    """Current / previous represented versions from past rows only.

    Current is the ``game_version_id`` at the latest ``start_time``
    with ``start_time <= past_end``. If several versions share that
    timestamp, the largest id is used. Previous is the represented
    version immediately before current in first-seen ``start_time``
    order among those past rows. Evaluation-window versions are ignored.
    """
    start_time = pd.to_datetime(context["start_time"])
    past = context.loc[start_time <= pd.Timestamp(past_end)]
    if past.empty:
        return None, None
    first_seen = (
        past.groupby("game_version_id", dropna=True)["start_time"]
        .min()
        .sort_values(kind="stable")
    )
    ordered = [int(value) for value in first_seen.index]
    latest_time = past["start_time"].max()
    at_latest = past.loc[past["start_time"] == latest_time, "game_version_id"]
    current = int(pd.Series(at_latest).dropna().max())
    if current not in ordered:
        return current, None
    idx = ordered.index(current)
    previous = ordered[idx - 1] if idx > 0 else None
    return current, previous


def eligible_past_mask(
    context: pd.DataFrame,
    *,
    policy: str,
    past_end: datetime,
) -> pd.Series:
    """Boolean mask over ``context`` for rows allowed as train/validation.

    Rows after ``past_end`` are always False. ``policy`` must be one of
    ``MEMORY_POLICIES``.
    """
    if policy not in MEMORY_POLICIES:
        raise ValueError(f"unknown memory policy {policy!r}")
    start_time = pd.to_datetime(context["start_time"])
    boundary = pd.Timestamp(past_end)
    past = start_time <= boundary
    if policy == POLICY_EXPANDING:
        return past
    if policy in _CALENDAR_DAYS:
        cutoff = calendar_cutoff(boundary, days=_CALENDAR_DAYS[policy])
        return past & (start_time >= cutoff)
    current, previous = current_and_previous_version_ids(
        context, past_end=boundary
    )
    allowed: set[int] = set()
    if current is not None:
        allowed.add(current)
    if previous is not None:
        allowed.add(previous)
    versions = context["game_version_id"]
    return past & versions.isin(allowed)


def _unique_versions(frame: pd.DataFrame) -> tuple[int, ...]:
    values = frame["game_version_id"].dropna().astype(int)
    return tuple(sorted(values.unique().tolist()))


def _coverage_row(
    *,
    fold_id: int,
    policy: str,
    train: DatasetPartition | None,
    validation: DatasetPartition | None,
    test: DatasetPartition,
    skipped: bool,
    skip_reason: str | None,
    current_version_id: int | None,
    previous_version_id: int | None,
) -> dict[str, object]:
    def _range(partition: DatasetPartition | None) -> tuple[object, object]:
        if partition is None or len(partition) == 0:
            return pd.NaT, pd.NaT
        times = partition.context["start_time"]
        return times.min(), times.max()

    train_start, train_end = _range(train)
    val_start, val_end = _range(validation)
    test_start, test_end = _range(test)
    return {
        "fold_id": fold_id,
        "policy": policy,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "n_train": 0 if train is None else len(train),
        "n_validation": 0 if validation is None else len(validation),
        "n_evaluation": len(test),
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": val_start,
        "validation_end": val_end,
        "evaluation_start": test_start,
        "evaluation_end": test_end,
        "train_game_versions": (
            () if train is None else _unique_versions(train.context)
        ),
        "validation_game_versions": (
            () if validation is None else _unique_versions(validation.context)
        ),
        "evaluation_game_versions": _unique_versions(test.context),
        "current_version_id": current_version_id,
        "previous_version_id": previous_version_id,
    }


@dataclass(frozen=True)
class MemoryRestrictedFold:
    """One walk-forward fold after a training-memory filter.

    ``test`` is the original fold's OOS partition, unchanged. Train and
    validation are a trailing split of the policy-eligible past.
    """

    fold_id: int
    policy: str
    train: DatasetPartition | None
    validation: DatasetPartition | None
    test: DatasetPartition
    train_end: datetime | None
    validation_end: datetime
    test_end: datetime
    skipped: bool
    skip_reason: str | None
    current_version_id: int | None
    previous_version_id: int | None
    coverage: dict[str, object]


def restrict_fold_to_memory(
    dataset: ModelReadyDataset,
    fold: WalkForwardFold,
    *,
    policy: str,
    train_fraction_of_past: float,
) -> MemoryRestrictedFold:
    """Filter ``fold``'s past by ``policy``; keep the same OOS test rows.

    ``C`` validation is a trailing slice of the *filtered* past, never
    of rows outside the memory window. Evaluation matches are not used
    to decide eligibility.
    """
    past_end = fold.validation_end
    current, previous = current_and_previous_version_ids(
        dataset.context, past_end=past_end
    )
    eligible = eligible_past_mask(
        dataset.context, policy=policy, past_end=past_end
    )
    test_ids = set(fold.test.context["match_id"])
    if eligible[dataset.context["match_id"].isin(test_ids)].any():
        raise ChronologicalSplitError(
            f"{policy} eligibility leaked evaluation rows into the past"
        )

    def _skipped(reason: str) -> MemoryRestrictedFold:
        coverage = _coverage_row(
            fold_id=fold.fold_id,
            policy=policy,
            train=None,
            validation=None,
            test=fold.test,
            skipped=True,
            skip_reason=reason,
            current_version_id=current,
            previous_version_id=previous,
        )
        return MemoryRestrictedFold(
            fold_id=fold.fold_id,
            policy=policy,
            train=None,
            validation=None,
            test=fold.test,
            train_end=None,
            validation_end=past_end,
            test_end=fold.test_end,
            skipped=True,
            skip_reason=reason,
            current_version_id=current,
            previous_version_id=previous,
            coverage=coverage,
        )

    eligible_times = dataset.context.loc[eligible, "start_time"]
    if eligible_times.empty:
        return _skipped("no eligible training rows under this memory policy")
    try:
        train_end = _train_end_within_past(
            eligible_times, pd.Timestamp(past_end), train_fraction_of_past
        )
    except ChronologicalSplitError as exc:
        return _skipped(str(exc))

    start_time = pd.to_datetime(dataset.context["start_time"])
    train_mask = eligible & (start_time <= train_end)
    validation_mask = (
        eligible & (start_time > train_end) & (start_time <= pd.Timestamp(past_end))
    )
    train = _partition(dataset, train_mask)
    validation = _partition(dataset, validation_mask)
    if len(train) == 0:
        return _skipped("empty train partition after memory filter")
    if len(validation) == 0:
        return _skipped("empty validation partition after memory filter")
    if train.y.nunique() < 2:
        return _skipped("training labels are a single class after memory filter")
    if not (
        train.context["start_time"].max() < validation.context["start_time"].min()
    ):
        raise ChronologicalSplitError(
            f"{policy} fold {fold.fold_id} train/validation overlap"
        )
    if not (
        validation.context["start_time"].max()
        < fold.test.context["start_time"].min()
    ):
        raise ChronologicalSplitError(
            f"{policy} fold {fold.fold_id} validation/test overlap"
        )
    coverage = _coverage_row(
        fold_id=fold.fold_id,
        policy=policy,
        train=train,
        validation=validation,
        test=fold.test,
        skipped=False,
        skip_reason=None,
        current_version_id=current,
        previous_version_id=previous,
    )
    return MemoryRestrictedFold(
        fold_id=fold.fold_id,
        policy=policy,
        train=train,
        validation=validation,
        test=fold.test,
        train_end=train_end,
        validation_end=past_end,
        test_end=fold.test_end,
        skipped=False,
        skip_reason=None,
        current_version_id=current,
        previous_version_id=previous,
        coverage=coverage,
    )
