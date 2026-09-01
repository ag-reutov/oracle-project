"""Slice 9: frozen-model temporal holdout protocol.

Evaluation protocol only. No new features. Does not change Elo,
production ``FEATURE_COLUMNS``, walk-forward fold construction, or
Slices 0–8 specs.

Candidate specification (frozen)
--------------------------------
Elo + *unconditional* Career Player × Hero
(``logistic_elo_plus_player_hero`` / ``ELO_PLUS_PLAYER_HERO_COLUMNS``).
Slice 8 gates, interactions, and TRAIN/VAL threshold search are not
part of this specification.

Development / OOS boundary (frozen)
-----------------------------------
The Slice 8 expanding-window frame used the entire then-current
post-draft corpus. The last fold's ``test_end`` is the latest timestamp
that informed Slices 7–8. That instant is recorded here and must not
move when later matches arrive.

Holdout
-------
Matches with ``start_time > FROZEN_DEVELOPMENT_END`` only. Equal-
``start_time`` matches at the boundary stay in development. The holdout
must not be used to choose thresholds, gates, hyperparameters, feature
definitions, or preprocessing. Nested ``C`` selection and imputer/scaler
fit use development TRAIN/VAL only.

This module records the protocol and inventories later matches. It does
not score the holdout from ``record_frozen_holdout_protocol``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from dota_predictor.features.duckdb_layer import MATCHES_VIEW, FeatureDuckDBConnection
from dota_predictor.features.player_hero_meta_comparison import MATCH_ID_COLUMN
from dota_predictor.training.dataset import ModelReadyDataset, TrainingDatasetError
from dota_predictor.training.evaluation import REGULARIZATION_CANDIDATES
from dota_predictor.training.feature_sets import (
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    SLICE8_INTERACTION_COLUMNS,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_REFERENCE_SPEC,
    BlockAblationSpec,
)
from dota_predictor.training.preprocessing import PreprocessingSpec
from dota_predictor.training.slice8_player_hero_gating import (
    assert_slice7_slice8_identity,
    build_slice8_model_ready_dataset,
)
from dota_predictor.training.split import ChronologicalSplitError, DatasetPartition
from dota_predictor.training.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG,
    WalkForwardConfig,
    WalkForwardFold,
    _partition,
    _train_end_within_past,
    resolve_walk_forward_folds,
)

__all__ = [
    "FROZEN_DEVELOPMENT_END",
    "FROZEN_DEVELOPMENT_MATCH_COUNT",
    "FROZEN_DEVELOPMENT_OOS_MATCH_COUNT",
    "FROZEN_PREPROCESSING_SPEC",
    "INSPECTED_LATER_T1_MAIN_EVENT_END",
    "INSPECTED_LATER_T1_MAIN_EVENT_LEAGUE_ID",
    "INSPECTED_LATER_T1_MAIN_EVENT_N",
    "INSPECTED_LATER_T1_MAIN_EVENT_NAME",
    "INSPECTED_LATER_T1_MAIN_EVENT_START",
    "FrozenHoldoutEmptyError",
    "FrozenHoldoutProtocol",
    "FrozenHoldoutSplit",
    "HoldoutInventory",
    "assert_development_frame_excludes_holdout",
    "development_end_from_slice8_frame",
    "holdout_mask",
    "inventory_holdout",
    "record_frozen_holdout_protocol",
    "resolve_frozen_holdout_split",
    "subset_model_ready_dataset",
    "utc_datetime",
]


# Latest timestamp in the Slice 8 development/OOS frame (fold 4
# ``test_end`` / corpus max ``start_time``) on the 5967-match
# 2024-02-04 .. 2026-07-19 professional export. Inclusive: matches at
# this instant remain development.
FROZEN_DEVELOPMENT_END = datetime(2026, 7, 19, 17, 49, 1, tzinfo=UTC)
FROZEN_DEVELOPMENT_MATCH_COUNT = 5967
FROZEN_DEVELOPMENT_OOS_MATCH_COUNT = 4773
FROZEN_PREPROCESSING_SPEC = PreprocessingSpec()

# Later T1 main-event matches inspected 2026-09-01 (OpenDota explorer).
# Not in the canonical corpus, not allowlisted, not used for tuning.
# The International 2026 is strictly after ``FROZEN_DEVELOPMENT_END``.
INSPECTED_LATER_T1_MAIN_EVENT_LEAGUE_ID = 19719
INSPECTED_LATER_T1_MAIN_EVENT_NAME = "The International 2026"
INSPECTED_LATER_T1_MAIN_EVENT_N = 147
INSPECTED_LATER_T1_MAIN_EVENT_START = datetime(2026, 8, 13, 3, 3, 26, tzinfo=UTC)
INSPECTED_LATER_T1_MAIN_EVENT_END = datetime(2026, 8, 23, 12, 8, 38, tzinfo=UTC)


class FrozenHoldoutEmptyError(TrainingDatasetError):
    """Raised when a holdout scoring path is requested with no later matches."""


def utc_datetime(value: object) -> datetime:
    """Convert a timezone-aware timestamp to UTC ``datetime``."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise TrainingDatasetError("holdout timestamps must be timezone-aware")
    converted = stamp.tz_convert("UTC")
    return datetime(
        converted.year,
        converted.month,
        converted.day,
        converted.hour,
        converted.minute,
        converted.second,
        converted.microsecond,
        tzinfo=UTC,
    )


def holdout_mask(start_time: pd.Series, development_end: datetime) -> pd.Series:
    """True for matches strictly after the frozen development boundary."""
    return start_time > pd.Timestamp(development_end)


def subset_model_ready_dataset(
    dataset: ModelReadyDataset, mask: pd.Series
) -> ModelReadyDataset:
    """Row subset of ``dataset`` with a fresh ``RangeIndex``."""
    return ModelReadyDataset(
        X=dataset.X.loc[mask].reset_index(drop=True).copy(),
        y=dataset.y.loc[mask].reset_index(drop=True).copy(),
        context=dataset.context.loc[mask].reset_index(drop=True).copy(),
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        identity_columns=dataset.identity_columns,
    )


def development_end_from_slice8_frame(
    dataset: ModelReadyDataset,
    *,
    config: WalkForwardConfig | None = None,
) -> datetime:
    """Latest timestamp used by expanding-window TEST on ``dataset``.

    With the default 5-block layout this is also ``max(start_time)``.
    """
    folds = resolve_walk_forward_folds(dataset, config=config)
    return utc_datetime(folds[-1].test_end)


def assert_development_frame_excludes_holdout(
    context: pd.DataFrame,
    *,
    development_end: datetime = FROZEN_DEVELOPMENT_END,
) -> None:
    """Refuse a Slice 7/8-style frame that already contains holdout matches."""
    if context.empty:
        raise TrainingDatasetError("development frame is empty")
    latest = utc_datetime(context["start_time"].max())
    if latest > utc_datetime(development_end):
        raise TrainingDatasetError(
            "development/OOS frame includes matches after the frozen "
            f"holdout boundary {utc_datetime(development_end).isoformat()}; "
            "those matches are untouched holdout and must not be used to "
            "retune thresholds, gates, C, or preprocessing"
        )


@dataclass(frozen=True)
class HoldoutInventory:
    """Later-match census. No predictions or metrics."""

    n: int
    start: datetime | None
    end: datetime | None
    n_leagues: int
    n_game_versions: int
    match_ids: tuple[int, ...]


def inventory_holdout(
    context: pd.DataFrame, *, development_end: datetime
) -> HoldoutInventory:
    """Count later matches in ``context`` without scoring them."""
    later = context.loc[holdout_mask(context["start_time"], development_end)]
    if later.empty:
        return HoldoutInventory(
            n=0,
            start=None,
            end=None,
            n_leagues=0,
            n_game_versions=0,
            match_ids=(),
        )
    ordered = later.sort_values(
        ["start_time", MATCH_ID_COLUMN], kind="stable"
    )
    n_versions = 0
    if "game_version_id" in ordered.columns:
        n_versions = int(ordered["game_version_id"].nunique(dropna=True))
    n_leagues = 0
    if "league_id" in ordered.columns:
        n_leagues = int(ordered["league_id"].nunique(dropna=True))
    return HoldoutInventory(
        n=len(ordered),
        start=utc_datetime(ordered["start_time"].min()),
        end=utc_datetime(ordered["start_time"].max()),
        n_leagues=n_leagues,
        n_game_versions=n_versions,
        match_ids=tuple(int(value) for value in ordered[MATCH_ID_COLUMN]),
    )


@dataclass(frozen=True)
class FrozenHoldoutSplit:
    """Development TRAIN/VAL plus untouched later holdout."""

    train: DatasetPartition
    validation: DatasetPartition
    holdout: DatasetPartition
    train_end: datetime
    validation_end: datetime
    development_end: datetime


def resolve_frozen_holdout_split(
    dataset: ModelReadyDataset,
    *,
    development_end: datetime,
    config: WalkForwardConfig | None = None,
    require_holdout: bool = False,
) -> FrozenHoldoutSplit:
    """Nested development split; holdout is strictly later matches.

    ``C`` and preprocessing may be fit on TRAIN and chosen on VAL.
    Holdout rows never enter those partitions.
    """
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    end = utc_datetime(development_end)
    start_time = dataset.context["start_time"]
    development_mask = ~holdout_mask(start_time, end)
    holdout = _partition(dataset, holdout_mask(start_time, end))
    if require_holdout and len(holdout) == 0:
        raise FrozenHoldoutEmptyError(
            "no professional matches after "
            f"{end.isoformat()}; holdout scoring is blocked"
        )
    development = start_time[development_mask]
    if development.empty:
        raise ChronologicalSplitError(
            "frozen holdout split has an empty development partition"
        )
    past_end = pd.Timestamp(end)
    train_end = utc_datetime(
        _train_end_within_past(
            dataset.context["start_time"],
            past_end,
            resolved.train_fraction_of_past,
        )
    )
    train_mask = start_time <= pd.Timestamp(train_end)
    validation_mask = (start_time > pd.Timestamp(train_end)) & (
        start_time <= past_end
    )
    train = _partition(dataset, train_mask)
    validation = _partition(dataset, validation_mask)
    if len(train) == 0 or len(validation) == 0:
        raise ChronologicalSplitError(
            "frozen holdout nested split produced an empty train or "
            "validation partition"
        )
    if not train.context["start_time"].max() < validation.context["start_time"].min():
        raise ChronologicalSplitError(
            "frozen holdout train/validation partitions overlap in time"
        )
    if len(holdout) and not (
        validation.context["start_time"].max() < holdout.context["start_time"].min()
    ):
        raise ChronologicalSplitError(
            "frozen holdout validation/holdout partitions overlap in time"
        )
    if len(holdout) and not utc_datetime(holdout.context["start_time"].min()) > end:
        raise ChronologicalSplitError(
            "holdout is not strictly after the frozen development boundary"
        )
    return FrozenHoldoutSplit(
        train=train,
        validation=validation,
        holdout=holdout,
        train_end=train_end,
        validation_end=end,
        development_end=end,
    )


@dataclass(frozen=True)
class FrozenHoldoutProtocol:
    """Recorded boundary, frozen spec, and later-match inventory.

    ``evaluated`` is always False for ``record_frozen_holdout_protocol``.
    """

    candidate_spec: BlockAblationSpec
    reference_spec: BlockAblationSpec
    preprocessing_spec: PreprocessingSpec
    regularization_candidates: tuple[float, ...]
    development_end: datetime
    development_start: datetime
    n_development: int
    n_development_oos: int
    holdout: HoldoutInventory
    canonical_later: HoldoutInventory
    n_slice8_post_draft_matches: int
    evaluated: bool = False


def _oos_match_count(folds: tuple[WalkForwardFold, ...]) -> int:
    ids: set[int] = set()
    for fold in folds:
        ids.update(int(value) for value in fold.test.context[MATCH_ID_COLUMN])
    return len(ids)


def _canonical_later_inventory(
    store: FeatureDuckDBConnection, *, development_end: datetime
) -> HoldoutInventory:
    frame = store.sql(
        f"""
        SELECT match_id, start_time, league_id, game_version_id
        FROM {MATCHES_VIEW}
        """
    ).df()
    return inventory_holdout(frame, development_end=development_end)


def record_frozen_holdout_protocol(
    store: FeatureDuckDBConnection,
    *,
    config: WalkForwardConfig | None = None,
    development_end: datetime | None = None,
    require_recorded_census: bool = False,
) -> FrozenHoldoutProtocol:
    """Assemble the Slice 8 frame, record the freeze, inventory later matches.

    Does not fit models, select ``C``, or score holdout rows.
    """
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    assembly = build_slice8_model_ready_dataset(store)
    assert_slice7_slice8_identity(assembly.slice7, assembly, config=resolved)
    extra = set(SLICE8_INTERACTION_COLUMNS)
    if extra & set(SLICE9_CANDIDATE_SPEC.feature_columns):
        raise TrainingDatasetError(
            "frozen Career spec must not include Slice 8 interaction columns"
        )
    if tuple(SLICE9_CANDIDATE_SPEC.feature_columns) != ELO_PLUS_PLAYER_HERO_COLUMNS:
        raise TrainingDatasetError(
            "frozen candidate must be Elo + unconditional Career Player × Hero"
        )

    recorded_end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    start_time = assembly.dataset.context["start_time"]
    development_mask = ~holdout_mask(start_time, recorded_end)
    development = subset_model_ready_dataset(assembly.dataset, development_mask)
    if len(development) == 0:
        raise TrainingDatasetError(
            "no development matches at or before "
            f"{recorded_end.isoformat()}"
        )
    folds = resolve_walk_forward_folds(development, config=resolved)
    frame_end = development_end_from_slice8_frame(development, config=resolved)
    latest_development = utc_datetime(development.context["start_time"].max())
    if latest_development != recorded_end:
        raise TrainingDatasetError(
            "Slice 8 development subset max start_time "
            f"{latest_development.isoformat()} does not match the recorded "
            f"boundary {recorded_end.isoformat()}"
        )
    if frame_end != recorded_end:
        raise TrainingDatasetError(
            "Slice 8 last-fold test_end "
            f"{frame_end.isoformat()} does not match the recorded boundary "
            f"{recorded_end.isoformat()}"
        )
    if require_recorded_census:
        n_oos = _oos_match_count(folds)
        if len(development) != FROZEN_DEVELOPMENT_MATCH_COUNT:
            raise TrainingDatasetError(
                "development match count "
                f"{len(development)} != recorded census "
                f"{FROZEN_DEVELOPMENT_MATCH_COUNT}"
            )
        if n_oos != FROZEN_DEVELOPMENT_OOS_MATCH_COUNT:
            raise TrainingDatasetError(
                "development OOS match count "
                f"{n_oos} != recorded census "
                f"{FROZEN_DEVELOPMENT_OOS_MATCH_COUNT}"
            )

    holdout = inventory_holdout(
        assembly.dataset.context, development_end=recorded_end
    )
    canonical_later = _canonical_later_inventory(
        store, development_end=recorded_end
    )
    return FrozenHoldoutProtocol(
        candidate_spec=SLICE9_CANDIDATE_SPEC,
        reference_spec=SLICE9_REFERENCE_SPEC,
        preprocessing_spec=FROZEN_PREPROCESSING_SPEC,
        regularization_candidates=REGULARIZATION_CANDIDATES,
        development_end=recorded_end,
        development_start=utc_datetime(development.context["start_time"].min()),
        n_development=len(development),
        n_development_oos=_oos_match_count(folds),
        holdout=holdout,
        canonical_later=canonical_later,
        n_slice8_post_draft_matches=assembly.n_post_draft_matches,
        evaluated=False,
    )
