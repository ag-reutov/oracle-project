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

This module records the protocol, inventories later matches, and runs
the one-shot holdout evaluation. ``record_frozen_holdout_protocol``
does not score. ``evaluate_frozen_holdout`` scores once, persists a
lock, and refuses later calls that would re-select ``C`` or redefine
the frozen experiment.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import MATCHES_VIEW, FeatureDuckDBConnection
from dota_predictor.features.player_hero_meta_comparison import MATCH_ID_COLUMN
from dota_predictor.training.dataset import ModelReadyDataset, TrainingDatasetError
from dota_predictor.training.evaluation import (
    REGULARIZATION_CANDIDATES,
    _fit_logistic,
    _select_regularization,
    evaluate_predictor,
)
from dota_predictor.training.feature_sets import (
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    SLICE8_INTERACTION_COLUMNS,
    SLICE8_MATCH_MEAN_CAREER_GAMES,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_REFERENCE_SPEC,
    SLICE9_REFERENCE_SPEC_NAME,
    BlockAblationSpec,
)
from dota_predictor.training.logistic_model import LogisticRegressionConfig
from dota_predictor.training.metrics import (
    EvaluationMetrics,
    bootstrap_mean_ci,
    per_sample_brier,
    per_sample_log_loss,
)
from dota_predictor.training.preprocessing import PreprocessingSpec
from dota_predictor.training.slice8_player_hero_gating import (
    EVIDENCE_BIN_ORDER,
    Slice8Assembly,
    assert_slice7_slice8_identity,
    assign_train_tertile_bin,
    build_slice8_model_ready_dataset,
    train_tertile_edges,
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
    "FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES",
    "FROZEN_HOLDOUT_BOOTSTRAP_SEED",
    "FROZEN_HOLDOUT_EVALUATION_FILENAME",
    "FROZEN_HOLDOUT_EXPECTED_LEAGUE_ID",
    "FROZEN_HOLDOUT_EXPECTED_N",
    "FROZEN_HOLDOUT_PREDICTIONS_FILENAME",
    "FROZEN_PREPROCESSING_SPEC",
    "INSPECTED_LATER_T1_MAIN_EVENT_END",
    "INSPECTED_LATER_T1_MAIN_EVENT_LEAGUE_ID",
    "INSPECTED_LATER_T1_MAIN_EVENT_N",
    "INSPECTED_LATER_T1_MAIN_EVENT_NAME",
    "INSPECTED_LATER_T1_MAIN_EVENT_START",
    "FrozenHoldoutAlreadyEvaluatedError",
    "FrozenHoldoutEmptyError",
    "FrozenHoldoutEvaluation",
    "FrozenHoldoutProtocol",
    "FrozenHoldoutSplit",
    "HoldoutInventory",
    "assert_development_frame_excludes_holdout",
    "assert_frozen_holdout_ready",
    "chronology_bins",
    "development_end_from_slice8_frame",
    "evaluate_frozen_holdout",
    "holdout_mask",
    "inventory_holdout",
    "load_frozen_holdout_eval_dir",
    "load_frozen_holdout_evaluation",
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

# One-shot holdout census. Must match the ingested TI 2026 main event.
FROZEN_HOLDOUT_EXPECTED_N = INSPECTED_LATER_T1_MAIN_EVENT_N
FROZEN_HOLDOUT_EXPECTED_LEAGUE_ID = INSPECTED_LATER_T1_MAIN_EVENT_LEAGUE_ID
FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES = 10_000
FROZEN_HOLDOUT_BOOTSTRAP_SEED = 0
FROZEN_HOLDOUT_EVALUATION_FILENAME = "evaluation.json"
FROZEN_HOLDOUT_PREDICTIONS_FILENAME = "predictions.parquet"
_DEFAULT_EVAL_DIR = Path("models") / "slice9_frozen_holdout"
CHRONOLOGY_BIN_ORDER: tuple[str, ...] = ("early", "middle", "late")
WINNER_SIDE_ORDER: tuple[str, ...] = ("Radiant", "Dire")


class FrozenHoldoutEmptyError(TrainingDatasetError):
    """Raised when a holdout scoring path is requested with no later matches."""


class FrozenHoldoutAlreadyEvaluatedError(TrainingDatasetError):
    """Raised when a second scoring run would re-fit or redefine the freeze."""

    def __init__(self, path: Path, payload: dict[str, Any] | None = None) -> None:
        self.path = path
        self.payload = payload or {}
        super().__init__(
            f"frozen holdout already evaluated at {path}; refusing to "
            "re-select C, refit preprocessing, or redefine the experiment"
        )


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
    ``evaluate_frozen_holdout`` returns a copy with ``evaluated=True``.
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


@dataclass(frozen=True)
class _FrozenHoldoutState:
    protocol: FrozenHoldoutProtocol
    assembly: Slice8Assembly
    development: ModelReadyDataset


def _assert_frozen_specification() -> None:
    extra = set(SLICE8_INTERACTION_COLUMNS)
    if extra & set(SLICE9_CANDIDATE_SPEC.feature_columns):
        raise TrainingDatasetError(
            "frozen Career spec must not include Slice 8 interaction columns"
        )
    if tuple(SLICE9_CANDIDATE_SPEC.feature_columns) != ELO_PLUS_PLAYER_HERO_COLUMNS:
        raise TrainingDatasetError(
            "frozen candidate must be Elo + unconditional Career Player × Hero"
        )
    if SLICE9_CANDIDATE_SPEC.name != SLICE9_CANDIDATE_SPEC_NAME:
        raise TrainingDatasetError("frozen candidate spec name drifted")
    if SLICE9_REFERENCE_SPEC.name != SLICE9_REFERENCE_SPEC_NAME:
        raise TrainingDatasetError("frozen reference spec name drifted")
    if tuple(SLICE9_REFERENCE_SPEC.feature_columns) == ELO_PLUS_PLAYER_HERO_COLUMNS:
        raise TrainingDatasetError(
            "frozen reference must be Elo only, not Career Player × Hero"
        )


def _prepare_frozen_holdout(
    store: FeatureDuckDBConnection,
    *,
    config: WalkForwardConfig | None = None,
    development_end: datetime | None = None,
    require_recorded_census: bool = False,
) -> _FrozenHoldoutState:
    """Assemble the Slice 8 frame, record the freeze, inventory later matches."""
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    assembly = build_slice8_model_ready_dataset(store)
    assert_slice7_slice8_identity(assembly.slice7, assembly, config=resolved)
    _assert_frozen_specification()

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
    protocol = FrozenHoldoutProtocol(
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
    return _FrozenHoldoutState(
        protocol=protocol, assembly=assembly, development=development
    )


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
    return _prepare_frozen_holdout(
        store,
        config=config,
        development_end=development_end,
        require_recorded_census=require_recorded_census,
    ).protocol


def load_frozen_holdout_eval_dir(*, root: Path | None = None) -> Path:
    """Directory for the one-shot evaluation lock and predictions.

    ``SLICE9_HOLDOUT_EVAL_DIR`` overrides the default
    ``models/slice9_frozen_holdout`` under ``root``.
    """
    raw = os.environ.get("SLICE9_HOLDOUT_EVAL_DIR", "").strip()
    if raw:
        return Path(raw)
    base = root if root is not None else Path.cwd()
    return base / _DEFAULT_EVAL_DIR


def chronology_bins(start_time: pd.Series) -> pd.Series:
    """Equal-count early/middle/late bins over the provided timestamps.

    Descriptive only. Does not use outcomes. Ties follow ``start_time``
    order already present on the series.
    """
    n = len(start_time)
    if n == 0:
        return pd.Series(dtype="object")
    order = np.argsort(start_time.to_numpy(), kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    labels = np.asarray(CHRONOLOGY_BIN_ORDER)
    return pd.Series(labels[(ranks * 3) // n], index=start_time.index)


def load_frozen_holdout_evaluation(output_dir: Path) -> dict[str, Any]:
    """Read the persisted one-shot lock. Raises if it is missing."""
    path = output_dir / FROZEN_HOLDOUT_EVALUATION_FILENAME
    if not path.is_file():
        raise TrainingDatasetError(f"frozen holdout evaluation lock not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_frozen_holdout_ready(
    protocol: FrozenHoldoutProtocol,
    split: FrozenHoldoutSplit,
    *,
    expected_holdout_n: int | None = None,
    expected_holdout_league_id: int | None = None,
    require_recorded_census: bool = False,
) -> None:
    """Refuse scoring unless the freeze, census, and holdout identity hold."""
    if protocol.candidate_spec != SLICE9_CANDIDATE_SPEC:
        raise TrainingDatasetError("candidate spec does not match the Slice 9 freeze")
    if protocol.reference_spec != SLICE9_REFERENCE_SPEC:
        raise TrainingDatasetError("reference spec does not match the Slice 9 freeze")
    if protocol.preprocessing_spec != FROZEN_PREPROCESSING_SPEC:
        raise TrainingDatasetError("preprocessing spec does not match the Slice 9 freeze")
    if protocol.regularization_candidates != REGULARIZATION_CANDIDATES:
        raise TrainingDatasetError("C grid does not match the Slice 9 freeze")
    extra = set(SLICE8_INTERACTION_COLUMNS) & set(
        protocol.candidate_spec.feature_columns
    )
    if extra:
        raise TrainingDatasetError(
            "frozen Career spec must not include Slice 8 interaction columns"
        )
    if protocol.development_end != split.development_end:
        raise TrainingDatasetError("split development_end drifted from the protocol")
    if require_recorded_census:
        if protocol.n_development != FROZEN_DEVELOPMENT_MATCH_COUNT:
            raise TrainingDatasetError(
                "development match count "
                f"{protocol.n_development} != recorded census "
                f"{FROZEN_DEVELOPMENT_MATCH_COUNT}"
            )
        if protocol.n_development_oos != FROZEN_DEVELOPMENT_OOS_MATCH_COUNT:
            raise TrainingDatasetError(
                "development OOS match count "
                f"{protocol.n_development_oos} != recorded census "
                f"{FROZEN_DEVELOPMENT_OOS_MATCH_COUNT}"
            )
    if expected_holdout_n is not None:
        if protocol.holdout.n != expected_holdout_n:
            raise TrainingDatasetError(
                f"holdout N={protocol.holdout.n} != expected {expected_holdout_n}"
            )
        if protocol.canonical_later.n != expected_holdout_n:
            raise TrainingDatasetError(
                "canonical later N="
                f"{protocol.canonical_later.n} != expected {expected_holdout_n}"
            )
        if len(split.holdout) != expected_holdout_n:
            raise TrainingDatasetError(
                f"split holdout N={len(split.holdout)} != expected {expected_holdout_n}"
            )
    if len(split.holdout) == 0:
        raise FrozenHoldoutEmptyError(
            "no professional matches after "
            f"{protocol.development_end.isoformat()}; holdout scoring is blocked"
        )
    holdout_times = split.holdout.context["start_time"]
    if not utc_datetime(holdout_times.min()) > protocol.development_end:
        raise TrainingDatasetError(
            "holdout is not strictly after the frozen development boundary"
        )
    if expected_holdout_league_id is not None:
        leagues = {
            int(value) for value in split.holdout.context["league_id"].tolist()
        }
        if leagues != {expected_holdout_league_id}:
            raise TrainingDatasetError(
                "holdout leagues "
                f"{sorted(leagues)} != {{{expected_holdout_league_id}}}"
            )


def _metrics_payload(metrics: EvaluationMetrics) -> dict[str, float | int]:
    return {
        "log_loss": metrics.log_loss,
        "brier_score": metrics.brier_score,
        "roc_auc": metrics.roc_auc,
        "ece": metrics.expected_calibration_error,
        "accuracy_at_0.5": metrics.accuracy_at_0_5,
        "n": metrics.n_samples,
    }


def _diagnostic_rows(
    predictions: pd.DataFrame, *, column: str, order: tuple[str, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in order:
        subset = predictions.loc[predictions[column] == label]
        if subset.empty:
            rows.append(
                {
                    column: label,
                    "n": 0,
                    "candidate_log_loss": float("nan"),
                    "elo_log_loss": float("nan"),
                    "mean_delta_log_loss": float("nan"),
                    "n_candidate_better": 0,
                }
            )
            continue
        rows.append(
            {
                column: label,
                "n": len(subset),
                "candidate_log_loss": float(subset["candidate_log_loss"].mean()),
                "elo_log_loss": float(subset["elo_log_loss"].mean()),
                "mean_delta_log_loss": float(subset["delta_log_loss"].mean()),
                "n_candidate_better": int((subset["delta_log_loss"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FrozenHoldoutEvaluation:
    """One-shot holdout scores. Not a production-model decision."""

    protocol: FrozenHoldoutProtocol
    split: FrozenHoldoutSplit
    selected_C: dict[str, float]
    regularization_comparison: pd.DataFrame
    reference_metrics: EvaluationMetrics
    candidate_metrics: EvaluationMetrics
    paired_delta_log_loss: float
    paired_delta_brier: float
    n_candidate_better_log_loss: int
    mean_paired_log_loss_diff: float
    median_paired_log_loss_diff: float
    bootstrap_delta_log_loss_ci95: tuple[float, float]
    predictions: pd.DataFrame
    chronology: pd.DataFrame
    winner_side: pd.DataFrame
    career_evidence: pd.DataFrame
    output_dir: Path | None = None


def evaluate_frozen_holdout(
    store: FeatureDuckDBConnection,
    *,
    config: WalkForwardConfig | None = None,
    development_end: datetime | None = None,
    require_recorded_census: bool = False,
    expected_holdout_n: int | None = None,
    expected_holdout_league_id: int | None = None,
    output_dir: Path | None = None,
) -> FrozenHoldoutEvaluation:
    """Score the frozen Elo vs Career specs on later matches once.

    Fits preprocessing and selects ``C`` on development TRAIN/VAL only.
    Holdout rows are not used for those decisions. A lock in
    ``output_dir`` makes a later call refuse to re-fit.
    """
    if output_dir is not None:
        lock_path = output_dir / FROZEN_HOLDOUT_EVALUATION_FILENAME
        if lock_path.is_file():
            raise FrozenHoldoutAlreadyEvaluatedError(
                lock_path, json.loads(lock_path.read_text(encoding="utf-8"))
            )

    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    if require_recorded_census:
        if expected_holdout_n is None:
            expected_holdout_n = FROZEN_HOLDOUT_EXPECTED_N
        if expected_holdout_league_id is None:
            expected_holdout_league_id = FROZEN_HOLDOUT_EXPECTED_LEAGUE_ID

    state = _prepare_frozen_holdout(
        store,
        config=resolved,
        development_end=development_end,
        require_recorded_census=require_recorded_census,
    )
    protocol = state.protocol
    split = resolve_frozen_holdout_split(
        state.assembly.dataset,
        development_end=protocol.development_end,
        config=resolved,
        require_holdout=True,
    )
    assert_frozen_holdout_ready(
        protocol,
        split,
        expected_holdout_n=expected_holdout_n,
        expected_holdout_league_id=expected_holdout_league_id,
        require_recorded_census=require_recorded_census,
    )
    train_ids = {int(v) for v in split.train.context[MATCH_ID_COLUMN]}
    val_ids = {int(v) for v in split.validation.context[MATCH_ID_COLUMN]}
    holdout_ids = {int(v) for v in split.holdout.context[MATCH_ID_COLUMN]}
    if train_ids & holdout_ids or val_ids & holdout_ids:
        raise TrainingDatasetError("holdout rows leaked into TRAIN/VAL")

    specs: tuple[BlockAblationSpec, ...] = (
        protocol.reference_spec,
        protocol.candidate_spec,
    )
    selected_c: dict[str, float] = {}
    comparison_frames: list[pd.DataFrame] = []
    fitted = {}
    for spec in specs:
        c, comparison = _select_regularization(
            split.train, split.validation, spec.feature_columns
        )
        selected_c[spec.name] = c
        labeled = comparison.copy()
        labeled.insert(0, "model", spec.name)
        comparison_frames.append(labeled)
        fitted[spec.name] = _fit_logistic(
            split.train,
            spec.feature_columns,
            config=LogisticRegressionConfig(
                C=c, preprocessing=protocol.preprocessing_spec
            ),
        )

    reference_eval = evaluate_predictor(
        protocol.reference_spec.name, split.holdout, fitted[protocol.reference_spec.name]
    )
    candidate_eval = evaluate_predictor(
        protocol.candidate_spec.name, split.holdout, fitted[protocol.candidate_spec.name]
    )
    y = reference_eval.predictions.y_true.reset_index(drop=True)
    p_elo = reference_eval.predictions.p_radiant_win.reset_index(drop=True)
    p_cand = candidate_eval.predictions.p_radiant_win.reset_index(drop=True)
    context = reference_eval.predictions.context.reset_index(drop=True)
    elo_ll = per_sample_log_loss(y, p_elo)
    cand_ll = per_sample_log_loss(y, p_cand)
    delta_ll = cand_ll - elo_ll
    elo_brier = per_sample_brier(y, p_elo)
    cand_brier = per_sample_brier(y, p_cand)
    delta_brier = cand_brier - elo_brier
    ci_lo, ci_hi = bootstrap_mean_ci(
        delta_ll,
        n_resamples=FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES,
        random_state=FROZEN_HOLDOUT_BOOTSTRAP_SEED,
    )

    holdout_x = split.holdout.X.reset_index(drop=True)
    career_values = holdout_x[SLICE8_MATCH_MEAN_CAREER_GAMES]
    q_low, q_high = train_tertile_edges(
        split.train.X[SLICE8_MATCH_MEAN_CAREER_GAMES]
    )
    career_bins = [
        assign_train_tertile_bin(
            None if pd.isna(value) else float(value), q_low=q_low, q_high=q_high
        )
        for value in career_values
    ]
    winner = np.where(y.to_numpy(dtype=int) == 1, "Radiant", "Dire")
    chrono = chronology_bins(context["start_time"])

    predictions = pd.DataFrame(
        {
            MATCH_ID_COLUMN: context[MATCH_ID_COLUMN].to_numpy(),
            "start_time": context["start_time"].to_numpy(),
            "league_id": context["league_id"].to_numpy(),
            "y_true": y.to_numpy(),
            "p_elo": p_elo.to_numpy(),
            "p_candidate": p_cand.to_numpy(),
            "elo_log_loss": elo_ll,
            "candidate_log_loss": cand_ll,
            "delta_log_loss": delta_ll,
            "elo_brier": elo_brier,
            "candidate_brier": cand_brier,
            "delta_brier": delta_brier,
            "chronology_bin": chrono.to_numpy(),
            "winner_side": winner,
            "career_evidence_bin": career_bins,
        }
    )
    chronology = _diagnostic_rows(
        predictions, column="chronology_bin", order=CHRONOLOGY_BIN_ORDER
    )
    winner_side = _diagnostic_rows(
        predictions, column="winner_side", order=WINNER_SIDE_ORDER
    )
    career_evidence = _diagnostic_rows(
        predictions, column="career_evidence_bin", order=EVIDENCE_BIN_ORDER
    )

    scored = replace(protocol, evaluated=True)
    report = FrozenHoldoutEvaluation(
        protocol=scored,
        split=split,
        selected_C=selected_c,
        regularization_comparison=pd.concat(comparison_frames, ignore_index=True),
        reference_metrics=reference_eval.metrics,
        candidate_metrics=candidate_eval.metrics,
        paired_delta_log_loss=float(delta_ll.mean()),
        paired_delta_brier=float(delta_brier.mean()),
        n_candidate_better_log_loss=int((delta_ll < 0).sum()),
        mean_paired_log_loss_diff=float(delta_ll.mean()),
        median_paired_log_loss_diff=float(np.median(delta_ll)),
        bootstrap_delta_log_loss_ci95=(ci_lo, ci_hi),
        predictions=predictions,
        chronology=chronology,
        winner_side=winner_side,
        career_evidence=career_evidence,
        output_dir=output_dir,
    )
    if output_dir is not None:
        _persist_frozen_holdout_evaluation(report, output_dir)
    return report


def _persist_frozen_holdout_evaluation(
    report: FrozenHoldoutEvaluation, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = report.protocol
    payload = {
        "evaluated": True,
        "production_decision": None,
        "development_end": protocol.development_end.isoformat(),
        "n_development": protocol.n_development,
        "n_development_oos": protocol.n_development_oos,
        "n_holdout": protocol.holdout.n,
        "holdout_start": protocol.holdout.start.isoformat()
        if protocol.holdout.start
        else None,
        "holdout_end": protocol.holdout.end.isoformat()
        if protocol.holdout.end
        else None,
        "holdout_league_ids": sorted(
            {int(v) for v in report.predictions["league_id"].tolist()}
        ),
        "candidate_spec": protocol.candidate_spec.name,
        "reference_spec": protocol.reference_spec.name,
        "candidate_columns": list(protocol.candidate_spec.feature_columns),
        "reference_columns": list(protocol.reference_spec.feature_columns),
        "preprocessing": asdict(protocol.preprocessing_spec),
        "C_grid": list(protocol.regularization_candidates),
        "selected_C": report.selected_C,
        "regularization_comparison": report.regularization_comparison.to_dict(
            orient="records"
        ),
        "reference_metrics": _metrics_payload(report.reference_metrics),
        "candidate_metrics": _metrics_payload(report.candidate_metrics),
        "paired_delta_log_loss": report.paired_delta_log_loss,
        "paired_delta_brier": report.paired_delta_brier,
        "n_candidate_better_log_loss": report.n_candidate_better_log_loss,
        "mean_paired_log_loss_diff": report.mean_paired_log_loss_diff,
        "median_paired_log_loss_diff": report.median_paired_log_loss_diff,
        "bootstrap_delta_log_loss_ci95": list(report.bootstrap_delta_log_loss_ci95),
        "bootstrap_n": FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": FROZEN_HOLDOUT_BOOTSTRAP_SEED,
        "n_train": len(report.split.train),
        "n_validation": len(report.split.validation),
        "holdout_match_ids": [int(v) for v in report.predictions[MATCH_ID_COLUMN]],
        "chronology": report.chronology.to_dict(orient="records"),
        "winner_side": report.winner_side.to_dict(orient="records"),
        "career_evidence": report.career_evidence.to_dict(orient="records"),
        "career_evidence_note": (
            "LOW/MEDIUM/HIGH bins use development TRAIN tertiles of "
            f"{SLICE8_MATCH_MEAN_CAREER_GAMES}; diagnostic only"
        ),
    }
    (output_dir / FROZEN_HOLDOUT_EVALUATION_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report.predictions.to_parquet(output_dir / FROZEN_HOLDOUT_PREDICTIONS_FILENAME)
