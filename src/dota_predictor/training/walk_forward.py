"""Expanding walk-forward evaluation for post-draft block ablation.

The holdout split in ``training.split`` is a single train/validation/test
cut. This module layers expanding-window folds on the same
``ModelReadyDataset`` without changing that contract.

Each fold:
* uses only matches strictly before the test window for fitting
* takes a trailing slice of that past as validation for ``C``
* evaluates once on the next time block

Equal-``start_time`` matches stay in one partition. Paired log-loss
deltas are spec minus Elo on the same test matches (negative = spec
better). Game-version breakdowns are diagnostics on those OOS rows,
not a second split.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from dota_predictor.training.dataset import ModelReadyDataset
from dota_predictor.training.evaluation import (
    _fit_logistic,
    _select_regularization,
    evaluate_predictor,
)
from dota_predictor.training.feature_sets import (
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    BlockAblationSpec,
)
from dota_predictor.training.logistic_model import LogisticRegressionConfig
from dota_predictor.training.metrics import evaluate_probabilities, per_sample_log_loss
from dota_predictor.training.split import ChronologicalSplitError, DatasetPartition

__all__ = [
    "DEFAULT_WALK_FORWARD_CONFIG",
    "ELO_BLOCK_SPEC_NAME",
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardReport",
    "resolve_walk_forward_folds",
    "run_post_draft_walk_forward",
]

ELO_BLOCK_SPEC_NAME = "logistic_elo_only"


@dataclass(frozen=True)
class WalkForwardConfig:
    """Expanding-window fold layout.

    ``n_blocks`` consecutive time blocks of roughly equal match count.
    Fold ``k`` trains on blocks ``1..k`` (split into train + trailing
    validation) and tests on block ``k+1``. ``train_fraction_of_past``
    is the share of the pre-test mass used for fitting; the remainder
    is validation. The default ``0.70 / 0.85`` matches the holdout's
    70:15 split of that pre-test mass.
    """

    n_blocks: int = 5
    train_fraction_of_past: float = 0.70 / 0.85

    def __post_init__(self) -> None:
        if self.n_blocks < 2:
            raise ChronologicalSplitError("n_blocks must be >= 2")
        if not (0.0 < self.train_fraction_of_past < 1.0):
            raise ChronologicalSplitError(
                "train_fraction_of_past must be in (0, 1)"
            )


DEFAULT_WALK_FORWARD_CONFIG = WalkForwardConfig()


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window fold with nested validation for ``C``."""

    fold_id: int
    train: DatasetPartition
    validation: DatasetPartition
    test: DatasetPartition
    train_end: datetime
    validation_end: datetime
    test_end: datetime


@dataclass
class WalkForwardReport:
    """Walk-forward block ablation on one assembled post-draft matrix."""

    preprocessing_spec: object
    config: WalkForwardConfig
    specs: tuple[BlockAblationSpec, ...]
    folds: tuple[WalkForwardFold, ...]
    selected_C: pd.DataFrame
    fold_metrics: pd.DataFrame
    pooled_metrics: pd.DataFrame
    version_breakdown: pd.DataFrame
    version_fold_counts: pd.DataFrame
    oos_predictions: pd.DataFrame


def _partition(dataset: ModelReadyDataset, mask: pd.Series) -> DatasetPartition:
    return DatasetPartition(
        X=dataset.X.loc[mask].reset_index(drop=True),
        y=dataset.y.loc[mask].reset_index(drop=True),
        context=dataset.context.loc[mask].reset_index(drop=True),
    )


def _timestamp_cuts(start_time: pd.Series, n_blocks: int) -> list[pd.Timestamp]:
    counts = start_time.value_counts().sort_index()
    cumulative = counts.cumsum()
    total = int(cumulative.iloc[-1])
    values = cumulative.to_numpy()
    ends: list[pd.Timestamp] = []
    prev_idx = -1
    for block in range(n_blocks):
        target = ((block + 1) / n_blocks) * total
        idx = int(np.searchsorted(values, target, side="left"))
        idx = min(idx, len(cumulative) - 1)
        if idx <= prev_idx:
            idx = min(prev_idx + 1, len(cumulative) - 1)
        ends.append(cumulative.index[idx])
        prev_idx = idx
    ends[-1] = cumulative.index[-1]
    if len(set(ends)) != n_blocks:
        raise ChronologicalSplitError(
            "cannot resolve walk-forward blocks: start_time grouping is "
            f"too coarse for n_blocks={n_blocks}, n={total}"
        )
    for earlier, later in itertools.pairwise(ends):
        if not earlier < later:
            raise ChronologicalSplitError(
                "walk-forward block ends are not strictly increasing"
            )
    return ends


def _train_end_within_past(
    start_time: pd.Series, past_end: pd.Timestamp, train_fraction_of_past: float
) -> pd.Timestamp:
    past = start_time[start_time <= past_end]
    counts = past.value_counts().sort_index()
    cumulative = counts.cumsum()
    total = int(cumulative.iloc[-1])
    target = train_fraction_of_past * total
    idx = int(np.searchsorted(cumulative.to_numpy(), target, side="left"))
    idx = min(idx, len(cumulative) - 1)
    if idx == len(cumulative) - 1:
        idx -= 1
    if idx < 0:
        raise ChronologicalSplitError(
            "cannot split the pre-test window into non-empty train and "
            "validation partitions"
        )
    train_end = cumulative.index[idx]
    if not train_end < past_end:
        raise ChronologicalSplitError(
            "walk-forward validation would be empty for this past window"
        )
    return train_end


def resolve_walk_forward_folds(
    dataset: ModelReadyDataset,
    *,
    config: WalkForwardConfig | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Build expanding train/validation/test folds on ``dataset``.

    Block boundaries land on observed ``start_time`` values. Fold ``k``
    (1-based) tests block ``k+1`` and never uses that block, or any
    later block, for fitting or ``C`` selection.
    """
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    start_time = dataset.context["start_time"]
    block_ends = _timestamp_cuts(start_time, resolved.n_blocks)

    folds: list[WalkForwardFold] = []
    for fold_id, (past_end, test_end) in enumerate(
        itertools.pairwise(block_ends), start=1
    ):
        train_end = _train_end_within_past(
            start_time, past_end, resolved.train_fraction_of_past
        )
        train_mask = start_time <= train_end
        validation_mask = (start_time > train_end) & (start_time <= past_end)
        test_mask = (start_time > past_end) & (start_time <= test_end)
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
                    f"walk-forward fold {fold_id} produced an empty {name} "
                    "partition"
                )
        if not (
            train.context["start_time"].max()
            < validation.context["start_time"].min()
        ):
            raise ChronologicalSplitError(
                f"walk-forward fold {fold_id} train/validation overlap"
            )
        if not (
            validation.context["start_time"].max()
            < test.context["start_time"].min()
        ):
            raise ChronologicalSplitError(
                f"walk-forward fold {fold_id} validation/test overlap"
            )
        if test.context["start_time"].max() > test_end:
            raise ChronologicalSplitError(
                f"walk-forward fold {fold_id} test window exceeds test_end"
            )
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train=train,
                validation=validation,
                test=test,
                train_end=train_end,
                validation_end=past_end,
                test_end=test_end,
            )
        )
    return tuple(folds)


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    n = int(values.size)
    mean = float(values.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan")
    se = float(values.std(ddof=1) / np.sqrt(n))
    return mean, se


def _paired_stats(
    y: pd.Series, p_spec: pd.Series, p_elo: pd.Series
) -> dict[str, float]:
    spec_ll = per_sample_log_loss(y, p_spec)
    elo_ll = per_sample_log_loss(y, p_elo)
    delta = spec_ll - elo_ll
    mean_delta, se_delta = _mean_se(delta)
    return {
        "mean_delta_vs_elo": mean_delta,
        "se_delta_vs_elo": se_delta,
        "frac_better_than_elo": float((delta < 0).mean()),
        "elo_log_loss": float(elo_ll.mean()),
        "log_loss": float(spec_ll.mean()),
        "n": float(len(delta)),
    }


def run_post_draft_walk_forward(
    dataset: ModelReadyDataset,
    *,
    config: WalkForwardConfig | None = None,
) -> WalkForwardReport:
    """Walk-forward the six predefined Elo + draft-block specs.

    Same ``PreprocessingSpec`` and per-spec validation ``C`` selection
    as ``run_post_draft_block_ablation``, repeated inside each fold.
    History is not recomputed; column subsets of the assembled matrix
    are used as-is.
    """
    from dota_predictor.training.preprocessing import PreprocessingSpec

    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    specs = POST_DRAFT_BLOCK_ABLATION_SPECS
    spec_by_name = {spec.name: spec for spec in specs}
    if ELO_BLOCK_SPEC_NAME not in spec_by_name:
        raise ValueError(
            f"walk-forward requires {ELO_BLOCK_SPEC_NAME!r} among the "
            "predefined block specs"
        )

    preprocessing_spec = PreprocessingSpec()
    folds = resolve_walk_forward_folds(dataset, config=resolved)

    selected_c_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        fitted = {}
        selected_c: dict[str, float] = {}
        for spec in specs:
            c, _reg = _select_regularization(
                fold.train, fold.validation, spec.feature_columns
            )
            selected_c[spec.name] = c
            selected_c_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": spec.name,
                    "label": spec.label,
                    "C": c,
                    "n_train": len(fold.train),
                    "n_validation": len(fold.validation),
                    "n_test": len(fold.test),
                }
            )
            model = _fit_logistic(
                fold.train,
                spec.feature_columns,
                config=LogisticRegressionConfig(
                    C=c, preprocessing=preprocessing_spec
                ),
            )
            fitted[spec.name] = model

        evaluations = {
            spec.name: evaluate_predictor(spec.name, fold.test, fitted[spec.name])
            for spec in specs
        }
        elo_preds = evaluations[ELO_BLOCK_SPEC_NAME].predictions
        elo_p = elo_preds.p_radiant_win.reset_index(drop=True)
        y = elo_preds.y_true.reset_index(drop=True)
        context = elo_preds.context.reset_index(drop=True)

        for spec in specs:
            evaluation = evaluations[spec.name]
            p = evaluation.predictions.p_radiant_win.reset_index(drop=True)
            paired = _paired_stats(y, p, elo_p)
            metrics = evaluation.metrics
            fold_metric_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": spec.name,
                    "label": spec.label,
                    "n_features": len(spec.feature_columns),
                    "C": selected_c[spec.name],
                    "n_train": len(fold.train),
                    "n_validation": len(fold.validation),
                    "n_test": metrics.n_samples,
                    "log_loss": metrics.log_loss,
                    "brier_score": metrics.brier_score,
                    "accuracy_at_0.5": metrics.accuracy_at_0_5,
                    "roc_auc": metrics.roc_auc,
                    "ece": metrics.expected_calibration_error,
                    "mean_delta_vs_elo": paired["mean_delta_vs_elo"],
                    "se_delta_vs_elo": paired["se_delta_vs_elo"],
                    "frac_better_than_elo": paired["frac_better_than_elo"],
                    "train_end": fold.train_end,
                    "validation_end": fold.validation_end,
                    "test_end": fold.test_end,
                }
            )
            spec_ll = per_sample_log_loss(y, p)
            elo_ll = per_sample_log_loss(y, elo_p)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold_id": fold.fold_id,
                        "model": spec.name,
                        "label": spec.label,
                        "match_id": context["match_id"].to_numpy(),
                        "start_time": context["start_time"].to_numpy(),
                        "game_version_id": context["game_version_id"].to_numpy(),
                        "y_true": y.to_numpy(),
                        "p_spec": p.to_numpy(),
                        "p_elo": elo_p.to_numpy(),
                        "sample_log_loss": spec_ll,
                        "delta_vs_elo": spec_ll - elo_ll,
                    }
                )
            )

    oos_predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    selected_C = pd.DataFrame(selected_c_rows)
    pooled_metrics = _pooled_metrics(oos_predictions, specs)
    version_breakdown = _version_breakdown(oos_predictions, specs)
    version_fold_counts = (
        oos_predictions.loc[
            oos_predictions["model"] == ELO_BLOCK_SPEC_NAME,
            ["fold_id", "game_version_id"],
        ]
        .value_counts()
        .rename("n")
        .reset_index()
        .sort_values(["fold_id", "game_version_id"])
        .reset_index(drop=True)
    )

    return WalkForwardReport(
        preprocessing_spec=preprocessing_spec,
        config=resolved,
        specs=specs,
        folds=folds,
        selected_C=selected_C,
        fold_metrics=fold_metrics,
        pooled_metrics=pooled_metrics,
        version_breakdown=version_breakdown,
        version_fold_counts=version_fold_counts,
        oos_predictions=oos_predictions,
    )


def _pooled_metrics(
    oos: pd.DataFrame, specs: tuple[BlockAblationSpec, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        subset = oos.loc[oos["model"] == spec.name]
        y = subset["y_true"]
        p = subset["p_spec"]
        p_elo = subset["p_elo"]
        paired = _paired_stats(y, p, p_elo)
        full = evaluate_probabilities(y, p)
        rows.append(
            {
                "model": spec.name,
                "label": spec.label,
                "n_features": len(spec.feature_columns),
                "n": int(paired["n"]),
                "log_loss": full.log_loss,
                "brier_score": full.brier_score,
                "accuracy_at_0.5": full.accuracy_at_0_5,
                "roc_auc": full.roc_auc,
                "ece": full.expected_calibration_error,
                "mean_delta_vs_elo": paired["mean_delta_vs_elo"],
                "se_delta_vs_elo": paired["se_delta_vs_elo"],
                "frac_better_than_elo": paired["frac_better_than_elo"],
            }
        )
    return pd.DataFrame(rows)


def _version_breakdown(
    oos: pd.DataFrame, specs: tuple[BlockAblationSpec, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    versions = sorted(
        v for v in oos["game_version_id"].dropna().unique().tolist()
    )
    missing = oos["game_version_id"].isna().any()
    version_keys: list[object] = list(versions)
    if missing:
        version_keys.append(pd.NA)

    for version in version_keys:
        if pd.isna(version):
            version_mask = oos["game_version_id"].isna()
        else:
            version_mask = oos["game_version_id"] == version
        for spec in specs:
            subset = oos.loc[version_mask & (oos["model"] == spec.name)]
            if subset.empty:
                continue
            paired = _paired_stats(
                subset["y_true"], subset["p_spec"], subset["p_elo"]
            )
            rows.append(
                {
                    "game_version_id": version,
                    "model": spec.name,
                    "label": spec.label,
                    "n": int(paired["n"]),
                    "log_loss": paired["log_loss"],
                    "elo_log_loss": paired["elo_log_loss"],
                    "mean_delta_vs_elo": paired["mean_delta_vs_elo"],
                    "se_delta_vs_elo": paired["se_delta_vs_elo"],
                    "frac_better_than_elo": paired["frac_better_than_elo"],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["game_version_id", "model"], kind="stable").reset_index(
        drop=True
    )
