"""Slice 16: walk-forward evaluation of frozen player farming vs Elo.

One incremental test. The candidate is the Slice 15 named spec
``logistic_elo_plus_player_farming`` (Elo + ``mean_farming_shrunk_b_diff``).
The reference is the frozen logistic Elo spec ``logistic_elo_only``.

Does not redesign candidate B, ``k``, player history, or the five-player
Radiant − Dire mean. Does not add ``prior_n``, raw farming means, or
interactions. Does not change production ``FEATURE_COLUMNS`` or Slice 9
specs.

Holdout policy
--------------
The frozen Slice 9 holdout remains reserved until an explicit later
promotion of a production spec. Slice 9's one-shot scorer is locked to
Career Player × Hero. Later research slices (10–15) evaluate on the
development frame only. Slice 16 follows that policy: expanding-window
OOS on ``start_time <= FROZEN_DEVELOPMENT_END``. The holdout is not
scored and is not used to choose ``C`` or preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.player_farming_comparison import (
    FARMING_CAUSAL_B_COLUMN,
    MATCH_ID_COLUMN,
    PLAYER_FARMING_COMPARISON_METRIC_COLUMNS,
    PLAYER_FARMING_FEATURE_COLUMNS,
    PLAYER_FARMING_REQUIRED_COLUMNS,
    PLAYER_FARMING_STATE_FEATURE_COLUMNS,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
    build_pre_draft_snapshot,
)
from dota_predictor.features.team_elo import (
    DEFAULT_ELO_CONFIG,
    TEAM_ELO_DELTA_COLUMN,
    TEAM_ELO_FEATURE_COLUMNS,
    EloConfig,
)
from dota_predictor.training.dataset import ModelReadyDataset, TrainingDatasetError
from dota_predictor.training.evaluation import _fit_logistic
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_PLAYER_FARMING_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
    SLICE15_CANDIDATE_SPEC,
    SLICE15_CANDIDATE_SPEC_NAME,
    SLICE15_FROZEN_SPECS,
    SLICE15_REFERENCE_SPEC,
    SLICE15_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.logistic_model import (
    LogisticRegressionConfig,
    standardized_coefficients,
)
from dota_predictor.training.metrics import (
    bootstrap_mean_ci,
    evaluate_probabilities,
    per_sample_brier,
    per_sample_log_loss,
)
from dota_predictor.training.player_farming_comparison import (
    build_player_farming_comparison,
)
from dota_predictor.training.player_farming_state import (
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    SLICE14_STATE_COLUMNS,
)
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    _jsonable_value,
    _numeric,
    _pearson,
    _std,
    restrict_development,
)
from dota_predictor.training.preprocessing import MISSINGNESS_INDICATOR_SUFFIX
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    FROZEN_DEVELOPMENT_MATCH_COUNT,
    FROZEN_DEVELOPMENT_OOS_MATCH_COUNT,
    FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES,
    FROZEN_HOLDOUT_BOOTSTRAP_SEED,
    assert_development_frame_excludes_holdout,
    utc_datetime,
)
from dota_predictor.training.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG,
    ELO_BLOCK_SPEC_NAME,
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardReport,
    run_post_draft_walk_forward,
)

__all__ = [
    "ABS_FARMING_BUCKET_COUNT",
    "CLASSIFICATION_A",
    "CLASSIFICATION_B",
    "CLASSIFICATION_C",
    "FARMING_FEATURE_COLUMN",
    "HOLDOUT_POLICY",
    "PREDICTION_MOVE_THRESHOLDS",
    "SLICE16_BOOTSTRAP_RESAMPLES",
    "SLICE16_BOOTSTRAP_SEED",
    "SLICE16_CANDIDATE_SPEC",
    "SLICE16_CANDIDATE_SPEC_NAME",
    "SLICE16_FROZEN_SPECS",
    "SLICE16_REFERENCE_SPEC",
    "SLICE16_REFERENCE_SPEC_NAME",
    "Slice16Assembly",
    "Slice16BenchmarkReport",
    "assign_abs_quantile_bucket",
    "build_slice16_model_ready_dataset",
    "classify_slice16",
    "run_slice16_player_farming_benchmark",
    "slice16_report_to_jsonable",
]


# Repository names. User-facing "logistic_elo" is ``logistic_elo_only``.
SLICE16_REFERENCE_SPEC_NAME = SLICE15_REFERENCE_SPEC_NAME
SLICE16_CANDIDATE_SPEC_NAME = SLICE15_CANDIDATE_SPEC_NAME
SLICE16_REFERENCE_SPEC = SLICE15_REFERENCE_SPEC
SLICE16_CANDIDATE_SPEC = SLICE15_CANDIDATE_SPEC
SLICE16_FROZEN_SPECS = SLICE15_FROZEN_SPECS
FARMING_FEATURE_COLUMN = PLAYER_FARMING_FEATURE_COLUMNS[0]

SLICE16_BOOTSTRAP_RESAMPLES = FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES
SLICE16_BOOTSTRAP_SEED = FROZEN_HOLDOUT_BOOTSTRAP_SEED
ABS_FARMING_BUCKET_COUNT = 4
PREDICTION_MOVE_THRESHOLDS: tuple[float, ...] = (0.01, 0.02, 0.05)

HOLDOUT_POLICY = (
    "development_oos_only: frozen Slice 9 holdout remains reserved until "
    "an explicit later promotion of a production spec. Slice 9's one-shot "
    "scorer is locked to Career Player × Hero. Feature/model choices for "
    "farming were frozen in Slices 13–15, but this research candidate is "
    "not that promoted spec, so Slice 16 scores development walk-forward "
    "OOS only (start_time <= FROZEN_DEVELOPMENT_END)."
)

CLASSIFICATION_A = "A — consistent incremental signal beyond Elo"
CLASSIFICATION_B = (
    "B — weak/mixed incremental signal; retain state but do not promote"
)
CLASSIFICATION_C = "C — no useful incremental win signal beyond Elo"

_FORBIDDEN_MODEL_COLUMNS: tuple[str, ...] = (
    "farming_prior_n",
    "mean_farming_prior_n",
    "mean_farming_prior_n_diff",
    "min_farming_prior_n",
    "min_farming_prior_n_diff",
    "hero_id",
    "position",
    "position_number",
    "num_last_hits",
    "duration_seconds",
    "last_hits_per_minute",
    FARMING_CAUSAL_B_COLUMN,
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class Slice16Assembly:
    """Elo + frozen farming comparison on the development frame."""

    dataset: ModelReadyDataset
    n_snapshot_matches: int
    n_farming_comparison_matches: int
    n_missing_farming_join: int
    n_holdout_excluded: int


@dataclass
class Slice16BenchmarkReport:
    """Walk-forward paired evaluation of frozen farming vs logistic Elo."""

    development_end: datetime
    holdout_policy: str
    n_development_matches: int
    n_holdout_excluded: int
    n_oos: int
    frozen_k: float
    assembly: Slice16Assembly
    walk_forward: WalkForwardReport
    fold_table: pd.DataFrame
    pooled: pd.DataFrame
    bootstrap: dict[str, object]
    coefficients: pd.DataFrame
    fold_sign_consistency: dict[str, object]
    farming_elo_correlation: float
    farming_distribution: pd.DataFrame
    elo_by_farming_bucket: pd.DataFrame
    prediction_movement: dict[str, object]
    magnitude_buckets: pd.DataFrame
    calibration: dict[str, object]
    integrity: dict[str, object]
    classification: str
    classification_rationale: str


def build_slice16_model_ready_dataset(
    store: FeatureDuckDBConnection,
    *,
    elo_config: EloConfig | None = None,
    development_end: datetime | None = None,
) -> Slice16Assembly:
    """PRE_DRAFT Elo plus the frozen farming comparison, development only.

    Farming state is attached with ``development_end`` set so later
    holdout box scores cannot enter the residualizer or player history.
    The current match still never contributes last hits, duration,
    position, hero, or result to ``mean_farming_shrunk_b_diff``.
    """
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    resolved_elo = elo_config if elo_config is not None else DEFAULT_ELO_CONFIG
    snapshot = build_pre_draft_snapshot(store, elo_config=resolved_elo)
    full = snapshot.to_frame()
    n_snapshot = int(full[MATCH_ID_COLUMN].nunique())
    stamp = pd.to_datetime(full["start_time"], utc=True)
    holdout_n = int((stamp > pd.Timestamp(end)).sum())
    development = restrict_development(full, development_end=end)
    comparison = build_player_farming_comparison(
        store, k=FROZEN_SHRINKAGE_K, elo_config=resolved_elo, development_end=end
    )
    farming = comparison.loc[:, [MATCH_ID_COLUMN, *PLAYER_FARMING_FEATURE_COLUMNS]]
    merged = development.merge(
        farming,
        on=MATCH_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    ordered = merged.sort_values(
        ["start_time", MATCH_ID_COLUMN], kind="stable"
    ).reset_index(drop=True)
    feature_columns = ELO_PLUS_PLAYER_FARMING_COLUMNS
    missing = [name for name in feature_columns if name not in ordered.columns]
    if missing:
        raise TrainingDatasetError(
            "Slice 16 assembly is missing required columns: "
            f"{missing}"
        )
    overlap = set(FEATURE_COLUMNS) & set(PLAYER_FARMING_FEATURE_COLUMNS)
    if overlap:
        raise TrainingDatasetError(
            "farming candidate columns must not appear in FEATURE_COLUMNS: "
            f"{sorted(overlap)}"
        )
    forbidden = [name for name in _FORBIDDEN_MODEL_COLUMNS if name in feature_columns]
    if forbidden:
        raise TrainingDatasetError(
            "Slice 16 feature matrix must not include "
            f"{forbidden}"
        )
    dataset = ModelReadyDataset(
        X=ordered[list(feature_columns)].copy(),
        y=ordered[TARGET_COLUMN].copy(),
        context=ordered[list(IDENTITY_COLUMNS)].copy(),
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        identity_columns=IDENTITY_COLUMNS,
    )
    assert_development_frame_excludes_holdout(
        dataset.context, development_end=end
    )
    present_ids = set(farming[MATCH_ID_COLUMN].tolist()) if len(farming) else set()
    n_missing_join = int((~ordered[MATCH_ID_COLUMN].isin(present_ids)).sum())
    return Slice16Assembly(
        dataset=dataset,
        n_snapshot_matches=n_snapshot,
        n_farming_comparison_matches=int(farming[MATCH_ID_COLUMN].nunique())
        if len(farming)
        else 0,
        n_missing_farming_join=n_missing_join,
        n_holdout_excluded=holdout_n,
    )


def _quantile_edges(values: np.ndarray, n_buckets: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    probs = np.linspace(0.0, 1.0, n_buckets + 1)
    edges = np.quantile(finite, probs)
    unique = np.unique(edges)
    if unique.size < 2:
        lo = float(unique[0])
        return np.array([lo, lo], dtype=float)
    return unique.astype(float)


def assign_abs_quantile_bucket(value: float, edges: np.ndarray) -> str:
    """Map ``|farming diff|`` onto precomputed quantile edges.

    Edges come from the OOS ``|mean_farming_shrunk_b_diff|`` distribution
    only. They are not chosen from prediction metrics.
    """
    if not np.isfinite(value):
        return "NULL"
    if edges.size < 2:
        return "Q1"
    n_bins = int(edges.size - 1)
    if edges[0] == edges[-1]:
        return "Q1"
    idx = int(np.searchsorted(edges, float(value), side="right") - 1)
    idx = min(max(idx, 0), n_bins - 1)
    return f"Q{idx + 1}"


def _paired_oos(walk_forward: WalkForwardReport) -> pd.DataFrame:
    oos = walk_forward.oos_predictions
    reference = oos.loc[oos["model"] == SLICE16_REFERENCE_SPEC_NAME].copy()
    candidate = oos.loc[oos["model"] == SLICE16_CANDIDATE_SPEC_NAME].copy()
    ref_ids = reference[MATCH_ID_COLUMN].to_numpy()
    cand_ids = candidate[MATCH_ID_COLUMN].to_numpy()
    if ref_ids.shape != cand_ids.shape or not np.array_equal(ref_ids, cand_ids):
        raise TrainingDatasetError(
            "reference and candidate OOS match_ids are not identical"
        )
    paired = candidate.rename(
        columns={
            "p_spec": "p_candidate",
            "sample_log_loss": "candidate_log_loss",
            "delta_vs_elo": "delta_log_loss",
        }
    )
    paired["p_reference"] = reference["p_spec"].to_numpy()
    paired["reference_log_loss"] = reference["sample_log_loss"].to_numpy()
    paired["prediction_delta"] = paired["p_candidate"] - paired["p_reference"]
    paired["reference_brier"] = per_sample_brier(
        paired["y_true"], paired["p_reference"]
    )
    paired["candidate_brier"] = per_sample_brier(
        paired["y_true"], paired["p_candidate"]
    )
    return paired.reset_index(drop=True)


def _attach_features(
    paired: pd.DataFrame, dataset: ModelReadyDataset
) -> pd.DataFrame:
    features = dataset.context[[MATCH_ID_COLUMN]].copy()
    features[TEAM_ELO_DELTA_COLUMN] = dataset.X[TEAM_ELO_DELTA_COLUMN].to_numpy()
    features[FARMING_FEATURE_COLUMN] = dataset.X[FARMING_FEATURE_COLUMN].to_numpy()
    return paired.merge(
        features, on=MATCH_ID_COLUMN, how="left", validate="one_to_one"
    )


def _farming_coefficient(table: pd.DataFrame) -> float:
    rows = table.loc[table["feature"] == FARMING_FEATURE_COLUMN]
    if rows.empty:
        return float("nan")
    return float(rows["coefficient"].iloc[0])


def _fold_coefficients(
    folds: tuple[WalkForwardFold, ...], selected_c: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in folds:
        c_lookup = selected_c.loc[selected_c["fold_id"] == fold.fold_id]
        for spec in SLICE16_FROZEN_SPECS:
            c = float(c_lookup.loc[c_lookup["model"] == spec.name, "C"].iloc[0])
            model = _fit_logistic(
                fold.train,
                spec.feature_columns,
                config=LogisticRegressionConfig(C=c),
            )
            coef = standardized_coefficients(model)
            farming = _farming_coefficient(coef)
            train_farming = _numeric(fold.train.X[FARMING_FEATURE_COLUMN])
            finite = train_farming.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": spec.name,
                    "C": c,
                    "farming_coefficient": farming
                    if spec.name == SLICE16_CANDIDATE_SPEC_NAME
                    else float("nan"),
                    "train_farming_mean": (
                        float(finite.mean()) if finite.size else float("nan")
                    ),
                    "train_farming_std": (
                        _std(finite) if finite.size else float("nan")
                    ),
                    "n_train": len(fold.train),
                    "missingness_indicator_in_model": any(
                        str(name).endswith(MISSINGNESS_INDICATOR_SUFFIX)
                        for name in coef["feature"].tolist()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _period(partition_times: pd.Series) -> tuple[datetime, datetime]:
    return utc_datetime(partition_times.min()), utc_datetime(partition_times.max())


def _fold_table(
    walk_forward: WalkForwardReport,
    paired: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cand_coef = coefficients.loc[
        coefficients["model"] == SLICE16_CANDIDATE_SPEC_NAME
    ].set_index("fold_id")
    for fold in walk_forward.folds:
        subset = paired.loc[paired["fold_id"] == fold.fold_id]
        train_start, train_end = _period(fold.train.context["start_time"])
        val_start, val_end = _period(fold.validation.context["start_time"])
        test_start, test_end = _period(fold.test.context["start_time"])
        y = subset["y_true"]
        ref_metrics = evaluate_probabilities(y, subset["p_reference"])
        cand_metrics = evaluate_probabilities(y, subset["p_candidate"])
        delta = per_sample_log_loss(y, subset["p_candidate"]) - per_sample_log_loss(
            y, subset["p_reference"]
        )
        coef_row = cand_coef.loc[fold.fold_id]
        rows.append(
            {
                "fold": fold.fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": val_start,
                "validation_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
                "n_matches": len(subset),
                "reference_log_loss": ref_metrics.log_loss,
                "candidate_log_loss": cand_metrics.log_loss,
                "paired_delta_log_loss": float(delta.mean()),
                "reference_brier": ref_metrics.brier_score,
                "candidate_brier": cand_metrics.brier_score,
                "farming_coefficient": float(coef_row["farming_coefficient"]),
                "train_farming_mean": float(coef_row["train_farming_mean"]),
                "train_farming_std": float(coef_row["train_farming_std"]),
                "C_reference": float(
                    walk_forward.selected_C.loc[
                        (walk_forward.selected_C["fold_id"] == fold.fold_id)
                        & (
                            walk_forward.selected_C["model"]
                            == SLICE16_REFERENCE_SPEC_NAME
                        ),
                        "C",
                    ].iloc[0]
                ),
                "C_candidate": float(coef_row["C"]),
            }
        )
    return pd.DataFrame(rows)


def _pooled_row(paired: pd.DataFrame) -> pd.DataFrame:
    y = paired["y_true"]
    ref = evaluate_probabilities(y, paired["p_reference"])
    cand = evaluate_probabilities(y, paired["p_candidate"])
    delta = paired["candidate_log_loss"] - paired["reference_log_loss"]
    return pd.DataFrame(
        [
            {
                "n": len(paired),
                "reference_log_loss": ref.log_loss,
                "candidate_log_loss": cand.log_loss,
                "paired_delta_log_loss": float(delta.mean()),
                "reference_brier": ref.brier_score,
                "candidate_brier": cand.brier_score,
                "reference_ece": ref.expected_calibration_error,
                "candidate_ece": cand.expected_calibration_error,
                "reference_roc_auc": ref.roc_auc,
                "candidate_roc_auc": cand.roc_auc,
            }
        ]
    )


def _bootstrap_delta(delta: np.ndarray) -> dict[str, object]:
    observed = float(delta.mean()) if delta.size else float("nan")
    ci_lo, ci_hi = bootstrap_mean_ci(
        delta,
        n_resamples=SLICE16_BOOTSTRAP_RESAMPLES,
        random_state=SLICE16_BOOTSTRAP_SEED,
    )
    rng = np.random.default_rng(SLICE16_BOOTSTRAP_SEED)
    if delta.size == 0:
        frac_negative = float("nan")
        boot_mean = float("nan")
    else:
        draws = rng.choice(
            delta, size=(SLICE16_BOOTSTRAP_RESAMPLES, delta.size), replace=True
        )
        means = draws.mean(axis=1)
        frac_negative = float((means < 0.0).mean())
        boot_mean = float(means.mean())
    return {
        "observed_delta": observed,
        "bootstrap_mean_delta": boot_mean,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "frac_delta_negative": frac_negative,
        "n_resamples": SLICE16_BOOTSTRAP_RESAMPLES,
        "seed": SLICE16_BOOTSTRAP_SEED,
        "grouping": "oos_match",
    }


def _sign(value: float) -> int:
    if not np.isfinite(value) or value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def _fold_sign_consistency(fold_table: pd.DataFrame) -> dict[str, object]:
    deltas = fold_table["paired_delta_log_loss"].to_numpy(dtype=float)
    coefs = fold_table["farming_coefficient"].to_numpy(dtype=float)
    n_neg = int((deltas < 0.0).sum())
    n_pos = int((deltas > 0.0).sum())
    n_zero = int((deltas == 0.0).sum()) + int((~np.isfinite(deltas)).sum())
    coef_signs = [_sign(value) for value in coefs]
    n_pos_coef = coef_signs.count(1)
    n_neg_coef = coef_signs.count(-1)
    unique_nonzero = {sign for sign in coef_signs if sign != 0}
    return {
        "n_folds": len(fold_table),
        "n_folds_delta_negative": n_neg,
        "n_folds_delta_positive": n_pos,
        "n_folds_delta_zero": n_zero,
        "delta_by_fold": [float(value) for value in deltas],
        "coefficient_by_fold": [float(value) for value in coefs],
        "coefficient_sign_by_fold": coef_signs,
        "n_folds_coefficient_positive": n_pos_coef,
        "n_folds_coefficient_negative": n_neg_coef,
        "coefficient_sign_stable": len(unique_nonzero) <= 1,
    }


def _distribution(values: pd.Series, *, column: str) -> pd.DataFrame:
    finite = _numeric(values).to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        stats = {
            "column": column,
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
        return pd.DataFrame([stats])
    return pd.DataFrame(
        [
            {
                "column": column,
                "n": int(finite.size),
                "mean": float(finite.mean()),
                "std": _std(finite),
                "median": float(np.median(finite)),
                "p05": float(np.quantile(finite, 0.05)),
                "p25": float(np.quantile(finite, 0.25)),
                "p75": float(np.quantile(finite, 0.75)),
                "p95": float(np.quantile(finite, 0.95)),
                "min": float(finite.min()),
                "max": float(finite.max()),
            }
        ]
    )


def _prediction_movement(delta: np.ndarray) -> dict[str, object]:
    abs_delta = np.abs(delta)
    finite = abs_delta[np.isfinite(abs_delta)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean_abs": float("nan"),
            "median_abs": float("nan"),
            "p90_abs": float("nan"),
            "p95_abs": float("nan"),
            "max_abs": float("nan"),
            "frac_ge_1pp": float("nan"),
            "frac_ge_2pp": float("nan"),
            "frac_ge_5pp": float("nan"),
        }
    return {
        "n": int(finite.size),
        "mean_abs": float(finite.mean()),
        "median_abs": float(np.median(finite)),
        "p90_abs": float(np.quantile(finite, 0.90)),
        "p95_abs": float(np.quantile(finite, 0.95)),
        "max_abs": float(finite.max()),
        "frac_ge_1pp": float((finite >= PREDICTION_MOVE_THRESHOLDS[0]).mean()),
        "frac_ge_2pp": float((finite >= PREDICTION_MOVE_THRESHOLDS[1]).mean()),
        "frac_ge_5pp": float((finite >= PREDICTION_MOVE_THRESHOLDS[2]).mean()),
    }


def _magnitude_buckets(paired: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    abs_diff = np.abs(_numeric(paired[FARMING_FEATURE_COLUMN]).to_numpy(dtype=float))
    edges = _quantile_edges(abs_diff, ABS_FARMING_BUCKET_COUNT)
    labels = [
        assign_abs_quantile_bucket(value, edges) for value in abs_diff
    ]
    work = paired.copy()
    work["abs_farming_diff"] = abs_diff
    work["abs_farming_bucket"] = labels
    rows: list[dict[str, object]] = []
    order = sorted(
        {label for label in labels if label != "NULL"},
        key=lambda item: int(item[1:]) if item.startswith("Q") else 99,
    )
    if "NULL" in labels:
        order.append("NULL")
    for label in order:
        subset = work.loc[work["abs_farming_bucket"] == label]
        if subset.empty:
            continue
        y = subset["y_true"]
        ref_ll = float(per_sample_log_loss(y, subset["p_reference"]).mean())
        cand_ll = float(per_sample_log_loss(y, subset["p_candidate"]).mean())
        rows.append(
            {
                "bucket": label,
                "n": len(subset),
                "mean_abs_farming_diff": float(
                    np.nanmean(subset["abs_farming_diff"].to_numpy(dtype=float))
                ),
                "reference_log_loss": ref_ll,
                "candidate_log_loss": cand_ll,
                "paired_delta_log_loss": cand_ll - ref_ll,
            }
        )
    return pd.DataFrame(rows), edges


def _elo_by_farming_bucket(paired: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    abs_diff = np.abs(_numeric(paired[FARMING_FEATURE_COLUMN]).to_numpy(dtype=float))
    labels = [
        assign_abs_quantile_bucket(value, edges) for value in abs_diff
    ]
    work = paired.copy()
    work["abs_farming_bucket"] = labels
    rows: list[dict[str, object]] = []
    order = sorted(
        {label for label in labels if label != "NULL"},
        key=lambda item: int(item[1:]) if item.startswith("Q") else 99,
    )
    for label in order:
        subset = work.loc[work["abs_farming_bucket"] == label]
        elo = _numeric(subset[TEAM_ELO_DELTA_COLUMN]).to_numpy(dtype=float)
        finite = elo[np.isfinite(elo)]
        rows.append(
            {
                "bucket": label,
                "n": len(subset),
                "mean_elo_diff": float(finite.mean()) if finite.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _calibration(paired: pd.DataFrame) -> dict[str, object]:
    ref = evaluate_probabilities(paired["y_true"], paired["p_reference"])
    cand = evaluate_probabilities(paired["y_true"], paired["p_candidate"])
    return {
        "reference_brier": ref.brier_score,
        "candidate_brier": cand.brier_score,
        "reference_ece": ref.expected_calibration_error,
        "candidate_ece": cand.expected_calibration_error,
        "reference_bins": ref.calibration_table,
        "candidate_bins": cand.calibration_table,
        "intercept_slope": None,
        "intercept_slope_available": False,
    }


def classify_slice16(
    *,
    pooled_delta: float,
    ci_low: float,
    ci_high: float,
    frac_delta_negative: float,
    n_folds_delta_negative: int,
    n_folds_delta_positive: int,
    n_folds: int,
    coefficient_sign_stable: bool,
    reference_brier: float,
    candidate_brier: float,
    reference_ece: float,
    candidate_ece: float,
    mean_abs_prediction_delta: float,
) -> tuple[str, str]:
    """Map frozen-protocol diagnostics onto A / B / C. Does not retune."""
    ci_excludes_zero_negative = (
        np.isfinite(ci_high) and np.isfinite(pooled_delta) and ci_high < 0.0
    )
    ci_excludes_zero_positive = (
        np.isfinite(ci_low) and np.isfinite(pooled_delta) and ci_low > 0.0
    )
    majority_negative = (
        n_folds >= 2
        and n_folds_delta_negative >= max(2, n_folds - 1)
        and n_folds_delta_positive == 0
    )
    mixed_folds = n_folds_delta_negative > 0 and n_folds_delta_positive > 0
    isolated_fold = (
        n_folds >= 3
        and n_folds_delta_negative == 1
        and pooled_delta < 0.0
    )
    brier_worse = (
        np.isfinite(candidate_brier)
        and np.isfinite(reference_brier)
        and candidate_brier > reference_brier + 1e-4
    )
    ece_much_worse = (
        np.isfinite(candidate_ece)
        and np.isfinite(reference_ece)
        and candidate_ece > reference_ece + 0.02
    )
    tiny_movement = (
        np.isfinite(mean_abs_prediction_delta) and mean_abs_prediction_delta < 0.002
    )

    if (
        np.isfinite(pooled_delta)
        and pooled_delta < 0.0
        and ci_excludes_zero_negative
        and majority_negative
        and not isolated_fold
        and coefficient_sign_stable
        and not brier_worse
        and not ece_much_worse
    ):
        return (
            CLASSIFICATION_A,
            (
                "Pooled paired Δ log loss is negative, the bootstrap CI "
                "excludes zero on the improvement side, folds agree, the "
                "farming coefficient sign is stable, and calibration does "
                "not degrade."
            ),
        )
    if (
        (np.isfinite(pooled_delta) and pooled_delta > 0.0 and ci_excludes_zero_positive)
        or (n_folds >= 2 and n_folds_delta_positive == n_folds)
        or (
            np.isfinite(pooled_delta)
            and pooled_delta >= 0.0
            and tiny_movement
            and (not np.isfinite(frac_delta_negative) or frac_delta_negative < 0.5)
        )
        or (brier_worse and ece_much_worse and pooled_delta >= 0.0)
    ):
        return (
            CLASSIFICATION_C,
            (
                "The candidate does not improve on Elo: pooled Δ is "
                "non-negative with evidence against improvement, folds do "
                "not support a gain, and/or prediction quality worsens."
            ),
        )
    reasons: list[str] = []
    if not np.isfinite(pooled_delta) or abs(pooled_delta) < 1e-5:
        reasons.append("pooled Δ is near zero")
    elif pooled_delta < 0.0 and not ci_excludes_zero_negative:
        reasons.append("pooled Δ is negative but the CI includes zero")
    elif pooled_delta > 0.0 and not ci_excludes_zero_positive:
        reasons.append("pooled Δ is positive but the CI includes zero")
    if mixed_folds:
        reasons.append("folds disagree on the sign of Δ")
    if isolated_fold:
        reasons.append("pooled improvement is concentrated in one fold")
    if not coefficient_sign_stable:
        reasons.append("farming coefficient sign is unstable")
    if brier_worse:
        reasons.append("candidate Brier is worse")
    if tiny_movement:
        reasons.append("prediction movement is tiny")
    if not reasons:
        reasons.append("the incremental effect is too small or uncertain to promote")
    return CLASSIFICATION_B, "; ".join(reasons) + "."


def _integrity(
    store: FeatureDuckDBConnection,
    assembly: Slice16Assembly,
    paired: pd.DataFrame,
) -> dict[str, object]:
    dataset = assembly.dataset
    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    ref_ids = paired[MATCH_ID_COLUMN].to_numpy()
    return {
        "holdout_policy": HOLDOUT_POLICY,
        "development_end": FROZEN_DEVELOPMENT_END.isoformat(),
        "frozen_shrinkage_k": FROZEN_SHRINKAGE_K,
        "frozen_candidate_b": FROZEN_CANDIDATE_B,
        "k_re_searched": False,
        "alternative_farming_features_searched": False,
        "holdout_scored": False,
        "holdout_used_for_c": False,
        "holdout_used_for_feature": False,
        "stratz_called": False,
        "reference_spec": SLICE16_REFERENCE_SPEC_NAME,
        "candidate_spec": SLICE16_CANDIDATE_SPEC_NAME,
        "reference_columns": list(SLICE16_REFERENCE_SPEC.feature_columns),
        "candidate_columns": list(SLICE16_CANDIDATE_SPEC.feature_columns),
        "reference_is_elo_only": list(SLICE16_REFERENCE_SPEC.feature_columns)
        == list(ELO_ONLY_FEATURE_COLUMNS)
        == list(TEAM_ELO_FEATURE_COLUMNS),
        "candidate_is_elo_plus_farming": list(
            SLICE16_CANDIDATE_SPEC.feature_columns
        )
        == list(ELO_PLUS_PLAYER_FARMING_COLUMNS),
        "candidate_excludes_prior_n": all(
            "prior_n" not in name for name in SLICE16_CANDIDATE_SPEC.feature_columns
        ),
        "forbidden_columns_absent_from_x": all(
            name not in dataset.X.columns for name in _FORBIDDEN_MODEL_COLUMNS
        ),
        "hero_id_in_required_columns": "hero_id" in PLAYER_FARMING_REQUIRED_COLUMNS,
        "position_in_required_columns": "position" in PLAYER_FARMING_REQUIRED_COLUMNS,
        "causal_b_in_state_feature_columns": (
            FARMING_CAUSAL_B_COLUMN in PLAYER_FARMING_STATE_FEATURE_COLUMNS
        ),
        "causal_b_in_comparison_columns": (
            FARMING_CAUSAL_B_COLUMN in PLAYER_FARMING_COMPARISON_METRIC_COLUMNS
        ),
        "state_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in SLICE14_STATE_COLUMNS
        ),
        "farming_in_feature_columns": FARMING_FEATURE_COLUMN in FEATURE_COLUMNS,
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_names": [spec.name for spec in SLICE9_FROZEN_SPECS],
        "slice9_reference_unchanged": (
            SLICE9_REFERENCE_SPEC_NAME == ELO_BLOCK_SPEC_NAME
        ),
        "slice9_candidate_unchanged": (
            SLICE9_CANDIDATE_SPEC_NAME == "logistic_elo_plus_player_hero"
        ),
        "slice9_candidate_excludes_farming": FARMING_FEATURE_COLUMN
        not in SLICE9_CANDIDATE_SPEC.feature_columns,
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "identical_oos_match_ids": True,
        "n_oos_match_ids": len(np.unique(ref_ids)),
        "population_matches_expected": (
            len(dataset) == FROZEN_DEVELOPMENT_MATCH_COUNT
        ),
        "oos_count_matches_frozen_frame": (
            len(paired) == FROZEN_DEVELOPMENT_OOS_MATCH_COUNT
        ),
        "n_holdout_excluded": assembly.n_holdout_excluded,
        "n_missing_farming_join": assembly.n_missing_farming_join,
        "team_aggregation": "arithmetic_five_player_side_mean",
        "orientation": "radiant_minus_dire",
        "cold_start": 0.0,
        "walk_forward_config_n_blocks": DEFAULT_WALK_FORWARD_CONFIG.n_blocks,
    }


def run_slice16_player_farming_benchmark(
    store: FeatureDuckDBConnection,
    *,
    config: WalkForwardConfig | None = None,
    elo_config: EloConfig | None = None,
    development_end: datetime | None = None,
) -> Slice16BenchmarkReport:
    """Walk-forward paired test of frozen farming vs logistic Elo."""
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    assembly = build_slice16_model_ready_dataset(
        store, elo_config=elo_config, development_end=end
    )
    walk_forward = run_post_draft_walk_forward(
        assembly.dataset,
        config=resolved,
        specs=SLICE16_FROZEN_SPECS,
    )
    paired = _attach_features(_paired_oos(walk_forward), assembly.dataset)
    coefficients = _fold_coefficients(walk_forward.folds, walk_forward.selected_C)
    fold_table = _fold_table(walk_forward, paired, coefficients)
    pooled = _pooled_row(paired)
    bootstrap = _bootstrap_delta(
        (paired["candidate_log_loss"] - paired["reference_log_loss"]).to_numpy(
            dtype=float
        )
    )
    consistency = _fold_sign_consistency(fold_table)
    farming_elo_corr = _pearson(
        paired[FARMING_FEATURE_COLUMN], paired[TEAM_ELO_DELTA_COLUMN]
    )
    farming_distribution = _distribution(
        paired[FARMING_FEATURE_COLUMN], column=FARMING_FEATURE_COLUMN
    )
    magnitude_buckets, edges = _magnitude_buckets(paired)
    elo_by_bucket = _elo_by_farming_bucket(paired, edges)
    movement = _prediction_movement(
        paired["prediction_delta"].to_numpy(dtype=float)
    )
    calibration = _calibration(paired)
    classification, rationale = classify_slice16(
        pooled_delta=float(pooled.iloc[0]["paired_delta_log_loss"]),
        ci_low=float(bootstrap["ci95_low"]),
        ci_high=float(bootstrap["ci95_high"]),
        frac_delta_negative=float(bootstrap["frac_delta_negative"]),
        n_folds_delta_negative=int(consistency["n_folds_delta_negative"]),
        n_folds_delta_positive=int(consistency["n_folds_delta_positive"]),
        n_folds=int(consistency["n_folds"]),
        coefficient_sign_stable=bool(consistency["coefficient_sign_stable"]),
        reference_brier=float(pooled.iloc[0]["reference_brier"]),
        candidate_brier=float(pooled.iloc[0]["candidate_brier"]),
        reference_ece=float(pooled.iloc[0]["reference_ece"]),
        candidate_ece=float(pooled.iloc[0]["candidate_ece"]),
        mean_abs_prediction_delta=float(movement["mean_abs"]),
    )
    integrity = _integrity(store, assembly, paired)
    integrity["abs_farming_bucket_edges"] = [float(edge) for edge in edges]
    return Slice16BenchmarkReport(
        development_end=end,
        holdout_policy=HOLDOUT_POLICY,
        n_development_matches=len(assembly.dataset),
        n_holdout_excluded=assembly.n_holdout_excluded,
        n_oos=len(paired),
        frozen_k=FROZEN_SHRINKAGE_K,
        assembly=assembly,
        walk_forward=walk_forward,
        fold_table=fold_table,
        pooled=pooled,
        bootstrap=bootstrap,
        coefficients=coefficients,
        fold_sign_consistency=consistency,
        farming_elo_correlation=farming_elo_corr,
        farming_distribution=farming_distribution,
        elo_by_farming_bucket=elo_by_bucket,
        prediction_movement=movement,
        magnitude_buckets=magnitude_buckets,
        calibration=calibration,
        integrity=integrity,
        classification=classification,
        classification_rationale=rationale,
    )


def slice16_report_to_jsonable(report: Slice16BenchmarkReport) -> dict[str, object]:
    """JSON-safe dump of the Slice 16 walk-forward report."""
    return {
        "development_end": report.development_end.isoformat(),
        "holdout_policy": report.holdout_policy,
        "n_development_matches": report.n_development_matches,
        "n_holdout_excluded": report.n_holdout_excluded,
        "n_oos": report.n_oos,
        "frozen_k": report.frozen_k,
        "fold_table": _jsonable_value(report.fold_table),
        "pooled": _jsonable_value(report.pooled),
        "bootstrap": _jsonable_value(report.bootstrap),
        "coefficients": _jsonable_value(report.coefficients),
        "fold_sign_consistency": _jsonable_value(report.fold_sign_consistency),
        "farming_elo_correlation": _jsonable_value(report.farming_elo_correlation),
        "farming_distribution": _jsonable_value(report.farming_distribution),
        "elo_by_farming_bucket": _jsonable_value(report.elo_by_farming_bucket),
        "prediction_movement": _jsonable_value(report.prediction_movement),
        "magnitude_buckets": _jsonable_value(report.magnitude_buckets),
        "calibration": {
            "reference_brier": report.calibration["reference_brier"],
            "candidate_brier": report.calibration["candidate_brier"],
            "reference_ece": report.calibration["reference_ece"],
            "candidate_ece": report.calibration["candidate_ece"],
            "reference_bins": _jsonable_value(report.calibration["reference_bins"]),
            "candidate_bins": _jsonable_value(report.calibration["candidate_bins"]),
            "intercept_slope_available": report.calibration[
                "intercept_slope_available"
            ],
        },
        "integrity": _jsonable_value(report.integrity),
        "classification": report.classification,
        "classification_rationale": report.classification_rationale,
        "n_missing_farming_join": report.assembly.n_missing_farming_join,
        "n_farming_comparison_matches": (
            report.assembly.n_farming_comparison_matches
        ),
    }
