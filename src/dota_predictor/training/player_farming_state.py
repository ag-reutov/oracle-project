"""Slice 14: leakage-safe historical player farming state.

Research only. Candidate B's *definition* is frozen from Slice 13:

    LHPM = num_last_hits / (duration_seconds / 60)
    B = z(OLS residual of LHPM ~ position dummies + duration_seconds)

This module does **not** reuse Slice 13's globally fitted residualizer.
Nuisance estimates (position+duration coefficients and residual scale)
are fit on ``start_time < T`` only. Same-timestamp rows are mutually
blind. Insufficient residualizer history yields NULL B rather than a
silent full-development fallback.

Player state for appearance ``M`` uses that player's eligible prior B
observations with ``H.start_time < M.start_time``. Shrinkage is toward
zero because causal B is standardized around the population baseline:

    shrunk_b = n / (n + k) * prior_mean_b
    shrunk_b = 0 when n = 0

``k`` is chosen on an earlier development timestamp split by how well
prior state predicts the *next causal B*, never from match outcomes.
This module does not add production features, does not train a win
model, and does not touch the frozen holdout.
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
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.features.team_elo import DEFAULT_ELO_CONFIG, EloConfig
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    _jsonable_value,
    _numeric,
    _pearson,
    _spearman,
    _std,
    build_player_performance_frame,
    explicit_position_mask,
    first_half_second_half_correlation,
    per_minute,
    restrict_development,
    slope_coefficient,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    FROZEN_DEVELOPMENT_MATCH_COUNT,
    FROZEN_HOLDOUT_EXPECTED_N,
    utc_datetime,
)
from dota_predictor.training.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG,
    _train_end_within_past,
)

__all__ = [
    "CAUSAL_B_COLUMN",
    "EXPECTED_DEVELOPMENT_PLAYER_ROWS",
    "FROZEN_CANDIDATE_B",
    "FROZEN_SHRINKAGE_K",
    "HISTORY_N_BUCKETS",
    "MIN_EMPIRICAL_BAYES_PLAYER_N",
    "RESIDUALIZER_N_PARAMS",
    "SHRINKAGE_GRID",
    "SLICE14_STATE_COLUMNS",
    "Slice14DiagnosticReport",
    "attach_causal_candidate_b",
    "attach_player_farming_state",
    "causal_position_duration_design",
    "classify_slice14",
    "development_tune_end",
    "empirical_bayes_k",
    "farming_shrinkage_weight",
    "farming_shrunk_b",
    "fit_causal_residualizer",
    "history_n_bucket",
    "prior_farming_history",
    "run_player_farming_state_diagnostics",
    "select_shrinkage_k",
    "slice14_report_to_jsonable",
]


FROZEN_CANDIDATE_B = CANDIDATE_B
CAUSAL_B_COLUMN = "farming_causal_b"
RESIDUALIZER_N_PARAMS = 6
RESIDUAL_STD_FLOOR = 1e-12
DURATION_VARIANCE_FLOOR = 1e-12
SHRINKAGE_GRID: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
# Frozen after the development-only grid + later-development confirmation.
# ``select_shrinkage_k`` is the authority; this constant documents the
# chosen grid point so later slices do not re-search.
FROZEN_SHRINKAGE_K = 5.0
MIN_EMPIRICAL_BAYES_PLAYER_N = 8
MIN_EMPIRICAL_BAYES_PLAYERS = 8
EQUIVALENT_RMSE_RATIO = 1.01
ESTABLISHED_RMSE_CAP_RATIO = 1.05
_REPEATABILITY_FLOOR = 0.10
_CORR_WEAK = 0.05
MIN_LOW_N_VALIDATION_ROWS = 100
OVERALL_RMSE_CAP_RATIO = 1.01
EXPECTED_DEVELOPMENT_PLAYER_ROWS = 59_670
EXPECTED_HOLDOUT_PLAYER_ROWS = FROZEN_HOLDOUT_EXPECTED_N * 10
RTM_MIN_EACH = 5
RTM_EXTREME_QUANTILE = 0.90

HISTORY_N_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1–2", 1, 2),
    ("3–5", 3, 5),
    ("6–10", 6, 10),
    ("11–20", 11, 20),
    ("21–50", 21, 50),
    (">50", 51, None),
)

SLICE14_STATE_COLUMNS: tuple[str, ...] = (
    CAUSAL_B_COLUMN,
    "farming_residualizer_n",
    "farming_prior_n",
    "farming_prior_sum_b",
    "farming_prior_mean_b",
    "farming_shrinkage_weight",
    "farming_shrunk_b",
)

_POSITION_DUMMY_LEVELS: tuple[int, ...] = (2, 3, 4, 5)


@dataclass(frozen=True)
class Slice14DiagnosticReport:
    development_end: datetime
    tune_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    selected_k: float
    selected_k_justification: str
    classification: pd.DataFrame
    coverage: pd.DataFrame
    residualizer_warmup: pd.DataFrame
    split: pd.DataFrame
    shrinkage_grid_tune: pd.DataFrame
    shrinkage_grid_validation: pd.DataFrame
    empirical_bayes: pd.DataFrame
    history_bucket_tune: pd.DataFrame
    history_bucket_validation: pd.DataFrame
    state_distribution: pd.DataFrame
    persistence: pd.DataFrame
    consecutive_persistence: pd.DataFrame
    first_half_second_half: pd.DataFrame
    regression_to_mean: pd.DataFrame
    integrity: dict[str, object]


def causal_position_duration_design(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed intercept + position 2–5 dummies + ``duration_seconds``.

    Position 1 is the reference level, matching Slice 13's drop-first
    dummies when all five explicit positions are present. Missing
    position stays NULL in every column. The column set does not depend
    on which positions happen to appear in ``frame``, so a later match
    cannot change the design used at an earlier timestamp.
    """
    eligible = explicit_position_mask(frame)
    numbers = _numeric(frame["position_number"])
    intercept = pd.Series(1.0, index=frame.index, dtype=float).where(eligible)
    columns: dict[str, pd.Series] = {"intercept": intercept}
    for level in _POSITION_DUMMY_LEVELS:
        dummy = (numbers == float(level)).astype(float)
        columns[f"pos_{level}"] = dummy.where(eligible)
    duration = _numeric(frame["duration_seconds"]).where(eligible)
    columns["duration_seconds"] = duration
    return pd.DataFrame(columns, index=frame.index)


def _moments_from_design(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    """XtX, Xty, yty, n, and position-1–5 counts from a design block."""
    xtx = x.T @ x
    xty = x.T @ y
    yty = float(y @ y)
    n = int(x.shape[0])
    dummies = x[:, 1:5]
    counts = np.zeros(5, dtype=int)
    counts[0] = int(np.sum(dummies.sum(axis=1) == 0.0))
    counts[1:] = np.sum(dummies == 1.0, axis=0)
    return xtx, xty, yty, n, counts


def _fit_from_moments(
    xtx: np.ndarray,
    xty: np.ndarray,
    yty: float,
    n: int,
    pos_counts: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Identify the Slice 13 design from prior-only sufficient statistics."""
    if n < RESIDUALIZER_N_PARAMS:
        return None
    if int(pos_counts.min()) < 1:
        return None
    mean_duration = xtx[0, 5] / n
    duration_var = xtx[5, 5] / n - mean_duration * mean_duration
    if duration_var <= DURATION_VARIANCE_FLOOR:
        return None
    sign, _logdet = np.linalg.slogdet(xtx)
    if sign <= 0:
        return None
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(coef).all():
        return None
    sse = yty - float(coef @ xty)
    sum_resid = float(xty[0] - xtx[0] @ coef)
    mu = sum_resid / n
    sse_centered = sse - n * mu * mu
    if sse_centered < 0.0 and sse_centered > -1e-9:
        sse_centered = 0.0
    if sse_centered < 0.0:
        return None
    sigma = float(np.sqrt(sse_centered / n))
    if sigma <= RESIDUAL_STD_FLOOR:
        return None
    return coef, mu, sigma


def fit_causal_residualizer(
    design: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Fit ``y ~ position + duration`` on the supplied prior rows only.

    Returns ``(coef, residual_mean, residual_std)`` with population
    (ddof=0) residual scale, or ``None`` when the prior window cannot
    identify the Slice 13 design. Never falls back to a larger sample.
    """
    if isinstance(design, pd.DataFrame):
        x = design.to_numpy(dtype=float)
    else:
        x = np.asarray(design, dtype=float)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[1] != RESIDUALIZER_N_PARAMS:
        return None
    finite = np.isfinite(x).all(axis=1) & np.isfinite(yv)
    x = x[finite]
    yv = yv[finite]
    if x.shape[0] == 0:
        return None
    xtx, xty, yty, n, pos_counts = _moments_from_design(x, yv)
    return _fit_from_moments(xtx, xty, yty, n, pos_counts)


def attach_causal_candidate_b(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe candidate B and residualizer sample size.

    For each timestamp ``T``, coefficients and residual scale are fit on
    eligible rows with ``start_time < T`` and applied to every row at
    ``T``. Rows at ``T`` never enter one another's baseline. When the
    prior window is unidentified, ``farming_causal_b`` is NULL.
    """
    out = frame.copy()
    out["last_hits_per_minute"] = per_minute(
        out["num_last_hits"], out["duration_seconds"]
    )
    eligible = explicit_position_mask(out) & out["last_hits_per_minute"].notna()
    design = causal_position_duration_design(out)
    b = pd.Series(np.nan, index=out.index, dtype=float)
    residualizer_n = pd.Series(0, index=out.index, dtype=int)
    if not bool(eligible.any()):
        out[CAUSAL_B_COLUMN] = b
        out["farming_residualizer_n"] = residualizer_n
        return out

    times = pd.to_datetime(out["start_time"], utc=True).to_numpy()
    eligible_pos = np.flatnonzero(eligible.to_numpy())
    order = np.argsort(times[eligible_pos], kind="mergesort")
    sorted_pos = eligible_pos[order]
    sorted_times = times[sorted_pos]
    x_all = design.to_numpy(dtype=float)[sorted_pos]
    y_all = out["last_hits_per_minute"].to_numpy(dtype=float)[sorted_pos]
    cuts = np.r_[True, sorted_times[1:] != sorted_times[:-1]]
    starts = np.flatnonzero(cuts)
    bounds = np.r_[starts, len(sorted_pos)]
    xtx = np.zeros((RESIDUALIZER_N_PARAMS, RESIDUALIZER_N_PARAMS), dtype=float)
    xty = np.zeros(RESIDUALIZER_N_PARAMS, dtype=float)
    yty = 0.0
    n_prior = 0
    pos_counts = np.zeros(5, dtype=int)
    for i in range(len(starts)):
        lo = int(bounds[i])
        hi = int(bounds[i + 1])
        residualizer_n.iloc[sorted_pos[lo:hi]] = n_prior
        fitted = _fit_from_moments(xtx, xty, yty, n_prior, pos_counts)
        if fitted is not None:
            coef, mu, sigma = fitted
            resid = y_all[lo:hi] - x_all[lo:hi] @ coef
            b.iloc[sorted_pos[lo:hi]] = (resid - mu) / sigma
        block_x = x_all[lo:hi]
        block_y = y_all[lo:hi]
        d_xtx, d_xty, d_yty, d_n, d_pos = _moments_from_design(block_x, block_y)
        xtx += d_xtx
        xty += d_xty
        yty += d_yty
        n_prior += d_n
        pos_counts += d_pos
    out[CAUSAL_B_COLUMN] = b
    out["farming_residualizer_n"] = residualizer_n
    return out


def prior_farming_history(
    frame: pd.DataFrame, column: str = CAUSAL_B_COLUMN
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Strictly prior player ``n``, ``sum``, and ``mean`` of causal B.

    ``H.start_time < M.start_time``. Same-timestamp appearances are
    mutually blind. Non-finite B does not increment ``n``. Mean is NULL
    when ``n = 0``.
    """
    values = _numeric(frame[column])
    means = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0, index=frame.index, dtype=int)
    sums = pd.Series(0.0, index=frame.index, dtype=float)
    if frame.empty:
        return counts, sums, means
    times = pd.to_datetime(frame["start_time"], utc=True)
    work = pd.DataFrame(
        {
            "player_id": frame["player_id"].to_numpy(),
            "time": times.to_numpy(),
            "value": values.to_numpy(dtype=float),
        },
        index=frame.index,
    )
    for _player_id, group in work.groupby("player_id", sort=False):
        ordered = group.sort_values("time", kind="mergesort")
        stamps = ordered["time"].to_numpy()
        vals = ordered["value"].to_numpy(dtype=float)
        player_means = np.full(len(ordered), np.nan)
        player_counts = np.zeros(len(ordered), dtype=int)
        player_sums = np.zeros(len(ordered), dtype=float)
        for i in range(len(ordered)):
            prior = (stamps < stamps[i]) & np.isfinite(vals)
            n = int(prior.sum())
            player_counts[i] = n
            if n > 0:
                total = float(vals[prior].sum())
                player_sums[i] = total
                player_means[i] = total / n
        means.loc[ordered.index] = player_means
        counts.loc[ordered.index] = player_counts
        sums.loc[ordered.index] = player_sums
    return counts, sums, means


def farming_shrinkage_weight(prior_n: float, *, k: float) -> float:
    """Evidence fraction ``n / (n + k)``. Zero when ``n = 0``.

    ``k = 0`` is the unshrunk reference: weight is 1 for ``n > 0``.
    """
    if k < 0.0:
        raise ValueError(f"shrinkage k must be >= 0, got {k}")
    n = float(prior_n)
    if n <= 0.0:
        return 0.0
    if k == 0.0:
        return 1.0
    return n / (n + k)


def farming_shrunk_b(
    mean_b: float | None, prior_n: float, *, k: float
) -> float:
    """``n / (n + k) * mean``. Exactly 0 when ``n = 0``."""
    weight = farming_shrinkage_weight(prior_n, k=k)
    if weight == 0.0 or mean_b is None or (
        isinstance(mean_b, float) and not np.isfinite(mean_b)
    ):
        return 0.0
    return weight * float(mean_b)


def apply_farming_shrinkage(
    mean_b: pd.Series, prior_n: pd.Series, *, k: float
) -> tuple[pd.Series, pd.Series]:
    """Vectorized shrinkage toward zero. ``k = 0`` is allowed."""
    if k < 0.0:
        raise ValueError(f"shrinkage k must be >= 0, got {k}")
    n = pd.to_numeric(prior_n, errors="coerce").fillna(0.0).astype(float)
    mean = pd.to_numeric(mean_b, errors="coerce")
    if k == 0.0:
        weight = pd.Series(np.where(n > 0.0, 1.0, 0.0), index=mean_b.index)
    else:
        weight = n / (n + k)
        weight = weight.mask(n <= 0.0, 0.0)
    shrunk = weight * mean
    shrunk = shrunk.fillna(0.0)
    shrunk = shrunk.mask(n <= 0.0, 0.0)
    return shrunk.astype(float), weight.astype(float)


def attach_player_farming_state(
    frame: pd.DataFrame, *, k: float = FROZEN_SHRINKAGE_K
) -> pd.DataFrame:
    """Causal B plus strictly prior player farming state at shrinkage ``k``.

    Does not recompute causal B when ``farming_causal_b`` is already
    present, so leakage tests can mutate B independently of last hits.
    """
    if k < 0.0:
        raise ValueError(f"shrinkage k must be >= 0, got {k}")
    if CAUSAL_B_COLUMN in frame.columns:
        out = frame.copy()
    else:
        out = attach_causal_candidate_b(frame)
    counts, sums, means = prior_farming_history(out, CAUSAL_B_COLUMN)
    shrunk, weight = apply_farming_shrinkage(means, counts, k=k)
    out["farming_prior_n"] = counts
    out["farming_prior_sum_b"] = sums
    out["farming_prior_mean_b"] = means
    out["farming_shrinkage_weight"] = weight
    out["farming_shrunk_b"] = shrunk
    return out


def history_n_bucket(prior_n: float) -> str:
    """Map a prior count onto the Slice 14 history-size labels."""
    n = int(prior_n)
    for label, low, high in HISTORY_N_BUCKETS:
        if high is None:
            if n >= low:
                return label
        elif low <= n <= high:
            return label
    return "0"


def development_tune_end(
    start_time: pd.Series, *, development_end: datetime | None = None
) -> datetime:
    """Timestamp closing the earlier development (tune) partition.

    Reuses the walk-forward nested-split helper: same-timestamp groups
    stay together, the cut is a realized ``start_time``, and the
    fraction is ``DEFAULT_WALK_FORWARD_CONFIG.train_fraction_of_past``.
    """
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    stamp = pd.to_datetime(start_time, utc=True)
    train_end = _train_end_within_past(
        stamp,
        pd.Timestamp(end),
        DEFAULT_WALK_FORWARD_CONFIG.train_fraction_of_past,
    )
    return utc_datetime(train_end)


def _finite_pair(actual: pd.Series, predicted: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    a = _numeric(actual)
    p = _numeric(predicted)
    mask = a.notna() & p.notna() & np.isfinite(a.to_numpy()) & np.isfinite(
        p.to_numpy()
    )
    return a[mask].to_numpy(dtype=float), p[mask].to_numpy(dtype=float)


def _prediction_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, object]:
    y, yhat = _finite_pair(actual, predicted)
    n = int(y.size)
    if n == 0:
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "slope": float("nan"),
            "pred_mean": float("nan"),
            "pred_std": float("nan"),
            "actual_mean": float("nan"),
            "actual_std": float("nan"),
        }
    err = y - yhat
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": _pearson(pd.Series(y), pd.Series(yhat)),
        "spearman": _spearman(pd.Series(y), pd.Series(yhat)),
        "slope": slope_coefficient(pd.Series(y), pd.Series(yhat)),
        "pred_mean": float(yhat.mean()),
        "pred_std": _std(yhat),
        "actual_mean": float(y.mean()),
        "actual_std": _std(y),
    }


def _bucket_metrics(
    frame: pd.DataFrame, predicted: pd.Series, *, k: float, split: str
) -> pd.DataFrame:
    actual = _numeric(frame[CAUSAL_B_COLUMN])
    n_prior = pd.to_numeric(frame["farming_prior_n"], errors="coerce").fillna(0)
    rows: list[dict[str, object]] = []
    observed = actual.notna()
    overall = _prediction_metrics(actual[observed], predicted[observed])
    rows.append(
        {
            "k": k,
            "split": split,
            "bucket": "all",
            **overall,
        }
    )
    for label, low, high in HISTORY_N_BUCKETS:
        if high is None:
            mask = observed & (n_prior >= low)
        else:
            mask = observed & (n_prior >= low) & (n_prior <= high)
        metrics = _prediction_metrics(actual[mask], predicted[mask])
        rows.append({"k": k, "split": split, "bucket": label, **metrics})
    return pd.DataFrame(rows)


def _grid_summary(bucket_table: pd.DataFrame, *, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for k, group in bucket_table.groupby("k", sort=False):
        by_bucket = group.set_index("bucket")

        def _val(name: str, column: str = "rmse") -> float:
            if name not in by_bucket.index:
                return float("nan")
            return float(by_bucket.loc[name, column])

        rows.append(
            {
                "k": float(k),
                "split": split,
                "n_eval": int(by_bucket.loc["all", "n"]) if "all" in by_bucket.index else 0,
                "rmse": _val("all"),
                "mae": _val("all", "mae"),
                "pearson": _val("all", "pearson"),
                "spearman": _val("all", "spearman"),
                "slope": _val("all", "slope"),
                "pred_mean": _val("all", "pred_mean"),
                "pred_std": _val("all", "pred_std"),
                "rmse_n_0": _val("0"),
                "rmse_n_1_2": _val("1–2"),
                "rmse_n_3_5": _val("3–5"),
                "rmse_n_6_10": _val("6–10"),
                "rmse_n_11_20": _val("11–20"),
                "rmse_n_21_50": _val("21–50"),
                "rmse_n_gt_50": _val(">50"),
                "rmse_n_gt_20": _combined_rmse(group, ("21–50", ">50")),
                "low_n_rmse": _nanmean([_val("1–2"), _val("3–5")]),
                "n_low_n": int(
                    (0 if "1–2" not in by_bucket.index else int(by_bucket.loc["1–2", "n"]))
                    + (0 if "3–5" not in by_bucket.index else int(by_bucket.loc["3–5", "n"]))
                ),
                "pearson_n_1_2": _val("1–2", "pearson"),
                "pearson_n_gt_20": _combined_pearson(group, ("21–50", ">50")),
            }
        )
    return pd.DataFrame(rows)


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _combined_rmse(group: pd.DataFrame, buckets: tuple[str, ...]) -> float:
    subset = group.loc[group["bucket"].isin(buckets)]
    n = pd.to_numeric(subset["n"], errors="coerce").to_numpy(dtype=float)
    rmse = pd.to_numeric(subset["rmse"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(n) & np.isfinite(rmse) & (n > 0)
    if not bool(mask.any()):
        return float("nan")
    sse = np.sum((rmse[mask] ** 2) * n[mask])
    total = np.sum(n[mask])
    if total <= 0:
        return float("nan")
    return float(np.sqrt(sse / total))


def _combined_pearson(group: pd.DataFrame, buckets: tuple[str, ...]) -> float:
    subset = group.loc[group["bucket"].isin(buckets)]
    n = pd.to_numeric(subset["n"], errors="coerce")
    r = pd.to_numeric(subset["pearson"], errors="coerce")
    mask = n.notna() & r.notna() & (n > 0) & np.isfinite(r)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(r[mask].to_numpy(dtype=float), weights=n[mask].to_numpy(dtype=float)))


def select_shrinkage_k(tune_grid: pd.DataFrame) -> tuple[float, str]:
    """Choose ``k`` from the tune-set grid.

    Prefers a broad, stable low-``n`` RMSE improvement over a razor-thin
    optimum. Among essentially equivalent neighbors, pick the larger
    ``k`` (stronger shrinkage). ``k`` that inflate established-player
    RMSE by more than 5% versus ``k=0`` are discarded.
    """
    if tune_grid.empty:
        return 0.0, "Empty tune grid; defaulting to unshrunk k=0."
    work = tune_grid.copy()
    k0 = work.loc[work["k"] == 0.0]
    if k0.empty:
        return float(work.iloc[0]["k"]), "No k=0 reference row; using the first grid point."
    baseline = k0.iloc[0]
    established_k0 = float(baseline["rmse_n_gt_20"])
    cap = (
        established_k0 * ESTABLISHED_RMSE_CAP_RATIO
        if np.isfinite(established_k0)
        else float("inf")
    )
    positive = work.loc[work["k"] > 0.0].copy()
    if positive.empty:
        return 0.0, "Grid contained only k=0."
    established = pd.to_numeric(positive["rmse_n_gt_20"], errors="coerce")
    keep = positive.loc[established.isna() | (established <= cap)]
    if keep.empty:
        return (
            0.0,
            "No positive k kept established-player RMSE within 5% of k=0.",
        )
    low_n = pd.to_numeric(keep["low_n_rmse"], errors="coerce")
    overall = pd.to_numeric(keep["rmse"], errors="coerce")
    if not bool(low_n.notna().any()):
        if not bool(overall.notna().any()):
            return (
                0.0,
                "Tune split had no observed causal B; defaulting to unshrunk k=0.",
            )
        best_overall = float(overall.min())
        equiv = keep.loc[overall <= best_overall * EQUIVALENT_RMSE_RATIO]
        if equiv.empty:
            return 0.0, "No equivalent overall-RMSE neighbors; defaulting to k=0."
        selected = float(equiv["k"].max())
        return selected, (
            "Low-n RMSE was unavailable; selected the largest k within "
            f"{EQUIVALENT_RMSE_RATIO:.0%} of best overall tune RMSE."
        )
    best_low = float(low_n.min())
    equiv = keep.loc[low_n <= best_low * EQUIVALENT_RMSE_RATIO]
    if equiv.empty:
        return 0.0, "No equivalent low-n neighbors; defaulting to k=0."
    selected = float(equiv["k"].max())
    k0_low = float(baseline["low_n_rmse"])
    selected_row = keep.loc[keep["k"] == selected].iloc[0]
    selected_low = float(selected_row["low_n_rmse"])
    improved = np.isfinite(k0_low) and np.isfinite(selected_low) and selected_low < k0_low
    if improved:
        justification = (
            f"Selected k={selected:g} as the strongest shrinkage among "
            "tune-set neighbors within 1% of the best low-n RMSE, without "
            "inflating established-player RMSE by more than 5% versus k=0."
        )
    else:
        justification = (
            f"Selected k={selected:g} from equivalent low-n neighbors, but "
            "low-n RMSE did not beat k=0 on the tune split."
        )
    return selected, justification


def empirical_bayes_k(
    frame: pd.DataFrame, *, min_player_n: int = MIN_EMPIRICAL_BAYES_PLAYER_N
) -> pd.DataFrame:
    """Sanity-check ``k ≈ within-player var / between-player var``.

    Restricted to the supplied rows (tune split). Players with fewer
    than ``min_player_n`` causal B observations are excluded. This is
    not a search and is not used to override the grid.
    """
    values = _numeric(frame[CAUSAL_B_COLUMN])
    work = pd.DataFrame(
        {"player_id": frame["player_id"], "b": values},
        index=frame.index,
    ).dropna()
    if work.empty:
        return pd.DataFrame(
            [
                {
                    "k": float("nan"),
                    "within_player_variance": float("nan"),
                    "between_player_variance": float("nan"),
                    "n_players": 0,
                    "n_appearances": 0,
                    "min_player_n": min_player_n,
                    "used_for_state": False,
                }
            ]
        )
    grouped = work.groupby("player_id")["b"]
    stats = grouped.agg(n="size", mean="mean", var=lambda s: float(s.var(ddof=1)))
    eligible = stats.loc[stats["n"] >= min_player_n]
    n_players = int(len(eligible))
    n_appearances = int(eligible["n"].sum()) if n_players else 0
    if n_players < MIN_EMPIRICAL_BAYES_PLAYERS:
        return pd.DataFrame(
            [
                {
                    "k": float("nan"),
                    "within_player_variance": float("nan"),
                    "between_player_variance": float("nan"),
                    "n_players": n_players,
                    "n_appearances": n_appearances,
                    "min_player_n": min_player_n,
                    "used_for_state": False,
                }
            ]
        )
    within = float(eligible["var"].mean())
    sampling = float((eligible["var"] / eligible["n"]).mean())
    between_raw = float(eligible["mean"].var(ddof=1))
    between = max(0.0, between_raw - sampling)
    k = float("inf") if between <= 0.0 else float(within / between)
    return pd.DataFrame(
        [
            {
                "k": k,
                "within_player_variance": within,
                "between_player_variance": between,
                "between_raw_var_of_means": between_raw,
                "mean_sampling_variance": sampling,
                "n_players": n_players,
                "n_appearances": n_appearances,
                "min_player_n": min_player_n,
                "used_for_state": False,
            }
        ]
    )


def _distribution_row(values: np.ndarray, *, kind: str, subset: str) -> dict[str, object]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "kind": kind,
            "subset": subset,
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
    return {
        "kind": kind,
        "subset": subset,
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


def state_distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    raw = _numeric(frame["farming_prior_mean_b"]).to_numpy(dtype=float)
    shrunk = _numeric(frame["farming_shrunk_b"]).to_numpy(dtype=float)
    n = pd.to_numeric(frame["farming_prior_n"], errors="coerce").fillna(0)
    rows = [
        _distribution_row(raw, kind="raw_mean_b", subset="n>=1"),
        _distribution_row(shrunk, kind="shrunk_b", subset="all"),
        _distribution_row(shrunk[n.to_numpy() == 0], kind="shrunk_b", subset="n=0"),
        _distribution_row(shrunk[n.to_numpy() >= 1], kind="shrunk_b", subset="n>=1"),
        _distribution_row(
            raw[((n >= 1) & (n <= 2)).to_numpy()],
            kind="raw_mean_b",
            subset="n=1–2",
        ),
        _distribution_row(
            shrunk[((n >= 1) & (n <= 2)).to_numpy()],
            kind="shrunk_b",
            subset="n=1–2",
        ),
    ]
    return pd.DataFrame(rows)


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = explicit_position_mask(frame) & _numeric(
        frame["last_hits_per_minute"]
    ).notna()
    causal = _numeric(frame[CAUSAL_B_COLUMN]).notna()
    n = pd.to_numeric(frame["farming_prior_n"], errors="coerce").fillna(0)
    n_values = n.to_numpy(dtype=int)
    return pd.DataFrame(
        [
            {
                "n_eligible_appearances": int(eligible.sum()),
                "n_causal_b_available": int(causal.sum()),
                "n_residualizer_warmup_loss": int((eligible & ~causal).sum()),
                "n_prior_0": int((n == 0).sum()),
                "n_prior_ge_1": int((n >= 1).sum()),
                "n_prior_ge_5": int((n >= 5).sum()),
                "n_prior_ge_10": int((n >= 10).sum()),
                "n_prior_ge_20": int((n >= 20).sum()),
                "n_players": int(frame["player_id"].nunique()),
                "n_players_with_causal_b": int(
                    frame.loc[causal, "player_id"].nunique()
                )
                if bool(causal.any())
                else 0,
                "prior_n_mean": float(n_values.mean()) if n_values.size else float("nan"),
                "prior_n_median": float(np.median(n_values))
                if n_values.size
                else float("nan"),
                "prior_n_p95": float(np.quantile(n_values, 0.95))
                if n_values.size
                else float("nan"),
                "prior_n_max": int(n_values.max()) if n_values.size else 0,
            }
        ]
    )


def residualizer_warmup_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = explicit_position_mask(frame) & _numeric(
        frame["last_hits_per_minute"]
    ).notna()
    causal = _numeric(frame[CAUSAL_B_COLUMN]).notna()
    ready = frame.loc[eligible & causal]
    first_ready = (
        pd.to_datetime(ready["start_time"], utc=True).min()
        if not ready.empty
        else pd.NaT
    )
    n_prior = pd.to_numeric(
        frame.loc[eligible & causal, "farming_residualizer_n"], errors="coerce"
    )
    return pd.DataFrame(
        [
            {
                "n_eligible": int(eligible.sum()),
                "n_causal_b": int(causal.sum()),
                "n_warmup_loss": int((eligible & ~causal).sum()),
                "warmup_loss_fraction": (
                    float((eligible & ~causal).sum()) / float(eligible.sum())
                    if int(eligible.sum())
                    else float("nan")
                ),
                "first_ready_start_time": (
                    utc_datetime(first_ready).isoformat()
                    if pd.notna(first_ready)
                    else None
                ),
                "min_residualizer_n_when_ready": (
                    int(n_prior.min()) if n_prior.notna().any() else 0
                ),
                "global_fit_fallback_used": False,
            }
        ]
    )


def persistence_table(frame: pd.DataFrame, *, k: float) -> pd.DataFrame:
    actual = _numeric(frame[CAUSAL_B_COLUMN])
    raw = _numeric(frame["farming_prior_mean_b"])
    shrunk = _numeric(frame["farming_shrunk_b"])
    n = pd.to_numeric(frame["farming_prior_n"], errors="coerce").fillna(0)
    rows: list[dict[str, object]] = []
    observed = actual.notna()
    for label, low, high in (("all", 0, None), *HISTORY_N_BUCKETS):
        if label == "all":
            mask = observed
        elif high is None:
            mask = observed & (n >= low)
        else:
            mask = observed & (n >= low) & (n <= high)
        rows.append(
            {
                "k": k,
                "bucket": label,
                "n_rows": int(mask.sum()),
                "n_players": int(frame.loc[mask, "player_id"].nunique())
                if int(mask.sum())
                else 0,
                "pearson_raw_mean": _pearson(raw[mask], actual[mask]),
                "pearson_shrunk": _pearson(shrunk[mask], actual[mask]),
                "spearman_raw_mean": _spearman(raw[mask], actual[mask]),
                "spearman_shrunk": _spearman(shrunk[mask], actual[mask]),
            }
        )
    return pd.DataFrame(rows)


def consecutive_persistence_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Same-player consecutive-appearance persistence of B and of state."""
    values = _numeric(frame[CAUSAL_B_COLUMN])
    work = frame.loc[
        :,
        ["player_id", "start_time", CAUSAL_B_COLUMN, "farming_shrunk_b", "farming_prior_n"],
    ].copy()
    work[CAUSAL_B_COLUMN] = values
    work["farming_shrunk_b"] = _numeric(work["farming_shrunk_b"])
    work["start_time"] = pd.to_datetime(work["start_time"], utc=True)
    b_now: list[float] = []
    b_next: list[float] = []
    state_now: list[float] = []
    state_next: list[float] = []
    n_now: list[int] = []
    for _player_id, group in work.groupby("player_id", sort=False):
        ordered = group.sort_values("start_time", kind="mergesort")
        stamps = ordered["start_time"].to_numpy()
        b = ordered[CAUSAL_B_COLUMN].to_numpy(dtype=float)
        st = ordered["farming_shrunk_b"].to_numpy(dtype=float)
        nn = ordered["farming_prior_n"].to_numpy(dtype=int)
        for i in range(len(ordered) - 1):
            if not (stamps[i] < stamps[i + 1]):
                continue
            if np.isfinite(b[i]) and np.isfinite(b[i + 1]):
                b_now.append(float(b[i]))
                b_next.append(float(b[i + 1]))
            if np.isfinite(st[i]) and np.isfinite(st[i + 1]):
                state_now.append(float(st[i]))
                state_next.append(float(st[i + 1]))
                n_now.append(int(nn[i]))
    return pd.DataFrame(
        [
            {
                "kind": "causal_b_consecutive",
                "n_pairs": len(b_now),
                "pearson": _pearson(pd.Series(b_now), pd.Series(b_next))
                if b_now
                else float("nan"),
                "spearman": _spearman(pd.Series(b_now), pd.Series(b_next))
                if b_now
                else float("nan"),
            },
            {
                "kind": "shrunk_state_consecutive",
                "n_pairs": len(state_now),
                "pearson": _pearson(pd.Series(state_now), pd.Series(state_next))
                if state_now
                else float("nan"),
                "spearman": _spearman(pd.Series(state_now), pd.Series(state_next))
                if state_now
                else float("nan"),
            },
        ]
    )


def regression_to_mean_table(
    frame: pd.DataFrame,
    *,
    k: float,
    min_each: int = RTM_MIN_EACH,
) -> pd.DataFrame:
    """Extreme early mean B versus later mean B, raw vs shrunk."""
    values = _numeric(frame[CAUSAL_B_COLUMN])
    work = frame.loc[values.notna(), ["player_id", "start_time", CAUSAL_B_COLUMN]].copy()
    work[CAUSAL_B_COLUMN] = values.loc[work.index]
    work["start_time"] = pd.to_datetime(work["start_time"], utc=True)
    early_means: list[float] = []
    late_means: list[float] = []
    early_n: list[int] = []
    late_n: list[int] = []
    for _player_id, group in work.groupby("player_id", sort=False):
        ordered = group.sort_values("start_time", kind="mergesort")
        n = len(ordered)
        split = n // 2
        if split < min_each or (n - split) < min_each:
            continue
        early = ordered.iloc[:split][CAUSAL_B_COLUMN].to_numpy(dtype=float)
        late = ordered.iloc[split:][CAUSAL_B_COLUMN].to_numpy(dtype=float)
        early_means.append(float(early.mean()))
        late_means.append(float(late.mean()))
        early_n.append(int(early.size))
        late_n.append(int(late.size))
    if not early_means:
        return pd.DataFrame(
            [
                {
                    "subset": "none",
                    "n_players": 0,
                    "k": k,
                    "early_raw_mean": float("nan"),
                    "early_shrunk_mean": float("nan"),
                    "late_mean": float("nan"),
                    "rmse_raw": float("nan"),
                    "rmse_shrunk": float("nan"),
                    "mae_raw": float("nan"),
                    "mae_shrunk": float("nan"),
                    "mean_abs_early_raw": float("nan"),
                    "mean_abs_early_shrunk": float("nan"),
                    "mean_abs_late": float("nan"),
                    "mean_late_over_early_raw": float("nan"),
                }
            ]
        )
    early_arr = np.asarray(early_means, dtype=float)
    late_arr = np.asarray(late_means, dtype=float)
    n_arr = np.asarray(early_n, dtype=float)
    shrunk_early = np.array(
        [
            farming_shrunk_b(float(mean), float(n), k=k)
            for mean, n in zip(early_arr, n_arr, strict=True)
        ],
        dtype=float,
    )
    abs_early = np.abs(early_arr)
    threshold = float(np.quantile(abs_early, RTM_EXTREME_QUANTILE))
    extreme = abs_early >= threshold

    def _block(mask: np.ndarray, subset: str) -> dict[str, object]:
        e = early_arr[mask]
        s = shrunk_early[mask]
        l = late_arr[mask]
        if e.size == 0:
            return {
                "subset": subset,
                "n_players": 0,
                "k": k,
                "early_raw_mean": float("nan"),
                "early_shrunk_mean": float("nan"),
                "late_mean": float("nan"),
                "rmse_raw": float("nan"),
                "rmse_shrunk": float("nan"),
                "mae_raw": float("nan"),
                "mae_shrunk": float("nan"),
                "mean_abs_early_raw": float("nan"),
                "mean_abs_early_shrunk": float("nan"),
                "mean_abs_late": float("nan"),
                "mean_late_over_early_raw": float("nan"),
                "extreme_abs_threshold": threshold,
            }
        ratio = np.divide(
            l, e, out=np.full(e.shape, np.nan), where=np.abs(e) > 1e-12
        )
        return {
            "subset": subset,
            "n_players": int(e.size),
            "k": k,
            "early_raw_mean": float(e.mean()),
            "early_shrunk_mean": float(s.mean()),
            "late_mean": float(l.mean()),
            "rmse_raw": float(np.sqrt(np.mean((l - e) ** 2))),
            "rmse_shrunk": float(np.sqrt(np.mean((l - s) ** 2))),
            "mae_raw": float(np.mean(np.abs(l - e))),
            "mae_shrunk": float(np.mean(np.abs(l - s))),
            "mean_abs_early_raw": float(np.mean(np.abs(e))),
            "mean_abs_early_shrunk": float(np.mean(np.abs(s))),
            "mean_abs_late": float(np.mean(np.abs(l))),
            "mean_late_over_early_raw": float(np.nanmean(ratio)),
            "extreme_abs_threshold": threshold,
        }

    return pd.DataFrame(
        [
            _block(np.ones(early_arr.shape, dtype=bool), "all_paired"),
            _block(extreme, "extreme_early_abs_ge_p90"),
        ]
    )


def classify_slice14(
    *,
    tune_grid: pd.DataFrame,
    validation_grid: pd.DataFrame,
    selected_k: float,
    persistence: pd.DataFrame,
    first_half: pd.DataFrame,
) -> pd.DataFrame:
    """Map temporal persistence and shrinkage confirmation onto A / B / C.

    Shrinkage is judged on the tune split, where low-``n`` cells are
    populated. Validation must confirm that the frozen ``k`` still
    predicts later causal B and does not inflate overall or
    established-player RMSE. Tiny validation low-``n`` buckets are not
    allowed to veto a well-sampled tune result. Does not use win rate,
    Elo, log loss, AUC, or betting metrics.
    """
    val_sel = validation_grid.loc[validation_grid["k"] == selected_k]
    val_k0 = validation_grid.loc[validation_grid["k"] == 0.0]
    tune_sel = tune_grid.loc[tune_grid["k"] == selected_k]
    tune_k0 = tune_grid.loc[tune_grid["k"] == 0.0]
    pearson = (
        float(val_sel.iloc[0]["pearson"]) if not val_sel.empty else float("nan")
    )
    val_rmse = float(val_sel.iloc[0]["rmse"]) if not val_sel.empty else float("nan")
    val_rmse_k0 = float(val_k0.iloc[0]["rmse"]) if not val_k0.empty else float("nan")
    low_n = (
        float(val_sel.iloc[0]["low_n_rmse"]) if not val_sel.empty else float("nan")
    )
    low_n_k0 = (
        float(val_k0.iloc[0]["low_n_rmse"]) if not val_k0.empty else float("nan")
    )
    n_low_n_val = (
        int(val_sel.iloc[0]["n_low_n"])
        if not val_sel.empty and "n_low_n" in val_sel.columns
        else 0
    )
    tune_low_n = (
        float(tune_sel.iloc[0]["low_n_rmse"]) if not tune_sel.empty else float("nan")
    )
    tune_low_n_k0 = (
        float(tune_k0.iloc[0]["low_n_rmse"]) if not tune_k0.empty else float("nan")
    )
    established = (
        float(val_sel.iloc[0]["rmse_n_gt_20"]) if not val_sel.empty else float("nan")
    )
    established_k0 = (
        float(val_k0.iloc[0]["rmse_n_gt_20"]) if not val_k0.empty else float("nan")
    )
    half_r = (
        float(first_half.iloc[0]["pearson"]) if not first_half.empty else float("nan")
    )
    persist_r = float("nan")
    if not persistence.empty:
        all_row = persistence.loc[persistence["bucket"] == "all"]
        if not all_row.empty:
            persist_r = float(all_row.iloc[0]["pearson_shrunk"])

    tune_low_n_improves = (
        np.isfinite(tune_low_n)
        and np.isfinite(tune_low_n_k0)
        and tune_low_n < tune_low_n_k0
    )
    val_low_n_improves = (
        np.isfinite(low_n) and np.isfinite(low_n_k0) and low_n < low_n_k0
    )
    if n_low_n_val >= MIN_LOW_N_VALIDATION_ROWS:
        low_n_confirmed = val_low_n_improves
        low_n_source = "validation"
    else:
        low_n_confirmed = tune_low_n_improves
        low_n_source = "tune"
    established_ok = (not np.isfinite(established)) or (
        np.isfinite(established_k0)
        and established <= established_k0 * ESTABLISHED_RMSE_CAP_RATIO
    )
    overall_ok = (not np.isfinite(val_rmse)) or (
        np.isfinite(val_rmse_k0) and val_rmse <= val_rmse_k0 * OVERALL_RMSE_CAP_RATIO
    )
    signal = (
        (np.isfinite(pearson) and pearson >= _REPEATABILITY_FLOOR)
        or (np.isfinite(persist_r) and persist_r >= _REPEATABILITY_FLOOR)
        or (np.isfinite(half_r) and half_r >= _REPEATABILITY_FLOOR)
    )
    weak = (not np.isfinite(pearson) or pearson < _CORR_WEAK) and (
        not np.isfinite(half_r) or half_r < _CORR_WEAK
    )

    if weak:
        classification = "C"
        rationale = (
            "Causal candidate B does not persist as a player-level "
            "farming tendency once the residualizer itself is causal."
        )
        next_slice = (
            "Do not freeze a historical farming state. Treat last-hit "
            "residuals as match-level outcomes, not a player trait."
        )
    elif (
        selected_k > 0.0
        and signal
        and low_n_confirmed
        and established_ok
        and overall_ok
    ):
        classification = "A"
        rationale = (
            "Prior-only causal B is a repeatable player farming tendency, "
            f"and k={selected_k:g} improves low-history estimates on the "
            f"{low_n_source} split without destroying later-development "
            "signal for established players."
        )
        next_slice = (
            "Freeze the historical farming state and k for a later "
            "feature-evaluation slice. Do not run a win-model benchmark now."
        )
    else:
        classification = "B"
        rationale = (
            "Causal B shows player-level persistence, but the shrinkage "
            "constant is not yet a broad, stable improvement over the "
            "unshrunk mean."
        )
        next_slice = (
            "Keep the causal B construction and unshrunk prior mean; do "
            "not freeze k. Do not run a win-model benchmark."
        )
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "selected_k": selected_k,
                "rationale": rationale,
                "next_slice": next_slice,
                "validation_pearson_selected_k": pearson,
                "validation_rmse_selected_k": val_rmse,
                "validation_rmse_k0": val_rmse_k0,
                "validation_low_n_rmse_selected_k": low_n,
                "validation_low_n_rmse_k0": low_n_k0,
                "validation_n_low_n": n_low_n_val,
                "tune_low_n_rmse_selected_k": tune_low_n,
                "tune_low_n_rmse_k0": tune_low_n_k0,
                "validation_established_rmse_selected_k": established,
                "validation_established_rmse_k0": established_k0,
                "first_half_second_half_pearson": half_r,
                "persistence_pearson_shrunk": persist_r,
                "low_n_confirmed": low_n_confirmed,
                "low_n_source": low_n_source,
                "established_ok": established_ok,
                "overall_ok": overall_ok,
                "signal": signal,
            }
        ]
    )


def _evaluate_k_grid(
    frame: pd.DataFrame, *, split: str, ks: tuple[float, ...] = SHRINKAGE_GRID
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bucket_parts: list[pd.DataFrame] = []
    for k in ks:
        shrunk, _weight = apply_farming_shrinkage(
            frame["farming_prior_mean_b"], frame["farming_prior_n"], k=k
        )
        bucket_parts.append(_bucket_metrics(frame, shrunk, k=k, split=split))
    buckets = pd.concat(bucket_parts, ignore_index=True)
    return _grid_summary(buckets, split=split), buckets


def run_player_farming_state_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice14DiagnosticReport:
    """Development-only Slice 14 farming-state research. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    development = attach_causal_candidate_b(development)
    counts, sums, means = prior_farming_history(development, CAUSAL_B_COLUMN)
    development["farming_prior_n"] = counts
    development["farming_prior_sum_b"] = sums
    development["farming_prior_mean_b"] = means

    tune_end = development_tune_end(development["start_time"], development_end=end)
    dev_times = pd.to_datetime(development["start_time"], utc=True)
    tune_mask = dev_times <= pd.Timestamp(tune_end)
    val_mask = (dev_times > pd.Timestamp(tune_end)) & (dev_times <= pd.Timestamp(end))
    tune = development.loc[tune_mask].copy()
    validation = development.loc[val_mask].copy()

    tune_grid, tune_buckets = _evaluate_k_grid(tune, split="tune")
    selected_k, justification = select_shrinkage_k(tune_grid)
    val_grid, val_buckets = _evaluate_k_grid(validation, split="validation")

    shrunk, weight = apply_farming_shrinkage(
        development["farming_prior_mean_b"],
        development["farming_prior_n"],
        k=selected_k,
    )
    development["farming_shrinkage_weight"] = weight
    development["farming_shrunk_b"] = shrunk

    eb = empirical_bayes_k(tune)
    coverage = coverage_table(development)
    warmup = residualizer_warmup_table(development)
    split = pd.DataFrame(
        [
            {
                "tune_end": tune_end.isoformat(),
                "development_end": end.isoformat(),
                "train_fraction_of_past": (
                    DEFAULT_WALK_FORWARD_CONFIG.train_fraction_of_past
                ),
                "n_tune_rows": int(len(tune)),
                "n_tune_players": int(tune["player_id"].nunique()) if len(tune) else 0,
                "n_tune_matches": int(tune["match_id"].nunique()) if len(tune) else 0,
                "n_validation_rows": int(len(validation)),
                "n_validation_players": (
                    int(validation["player_id"].nunique()) if len(validation) else 0
                ),
                "n_validation_matches": (
                    int(validation["match_id"].nunique()) if len(validation) else 0
                ),
                "tune_min_start_time": (
                    pd.to_datetime(tune["start_time"], utc=True).min().isoformat()
                    if len(tune)
                    else None
                ),
                "tune_max_start_time": (
                    pd.to_datetime(tune["start_time"], utc=True).max().isoformat()
                    if len(tune)
                    else None
                ),
                "validation_min_start_time": (
                    pd.to_datetime(validation["start_time"], utc=True).min().isoformat()
                    if len(validation)
                    else None
                ),
                "validation_max_start_time": (
                    pd.to_datetime(validation["start_time"], utc=True).max().isoformat()
                    if len(validation)
                    else None
                ),
            }
        ]
    )
    distribution = state_distribution_table(development)
    persistence = persistence_table(development, k=selected_k)
    consecutive = consecutive_persistence_table(development)
    halves = pd.DataFrame(
        [first_half_second_half_correlation(development, CAUSAL_B_COLUMN)]
    )
    rtm = regression_to_mean_table(development, k=selected_k)
    classification = classify_slice14(
        tune_grid=tune_grid,
        validation_grid=val_grid,
        selected_k=selected_k,
        persistence=persistence,
        first_half=halves,
    )

    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    later_than_end = bool((dev_times > pd.Timestamp(end)).any())
    integrity = {
        "development_end": end.isoformat(),
        "tune_end": tune_end.isoformat(),
        "frozen_candidate": FROZEN_CANDIDATE_B,
        "causal_b_column": CAUSAL_B_COLUMN,
        "selected_k": selected_k,
        "frozen_shrinkage_k_constant": FROZEN_SHRINKAGE_K,
        "ti2026_used_for_k": False,
        "holdout_used_for_k": False,
        "holdout_rows_in_development": later_than_end,
        "stratz_called": False,
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "state_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in SLICE14_STATE_COLUMNS
        ),
        "candidate_b_in_feature_columns": FROZEN_CANDIDATE_B in FEATURE_COLUMNS,
        "state_in_all_feature_columns": any(
            name in ALL_FEATURE_COLUMNS for name in SLICE14_STATE_COLUMNS
        ),
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "missing_position_imputed": False,
        "hero_used_in_residualizer": False,
        "global_residualizer_fallback_used": False,
        "player_rating_persisted": False,
        "player_farming_state_persisted": False,
        "model_trained": False,
        "win_model_benchmarked": False,
        "shrinkage_chosen_from_outcomes": False,
        "empirical_bayes_used_for_k": False,
        "population_matches_expected": (
            int(development["match_id"].nunique()) == FROZEN_DEVELOPMENT_MATCH_COUNT
            and len(development) == EXPECTED_DEVELOPMENT_PLAYER_ROWS
        ),
        "n_holdout_excluded": len(holdout),
    }
    return Slice14DiagnosticReport(
        development_end=end,
        tune_end=tune_end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        selected_k=selected_k,
        selected_k_justification=justification,
        classification=classification,
        coverage=coverage,
        residualizer_warmup=warmup,
        split=split,
        shrinkage_grid_tune=tune_grid,
        shrinkage_grid_validation=val_grid,
        empirical_bayes=eb,
        history_bucket_tune=tune_buckets,
        history_bucket_validation=val_buckets,
        state_distribution=distribution,
        persistence=persistence,
        consecutive_persistence=consecutive,
        first_half_second_half=halves,
        regression_to_mean=rtm,
        integrity=integrity,
    )


def slice14_report_to_jsonable(report: Slice14DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 14 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "tune_end": report.tune_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "selected_k": report.selected_k,
        "selected_k_justification": report.selected_k_justification,
        "frozen_candidate": FROZEN_CANDIDATE_B,
        "causal_b_column": CAUSAL_B_COLUMN,
        "shrinkage_grid": list(SHRINKAGE_GRID),
        "classification": _jsonable_value(report.classification),
        "coverage": _jsonable_value(report.coverage),
        "residualizer_warmup": _jsonable_value(report.residualizer_warmup),
        "split": _jsonable_value(report.split),
        "shrinkage_grid_tune": _jsonable_value(report.shrinkage_grid_tune),
        "shrinkage_grid_validation": _jsonable_value(report.shrinkage_grid_validation),
        "empirical_bayes": _jsonable_value(report.empirical_bayes),
        "history_bucket_tune": _jsonable_value(report.history_bucket_tune),
        "history_bucket_validation": _jsonable_value(report.history_bucket_validation),
        "state_distribution": _jsonable_value(report.state_distribution),
        "persistence": _jsonable_value(report.persistence),
        "consecutive_persistence": _jsonable_value(report.consecutive_persistence),
        "first_half_second_half": _jsonable_value(report.first_half_second_half),
        "regression_to_mean": _jsonable_value(report.regression_to_mean),
        "integrity": _jsonable_value(report.integrity),
    }
