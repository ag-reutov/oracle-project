"""Slice 18: leakage-safe historical player combat state.

Research only. Candidate C's *definition* is frozen from Slice 17:

    share = hero_damage / sum(hero_damage | match_id, side)
    C = share - mean(share | explicit position 1–5)

This module does **not** reuse Slice 17's globally estimated position
means. For an appearance at timestamp ``T``:

    share_T uses the same-match team denominator (post-match target)
    position_mean_T = mean(historical share | same position, start_time < T)
    C_T = share_T - position_mean_T

Same-timestamp rows are mutually blind. Insufficient prior history for
that position yields NULL C rather than a 0.20 / global / cross-position
fallback.

Player state for appearance ``M`` uses that player's eligible prior C
observations with ``H.start_time < M.start_time``. Shrinkage is toward
zero because causal C is already position-adjusted:

    shrunk_c = n / (n + k) * prior_mean_c
    shrunk_c = 0 when n = 0

``k`` is chosen on an earlier development timestamp split by how well
prior state predicts the *next causal C*, never from match outcomes.
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
from dota_predictor.training.combat_performance_target import (
    COMBAT_C,
    COMBAT_C_POSITION,
    FROZEN_COMBAT_CANDIDATE,
    complete_side_mask,
    hero_damage_share,
    team_sum,
)
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.player_farming_state import (
    EQUIVALENT_RMSE_RATIO,
    ESTABLISHED_RMSE_CAP_RATIO,
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    HISTORY_N_BUCKETS,
    MIN_LOW_N_VALIDATION_ROWS,
    OVERALL_RMSE_CAP_RATIO,
    SHRINKAGE_GRID,
    apply_farming_shrinkage,
    attach_player_farming_state,
    development_tune_end,
    farming_shrinkage_weight,
    farming_shrunk_b,
    prior_farming_history,
)
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    EXPLICIT_POSITION_NUMBERS,
    _jsonable_value,
    _numeric,
    _pearson,
    _spearman,
    _std,
    build_player_performance_frame,
    explicit_position_mask,
    first_half_second_half_correlation,
    restrict_development,
    slope_coefficient,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    FROZEN_DEVELOPMENT_MATCH_COUNT,
    utc_datetime,
)
from dota_predictor.training.walk_forward import DEFAULT_WALK_FORWARD_CONFIG

__all__ = [
    "CAUSAL_C_COLUMN",
    "COMBAT_SHRINKAGE_GRID",
    "EXPECTED_DEVELOPMENT_PLAYER_ROWS",
    "FROZEN_COMBAT_CANDIDATE",
    "FROZEN_COMBAT_SHRINKAGE_K",
    "SLICE18_STATE_COLUMNS",
    "Slice18DiagnosticReport",
    "attach_causal_candidate_c",
    "attach_player_combat_state",
    "classify_slice18",
    "combat_shrinkage_weight",
    "combat_shrunk_c",
    "prior_combat_history",
    "run_player_combat_state_diagnostics",
    "select_combat_shrinkage_k",
    "slice18_report_to_jsonable",
]


CAUSAL_C_COLUMN = "combat_causal_c"
COMBAT_SHRINKAGE_GRID = SHRINKAGE_GRID
# Methodological freeze after the development-only grid + later-development
# confirmation. ``select_combat_shrinkage_k`` is the authority; this
# constant documents the chosen grid point so later slices do not
# re-search. Combat k is independent of farming ``k=5``. Not a
# production FEATURE_COLUMNS freeze.
FROZEN_COMBAT_SHRINKAGE_K = 20.0
MIN_EMPIRICAL_BAYES_PLAYER_N = 8
MIN_EMPIRICAL_BAYES_PLAYERS = 8
_REPEATABILITY_FLOOR = 0.10
_CORR_WEAK = 0.05
EXPECTED_DEVELOPMENT_PLAYER_ROWS = 59_670
RTM_MIN_EACH = 5
RTM_EXTREME_QUANTILE = 0.90

SLICE18_STATE_COLUMNS: tuple[str, ...] = (
    CAUSAL_C_COLUMN,
    "combat_position_baseline",
    "combat_position_baseline_n",
    "combat_prior_n",
    "combat_prior_sum_c",
    "combat_prior_mean_c",
    "combat_shrinkage_weight",
    "combat_shrunk_c",
)

GATE_A = "A — freeze historical combat state + combat k for later feature evaluation"
GATE_B = (
    "B — combat tendency is real but historical-state shrinkage is not yet defensible"
)
GATE_C = (
    "C — frozen combat candidate does not produce a sufficiently stable "
    "historical player state"
)


@dataclass(frozen=True)
class Slice18DiagnosticReport:
    development_end: datetime
    tune_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    selected_k: float
    selected_k_justification: str
    classification: pd.DataFrame
    coverage: pd.DataFrame
    position_baseline_warmup: pd.DataFrame
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
    farming_relationship: pd.DataFrame
    integrity: dict[str, object]


def attach_causal_candidate_c(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe candidate C using a prior-only position baseline.

    ``hero_damage_share`` uses the current-match team denominator. The
    position mean at ``T`` uses only explicit-position shares with
    ``start_time < T``. Rows at ``T`` never enter one another's baseline.
    When that position has no prior share, ``combat_causal_c`` is NULL.
    """
    out = frame.copy()
    share = hero_damage_share(out)
    out[COMBAT_C] = share
    c = pd.Series(np.nan, index=out.index, dtype=float)
    baseline = pd.Series(np.nan, index=out.index, dtype=float)
    baseline_n = pd.Series(0, index=out.index, dtype=int)
    eligible = explicit_position_mask(out) & share.notna()
    if not bool(eligible.any()):
        out[CAUSAL_C_COLUMN] = c
        out["combat_position_baseline"] = baseline
        out["combat_position_baseline_n"] = baseline_n
        return out

    times = pd.to_datetime(out["start_time"], utc=True).to_numpy()
    positions = _numeric(out["position_number"]).to_numpy(dtype=float)
    shares = share.to_numpy(dtype=float)
    eligible_idx = np.flatnonzero(eligible.to_numpy())
    order = np.argsort(times[eligible_idx], kind="mergesort")
    sorted_idx = eligible_idx[order]
    sorted_times = times[sorted_idx]
    sorted_pos = positions[sorted_idx]
    sorted_share = shares[sorted_idx]
    cuts = np.r_[True, sorted_times[1:] != sorted_times[:-1]]
    starts = np.flatnonzero(cuts)
    bounds = np.r_[starts, len(sorted_idx)]
    pos_sum = np.zeros(6, dtype=float)
    pos_count = np.zeros(6, dtype=int)
    c_vals = np.full(len(out), np.nan, dtype=float)
    base_vals = np.full(len(out), np.nan, dtype=float)
    n_vals = np.zeros(len(out), dtype=int)
    for i in range(len(starts)):
        lo = int(bounds[i])
        hi = int(bounds[i + 1])
        block = sorted_idx[lo:hi]
        block_pos = sorted_pos[lo:hi]
        block_share = sorted_share[lo:hi]
        for j, row_pos in enumerate(block):
            position = int(block_pos[j])
            n_prior = int(pos_count[position])
            n_vals[row_pos] = n_prior
            if n_prior > 0:
                mean = pos_sum[position] / n_prior
                base_vals[row_pos] = mean
                c_vals[row_pos] = block_share[j] - mean
        for j in range(len(block)):
            position = int(block_pos[j])
            pos_sum[position] += float(block_share[j])
            pos_count[position] += 1
    out[CAUSAL_C_COLUMN] = c_vals
    out["combat_position_baseline"] = base_vals
    out["combat_position_baseline_n"] = n_vals
    return out


def prior_combat_history(
    frame: pd.DataFrame, column: str = CAUSAL_C_COLUMN
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Strictly prior player ``n``, ``sum``, and ``mean`` of causal C.

    ``H.start_time < M.start_time``. Same-timestamp appearances are
    mutually blind. Non-finite C does not increment ``n``. Mean is NULL
    when ``n = 0``.
    """
    return prior_farming_history(frame, column)


def combat_shrinkage_weight(prior_n: float, *, k: float) -> float:
    """Evidence fraction ``n / (n + k)``. Zero when ``n = 0``."""
    return farming_shrinkage_weight(prior_n, k=k)


def combat_shrunk_c(mean_c: float | None, prior_n: float, *, k: float) -> float:
    """``n / (n + k) * mean``. Exactly 0 when ``n = 0``."""
    return farming_shrunk_b(mean_c, prior_n, k=k)


def apply_combat_shrinkage(
    mean_c: pd.Series, prior_n: pd.Series, *, k: float
) -> tuple[pd.Series, pd.Series]:
    """Vectorized shrinkage toward zero. ``k = 0`` is allowed."""
    return apply_farming_shrinkage(mean_c, prior_n, k=k)


def attach_player_combat_state(
    frame: pd.DataFrame, *, k: float = FROZEN_COMBAT_SHRINKAGE_K
) -> pd.DataFrame:
    """Causal C plus strictly prior player combat state at shrinkage ``k``.

    Does not recompute causal C when ``combat_causal_c`` is already
    present, so leakage tests can mutate C independently of damage.
    """
    if k < 0.0:
        raise ValueError(f"shrinkage k must be >= 0, got {k}")
    if CAUSAL_C_COLUMN in frame.columns:
        out = frame.copy()
    else:
        out = attach_causal_candidate_c(frame)
    counts, sums, means = prior_combat_history(out, CAUSAL_C_COLUMN)
    shrunk, weight = apply_combat_shrinkage(means, counts, k=k)
    out["combat_prior_n"] = counts
    out["combat_prior_sum_c"] = sums
    out["combat_prior_mean_c"] = means
    out["combat_shrinkage_weight"] = weight
    out["combat_shrunk_c"] = shrunk
    return out


def _finite_pair(
    actual: pd.Series, predicted: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
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
    actual = _numeric(frame[CAUSAL_C_COLUMN])
    n_prior = pd.to_numeric(frame["combat_prior_n"], errors="coerce").fillna(0)
    rows: list[dict[str, object]] = []
    observed = actual.notna()
    overall = _prediction_metrics(actual[observed], predicted[observed])
    rows.append({"k": k, "split": split, "bucket": "all", **overall})
    for label, low, high in HISTORY_N_BUCKETS:
        if high is None:
            mask = observed & (n_prior >= low)
        else:
            mask = observed & (n_prior >= low) & (n_prior <= high)
        metrics = _prediction_metrics(actual[mask], predicted[mask])
        rows.append({"k": k, "split": split, "bucket": label, **metrics})
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
    return float(
        np.average(r[mask].to_numpy(dtype=float), weights=n[mask].to_numpy(dtype=float))
    )


def _grid_summary(bucket_table: pd.DataFrame, *, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for k, group in bucket_table.groupby("k", sort=False):
        by_bucket = group.set_index("bucket")

        def _val(
            name: str, column: str = "rmse", table: pd.DataFrame = by_bucket
        ) -> float:
            if name not in table.index:
                return float("nan")
            return float(table.loc[name, column])

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


def _central_grid_k(values: np.ndarray) -> float:
    """Median of a sorted grid. Even length takes the larger central value."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return 0.0
    mid = int(ordered.size // 2)
    return float(ordered[mid])


def select_combat_shrinkage_k(tune_grid: pd.DataFrame) -> tuple[float, str]:
    """Choose combat ``k`` from the tune-set grid.

    Combat C's RMSE surface is flatter than farming B's, so the Slice 14
    rule of taking the *largest* equivalent ``k`` overshrinks toward
    zero. Among neighbors within 1% of the best low-``n`` RMSE that do
    not inflate established-player RMSE by more than 5%, pick the
    central grid point (larger of the two central values when even).
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
                "Tune split had no observed causal C; defaulting to unshrunk k=0.",
            )
        best_overall = float(overall.min())
        equiv = keep.loc[overall <= best_overall * EQUIVALENT_RMSE_RATIO]
        if equiv.empty:
            return 0.0, "No equivalent overall-RMSE neighbors; defaulting to k=0."
        selected = _central_grid_k(equiv["k"].to_numpy(dtype=float))
        return selected, (
            "Low-n RMSE was unavailable; selected the central k within "
            f"{EQUIVALENT_RMSE_RATIO:.0%} of best overall tune RMSE."
        )
    best_low = float(low_n.min())
    equiv = keep.loc[low_n <= best_low * EQUIVALENT_RMSE_RATIO]
    if equiv.empty:
        return 0.0, "No equivalent low-n neighbors; defaulting to k=0."
    selected = _central_grid_k(equiv["k"].to_numpy(dtype=float))
    k0_low = float(baseline["low_n_rmse"])
    selected_row = keep.loc[keep["k"] == selected].iloc[0]
    selected_low = float(selected_row["low_n_rmse"])
    improved = np.isfinite(k0_low) and np.isfinite(selected_low) and selected_low < k0_low
    if improved:
        justification = (
            f"Selected k={selected:g} as the central plateau point among "
            "tune-set neighbors within 1% of the best low-n RMSE, without "
            "inflating established-player RMSE by more than 5% versus k=0. "
            "Combat RMSE is too flat to take the strongest equivalent k."
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
    than ``min_player_n`` causal C observations are excluded. This is
    not a search and is not used to override the grid.
    """
    values = _numeric(frame[CAUSAL_C_COLUMN])
    work = pd.DataFrame(
        {"player_id": frame["player_id"], "c": values},
        index=frame.index,
    ).dropna()
    empty = {
        "k": float("nan"),
        "within_player_variance": float("nan"),
        "between_player_variance": float("nan"),
        "n_players": 0,
        "n_appearances": 0,
        "min_player_n": min_player_n,
        "used_for_state": False,
    }
    if work.empty:
        return pd.DataFrame([empty])
    grouped = work.groupby("player_id")["c"]
    stats = grouped.agg(n="size", mean="mean", var=lambda s: float(s.var(ddof=1)))
    eligible = stats.loc[stats["n"] >= min_player_n]
    n_players = len(eligible)
    n_appearances = int(eligible["n"].sum()) if n_players else 0
    if n_players < MIN_EMPIRICAL_BAYES_PLAYERS:
        empty["n_players"] = n_players
        empty["n_appearances"] = n_appearances
        return pd.DataFrame([empty])
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
    raw = _numeric(frame["combat_prior_mean_c"]).to_numpy(dtype=float)
    shrunk = _numeric(frame["combat_shrunk_c"]).to_numpy(dtype=float)
    n = pd.to_numeric(frame["combat_prior_n"], errors="coerce").fillna(0)
    n_arr = n.to_numpy()
    rows = [
        _distribution_row(raw, kind="raw_mean_c", subset="n>=1"),
        _distribution_row(shrunk, kind="shrunk_c", subset="all"),
        _distribution_row(shrunk[n_arr == 0], kind="shrunk_c", subset="n=0"),
        _distribution_row(shrunk[n_arr >= 1], kind="shrunk_c", subset="n>=1"),
        _distribution_row(
            raw[((n >= 1) & (n <= 2)).to_numpy()],
            kind="raw_mean_c",
            subset="n=1–2",
        ),
        _distribution_row(
            shrunk[((n >= 1) & (n <= 2)).to_numpy()],
            kind="shrunk_c",
            subset="n=1–2",
        ),
        _distribution_row(
            raw[(n >= 21).to_numpy()],
            kind="raw_mean_c",
            subset="n>=21",
        ),
        _distribution_row(
            shrunk[(n >= 21).to_numpy()],
            kind="shrunk_c",
            subset="n>=21",
        ),
    ]
    return pd.DataFrame(rows)


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    share = (
        _numeric(frame[COMBAT_C])
        if COMBAT_C in frame.columns
        else hero_damage_share(frame)
    )
    complete = complete_side_mask(frame, "hero_damage")
    team = team_sum(frame, "hero_damage")
    explicit = explicit_position_mask(frame)
    causal = _numeric(frame[CAUSAL_C_COLUMN]).notna()
    eligible = explicit & share.notna()
    n = pd.to_numeric(frame["combat_prior_n"], errors="coerce").fillna(0)
    n_values = n.to_numpy(dtype=int)
    return pd.DataFrame(
        [
            {
                "n_eligible_explicit_position": int(explicit.sum()),
                "n_share_available": int(share.notna().sum()),
                "n_causal_c_available": int(causal.sum()),
                "n_position_baseline_warmup_loss": int((eligible & ~causal).sum()),
                "n_side_damage_incomplete": int((~complete).sum()),
                "n_side_damage_sum_zero": int(
                    (complete & team.notna() & (team == 0.0)).sum()
                ),
                "n_position_missing_invalid": int((~explicit).sum()),
                "n_causal_position_baseline_unavailable": int(
                    (eligible & ~causal).sum()
                ),
                "n_prior_0": int((n == 0).sum()),
                "n_prior_ge_1": int((n >= 1).sum()),
                "n_prior_ge_5": int((n >= 5).sum()),
                "n_prior_ge_10": int((n >= 10).sum()),
                "n_prior_ge_20": int((n >= 20).sum()),
                "n_players": int(frame["player_id"].nunique()),
                "n_players_with_causal_c": int(
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


def _first_ready_iso(times: pd.Series) -> str | None:
    if times.empty or times.isna().all():
        return None
    first = pd.to_datetime(times, utc=True).min()
    if pd.isna(first):
        return None
    return utc_datetime(first).isoformat()


def position_baseline_warmup_table(frame: pd.DataFrame) -> pd.DataFrame:
    share = (
        _numeric(frame[COMBAT_C])
        if COMBAT_C in frame.columns
        else hero_damage_share(frame)
    )
    explicit = explicit_position_mask(frame)
    eligible = explicit & share.notna()
    causal = _numeric(frame[CAUSAL_C_COLUMN]).notna()
    positions = _numeric(frame["position_number"])
    rows: list[dict[str, object]] = []
    n_eligible = int(eligible.sum())
    n_causal = int((eligible & causal).sum())
    n_loss = int((eligible & ~causal).sum())
    ready = frame.loc[eligible & causal]
    rows.append(
        {
            "position": "all",
            "n_eligible": n_eligible,
            "n_causal_c": n_causal,
            "n_warmup_loss": n_loss,
            "warmup_loss_fraction": (
                float(n_loss) / float(n_eligible) if n_eligible else float("nan")
            ),
            "first_ready_start_time": _first_ready_iso(ready["start_time"]),
            "min_baseline_n_when_ready": (
                int(
                    pd.to_numeric(
                        ready["combat_position_baseline_n"], errors="coerce"
                    ).min()
                )
                if not ready.empty
                else 0
            ),
            "global_mean_fallback_used": False,
            "uniform_share_fallback_used": False,
            "cross_position_fallback_used": False,
        }
    )
    for number in EXPLICIT_POSITION_NUMBERS:
        at_pos = eligible & (positions == float(number))
        ready_pos = at_pos & causal
        n_pos = int(at_pos.sum())
        n_ready = int(ready_pos.sum())
        n_pos_loss = int((at_pos & ~causal).sum())
        rows.append(
            {
                "position": int(number),
                "n_eligible": n_pos,
                "n_causal_c": n_ready,
                "n_warmup_loss": n_pos_loss,
                "warmup_loss_fraction": (
                    float(n_pos_loss) / float(n_pos) if n_pos else float("nan")
                ),
                "first_ready_start_time": _first_ready_iso(
                    frame.loc[ready_pos, "start_time"]
                ),
                "min_baseline_n_when_ready": (
                    int(
                        pd.to_numeric(
                            frame.loc[ready_pos, "combat_position_baseline_n"],
                            errors="coerce",
                        ).min()
                    )
                    if n_ready
                    else 0
                ),
                "global_mean_fallback_used": False,
                "uniform_share_fallback_used": False,
                "cross_position_fallback_used": False,
            }
        )
    return pd.DataFrame(rows)


def persistence_table(frame: pd.DataFrame, *, k: float) -> pd.DataFrame:
    actual = _numeric(frame[CAUSAL_C_COLUMN])
    raw = _numeric(frame["combat_prior_mean_c"])
    shrunk = _numeric(frame["combat_shrunk_c"])
    n = pd.to_numeric(frame["combat_prior_n"], errors="coerce").fillna(0)
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
    """Same-player consecutive-appearance persistence of C and of state.

    Expanding historical states overlap heavily; consecutive-state
    correlation is therefore not strong independent evidence of
    persistence.
    """
    values = _numeric(frame[CAUSAL_C_COLUMN])
    work = frame.loc[
        :,
        [
            "player_id",
            "start_time",
            CAUSAL_C_COLUMN,
            "combat_shrunk_c",
            "combat_prior_n",
        ],
    ].copy()
    work[CAUSAL_C_COLUMN] = values
    work["combat_shrunk_c"] = _numeric(work["combat_shrunk_c"])
    work["start_time"] = pd.to_datetime(work["start_time"], utc=True)
    c_now: list[float] = []
    c_next: list[float] = []
    state_now: list[float] = []
    state_next: list[float] = []
    for _player_id, group in work.groupby("player_id", sort=False):
        ordered = group.sort_values("start_time", kind="mergesort")
        stamps = ordered["start_time"].to_numpy()
        c_vals = ordered[CAUSAL_C_COLUMN].to_numpy(dtype=float)
        st = ordered["combat_shrunk_c"].to_numpy(dtype=float)
        for i in range(len(ordered) - 1):
            if not (stamps[i] < stamps[i + 1]):
                continue
            if np.isfinite(c_vals[i]) and np.isfinite(c_vals[i + 1]):
                c_now.append(float(c_vals[i]))
                c_next.append(float(c_vals[i + 1]))
            if np.isfinite(st[i]) and np.isfinite(st[i + 1]):
                state_now.append(float(st[i]))
                state_next.append(float(st[i + 1]))
    return pd.DataFrame(
        [
            {
                "kind": "causal_c_consecutive",
                "n_pairs": len(c_now),
                "pearson": _pearson(pd.Series(c_now), pd.Series(c_next))
                if c_now
                else float("nan"),
                "spearman": _spearman(pd.Series(c_now), pd.Series(c_next))
                if c_now
                else float("nan"),
                "note": "Independent consecutive target observations.",
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
                "note": (
                    "Expanding states overlap heavily; not strong independent "
                    "evidence of persistence."
                ),
            },
        ]
    )


def regression_to_mean_table(
    frame: pd.DataFrame,
    *,
    k: float,
    min_each: int = RTM_MIN_EACH,
) -> pd.DataFrame:
    """Extreme early mean C versus later mean C, raw vs shrunk."""
    values = _numeric(frame[CAUSAL_C_COLUMN])
    work = frame.loc[values.notna(), ["player_id", "start_time", CAUSAL_C_COLUMN]].copy()
    work[CAUSAL_C_COLUMN] = values.loc[work.index]
    work["start_time"] = pd.to_datetime(work["start_time"], utc=True)
    early_means: list[float] = []
    late_means: list[float] = []
    early_n: list[int] = []
    for _player_id, group in work.groupby("player_id", sort=False):
        ordered = group.sort_values("start_time", kind="mergesort")
        n = len(ordered)
        split = n // 2
        if split < min_each or (n - split) < min_each:
            continue
        early = ordered.iloc[:split][CAUSAL_C_COLUMN].to_numpy(dtype=float)
        late = ordered.iloc[split:][CAUSAL_C_COLUMN].to_numpy(dtype=float)
        early_means.append(float(early.mean()))
        late_means.append(float(late.mean()))
        early_n.append(int(early.size))
    empty_row = {
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
        "extreme_abs_threshold": float("nan"),
    }
    if not early_means:
        appearance_rows = _low_history_extreme_rows(frame, k=k)
        return pd.DataFrame([empty_row, *appearance_rows])
    early_arr = np.asarray(early_means, dtype=float)
    late_arr = np.asarray(late_means, dtype=float)
    n_arr = np.asarray(early_n, dtype=float)
    shrunk_early = np.array(
        [
            combat_shrunk_c(float(mean), float(n), k=k)
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
        late = late_arr[mask]
        if e.size == 0:
            row = dict(empty_row)
            row["subset"] = subset
            row["extreme_abs_threshold"] = threshold
            return row
        ratio = np.divide(
            late, e, out=np.full(e.shape, np.nan), where=np.abs(e) > 1e-12
        )
        return {
            "subset": subset,
            "n_players": int(e.size),
            "k": k,
            "early_raw_mean": float(e.mean()),
            "early_shrunk_mean": float(s.mean()),
            "late_mean": float(late.mean()),
            "rmse_raw": float(np.sqrt(np.mean((late - e) ** 2))),
            "rmse_shrunk": float(np.sqrt(np.mean((late - s) ** 2))),
            "mae_raw": float(np.mean(np.abs(late - e))),
            "mae_shrunk": float(np.mean(np.abs(late - s))),
            "mean_abs_early_raw": float(np.mean(np.abs(e))),
            "mean_abs_early_shrunk": float(np.mean(np.abs(s))),
            "mean_abs_late": float(np.mean(np.abs(late))),
            "mean_late_over_early_raw": float(np.nanmean(ratio)),
            "extreme_abs_threshold": threshold,
        }

    player_rows = [
        _block(np.ones(early_arr.shape, dtype=bool), "all_paired"),
        _block(extreme, "extreme_early_abs_ge_p90"),
    ]
    return pd.DataFrame(player_rows + _low_history_extreme_rows(frame, k=k))


def _low_history_extreme_rows(frame: pd.DataFrame, *, k: float) -> list[dict[str, object]]:
    """Appearance-level extremes among n=1–2 before versus after shrinkage."""
    n = pd.to_numeric(frame["combat_prior_n"], errors="coerce").fillna(0)
    raw = _numeric(frame["combat_prior_mean_c"])
    shrunk = _numeric(frame["combat_shrunk_c"])
    low = (n >= 1) & (n <= 2) & raw.notna()
    raw_v = raw[low].to_numpy(dtype=float)
    shrunk_v = shrunk[low].to_numpy(dtype=float)
    if raw_v.size == 0:
        return [
            {
                "subset": "low_history_n_1_2",
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
                "extreme_abs_threshold": float("nan"),
            }
        ]
    abs_raw = np.abs(raw_v)
    threshold = float(np.quantile(abs_raw, RTM_EXTREME_QUANTILE))
    extreme = abs_raw >= threshold

    def _appearance_block(mask: np.ndarray, subset: str) -> dict[str, object]:
        r = raw_v[mask]
        s = shrunk_v[mask]
        return {
            "subset": subset,
            "n_players": int(r.size),
            "k": k,
            "early_raw_mean": float(r.mean()),
            "early_shrunk_mean": float(s.mean()),
            "late_mean": float("nan"),
            "rmse_raw": float("nan"),
            "rmse_shrunk": float("nan"),
            "mae_raw": float("nan"),
            "mae_shrunk": float("nan"),
            "mean_abs_early_raw": float(np.mean(np.abs(r))),
            "mean_abs_early_shrunk": float(np.mean(np.abs(s))),
            "mean_abs_late": float("nan"),
            "mean_late_over_early_raw": float("nan"),
            "extreme_abs_threshold": threshold,
        }

    return [
        _appearance_block(np.ones(raw_v.shape, dtype=bool), "low_history_n_1_2"),
        _appearance_block(extreme, "low_history_n_1_2_extreme_p90"),
    ]


def farming_relationship_table(
    combat: pd.DataFrame, farming: pd.DataFrame
) -> pd.DataFrame:
    """Diagnostic Pearson of shrunk combat vs frozen farming state.

    Does not residualize, combine, or alter ``k``.
    """
    left = combat.loc[
        :,
        [
            "match_id",
            "player_id",
            "position_number",
            "combat_shrunk_c",
            "combat_prior_n",
            CAUSAL_C_COLUMN,
        ],
    ].copy()
    right = farming.loc[
        :,
        ["match_id", "player_id", "farming_shrunk_b", "farming_prior_n"],
    ].copy()
    joined = left.merge(right, on=["match_id", "player_id"], how="inner")
    both = (
        _numeric(joined["combat_shrunk_c"]).notna()
        & _numeric(joined["farming_shrunk_b"]).notna()
    )
    rows: list[dict[str, object]] = [
        {
            "subset": "all_both_states",
            "n_rows": int(both.sum()),
            "n_players": int(joined.loc[both, "player_id"].nunique())
            if bool(both.any())
            else 0,
            "pearson": _pearson(
                joined.loc[both, "combat_shrunk_c"],
                joined.loc[both, "farming_shrunk_b"],
            ),
            "spearman": _spearman(
                joined.loc[both, "combat_shrunk_c"],
                joined.loc[both, "farming_shrunk_b"],
            ),
        }
    ]
    positions = _numeric(joined["position_number"])
    for number in EXPLICIT_POSITION_NUMBERS:
        mask = both & (positions == float(number))
        rows.append(
            {
                "subset": f"position_{number}",
                "n_rows": int(mask.sum()),
                "n_players": int(joined.loc[mask, "player_id"].nunique())
                if int(mask.sum())
                else 0,
                "pearson": _pearson(
                    joined.loc[mask, "combat_shrunk_c"],
                    joined.loc[mask, "farming_shrunk_b"],
                ),
                "spearman": _spearman(
                    joined.loc[mask, "combat_shrunk_c"],
                    joined.loc[mask, "farming_shrunk_b"],
                ),
            }
        )
    history = both & (pd.to_numeric(joined["combat_prior_n"], errors="coerce") >= 1)
    rows.append(
        {
            "subset": "combat_n_ge_1",
            "n_rows": int(history.sum()),
            "n_players": int(joined.loc[history, "player_id"].nunique())
            if bool(history.any())
            else 0,
            "pearson": _pearson(
                joined.loc[history, "combat_shrunk_c"],
                joined.loc[history, "farming_shrunk_b"],
            ),
            "spearman": _spearman(
                joined.loc[history, "combat_shrunk_c"],
                joined.loc[history, "farming_shrunk_b"],
            ),
        }
    )
    return pd.DataFrame(rows)


def classify_slice18(
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
    predicts later causal C and does not inflate overall or
    established-player RMSE. Does not use win rate, Elo, log loss, or
    farming as a selection criterion.
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
    ) and (not np.isfinite(persist_r) or persist_r < _CORR_WEAK)

    if weak:
        classification = "C"
        rationale = (
            "Causal candidate C does not persist as a player-level "
            "combat tendency once the position baseline itself is causal."
        )
        next_slice = (
            "Do not freeze a historical combat state. Treat damage-share "
            "residuals as match-level outcomes, not a player trait."
        )
        gate = GATE_C
    elif (
        selected_k > 0.0
        and signal
        and low_n_confirmed
        and established_ok
        and overall_ok
    ):
        classification = "A"
        rationale = (
            "Prior-only causal C is a repeatable player combat tendency, "
            f"and k={selected_k:g} improves low-history estimates on the "
            f"{low_n_source} split without destroying later-development "
            "signal for established players."
        )
        next_slice = (
            "Freeze the historical combat state and k for a later "
            "feature-evaluation slice. Do not aggregate to team features "
            "or run a win-model benchmark now."
        )
        gate = GATE_A
    else:
        classification = "B"
        rationale = (
            "Causal C shows player-level persistence, but the shrinkage "
            "constant is not yet a broad, stable improvement over the "
            "unshrunk mean."
        )
        next_slice = (
            "Keep the causal C construction and unshrunk prior mean; do "
            "not freeze k. Do not run a win-model benchmark."
        )
        gate = GATE_B
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "gate": gate,
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
    frame: pd.DataFrame, *, split: str, ks: tuple[float, ...] = COMBAT_SHRINKAGE_GRID
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bucket_parts: list[pd.DataFrame] = []
    for k in ks:
        shrunk, _weight = apply_combat_shrinkage(
            frame["combat_prior_mean_c"], frame["combat_prior_n"], k=k
        )
        bucket_parts.append(_bucket_metrics(frame, shrunk, k=k, split=split))
    buckets = pd.concat(bucket_parts, ignore_index=True)
    return _grid_summary(buckets, split=split), buckets


def run_player_combat_state_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice18DiagnosticReport:
    """Development-only Slice 18 combat-state research. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    development = attach_causal_candidate_c(development)
    counts, sums, means = prior_combat_history(development, CAUSAL_C_COLUMN)
    development["combat_prior_n"] = counts
    development["combat_prior_sum_c"] = sums
    development["combat_prior_mean_c"] = means

    tune_end = development_tune_end(development["start_time"], development_end=end)
    dev_times = pd.to_datetime(development["start_time"], utc=True)
    tune_mask = dev_times <= pd.Timestamp(tune_end)
    val_mask = (dev_times > pd.Timestamp(tune_end)) & (dev_times <= pd.Timestamp(end))
    tune = development.loc[tune_mask].copy()
    validation = development.loc[val_mask].copy()

    tune_grid, tune_buckets = _evaluate_k_grid(tune, split="tune")
    selected_k, justification = select_combat_shrinkage_k(tune_grid)
    val_grid, val_buckets = _evaluate_k_grid(validation, split="validation")

    shrunk, weight = apply_combat_shrinkage(
        development["combat_prior_mean_c"],
        development["combat_prior_n"],
        k=selected_k,
    )
    development["combat_shrinkage_weight"] = weight
    development["combat_shrunk_c"] = shrunk

    farming = attach_player_farming_state(development, k=FROZEN_SHRINKAGE_K)
    farming_rel = farming_relationship_table(development, farming)

    eb = empirical_bayes_k(tune)
    coverage = coverage_table(development)
    warmup = position_baseline_warmup_table(development)
    split = pd.DataFrame(
        [
            {
                "tune_end": tune_end.isoformat(),
                "development_end": end.isoformat(),
                "train_fraction_of_past": (
                    DEFAULT_WALK_FORWARD_CONFIG.train_fraction_of_past
                ),
                "n_tune_rows": len(tune),
                "n_tune_players": int(tune["player_id"].nunique()) if len(tune) else 0,
                "n_tune_matches": int(tune["match_id"].nunique()) if len(tune) else 0,
                "n_validation_rows": len(validation),
                "n_validation_players": (
                    int(validation["player_id"].nunique()) if len(validation) else 0
                ),
                "n_validation_matches": (
                    int(validation["match_id"].nunique()) if len(validation) else 0
                ),
                "n_holdout_excluded": len(holdout),
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
        [first_half_second_half_correlation(development, CAUSAL_C_COLUMN)]
    )
    rtm = regression_to_mean_table(development, k=selected_k)
    classification = classify_slice18(
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
        "frozen_combat_candidate": FROZEN_COMBAT_CANDIDATE,
        "slice17_candidate_unchanged": FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION,
        "causal_c_column": CAUSAL_C_COLUMN,
        "selected_k": selected_k,
        "frozen_combat_shrinkage_k_constant": FROZEN_COMBAT_SHRINKAGE_K,
        "farming_candidate_b": FROZEN_CANDIDATE_B,
        "farming_candidate_b_unchanged": FROZEN_CANDIDATE_B == CANDIDATE_B,
        "farming_frozen_shrinkage_k": FROZEN_SHRINKAGE_K,
        "farming_k_is_5": FROZEN_SHRINKAGE_K == 5.0,
        "combat_k_copied_from_farming": selected_k == FROZEN_SHRINKAGE_K,
        "ti2026_used_for_k": False,
        "holdout_used_for_k": False,
        "holdout_used_for_validation": False,
        "holdout_used_for_eb": False,
        "holdout_rows_in_development": later_than_end,
        "stratz_called": False,
        "ingestion_modified": False,
        "schema_modified": False,
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "state_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in SLICE18_STATE_COLUMNS
        ),
        "candidate_c_in_feature_columns": COMBAT_C_POSITION in FEATURE_COLUMNS,
        "state_in_all_feature_columns": any(
            name in ALL_FEATURE_COLUMNS for name in SLICE18_STATE_COLUMNS
        ),
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "missing_position_imputed": False,
        "hero_used_in_state": False,
        "current_result_used_in_state": False,
        "farming_used_as_input": False,
        "team_combat_feature_created": False,
        "global_position_mean_fallback_used": False,
        "uniform_share_fallback_used": False,
        "player_rating_persisted": False,
        "player_combat_state_persisted": False,
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
    return Slice18DiagnosticReport(
        development_end=end,
        tune_end=tune_end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        selected_k=selected_k,
        selected_k_justification=justification,
        classification=classification,
        coverage=coverage,
        position_baseline_warmup=warmup,
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
        farming_relationship=farming_rel,
        integrity=integrity,
    )


def slice18_report_to_jsonable(report: Slice18DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 18 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "tune_end": report.tune_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "selected_k": report.selected_k,
        "selected_k_justification": report.selected_k_justification,
        "frozen_combat_candidate": FROZEN_COMBAT_CANDIDATE,
        "causal_c_column": CAUSAL_C_COLUMN,
        "shrinkage_grid": list(COMBAT_SHRINKAGE_GRID),
        "frozen_combat_shrinkage_k": FROZEN_COMBAT_SHRINKAGE_K,
        "farming_frozen_shrinkage_k": FROZEN_SHRINKAGE_K,
        "classification": _jsonable_value(report.classification),
        "coverage": _jsonable_value(report.coverage),
        "position_baseline_warmup": _jsonable_value(report.position_baseline_warmup),
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
        "farming_relationship": _jsonable_value(report.farming_relationship),
        "integrity": _jsonable_value(report.integrity),
    }
