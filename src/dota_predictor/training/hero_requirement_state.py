"""Slice 22: leakage-safe historical hero×position requirement states.

Research only. This module does not persist a hero rating, does not add
production features, does not construct a player×hero fit score, does
not aggregate to team, and does not train a win model. Requirement
columns never enter ``FEATURE_COLUMNS``.

Question
--------
Can we estimate stable causal historical hero×position requirement
states, with defensible shrinkage, for later player↔hero fit
construction?

These are **hero-role requirements / identities**, not strength
ratings. Shrinkage is toward zero because both frozen targets are
already expressed relative to a position baseline.

Frozen Slice 21 targets
-----------------------
Farming: ``farming_causal_b``, keyed by hero_id × explicit position 1–5.
Combat: ``combat_causal_c``, keyed by hero_id × explicit position 1–5.

Primary state semantics
-----------------------
For appearance ``(P, H, R, T)`` the leave-current-player-out (LPO)
history is:

    Q.player_id != P
    Q.hero_id = H
    Q.position = R
    Q.start_time < T
    finite frozen target

Same-timestamp rows are mutually blind. Future and holdout rows never
enter. There is no fallback to a full-development mean.

Current post-match position is used **diagnostically**. It is not a
PRE_DRAFT production input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from dota_predictor.data.canonical_schema import MATCH_PLAYER_BOX_SCORE_COLUMNS
from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.features.team_elo import DEFAULT_ELO_CONFIG, EloConfig
from dota_predictor.training.combat_performance_target import (
    COMBAT_C_POSITION,
    FROZEN_COMBAT_CANDIDATE,
)
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.hero_performance_profile import (
    HERO_COMBAT_PROFILE_KEY,
    HERO_COMBAT_PROFILE_TARGET,
    HERO_FARMING_PROFILE_KEY,
    HERO_FARMING_PROFILE_TARGET,
    MIN_HALF_HERO_POSITION,
    PLAYER_X_HERO_FIT_NAMES,
    SPECIALIST_TOP_SHARE,
    _adjacent_window_stability,
    assign_chronological_blocks,
    attach_hero_profile_observations,
    group_split_half,
)
from dota_predictor.training.player_combat_state import (
    CAUSAL_C_COLUMN,
    FROZEN_COMBAT_SHRINKAGE_K,
)
from dota_predictor.training.player_farming_state import (
    CAUSAL_B_COLUMN,
    EQUIVALENT_RMSE_RATIO,
    ESTABLISHED_RMSE_CAP_RATIO,
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    HISTORY_N_BUCKETS,
    MIN_LOW_N_VALIDATION_ROWS,
    OVERALL_RMSE_CAP_RATIO,
    apply_farming_shrinkage,
    development_tune_end,
    farming_shrinkage_weight,
    farming_shrunk_b,
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
    restrict_development,
    slope_coefficient,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    utc_datetime,
)

__all__ = [
    "CLASSIFICATION_A",
    "CLASSIFICATION_B",
    "CLASSIFICATION_C",
    "FROZEN_HERO_COMBAT_SHRINKAGE_K",
    "FROZEN_HERO_FARM_SHRINKAGE_K",
    "HERO_COMBAT_SHRINKAGE_GRID",
    "HERO_FARM_SHRINKAGE_GRID",
    "PREFERRED_TUNE_END",
    "SLICE22_STATE_COLUMNS",
    "Slice22DiagnosticReport",
    "attach_hero_requirement_state",
    "classify_slice22",
    "hero_requirement_shrinkage_weight",
    "hero_requirement_shrunk",
    "prior_hero_position_history",
    "run_hero_requirement_state_diagnostics",
    "select_hero_combat_shrinkage_k",
    "select_hero_farm_shrinkage_k",
    "slice22_report_to_jsonable",
]


PREFERRED_TUNE_END = datetime(2026, 2, 7, 15, 4, 47, tzinfo=UTC)
HERO_FARM_SHRINKAGE_GRID: tuple[float, ...] = (
    0.0,
    2.0,
    5.0,
    10.0,
    20.0,
    40.0,
    80.0,
)
HERO_COMBAT_SHRINKAGE_GRID: tuple[float, ...] = HERO_FARM_SHRINKAGE_GRID
# Methodological freeze after Slice 22 development diagnostics.
# Selection remains the authority of ``select_hero_*_shrinkage_k``;
# these constants record the independently chosen grid points (both
# happened to be 2). Research-state shrinkage only; not production
# features and not a player×hero fit score.
FROZEN_HERO_FARM_SHRINKAGE_K = 2.0
FROZEN_HERO_COMBAT_SHRINKAGE_K = 2.0
MIN_EMPIRICAL_BAYES_CELL_N = 10
MIN_EMPIRICAL_BAYES_CELLS = 10
MIN_BLOCK_PROFILE_N = 10
RTM_MIN_EACH = 5
RTM_EXTREME_QUANTILE = 0.90
PATCH_MIN_VERSION_N = 50
PATCH_RMSE_RATIO = 1.25
_REPEATABILITY_FLOOR = 0.10
_LPO_DESTROY_DELTA = 0.15
_PATCH_CORR_SOFT = 0.10
UNIQUE_PLAYER_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1–2", 1, 2),
    ("3–5", 3, 5),
    ("6–10", 6, 10),
    (">10", 11, None),
)
COVERAGE_N_THRESHOLDS: tuple[int, ...] = (1, 5, 10, 20, 50)

SLICE22_STATE_COLUMNS: tuple[str, ...] = (
    "hero_farming_prior_n",
    "hero_farming_prior_sum_b",
    "hero_farming_prior_mean_b",
    "hero_farming_shrunk_b",
    "hero_combat_prior_n",
    "hero_combat_prior_sum_c",
    "hero_combat_prior_mean_c",
    "hero_combat_shrunk_c",
)

CLASSIFICATION_A = (
    "A — freeze LPO hero farming/combat requirement states + shrinkage "
    "for later fit construction"
)
CLASSIFICATION_B = (
    "B — hero requirement states are real but one dimension or temporal "
    "assumption needs another diagnostic slice"
)
CLASSIFICATION_C = (
    "C — frozen hero×position targets do not produce usable causal "
    "requirement states"
)

HERO_POSITION_GROUP = ("hero_id", "position_number")


@dataclass(frozen=True)
class Slice22DiagnosticReport:
    development_end: datetime
    tune_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    selected_k_farm: float
    selected_k_combat: float
    selected_k_farm_justification: str
    selected_k_combat_justification: str
    farming_semantics: dict[str, object]
    combat_semantics: dict[str, object]
    classification: pd.DataFrame
    split: pd.DataFrame
    farming_coverage: pd.DataFrame
    combat_coverage: pd.DataFrame
    farming_inclusive_vs_lpo: pd.DataFrame
    combat_inclusive_vs_lpo: pd.DataFrame
    farming_grid_tune: pd.DataFrame
    farming_grid_validation: pd.DataFrame
    combat_grid_tune: pd.DataFrame
    combat_grid_validation: pd.DataFrame
    farming_empirical_bayes: pd.DataFrame
    combat_empirical_bayes: pd.DataFrame
    farming_history_bucket_tune: pd.DataFrame
    farming_history_bucket_validation: pd.DataFrame
    combat_history_bucket_tune: pd.DataFrame
    combat_history_bucket_validation: pd.DataFrame
    farming_unique_player: pd.DataFrame
    combat_unique_player: pd.DataFrame
    farming_specialist: pd.DataFrame
    combat_specialist: pd.DataFrame
    farming_state_distribution: pd.DataFrame
    combat_state_distribution: pd.DataFrame
    farming_persistence: pd.DataFrame
    combat_persistence: pd.DataFrame
    farming_split_half: pd.DataFrame
    combat_split_half: pd.DataFrame
    farming_temporal_blocks: pd.DataFrame
    combat_temporal_blocks: pd.DataFrame
    farming_patch: pd.DataFrame
    combat_patch: pd.DataFrame
    farming_regression_to_mean: pd.DataFrame
    combat_regression_to_mean: pd.DataFrame
    player_state_relationship: pd.DataFrame
    cross_dimension: pd.DataFrame
    integrity: dict[str, object]


def hero_requirement_shrinkage_weight(prior_n: float, *, k: float) -> float:
    """Evidence fraction ``n / (n + k)``. Zero when ``n = 0``."""
    return farming_shrinkage_weight(prior_n, k=k)


def hero_requirement_shrunk(
    mean: float | None, prior_n: float, *, k: float
) -> float:
    """``n / (n + k) * mean``. Exactly 0 when ``n = 0``."""
    return farming_shrunk_b(mean, prior_n, k=k)


def apply_hero_requirement_shrinkage(
    mean: pd.Series, prior_n: pd.Series, *, k: float
) -> tuple[pd.Series, pd.Series]:
    """Vectorized shrinkage toward zero. ``k = 0`` is allowed."""
    return apply_farming_shrinkage(mean, prior_n, k=k)


def _hero_position_keys(frame: pd.DataFrame) -> pd.Series:
    """``(hero_id, position)`` keys for explicit positions 1–5 only."""
    hero = _numeric(frame["hero_id"])
    eligible = explicit_position_mask(frame) & hero.notna()
    keys = pd.Series(pd.NA, index=frame.index, dtype="object")
    pos = _numeric(frame["position_number"])
    keys.loc[eligible] = [
        (int(h), int(p))
        for h, p in zip(
            hero.loc[eligible].to_numpy(dtype=float),
            pos.loc[eligible].to_numpy(dtype=float),
            strict=True,
        )
    ]
    return keys


def prior_hero_position_history(
    frame: pd.DataFrame,
    column: str,
    *,
    leave_player_out: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Strictly prior hero×position ``n``, ``sum``, ``mean``, unique players, top share.

    History is ``start_time < T``, same hero, same explicit position 1–5.
    Non-finite values do not increment ``n``. Same-timestamp rows are
    mutually blind. When ``leave_player_out`` is true, the current
    player's own earlier observations on the same key are excluded.
    Mean and top-player share are NULL when ``n = 0``.
    """
    values = _numeric(frame[column])
    keys = _hero_position_keys(frame)
    n_out = pd.Series(0, index=frame.index, dtype=int)
    sum_out = pd.Series(0.0, index=frame.index, dtype=float)
    mean_out = pd.Series(np.nan, index=frame.index, dtype=float)
    unique_out = pd.Series(0, index=frame.index, dtype=int)
    top_out = pd.Series(np.nan, index=frame.index, dtype=float)
    if frame.empty:
        return n_out, sum_out, mean_out, unique_out, top_out

    times = pd.to_datetime(frame["start_time"], utc=True).to_numpy()
    players = _numeric(frame["player_id"]).to_numpy(dtype=float)
    # Eligible *history contributors* have a key and a finite target.
    contributor = values.notna() & keys.notna()
    # State is attached to every explicit hero×position appearance,
    # including those whose current target is still NULL.
    attachable = keys.notna()
    if not bool(attachable.any()):
        return n_out, sum_out, mean_out, unique_out, top_out

    attach_idx = np.flatnonzero(attachable.to_numpy())
    order = np.argsort(times[attach_idx], kind="mergesort")
    sorted_idx = attach_idx[order]
    sorted_times = times[sorted_idx]
    sorted_values = values.to_numpy(dtype=float)[sorted_idx]
    sorted_keys = keys.to_numpy()[sorted_idx]
    sorted_players = players[sorted_idx]
    sorted_contrib = contributor.to_numpy()[sorted_idx]
    cuts = np.r_[True, sorted_times[1:] != sorted_times[:-1]]
    starts = np.flatnonzero(cuts)
    bounds = np.r_[starts, len(sorted_idx)]

    group_sum: dict[tuple[int, int], float] = {}
    group_n: dict[tuple[int, int], int] = {}
    player_sum: dict[tuple[int, int, int], float] = {}
    player_n: dict[tuple[int, int, int], int] = {}
    player_counts: dict[tuple[int, int], dict[int, int]] = {}

    n_vals = np.zeros(len(frame), dtype=int)
    sum_vals = np.zeros(len(frame), dtype=float)
    mean_vals = np.full(len(frame), np.nan, dtype=float)
    unique_vals = np.zeros(len(frame), dtype=int)
    top_vals = np.full(len(frame), np.nan, dtype=float)

    for i in range(len(starts)):
        lo = int(bounds[i])
        hi = int(bounds[i + 1])
        for j in range(lo, hi):
            key = sorted_keys[j]
            player = int(sorted_players[j])
            n_prior = int(group_n.get(key, 0))
            total = float(group_sum.get(key, 0.0))
            counts = player_counts.get(key, {})
            if leave_player_out:
                own_key = (key[0], key[1], player)
                n_prior -= int(player_n.get(own_key, 0))
                total -= float(player_sum.get(own_key, 0.0))
            row_pos = int(sorted_idx[j])
            n_vals[row_pos] = n_prior
            sum_vals[row_pos] = total
            if n_prior > 0:
                mean_vals[row_pos] = total / n_prior
            unique_lpo = 0
            max_other = 0
            for pid, count in counts.items():
                if leave_player_out and pid == player:
                    continue
                if count > 0:
                    unique_lpo += 1
                    max_other = max(max_other, count)
            unique_vals[row_pos] = unique_lpo
            if n_prior > 0:
                top_vals[row_pos] = max_other / n_prior
        for j in range(lo, hi):
            if not bool(sorted_contrib[j]):
                continue
            key = sorted_keys[j]
            player = int(sorted_players[j])
            value = float(sorted_values[j])
            group_sum[key] = float(group_sum.get(key, 0.0)) + value
            group_n[key] = int(group_n.get(key, 0)) + 1
            own_key = (key[0], key[1], player)
            player_sum[own_key] = float(player_sum.get(own_key, 0.0)) + value
            player_n[own_key] = int(player_n.get(own_key, 0)) + 1
            cell = player_counts.setdefault(key, {})
            cell[player] = int(cell.get(player, 0)) + 1

    n_out.loc[:] = n_vals
    sum_out.loc[:] = sum_vals
    mean_out.loc[:] = mean_vals
    unique_out.loc[:] = unique_vals
    top_out.loc[:] = top_vals
    return n_out, sum_out, mean_out, unique_out, top_out


def attach_hero_requirement_state(
    frame: pd.DataFrame,
    *,
    k_farm: float = FROZEN_HERO_FARM_SHRINKAGE_K,
    k_combat: float = FROZEN_HERO_COMBAT_SHRINKAGE_K,
) -> pd.DataFrame:
    """LPO hero×position requirement state plus inclusive diagnostic columns.

    Does not recompute causal B/C when they are already present. Player
    farming ``k=5`` and combat ``k=20`` are unchanged. Shrinkage of the
    *hero* state uses ``k_farm`` / ``k_combat`` independently.
    """
    if k_farm < 0.0:
        raise ValueError(f"farming hero shrinkage k must be >= 0, got {k_farm}")
    if k_combat < 0.0:
        raise ValueError(f"combat hero shrinkage k must be >= 0, got {k_combat}")
    if CAUSAL_B_COLUMN in frame.columns and CAUSAL_C_COLUMN in frame.columns:
        out = frame.copy()
    else:
        out = attach_hero_profile_observations(frame)

    farm_n, farm_sum, farm_mean, farm_unique, farm_top = prior_hero_position_history(
        out, CAUSAL_B_COLUMN, leave_player_out=True
    )
    farm_in_n, farm_in_sum, farm_in_mean, _, _ = prior_hero_position_history(
        out, CAUSAL_B_COLUMN, leave_player_out=False
    )
    farm_shrunk, farm_weight = apply_hero_requirement_shrinkage(
        farm_mean, farm_n, k=k_farm
    )
    out["hero_farming_prior_n"] = farm_n
    out["hero_farming_prior_sum_b"] = farm_sum
    out["hero_farming_prior_mean_b"] = farm_mean
    out["hero_farming_shrinkage_weight"] = farm_weight
    out["hero_farming_shrunk_b"] = farm_shrunk
    out["hero_farming_unique_prior_players"] = farm_unique
    out["hero_farming_top_player_share"] = farm_top
    out["hero_farming_inclusive_prior_n"] = farm_in_n
    out["hero_farming_inclusive_prior_sum_b"] = farm_in_sum
    out["hero_farming_inclusive_prior_mean_b"] = farm_in_mean
    out["hero_farming_current_player_prior_n"] = farm_in_n - farm_n

    combat_n, combat_sum, combat_mean, combat_unique, combat_top = (
        prior_hero_position_history(out, CAUSAL_C_COLUMN, leave_player_out=True)
    )
    combat_in_n, combat_in_sum, combat_in_mean, _, _ = prior_hero_position_history(
        out, CAUSAL_C_COLUMN, leave_player_out=False
    )
    combat_shrunk, combat_weight = apply_hero_requirement_shrinkage(
        combat_mean, combat_n, k=k_combat
    )
    out["hero_combat_prior_n"] = combat_n
    out["hero_combat_prior_sum_c"] = combat_sum
    out["hero_combat_prior_mean_c"] = combat_mean
    out["hero_combat_shrinkage_weight"] = combat_weight
    out["hero_combat_shrunk_c"] = combat_shrunk
    out["hero_combat_unique_prior_players"] = combat_unique
    out["hero_combat_top_player_share"] = combat_top
    out["hero_combat_inclusive_prior_n"] = combat_in_n
    out["hero_combat_inclusive_prior_sum_c"] = combat_in_sum
    out["hero_combat_inclusive_prior_mean_c"] = combat_in_mean
    out["hero_combat_current_player_prior_n"] = combat_in_n - combat_n
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


def _bucket_metrics(
    frame: pd.DataFrame,
    predicted: pd.Series,
    *,
    target: str,
    n_column: str,
    k: float,
    split: str,
) -> pd.DataFrame:
    actual = _numeric(frame[target])
    n_prior = pd.to_numeric(frame[n_column], errors="coerce").fillna(0)
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


def _evaluate_k_grid(
    frame: pd.DataFrame,
    *,
    target: str,
    mean_column: str,
    n_column: str,
    split: str,
    ks: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bucket_parts: list[pd.DataFrame] = []
    for k in ks:
        shrunk, _weight = apply_hero_requirement_shrinkage(
            frame[mean_column], frame[n_column], k=k
        )
        bucket_parts.append(
            _bucket_metrics(
                frame, shrunk, target=target, n_column=n_column, k=k, split=split
            )
        )
    buckets = pd.concat(bucket_parts, ignore_index=True)
    return _grid_summary(buckets, split=split), buckets


def select_hero_farm_shrinkage_k(tune_grid: pd.DataFrame) -> tuple[float, str]:
    """Choose farming hero ``k`` from the tune-set grid.

    Prefers a stable low-``n`` RMSE improvement. Among essentially
    equivalent neighbors, pick the larger ``k``. ``k`` that inflate
    established-cell RMSE by more than 5% versus ``k=0`` are discarded.
    """
    return _select_hero_k(tune_grid, plateau="largest", dimension="farming")


def select_hero_combat_shrinkage_k(tune_grid: pd.DataFrame) -> tuple[float, str]:
    """Choose combat hero ``k`` independently of farming.

    Combat RMSE surfaces tend to be flatter, so among neighbors within
    1% of the best low-``n`` RMSE pick the central grid point.
    """
    return _select_hero_k(tune_grid, plateau="central", dimension="combat")


def _select_hero_k(
    tune_grid: pd.DataFrame, *, plateau: str, dimension: str
) -> tuple[float, str]:
    if tune_grid.empty:
        return 0.0, f"Empty {dimension} tune grid; defaulting to unshrunk k=0."
    work = tune_grid.copy()
    k0 = work.loc[work["k"] == 0.0]
    if k0.empty:
        return (
            float(work.iloc[0]["k"]),
            "No k=0 reference row; using the first grid point.",
        )
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
            "No positive k kept established-cell RMSE within 5% of k=0.",
        )
    low_n = pd.to_numeric(keep["low_n_rmse"], errors="coerce")
    overall = pd.to_numeric(keep["rmse"], errors="coerce")
    if not bool(low_n.notna().any()):
        if not bool(overall.notna().any()):
            return (
                0.0,
                f"Tune split had no observed {dimension} target; defaulting to k=0.",
            )
        best_overall = float(overall.min())
        equiv = keep.loc[overall <= best_overall * EQUIVALENT_RMSE_RATIO]
        if equiv.empty:
            return 0.0, "No equivalent overall-RMSE neighbors; defaulting to k=0."
        selected = _plateau_k(equiv["k"].to_numpy(dtype=float), plateau=plateau)
        return selected, (
            f"Low-n RMSE was unavailable; selected the {plateau} k within "
            f"{EQUIVALENT_RMSE_RATIO:.0%} of best overall tune RMSE."
        )
    best_low = float(low_n.min())
    equiv = keep.loc[low_n <= best_low * EQUIVALENT_RMSE_RATIO]
    if equiv.empty:
        return 0.0, "No equivalent low-n neighbors; defaulting to k=0."
    selected = _plateau_k(equiv["k"].to_numpy(dtype=float), plateau=plateau)
    k0_low = float(baseline["low_n_rmse"])
    selected_row = keep.loc[keep["k"] == selected].iloc[0]
    selected_low = float(selected_row["low_n_rmse"])
    improved = np.isfinite(k0_low) and np.isfinite(selected_low) and selected_low < k0_low
    if plateau == "largest":
        style = "strongest shrinkage among"
    else:
        style = "central plateau point among"
    if improved:
        justification = (
            f"Selected k={selected:g} as the {style} tune-set neighbors "
            "within 1% of the best low-n RMSE, without inflating "
            "established-cell RMSE by more than 5% versus k=0."
        )
    else:
        justification = (
            f"Selected k={selected:g} from equivalent low-n neighbors, but "
            "low-n RMSE did not beat k=0 on the tune split."
        )
    return selected, justification


def _central_grid_k(values: np.ndarray) -> float:
    """Median of a sorted grid. Even length takes the larger central value."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return 0.0
    mid = int(ordered.size // 2)
    return float(ordered[mid])


def _plateau_k(values: np.ndarray, *, plateau: str) -> float:
    if plateau == "central":
        return _central_grid_k(values)
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return 0.0
    return float(ordered[-1])


def empirical_bayes_hero_k(
    frame: pd.DataFrame,
    *,
    target: str,
    min_cell_n: int = MIN_EMPIRICAL_BAYES_CELL_N,
) -> pd.DataFrame:
    """Sanity-check ``k ≈ within-cell var / between-cell var``.

    Restricted to the supplied rows (tune split). Cells with fewer than
    ``min_cell_n`` frozen-target observations are excluded. This is not
    a search and is not used to override the grid.
    """
    values = _numeric(frame[target])
    keys = _hero_position_keys(frame)
    work = pd.DataFrame({"key": keys, "value": values}, index=frame.index)
    work = work.loc[work["key"].notna() & work["value"].notna()]
    empty = {
        "k": float("nan"),
        "within_variance": float("nan"),
        "between_variance": float("nan"),
        "n_cells": 0,
        "n_observations": 0,
        "min_observations": min_cell_n,
        "used_for_state": False,
        "beyond_grid": False,
    }
    if work.empty:
        return pd.DataFrame([empty])
    stats = work.groupby("key")["value"].agg(
        n="size", mean="mean", var=lambda s: float(s.var(ddof=1))
    )
    eligible = stats.loc[stats["n"] >= min_cell_n]
    n_cells = len(eligible)
    n_obs = int(eligible["n"].sum()) if n_cells else 0
    if n_cells < MIN_EMPIRICAL_BAYES_CELLS:
        empty["n_cells"] = n_cells
        empty["n_observations"] = n_obs
        return pd.DataFrame([empty])
    within = float(eligible["var"].mean())
    sampling = float((eligible["var"] / eligible["n"]).mean())
    between_raw = float(eligible["mean"].var(ddof=1))
    between = max(0.0, between_raw - sampling)
    k = float("inf") if between <= 0.0 else float(within / between)
    grid_max = float(max(HERO_FARM_SHRINKAGE_GRID))
    beyond = bool(np.isfinite(k) and k > grid_max) or (not np.isfinite(k) and k == float("inf"))
    return pd.DataFrame(
        [
            {
                "k": k,
                "within_variance": within,
                "between_variance": between,
                "between_raw_var_of_means": between_raw,
                "mean_sampling_variance": sampling,
                "n_cells": n_cells,
                "n_observations": n_obs,
                "min_observations": min_cell_n,
                "used_for_state": False,
                "beyond_grid": beyond,
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


def state_distribution_table(
    frame: pd.DataFrame, *, mean_column: str, shrunk_column: str, n_column: str
) -> pd.DataFrame:
    raw = _numeric(frame[mean_column]).to_numpy(dtype=float)
    shrunk = _numeric(frame[shrunk_column]).to_numpy(dtype=float)
    n = pd.to_numeric(frame[n_column], errors="coerce").fillna(0)
    n_arr = n.to_numpy()
    return pd.DataFrame(
        [
            _distribution_row(raw, kind="raw_mean", subset="n>=1"),
            _distribution_row(shrunk, kind="shrunk", subset="all"),
            _distribution_row(shrunk[n_arr == 0], kind="shrunk", subset="n=0"),
            _distribution_row(shrunk[n_arr >= 1], kind="shrunk", subset="n>=1"),
            _distribution_row(raw[(n_arr >= 1) & (n_arr <= 2)], kind="raw_mean", subset="n=1–2"),
            _distribution_row(
                shrunk[(n_arr >= 1) & (n_arr <= 2)], kind="shrunk", subset="n=1–2"
            ),
            _distribution_row(raw[(n_arr >= 1) & (n_arr <= 5)], kind="raw_mean", subset="n=1–5"),
            _distribution_row(
                shrunk[(n_arr >= 1) & (n_arr <= 5)], kind="shrunk", subset="n=1–5"
            ),
        ]
    )


def coverage_table(
    frame: pd.DataFrame,
    *,
    target: str,
    n_column: str,
    inclusive_n_column: str,
) -> pd.DataFrame:
    eligible = explicit_position_mask(frame) & _numeric(frame["hero_id"]).notna()
    causal = eligible & _numeric(frame[target]).notna()
    n = pd.to_numeric(frame[n_column], errors="coerce").fillna(0)
    inclusive = pd.to_numeric(frame[inclusive_n_column], errors="coerce").fillna(0)
    work = frame.loc[eligible]
    keys = _hero_position_keys(work)
    n_values = n.loc[eligible].to_numpy(dtype=int)
    row: dict[str, object] = {
        "n_eligible_appearances": int(eligible.sum()),
        "n_causal_target_available": int(causal.sum()),
        "n_lpo_0": int((n.loc[eligible] == 0).sum()),
        "n_inclusive_gt0_lpo_0": int(
            ((inclusive.loc[eligible] >= 1) & (n.loc[eligible] == 0)).sum()
        ),
        "unique_hero_position_cells": int(keys.dropna().nunique()),
        "unique_heroes": int(_numeric(work["hero_id"]).nunique()) if len(work) else 0,
        "unique_players": int(work["player_id"].nunique()) if len(work) else 0,
        "prior_n_mean": float(n_values.mean()) if n_values.size else float("nan"),
        "prior_n_median": float(np.median(n_values)) if n_values.size else float("nan"),
        "prior_n_p95": float(np.quantile(n_values, 0.95)) if n_values.size else float("nan"),
        "prior_n_max": int(n_values.max()) if n_values.size else 0,
    }
    for threshold in COVERAGE_N_THRESHOLDS:
        row[f"n_lpo_ge_{threshold}"] = int((n.loc[eligible] >= threshold).sum())
    return pd.DataFrame([row])


def inclusive_vs_lpo_table(
    frame: pd.DataFrame,
    *,
    target: str,
    lpo_mean: str,
    inclusive_mean: str,
    lpo_n: str,
    inclusive_n: str,
    current_player_n: str,
) -> pd.DataFrame:
    actual = _numeric(frame[target])
    lpo = _numeric(frame[lpo_mean])
    incl = _numeric(frame[inclusive_mean])
    n_lpo = pd.to_numeric(frame[lpo_n], errors="coerce").fillna(0)
    n_incl = pd.to_numeric(frame[inclusive_n], errors="coerce").fillna(0)
    own = pd.to_numeric(frame[current_player_n], errors="coerce").fillna(0)
    observed = actual.notna()
    paired = observed & lpo.notna() & incl.notna() & (n_lpo >= 1) & (n_incl >= 1)
    major = paired & (n_incl > 0) & ((own / n_incl) >= SPECIALIST_TOP_SHARE)
    rows: list[dict[str, object]] = []
    for label, mask in (
        ("all_paired_n>=1", paired),
        ("current_player_major_share>=0.5", major),
        ("current_player_not_major", paired & ~major),
    ):
        subset = frame.loc[mask]
        if subset.empty:
            rows.append(
                {
                    "subset": label,
                    "n": 0,
                    "pearson_lpo_vs_inclusive": float("nan"),
                    "mean_abs_difference": float("nan"),
                    "rmse_lpo": float("nan"),
                    "rmse_inclusive": float("nan"),
                    "pearson_lpo_vs_target": float("nan"),
                    "pearson_inclusive_vs_target": float("nan"),
                }
            )
            continue
        lpo_s = _numeric(subset[lpo_mean])
        incl_s = _numeric(subset[inclusive_mean])
        y = _numeric(subset[target])
        abs_delta = (lpo_s - incl_s).abs()
        lpo_metrics = _prediction_metrics(y, lpo_s)
        incl_metrics = _prediction_metrics(y, incl_s)
        rows.append(
            {
                "subset": label,
                "n": int(mask.sum()),
                "pearson_lpo_vs_inclusive": _pearson(lpo_s, incl_s),
                "mean_abs_difference": float(abs_delta.mean()),
                "rmse_lpo": lpo_metrics["rmse"],
                "rmse_inclusive": incl_metrics["rmse"],
                "pearson_lpo_vs_target": lpo_metrics["pearson"],
                "pearson_inclusive_vs_target": incl_metrics["pearson"],
            }
        )
    return pd.DataFrame(rows)


def persistence_table(
    frame: pd.DataFrame,
    *,
    k: float,
    target: str,
    raw_column: str,
    shrunk_column: str,
    n_column: str,
) -> pd.DataFrame:
    actual = _numeric(frame[target])
    raw = _numeric(frame[raw_column])
    shrunk = _numeric(frame[shrunk_column])
    n = pd.to_numeric(frame[n_column], errors="coerce").fillna(0)
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
                "n_hero_position_cells": int(
                    _hero_position_keys(frame.loc[mask]).dropna().nunique()
                )
                if int(mask.sum())
                else 0,
                "pearson_raw_mean": _pearson(raw[mask], actual[mask]),
                "pearson_shrunk": _pearson(shrunk[mask], actual[mask]),
                "spearman_raw_mean": _spearman(raw[mask], actual[mask]),
                "spearman_shrunk": _spearman(shrunk[mask], actual[mask]),
            }
        )
    return pd.DataFrame(rows)


def unique_player_table(
    frame: pd.DataFrame,
    *,
    target: str,
    predicted: pd.Series,
    unique_column: str,
) -> pd.DataFrame:
    actual = _numeric(frame[target])
    unique = pd.to_numeric(frame[unique_column], errors="coerce").fillna(0)
    observed = actual.notna()
    rows: list[dict[str, object]] = []
    overall = _prediction_metrics(actual[observed], predicted[observed])
    rows.append({"bucket": "all", "kind": "unique_prior_players", **overall})
    for label, low, high in UNIQUE_PLAYER_BUCKETS:
        if high is None:
            mask = observed & (unique >= low)
        else:
            mask = observed & (unique >= low) & (unique <= high)
        metrics = _prediction_metrics(actual[mask], predicted[mask])
        rows.append(
            {"bucket": label, "kind": "unique_prior_players", **metrics}
        )
    return pd.DataFrame(rows)


def specialist_table(
    frame: pd.DataFrame,
    *,
    target: str,
    predicted: pd.Series,
    top_share_column: str,
) -> pd.DataFrame:
    actual = _numeric(frame[target])
    share = _numeric(frame[top_share_column])
    observed = actual.notna()
    rows: list[dict[str, object]] = []
    for label, mask in (
        ("top_share>=0.50", observed & share.notna() & (share >= SPECIALIST_TOP_SHARE)),
        ("top_share<0.50", observed & share.notna() & (share < SPECIALIST_TOP_SHARE)),
        ("top_share_missing", observed & share.isna()),
    ):
        metrics = _prediction_metrics(actual[mask], predicted[mask])
        rows.append({"bucket": label, "kind": "top_player_share", **metrics})
    return pd.DataFrame(rows)


def regression_to_mean_table(
    frame: pd.DataFrame,
    *,
    k: float,
    target: str,
) -> pd.DataFrame:
    """Low-history extremes and split-period cell means, raw vs shrunk."""
    values = _numeric(frame[target])
    keys = _hero_position_keys(frame)
    work = pd.DataFrame(
        {
            "key": keys,
            "start_time": pd.to_datetime(frame["start_time"], utc=True),
            "value": values,
        },
        index=frame.index,
    )
    work = work.loc[work["key"].notna() & work["value"].notna()]
    appearance_n = pd.to_numeric(
        frame.loc[work.index, "hero_farming_prior_n"]
        if target == CAUSAL_B_COLUMN
        else frame.loc[work.index, "hero_combat_prior_n"],
        errors="coerce",
    ).fillna(0)
    appearance_mean = _numeric(
        frame.loc[work.index, "hero_farming_prior_mean_b"]
        if target == CAUSAL_B_COLUMN
        else frame.loc[work.index, "hero_combat_prior_mean_c"]
    )
    appearance_shrunk = _numeric(
        frame.loc[work.index, "hero_farming_shrunk_b"]
        if target == CAUSAL_B_COLUMN
        else frame.loc[work.index, "hero_combat_shrunk_c"]
    )
    y = values.loc[work.index]
    low_n = (appearance_n >= 1) & (appearance_n <= 5) & appearance_mean.notna() & y.notna()
    rows: list[dict[str, object]] = []

    def _error_block(
        mask: pd.Series, *, subset: str, raw: pd.Series, shrunk: pd.Series, actual: pd.Series
    ) -> dict[str, object]:
        if not bool(mask.any()):
            return {
                "subset": subset,
                "n": 0,
                "k": k,
                "raw_mean": float("nan"),
                "shrunk_mean": float("nan"),
                "actual_mean": float("nan"),
                "rmse_raw": float("nan"),
                "rmse_shrunk": float("nan"),
                "mae_raw": float("nan"),
                "mae_shrunk": float("nan"),
                "mean_abs_raw": float("nan"),
                "mean_abs_shrunk": float("nan"),
                "mean_abs_actual": float("nan"),
            }
        r = raw[mask].to_numpy(dtype=float)
        s = shrunk[mask].to_numpy(dtype=float)
        a = actual[mask].to_numpy(dtype=float)
        return {
            "subset": subset,
            "n": int(mask.sum()),
            "k": k,
            "raw_mean": float(r.mean()),
            "shrunk_mean": float(s.mean()),
            "actual_mean": float(a.mean()),
            "rmse_raw": float(np.sqrt(np.mean((a - r) ** 2))),
            "rmse_shrunk": float(np.sqrt(np.mean((a - s) ** 2))),
            "mae_raw": float(np.mean(np.abs(a - r))),
            "mae_shrunk": float(np.mean(np.abs(a - s))),
            "mean_abs_raw": float(np.mean(np.abs(r))),
            "mean_abs_shrunk": float(np.mean(np.abs(s))),
            "mean_abs_actual": float(np.mean(np.abs(a))),
        }

    rows.append(
        _error_block(
            low_n,
            subset="low_n_1_5_all",
            raw=appearance_mean,
            shrunk=appearance_shrunk,
            actual=y,
        )
    )
    if bool(low_n.any()):
        abs_raw = appearance_mean[low_n].abs()
        threshold = float(np.quantile(abs_raw.to_numpy(dtype=float), RTM_EXTREME_QUANTILE))
        extreme = low_n & (appearance_mean.abs() >= threshold)
        rows.append(
            _error_block(
                extreme,
                subset="low_n_1_5_extreme_abs_ge_p90",
                raw=appearance_mean,
                shrunk=appearance_shrunk,
                actual=y,
            )
        )
    else:
        rows.append(
            _error_block(
                low_n,
                subset="low_n_1_5_extreme_abs_ge_p90",
                raw=appearance_mean,
                shrunk=appearance_shrunk,
                actual=y,
            )
        )

    early_means: list[float] = []
    late_means: list[float] = []
    early_n: list[int] = []
    for _key, group in work.groupby("key", sort=False):
        ordered = group.sort_values("start_time", kind="mergesort")
        size = len(ordered)
        split = size // 2
        if split < RTM_MIN_EACH or (size - split) < RTM_MIN_EACH:
            continue
        early_means.append(float(ordered.iloc[:split]["value"].mean()))
        late_means.append(float(ordered.iloc[split:]["value"].mean()))
        early_n.append(int(split))
    if not early_means:
        rows.append(
            {
                "subset": "split_period_all_paired",
                "n": 0,
                "k": k,
                "raw_mean": float("nan"),
                "shrunk_mean": float("nan"),
                "actual_mean": float("nan"),
                "rmse_raw": float("nan"),
                "rmse_shrunk": float("nan"),
                "mae_raw": float("nan"),
                "mae_shrunk": float("nan"),
                "mean_abs_raw": float("nan"),
                "mean_abs_shrunk": float("nan"),
                "mean_abs_actual": float("nan"),
            }
        )
        rows.append(
            {
                "subset": "split_period_extreme_early_abs_ge_p90",
                "n": 0,
                "k": k,
                "raw_mean": float("nan"),
                "shrunk_mean": float("nan"),
                "actual_mean": float("nan"),
                "rmse_raw": float("nan"),
                "rmse_shrunk": float("nan"),
                "mae_raw": float("nan"),
                "mae_shrunk": float("nan"),
                "mean_abs_raw": float("nan"),
                "mean_abs_shrunk": float("nan"),
                "mean_abs_actual": float("nan"),
            }
        )
        return pd.DataFrame(rows)

    early_arr = np.asarray(early_means, dtype=float)
    late_arr = np.asarray(late_means, dtype=float)
    n_arr = np.asarray(early_n, dtype=float)
    shrunk_early = np.array(
        [
            hero_requirement_shrunk(float(mean), float(n), k=k)
            for mean, n in zip(early_arr, n_arr, strict=True)
        ],
        dtype=float,
    )

    def _split_block(mask: np.ndarray, subset: str) -> dict[str, object]:
        e = early_arr[mask]
        s = shrunk_early[mask]
        l = late_arr[mask]
        if e.size == 0:
            return {
                "subset": subset,
                "n": 0,
                "k": k,
                "raw_mean": float("nan"),
                "shrunk_mean": float("nan"),
                "actual_mean": float("nan"),
                "rmse_raw": float("nan"),
                "rmse_shrunk": float("nan"),
                "mae_raw": float("nan"),
                "mae_shrunk": float("nan"),
                "mean_abs_raw": float("nan"),
                "mean_abs_shrunk": float("nan"),
                "mean_abs_actual": float("nan"),
            }
        return {
            "subset": subset,
            "n": int(e.size),
            "k": k,
            "raw_mean": float(e.mean()),
            "shrunk_mean": float(s.mean()),
            "actual_mean": float(l.mean()),
            "rmse_raw": float(np.sqrt(np.mean((l - e) ** 2))),
            "rmse_shrunk": float(np.sqrt(np.mean((l - s) ** 2))),
            "mae_raw": float(np.mean(np.abs(l - e))),
            "mae_shrunk": float(np.mean(np.abs(l - s))),
            "mean_abs_raw": float(np.mean(np.abs(e))),
            "mean_abs_shrunk": float(np.mean(np.abs(s))),
            "mean_abs_actual": float(np.mean(np.abs(l))),
        }

    rows.append(_split_block(np.ones(early_arr.shape, dtype=bool), "split_period_all_paired"))
    abs_early = np.abs(early_arr)
    threshold = float(np.quantile(abs_early, RTM_EXTREME_QUANTILE))
    rows.append(
        _split_block(abs_early >= threshold, "split_period_extreme_early_abs_ge_p90")
    )
    return pd.DataFrame(rows)


def patch_error_table(
    frame: pd.DataFrame,
    *,
    target: str,
    predicted: pd.Series,
    min_n: int = PATCH_MIN_VERSION_N,
) -> pd.DataFrame:
    actual = _numeric(frame[target])
    pred = _numeric(predicted)
    version = frame["game_version_id"]
    observed = actual.notna() & pred.notna()
    rows: list[dict[str, object]] = []
    overall = _prediction_metrics(actual[observed], pred[observed])
    rows.append({"game_version_id": "all", **overall, "material_rmse_increase": False})
    version_rows: list[dict[str, object]] = []
    for vid, group in frame.loc[observed].groupby(version, dropna=False):
        metrics = _prediction_metrics(actual.loc[group.index], pred.loc[group.index])
        version_rows.append({"game_version_id": vid, **metrics})
    version_table = pd.DataFrame(version_rows)
    enough = version_table.loc[version_table["n"] >= min_n] if not version_table.empty else version_table
    median_rmse = (
        float(enough["rmse"].median()) if not enough.empty else float("nan")
    )
    for row in version_rows:
        rmse = float(row["rmse"]) if row["rmse"] is not None else float("nan")
        pearson = float(row["pearson"]) if row["pearson"] is not None else float("nan")
        material = bool(
            np.isfinite(rmse)
            and np.isfinite(median_rmse)
            and int(row["n"]) >= min_n
            and rmse >= median_rmse * PATCH_RMSE_RATIO
            and (not np.isfinite(pearson) or pearson < _PATCH_CORR_SOFT)
        )
        rows.append({**row, "material_rmse_increase": material})
    return pd.DataFrame(rows)


def player_state_relationship_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    farm_hero = _numeric(frame["hero_farming_shrunk_b"])
    farm_player = _numeric(frame["farming_shrunk_b"])
    combat_hero = _numeric(frame["hero_combat_shrunk_c"])
    combat_player = _numeric(frame["combat_shrunk_c"])
    farm_n = pd.to_numeric(frame["hero_farming_prior_n"], errors="coerce").fillna(0)
    combat_n = pd.to_numeric(frame["hero_combat_prior_n"], errors="coerce").fillna(0)
    for label, hero, player, n_prior, extra_mask in (
        (
            "farming_all_n>=1",
            farm_hero,
            farm_player,
            farm_n,
            pd.Series(True, index=frame.index),
        ),
        (
            "combat_all_n>=1",
            combat_hero,
            combat_player,
            combat_n,
            pd.Series(True, index=frame.index),
        ),
    ):
        mask = (n_prior >= 1) & hero.notna() & player.notna() & extra_mask
        rows.append(
            {
                "subset": label,
                "n": int(mask.sum()),
                "pearson": _pearson(hero[mask], player[mask]) if int(mask.sum()) else float("nan"),
                "spearman": _spearman(hero[mask], player[mask])
                if int(mask.sum())
                else float("nan"),
            }
        )
    positions = _numeric(frame["position_number"])
    for position in EXPLICIT_POSITION_NUMBERS:
        pos_mask = positions == float(position)
        for dim, hero, player, n_prior in (
            ("farming", farm_hero, farm_player, farm_n),
            ("combat", combat_hero, combat_player, combat_n),
        ):
            mask = pos_mask & (n_prior >= 1) & hero.notna() & player.notna()
            rows.append(
                {
                    "subset": f"{dim}_position_{position}_n>=1",
                    "n": int(mask.sum()),
                    "pearson": _pearson(hero[mask], player[mask])
                    if int(mask.sum())
                    else float("nan"),
                    "spearman": _spearman(hero[mask], player[mask])
                    if int(mask.sum())
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def cross_dimension_table(frame: pd.DataFrame) -> pd.DataFrame:
    farm = _numeric(frame["hero_farming_shrunk_b"])
    combat = _numeric(frame["hero_combat_shrunk_c"])
    farm_n = pd.to_numeric(frame["hero_farming_prior_n"], errors="coerce").fillna(0)
    combat_n = pd.to_numeric(frame["hero_combat_prior_n"], errors="coerce").fillna(0)
    appearance_mask = (farm_n >= 1) & (combat_n >= 1) & farm.notna() & combat.notna()
    keys = _hero_position_keys(frame)
    cell_work = pd.DataFrame(
        {
            "key": keys,
            "farm": farm,
            "combat": combat,
            "farm_n": farm_n,
            "combat_n": combat_n,
        }
    )
    cell_work = cell_work.loc[
        cell_work["key"].notna()
        & (cell_work["farm_n"] >= 1)
        & (cell_work["combat_n"] >= 1)
    ]
    if cell_work.empty:
        cell_farm = pd.Series(dtype=float)
        cell_combat = pd.Series(dtype=float)
    else:
        last = cell_work.sort_index().groupby("key", sort=False).tail(1)
        cell_farm = last["farm"]
        cell_combat = last["combat"]
    return pd.DataFrame(
        [
            {
                "unit": "appearances_both_n>=1",
                "n": int(appearance_mask.sum()),
                "pearson": _pearson(farm[appearance_mask], combat[appearance_mask])
                if int(appearance_mask.sum())
                else float("nan"),
                "spearman": _spearman(farm[appearance_mask], combat[appearance_mask])
                if int(appearance_mask.sum())
                else float("nan"),
            },
            {
                "unit": "hero_position_cells_last_state",
                "n": len(cell_farm),
                "pearson": _pearson(cell_farm, cell_combat) if len(cell_farm) else float("nan"),
                "spearman": _spearman(cell_farm, cell_combat)
                if len(cell_farm)
                else float("nan"),
            },
        ]
    )


def split_summary(
    frame: pd.DataFrame,
    *,
    mask: pd.Series,
    split: str,
    tune_end: datetime,
    development_end: datetime,
) -> dict[str, object]:
    subset = frame.loc[mask]
    keys = _hero_position_keys(subset)
    return {
        "split": split,
        "tune_end": tune_end.isoformat(),
        "development_end": development_end.isoformat(),
        "n_rows": len(subset),
        "n_matches": int(subset["match_id"].nunique()) if len(subset) else 0,
        "n_players": int(subset["player_id"].nunique()) if len(subset) else 0,
        "unique_hero_position_cells": int(keys.dropna().nunique()),
        "unique_heroes": int(_numeric(subset["hero_id"]).nunique()) if len(subset) else 0,
        "min_start_time": (
            pd.to_datetime(subset["start_time"], utc=True).min().isoformat()
            if len(subset)
            else None
        ),
        "max_start_time": (
            pd.to_datetime(subset["start_time"], utc=True).max().isoformat()
            if len(subset)
            else None
        ),
        "matches_preferred_tune_end": tune_end == PREFERRED_TUNE_END,
    }


def _dimension_gate(
    *,
    name: str,
    selected_k: float,
    tune_grid: pd.DataFrame,
    validation_grid: pd.DataFrame,
    persistence: pd.DataFrame,
    split_half: pd.DataFrame,
    inclusive_vs_lpo: pd.DataFrame,
    unique_player: pd.DataFrame,
    specialist: pd.DataFrame,
    patch: pd.DataFrame,
    empirical_bayes: pd.DataFrame,
) -> dict[str, object]:
    val_sel = validation_grid.loc[validation_grid["k"] == selected_k]
    val_k0 = validation_grid.loc[validation_grid["k"] == 0.0]
    tune_sel = tune_grid.loc[tune_grid["k"] == selected_k]
    tune_k0 = tune_grid.loc[tune_grid["k"] == 0.0]
    pearson = float(val_sel.iloc[0]["pearson"]) if not val_sel.empty else float("nan")
    val_rmse = float(val_sel.iloc[0]["rmse"]) if not val_sel.empty else float("nan")
    val_rmse_k0 = float(val_k0.iloc[0]["rmse"]) if not val_k0.empty else float("nan")
    low_n = float(val_sel.iloc[0]["low_n_rmse"]) if not val_sel.empty else float("nan")
    low_n_k0 = float(val_k0.iloc[0]["low_n_rmse"]) if not val_k0.empty else float("nan")
    n_low_n_val = (
        int(val_sel.iloc[0]["n_low_n"])
        if not val_sel.empty and "n_low_n" in val_sel.columns
        else 0
    )
    tune_low_n = float(tune_sel.iloc[0]["low_n_rmse"]) if not tune_sel.empty else float("nan")
    tune_low_n_k0 = float(tune_k0.iloc[0]["low_n_rmse"]) if not tune_k0.empty else float("nan")
    established = (
        float(val_sel.iloc[0]["rmse_n_gt_20"]) if not val_sel.empty else float("nan")
    )
    established_k0 = (
        float(val_k0.iloc[0]["rmse_n_gt_20"]) if not val_k0.empty else float("nan")
    )
    half_r = (
        float(split_half.iloc[0]["pearson"]) if not split_half.empty else float("nan")
    )
    persist_r = float("nan")
    if not persistence.empty:
        all_row = persistence.loc[persistence["bucket"] == "all"]
        if not all_row.empty:
            persist_r = float(all_row.iloc[0]["pearson_shrunk"])
    lpo_r = float("nan")
    incl_r = float("nan")
    if not inclusive_vs_lpo.empty:
        paired = inclusive_vs_lpo.loc[inclusive_vs_lpo["subset"] == "all_paired_n>=1"]
        if not paired.empty:
            lpo_r = float(paired.iloc[0]["pearson_lpo_vs_target"])
            incl_r = float(paired.iloc[0]["pearson_inclusive_vs_target"])
    specialist_r = float("nan")
    if not specialist.empty:
        heavy = specialist.loc[specialist["bucket"] == "top_share>=0.50"]
        if not heavy.empty:
            specialist_r = float(heavy.iloc[0]["pearson"])
    sparse_r = float("nan")
    if not unique_player.empty:
        sparse = unique_player.loc[unique_player["bucket"] == "1–2"]
        if not sparse.empty:
            sparse_r = float(sparse.iloc[0]["pearson"])
    patch_material = False
    if not patch.empty and "material_rmse_increase" in patch.columns:
        patch_material = bool(patch["material_rmse_increase"].fillna(False).any())
    eb_beyond = False
    eb_k = float("nan")
    if not empirical_bayes.empty:
        eb_beyond = bool(empirical_bayes.iloc[0].get("beyond_grid", False))
        eb_k = float(empirical_bayes.iloc[0]["k"])

    tune_low_n_improves = (
        np.isfinite(tune_low_n)
        and np.isfinite(tune_low_n_k0)
        and tune_low_n < tune_low_n_k0
    )
    val_low_n_improves = np.isfinite(low_n) and np.isfinite(low_n_k0) and low_n < low_n_k0
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
        or (np.isfinite(lpo_r) and lpo_r >= _REPEATABILITY_FLOOR)
    )
    lpo_destroyed = (
        np.isfinite(incl_r)
        and incl_r >= 0.20
        and np.isfinite(lpo_r)
        and lpo_r < _REPEATABILITY_FLOOR
        and (incl_r - lpo_r) >= _LPO_DESTROY_DELTA
    )
    specialist_unusable = (
        np.isfinite(specialist_r) and specialist_r < 0.0 and signal
    )
    k_unstable = selected_k <= 0.0 or not low_n_confirmed

    if lpo_destroyed or not signal:
        grade = "C"
        rationale = (
            f"{name}: LPO causal requirement does not predict the next "
            "frozen target (or LPO destroys the inclusive profile signal)."
        )
    elif (
        selected_k > 0.0
        and signal
        and low_n_confirmed
        and established_ok
        and overall_ok
        and not patch_material
        and not specialist_unusable
    ):
        grade = "A"
        rationale = (
            f"{name}: LPO historical state predicts the next target, "
            f"k={selected_k:g} improves low-n RMSE on the {low_n_source} "
            "split, established cells are not harmed, specialist cells "
            "remain usable, and patch drift does not invalidate expanding "
            "history."
        )
    else:
        grade = "B"
        notes: list[str] = []
        if k_unstable:
            notes.append("shrinkage k is unstable or does not improve low-n")
        if patch_material:
            notes.append("patch/version error increases materially")
        if specialist_unusable:
            notes.append("specialist-heavy cells lose signal")
        if not established_ok:
            notes.append("established-cell RMSE inflated")
        if not overall_ok:
            notes.append("overall validation RMSE inflated vs k=0")
        if eb_beyond:
            notes.append(f"EB k={eb_k:g} lies beyond the grid")
        rationale = (
            f"{name}: requirement signal is real, but "
            + ("; ".join(notes) if notes else "a temporal/shrinkage assumption needs another slice")
            + "."
        )
    return {
        "dimension": name,
        "grade": grade,
        "selected_k": selected_k,
        "rationale": rationale,
        "signal": signal,
        "lpo_destroyed": lpo_destroyed,
        "low_n_confirmed": low_n_confirmed,
        "low_n_source": low_n_source,
        "established_ok": established_ok,
        "overall_ok": overall_ok,
        "patch_material": patch_material,
        "specialist_unusable": specialist_unusable,
        "validation_pearson": pearson,
        "persistence_pearson_shrunk": persist_r,
        "split_half_pearson": half_r,
        "lpo_vs_target_pearson": lpo_r,
        "inclusive_vs_target_pearson": incl_r,
        "sparse_unique_1_2_pearson": sparse_r,
        "specialist_pearson": specialist_r,
        "eb_k": eb_k,
        "eb_beyond_grid": eb_beyond,
    }


def classify_slice22(
    *,
    farming_gate: dict[str, object],
    combat_gate: dict[str, object],
) -> pd.DataFrame:
    """Map per-dimension gates onto a single Slice 22 classification."""
    farm_grade = str(farming_gate["grade"])
    combat_grade = str(combat_gate["grade"])
    if farm_grade == "C" and combat_grade == "C":
        classification = "C"
        gate = CLASSIFICATION_C
        next_slice = (
            "Do not freeze hero requirement states. Causal LPO construction "
            "did not retain the Slice 21 profile signal."
        )
    elif farm_grade == "A" and combat_grade == "A":
        classification = "A"
        gate = CLASSIFICATION_A
        next_slice = (
            "Freeze both LPO hero requirement states and their shrinkage "
            "constants for a later player×hero fit slice. Do not build fit now."
        )
    else:
        classification = "B"
        gate = CLASSIFICATION_B
        next_slice = (
            "Keep the LPO construction. Resolve the weaker dimension or "
            "temporal assumption before fit construction."
        )
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "gate": gate,
                "farming_grade": farm_grade,
                "combat_grade": combat_grade,
                "farming_k": farming_gate["selected_k"],
                "combat_k": combat_gate["selected_k"],
                "farming_rationale": farming_gate["rationale"],
                "combat_rationale": combat_gate["rationale"],
                "next_slice": next_slice,
            }
        ]
    )


def _semantics(*, dimension: str, target: str, k: float) -> dict[str, object]:
    return {
        "dimension": dimension,
        "target": target,
        "key": "hero_id × explicit position 1–5",
        "history_filter": (
            "start_time < T; same hero_id; same explicit position; "
            "player_id != current player; finite frozen target"
        ),
        "same_timestamp": "mutually blind",
        "n_0_raw_mean": "NULL",
        "n_0_shrunk": 0.0,
        "shrinkage": "n / (n + k) * prior_mean, toward 0",
        "selected_k": k,
        "current_position": (
            "diagnostic realized post-match position; not a PRE_DRAFT feature"
        ),
        "fit_constructed": False,
    }


def run_hero_requirement_state_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice22DiagnosticReport:
    """Development-only Slice 22 hero-requirement research. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    development = attach_hero_profile_observations(development)
    development = attach_hero_requirement_state(
        development,
        k_farm=0.0,
        k_combat=0.0,
    )

    tune_end = development_tune_end(development["start_time"], development_end=end)
    dev_times = pd.to_datetime(development["start_time"], utc=True)
    tune_mask = dev_times <= pd.Timestamp(tune_end)
    val_mask = (dev_times > pd.Timestamp(tune_end)) & (dev_times <= pd.Timestamp(end))
    tune = development.loc[tune_mask].copy()
    validation = development.loc[val_mask].copy()

    farm_tune_grid, farm_tune_buckets = _evaluate_k_grid(
        tune,
        target=CAUSAL_B_COLUMN,
        mean_column="hero_farming_prior_mean_b",
        n_column="hero_farming_prior_n",
        split="tune",
        ks=HERO_FARM_SHRINKAGE_GRID,
    )
    farm_k, farm_why = select_hero_farm_shrinkage_k(farm_tune_grid)
    farm_val_grid, farm_val_buckets = _evaluate_k_grid(
        validation,
        target=CAUSAL_B_COLUMN,
        mean_column="hero_farming_prior_mean_b",
        n_column="hero_farming_prior_n",
        split="validation",
        ks=HERO_FARM_SHRINKAGE_GRID,
    )
    combat_tune_grid, combat_tune_buckets = _evaluate_k_grid(
        tune,
        target=CAUSAL_C_COLUMN,
        mean_column="hero_combat_prior_mean_c",
        n_column="hero_combat_prior_n",
        split="tune",
        ks=HERO_COMBAT_SHRINKAGE_GRID,
    )
    combat_k, combat_why = select_hero_combat_shrinkage_k(combat_tune_grid)
    combat_val_grid, combat_val_buckets = _evaluate_k_grid(
        validation,
        target=CAUSAL_C_COLUMN,
        mean_column="hero_combat_prior_mean_c",
        n_column="hero_combat_prior_n",
        split="validation",
        ks=HERO_COMBAT_SHRINKAGE_GRID,
    )

    farm_shrunk, farm_weight = apply_hero_requirement_shrinkage(
        development["hero_farming_prior_mean_b"],
        development["hero_farming_prior_n"],
        k=farm_k,
    )
    combat_shrunk, combat_weight = apply_hero_requirement_shrinkage(
        development["hero_combat_prior_mean_c"],
        development["hero_combat_prior_n"],
        k=combat_k,
    )
    development["hero_farming_shrinkage_weight"] = farm_weight
    development["hero_farming_shrunk_b"] = farm_shrunk
    development["hero_combat_shrinkage_weight"] = combat_weight
    development["hero_combat_shrunk_c"] = combat_shrunk

    farm_eb = empirical_bayes_hero_k(tune, target=CAUSAL_B_COLUMN)
    combat_eb = empirical_bayes_hero_k(tune, target=CAUSAL_C_COLUMN)
    if bool(farm_eb.iloc[0]["beyond_grid"]):
        farm_why = (
            farm_why
            + " EB estimate lies beyond the explicit grid; the grid was not extended."
        )
    if bool(combat_eb.iloc[0]["beyond_grid"]):
        combat_why = (
            combat_why
            + " EB estimate lies beyond the explicit grid; the grid was not extended."
        )

    split = pd.DataFrame(
        [
            split_summary(
                development,
                mask=tune_mask,
                split="tune",
                tune_end=tune_end,
                development_end=end,
            ),
            split_summary(
                development,
                mask=val_mask,
                split="validation",
                tune_end=tune_end,
                development_end=end,
            ),
        ]
    )
    farming_coverage = coverage_table(
        development,
        target=CAUSAL_B_COLUMN,
        n_column="hero_farming_prior_n",
        inclusive_n_column="hero_farming_inclusive_prior_n",
    )
    combat_coverage = coverage_table(
        development,
        target=CAUSAL_C_COLUMN,
        n_column="hero_combat_prior_n",
        inclusive_n_column="hero_combat_inclusive_prior_n",
    )
    farming_inclusive_vs_lpo = inclusive_vs_lpo_table(
        development,
        target=CAUSAL_B_COLUMN,
        lpo_mean="hero_farming_prior_mean_b",
        inclusive_mean="hero_farming_inclusive_prior_mean_b",
        lpo_n="hero_farming_prior_n",
        inclusive_n="hero_farming_inclusive_prior_n",
        current_player_n="hero_farming_current_player_prior_n",
    )
    combat_inclusive_vs_lpo = inclusive_vs_lpo_table(
        development,
        target=CAUSAL_C_COLUMN,
        lpo_mean="hero_combat_prior_mean_c",
        inclusive_mean="hero_combat_inclusive_prior_mean_c",
        lpo_n="hero_combat_prior_n",
        inclusive_n="hero_combat_inclusive_prior_n",
        current_player_n="hero_combat_current_player_prior_n",
    )
    farming_distribution = state_distribution_table(
        development,
        mean_column="hero_farming_prior_mean_b",
        shrunk_column="hero_farming_shrunk_b",
        n_column="hero_farming_prior_n",
    )
    combat_distribution = state_distribution_table(
        development,
        mean_column="hero_combat_prior_mean_c",
        shrunk_column="hero_combat_shrunk_c",
        n_column="hero_combat_prior_n",
    )
    farming_persistence = persistence_table(
        development,
        k=farm_k,
        target=CAUSAL_B_COLUMN,
        raw_column="hero_farming_prior_mean_b",
        shrunk_column="hero_farming_shrunk_b",
        n_column="hero_farming_prior_n",
    )
    combat_persistence = persistence_table(
        development,
        k=combat_k,
        target=CAUSAL_C_COLUMN,
        raw_column="hero_combat_prior_mean_c",
        shrunk_column="hero_combat_shrunk_c",
        n_column="hero_combat_prior_n",
    )
    farming_unique = unique_player_table(
        development,
        target=CAUSAL_B_COLUMN,
        predicted=development["hero_farming_shrunk_b"],
        unique_column="hero_farming_unique_prior_players",
    )
    combat_unique = unique_player_table(
        development,
        target=CAUSAL_C_COLUMN,
        predicted=development["hero_combat_shrunk_c"],
        unique_column="hero_combat_unique_prior_players",
    )
    farming_specialist = specialist_table(
        development,
        target=CAUSAL_B_COLUMN,
        predicted=development["hero_farming_shrunk_b"],
        top_share_column="hero_farming_top_player_share",
    )
    combat_specialist = specialist_table(
        development,
        target=CAUSAL_C_COLUMN,
        predicted=development["hero_combat_shrunk_c"],
        top_share_column="hero_combat_top_player_share",
    )
    farming_half = pd.DataFrame(
        [
            {
                "dimension": "farming",
                **group_split_half(
                    development,
                    value_column=CAUSAL_B_COLUMN,
                    group_columns=HERO_POSITION_GROUP,
                    min_each=MIN_HALF_HERO_POSITION,
                ),
            }
        ]
    )
    combat_half = pd.DataFrame(
        [
            {
                "dimension": "combat",
                **group_split_half(
                    development,
                    value_column=CAUSAL_C_COLUMN,
                    group_columns=HERO_POSITION_GROUP,
                    min_each=MIN_HALF_HERO_POSITION,
                ),
            }
        ]
    )
    development = development.assign(chrono_block=assign_chronological_blocks(development))
    farming_blocks = _adjacent_window_stability(
        development,
        value_column=CAUSAL_B_COLUMN,
        group_columns=HERO_POSITION_GROUP,
        window_column="chrono_block",
        min_n=MIN_BLOCK_PROFILE_N,
        material_shift=0.50,
    )
    combat_blocks = _adjacent_window_stability(
        development,
        value_column=CAUSAL_C_COLUMN,
        group_columns=HERO_POSITION_GROUP,
        window_column="chrono_block",
        min_n=MIN_BLOCK_PROFILE_N,
        material_shift=0.05,
    )
    farming_patch = patch_error_table(
        development,
        target=CAUSAL_B_COLUMN,
        predicted=development["hero_farming_shrunk_b"],
    )
    combat_patch = patch_error_table(
        development,
        target=CAUSAL_C_COLUMN,
        predicted=development["hero_combat_shrunk_c"],
    )
    farming_rtm = regression_to_mean_table(
        development, k=farm_k, target=CAUSAL_B_COLUMN
    )
    combat_rtm = regression_to_mean_table(
        development, k=combat_k, target=CAUSAL_C_COLUMN
    )
    player_rel = player_state_relationship_table(development)
    cross = cross_dimension_table(development)

    farming_gate = _dimension_gate(
        name="farming",
        selected_k=farm_k,
        tune_grid=farm_tune_grid,
        validation_grid=farm_val_grid,
        persistence=farming_persistence,
        split_half=farming_half,
        inclusive_vs_lpo=farming_inclusive_vs_lpo,
        unique_player=farming_unique,
        specialist=farming_specialist,
        patch=farming_patch,
        empirical_bayes=farm_eb,
    )
    combat_gate = _dimension_gate(
        name="combat",
        selected_k=combat_k,
        tune_grid=combat_tune_grid,
        validation_grid=combat_val_grid,
        persistence=combat_persistence,
        split_half=combat_half,
        inclusive_vs_lpo=combat_inclusive_vs_lpo,
        unique_player=combat_unique,
        specialist=combat_specialist,
        patch=combat_patch,
        empirical_bayes=combat_eb,
    )
    classification = classify_slice22(
        farming_gate=farming_gate, combat_gate=combat_gate
    )

    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    integrity = {
        "development_end": end.isoformat(),
        "tune_end": tune_end.isoformat(),
        "preferred_tune_end": PREFERRED_TUNE_END.isoformat(),
        "tune_end_matches_preferred": tune_end == PREFERRED_TUNE_END,
        "holdout_used_for_k_selection": False,
        "holdout_used_for_validation": False,
        "holdout_used_for_eb": False,
        "stratz_called": False,
        "ingestion_modified": False,
        "schema_modified": False,
        "slice21_farming_target_unchanged": (
            HERO_FARMING_PROFILE_TARGET == CAUSAL_B_COLUMN
        ),
        "slice21_combat_target_unchanged": (
            HERO_COMBAT_PROFILE_TARGET == CAUSAL_C_COLUMN
        ),
        "slice21_farming_key_unchanged": HERO_FARMING_PROFILE_KEY
        == "hero_id × position",
        "slice21_combat_key_unchanged": HERO_COMBAT_PROFILE_KEY == "hero_id × position",
        "farming_candidate_b_unchanged": FROZEN_CANDIDATE_B == CANDIDATE_B,
        "farming_player_k_is_5": FROZEN_SHRINKAGE_K == 5.0,
        "combat_candidate_c_unchanged": FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION,
        "combat_player_k_is_20": FROZEN_COMBAT_SHRINKAGE_K == 20.0,
        "player_hero_fit_created": False,
        "current_position_resolved": False,
        "team_feature_created": False,
        "win_model_run": False,
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "requirement_in_feature_columns": any(
            name in FEATURE_COLUMNS
            for name in (
                *SLICE22_STATE_COLUMNS,
                *PLAYER_X_HERO_FIT_NAMES,
                CAUSAL_B_COLUMN,
                CAUSAL_C_COLUMN,
            )
        ),
        "requirement_in_snapshot_columns": any(
            name in SNAPSHOT_COLUMNS for name in (*SLICE22_STATE_COLUMNS, *PLAYER_X_HERO_FIT_NAMES)
        ),
        "requirement_in_pre_draft_sql": any(
            name in PRE_DRAFT_SNAPSHOT_SQL
            for name in (*SLICE22_STATE_COLUMNS, *PLAYER_X_HERO_FIT_NAMES)
        ),
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "match_player_box_score_field_count": len(MATCH_PLAYER_BOX_SCORE_COLUMNS),
        "n_holdout_excluded": len(holdout),
        "model_trained": False,
        "full_development_mean_fallback": False,
        "farming_gate": farming_gate,
        "combat_gate": combat_gate,
    }
    return Slice22DiagnosticReport(
        development_end=end,
        tune_end=tune_end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        selected_k_farm=farm_k,
        selected_k_combat=combat_k,
        selected_k_farm_justification=farm_why,
        selected_k_combat_justification=combat_why,
        farming_semantics=_semantics(dimension="farming", target=CAUSAL_B_COLUMN, k=farm_k),
        combat_semantics=_semantics(dimension="combat", target=CAUSAL_C_COLUMN, k=combat_k),
        classification=classification,
        split=split,
        farming_coverage=farming_coverage,
        combat_coverage=combat_coverage,
        farming_inclusive_vs_lpo=farming_inclusive_vs_lpo,
        combat_inclusive_vs_lpo=combat_inclusive_vs_lpo,
        farming_grid_tune=farm_tune_grid,
        farming_grid_validation=farm_val_grid,
        combat_grid_tune=combat_tune_grid,
        combat_grid_validation=combat_val_grid,
        farming_empirical_bayes=farm_eb,
        combat_empirical_bayes=combat_eb,
        farming_history_bucket_tune=farm_tune_buckets,
        farming_history_bucket_validation=farm_val_buckets,
        combat_history_bucket_tune=combat_tune_buckets,
        combat_history_bucket_validation=combat_val_buckets,
        farming_unique_player=farming_unique,
        combat_unique_player=combat_unique,
        farming_specialist=farming_specialist,
        combat_specialist=combat_specialist,
        farming_state_distribution=farming_distribution,
        combat_state_distribution=combat_distribution,
        farming_persistence=farming_persistence,
        combat_persistence=combat_persistence,
        farming_split_half=farming_half,
        combat_split_half=combat_half,
        farming_temporal_blocks=farming_blocks,
        combat_temporal_blocks=combat_blocks,
        farming_patch=farming_patch,
        combat_patch=combat_patch,
        farming_regression_to_mean=farming_rtm,
        combat_regression_to_mean=combat_rtm,
        player_state_relationship=player_rel,
        cross_dimension=cross,
        integrity=integrity,
    )


def slice22_report_to_jsonable(report: Slice22DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 22 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "tune_end": report.tune_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "selected_k_farm": report.selected_k_farm,
        "selected_k_combat": report.selected_k_combat,
        "selected_k_farm_justification": report.selected_k_farm_justification,
        "selected_k_combat_justification": report.selected_k_combat_justification,
        "frozen_hero_farm_k": FROZEN_HERO_FARM_SHRINKAGE_K,
        "frozen_hero_combat_k": FROZEN_HERO_COMBAT_SHRINKAGE_K,
        "farming_target": HERO_FARMING_PROFILE_TARGET,
        "combat_target": HERO_COMBAT_PROFILE_TARGET,
        "farming_key": HERO_FARMING_PROFILE_KEY,
        "combat_key": HERO_COMBAT_PROFILE_KEY,
        "farming_grid": list(HERO_FARM_SHRINKAGE_GRID),
        "combat_grid": list(HERO_COMBAT_SHRINKAGE_GRID),
        "farming_semantics": _jsonable_value(report.farming_semantics),
        "combat_semantics": _jsonable_value(report.combat_semantics),
        "classification": _jsonable_value(report.classification),
        "split": _jsonable_value(report.split),
        "farming_coverage": _jsonable_value(report.farming_coverage),
        "combat_coverage": _jsonable_value(report.combat_coverage),
        "farming_inclusive_vs_lpo": _jsonable_value(report.farming_inclusive_vs_lpo),
        "combat_inclusive_vs_lpo": _jsonable_value(report.combat_inclusive_vs_lpo),
        "farming_grid_tune": _jsonable_value(report.farming_grid_tune),
        "farming_grid_validation": _jsonable_value(report.farming_grid_validation),
        "combat_grid_tune": _jsonable_value(report.combat_grid_tune),
        "combat_grid_validation": _jsonable_value(report.combat_grid_validation),
        "farming_empirical_bayes": _jsonable_value(report.farming_empirical_bayes),
        "combat_empirical_bayes": _jsonable_value(report.combat_empirical_bayes),
        "farming_history_bucket_tune": _jsonable_value(report.farming_history_bucket_tune),
        "farming_history_bucket_validation": _jsonable_value(
            report.farming_history_bucket_validation
        ),
        "combat_history_bucket_tune": _jsonable_value(report.combat_history_bucket_tune),
        "combat_history_bucket_validation": _jsonable_value(
            report.combat_history_bucket_validation
        ),
        "farming_unique_player": _jsonable_value(report.farming_unique_player),
        "combat_unique_player": _jsonable_value(report.combat_unique_player),
        "farming_specialist": _jsonable_value(report.farming_specialist),
        "combat_specialist": _jsonable_value(report.combat_specialist),
        "farming_state_distribution": _jsonable_value(report.farming_state_distribution),
        "combat_state_distribution": _jsonable_value(report.combat_state_distribution),
        "farming_persistence": _jsonable_value(report.farming_persistence),
        "combat_persistence": _jsonable_value(report.combat_persistence),
        "farming_split_half": _jsonable_value(report.farming_split_half),
        "combat_split_half": _jsonable_value(report.combat_split_half),
        "farming_temporal_blocks": _jsonable_value(report.farming_temporal_blocks),
        "combat_temporal_blocks": _jsonable_value(report.combat_temporal_blocks),
        "farming_patch": _jsonable_value(report.farming_patch),
        "combat_patch": _jsonable_value(report.combat_patch),
        "farming_regression_to_mean": _jsonable_value(report.farming_regression_to_mean),
        "combat_regression_to_mean": _jsonable_value(report.combat_regression_to_mean),
        "player_state_relationship": _jsonable_value(report.player_state_relationship),
        "cross_dimension": _jsonable_value(report.cross_dimension),
        "integrity": _jsonable_value(report.integrity),
    }
