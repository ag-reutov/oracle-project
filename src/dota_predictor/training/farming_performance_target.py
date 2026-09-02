"""Slice 13: farming-performance target refinement and falsification.

Research only. This module does not persist a player rating, does not
write rolling player state, does not add production features, and does
not train a win model. Candidate columns never enter ``FEATURE_COLUMNS``.

Question
--------
After properly removing duration effects, how much of last-hit-rate
persistence belongs to the player rather than repeated hero choice or
team farming structure?

Population
----------
Matches with ``start_time <=`` the frozen Slice 9 development end
(``FROZEN_DEVELOPMENT_END``). Holdout / TI 2026 rows are excluded from
every summary. Box-score last hits are POST_MATCH observations of the
*current* appearance; they are the candidate *target*, not
prediction-time features.

Reuse
-----
Development cutoff, Elo expected win, position standardization,
position+duration residualization, repeatability, and hero R² helpers
come from Slice 12 (``player_performance_target``). This module does
not reimplement Elo or the frozen cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.features.team_elo import DEFAULT_ELO_CONFIG, EloConfig
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    CANDIDATE_COLUMN_NAMES,
    EXPLICIT_POSITION_NUMBERS,
    MIN_HERO_APPEARANCES,
    MIN_PRIOR_APPEARANCES,
    REPEATABILITY_PRIOR_THRESHOLDS,
    CandidateSpec,
    _categorical_r2,
    _jsonable_value,
    _loo_team_match_residual,
    _numeric,
    _pearson,
    _player_mean_vs_team_mean,
    _position_dummy_frame,
    _spearman,
    _std,
    build_player_performance_frame,
    explicit_position_mask,
    first_half_second_half_correlation,
    ols_residual,
    per_minute,
    position_duration_residual,
    position_r_squared,
    position_standardized,
    prior_player_history,
    repeatability_by_min_prior,
    restrict_development,
    slope_coefficient,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    FROZEN_DEVELOPMENT_MATCH_COUNT,
    FROZEN_HOLDOUT_EXPECTED_N,
    utc_datetime,
)

__all__ = [
    "EXPECTED_DEVELOPMENT_PLAYER_ROWS",
    "EXPECTED_EXPLICIT_POSITION_ROWS",
    "EXPECTED_HOLDOUT_PLAYER_ROWS",
    "EXPECTED_MISSING_POSITION_ROWS",
    "FARMING_CANDIDATE_COLUMN_NAMES",
    "FARMING_CANDIDATE_SPECS",
    "SLICE12_BASELINE_CANDIDATE",
    "SPARSE_GROUP_MIN_N",
    "TEAM_SWITCH_MIN_APPEARANCES",
    "TEAM_SWITCH_STRICT_MIN_APPEARANCES",
    "FarmingCandidateSpec",
    "Slice13DiagnosticReport",
    "attach_farming_candidates",
    "build_team_spells",
    "classify_slice13",
    "hero_excluded_prior_history",
    "pooled_group_labels",
    "position_duration_group_residual",
    "prior_history_excluding",
    "run_farming_performance_target_diagnostics",
    "slice13_report_to_jsonable",
    "team_period_centered",
    "team_switcher_table",
]


SLICE12_BASELINE_CANDIDATE = "last_hits_per_min_position_standardized"
CANDIDATE_A = SLICE12_BASELINE_CANDIDATE
CANDIDATE_B = "last_hits_per_min_position_duration_residual_z"
CANDIDATE_C = "last_hits_per_min_position_duration_hero_residual_z"
CANDIDATE_D = "last_hits_per_min_position_duration_hero_pos_residual_z"
SPARSE_GROUP_MIN_N = 5
SPARSE_LABEL = "__sparse__"
TEAM_SWITCH_MIN_APPEARANCES = 5
TEAM_SWITCH_STRICT_MIN_APPEARANCES = 10
TEAM_SWITCH_RANDOM_SEED = 202613
DURATION_QUINTILES = 5
_REPEATABILITY_FLOOR = 0.10
_POSITION_R2_NEUTRAL = 0.05
_DURATION_CORR_NEUTRAL = 0.05
_DURATION_CORR_SOFT = 0.15
_WIN_CORR_DISGUISED = 0.50
_HERO_DROP_FRACTION = 0.50

EXPECTED_DEVELOPMENT_PLAYER_ROWS = 59_670
EXPECTED_EXPLICIT_POSITION_ROWS = 59_640
EXPECTED_MISSING_POSITION_ROWS = 30
EXPECTED_HOLDOUT_PLAYER_ROWS = FROZEN_HOLDOUT_EXPECTED_N * 10


FarmingCandidateSpec = CandidateSpec


FARMING_CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name=CANDIDATE_A,
        family="farming",
        formula=(
            "z(num_last_hits / minutes | position); Slice 12 baseline, unchanged"
        ),
        adjustment_variables=("position",),
        hero_agnostic=True,
        source_fields=("num_last_hits",),
    ),
    CandidateSpec(
        name=CANDIDATE_B,
        family="farming",
        formula=(
            "z(OLS residual of LH/min ~ position dummies + duration_seconds)"
        ),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=True,
        source_fields=("num_last_hits",),
    ),
    CandidateSpec(
        name=CANDIDATE_C,
        family="farming",
        formula=(
            "z(OLS residual of LH/min ~ position dummies + duration_seconds "
            f"+ hero FE; heroes with n<{SPARSE_GROUP_MIN_N} pooled)"
        ),
        adjustment_variables=("position", "duration_seconds", "hero_id"),
        hero_agnostic=False,
        source_fields=("num_last_hits",),
    ),
    CandidateSpec(
        name=CANDIDATE_D,
        family="farming",
        formula=(
            "z(OLS residual of LH/min ~ position dummies + duration_seconds "
            f"+ hero×position FE; cells with n<{SPARSE_GROUP_MIN_N} pooled)"
        ),
        adjustment_variables=(
            "position",
            "duration_seconds",
            "hero_id",
            "position_number",
        ),
        hero_agnostic=False,
        source_fields=("num_last_hits",),
    ),
)
FARMING_CANDIDATE_COLUMN_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FARMING_CANDIDATE_SPECS
)
_FARMING_BY_NAME: dict[str, CandidateSpec] = {
    spec.name: spec for spec in FARMING_CANDIDATE_SPECS
}


@dataclass(frozen=True)
class Slice13DiagnosticReport:
    development_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    n_missing_position: int
    n_explicit_position: int
    coverage: pd.DataFrame
    formulas: pd.DataFrame
    duration_adjustment: pd.DataFrame
    duration_nonlinearity: pd.DataFrame
    residual_distribution: pd.DataFrame
    hero_adjustment: pd.DataFrame
    hero_pooling: pd.DataFrame
    hero_sample_sizes: pd.DataFrame
    candidate_comparison: pd.DataFrame
    candidate_position_means: pd.DataFrame
    repeatability: pd.DataFrame
    first_half_second_half: pd.DataFrame
    team_switcher: pd.DataFrame
    team_switcher_pairs: pd.DataFrame
    within_team_centered: pd.DataFrame
    hero_excluded_repeatability: pd.DataFrame
    winner_loser: pd.DataFrame
    patch_stability: pd.DataFrame
    patch_repeatability: pd.DataFrame
    cross_version_repeatability: pd.DataFrame
    falsification: pd.DataFrame
    classification: pd.DataFrame
    integrity: dict[str, object]


def pooled_group_labels(
    frame: pd.DataFrame,
    group_columns: tuple[str, ...],
    *,
    min_n: int = SPARSE_GROUP_MIN_N,
    eligible: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Label each eligible row with its group, pooling sparse groups.

    Rows that fail ``eligible`` stay null. Sparse groups share
    ``SPARSE_LABEL`` instead of being dropped.
    """
    if eligible is None:
        mask = pd.Series(True, index=frame.index)
    else:
        mask = eligible.reindex(frame.index).fillna(False).astype(bool)
    labels = pd.Series(pd.NA, index=frame.index, dtype="object")
    if not bool(mask.any()):
        summary = pd.DataFrame(
            [
                {
                    "n_rows_labeled": 0,
                    "n_groups_total": 0,
                    "n_groups_own_fe": 0,
                    "n_groups_pooled": 0,
                    "n_rows_own_fe": 0,
                    "n_rows_pooled": 0,
                    "min_n": min_n,
                    "group_columns": ", ".join(group_columns),
                }
            ]
        )
        return labels, summary

    keys = _group_key_series(frame.loc[mask], group_columns)
    counts = keys.value_counts(dropna=False)
    own = counts[counts >= min_n].index
    mapped = keys.where(keys.isin(own), other=SPARSE_LABEL)
    labels.loc[mask] = mapped.to_numpy()
    n_pooled_groups = int((~counts.index.isin(own)).sum())
    n_rows_pooled = int((mapped == SPARSE_LABEL).sum())
    summary = pd.DataFrame(
        [
            {
                "n_rows_labeled": int(mask.sum()),
                "n_groups_total": len(counts),
                "n_groups_own_fe": len(own),
                "n_groups_pooled": n_pooled_groups,
                "n_rows_own_fe": int(mask.sum()) - n_rows_pooled,
                "n_rows_pooled": n_rows_pooled,
                "min_n": min_n,
                "group_columns": ", ".join(group_columns),
            }
        ]
    )
    return labels, summary


def _residual_zscore(series: pd.Series) -> pd.Series:
    """Standardize a residual. Near-zero variance is treated as identically zero.

    Slice 12's ``_zscore`` divides by a tiny floating-point std after a
    perfect OLS fit and produces huge junk values. Farming candidates
    need the mathematically zero residual to stay zero.
    """
    values = _numeric(series)
    finite = values[values.notna()]
    if finite.empty:
        return values
    std = float(finite.std(ddof=0))
    if std <= 1e-12:
        return values - float(finite.mean())
    return (values - float(finite.mean())) / std


def _group_key_series(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    parts = [frame[column].astype("string") for column in columns]
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key


def position_duration_design(frame: pd.DataFrame) -> pd.DataFrame:
    """Intercept + position dummies (drop-first) + ``duration_seconds``."""
    design = _position_dummy_frame(frame)
    design["duration_seconds"] = _numeric(frame["duration_seconds"])
    return design


def position_duration_group_residual(
    frame: pd.DataFrame,
    column: str,
    group_labels: pd.Series,
) -> tuple[pd.Series, np.ndarray]:
    """OLS residual of ``column ~ position + duration + group FE``.

    Sparse groups already share a label (see ``pooled_group_labels``).
    NULL-position rows stay null. No rows are dropped for sparsity.
    """
    eligible = explicit_position_mask(frame)
    y = _numeric(frame[column]).where(eligible)
    numeric = position_duration_design(frame).where(eligible)
    dummies = pd.get_dummies(group_labels, prefix="g", drop_first=True, dtype=float)
    dummies = dummies.where(eligible & group_labels.notna())
    design = pd.concat([numeric, dummies], axis=1)
    return ols_residual(y, design)


def attach_farming_candidates(
    frame: pd.DataFrame, *, sparse_min_n: int = SPARSE_GROUP_MIN_N
) -> pd.DataFrame:
    """Add Slice 13 farming candidates A–D. Does not impute missing position.

    Candidate A is the Slice 12 ``last_hits_per_min_position_standardized``
    column, computed with the same ``position_standardized`` helper.
    """
    out = frame.copy()
    out["last_hits_per_minute"] = per_minute(
        out["num_last_hits"], out["duration_seconds"]
    )
    out[CANDIDATE_A] = position_standardized(out, "last_hits_per_minute")
    eligible = explicit_position_mask(out)
    pos_dur = position_duration_residual(out, "last_hits_per_minute")
    out["_lhpm_position_duration_residual"] = pos_dur
    out[CANDIDATE_B] = _residual_zscore(pos_dur)

    hero_labels, _hero_summary = pooled_group_labels(
        out, ("hero_id",), min_n=sparse_min_n, eligible=eligible
    )
    hero_resid, _hero_coef = position_duration_group_residual(
        out, "last_hits_per_minute", hero_labels
    )
    out["_lhpm_position_duration_hero_residual"] = hero_resid
    out[CANDIDATE_C] = _residual_zscore(hero_resid)

    cell_labels, _cell_summary = pooled_group_labels(
        out,
        ("hero_id", "position_number"),
        min_n=sparse_min_n,
        eligible=eligible,
    )
    cell_resid, _cell_coef = position_duration_group_residual(
        out, "last_hits_per_minute", cell_labels
    )
    out["_lhpm_position_duration_hero_pos_residual"] = cell_resid
    out[CANDIDATE_D] = _residual_zscore(cell_resid)
    return out


def prior_history_excluding(
    frame: pd.DataFrame,
    column: str,
    exclude_column: str,
) -> tuple[pd.Series, pd.Series]:
    """Strictly prior player mean excluding rows matching ``exclude_column``.

    ``H.start_time < M.start_time``. Same-timestamp rows are mutually
    blind. Current/future player identity is not used except as the
    grouping key for *past* appearances of the same player.
    """
    values = _numeric(frame[column])
    means = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0, index=frame.index, dtype=int)
    if frame.empty:
        return means, counts
    times = pd.to_datetime(frame["start_time"], utc=True)
    excluded = frame[exclude_column]
    work = pd.DataFrame(
        {
            "player_id": frame["player_id"].to_numpy(),
            "time": times.to_numpy(),
            "value": values.to_numpy(dtype=float),
            "exclude": excluded.to_numpy(),
        },
        index=frame.index,
    )
    for _player_id, group in work.groupby("player_id", sort=False):
        ordered = group.sort_values("time", kind="mergesort")
        stamps = ordered["time"].to_numpy()
        vals = ordered["value"].to_numpy(dtype=float)
        keys = ordered["exclude"].to_numpy()
        player_means = np.full(len(ordered), np.nan)
        player_counts = np.zeros(len(ordered), dtype=int)
        for i in range(len(ordered)):
            prior = (
                (stamps < stamps[i])
                & np.isfinite(vals)
                & _values_not_equal(keys, keys[i])
            )
            player_counts[i] = int(prior.sum())
            if player_counts[i] > 0:
                player_means[i] = float(vals[prior].mean())
        means.loc[ordered.index] = player_means
        counts.loc[ordered.index] = player_counts
    return means, counts


def hero_excluded_prior_history(
    frame: pd.DataFrame, column: str
) -> tuple[pd.Series, pd.Series]:
    """Prior player mean of ``column`` excluding the current hero."""
    return prior_history_excluding(frame, column, "hero_id")


def _values_not_equal(values: np.ndarray, current: object) -> np.ndarray:
    if current is None or (isinstance(current, float) and not np.isfinite(current)):
        return np.array(
            [
                item is not None
                and not (isinstance(item, float) and not np.isfinite(item))
                for item in values
            ],
            dtype=bool,
        )
    compared = values == current
    missing = pd.isna(values)
    return (~compared) & (~missing)


def _repeatability_from_prior(
    frame: pd.DataFrame,
    column: str,
    prior_mean: pd.Series,
    prior_n: pd.Series,
    *,
    thresholds: tuple[int, ...] = REPEATABILITY_PRIOR_THRESHOLDS,
    kind: str,
) -> pd.DataFrame:
    current = _numeric(frame[column])
    rows: list[dict[str, object]] = []
    for minimum in thresholds:
        mask = (prior_n >= minimum) & current.notna() & prior_mean.notna()
        rows.append(
            {
                "candidate": column,
                "kind": kind,
                "min_prior_appearances": minimum,
                "n_rows": int(mask.sum()),
                "n_players": int(frame.loc[mask, "player_id"].nunique())
                if int(mask.sum())
                else 0,
                "pearson": _pearson(prior_mean[mask], current[mask]),
                "spearman": _spearman(prior_mean[mask], current[mask]),
            }
        )
    return pd.DataFrame(rows)


def duration_nonlinearity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Diagnostics for leftover duration shape after linear residualization.

    Does not search alternative target formulas. Candidate B remains the
    linear position+duration residual.
    """
    residual = _numeric(frame["_lhpm_position_duration_residual"])
    duration = _numeric(frame["duration_seconds"])
    eligible = explicit_position_mask(frame) & residual.notna() & duration.notna()
    y = residual.loc[eligible]
    d = duration.loc[eligible]
    d2 = d.to_numpy(dtype=float) ** 2
    log_d = np.log(d.to_numpy(dtype=float))
    log_d = np.where(d.to_numpy(dtype=float) > 0.0, log_d, np.nan)
    minutes = d / 60.0
    work = frame.loc[eligible].assign(
        _resid=y.to_numpy(dtype=float),
        _duration=d.to_numpy(dtype=float),
    )
    try:
        work["_q"] = pd.qcut(
            work["_duration"], DURATION_QUINTILES, labels=False, duplicates="drop"
        )
    except ValueError:
        work["_q"] = np.nan
    quintile_means: dict[str, float] = {}
    quintile_abs_max = float("nan")
    if work["_q"].notna().any():
        grouped = work.groupby("_q", dropna=True)["_resid"]
        means = grouped.mean()
        for q, value in means.items():
            quintile_means[f"quintile_{int(q) + 1}_mean"] = float(value)
        quintile_abs_max = float(np.nanmax(np.abs(means.to_numpy(dtype=float))))
    return pd.DataFrame(
        [
            {
                "n": int(eligible.sum()),
                "pearson_residual_duration": _pearson(y, d),
                "pearson_residual_duration_squared": _pearson(
                    y, pd.Series(d2, index=y.index)
                ),
                "pearson_residual_log_duration": _pearson(
                    y, pd.Series(log_d, index=y.index)
                ),
                "pearson_residual_duration_minutes": _pearson(y, minutes),
                "spearman_residual_duration": _spearman(y, d),
                "quintile_abs_max_mean": quintile_abs_max,
                **quintile_means,
            }
        ]
    )


def residual_distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in FARMING_CANDIDATE_SPECS:
        values = _numeric(frame[spec.name])
        finite = values.dropna().to_numpy(dtype=float)
        rows.append(
            {
                "candidate": spec.name,
                "n": int(finite.size),
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "std": _std(finite),
                "min": float(finite.min()) if finite.size else float("nan"),
                "max": float(finite.max()) if finite.size else float("nan"),
                "q01": float(np.quantile(finite, 0.01)) if finite.size else float("nan"),
                "q05": float(np.quantile(finite, 0.05)) if finite.size else float("nan"),
                "q50": float(np.quantile(finite, 0.50)) if finite.size else float("nan"),
                "q95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
                "q99": float(np.quantile(finite, 0.99)) if finite.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def farming_position_means_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = explicit_position_mask(frame)
    for spec in FARMING_CANDIDATE_SPECS:
        values = _numeric(frame[spec.name])
        present = eligible & values.notna()
        row: dict[str, object] = {
            "candidate": spec.name,
            "n_explicit_position": int(present.sum()),
            "position_r2": position_r_squared(frame.assign(_c=values), "_c"),
        }
        for number in EXPLICIT_POSITION_NUMBERS:
            subset = values.loc[present & (frame["position_number"] == number)]
            finite = subset.to_numpy(dtype=float)
            row[f"pos{number}_n"] = int(finite.size)
            row[f"pos{number}_mean"] = (
                float(finite.mean()) if finite.size else float("nan")
            )
            row[f"pos{number}_std"] = _std(finite)
        rows.append(row)
    return pd.DataFrame(rows)


def hero_sample_size_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[explicit_position_mask(frame)]
    counts = (
        eligible.groupby("hero_id", dropna=False)
        .size()
        .rename("n_appearances")
        .reset_index()
        .sort_values("n_appearances", ascending=False, kind="mergesort")
    )
    counts["own_fe"] = counts["n_appearances"] >= SPARSE_GROUP_MIN_N
    return counts


def hero_adjustment_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[explicit_position_mask(frame)].copy()
    lhpm = _numeric(eligible["last_hits_per_minute"])
    rows: list[dict[str, object]] = []
    for spec in FARMING_CANDIDATE_SPECS:
        values = _numeric(eligible[spec.name])
        work = eligible.loc[values.notna() & lhpm.notna()]
        hero_r2 = _categorical_r2(work[spec.name], work["hero_id"])
        residual = work.assign(_resid=work[spec.name])
        within = _categorical_r2(residual["_resid"], residual["hero_id"])
        pos_hero = work.assign(
            _cell=_group_key_series(work, ("hero_id", "position_number"))
        )
        hero_pos_r2 = _categorical_r2(pos_hero[spec.name], pos_hero["_cell"])
        hero_n = work.groupby("hero_id").size()
        rows.append(
            {
                "candidate": spec.name,
                "n": len(work),
                "n_heroes": int(work["hero_id"].nunique()),
                "n_heroes_ge_min_fe": int((hero_n >= SPARSE_GROUP_MIN_N).sum()),
                "n_heroes_ge_slice12_min": int((hero_n >= MIN_HERO_APPEARANCES).sum()),
                "hero_r2": hero_r2,
                "hero_r2_on_candidate": within,
                "hero_position_r2": hero_pos_r2,
            }
        )
    raw_work = eligible.loc[lhpm.notna()]
    if not raw_work.empty:
        rows.insert(
            0,
            {
                "candidate": "last_hits_per_minute",
                "n": len(raw_work),
                "n_heroes": int(raw_work["hero_id"].nunique()),
                "n_heroes_ge_min_fe": int(
                    (
                        raw_work.groupby("hero_id").size() >= SPARSE_GROUP_MIN_N
                    ).sum()
                ),
                "n_heroes_ge_slice12_min": int(
                    (
                        raw_work.groupby("hero_id").size() >= MIN_HERO_APPEARANCES
                    ).sum()
                ),
                "hero_r2": _categorical_r2(
                    raw_work["last_hits_per_minute"], raw_work["hero_id"]
                ),
                "hero_r2_on_candidate": _categorical_r2(
                    raw_work["last_hits_per_minute"], raw_work["hero_id"]
                ),
                "hero_position_r2": _categorical_r2(
                    raw_work["last_hits_per_minute"],
                    _group_key_series(raw_work, ("hero_id", "position_number")),
                ),
            },
        )
    return pd.DataFrame(rows)


def candidate_comparison_table(
    frame: pd.DataFrame,
    repeatability: pd.DataFrame,
    halves: pd.DataFrame,
    hero_adj: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    half_by = halves.set_index("candidate") if not halves.empty else pd.DataFrame()
    hero_by = (
        hero_adj.set_index("candidate") if not hero_adj.empty else pd.DataFrame()
    )
    by_rep = {name: group for name, group in repeatability.groupby("candidate")}
    for spec in FARMING_CANDIDATE_SPECS:
        values = _numeric(frame[spec.name])
        finite = values.dropna().to_numpy(dtype=float)
        pos_r2 = position_r_squared(frame.assign(_c=values), "_c")
        r10 = float("nan")
        n10 = 0
        n_players_10 = 0
        rep = by_rep.get(spec.name)
        if rep is not None:
            at10 = rep.loc[rep["min_prior_appearances"] == MIN_PRIOR_APPEARANCES]
            if not at10.empty:
                r10 = float(at10.iloc[0]["pearson"])
                n10 = int(at10.iloc[0]["n_rows"])
                n_players_10 = int(at10.iloc[0]["n_players"])
        half_r = (
            float(half_by.loc[spec.name, "pearson"])
            if spec.name in half_by.index
            else float("nan")
        )
        hero_r2 = (
            float(hero_by.loc[spec.name, "hero_r2"])
            if spec.name in hero_by.index
            else float("nan")
        )
        rows.append(
            {
                "candidate": spec.name,
                "formula": spec.formula,
                "adjustment_variables": ", ".join(spec.adjustment_variables),
                "n": int(values.notna().sum()),
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "std": _std(finite),
                "position_r2": pos_r2,
                "pearson_duration": _pearson(values, frame["duration_seconds"]),
                "pearson_team_won": _pearson(values, frame["team_won"]),
                "pearson_elo_expected_win": _pearson(
                    values, frame["elo_expected_win"]
                ),
                "elo_slope": slope_coefficient(values, frame["elo_expected_win"]),
                "hero_r2": hero_r2,
                "repeatability_pearson_min_prior_10": r10,
                "repeatability_n_min_prior_10": n10,
                "repeatability_n_players_min_prior_10": n_players_10,
                "first_half_second_half_pearson": half_r,
            }
        )
    return pd.DataFrame(rows)


def build_team_spells(
    frame: pd.DataFrame,
    column: str,
    *,
    min_appearances: int,
) -> pd.DataFrame:
    """Consecutive canonical-``team_id`` spells with a candidate mean.

    A spell is a run of appearances for one ``team_id`` after sorting
    by ``start_time`` (then ``match_id``). Future spells do not enter
    an earlier spell's mean. Spells shorter than ``min_appearances``
    are omitted so trivial stand-ins are not treated as evidence.
    """
    values = _numeric(frame[column])
    work = frame.loc[
        values.notna() & frame["team_id"].notna() & frame["player_id"].notna(),
        ["player_id", "team_id", "match_id", "start_time", column],
    ].copy()
    work[column] = values.loc[work.index]
    work["start_time"] = pd.to_datetime(work["start_time"], utc=True)
    if work.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "team_id",
                "spell_index",
                "n_appearances",
                "start_time_first",
                "start_time_last",
                "match_id_first",
                "match_id_last",
                "mean_value",
                "candidate",
                "min_appearances",
            ]
        )
    rows: list[dict[str, object]] = []
    for player_id, group in work.groupby("player_id", sort=False):
        ordered = group.sort_values(
            ["start_time", "match_id"], kind="mergesort"
        )
        team_ids = ordered["team_id"].to_numpy()
        if len(ordered) == 0:
            continue
        spell_starts = [0]
        for i in range(1, len(ordered)):
            if team_ids[i] != team_ids[i - 1]:
                spell_starts.append(i)
        spell_starts.append(len(ordered))
        spell_index = 0
        for begin, end in pairwise(spell_starts):
            spell = ordered.iloc[begin:end]
            n = len(spell)
            if n < min_appearances:
                continue
            rows.append(
                {
                    "player_id": int(player_id),
                    "team_id": int(spell["team_id"].iloc[0]),
                    "spell_index": spell_index,
                    "n_appearances": n,
                    "start_time_first": spell["start_time"].iloc[0],
                    "start_time_last": spell["start_time"].iloc[-1],
                    "match_id_first": int(spell["match_id"].iloc[0]),
                    "match_id_last": int(spell["match_id"].iloc[-1]),
                    "mean_value": float(_numeric(spell[column]).mean()),
                    "candidate": column,
                    "min_appearances": min_appearances,
                }
            )
            spell_index += 1
    return pd.DataFrame(rows)


def team_switcher_table(
    frame: pd.DataFrame,
    column: str,
    *,
    min_appearances: tuple[int, ...] = (
        TEAM_SWITCH_MIN_APPEARANCES,
        TEAM_SWITCH_STRICT_MIN_APPEARANCES,
    ),
    random_seed: int = TEAM_SWITCH_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same-player old-team vs new-team means, plus simple baselines."""
    summary_rows: list[dict[str, object]] = []
    pair_parts: list[pd.DataFrame] = []
    for minimum in min_appearances:
        spells = build_team_spells(frame, column, min_appearances=minimum)
        pairs = _spell_transitions(spells)
        pairs["candidate"] = column
        pairs["min_appearances"] = minimum
        pair_parts.append(pairs)
        n_players = int(pairs["player_id"].nunique()) if not pairs.empty else 0
        n_transitions = len(pairs)
        if pairs.empty:
            summary_rows.append(
                {
                    "candidate": column,
                    "min_appearances": minimum,
                    "n_qualifying_players": 0,
                    "n_team_transitions": 0,
                    "n_spells": len(spells),
                    "pearson_old_new": float("nan"),
                    "spearman_old_new": float("nan"),
                    "mean_absolute_change": float("nan"),
                    "pearson_random_different_player": float("nan"),
                    "pearson_different_players_same_team": float("nan"),
                    "n_different_player_same_team_pairs": 0,
                }
            )
            continue
        old_new = _pearson(pairs["old_mean"], pairs["new_mean"])
        abs_change = float(np.mean(np.abs(pairs["new_mean"] - pairs["old_mean"])))
        rng = np.random.default_rng(random_seed + minimum)
        shuffled = pairs["new_mean"].to_numpy(dtype=float).copy()
        rng.shuffle(shuffled)
        random_r = _pearson(pairs["old_mean"], pd.Series(shuffled, index=pairs.index))
        same_team = _different_players_same_team(frame, column, min_appearances=minimum)
        summary_rows.append(
            {
                "candidate": column,
                "min_appearances": minimum,
                "n_qualifying_players": n_players,
                "n_team_transitions": n_transitions,
                "n_spells": len(spells),
                "pearson_old_new": old_new,
                "spearman_old_new": _spearman(pairs["old_mean"], pairs["new_mean"]),
                "mean_absolute_change": abs_change,
                "pearson_random_different_player": random_r,
                "pearson_different_players_same_team": same_team["pearson"],
                "n_different_player_same_team_pairs": same_team["n_pairs"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    pairs_out = (
        pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    )
    return summary, pairs_out


def _spell_transitions(spells: pd.DataFrame) -> pd.DataFrame:
    if spells.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "old_team_id",
                "new_team_id",
                "old_spell_index",
                "new_spell_index",
                "old_n",
                "new_n",
                "old_mean",
                "new_mean",
                "old_end",
                "new_start",
            ]
        )
    rows: list[dict[str, object]] = []
    for player_id, group in spells.groupby("player_id", sort=False):
        ordered = group.sort_values("spell_index", kind="mergesort")
        teams = ordered["team_id"].to_numpy()
        for i in range(len(ordered) - 1):
            if teams[i] == teams[i + 1]:
                continue
            earlier = ordered.iloc[i]
            later = ordered.iloc[i + 1]
            if pd.Timestamp(later["start_time_first"]) < pd.Timestamp(
                earlier["start_time_last"]
            ):
                continue
            rows.append(
                {
                    "player_id": int(player_id),
                    "old_team_id": int(earlier["team_id"]),
                    "new_team_id": int(later["team_id"]),
                    "old_spell_index": int(earlier["spell_index"]),
                    "new_spell_index": int(later["spell_index"]),
                    "old_n": int(earlier["n_appearances"]),
                    "new_n": int(later["n_appearances"]),
                    "old_mean": float(earlier["mean_value"]),
                    "new_mean": float(later["mean_value"]),
                    "old_end": earlier["start_time_last"],
                    "new_start": later["start_time_first"],
                }
            )
    return pd.DataFrame(rows)


def _different_players_same_team(
    frame: pd.DataFrame, column: str, *, min_appearances: int
) -> dict[str, object]:
    values = _numeric(frame[column])
    work = frame.loc[values.notna(), ["player_id", "team_id", column]].copy()
    work[column] = values.loc[work.index]
    counts = work.groupby(["team_id", "player_id"]).size().rename("n")
    keep = counts[counts >= min_appearances].reset_index()
    if keep.empty:
        return {"pearson": float("nan"), "n_pairs": 0}
    means = (
        work.merge(keep[["team_id", "player_id"]], on=["team_id", "player_id"])
        .groupby(["team_id", "player_id"])[column]
        .mean()
        .reset_index()
    )
    left = means.rename(
        columns={"player_id": "player_a", column: "mean_a"}
    )
    right = means.rename(
        columns={"player_id": "player_b", column: "mean_b"}
    )
    pairs = left.merge(right, on="team_id")
    pairs = pairs.loc[pairs["player_a"] < pairs["player_b"]]
    if pairs.empty:
        return {"pearson": float("nan"), "n_pairs": 0}
    return {
        "pearson": _pearson(pairs["mean_a"], pairs["mean_b"]),
        "n_pairs": len(pairs),
    }


def team_period_centered(
    frame: pd.DataFrame,
    column: str,
    *,
    period_columns: tuple[str, ...] = ("team_id", "game_version_id"),
) -> pd.Series:
    """Appearance minus leave-one-player-out team-period mean.

    The current player's other appearances in the same period are
    excluded from the teammate mean, so a high-volume player does not
    define the structure being subtracted. Periods with no *other*
    player remain null.
    """
    values = _numeric(frame[column])
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    work = pd.DataFrame(
        {
            "value": values,
            "player_id": frame["player_id"],
            **{column_name: frame[column_name] for column_name in period_columns},
        },
        index=frame.index,
    )
    work = work.loc[work["value"].notna() & work["player_id"].notna()]
    for column_name in period_columns:
        work = work.loc[work[column_name].notna()]
    if work.empty:
        return out
    group_cols = list(period_columns)
    g_sum = work.groupby(group_cols, sort=False)["value"].transform("sum")
    g_cnt = work.groupby(group_cols, sort=False)["value"].transform("count")
    player_cols = group_cols + ["player_id"]
    p_sum = work.groupby(player_cols, sort=False)["value"].transform("sum")
    p_cnt = work.groupby(player_cols, sort=False)["value"].transform("count")
    others_n = g_cnt - p_cnt
    others_mean = (g_sum - p_sum) / others_n
    centered = work["value"] - others_mean
    out.loc[centered.index] = centered.where(others_n > 0)
    return out


def _within_team_repeatability(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    centered = team_period_centered(frame, column)
    work = frame.assign(_centered=centered)
    player_team = _player_mean_vs_team_mean(frame, column)
    loo = _loo_team_match_residual(frame, column)
    loo_frame = frame.assign(_loo=loo)
    loo_rep = repeatability_by_min_prior(loo_frame, "_loo")
    centered_rep = repeatability_by_min_prior(work, "_centered")
    rows: list[dict[str, object]] = []
    for minimum in REPEATABILITY_PRIOR_THRESHOLDS:
        c_at = centered_rep.loc[centered_rep["min_prior_appearances"] == minimum]
        l_at = loo_rep.loc[loo_rep["min_prior_appearances"] == minimum]
        rows.append(
            {
                "candidate": column,
                "min_prior_appearances": minimum,
                "n_centered_rows": int(c_at.iloc[0]["n_rows"]) if not c_at.empty else 0,
                "n_centered_players": int(c_at.iloc[0]["n_players"])
                if not c_at.empty
                else 0,
                "pearson_team_period_centered": (
                    float(c_at.iloc[0]["pearson"]) if not c_at.empty else float("nan")
                ),
                "n_loo_teammate_rows": int(l_at.iloc[0]["n_rows"])
                if not l_at.empty
                else 0,
                "pearson_loo_teammate_residual": (
                    float(l_at.iloc[0]["pearson"]) if not l_at.empty else float("nan")
                ),
                **player_team,
            }
        )
    return pd.DataFrame(rows)


def winner_loser_repeatability_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    winners = frame.loc[frame["team_won"] == 1]
    losers = frame.loc[frame["team_won"] == 0]
    win_rep = repeatability_by_min_prior(winners, column)
    lose_rep = repeatability_by_min_prior(losers, column)
    win_rep = win_rep.assign(subset="winners")
    lose_rep = lose_rep.assign(subset="losers")
    return pd.concat([win_rep, lose_rep], ignore_index=True)


def patch_stability_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for version, group in frame.groupby("game_version_id", dropna=False):
        for spec in FARMING_CANDIDATE_SPECS:
            values = _numeric(group[spec.name]).dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "game_version_id": version,
                    "candidate": spec.name,
                    "n": int(values.size),
                    "mean": float(values.mean()) if values.size else float("nan"),
                    "std": _std(values),
                    "median": float(np.median(values)) if values.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def patch_repeatability_table(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for version, group in frame.groupby("game_version_id", dropna=False):
        if len(group) < 50:
            continue
        for spec in FARMING_CANDIDATE_SPECS:
            rep = repeatability_by_min_prior(group, spec.name)
            rep = rep.assign(game_version_id=version, kind="within_version")
            parts.append(rep)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def classify_slice13(report: Slice13DiagnosticReport) -> pd.DataFrame:
    """Map Slice 13 tables onto the A / B / C decision gate.

    Thresholds are documented floors, not quantities tuned against
    team victory. Candidate C/D are falsification instruments; a freeze
    recommendation names the duration-adjusted formula (B) when A-gate
    passes.
    """
    comparison = report.candidate_comparison
    if comparison.empty:
        return pd.DataFrame(
            [
                {
                    "classification": "C",
                    "recommended_candidate": None,
                    "recommended_formula": None,
                    "rationale": "No candidate comparison rows were produced.",
                    "next_slice": (
                        "Do not freeze a farming target; Slice 12 apparent "
                        "persistence was not recoverable in Slice 13."
                    ),
                }
            ]
        )
    b_rows = comparison.loc[comparison["candidate"] == CANDIDATE_B]
    if b_rows.empty:
        b_rows = comparison.loc[comparison["candidate"] == CANDIDATE_A]
    row = b_rows.iloc[0]
    r10 = float(row["repeatability_pearson_min_prior_10"])
    duration_r = float(row["pearson_duration"])
    pos_r2 = float(row["position_r2"])
    win_r = float(row["pearson_team_won"])
    hero_r2 = float(row["hero_r2"]) if "hero_r2" in row else float("nan")

    hero_ex = report.hero_excluded_repeatability
    hero_ex_r10 = _metric_at_prior_10(hero_ex, CANDIDATE_B, "pearson")
    if not np.isfinite(hero_ex_r10):
        hero_ex_r10 = _metric_at_prior_10(hero_ex, CANDIDATE_A, "pearson")

    switch = report.team_switcher
    switch_r = float("nan")
    switch_n = 0
    if not switch.empty:
        primary = switch.loc[
            (switch["candidate"] == CANDIDATE_B)
            & (switch["min_appearances"] == TEAM_SWITCH_MIN_APPEARANCES)
        ]
        if primary.empty:
            primary = switch.loc[
                (switch["candidate"] == CANDIDATE_A)
                & (switch["min_appearances"] == TEAM_SWITCH_MIN_APPEARANCES)
            ]
        if not primary.empty:
            switch_r = float(primary.iloc[0]["pearson_old_new"])
            switch_n = int(primary.iloc[0]["n_qualifying_players"])

    centered = report.within_team_centered
    centered_r10 = _metric_at_prior_10(
        centered, CANDIDATE_B, "pearson_team_period_centered"
    )
    if not np.isfinite(centered_r10):
        centered_r10 = _metric_at_prior_10(
            centered, CANDIDATE_A, "pearson_team_period_centered"
        )

    duration_neutral = np.isfinite(duration_r) and abs(duration_r) <= _DURATION_CORR_NEUTRAL
    duration_soft = np.isfinite(duration_r) and abs(duration_r) <= _DURATION_CORR_SOFT
    repeatable = np.isfinite(r10) and r10 >= _REPEATABILITY_FLOOR
    role_neutral = np.isfinite(pos_r2) and abs(pos_r2) <= _POSITION_R2_NEUTRAL
    not_win_label = np.isfinite(win_r) and abs(win_r) < _WIN_CORR_DISGUISED
    hero_survives = np.isfinite(hero_ex_r10) and hero_ex_r10 >= _REPEATABILITY_FLOOR
    if np.isfinite(r10) and r10 > 0 and np.isfinite(hero_ex_r10):
        hero_survives = hero_survives and hero_ex_r10 >= r10 * _HERO_DROP_FRACTION
    team_survives = (
        switch_n >= 8
        and np.isfinite(switch_r)
        and switch_r >= _REPEATABILITY_FLOOR
        and np.isfinite(centered_r10)
        and centered_r10 >= _REPEATABILITY_FLOOR
    )
    team_weak = (switch_n < 8) or (
        np.isfinite(switch_r) and switch_r < _REPEATABILITY_FLOOR
    ) or (np.isfinite(centered_r10) and centered_r10 < _REPEATABILITY_FLOOR)

    freeze = (
        repeatable
        and duration_neutral
        and role_neutral
        and not_win_label
        and hero_survives
        and team_survives
    )
    persistent = repeatable and duration_soft and role_neutral and not_win_label
    if freeze:
        classification = "A"
        rationale = (
            "Duration-adjusted last-hit rate remains repeatable, "
            "duration-neutral, and player-specific after hero and team checks."
        )
        next_slice = (
            "Freeze candidate B as a farming-tendency target and, in the "
            "next slice only, estimate a shrunk historical player farming "
            "state. Do not treat it as overall player skill."
        )
        recommended = CANDIDATE_B
        formula = _FARMING_BY_NAME[CANDIDATE_B].formula
    elif persistent and (not hero_survives or team_weak or not duration_neutral):
        classification = "B"
        rationale = (
            "Player-level farming persistence survives duration adjustment "
            "but hero and/or team structure remain too important to freeze "
            "one target."
        )
        next_slice = (
            "Keep candidate B as the working residual and run one narrow "
            "follow-up: quantify how much remaining persistence is "
            "player×hero farming versus teammate/team-period structure, "
            "without building a rating."
        )
        recommended = CANDIDATE_B
        formula = _FARMING_BY_NAME[CANDIDATE_B].formula
    else:
        classification = "C"
        rationale = (
            "Duration, hero, or team correction substantially removes the "
            "Slice 12 last-hit-rate persistence, so the apparent player "
            "signal was mostly contextual."
        )
        next_slice = (
            "Do not freeze a farming target. If work continues, treat last "
            "hits as a contextual outcome rather than a player trait."
        )
        recommended = None
        formula = None

    return pd.DataFrame(
        [
            {
                "classification": classification,
                "recommended_candidate": recommended,
                "recommended_formula": formula,
                "rationale": rationale,
                "next_slice": next_slice,
                "b_repeatability_min_prior_10": r10,
                "b_pearson_duration": duration_r,
                "b_position_r2": pos_r2,
                "b_pearson_team_won": win_r,
                "b_hero_r2": hero_r2,
                "b_hero_excluded_r10": hero_ex_r10,
                "b_team_switch_r": switch_r,
                "b_team_switch_n_players": switch_n,
                "b_team_centered_r10": centered_r10,
                "duration_neutral": duration_neutral,
                "hero_survives": hero_survives,
                "team_survives": team_survives,
            }
        ]
    )


def _metric_at_prior_10(table: pd.DataFrame, candidate: str, column: str) -> float:
    if table.empty or "candidate" not in table.columns:
        return float("nan")
    work = table.loc[table["candidate"] == candidate]
    if "min_prior_appearances" in work.columns:
        work = work.loc[work["min_prior_appearances"] == MIN_PRIOR_APPEARANCES]
    if work.empty or column not in work.columns:
        return float("nan")
    return float(work.iloc[0][column])


def run_farming_performance_target_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
    sparse_min_n: int = SPARSE_GROUP_MIN_N,
) -> Slice13DiagnosticReport:
    """Development-only Slice 13 farming-target research. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    development = attach_farming_candidates(development, sparse_min_n=sparse_min_n)
    n_missing_position = int((~explicit_position_mask(development)).sum())
    n_explicit = int(explicit_position_mask(development).sum())
    coverage = pd.DataFrame(
        [
            {
                "n_development_matches": int(development["match_id"].nunique()),
                "n_development_player_rows": len(development),
                "n_explicit_position": n_explicit,
                "n_missing_position": n_missing_position,
                "n_holdout_excluded": len(holdout),
                "n_holdout_matches": int(holdout["match_id"].nunique())
                if len(holdout)
                else 0,
                "expected_development_matches": FROZEN_DEVELOPMENT_MATCH_COUNT,
                "expected_development_player_rows": EXPECTED_DEVELOPMENT_PLAYER_ROWS,
                "expected_explicit_position": EXPECTED_EXPLICIT_POSITION_ROWS,
                "expected_missing_position": EXPECTED_MISSING_POSITION_ROWS,
                "expected_holdout_player_rows": EXPECTED_HOLDOUT_PLAYER_ROWS,
            }
        ]
    )
    formulas = pd.DataFrame(
        [
            {
                "candidate": spec.name,
                "family": spec.family,
                "formula": spec.formula,
                "adjustment_variables": ", ".join(spec.adjustment_variables),
                "hero_agnostic": spec.hero_agnostic,
                "role": (
                    "baseline"
                    if spec.name == CANDIDATE_A
                    else "duration_adjusted"
                    if spec.name == CANDIDATE_B
                    else "falsification"
                ),
            }
            for spec in FARMING_CANDIDATE_SPECS
        ]
    )
    duration_adjustment = pd.DataFrame(
        [
            {
                "candidate": spec.name,
                "pearson_duration": _pearson(
                    development[spec.name], development["duration_seconds"]
                ),
                "spearman_duration": _spearman(
                    development[spec.name], development["duration_seconds"]
                ),
                "position_r2": position_r_squared(
                    development.assign(_c=_numeric(development[spec.name])), "_c"
                ),
            }
            for spec in FARMING_CANDIDATE_SPECS
        ]
    )
    duration_nonlinearity = duration_nonlinearity_table(development)
    residual_distribution = residual_distribution_table(development)
    hero_sample_sizes = hero_sample_size_table(development)
    hero_adjustment = hero_adjustment_table(development)
    eligible = explicit_position_mask(development)
    _hero_labels, hero_summary = pooled_group_labels(
        development, ("hero_id",), min_n=sparse_min_n, eligible=eligible
    )
    _cell_labels, cell_summary = pooled_group_labels(
        development,
        ("hero_id", "position_number"),
        min_n=sparse_min_n,
        eligible=eligible,
    )
    hero_pooling = pd.concat(
        [
            hero_summary.assign(adjustment="hero"),
            cell_summary.assign(adjustment="hero_position"),
        ],
        ignore_index=True,
    )

    repeat_parts = [
        repeatability_by_min_prior(development, spec.name)
        for spec in FARMING_CANDIDATE_SPECS
    ]
    repeatability = pd.concat(repeat_parts, ignore_index=True)
    halves = pd.DataFrame(
        [
            first_half_second_half_correlation(development, spec.name)
            for spec in FARMING_CANDIDATE_SPECS
        ]
    )
    comparison = candidate_comparison_table(
        development, repeatability, halves, hero_adjustment
    )
    position_means = farming_position_means_table(development)

    switch_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []
    for spec in FARMING_CANDIDATE_SPECS:
        summary, pairs = team_switcher_table(development, spec.name)
        switch_parts.append(summary)
        pair_parts.append(pairs)
    team_switcher = pd.concat(switch_parts, ignore_index=True)
    team_switcher_pairs = pd.concat(pair_parts, ignore_index=True)

    centered_parts = [
        _within_team_repeatability(development, spec.name)
        for spec in FARMING_CANDIDATE_SPECS
    ]
    within_team_centered = pd.concat(centered_parts, ignore_index=True)

    hero_ex_parts: list[pd.DataFrame] = []
    for spec in FARMING_CANDIDATE_SPECS:
        prior_mean, prior_n = hero_excluded_prior_history(development, spec.name)
        hero_ex_parts.append(
            _repeatability_from_prior(
                development,
                spec.name,
                prior_mean,
                prior_n,
                kind="hero_excluded",
            )
        )
        ordinary_mean, ordinary_n = prior_player_history(development, spec.name)
        hero_ex_parts.append(
            _repeatability_from_prior(
                development,
                spec.name,
                ordinary_mean,
                ordinary_n,
                kind="ordinary",
            )
        )
    hero_excluded_repeatability = pd.concat(hero_ex_parts, ignore_index=True)

    winner_loser_parts = [
        winner_loser_repeatability_table(development, spec.name)
        for spec in FARMING_CANDIDATE_SPECS
    ]
    winner_loser = pd.concat(winner_loser_parts, ignore_index=True)
    patch_stability = patch_stability_table(development)
    patch_repeatability = patch_repeatability_table(development)

    cross_parts: list[pd.DataFrame] = []
    for spec in FARMING_CANDIDATE_SPECS:
        prior_mean, prior_n = prior_history_excluding(
            development, spec.name, "game_version_id"
        )
        cross_parts.append(
            _repeatability_from_prior(
                development,
                spec.name,
                prior_mean,
                prior_n,
                kind="cross_version",
            )
        )
    cross_version_repeatability = pd.concat(cross_parts, ignore_index=True)

    falsification_rows: list[dict[str, object]] = []
    for spec in FARMING_CANDIDATE_SPECS:
        hero_ex_at = hero_excluded_repeatability.loc[
            (hero_excluded_repeatability["candidate"] == spec.name)
            & (hero_excluded_repeatability["kind"] == "hero_excluded")
            & (
                hero_excluded_repeatability["min_prior_appearances"]
                == MIN_PRIOR_APPEARANCES
            )
        ]
        ordinary_at = hero_excluded_repeatability.loc[
            (hero_excluded_repeatability["candidate"] == spec.name)
            & (hero_excluded_repeatability["kind"] == "ordinary")
            & (
                hero_excluded_repeatability["min_prior_appearances"]
                == MIN_PRIOR_APPEARANCES
            )
        ]
        win_at = winner_loser.loc[
            (winner_loser["candidate"] == spec.name)
            & (winner_loser["subset"] == "winners")
            & (winner_loser["min_prior_appearances"] == MIN_PRIOR_APPEARANCES)
        ]
        lose_at = winner_loser.loc[
            (winner_loser["candidate"] == spec.name)
            & (winner_loser["subset"] == "losers")
            & (winner_loser["min_prior_appearances"] == MIN_PRIOR_APPEARANCES)
        ]
        switch_at = team_switcher.loc[
            (team_switcher["candidate"] == spec.name)
            & (team_switcher["min_appearances"] == TEAM_SWITCH_MIN_APPEARANCES)
        ]
        centered_at = within_team_centered.loc[
            (within_team_centered["candidate"] == spec.name)
            & (
                within_team_centered["min_prior_appearances"]
                == MIN_PRIOR_APPEARANCES
            )
        ]
        cross_at = cross_version_repeatability.loc[
            (cross_version_repeatability["candidate"] == spec.name)
            & (
                cross_version_repeatability["min_prior_appearances"]
                == MIN_PRIOR_APPEARANCES
            )
        ]
        falsification_rows.append(
            {
                "candidate": spec.name,
                "ordinary_repeatability_min_prior_10": (
                    float(ordinary_at.iloc[0]["pearson"])
                    if not ordinary_at.empty
                    else float("nan")
                ),
                "hero_excluded_repeatability_min_prior_10": (
                    float(hero_ex_at.iloc[0]["pearson"])
                    if not hero_ex_at.empty
                    else float("nan")
                ),
                "winners_only_repeatability_min_prior_10": (
                    float(win_at.iloc[0]["pearson"])
                    if not win_at.empty
                    else float("nan")
                ),
                "losers_only_repeatability_min_prior_10": (
                    float(lose_at.iloc[0]["pearson"])
                    if not lose_at.empty
                    else float("nan")
                ),
                "team_switch_pearson_min_5": (
                    float(switch_at.iloc[0]["pearson_old_new"])
                    if not switch_at.empty
                    else float("nan")
                ),
                "team_switch_n_players_min_5": (
                    int(switch_at.iloc[0]["n_qualifying_players"])
                    if not switch_at.empty
                    else 0
                ),
                "team_period_centered_repeatability_min_prior_10": (
                    float(centered_at.iloc[0]["pearson_team_period_centered"])
                    if not centered_at.empty
                    else float("nan")
                ),
                "cross_version_repeatability_min_prior_10": (
                    float(cross_at.iloc[0]["pearson"])
                    if not cross_at.empty
                    else float("nan")
                ),
            }
        )
    falsification = pd.DataFrame(falsification_rows)

    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    integrity = {
        "development_end": end.isoformat(),
        "ti2026_used_for_target_definition": False,
        "stratz_called": False,
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "candidate_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in FARMING_CANDIDATE_COLUMN_NAMES
        ),
        "slice12_candidate_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in CANDIDATE_COLUMN_NAMES
        ),
        "candidate_in_all_feature_columns": any(
            name in ALL_FEATURE_COLUMNS for name in FARMING_CANDIDATE_COLUMN_NAMES
        ),
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "elo_implementation": "match_elo_expected_wins",
        "elo_modified": False,
        "missing_position_imputed": False,
        "player_rating_persisted": False,
        "player_farming_state_persisted": False,
        "shrinkage_introduced": False,
        "model_trained": False,
        "candidate_a_is_slice12_baseline": CANDIDATE_A == SLICE12_BASELINE_CANDIDATE,
        "population_matches_expected": (
            int(development["match_id"].nunique()) == FROZEN_DEVELOPMENT_MATCH_COUNT
            and len(development) == EXPECTED_DEVELOPMENT_PLAYER_ROWS
            and n_explicit == EXPECTED_EXPLICIT_POSITION_ROWS
            and n_missing_position == EXPECTED_MISSING_POSITION_ROWS
        ),
    }
    report = Slice13DiagnosticReport(
        development_end=end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        n_missing_position=n_missing_position,
        n_explicit_position=n_explicit,
        coverage=coverage,
        formulas=formulas,
        duration_adjustment=duration_adjustment,
        duration_nonlinearity=duration_nonlinearity,
        residual_distribution=residual_distribution,
        hero_adjustment=hero_adjustment,
        hero_pooling=hero_pooling,
        hero_sample_sizes=hero_sample_sizes,
        candidate_comparison=comparison,
        candidate_position_means=position_means,
        repeatability=repeatability,
        first_half_second_half=halves,
        team_switcher=team_switcher,
        team_switcher_pairs=team_switcher_pairs,
        within_team_centered=within_team_centered,
        hero_excluded_repeatability=hero_excluded_repeatability,
        winner_loser=winner_loser,
        patch_stability=patch_stability,
        patch_repeatability=patch_repeatability,
        cross_version_repeatability=cross_version_repeatability,
        falsification=falsification,
        classification=pd.DataFrame(),
        integrity=integrity,
    )
    classification = classify_slice13(report)
    return Slice13DiagnosticReport(
        **{**report.__dict__, "classification": classification}
    )


def slice13_report_to_jsonable(report: Slice13DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 13 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "n_missing_position": report.n_missing_position,
        "n_explicit_position": report.n_explicit_position,
        "coverage": _jsonable_value(report.coverage),
        "formulas": _jsonable_value(report.formulas),
        "duration_adjustment": _jsonable_value(report.duration_adjustment),
        "duration_nonlinearity": _jsonable_value(report.duration_nonlinearity),
        "residual_distribution": _jsonable_value(report.residual_distribution),
        "hero_adjustment": _jsonable_value(report.hero_adjustment),
        "hero_pooling": _jsonable_value(report.hero_pooling),
        "hero_sample_sizes": _jsonable_value(report.hero_sample_sizes),
        "candidate_comparison": _jsonable_value(report.candidate_comparison),
        "candidate_position_means": _jsonable_value(report.candidate_position_means),
        "repeatability": _jsonable_value(report.repeatability),
        "first_half_second_half": _jsonable_value(report.first_half_second_half),
        "team_switcher": _jsonable_value(report.team_switcher),
        "within_team_centered": _jsonable_value(report.within_team_centered),
        "hero_excluded_repeatability": _jsonable_value(
            report.hero_excluded_repeatability
        ),
        "winner_loser": _jsonable_value(report.winner_loser),
        "patch_stability": _jsonable_value(report.patch_stability),
        "patch_repeatability": _jsonable_value(report.patch_repeatability),
        "cross_version_repeatability": _jsonable_value(
            report.cross_version_repeatability
        ),
        "falsification": _jsonable_value(report.falsification),
        "classification": _jsonable_value(report.classification),
        "integrity": _jsonable_value(report.integrity),
        "candidate_specs": [
            {
                "name": spec.name,
                "family": spec.family,
                "formula": spec.formula,
                "adjustment_variables": list(spec.adjustment_variables),
                "hero_agnostic": spec.hero_agnostic,
                "source_fields": list(spec.source_fields),
            }
            for spec in FARMING_CANDIDATE_SPECS
        ],
    }
