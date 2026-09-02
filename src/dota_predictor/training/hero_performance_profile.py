"""Slice 21: leakage-safe hero resource / combat *profile* diagnostics.

Research only. This module does not persist a hero rating, does not add
production features, does not construct a player×hero fit score, does
not aggregate to team, and does not train a win model. Profile columns
never enter ``FEATURE_COLUMNS``.

Question
--------
Do heroes themselves have stable, leakage-safe resource-demand and
combat-contribution profiles that could later be compared with frozen
player farming B and combat C tendencies?

These are **requirements / roles / tendencies**, not win ratings. A
hero with high historical farm demand is not necessarily strong.

Population
----------
Matches with ``start_time <=`` the frozen Slice 9 development end
(``FROZEN_DEVELOPMENT_END``). Holdout / TI 2026 rows are excluded from
every summary. Box-score values are POST_MATCH observations of
*historical* appearances used to build a profile; they are never
PRE_DRAFT inputs for the current match.

Reuse
-----
Causal farming candidate B (``farming_causal_b``) and causal combat
candidate C (``combat_causal_c``) are the appearance-level observations.
This module groups those existing quantities by hero and by
hero×position. It does not change farming ``k=5``, combat ``k=20``,
candidate B, or candidate C.

Current position
----------------
Historical explicit position labels are used **diagnostically** to test
whether profiles must be position-conditioned. Current post-match
position is not treated as a PRE_DRAFT production input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

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
    COMBAT_C,
    COMBAT_C_POSITION,
    FROZEN_COMBAT_CANDIDATE,
    hero_damage_share,
)
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.player_combat_state import (
    CAUSAL_C_COLUMN,
    FROZEN_COMBAT_SHRINKAGE_K,
    attach_player_combat_state,
)
from dota_predictor.training.player_farming_state import (
    CAUSAL_B_COLUMN,
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    attach_player_farming_state,
)
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    EXPLICIT_POSITION_NUMBERS,
    _jsonable_value,
    _numeric,
    _pearson,
    _spearman,
    build_player_performance_frame,
    explicit_position_mask,
    per_minute,
    position_r_squared,
    restrict_development,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    utc_datetime,
)
from dota_predictor.training.walk_forward import DEFAULT_WALK_FORWARD_CONFIG

__all__ = [
    "CLASSIFICATION_A",
    "CLASSIFICATION_B",
    "CLASSIFICATION_C",
    "COMBAT_C1",
    "COMBAT_C2",
    "COVERAGE_THRESHOLDS",
    "FARMING_F1",
    "FARMING_F2",
    "HERO_COMBAT_PROFILE_KEY",
    "HERO_COMBAT_PROFILE_TARGET",
    "HERO_FARMING_PROFILE_KEY",
    "HERO_FARMING_PROFILE_TARGET",
    "PROFILE_SPECS",
    "ProfileSpec",
    "Slice21DiagnosticReport",
    "all_observation_and_leave_player_out",
    "assign_chronological_blocks",
    "attach_causal_group_mean",
    "attach_hero_profile_observations",
    "classify_slice21",
    "coverage_threshold_table",
    "flex_hero_table",
    "group_split_half",
    "group_variance_decomposition",
    "player_concentration_table",
    "player_demeaned_values",
    "position_usage_table",
    "run_hero_performance_profile_diagnostics",
    "slice21_report_to_jsonable",
]


FARMING_F1 = "F1"
FARMING_F2 = "F2"
COMBAT_C1 = "C1"
COMBAT_C2 = "C2"
KEY_HERO = "hero_id"
KEY_HERO_POSITION = "hero_id × position"
COVERAGE_THRESHOLDS: tuple[int, ...] = (5, 10, 20, 50, 100)
MIN_HALF_HERO = 10
MIN_HALF_HERO_POSITION = 5
MIN_VARIANCE_GROUP_N = 10
MIN_VARIANCE_GROUPS = 10
MIN_PATCH_PROFILE_N = 20
MIN_BLOCK_PROFILE_N = 10
MATERIAL_FARMING_SHIFT = 0.50
MATERIAL_COMBAT_SHIFT = 0.05
SPECIALIST_TOP_SHARE = 0.50
FLEX_DOMINANT_MAX = 0.70
_REPEATABILITY_FLOOR = 0.30
_REPEATABILITY_CLEAR = 0.50
_LPO_CORR_FLOOR = 0.90
_PATCH_CORR_SOFT = 0.70
_POSITION_R2_MATERIAL = 0.20
PLAYER_X_HERO_FIT_NAMES: tuple[str, ...] = (
    "player_hero_fit",
    "farming_fit",
    "combat_fit",
    "player_hero_farming_fit",
    "player_hero_combat_fit",
    "mean_farming_fit_diff",
    "mean_combat_fit_diff",
)

CLASSIFICATION_A = (
    "A — freeze stable hero farming/combat profile targets for "
    "historical-state construction"
)
CLASSIFICATION_B = (
    "B — hero profiles are promising but position/patch semantics "
    "need one more diagnostic slice"
)
CLASSIFICATION_C = (
    "C — landed data does not support sufficiently stable hero resource/combat profiles"
)

# Frozen after Slice 21 development diagnostics. Shrinkage, recency,
# patch weighting, and player×hero fit are later slices.
HERO_FARMING_PROFILE_TARGET = CAUSAL_B_COLUMN
HERO_FARMING_PROFILE_KEY = KEY_HERO_POSITION
HERO_COMBAT_PROFILE_TARGET = CAUSAL_C_COLUMN
HERO_COMBAT_PROFILE_KEY = KEY_HERO_POSITION


@dataclass(frozen=True)
class ProfileSpec:
    """One hero-profile representation. Not a production feature."""

    name: str
    dimension: str
    observation_column: str
    key: str
    group_columns: tuple[str, ...]
    formula: str


PROFILE_SPECS: tuple[ProfileSpec, ...] = (
    ProfileSpec(
        name=FARMING_F1,
        dimension="farming",
        observation_column=CAUSAL_B_COLUMN,
        key=KEY_HERO,
        group_columns=("hero_id",),
        formula=(
            "mean(farming_causal_b | hero_id); "
            "B = causal z(LHPM residual ~ position + duration)"
        ),
    ),
    ProfileSpec(
        name=FARMING_F2,
        dimension="farming",
        observation_column=CAUSAL_B_COLUMN,
        key=KEY_HERO_POSITION,
        group_columns=("hero_id", "position_number"),
        formula=(
            "mean(farming_causal_b | hero_id, explicit position 1–5); "
            "B is already position+duration adjusted"
        ),
    ),
    ProfileSpec(
        name=COMBAT_C1,
        dimension="combat",
        observation_column=CAUSAL_C_COLUMN,
        key=KEY_HERO,
        group_columns=("hero_id",),
        formula=(
            "mean(combat_causal_c | hero_id); "
            "C = causal (hero_damage_share - prior position mean)"
        ),
    ),
    ProfileSpec(
        name=COMBAT_C2,
        dimension="combat",
        observation_column=CAUSAL_C_COLUMN,
        key=KEY_HERO_POSITION,
        group_columns=("hero_id", "position_number"),
        formula=(
            "mean(combat_causal_c | hero_id, explicit position 1–5); "
            "C is already position-adjusted damage share"
        ),
    ),
)
_SPEC_BY_NAME: dict[str, ProfileSpec] = {spec.name: spec for spec in PROFILE_SPECS}


@dataclass(frozen=True)
class Slice21DiagnosticReport:
    development_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    n_missing_position: int
    n_explicit_position: int
    formulas: pd.DataFrame
    coverage: pd.DataFrame
    coverage_thresholds: pd.DataFrame
    position_usage: pd.DataFrame
    position_dependence: pd.DataFrame
    flex_heroes: pd.DataFrame
    circularity: pd.DataFrame
    player_concentration: pd.DataFrame
    player_demean: pd.DataFrame
    split_half: pd.DataFrame
    temporal_blocks: pd.DataFrame
    adjacent_block_stability: pd.DataFrame
    variance_decomposition: pd.DataFrame
    shrinkage_diagnosis: pd.DataFrame
    patch_stability: pd.DataFrame
    adjacent_patch_stability: pd.DataFrame
    same_hero_position_patch: pd.DataFrame
    player_state_relationship: pd.DataFrame
    cross_dimension: pd.DataFrame
    farming_comparison: pd.DataFrame
    combat_comparison: pd.DataFrame
    classification: pd.DataFrame
    integrity: dict[str, object]


def attach_hero_profile_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Add frozen causal B/C observations and player states.

    Does not rewrite candidate B, candidate C, farming ``k``, or combat
    ``k``. Player state is attached only so Slice 21 can correlate hero
    profiles with already-frozen player tendencies.
    """
    if CAUSAL_B_COLUMN in frame.columns:
        out = frame.copy()
    else:
        out = attach_player_farming_state(frame, k=FROZEN_SHRINKAGE_K)
    if CAUSAL_C_COLUMN not in out.columns or "combat_shrunk_c" not in out.columns:
        out = attach_player_combat_state(out, k=FROZEN_COMBAT_SHRINKAGE_K)
    if "farming_shrunk_b" not in out.columns:
        out = attach_player_farming_state(out, k=FROZEN_SHRINKAGE_K)
    if "last_hits_per_minute" not in out.columns:
        out["last_hits_per_minute"] = per_minute(
            out["num_last_hits"], out["duration_seconds"]
        )
    if COMBAT_C not in out.columns:
        out[COMBAT_C] = hero_damage_share(out)
    return out


def _group_key_series(frame: pd.DataFrame, group_columns: tuple[str, ...]) -> pd.Series:
    if group_columns == ("hero_id",):
        hero = _numeric(frame["hero_id"])
        keys = pd.Series(pd.NA, index=frame.index, dtype="object")
        valid = hero.notna()
        keys.loc[valid] = [
            (int(value),) for value in hero.loc[valid].to_numpy(dtype=float)
        ]
        return keys
    if group_columns == ("hero_id", "position_number"):
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
    raise ValueError(f"unsupported group columns: {group_columns}")


def attach_causal_group_mean(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: tuple[str, ...],
    out_column: str,
    n_column: str | None = None,
    leave_player_out: bool = False,
) -> pd.DataFrame:
    """Strictly prior group mean of ``value_column``.

    History is ``start_time < T``. Same-timestamp rows are mutually
    blind. Non-finite values do not enter the mean. When
    ``leave_player_out`` is true, the current player's own earlier
    appearances on the same key are also excluded.
    """
    out = frame.copy()
    values = _numeric(out[value_column])
    keys = _group_key_series(out, group_columns)
    means = pd.Series(np.nan, index=out.index, dtype=float)
    counts = pd.Series(0, index=out.index, dtype=int)
    eligible = values.notna() & keys.notna()
    n_name = n_column if n_column is not None else f"{out_column}_n"
    if not bool(eligible.any()):
        out[out_column] = means
        out[n_name] = counts
        return out

    times = pd.to_datetime(out["start_time"], utc=True).to_numpy()
    players = _numeric(out["player_id"]).to_numpy(dtype=float)
    eligible_idx = np.flatnonzero(eligible.to_numpy())
    order = np.argsort(times[eligible_idx], kind="mergesort")
    sorted_idx = eligible_idx[order]
    sorted_times = times[sorted_idx]
    sorted_values = values.to_numpy(dtype=float)[sorted_idx]
    sorted_keys = keys.to_numpy()[sorted_idx]
    sorted_players = players[sorted_idx]
    cuts = np.r_[True, sorted_times[1:] != sorted_times[:-1]]
    starts = np.flatnonzero(cuts)
    bounds = np.r_[starts, len(sorted_idx)]
    group_sum: dict[tuple[int, ...], float] = {}
    group_n: dict[tuple[int, ...], int] = {}
    player_sum: dict[tuple[object, ...], float] = {}
    player_n: dict[tuple[object, ...], int] = {}
    mean_vals = np.full(len(out), np.nan, dtype=float)
    n_vals = np.zeros(len(out), dtype=int)
    for i in range(len(starts)):
        lo = int(bounds[i])
        hi = int(bounds[i + 1])
        for j in range(lo, hi):
            key = sorted_keys[j]
            n_prior = int(group_n.get(key, 0))
            total = float(group_sum.get(key, 0.0))
            if leave_player_out:
                player_key = (key, int(sorted_players[j]))
                n_prior -= int(player_n.get(player_key, 0))
                total -= float(player_sum.get(player_key, 0.0))
            row_pos = int(sorted_idx[j])
            n_vals[row_pos] = n_prior
            if n_prior > 0:
                mean_vals[row_pos] = total / n_prior
        for j in range(lo, hi):
            key = sorted_keys[j]
            value = float(sorted_values[j])
            group_sum[key] = float(group_sum.get(key, 0.0)) + value
            group_n[key] = int(group_n.get(key, 0)) + 1
            if leave_player_out:
                player_key = (key, int(sorted_players[j]))
                player_sum[player_key] = float(player_sum.get(player_key, 0.0)) + value
                player_n[player_key] = int(player_n.get(player_key, 0)) + 1
    out[out_column] = mean_vals
    out[n_name] = n_vals
    return out


def all_observation_and_leave_player_out(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Full-development all-observation mean and leave-player-out mean.

    Diagnostic only. Not a causal production state.
    """
    values = _numeric(frame[value_column])
    keys = _group_key_series(frame, group_columns)
    work = pd.DataFrame(
        {
            "key": keys,
            "player_id": frame["player_id"].to_numpy(),
            "value": values,
        },
        index=frame.index,
    )
    eligible = work["key"].notna() & work["value"].notna()
    work = work.loc[eligible]
    empty = pd.DataFrame(
        {
            "all_mean": pd.Series(dtype=float),
            "lpo_mean": pd.Series(dtype=float),
            "group_n": pd.Series(dtype=int),
            "player_n": pd.Series(dtype=int),
        }
    )
    if work.empty:
        return empty
    group_sum = work.groupby("key", sort=False)["value"].transform("sum")
    group_n = work.groupby("key", sort=False)["value"].transform("size")
    player_sum = work.groupby(["key", "player_id"], sort=False)["value"].transform(
        "sum"
    )
    player_n = work.groupby(["key", "player_id"], sort=False)["value"].transform("size")
    other_n = group_n - player_n
    lpo = (group_sum - player_sum) / other_n
    out = pd.DataFrame(
        {
            "all_mean": group_sum / group_n,
            "lpo_mean": lpo.where(other_n > 0),
            "group_n": group_n.astype(int),
            "player_n": player_n.astype(int),
        },
        index=work.index,
    )
    return out


def player_demeaned_values(frame: pd.DataFrame, value_column: str) -> pd.Series:
    """Appearance minus the player's other-match mean of ``value_column``."""
    values = _numeric(frame[value_column])
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    work = pd.DataFrame(
        {"player_id": frame["player_id"].to_numpy(), "value": values},
        index=frame.index,
    ).dropna()
    if work.empty:
        return out
    player_sum = work.groupby("player_id")["value"].transform("sum")
    player_n = work.groupby("player_id")["value"].transform("size")
    other_n = player_n - 1
    player_loo = (player_sum - work["value"]) / other_n
    demeaned = work["value"] - player_loo.where(other_n > 0)
    out.loc[work.index] = demeaned
    return out


def player_concentration_table(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    value_column: str,
) -> pd.DataFrame:
    values = _numeric(frame[value_column])
    keys = _group_key_series(frame, group_columns)
    work = pd.DataFrame(
        {
            "key": keys,
            "player_id": frame["player_id"].to_numpy(),
            "match_id": frame["match_id"].to_numpy(),
            "value": values,
        }
    )
    work = work.loc[work["key"].notna() & work["value"].notna()]
    if work.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, group in work.groupby("key", sort=False):
        n = len(group)
        player_counts = group.groupby("player_id").size().sort_values(ascending=False)
        top1 = float(player_counts.iloc[0] / n) if len(player_counts) else float("nan")
        top3_n = int(player_counts.head(3).sum()) if len(player_counts) else 0
        row: dict[str, object] = {
            "hero_id": int(key[0]),
            "n_observations": n,
            "unique_players": int(player_counts.size),
            "unique_matches": int(group["match_id"].nunique()),
            "top_player_share": top1,
            "top_3_player_share": top3_n / n if n else float("nan"),
            "specialist_flag": bool(np.isfinite(top1) and top1 >= SPECIALIST_TOP_SHARE),
        }
        if len(group_columns) == 2:
            row["position_number"] = int(key[1])
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_threshold_table(counts: pd.Series, *, label: str) -> pd.DataFrame:
    values = pd.to_numeric(counts, errors="coerce").dropna().to_numpy(dtype=float)
    row: dict[str, object] = {
        "unit": label,
        "n_units": int(values.size),
        "median_n": float(np.median(values)) if values.size else float("nan"),
        "mean_n": float(values.mean()) if values.size else float("nan"),
        "min_n": float(values.min()) if values.size else float("nan"),
        "max_n": float(values.max()) if values.size else float("nan"),
        "p10_n": float(np.quantile(values, 0.10)) if values.size else float("nan"),
        "p90_n": float(np.quantile(values, 0.90)) if values.size else float("nan"),
    }
    for threshold in COVERAGE_THRESHOLDS:
        row[f"n_ge_{threshold}"] = int((values >= threshold).sum())
        row[f"share_ge_{threshold}"] = (
            float((values >= threshold).mean()) if values.size else float("nan")
        )
    return pd.DataFrame([row])


def position_usage_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = explicit_position_mask(frame)
    work = frame.loc[eligible, ["hero_id", "position_number", "match_id"]].copy()
    work["hero_id"] = _numeric(work["hero_id"])
    work["position_number"] = _numeric(work["position_number"])
    if work.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for hero_id, group in work.groupby("hero_id", sort=False):
        n = len(group)
        pos_counts = (
            group.groupby("position_number")
            .size()
            .reindex(list(EXPLICIT_POSITION_NUMBERS), fill_value=0)
        )
        shares = pos_counts.to_numpy(dtype=float) / n if n else np.zeros(5)
        used = int((pos_counts > 0).sum())
        dominant = float(shares.max()) if n else float("nan")
        entropy = _entropy(shares)
        row: dict[str, object] = {
            "hero_id": int(hero_id),
            "n_observations": n,
            "n_unique_matches": int(group["match_id"].nunique()),
            "n_unique_positions": used,
            "dominant_position": int(pos_counts.idxmax()) if n else None,
            "dominant_position_share": dominant,
            "position_entropy": entropy,
            "flex_flag": bool(np.isfinite(dominant) and dominant < FLEX_DOMINANT_MAX),
        }
        for number in EXPLICIT_POSITION_NUMBERS:
            row[f"pos{number}_n"] = int(pos_counts.loc[number])
            row[f"pos{number}_share"] = float(shares[number - 1])
        rows.append(row)
    return pd.DataFrame(rows)


def _entropy(shares: np.ndarray) -> float:
    positive = shares[shares > 0.0]
    if positive.size == 0:
        return float("nan")
    return float(-np.sum(positive * np.log(positive)))


def flex_hero_table(usage: pd.DataFrame, *, n: int = 12) -> pd.DataFrame:
    if usage.empty:
        return usage
    ranked = usage.sort_values(
        ["flex_flag", "position_entropy", "n_observations"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return ranked.head(n).reset_index(drop=True)


def group_split_half(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: tuple[str, ...],
    min_each: int,
) -> dict[str, object]:
    values = _numeric(frame[value_column])
    keys = _group_key_series(frame, group_columns)
    work = pd.DataFrame(
        {
            "key": keys,
            "start_time": pd.to_datetime(frame["start_time"], utc=True),
            "value": values,
        }
    )
    work = work.loc[work["key"].notna() & work["value"].notna()]
    early: list[float] = []
    late: list[float] = []
    n_paired = 0
    for _key, group in work.groupby("key", sort=False):
        ordered = group.sort_values("start_time", kind="mergesort")
        size = len(ordered)
        split = size // 2
        if split < min_each or (size - split) < min_each:
            continue
        early.append(float(ordered.iloc[:split]["value"].mean()))
        late.append(float(ordered.iloc[split:]["value"].mean()))
        n_paired += 1
    return {
        "n_profiles": n_paired,
        "min_observations_per_half": min_each,
        "pearson": _pearson(pd.Series(early), pd.Series(late))
        if n_paired
        else float("nan"),
        "spearman": _spearman(pd.Series(early), pd.Series(late))
        if n_paired
        else float("nan"),
    }


def group_variance_decomposition(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: tuple[str, ...],
    min_group_n: int = MIN_VARIANCE_GROUP_N,
) -> dict[str, object]:
    values = _numeric(frame[value_column])
    keys = _group_key_series(frame, group_columns)
    work = pd.DataFrame({"key": keys, "value": values})
    work = work.loc[work["key"].notna() & work["value"].notna()]
    empty = {
        "n_profiles": 0,
        "n_observations": 0,
        "min_group_n": min_group_n,
        "within_variance": float("nan"),
        "between_variance": float("nan"),
        "icc": float("nan"),
        "within_over_between": float("nan"),
    }
    if work.empty:
        return empty
    stats = work.groupby("key")["value"].agg(n="size", mean="mean", var="var")
    eligible = stats.loc[stats["n"] >= min_group_n]
    n_profiles = len(eligible)
    n_obs = int(eligible["n"].sum()) if n_profiles else 0
    if n_profiles < MIN_VARIANCE_GROUPS:
        empty["n_profiles"] = n_profiles
        empty["n_observations"] = n_obs
        return empty
    within = float(eligible["var"].mean())
    sampling = float((eligible["var"] / eligible["n"]).mean())
    between_raw = float(eligible["mean"].var(ddof=1))
    between = max(0.0, between_raw - sampling)
    icc = (
        float("nan")
        if (between + within) <= 0.0
        else float(between / (between + within))
    )
    ratio = (
        float("nan")
        if between <= 0.0 or not np.isfinite(between)
        else float(within / between)
    )
    return {
        "n_profiles": n_profiles,
        "n_observations": n_obs,
        "min_group_n": min_group_n,
        "within_variance": within,
        "between_variance": between,
        "icc": icc,
        "within_over_between": ratio,
    }


def assign_chronological_blocks(
    frame: pd.DataFrame, *, n_blocks: int | None = None
) -> pd.Series:
    """Map each row to a chronological match-count block (walk-forward style)."""
    blocks_n = (
        n_blocks if n_blocks is not None else DEFAULT_WALK_FORWARD_CONFIG.n_blocks
    )
    matches = (
        frame.loc[:, ["match_id", "start_time"]].drop_duplicates("match_id").copy()
    )
    matches["start_time"] = pd.to_datetime(matches["start_time"], utc=True)
    matches = matches.sort_values(["start_time", "match_id"], kind="mergesort")
    n = len(matches)
    if n == 0:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    order = np.arange(n)
    block = np.minimum(blocks_n, 1 + (order * blocks_n) // max(n, 1))
    mapped = pd.Series(block, index=matches["match_id"].to_numpy())
    return frame["match_id"].map(mapped)


def _adjacent_window_stability(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: tuple[str, ...],
    window_column: str,
    min_n: int,
    material_shift: float,
) -> pd.DataFrame:
    values = _numeric(frame[value_column])
    keys = _group_key_series(frame, group_columns)
    work = pd.DataFrame(
        {
            "key": keys,
            "window": frame[window_column],
            "value": values,
        }
    )
    work = work.loc[
        work["key"].notna() & work["value"].notna() & work["window"].notna()
    ]
    if work.empty:
        return pd.DataFrame()
    summary = (
        work.groupby(["window", "key"], sort=False)["value"]
        .agg(n="size", mean="mean")
        .reset_index()
    )
    windows = sorted(summary["window"].dropna().unique().tolist())
    rows: list[dict[str, object]] = []
    for left, right in pairwise(windows):
        a = summary.loc[(summary["window"] == left) & (summary["n"] >= min_n)]
        b = summary.loc[(summary["window"] == right) & (summary["n"] >= min_n)]
        joined = a.merge(b, on="key", suffixes=("_left", "_right"))
        n_profiles = len(joined)
        abs_delta = (
            (joined["mean_right"] - joined["mean_left"]).abs()
            if n_profiles
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "left_window": left,
                "right_window": right,
                "n_profiles": n_profiles,
                "min_n": min_n,
                "pearson": (
                    _pearson(joined["mean_left"], joined["mean_right"])
                    if n_profiles
                    else float("nan")
                ),
                "spearman": (
                    _spearman(joined["mean_left"], joined["mean_right"])
                    if n_profiles
                    else float("nan")
                ),
                "mean_abs_shift": (
                    float(abs_delta.mean()) if n_profiles else float("nan")
                ),
                "median_abs_shift": (
                    float(abs_delta.median()) if n_profiles else float("nan")
                ),
                "fraction_material_shift": (
                    float((abs_delta >= material_shift).mean())
                    if n_profiles
                    else float("nan")
                ),
                "material_shift_threshold": material_shift,
            }
        )
    return pd.DataFrame(rows)


def _circularity_row(frame: pd.DataFrame, spec: ProfileSpec) -> dict[str, object]:
    compared = all_observation_and_leave_player_out(
        frame,
        value_column=spec.observation_column,
        group_columns=spec.group_columns,
    )
    paired = compared.dropna(subset=["all_mean", "lpo_mean"])
    abs_delta = (paired["all_mean"] - paired["lpo_mean"]).abs()
    concentration = player_concentration_table(
        frame, group_columns=spec.group_columns, value_column=spec.observation_column
    )
    return {
        "representation": spec.name,
        "dimension": spec.dimension,
        "key": spec.key,
        "n_appearances": len(paired),
        "n_profiles": int(concentration["n_observations"].size)
        if not concentration.empty
        else 0,
        "pearson_all_vs_lpo": _pearson(paired["all_mean"], paired["lpo_mean"])
        if len(paired)
        else float("nan"),
        "spearman_all_vs_lpo": _spearman(paired["all_mean"], paired["lpo_mean"])
        if len(paired)
        else float("nan"),
        "mean_abs_difference": (
            float(abs_delta.mean()) if len(paired) else float("nan")
        ),
        "median_abs_difference": (
            float(abs_delta.median()) if len(paired) else float("nan")
        ),
        "median_top_player_share": (
            float(concentration["top_player_share"].median())
            if not concentration.empty
            else float("nan")
        ),
        "median_top_3_share": (
            float(concentration["top_3_player_share"].median())
            if not concentration.empty
            else float("nan")
        ),
        "median_unique_players": (
            float(concentration["unique_players"].median())
            if not concentration.empty
            else float("nan")
        ),
        "fraction_specialist": (
            float(concentration["specialist_flag"].mean())
            if not concentration.empty
            else float("nan")
        ),
    }


def _player_demean_row(frame: pd.DataFrame, spec: ProfileSpec) -> dict[str, object]:
    values = _numeric(frame[spec.observation_column])
    demeaned = player_demeaned_values(frame, spec.observation_column)
    work = frame.assign(_raw=values, _demeaned=demeaned)
    keys = _group_key_series(work, spec.group_columns)
    work = work.assign(_key=keys)
    eligible = work["_key"].notna() & work["_raw"].notna() & work["_demeaned"].notna()
    grouped = (
        work.loc[eligible]
        .groupby("_key", sort=False)
        .agg(
            raw_mean=("_raw", "mean"),
            demeaned_mean=("_demeaned", "mean"),
            n=("_raw", "size"),
        )
    )
    grouped = grouped.loc[grouped["n"] >= MIN_VARIANCE_GROUP_N]
    raw_icc = group_variance_decomposition(
        work.loc[eligible],
        value_column="_raw",
        group_columns=spec.group_columns,
    )
    demeaned_frame = work.loc[eligible].copy()
    demeaned_icc = group_variance_decomposition(
        demeaned_frame,
        value_column="_demeaned",
        group_columns=spec.group_columns,
    )
    return {
        "representation": spec.name,
        "dimension": spec.dimension,
        "key": spec.key,
        "n_profiles": len(grouped),
        "pearson_raw_vs_player_demeaned_means": (
            _pearson(grouped["raw_mean"], grouped["demeaned_mean"])
            if len(grouped)
            else float("nan")
        ),
        "icc_raw": raw_icc["icc"],
        "icc_player_demeaned": demeaned_icc["icc"],
        "between_raw": raw_icc["between_variance"],
        "between_player_demeaned": demeaned_icc["between_variance"],
    }


def _coverage_for_spec(frame: pd.DataFrame, spec: ProfileSpec) -> dict[str, object]:
    values = _numeric(frame[spec.observation_column])
    keys = _group_key_series(frame, spec.group_columns)
    eligible = values.notna() & keys.notna()
    n_rows = int(eligible.sum())
    counts = keys.loc[eligible].value_counts() if n_rows else pd.Series(dtype=int)
    return {
        "representation": spec.name,
        "dimension": spec.dimension,
        "key": spec.key,
        "n_eligible_rows": n_rows,
        "n_profiles": int(counts.size),
        "median_observations_per_profile": (
            float(counts.median()) if len(counts) else float("nan")
        ),
        "mean_observations_per_profile": (
            float(counts.mean()) if len(counts) else float("nan")
        ),
        **{
            f"n_ge_{threshold}": int((counts >= threshold).sum())
            for threshold in COVERAGE_THRESHOLDS
        },
    }


def _raw_demand_column(spec: ProfileSpec) -> str:
    """Unadjusted demand quantity used to test position conditioning.

    Frozen B/C already remove position, so they cannot answer whether a
    hero's *resource/combat requirement* differs by role. LHPM and
    own-team damage share can.
    """
    if spec.dimension == "farming":
        return "last_hits_per_minute"
    return COMBAT_C


def _within_hero_position_range(
    frame: pd.DataFrame, value_column: str
) -> tuple[int, float]:
    values = _numeric(frame[value_column])
    keys = _group_key_series(frame, ("hero_id", "position_number"))
    eligible = values.notna() & keys.notna()
    if not bool(eligible.any()):
        return 0, float("nan")
    pos_means = (
        pd.DataFrame({"key": keys.loc[eligible], "value": values.loc[eligible]})
        .groupby("key")["value"]
        .mean()
        .reset_index()
    )
    pos_means["hero_id"] = [key[0] for key in pos_means["key"]]
    spread = pos_means.groupby("hero_id")["value"].agg(["min", "max", "count"])
    multi = spread.loc[spread["count"] >= 2]
    if multi.empty:
        return 0, float("nan")
    return len(multi), float((multi["max"] - multi["min"]).mean())


def _position_dependence_row(
    frame: pd.DataFrame, spec: ProfileSpec, usage: pd.DataFrame
) -> dict[str, object]:
    raw_column = _raw_demand_column(spec)
    obs_r2 = position_r_squared(
        frame.assign(_v=_numeric(frame[spec.observation_column])), "_v"
    )
    raw_r2 = position_r_squared(frame.assign(_v=_numeric(frame[raw_column])), "_v")
    n_multi, mean_range = _within_hero_position_range(frame, raw_column)
    n_flex = int(usage["flex_flag"].sum()) if not usage.empty else 0
    return {
        "representation": spec.name,
        "dimension": spec.dimension,
        "raw_demand_column": raw_column,
        "observation_position_r2": obs_r2,
        "raw_position_r2": raw_r2,
        "n_heroes_with_2plus_positions": n_multi,
        "mean_within_hero_position_range": mean_range,
        "n_flex_heroes": n_flex,
        "median_dominant_position_share": (
            float(usage["dominant_position_share"].median())
            if not usage.empty
            else float("nan")
        ),
        "current_position_used_diagnostically": True,
        "current_position_treated_as_pre_draft": False,
    }


def _player_state_relationship(
    frame: pd.DataFrame, spec: ProfileSpec
) -> dict[str, object]:
    profile_col = f"{spec.name.lower()}_causal_mean"
    player_col = (
        "farming_shrunk_b" if spec.dimension == "farming" else "combat_shrunk_c"
    )
    if profile_col not in frame.columns or player_col not in frame.columns:
        return {
            "representation": spec.name,
            "dimension": spec.dimension,
            "n": 0,
            "pearson_overall": float("nan"),
            "spearman_overall": float("nan"),
        }
    left = _numeric(frame[player_col])
    right = _numeric(frame[profile_col])
    mask = left.notna() & right.notna()
    row: dict[str, object] = {
        "representation": spec.name,
        "dimension": spec.dimension,
        "n": int(mask.sum()),
        "pearson_overall": _pearson(left, right),
        "spearman_overall": _spearman(left, right),
    }
    eligible = explicit_position_mask(frame) & mask
    for number in EXPLICIT_POSITION_NUMBERS:
        subset = eligible & (frame["position_number"] == number)
        row[f"pos{number}_n"] = int(subset.sum())
        row[f"pos{number}_pearson"] = _pearson(left.loc[subset], right.loc[subset])
    return row


def _cross_dimension_row(
    frame: pd.DataFrame,
    farming_spec: ProfileSpec,
    combat_spec: ProfileSpec,
    *,
    min_n: int,
) -> dict[str, object]:
    farm_values = _numeric(frame[farming_spec.observation_column])
    combat_values = _numeric(frame[combat_spec.observation_column])
    keys = _group_key_series(frame, farming_spec.group_columns)
    work = pd.DataFrame({"key": keys, "farming": farm_values, "combat": combat_values})
    work = work.loc[
        work["key"].notna() & work["farming"].notna() & work["combat"].notna()
    ]
    grouped = work.groupby("key").agg(
        n=("farming", "size"),
        farming_mean=("farming", "mean"),
        combat_mean=("combat", "mean"),
    )
    grouped = grouped.loc[grouped["n"] >= min_n]
    return {
        "farming_representation": farming_spec.name,
        "combat_representation": combat_spec.name,
        "key": farming_spec.key,
        "min_n": min_n,
        "n_profiles": len(grouped),
        "pearson": (
            _pearson(grouped["farming_mean"], grouped["combat_mean"])
            if len(grouped)
            else float("nan")
        ),
        "spearman": (
            _spearman(grouped["farming_mean"], grouped["combat_mean"])
            if len(grouped)
            else float("nan")
        ),
    }


def _comparison_row(
    spec: ProfileSpec,
    *,
    coverage: pd.DataFrame,
    position: pd.DataFrame,
    split_half: pd.DataFrame,
    patch: pd.DataFrame,
    variance: pd.DataFrame,
    circularity: pd.DataFrame,
    caveat: str,
) -> dict[str, object]:
    cov = coverage.loc[coverage["representation"] == spec.name]
    pos = position.loc[position["representation"] == spec.name]
    half = split_half.loc[split_half["representation"] == spec.name]
    var = variance.loc[variance["representation"] == spec.name]
    circ = circularity.loc[circularity["representation"] == spec.name]
    if patch.empty or "representation" not in patch.columns:
        patch_rows = pd.DataFrame()
    else:
        patch_rows = patch.loc[patch["representation"] == spec.name]
    patch_pearson = (
        float(patch_rows["pearson"].median()) if not patch_rows.empty else float("nan")
    )

    def _val(table: pd.DataFrame, column: str) -> object:
        if table.empty or column not in table.columns:
            return float("nan")
        value = table.iloc[0][column]
        return value

    return {
        "representation": spec.name,
        "coverage_eligible_rows": _val(cov, "n_eligible_rows"),
        "n_profiles": _val(cov, "n_profiles"),
        "median_observations_per_profile": _val(cov, "median_observations_per_profile"),
        "position_dependence_r2": _val(pos, "raw_position_r2"),
        "observation_position_r2": _val(pos, "observation_position_r2"),
        "mean_within_hero_position_range": _val(pos, "mean_within_hero_position_range"),
        "split_half_pearson": _val(half, "pearson"),
        "split_half_n_profiles": _val(half, "n_profiles"),
        "patch_stability_median_pearson": patch_pearson,
        "between_within_icc": _val(var, "icc"),
        "player_concentration_median_top_share": _val(circ, "median_top_player_share"),
        "lpo_pearson": _val(circ, "pearson_all_vs_lpo"),
        "major_caveat": caveat,
    }


def _caveat_for(
    spec: ProfileSpec,
    *,
    split_half: float,
    patch_r: float,
    lpo_r: float,
    specialist_frac: float,
    pos_r2: float,
    coverage_ge20: int,
) -> str:
    notes: list[str] = []
    if spec.key == KEY_HERO:
        notes.append("hero-only mixes roles; diagnostic current position is POST_MATCH")
    else:
        notes.append(
            "hero×position uses diagnostic current position, not a PRE_DRAFT fact"
        )
    if np.isfinite(pos_r2) and pos_r2 >= _POSITION_R2_MATERIAL and spec.key == KEY_HERO:
        notes.append(f"raw demand still position-structured R²={pos_r2:.3f}")
    if np.isfinite(split_half) and split_half < _REPEATABILITY_FLOOR:
        notes.append("weak split-half repeatability")
    if np.isfinite(patch_r) and patch_r < _PATCH_CORR_SOFT:
        notes.append(f"patch-unstable adjacent r={patch_r:.3f}")
    if np.isfinite(lpo_r) and lpo_r < _LPO_CORR_FLOOR:
        notes.append("leave-player-out profile diverges (specialist mix)")
    if np.isfinite(specialist_frac) and specialist_frac >= 0.20:
        notes.append(f"{specialist_frac:.0%} of profiles specialist-dominated")
    if coverage_ge20 < 20:
        notes.append("sparse coverage at n>=20")
    notes.append("requirement/role profile, not hero strength")
    return "; ".join(notes)


def classify_slice21(report: Slice21DiagnosticReport) -> pd.DataFrame:
    """Map Slice 21 tables onto the A / B / C hero-profile gate.

    Choice of hero-only vs hero×position is not based on win metrics.
    """
    farming = report.farming_comparison
    combat = report.combat_comparison
    empty = pd.DataFrame(
        [
            {
                "classification": "C",
                "gate": CLASSIFICATION_C,
                "farming_key": None,
                "combat_key": None,
                "farming_target": None,
                "combat_target": None,
                "rationale": "No comparison rows were produced.",
                "next_slice": "Do not freeze a hero profile.",
            }
        ]
    )
    if farming.empty or combat.empty:
        return empty

    def _pick(
        table: pd.DataFrame,
        hero: str,
        hero_pos: str,
        *,
        material_range: float,
    ) -> tuple[str, str, str]:
        f1 = table.loc[table["representation"] == hero]
        f2 = table.loc[table["representation"] == hero_pos]
        if f1.empty or f2.empty:
            return hero_pos, "missing comparison row", "unresolved"
        r1 = float(f1.iloc[0]["split_half_pearson"])
        r2 = float(f2.iloc[0]["split_half_pearson"])
        n1 = int(f1.iloc[0]["n_profiles"])
        n2 = int(f2.iloc[0]["n_profiles"])
        pos_r2 = float(f1.iloc[0]["position_dependence_r2"])
        pos_range = float(f1.iloc[0]["mean_within_hero_position_range"])
        coverage_ok = n2 >= max(20, int(0.5 * n1)) if n1 else n2 >= 20
        position_matters = (
            np.isfinite(pos_r2) and pos_r2 >= _POSITION_R2_MATERIAL
        ) or (np.isfinite(pos_range) and pos_range >= material_range)
        more_stable = np.isfinite(r2) and (not np.isfinite(r1) or r2 >= r1 - 0.05)
        if position_matters and coverage_ok and more_stable:
            return KEY_HERO_POSITION, "position conditions the profile", "chosen"
        if (not position_matters) and np.isfinite(r1) and r1 >= _REPEATABILITY_CLEAR:
            return KEY_HERO, "hero identity dominates position", "chosen"
        if position_matters and not coverage_ok:
            return (
                KEY_HERO_POSITION,
                "position matters but cells are sparse",
                "unresolved",
            )
        return KEY_HERO_POSITION, "default to position-conditioned demand", "chosen"

    farm_key, farm_why, farm_status = _pick(
        farming, FARMING_F1, FARMING_F2, material_range=2.0
    )
    combat_key, combat_why, combat_status = _pick(
        combat, COMBAT_C1, COMBAT_C2, material_range=MATERIAL_COMBAT_SHIFT
    )

    def _metric(table: pd.DataFrame, name: str, column: str) -> float:
        rows = table.loc[table["representation"] == name]
        if rows.empty:
            return float("nan")
        return float(rows.iloc[0][column])

    farm_rep = FARMING_F2 if farm_key == KEY_HERO_POSITION else FARMING_F1
    combat_rep = COMBAT_C2 if combat_key == KEY_HERO_POSITION else COMBAT_C1
    farm_half = _metric(farming, farm_rep, "split_half_pearson")
    combat_half = _metric(combat, combat_rep, "split_half_pearson")
    farm_lpo = _metric(farming, farm_rep, "lpo_pearson")
    combat_lpo = _metric(combat, combat_rep, "lpo_pearson")
    farm_patch = _metric(farming, farm_rep, "patch_stability_median_pearson")
    combat_patch = _metric(combat, combat_rep, "patch_stability_median_pearson")
    farm_icc = _metric(farming, farm_rep, "between_within_icc")
    combat_icc = _metric(combat, combat_rep, "between_within_icc")

    reasons: list[str] = []
    if not np.isfinite(farm_half) or farm_half < _REPEATABILITY_FLOOR:
        reasons.append("farming split-half below floor")
    if not np.isfinite(combat_half) or combat_half < _REPEATABILITY_FLOOR:
        reasons.append("combat split-half below floor")
    if farm_status == "unresolved" or combat_status == "unresolved":
        reasons.append("hero-only vs hero×position unresolved")
    patch_unstable = (np.isfinite(farm_patch) and farm_patch < _PATCH_CORR_SOFT) or (
        np.isfinite(combat_patch) and combat_patch < _PATCH_CORR_SOFT
    )
    if patch_unstable:
        reasons.append("patch/version profile shifts are material")
    lpo_soft = (np.isfinite(farm_lpo) and farm_lpo < _LPO_CORR_FLOOR) or (
        np.isfinite(combat_lpo) and combat_lpo < _LPO_CORR_FLOOR
    )
    if lpo_soft:
        reasons.append("player concentration distorts some hero profiles")

    failed = any("split-half below floor" in item for item in reasons)
    if failed:
        classification = "C"
        gate = CLASSIFICATION_C
        next_slice = "Do not freeze a hero profile; temporal identity did not survive."
    elif reasons:
        classification = "B"
        gate = CLASSIFICATION_B
        next_slice = (
            "Keep the observation definitions but resolve position/patch/"
            "concentration before historical-state construction."
        )
    else:
        classification = "A"
        gate = CLASSIFICATION_A
        next_slice = (
            "Construct leakage-safe historical hero-profile state "
            "(shrinkage / recency / patch weighting later)."
        )

    rationale = (
        f"farming key={farm_key} ({farm_why}, split-half={farm_half:.3f}, "
        f"icc={farm_icc:.3f}, patch r={farm_patch:.3f}, LPO r={farm_lpo:.3f}); "
        f"combat key={combat_key} ({combat_why}, split-half={combat_half:.3f}, "
        f"icc={combat_icc:.3f}, patch r={combat_patch:.3f}, LPO r={combat_lpo:.3f})"
    )
    if reasons:
        rationale = rationale + " | " + "; ".join(reasons)
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "gate": gate,
                "farming_key": farm_key,
                "combat_key": combat_key,
                "farming_target": HERO_FARMING_PROFILE_TARGET,
                "combat_target": HERO_COMBAT_PROFILE_TARGET,
                "farming_representation": farm_rep,
                "combat_representation": combat_rep,
                "rationale": rationale,
                "next_slice": next_slice,
            }
        ]
    )


def run_hero_performance_profile_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice21DiagnosticReport:
    """Development-only Slice 21 hero-profile research. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    development = attach_hero_profile_observations(development)
    n_missing_position = int((~explicit_position_mask(development)).sum())
    n_explicit_position = int(explicit_position_mask(development).sum())

    for spec in PROFILE_SPECS:
        development = attach_causal_group_mean(
            development,
            value_column=spec.observation_column,
            group_columns=spec.group_columns,
            out_column=f"{spec.name.lower()}_causal_mean",
            n_column=f"{spec.name.lower()}_causal_n",
        )

    usage = position_usage_table(development)
    formulas = pd.DataFrame(
        [
            {
                "representation": spec.name,
                "dimension": spec.dimension,
                "observation": spec.observation_column,
                "key": spec.key,
                "formula": spec.formula,
                "hero_agnostic_observation": spec.observation_column
                in {CAUSAL_B_COLUMN, CAUSAL_C_COLUMN},
                "uses_match_result": False,
                "uses_current_box_score_as_state": False,
            }
            for spec in PROFILE_SPECS
        ]
    )
    coverage = pd.DataFrame(
        [_coverage_for_spec(development, spec) for spec in PROFILE_SPECS]
    )
    coverage_parts = [
        coverage_threshold_table(
            development.groupby("hero_id").size(), label="hero_observations"
        ),
        coverage_threshold_table(
            development.loc[explicit_position_mask(development)]
            .groupby(["hero_id", "position_number"])
            .size(),
            label="hero_position_observations",
        ),
        coverage_threshold_table(
            development.groupby("hero_id")["player_id"].nunique(),
            label="hero_unique_players",
        ),
        coverage_threshold_table(
            development.groupby("hero_id")["match_id"].nunique(),
            label="hero_unique_matches",
        ),
    ]
    coverage_thresholds = pd.concat(coverage_parts, ignore_index=True)

    position_dependence = pd.DataFrame(
        [_position_dependence_row(development, spec, usage) for spec in PROFILE_SPECS]
    )
    circularity = pd.DataFrame(
        [_circularity_row(development, spec) for spec in PROFILE_SPECS]
    )
    player_demean = pd.DataFrame(
        [_player_demean_row(development, spec) for spec in PROFILE_SPECS]
    )
    concentration_parts: list[pd.DataFrame] = []
    for spec in PROFILE_SPECS:
        table = player_concentration_table(
            development,
            group_columns=spec.group_columns,
            value_column=spec.observation_column,
        )
        if table.empty:
            continue
        table = table.assign(representation=spec.name, dimension=spec.dimension)
        concentration_parts.append(table)
    player_concentration = (
        pd.concat(concentration_parts, ignore_index=True)
        if concentration_parts
        else pd.DataFrame()
    )

    split_rows: list[dict[str, object]] = []
    variance_rows: list[dict[str, object]] = []
    for spec in PROFILE_SPECS:
        min_each = (
            MIN_HALF_HERO_POSITION if spec.key == KEY_HERO_POSITION else MIN_HALF_HERO
        )
        half = group_split_half(
            development,
            value_column=spec.observation_column,
            group_columns=spec.group_columns,
            min_each=min_each,
        )
        half.update(
            {
                "representation": spec.name,
                "dimension": spec.dimension,
                "key": spec.key,
            }
        )
        split_rows.append(half)
        var = group_variance_decomposition(
            development,
            value_column=spec.observation_column,
            group_columns=spec.group_columns,
        )
        var.update(
            {
                "representation": spec.name,
                "dimension": spec.dimension,
                "key": spec.key,
            }
        )
        variance_rows.append(var)
    split_half = pd.DataFrame(split_rows)
    variance_decomposition = pd.DataFrame(variance_rows)
    shrinkage_diagnosis = variance_decomposition.loc[
        :,
        [
            "representation",
            "dimension",
            "key",
            "within_variance",
            "between_variance",
            "within_over_between",
            "icc",
            "n_profiles",
        ],
    ].copy()

    development = development.assign(
        chrono_block=assign_chronological_blocks(development)
    )
    block_counts = (
        development.drop_duplicates("match_id")
        .groupby("chrono_block")
        .size()
        .rename("n_matches")
        .reset_index()
    )
    temporal_blocks = block_counts.assign(
        n_player_rows=development.groupby("chrono_block").size().to_numpy()
    )
    adjacent_parts: list[pd.DataFrame] = []
    patch_adj_parts: list[pd.DataFrame] = []
    same_pos_patch_parts: list[pd.DataFrame] = []
    patch_count_rows: list[dict[str, object]] = []
    for spec in PROFILE_SPECS:
        material = (
            MATERIAL_FARMING_SHIFT
            if spec.dimension == "farming"
            else MATERIAL_COMBAT_SHIFT
        )
        block_table = _adjacent_window_stability(
            development,
            value_column=spec.observation_column,
            group_columns=spec.group_columns,
            window_column="chrono_block",
            min_n=MIN_BLOCK_PROFILE_N,
            material_shift=material,
        )
        if not block_table.empty:
            adjacent_parts.append(
                block_table.assign(representation=spec.name, dimension=spec.dimension)
            )
        patch_table = _adjacent_window_stability(
            development,
            value_column=spec.observation_column,
            group_columns=spec.group_columns,
            window_column="game_version_id",
            min_n=MIN_PATCH_PROFILE_N,
            material_shift=material,
        )
        if not patch_table.empty:
            patch_adj_parts.append(
                patch_table.assign(representation=spec.name, dimension=spec.dimension)
            )
        if spec.key == KEY_HERO_POSITION:
            same_pos = _adjacent_window_stability(
                development,
                value_column=spec.observation_column,
                group_columns=("hero_id", "position_number"),
                window_column="game_version_id",
                min_n=MIN_PATCH_PROFILE_N,
                material_shift=material,
            )
            if not same_pos.empty:
                same_pos_patch_parts.append(
                    same_pos.assign(representation=spec.name, dimension=spec.dimension)
                )
        for version, group in development.groupby("game_version_id", dropna=False):
            values = _numeric(group[spec.observation_column])
            keys = _group_key_series(group, spec.group_columns)
            eligible = values.notna() & keys.notna()
            counts = (
                keys.loc[eligible].value_counts()
                if bool(eligible.any())
                else pd.Series(dtype=int)
            )
            patch_count_rows.append(
                {
                    "representation": spec.name,
                    "game_version_id": version,
                    "n_rows": int(eligible.sum()),
                    "n_profiles": int(counts.size),
                    "median_n_per_profile": (
                        float(counts.median()) if len(counts) else float("nan")
                    ),
                    "n_profiles_ge_20": int((counts >= 20).sum()) if len(counts) else 0,
                }
            )
    adjacent_block_stability = (
        pd.concat(adjacent_parts, ignore_index=True)
        if adjacent_parts
        else pd.DataFrame()
    )
    adjacent_patch_stability = (
        pd.concat(patch_adj_parts, ignore_index=True)
        if patch_adj_parts
        else pd.DataFrame()
    )
    same_hero_position_patch = (
        pd.concat(same_pos_patch_parts, ignore_index=True)
        if same_pos_patch_parts
        else pd.DataFrame()
    )
    patch_stability = pd.DataFrame(patch_count_rows)

    player_state_relationship = pd.DataFrame(
        [_player_state_relationship(development, spec) for spec in PROFILE_SPECS]
    )
    cross_dimension = pd.DataFrame(
        [
            _cross_dimension_row(
                development,
                _SPEC_BY_NAME[FARMING_F1],
                _SPEC_BY_NAME[COMBAT_C1],
                min_n=20,
            ),
            _cross_dimension_row(
                development,
                _SPEC_BY_NAME[FARMING_F2],
                _SPEC_BY_NAME[COMBAT_C2],
                min_n=10,
            ),
        ]
    )

    farming_rows: list[dict[str, object]] = []
    combat_rows: list[dict[str, object]] = []
    for spec in PROFILE_SPECS:
        cov = coverage.loc[coverage["representation"] == spec.name]
        circ = circularity.loc[circularity["representation"] == spec.name]
        half = split_half.loc[split_half["representation"] == spec.name]
        pos = position_dependence.loc[
            position_dependence["representation"] == spec.name
        ]
        if adjacent_patch_stability.empty:
            patch_rows = pd.DataFrame()
        else:
            patch_rows = adjacent_patch_stability.loc[
                adjacent_patch_stability["representation"] == spec.name
            ]
        coverage_ge20 = int(cov.iloc[0]["n_ge_20"]) if not cov.empty else 0
        caveat = _caveat_for(
            spec,
            split_half=(
                float(half.iloc[0]["pearson"]) if not half.empty else float("nan")
            ),
            patch_r=(
                float(patch_rows["pearson"].median())
                if not patch_rows.empty
                else float("nan")
            ),
            lpo_r=(
                float(circ.iloc[0]["pearson_all_vs_lpo"])
                if not circ.empty
                else float("nan")
            ),
            specialist_frac=(
                float(circ.iloc[0]["fraction_specialist"])
                if not circ.empty
                else float("nan")
            ),
            pos_r2=(
                float(pos.iloc[0]["raw_position_r2"]) if not pos.empty else float("nan")
            ),
            coverage_ge20=coverage_ge20,
        )
        row = _comparison_row(
            spec,
            coverage=coverage,
            position=position_dependence,
            split_half=split_half,
            patch=adjacent_patch_stability,
            variance=variance_decomposition,
            circularity=circularity,
            caveat=caveat,
        )
        if spec.dimension == "farming":
            farming_rows.append(row)
        else:
            combat_rows.append(row)
    farming_comparison = pd.DataFrame(farming_rows)
    combat_comparison = pd.DataFrame(combat_rows)

    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    integrity = {
        "development_end": end.isoformat(),
        "holdout_used_for_selection": False,
        "holdout_used_for_stability": False,
        "stratz_called": False,
        "ingestion_modified": False,
        "schema_modified": False,
        "farming_candidate_b_unchanged": FROZEN_CANDIDATE_B == CANDIDATE_B,
        "farming_k_is_5": FROZEN_SHRINKAGE_K == 5.0,
        "combat_candidate_c_unchanged": FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION,
        "combat_k_is_20": FROZEN_COMBAT_SHRINKAGE_K == 20.0,
        "player_hero_fit_created": False,
        "team_feature_created": False,
        "win_model_run": False,
        "hero_shrinkage_k_frozen": False,
        "current_result_used_for_profile": False,
        "current_box_score_used_as_pre_draft_state": False,
        "current_position_treated_as_pre_draft": False,
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "profile_in_feature_columns": any(
            name in FEATURE_COLUMNS
            for name in (
                HERO_FARMING_PROFILE_TARGET,
                HERO_COMBAT_PROFILE_TARGET,
                CAUSAL_B_COLUMN,
                CAUSAL_C_COLUMN,
                *PLAYER_X_HERO_FIT_NAMES,
            )
        ),
        "profile_in_snapshot_columns": any(
            name in SNAPSHOT_COLUMNS for name in PLAYER_X_HERO_FIT_NAMES
        ),
        "profile_in_pre_draft_sql": any(
            name in PRE_DRAFT_SNAPSHOT_SQL for name in PLAYER_X_HERO_FIT_NAMES
        ),
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "match_player_box_score_field_count": len(MATCH_PLAYER_BOX_SCORE_COLUMNS),
        "n_holdout_excluded": len(holdout),
        "model_trained": False,
        "player_rating_persisted": False,
    }
    report = Slice21DiagnosticReport(
        development_end=end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        n_missing_position=n_missing_position,
        n_explicit_position=n_explicit_position,
        formulas=formulas,
        coverage=coverage,
        coverage_thresholds=coverage_thresholds,
        position_usage=usage,
        position_dependence=position_dependence,
        flex_heroes=flex_hero_table(usage),
        circularity=circularity,
        player_concentration=player_concentration,
        player_demean=player_demean,
        split_half=split_half,
        temporal_blocks=temporal_blocks,
        adjacent_block_stability=adjacent_block_stability,
        variance_decomposition=variance_decomposition,
        shrinkage_diagnosis=shrinkage_diagnosis,
        patch_stability=patch_stability,
        adjacent_patch_stability=adjacent_patch_stability,
        same_hero_position_patch=same_hero_position_patch,
        player_state_relationship=player_state_relationship,
        cross_dimension=cross_dimension,
        farming_comparison=farming_comparison,
        combat_comparison=combat_comparison,
        classification=pd.DataFrame(),
        integrity=integrity,
    )
    classification = classify_slice21(report)
    return Slice21DiagnosticReport(
        **{**report.__dict__, "classification": classification}
    )


def slice21_report_to_jsonable(report: Slice21DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 21 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "n_missing_position": report.n_missing_position,
        "n_explicit_position": report.n_explicit_position,
        "frozen_farming_target": HERO_FARMING_PROFILE_TARGET,
        "frozen_farming_key": HERO_FARMING_PROFILE_KEY,
        "frozen_combat_target": HERO_COMBAT_PROFILE_TARGET,
        "frozen_combat_key": HERO_COMBAT_PROFILE_KEY,
        "formulas": _jsonable_value(report.formulas),
        "coverage": _jsonable_value(report.coverage),
        "coverage_thresholds": _jsonable_value(report.coverage_thresholds),
        "position_usage": _jsonable_value(report.position_usage),
        "position_dependence": _jsonable_value(report.position_dependence),
        "flex_heroes": _jsonable_value(report.flex_heroes),
        "circularity": _jsonable_value(report.circularity),
        "player_concentration": _jsonable_value(report.player_concentration),
        "player_demean": _jsonable_value(report.player_demean),
        "split_half": _jsonable_value(report.split_half),
        "temporal_blocks": _jsonable_value(report.temporal_blocks),
        "adjacent_block_stability": _jsonable_value(report.adjacent_block_stability),
        "variance_decomposition": _jsonable_value(report.variance_decomposition),
        "shrinkage_diagnosis": _jsonable_value(report.shrinkage_diagnosis),
        "patch_stability": _jsonable_value(report.patch_stability),
        "adjacent_patch_stability": _jsonable_value(report.adjacent_patch_stability),
        "same_hero_position_patch": _jsonable_value(report.same_hero_position_patch),
        "player_state_relationship": _jsonable_value(report.player_state_relationship),
        "cross_dimension": _jsonable_value(report.cross_dimension),
        "farming_comparison": _jsonable_value(report.farming_comparison),
        "combat_comparison": _jsonable_value(report.combat_comparison),
        "classification": _jsonable_value(report.classification),
        "integrity": _jsonable_value(report.integrity),
    }
