"""Slice 24: current-meta Hero × Position state diagnostics.

Research / state only. This module does not add production features, does
not train a win model, does not construct player×hero fit, and does not
overwrite frozen Slice 21/22 long-run requirement states.

Question
--------
Does Hero × Position have a meaningful **time-varying competitive meta
state** beyond its long-run historical identity?

Slice 21/22 describe relatively stable Hero × Position identity /
requirements. This slice investigates environmental state:

    Hero × Position × time

It does not collapse usage, Elo-adjusted outcomes, and requirement
drift into one “meta strength” score.

Population
----------
Matches with ``start_time <=`` the frozen Slice 9 development end
(``FROZEN_DEVELOPMENT_END``). Holdout / TI 2026 rows are excluded from
every summary used to classify or select a window.

Temporal integrity
------------------
For current match ``M`` at ``M.start_time`` every historical observation
satisfies ``historical.start_time < M.start_time``. Same-timestamp
matches are mutually blind. Current-match result, duration, box score,
and any future match never enter the *state*. Frozen farming/combat
targets and Elo residuals of *strictly prior* appearances are
observations used to build that state.

Position
--------
Only historically observed explicit positions 1–5 contribute, matching
Slice 21/22. Current post-match position is diagnostic, not a PRE_DRAFT
production input.

Windows
-------
Comparators are existing project conventions, not a new search:

* expanding long-run history
* recent 90 days (``hero_meta.RECENT_WINDOW_DAYS``)
* recent 180 days (walk-forward ``LAST_180D`` memory policy), diagnostic
  robustness only
* current ``game_version_id`` (hero-meta same-version convention)
* current + immediately previous represented version (memory policy)

90 days is the a priori recent window. 180 days is inspected on
development/tune data if 90-day coverage is too sparse to evaluate
persistence. Windows are not chosen against match-outcome log-loss.

Cold start
----------
Unseen H×P: counts are 0 and means/shares with a zero denominator are
NULL. Observed absence of H at a position that *was* played is share 0,
not NULL. No population-mean fallback is applied unless diagnostics
support a frozen hierarchy (they do not, a priori).

Research result
---------------
Classification **C — do not freeze**. Recent and current-version H×P
usage and Elo residuals differ from long-run history, but they do not
predict the next causal H×P observation better than expanding history.
Farming/combat requirement profiles are already captured by frozen
Slice 22 long-run states. This slice is diagnostic-only.
"""

from __future__ import annotations

from collections import defaultdict, deque
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
from dota_predictor.features.hero_meta import RECENT_WINDOW_DAYS
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
    MATERIAL_COMBAT_SHIFT,
    MATERIAL_FARMING_SHIFT,
    PLAYER_X_HERO_FIT_NAMES,
    assign_chronological_blocks,
    attach_hero_profile_observations,
)
from dota_predictor.training.hero_requirement_state import (
    FROZEN_HERO_COMBAT_SHRINKAGE_K,
    FROZEN_HERO_FARM_SHRINKAGE_K,
    PREFERRED_TUNE_END,
    SLICE22_STATE_COLUMNS,
    attach_hero_requirement_state,
)
from dota_predictor.training.player_combat_state import (
    CAUSAL_C_COLUMN,
    FROZEN_COMBAT_SHRINKAGE_K,
)
from dota_predictor.training.player_farming_state import (
    CAUSAL_B_COLUMN,
    EQUIVALENT_RMSE_RATIO,
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    HISTORY_N_BUCKETS,
    SHRINKAGE_GRID,
    apply_farming_shrinkage,
    development_tune_end,
    history_n_bucket,
)
from dota_predictor.training.player_hero_compatibility import (
    SLICE23_DIAGNOSTIC_ONLY,
    SLICE23_FIT_SCORE_FROZEN,
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
    "ELO_RESIDUAL_COLUMN",
    "HERO_POSITION_GROUP",
    "RECENT_WINDOW_DAYS_ALT",
    "RECENT_WINDOW_DAYS_PRIMARY",
    "SLICE24_DIAGNOSTIC_COLUMNS",
    "SLICE24_DIAGNOSTIC_ONLY",
    "SLICE24_FROZEN_COMPONENTS",
    "SLICE24_RESEARCH_CLASSIFICATION",
    "SLICE24_RESIDUAL_SHRINKAGE_K_FROZEN",
    "SLICE24_STATE_COLUMNS",
    "WINDOW_SPECS",
    "Slice24DiagnosticReport",
    "WindowSpec",
    "attach_elo_residual",
    "attach_hero_position_meta_state",
    "causal_previous_version_id",
    "classify_slice24",
    "run_hero_position_meta_diagnostics",
    "slice24_report_to_jsonable",
]


ELO_RESIDUAL_COLUMN = "hp_elo_residual"
HERO_POSITION_GROUP = ("hero_id", "position_number")
RECENT_WINDOW_DAYS_PRIMARY = RECENT_WINDOW_DAYS
RECENT_WINDOW_DAYS_ALT = 180
MIN_COMPARE_N = 5
MIN_WINDOW_PROFILE_N = 5
RTM_EXTREME_QUANTILE = 0.90
RTM_MIN_EACH = 5
COVERAGE_N_THRESHOLDS: tuple[int, ...] = (1, 5, 10, 20, 50)
MATERIAL_USAGE_SHARE_SHIFT = 0.10
MATERIAL_RESIDUAL_SHIFT = 0.05
USAGE_RANK_MATERIAL = 10
_VARIATION_CORR = 0.85
_PERSISTENCE_PEARSON_DELTA = 0.05
_SHARE_COMPARE_FLOOR = 0.05

VERSION_ALL = "all"
VERSION_CURRENT = "current"
VERSION_CURRENT_PLUS_PREVIOUS = "current_plus_previous"

CLASSIFICATION_A = (
    "A — freeze current-meta H×P state: temporal variation plus useful "
    "persistence, with a stable causal definition"
)
CLASSIFICATION_B = (
    "B — suggestive / partial: some current-meta information exists, "
    "but only certain dimensions or estimators are reliable"
)
CLASSIFICATION_C = (
    "C — do not freeze: recent/version-specific H×P state is mostly "
    "noise or adds no meaningful information beyond long-run state"
)

# Recorded after development diagnostics. Not a production-feature freeze.
# Classification C: recent/version H×P state varies, but does not persist
# usefully beyond long-run Slice 21/22 identity. Nothing is frozen.
SLICE24_RESEARCH_CLASSIFICATION = "C"
SLICE24_DIAGNOSTIC_ONLY = True
SLICE24_RESIDUAL_SHRINKAGE_K_FROZEN = False
SLICE24_FROZEN_COMPONENTS: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowSpec:
    """One causal H×P history filter. Not a production feature."""

    name: str
    window_days: int | None
    version_mode: str
    justification: str


WINDOW_SPECS: tuple[WindowSpec, ...] = (
    WindowSpec(
        name="expanding",
        window_days=None,
        version_mode=VERSION_ALL,
        justification="long-run H×P history; Slice 21/22 identity baseline",
    ),
    WindowSpec(
        name="recent_90d",
        window_days=RECENT_WINDOW_DAYS_PRIMARY,
        version_mode=VERSION_ALL,
        justification=(
            "hero_meta same-hero recent window "
            f"({RECENT_WINDOW_DAYS_PRIMARY} days); a priori current-meta calendar"
        ),
    ),
    WindowSpec(
        name="recent_180d",
        window_days=RECENT_WINDOW_DAYS_ALT,
        version_mode=VERSION_ALL,
        justification=(
            "walk-forward LAST_180D memory policy; diagnostic robustness "
            "if 90-day H×P cells are too sparse"
        ),
    ),
    WindowSpec(
        name="current_version",
        window_days=None,
        version_mode=VERSION_CURRENT,
        justification="hero_meta same-version window; current patch environment",
    ),
    WindowSpec(
        name="current_plus_previous",
        window_days=None,
        version_mode=VERSION_CURRENT_PLUS_PREVIOUS,
        justification=(
            "walk-forward CURRENT_PLUS_PREVIOUS_VERSION memory policy; "
            "patch-transition robustness"
        ),
    ),
)


def _state_columns_for_spec(spec: WindowSpec) -> tuple[str, ...]:
    prefix = f"hp_{spec.name}"
    return (
        f"{prefix}_n",
        f"{prefix}_pos_n",
        f"{prefix}_hero_n",
        f"{prefix}_hero_share_at_position",
        f"{prefix}_position_share_of_hero",
        f"{prefix}_elo_residual_n",
        f"{prefix}_elo_residual_mean",
        f"{prefix}_farming_n",
        f"{prefix}_farming_mean",
        f"{prefix}_combat_n",
        f"{prefix}_combat_mean",
    )


SLICE24_STATE_COLUMNS: tuple[str, ...] = tuple(
    column for spec in WINDOW_SPECS for column in _state_columns_for_spec(spec)
)
SLICE24_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    ELO_RESIDUAL_COLUMN,
    *SLICE24_STATE_COLUMNS,
)


@dataclass(frozen=True)
class Slice24DiagnosticReport:
    development_end: datetime
    tune_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    selected_recent_window: str
    selected_recent_window_justification: str
    residual_shrinkage_k: float
    residual_shrinkage_justification: str
    semantics: dict[str, object]
    classification: pd.DataFrame
    split: pd.DataFrame
    coverage: pd.DataFrame
    cold_start: pd.DataFrame
    history_size: pd.DataFrame
    estimator_comparison: pd.DataFrame
    residual_persistence: pd.DataFrame
    usage_persistence: pd.DataFrame
    version_transfer: pd.DataFrame
    same_version_persistence: pd.DataFrame
    requirement_drift: pd.DataFrame
    residual_shrinkage_tune: pd.DataFrame
    residual_shrinkage_validation: pd.DataFrame
    regression_to_mean: pd.DataFrame
    sample_size: pd.DataFrame
    integrity: dict[str, object]


class _Counts:
    """Mutable H×P / position / hero accumulators for one history filter."""

    def __init__(self) -> None:
        self.hp_n: dict[tuple[int, int], int] = defaultdict(int)
        self.pos_n: dict[int, int] = defaultdict(int)
        self.hero_n: dict[int, int] = defaultdict(int)
        self.val_n: dict[str, dict[tuple[int, int], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.val_sum: dict[str, dict[tuple[int, int], float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def add(
        self,
        hp: tuple[int, int],
        pos: int,
        hero: int,
        *,
        usage: bool,
        values: dict[str, float],
        sign: int = 1,
    ) -> None:
        if usage:
            self.hp_n[hp] += sign
            self.pos_n[pos] += sign
            self.hero_n[hero] += sign
        for name, value in values.items():
            self.val_n[name][hp] += sign
            self.val_sum[name][hp] += sign * value

    def lookup(
        self, hp: tuple[int, int], pos: int, hero: int, value_names: tuple[str, ...]
    ) -> dict[str, float]:
        return _fast_lookup([self], hp, pos, hero, value_names)


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


def attach_elo_residual(frame: pd.DataFrame) -> pd.DataFrame:
    """Team result minus pre-match Elo expected win.

    ``team_won`` is the current appearance's own result. The residual is
    an *observation* of a historical H×P appearance, never a current-match
    production feature. Same-timestamp Elo ratings already exclude the
    current match (``match_elo_expected_wins``).
    """
    out = frame.copy()
    won = _numeric(out["team_won"])
    expected = _numeric(out["elo_expected_win"])
    residual = won - expected
    residual = residual.mask(won.isna() | expected.isna())
    out[ELO_RESIDUAL_COLUMN] = residual
    return out


def causal_previous_version_id(frame: pd.DataFrame) -> pd.Series:
    """Immediately previous represented version among strictly prior rows.

    Versions are ordered by first-seen ``start_time`` among matches with
    ``start_time < T``. If the current version has not yet appeared
    before ``T``, previous is the last first-seen version before ``T``.
    Same-timestamp rows share the same prior version list.
    """
    out = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if frame.empty or "game_version_id" not in frame.columns:
        return out
    times = pd.to_datetime(frame["start_time"], utc=True).to_numpy()
    versions = _numeric(frame["game_version_id"]).to_numpy(dtype=float)
    order = np.argsort(times, kind="mergesort")
    sorted_times = times[order]
    cuts = np.r_[True, sorted_times[1:] != sorted_times[:-1]]
    starts = np.flatnonzero(cuts)
    bounds = np.r_[starts, len(order)]
    first_seen_order: list[int] = []
    seen: set[int] = set()
    prev_vals = np.full(len(frame), np.nan, dtype=float)
    for i in range(len(starts)):
        lo = int(bounds[i])
        hi = int(bounds[i + 1])
        for j in range(lo, hi):
            row = int(order[j])
            ver = versions[row]
            if not np.isfinite(ver):
                continue
            current = int(ver)
            if current in seen:
                idx = first_seen_order.index(current)
                if idx > 0:
                    prev_vals[row] = float(first_seen_order[idx - 1])
            elif first_seen_order:
                prev_vals[row] = float(first_seen_order[-1])
        for j in range(lo, hi):
            row = int(order[j])
            ver = versions[row]
            if not np.isfinite(ver):
                continue
            current = int(ver)
            if current not in seen:
                seen.add(current)
                first_seen_order.append(current)
    nullable = pd.array(prev_vals, dtype="Float64")
    out.loc[:] = nullable.astype("Int64")
    return out


def _value_map(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    available: dict[str, np.ndarray] = {}
    for name in (ELO_RESIDUAL_COLUMN, CAUSAL_B_COLUMN, CAUSAL_C_COLUMN):
        if name in frame.columns:
            available[name] = _numeric(frame[name]).to_numpy(dtype=float)
    return available


def _fast_lookup(
    counts_list: list[_Counts],
    hp: tuple[int, int],
    pos: int,
    hero: int,
    value_names: tuple[str, ...],
) -> dict[str, float]:
    n = 0
    pos_n = 0
    hero_n = 0
    val_n = {name: 0 for name in value_names}
    val_sum = {name: 0.0 for name in value_names}
    for counts in counts_list:
        n += int(counts.hp_n.get(hp, 0))
        pos_n += int(counts.pos_n.get(pos, 0))
        hero_n += int(counts.hero_n.get(hero, 0))
        for name in value_names:
            val_n[name] += int(counts.val_n[name].get(hp, 0))
            val_sum[name] += float(counts.val_sum[name].get(hp, 0.0))
    row: dict[str, float] = {"n": n, "pos_n": pos_n, "hero_n": hero_n}
    for name in value_names:
        row[f"{name}_n"] = val_n[name]
        row[f"{name}_sum"] = val_sum[name]
    return row


def _empty_counts() -> _Counts:
    return _Counts()


def _scan_window(
    frame: pd.DataFrame,
    spec: WindowSpec,
    *,
    previous_version: pd.Series,
) -> dict[str, np.ndarray]:
    n_rows = len(frame)
    value_map = _value_map(frame)
    value_names = tuple(value_map.keys())
    hp_n = np.zeros(n_rows, dtype=int)
    pos_n = np.zeros(n_rows, dtype=int)
    hero_n = np.zeros(n_rows, dtype=int)
    val_n = {name: np.zeros(n_rows, dtype=int) for name in value_names}
    val_sum = {name: np.zeros(n_rows, dtype=float) for name in value_names}
    if frame.empty:
        return {
            "n": hp_n,
            "pos_n": pos_n,
            "hero_n": hero_n,
            **{f"{name}_n": val_n[name] for name in value_names},
            **{f"{name}_sum": val_sum[name] for name in value_names},
        }

    times = pd.to_datetime(frame["start_time"], utc=True)
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    key_series = _hero_position_keys(frame)
    keys = key_series.to_numpy()
    versions = (
        _numeric(frame["game_version_id"]).to_numpy(dtype=float)
        if "game_version_id" in frame.columns
        else np.full(n_rows, np.nan)
    )
    prev_ver = _numeric(previous_version).to_numpy(dtype=float)
    attachable = key_series.notna().to_numpy()
    if not bool(attachable.any()):
        return {
            "n": hp_n,
            "pos_n": pos_n,
            "hero_n": hero_n,
            **{f"{name}_n": val_n[name] for name in value_names},
            **{f"{name}_sum": val_sum[name] for name in value_names},
        }

    ns_per_day = 24 * 60 * 60 * 10**9
    global_counts = _empty_counts()
    version_counts: dict[int, _Counts] = defaultdict(_empty_counts)
    pending: deque[
        tuple[int, int | None, tuple[int, int], int, int, bool, dict[str, float]]
    ] = deque()

    order = np.argsort(time_ns, kind="mergesort")
    sorted_ns = time_ns[order]
    cuts = np.r_[True, sorted_ns[1:] != sorted_ns[:-1]]
    starts = np.flatnonzero(cuts)
    bounds = np.r_[starts, len(order)]

    def _expire(cutoff_ns: int) -> None:
        while pending and pending[0][0] < cutoff_ns:
            _t, ver, hp, pos, hero, usage, values = pending.popleft()
            global_counts.add(hp, pos, hero, usage=usage, values=values, sign=-1)
            if ver is not None:
                version_counts[ver].add(
                    hp, pos, hero, usage=usage, values=values, sign=-1
                )

    for i in range(len(starts)):
        lo = int(bounds[i])
        hi = int(bounds[i + 1])
        group_ns = int(sorted_ns[lo])
        if spec.window_days is not None:
            _expire(group_ns - int(spec.window_days) * ns_per_day)
        for j in range(lo, hi):
            row = int(order[j])
            if not bool(attachable[row]):
                continue
            hp = keys[row]
            pos = int(hp[1])
            hero = int(hp[0])
            ver_raw = versions[row]
            current_ver = int(ver_raw) if np.isfinite(ver_raw) else None
            prev_raw = prev_ver[row]
            previous = int(prev_raw) if np.isfinite(prev_raw) else None
            if spec.version_mode == VERSION_ALL:
                selected = [global_counts]
            elif spec.version_mode == VERSION_CURRENT:
                if current_ver is None:
                    selected = []
                else:
                    selected = [version_counts[current_ver]]
            else:
                selected = []
                if current_ver is not None:
                    selected.append(version_counts[current_ver])
                if previous is not None:
                    selected.append(version_counts[previous])
            snap = (
                _fast_lookup(selected, hp, pos, hero, value_names)
                if selected
                else {
                    "n": 0.0,
                    "pos_n": 0.0,
                    "hero_n": 0.0,
                    **{f"{name}_n": 0.0 for name in value_names},
                    **{f"{name}_sum": 0.0 for name in value_names},
                }
            )
            hp_n[row] = int(snap["n"])
            pos_n[row] = int(snap["pos_n"])
            hero_n[row] = int(snap["hero_n"])
            for name in value_names:
                val_n[name][row] = int(snap[f"{name}_n"])
                val_sum[name][row] = float(snap[f"{name}_sum"])
        for j in range(lo, hi):
            row = int(order[j])
            if not bool(attachable[row]):
                continue
            hp = keys[row]
            pos = int(hp[1])
            hero = int(hp[0])
            values = {
                name: float(value_map[name][row])
                for name in value_names
                if np.isfinite(value_map[name][row])
            }
            ver_raw = versions[row]
            current_ver = int(ver_raw) if np.isfinite(ver_raw) else None
            event = (
                int(sorted_ns[j]),
                current_ver,
                hp,
                pos,
                hero,
                True,
                values,
            )
            global_counts.add(hp, pos, hero, usage=True, values=values)
            if current_ver is not None:
                version_counts[current_ver].add(
                    hp, pos, hero, usage=True, values=values
                )
            if spec.window_days is not None:
                pending.append(event)

    return {
        "n": hp_n,
        "pos_n": pos_n,
        "hero_n": hero_n,
        **{f"{name}_n": val_n[name] for name in value_names},
        **{f"{name}_sum": val_sum[name] for name in value_names},
    }


def _mean_from_sum(total: np.ndarray, n: np.ndarray) -> np.ndarray:
    out = np.full(len(n), np.nan, dtype=float)
    mask = n > 0
    out[mask] = total[mask] / n[mask]
    return out


def _share(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.full(len(numer), np.nan, dtype=float)
    mask = denom > 0
    out[mask] = numer[mask] / denom[mask]
    return out


def attach_hero_position_meta_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal H×P usage, Elo-residual, and requirement-window state.

    Inclusive of other players: this is environmental meta state, not
    Slice 22 leave-current-player-out identity. Does not recompute
    causal B/C. Does not fill NULL means with a population baseline.
    """
    if ELO_RESIDUAL_COLUMN in frame.columns:
        out = frame.copy()
    else:
        out = attach_elo_residual(frame)
    previous = causal_previous_version_id(out)
    out["hp_previous_version_id"] = previous
    for spec in WINDOW_SPECS:
        scanned = _scan_window(out, spec, previous_version=previous)
        prefix = f"hp_{spec.name}"
        out[f"{prefix}_n"] = scanned["n"]
        out[f"{prefix}_pos_n"] = scanned["pos_n"]
        out[f"{prefix}_hero_n"] = scanned["hero_n"]
        out[f"{prefix}_hero_share_at_position"] = _share(
            scanned["n"].astype(float), scanned["pos_n"].astype(float)
        )
        out[f"{prefix}_position_share_of_hero"] = _share(
            scanned["n"].astype(float), scanned["hero_n"].astype(float)
        )
        residual_n = scanned.get(
            f"{ELO_RESIDUAL_COLUMN}_n", np.zeros(len(out), dtype=int)
        )
        residual_sum = scanned.get(
            f"{ELO_RESIDUAL_COLUMN}_sum", np.zeros(len(out), dtype=float)
        )
        out[f"{prefix}_elo_residual_n"] = residual_n
        out[f"{prefix}_elo_residual_mean"] = _mean_from_sum(residual_sum, residual_n)
        farm_n = scanned.get(f"{CAUSAL_B_COLUMN}_n", np.zeros(len(out), dtype=int))
        farm_sum = scanned.get(
            f"{CAUSAL_B_COLUMN}_sum", np.zeros(len(out), dtype=float)
        )
        out[f"{prefix}_farming_n"] = farm_n
        out[f"{prefix}_farming_mean"] = _mean_from_sum(farm_sum, farm_n)
        combat_n = scanned.get(f"{CAUSAL_C_COLUMN}_n", np.zeros(len(out), dtype=int))
        combat_sum = scanned.get(
            f"{CAUSAL_C_COLUMN}_sum", np.zeros(len(out), dtype=float)
        )
        out[f"{prefix}_combat_n"] = combat_n
        out[f"{prefix}_combat_mean"] = _mean_from_sum(combat_sum, combat_n)
        explicit = explicit_position_mask(out)
        for column in _state_columns_for_spec(spec):
            if column.endswith("_n"):
                out.loc[~explicit, column] = 0
            else:
                out.loc[~explicit, column] = np.nan
    return out


def _finite_pair(
    actual: pd.Series, predicted: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    a = _numeric(actual)
    p = _numeric(predicted)
    mask = a.notna() & p.notna() & np.isfinite(a.to_numpy()) & np.isfinite(p.to_numpy())
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


def _zero_predictor(actual: pd.Series) -> dict[str, object]:
    predicted = pd.Series(0.0, index=actual.index)
    mask = _numeric(actual).notna()
    return _prediction_metrics(actual[mask], predicted[mask])


def _compare_estimators(
    frame: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    mean_suffix: str,
    n_suffix: str,
    material_shift: float,
    min_n: int = MIN_COMPARE_N,
) -> dict[str, object]:
    left = _numeric(frame[f"hp_{left_name}_{mean_suffix}"])
    right = _numeric(frame[f"hp_{right_name}_{mean_suffix}"])
    left_n = _numeric(frame[f"hp_{left_name}_{n_suffix}"])
    right_n = _numeric(frame[f"hp_{right_name}_{n_suffix}"])
    mask = (
        (left_n >= min_n)
        & (right_n >= min_n)
        & left.notna()
        & right.notna()
        & np.isfinite(left.to_numpy())
        & np.isfinite(right.to_numpy())
    )
    a = left[mask]
    b = right[mask]
    n = int(mask.sum())
    delta = b - a
    abs_delta = delta.abs()
    ranks_a = a.rank(method="average")
    ranks_b = b.rank(method="average")
    rank_delta = (ranks_b - ranks_a).abs()
    sign_change = (np.sign(a.to_numpy()) != np.sign(b.to_numpy())) & (
        (a.to_numpy() != 0.0) | (b.to_numpy() != 0.0)
    )
    return {
        "left": left_name,
        "right": right_name,
        "family": mean_suffix,
        "n_paired": n,
        "min_n": min_n,
        "pearson": _pearson(a, b) if n else float("nan"),
        "spearman": _spearman(a, b) if n else float("nan"),
        "mean_abs_change": float(abs_delta.mean()) if n else float("nan"),
        "median_abs_change": float(abs_delta.median()) if n else float("nan"),
        "fraction_material_change": (
            float((abs_delta >= material_shift).mean()) if n else float("nan")
        ),
        "material_shift": material_shift,
        "mean_abs_rank_change": float(rank_delta.mean()) if n else float("nan"),
        "fraction_rank_change_ge_10": (
            float((rank_delta >= USAGE_RANK_MATERIAL).mean()) if n else float("nan")
        ),
        "fraction_sign_change": float(sign_change.mean()) if n else float("nan"),
    }


def _coverage_row(frame: pd.DataFrame, spec: WindowSpec) -> dict[str, object]:
    explicit = explicit_position_mask(frame)
    subset = frame.loc[explicit]
    n_col = f"hp_{spec.name}_n"
    residual_n = f"hp_{spec.name}_elo_residual_n"
    n_vals = _numeric(subset[n_col])
    keys = _hero_position_keys(subset)
    ever = subset.loc[n_vals > 0, ["hero_id", "position_number"]].drop_duplicates()
    return {
        "estimator": spec.name,
        "n_explicit_appearances": int(explicit.sum()),
        "n_hp_with_prior": int(ever.shape[0]),
        "n_hp_keys_in_frame": int(
            subset.loc[keys.notna(), ["hero_id", "position_number"]]
            .drop_duplicates()
            .shape[0]
        ),
        "mean_prior_n": float(n_vals.mean()) if len(subset) else float("nan"),
        "median_prior_n": float(n_vals.median()) if len(subset) else float("nan"),
        "p90_prior_n": float(n_vals.quantile(0.90)) if len(subset) else float("nan"),
        "fraction_n_eq_0": float((n_vals == 0).mean()) if len(subset) else float("nan"),
        "fraction_residual_n_eq_0": (
            float((_numeric(subset[residual_n]) == 0).mean())
            if len(subset)
            else float("nan")
        ),
        "fraction_hero_share_null": (
            float(subset[f"hp_{spec.name}_hero_share_at_position"].isna().mean())
            if len(subset)
            else float("nan")
        ),
        "fraction_residual_mean_null": (
            float(subset[f"hp_{spec.name}_elo_residual_mean"].isna().mean())
            if len(subset)
            else float("nan")
        ),
        **{
            f"fraction_n_ge_{threshold}": float((n_vals >= threshold).mean())
            if len(subset)
            else float("nan")
            for threshold in COVERAGE_N_THRESHOLDS
        },
    }


def _cold_start_table(frame: pd.DataFrame) -> pd.DataFrame:
    explicit = frame.loc[explicit_position_mask(frame)]
    expanding_n = _numeric(explicit["hp_expanding_n"])
    current_n = _numeric(explicit["hp_current_version_n"])
    recent_n = _numeric(explicit["hp_recent_90d_n"])
    unseen = expanding_n == 0
    historical_not_current = (expanding_n > 0) & (current_n == 0)
    sparse_current = (current_n > 0) & (current_n < MIN_COMPARE_N)
    return pd.DataFrame(
        [
            {
                "case": "unseen_hp",
                "n_rows": int(unseen.sum()),
                "fraction_rows": float(unseen.mean())
                if len(explicit)
                else float("nan"),
                "behaviour": "n=0; residual/requirement means NULL; share 0 if position seen else NULL",
                "invents_information": False,
            },
            {
                "case": "historical_but_not_current_version",
                "n_rows": int(historical_not_current.sum()),
                "fraction_rows": (
                    float(historical_not_current.mean())
                    if len(explicit)
                    else float("nan")
                ),
                "behaviour": "current_version mean NULL; expanding/recent remain available as comparators",
                "invents_information": False,
            },
            {
                "case": "sparse_current_version",
                "n_rows": int(sparse_current.sum()),
                "fraction_rows": (
                    float(sparse_current.mean()) if len(explicit) else float("nan")
                ),
                "behaviour": "keep the sparse current-version estimate; do not blend until freeze evidence",
                "invents_information": False,
            },
            {
                "case": "no_recent_90d",
                "n_rows": int((recent_n == 0).sum()),
                "fraction_rows": float((recent_n == 0).mean())
                if len(explicit)
                else float("nan"),
                "behaviour": "recent_90d mean NULL; expanding remains the long-run identity",
                "invents_information": False,
            },
        ]
    )


def _history_size_table(frame: pd.DataFrame) -> pd.DataFrame:
    explicit = frame.loc[explicit_position_mask(frame)]
    rows: list[dict[str, object]] = []
    for spec in WINDOW_SPECS:
        n = _numeric(explicit[f"hp_{spec.name}_n"])
        buckets = n.map(lambda value: history_n_bucket(float(value)))
        counts = buckets.value_counts()
        for label, _low, _high in HISTORY_N_BUCKETS:
            rows.append(
                {
                    "estimator": spec.name,
                    "bucket": label,
                    "n_rows": int(counts.get(label, 0)),
                    "fraction_rows": (
                        float(counts.get(label, 0) / len(explicit))
                        if len(explicit)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _residual_persistence_table(
    frame: pd.DataFrame, *, split: str, min_n: int = MIN_COMPARE_N
) -> pd.DataFrame:
    actual = frame[ELO_RESIDUAL_COLUMN]
    rows: list[dict[str, object]] = []
    zero = _zero_predictor(actual)
    rows.append({"estimator": "zero", "split": split, "min_n": 0, **zero})
    for spec in WINDOW_SPECS:
        n = _numeric(frame[f"hp_{spec.name}_elo_residual_n"])
        predicted = frame[f"hp_{spec.name}_elo_residual_mean"]
        mask = n >= min_n
        metrics = _prediction_metrics(actual[mask], predicted[mask])
        rows.append(
            {
                "estimator": spec.name,
                "split": split,
                "min_n": min_n,
                **metrics,
            }
        )
        any_n = n > 0
        metrics_any = _prediction_metrics(actual[any_n], predicted[any_n])
        rows.append(
            {
                "estimator": spec.name,
                "split": split,
                "min_n": 1,
                **metrics_any,
            }
        )
    return pd.DataFrame(rows)


def _usage_block_persistence(frame: pd.DataFrame) -> pd.DataFrame:
    """Predict next chronological-block H×P position-share from prior state."""
    work = frame.loc[explicit_position_mask(frame)].copy()
    if work.empty:
        return pd.DataFrame()
    work["chrono_block"] = assign_chronological_blocks(work)
    keys = _hero_position_keys(work)
    work = work.loc[keys.notna()].copy()
    work["hp_key"] = keys.loc[work.index]
    realized = (
        work.groupby(["chrono_block", "hp_key"], sort=False)
        .size()
        .rename("block_n")
        .reset_index()
    )
    pos_n = (
        work.groupby(["chrono_block", "position_number"], sort=False)
        .size()
        .rename("block_pos_n")
        .reset_index()
    )
    realized["position_number"] = [key[1] for key in realized["hp_key"]]
    realized = realized.merge(pos_n, on=["chrono_block", "position_number"], how="left")
    realized["block_share"] = realized["block_n"] / realized["block_pos_n"]
    first = (
        work.sort_values(["chrono_block", "start_time"], kind="mergesort")
        .groupby(["chrono_block", "hp_key"], sort=False)
        .head(1)
    )
    first = first.merge(
        realized.loc[:, ["chrono_block", "hp_key", "block_share", "block_n"]],
        on=["chrono_block", "hp_key"],
        how="left",
    )
    rows: list[dict[str, object]] = []
    overall_actual: dict[str, list[pd.Series]] = {
        spec.name: [] for spec in WINDOW_SPECS
    }
    overall_pred: dict[str, list[pd.Series]] = {spec.name: [] for spec in WINDOW_SPECS}
    blocks = sorted(pd.unique(first["chrono_block"].dropna()))
    for left, right in pairwise(blocks):
        a = first.loc[first["chrono_block"] == left]
        b = first.loc[first["chrono_block"] == right]
        joined = a.merge(b, on="hp_key", suffixes=("_left", "_right"))
        eligible = joined.loc[joined["block_n_right"] >= MIN_WINDOW_PROFILE_N]
        if eligible.empty:
            continue
        actual = eligible["block_share_right"]
        for spec in WINDOW_SPECS:
            predicted = eligible[f"hp_{spec.name}_hero_share_at_position_left"]
            metrics = _prediction_metrics(actual, predicted)
            rows.append(
                {
                    "left_block": left,
                    "right_block": right,
                    "estimator": spec.name,
                    "split": "block_pair",
                    "min_n": MIN_WINDOW_PROFILE_N,
                    **metrics,
                }
            )
            overall_actual[spec.name].append(actual)
            overall_pred[spec.name].append(predicted)
    for spec in WINDOW_SPECS:
        if not overall_actual[spec.name]:
            continue
        actual = pd.concat(overall_actual[spec.name], ignore_index=True)
        predicted = pd.concat(overall_pred[spec.name], ignore_index=True)
        metrics = _prediction_metrics(actual, predicted)
        rows.append(
            {
                "left_block": "all",
                "right_block": "all",
                "estimator": spec.name,
                "split": "validation",
                "min_n": MIN_COMPARE_N,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _version_transfer_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Does version-V residual state predict version V+1 observations?"""
    work = frame.loc[explicit_position_mask(frame)].copy()
    if work.empty or "game_version_id" not in work.columns:
        return pd.DataFrame()
    work["game_version_id"] = _numeric(work["game_version_id"])
    first_seen = (
        work.groupby("game_version_id", dropna=True)["start_time"].min().sort_values()
    )
    ordered = [int(value) for value in first_seen.index]
    rows: list[dict[str, object]] = []
    for previous, current in pairwise(ordered):
        later = work.loc[work["game_version_id"] == current]
        if later.empty:
            continue
        actual = later[ELO_RESIDUAL_COLUMN]
        prev_mean = later["hp_current_plus_previous_elo_residual_mean"]
        expanding = later["hp_expanding_elo_residual_mean"]
        current_n = _numeric(later["hp_current_version_elo_residual_n"])
        # Early in a new version, current-version n is small; previous-version
        # mass lives in current_plus_previous while current_version is empty.
        new_version = current_n < MIN_COMPARE_N
        for label, predicted, mask in (
            ("expanding_all", expanding, pd.Series(True, index=later.index)),
            (
                "current_plus_previous_all",
                prev_mean,
                pd.Series(True, index=later.index),
            ),
            ("expanding_new_version_rows", expanding, new_version),
            ("current_plus_previous_new_version_rows", prev_mean, new_version),
        ):
            metrics = _prediction_metrics(actual[mask], predicted[mask])
            rows.append(
                {
                    "from_version": previous,
                    "to_version": current,
                    "estimator": label,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _same_version_persistence(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[explicit_position_mask(frame)].copy()
    if work.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for version, group in work.groupby("game_version_id", dropna=True):
        actual = group[ELO_RESIDUAL_COLUMN]
        n = _numeric(group["hp_current_version_elo_residual_n"])
        mask = n >= MIN_COMPARE_N
        current = _prediction_metrics(
            actual[mask], group.loc[mask, "hp_current_version_elo_residual_mean"]
        )
        expanding = _prediction_metrics(
            actual[mask], group.loc[mask, "hp_expanding_elo_residual_mean"]
        )
        rows.append(
            {
                "game_version_id": version,
                "estimator": "current_version",
                **current,
            }
        )
        rows.append(
            {
                "game_version_id": version,
                "estimator": "expanding",
                **expanding,
            }
        )
    return pd.DataFrame(rows)


def _requirement_drift_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = (
        ("farming_mean", "farming_n", MATERIAL_FARMING_SHIFT, CAUSAL_B_COLUMN),
        ("combat_mean", "combat_n", MATERIAL_COMBAT_SHIFT, CAUSAL_C_COLUMN),
        (
            "hero_share_at_position",
            "n",
            MATERIAL_USAGE_SHARE_SHIFT,
            "hero_share_at_position",
        ),
        (
            "elo_residual_mean",
            "elo_residual_n",
            MATERIAL_RESIDUAL_SHIFT,
            ELO_RESIDUAL_COLUMN,
        ),
    )
    comparators = [spec.name for spec in WINDOW_SPECS if spec.name != "expanding"]
    for mean_suffix, n_suffix, material, family in pairs:
        for name in comparators:
            rows.append(
                _compare_estimators(
                    frame,
                    left_name="expanding",
                    right_name=name,
                    mean_suffix=mean_suffix,
                    n_suffix=n_suffix,
                    material_shift=material,
                )
                | {"observation": family}
            )
    # Persistence of recent/version requirement vs expanding on next B/C.
    for target, mean_suffix, n_suffix in (
        (CAUSAL_B_COLUMN, "farming_mean", "farming_n"),
        (CAUSAL_C_COLUMN, "combat_mean", "combat_n"),
    ):
        if target not in frame.columns:
            continue
        actual = frame[target]
        for spec in WINDOW_SPECS:
            n = _numeric(frame[f"hp_{spec.name}_{n_suffix}"])
            predicted = frame[f"hp_{spec.name}_{mean_suffix}"]
            mask = n >= MIN_COMPARE_N
            metrics = _prediction_metrics(actual[mask], predicted[mask])
            rows.append(
                {
                    "left": "next_observation",
                    "right": spec.name,
                    "family": mean_suffix,
                    "observation": target,
                    "n_paired": metrics["n"],
                    "min_n": MIN_COMPARE_N,
                    "pearson": metrics["pearson"],
                    "spearman": metrics["spearman"],
                    "mean_abs_change": metrics["mae"],
                    "median_abs_change": float("nan"),
                    "fraction_material_change": float("nan"),
                    "material_shift": float("nan"),
                    "mean_abs_rank_change": float("nan"),
                    "fraction_rank_change_ge_10": float("nan"),
                    "fraction_sign_change": float("nan"),
                    "rmse": metrics["rmse"],
                    "kind": "predict_next_requirement",
                }
            )
    return pd.DataFrame(rows)


def _residual_shrinkage_grid(
    frame: pd.DataFrame, *, split: str, ks: tuple[float, ...] = SHRINKAGE_GRID
) -> pd.DataFrame:
    actual = frame[ELO_RESIDUAL_COLUMN]
    mean = frame["hp_expanding_elo_residual_mean"]
    n = frame["hp_expanding_elo_residual_n"]
    rows: list[dict[str, object]] = []
    for k in ks:
        shrunk, _weight = apply_farming_shrinkage(mean, n, k=k)
        observed = _numeric(n) > 0
        metrics = _prediction_metrics(actual[observed], shrunk[observed])
        rows.append({"k": k, "split": split, "subset": "n_gt_0", **metrics})
        established = _numeric(n) >= 20
        metrics_est = _prediction_metrics(actual[established], shrunk[established])
        rows.append({"k": k, "split": split, "subset": "n_ge_20", **metrics_est})
        sparse = (_numeric(n) > 0) & (_numeric(n) < MIN_COMPARE_N)
        metrics_sparse = _prediction_metrics(actual[sparse], shrunk[sparse])
        rows.append({"k": k, "split": split, "subset": "n_1_to_4", **metrics_sparse})
    return pd.DataFrame(rows)


def _select_residual_k(grid: pd.DataFrame) -> tuple[float, str]:
    subset = grid.loc[grid["subset"] == "n_gt_0"]
    if subset.empty:
        return 0.0, "no residual rows; k=0 (unshrunk / NULL at n=0)"
    rmse = pd.to_numeric(subset["rmse"], errors="coerce")
    finite = subset.loc[rmse.notna() & np.isfinite(rmse.to_numpy())]
    if finite.empty:
        return 0.0, "residual RMSE undefined; k=0"
    best = float(finite["rmse"].min())
    equivalent = finite.loc[
        pd.to_numeric(finite["rmse"], errors="coerce") <= best * EQUIVALENT_RMSE_RATIO
    ]
    # Prefer k=0 among equivalents: shrinkage is only justified by a real gain
    # at predicting the next residual, not by a win model.
    if 0.0 in set(pd.to_numeric(equivalent["k"], errors="coerce").tolist()):
        return 0.0, (
            "k=0 is equivalent to the best residual RMSE on tune; "
            "do not shrink toward 0 without a persistence gain"
        )
    chosen = float(equivalent.sort_values("k").iloc[0]["k"])
    return chosen, (
        f"smallest k within {EQUIVALENT_RMSE_RATIO:g} of best next-residual "
        f"RMSE ({best:.6f}) on tune; not evaluated on match outcomes"
    )


def _select_recent_window(
    persistence: pd.DataFrame, coverage: pd.DataFrame
) -> tuple[str, str]:
    """Diagnostic window choice on development/tune residual persistence."""
    tune = persistence.loc[
        (persistence["split"] == "tune")
        & (persistence["min_n"] == MIN_COMPARE_N)
        & (persistence["estimator"].isin(["recent_90d", "recent_180d", "expanding"]))
    ]
    cov = coverage.set_index("estimator")
    frac_90 = (
        float(cov.loc["recent_90d", "fraction_n_eq_0"])
        if "recent_90d" in cov.index
        else 1.0
    )
    justification_base = (
        f"{RECENT_WINDOW_DAYS_PRIMARY}d is the a priori recent window "
        "(hero_meta). 180d is LAST_180D robustness, not a win-model search."
    )
    if tune.empty:
        return "recent_90d", justification_base + " Tune persistence empty; keep 90d."
    by_name = tune.set_index("estimator")
    rmse_90 = (
        float(by_name.loc["recent_90d", "rmse"])
        if "recent_90d" in by_name.index
        else float("nan")
    )
    rmse_180 = (
        float(by_name.loc["recent_180d", "rmse"])
        if "recent_180d" in by_name.index
        else float("nan")
    )
    if (
        frac_90 > 0.80
        and np.isfinite(rmse_180)
        and (not np.isfinite(rmse_90) or rmse_180 * EQUIVALENT_RMSE_RATIO < rmse_90)
    ):
        return "recent_180d", (
            justification_base
            + f" 90d cold-start fraction={frac_90:.3f}; 180d has better "
            "next-residual RMSE on tune."
        )
    return "recent_90d", justification_base + " Keep 90d; 180d is diagnostic-only."


def _regression_to_mean_table(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[explicit_position_mask(frame)].copy()
    keys = _hero_position_keys(work)
    work = work.loc[keys.notna()].copy()
    work["hp_key"] = keys.loc[work.index]
    work["residual"] = _numeric(work[ELO_RESIDUAL_COLUMN])
    work = work.loc[work["residual"].notna()]
    if work.empty:
        return pd.DataFrame()
    work = work.sort_values(["hp_key", "start_time"], kind="mergesort")
    work["order"] = work.groupby("hp_key", sort=False).cumcount()
    sizes = work.groupby("hp_key", sort=False)["residual"].transform("size")
    first = work.loc[work["order"] < (sizes / 2.0)]
    second = work.loc[work["order"] >= (sizes / 2.0)]
    a = first.groupby("hp_key")["residual"].mean()
    b = second.groupby("hp_key")["residual"].mean()
    n_a = first.groupby("hp_key").size()
    n_b = second.groupby("hp_key").size()
    joined = pd.DataFrame({"first": a, "second": b, "n_first": n_a, "n_second": n_b})
    joined = joined.loc[
        (joined["n_first"] >= RTM_MIN_EACH) & (joined["n_second"] >= RTM_MIN_EACH)
    ]
    if joined.empty:
        return pd.DataFrame()
    q = float(joined["first"].abs().quantile(RTM_EXTREME_QUANTILE))
    extreme = joined.loc[joined["first"].abs() >= q]
    return pd.DataFrame(
        [
            {
                "scope": "all_paired_hp",
                "n_hp": len(joined),
                "pearson": _pearson(joined["first"], joined["second"]),
                "slope": slope_coefficient(joined["second"], joined["first"]),
                "mean_first": float(joined["first"].mean()),
                "mean_second": float(joined["second"].mean()),
            },
            {
                "scope": f"extreme_abs_q{RTM_EXTREME_QUANTILE:.2f}",
                "n_hp": len(extreme),
                "pearson": _pearson(extreme["first"], extreme["second"])
                if len(extreme)
                else float("nan"),
                "slope": (
                    slope_coefficient(extreme["second"], extreme["first"])
                    if len(extreme)
                    else float("nan")
                ),
                "mean_first": float(extreme["first"].mean())
                if len(extreme)
                else float("nan"),
                "mean_second": float(extreme["second"].mean())
                if len(extreme)
                else float("nan"),
            },
        ]
    )


def _sample_size_table(frame: pd.DataFrame) -> pd.DataFrame:
    actual = frame[ELO_RESIDUAL_COLUMN]
    rows: list[dict[str, object]] = []
    for spec in WINDOW_SPECS:
        n = _numeric(frame[f"hp_{spec.name}_elo_residual_n"])
        predicted = frame[f"hp_{spec.name}_elo_residual_mean"]
        buckets = n.map(lambda value: history_n_bucket(float(value)))
        for label, _low, _high in HISTORY_N_BUCKETS:
            mask = buckets == label
            metrics = _prediction_metrics(actual[mask], predicted[mask])
            rows.append({"estimator": spec.name, "bucket": label, **metrics})
    return pd.DataFrame(rows)


def _metric(
    table: pd.DataFrame, estimator: str, column: str, **filters: object
) -> float:
    subset = table
    if "estimator" in subset.columns:
        subset = subset.loc[subset["estimator"] == estimator]
    for key, value in filters.items():
        if key in subset.columns:
            subset = subset.loc[subset[key] == value]
    if subset.empty or column not in subset.columns:
        return float("nan")
    return float(pd.to_numeric(subset.iloc[0][column], errors="coerce"))


def _family_gate(
    *,
    name: str,
    comparison: pd.DataFrame,
    persistence: pd.DataFrame,
    expanding_rmse: float,
    candidate: str,
    variation_family: str,
    min_variation_change: float,
) -> dict[str, object]:
    varied = comparison.loc[
        (comparison["left"] == "expanding")
        & (comparison["right"] == candidate)
        & (comparison["family"] == variation_family)
    ]
    corr = float(varied.iloc[0]["pearson"]) if not varied.empty else float("nan")
    abs_change = (
        float(varied.iloc[0]["mean_abs_change"]) if not varied.empty else float("nan")
    )
    frac_material = (
        float(varied.iloc[0]["fraction_material_change"])
        if not varied.empty
        else float("nan")
    )
    sign_change = (
        float(varied.iloc[0]["fraction_sign_change"])
        if not varied.empty
        else float("nan")
    )
    cand_rmse = _metric(
        persistence, candidate, "rmse", split="validation", min_n=MIN_COMPARE_N
    )
    cand_pearson = _metric(
        persistence, candidate, "pearson", split="validation", min_n=MIN_COMPARE_N
    )
    exp_pearson = _metric(
        persistence, "expanding", "pearson", split="validation", min_n=MIN_COMPARE_N
    )
    variation = False
    if np.isfinite(corr) and corr < _VARIATION_CORR:
        variation = True
    if np.isfinite(abs_change) and abs_change >= min_variation_change:
        variation = True
    if np.isfinite(frac_material) and frac_material >= _SHARE_COMPARE_FLOOR:
        variation = True
    persistence_gain = False
    if (
        np.isfinite(cand_rmse)
        and np.isfinite(expanding_rmse)
        and cand_rmse * EQUIVALENT_RMSE_RATIO < expanding_rmse
    ):
        persistence_gain = True
    if (
        np.isfinite(cand_pearson)
        and np.isfinite(exp_pearson)
        and cand_pearson >= exp_pearson + _PERSISTENCE_PEARSON_DELTA
        and (not np.isfinite(cand_rmse) or cand_rmse <= expanding_rmse)
    ):
        persistence_gain = True
    if variation and persistence_gain:
        grade = "A"
        rationale = (
            f"{name}: {candidate} differs from long-run "
            f"(r={corr:.3f}, abs={abs_change:.4f}) and predicts the next "
            f"observation better than expanding (RMSE {cand_rmse:.4f} vs "
            f"{expanding_rmse:.4f})."
        )
    elif variation and not persistence_gain:
        grade = "C"
        rationale = (
            f"{name}: {candidate} moves relative to long-run "
            f"(r={corr:.3f}) but does not persist (RMSE {cand_rmse:.4f} vs "
            f"expanding {expanding_rmse:.4f}). Treat as noise."
        )
    elif persistence_gain and not variation:
        grade = "C"
        rationale = (
            f"{name}: {candidate} does not materially differ from long-run "
            f"(r={corr:.3f}); current-meta is not adding a distinct state."
        )
    else:
        grade = "C"
        rationale = (
            f"{name}: neither material temporal variation nor persistence "
            f"beyond long-run H×P history."
        )
    # Partial: variation is real and persistence is close but not decisive.
    if (
        grade == "C"
        and variation
        and np.isfinite(cand_rmse)
        and np.isfinite(expanding_rmse)
        and cand_rmse <= expanding_rmse * EQUIVALENT_RMSE_RATIO
    ):
        grade = "B"
        rationale = (
            f"{name}: {candidate} varies vs long-run (r={corr:.3f}) and is "
            "only equivalent, not clearly better, at next-observation RMSE. "
            "Suggestive; do not freeze a composite score."
        )
    return {
        "family": name,
        "grade": grade,
        "candidate": candidate,
        "rationale": rationale,
        "variation": variation,
        "persistence_gain": persistence_gain,
        "pearson_vs_expanding": corr,
        "mean_abs_change": abs_change,
        "fraction_material_change": frac_material,
        "fraction_sign_change": sign_change,
        "candidate_rmse": cand_rmse,
        "expanding_rmse": expanding_rmse,
        "candidate_pearson": cand_pearson,
        "expanding_pearson": exp_pearson,
    }


def classify_slice24(
    *,
    usage_gate: dict[str, object],
    residual_gate: dict[str, object],
    drift_gate: dict[str, object],
    selected_recent_window: str,
) -> pd.DataFrame:
    """Map family gates onto one Slice 24 classification."""
    grades = {
        "usage": str(usage_gate["grade"]),
        "elo_residual": str(residual_gate["grade"]),
        "requirement_drift": str(drift_gate["grade"]),
    }
    frozen: list[str] = []
    for family, grade in grades.items():
        if grade == "A":
            frozen.append(family)
    a_count = sum(grade == "A" for grade in grades.values())
    c_count = sum(grade == "C" for grade in grades.values())
    if a_count == 3:
        classification = "A"
        gate = CLASSIFICATION_A
        next_slice = (
            "Freeze the supported causal current-meta H×P families for "
            "downstream draft work. Do not build synergy/counters or a "
            "win-model benchmark in the next slice."
        )
    elif a_count == 0 and c_count == 3:
        classification = "C"
        gate = CLASSIFICATION_C
        next_slice = (
            "Do not freeze current-meta H×P state. Keep these diagnostics "
            "as evidence and proceed without a time-varying H×P layer."
        )
        frozen = []
    else:
        classification = "B"
        gate = CLASSIFICATION_B
        next_slice = (
            "Freeze only the supported family or families, if any. Do not "
            "create a broad composite meta-strength score. Requirement "
            "profiles remain the frozen Slice 22 long-run states."
        )
        frozen = [family for family, grade in grades.items() if grade == "A"]
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "gate": gate,
                "usage_grade": grades["usage"],
                "residual_grade": grades["elo_residual"],
                "drift_grade": grades["requirement_drift"],
                "selected_recent_window": selected_recent_window,
                "frozen_components": tuple(frozen),
                "usage_rationale": usage_gate["rationale"],
                "residual_rationale": residual_gate["rationale"],
                "drift_rationale": drift_gate["rationale"],
                "next_slice": next_slice,
                "fallback_hierarchy_frozen": False,
                "composite_meta_score": False,
            }
        ]
    )


def _semantics() -> dict[str, object]:
    return {
        "key": "hero_id × explicit position 1–5",
        "history_filter": "start_time < T; same explicit position observations only",
        "same_timestamp": "mutually blind",
        "current_match_result_in_state": False,
        "raw_win_rate_primary": False,
        "elo_residual": "team_won - pre-match Elo expected win",
        "leave_player_out": False,
        "reason_not_lpo": "environmental meta, not Slice 22 identity",
        "n_0_mean": "NULL",
        "n_0_share_if_denominator_positive": 0.0,
        "fallback_hierarchy": "not frozen",
        "windows": [
            {
                "name": spec.name,
                "window_days": spec.window_days,
                "version_mode": spec.version_mode,
                "justification": spec.justification,
            }
            for spec in WINDOW_SPECS
        ],
        "current_position": (
            "diagnostic realized post-match position; not a PRE_DRAFT feature"
        ),
        "slice22_overwritten": False,
        "player_hero_fit_constructed": False,
        "win_model_run": False,
    }


def run_hero_position_meta_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice24DiagnosticReport:
    """Development-only Slice 24 current-meta research. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    development = attach_hero_profile_observations(development)
    development = attach_hero_requirement_state(development)
    development = attach_hero_position_meta_state(development)

    tune_end = development_tune_end(development["start_time"], development_end=end)
    dev_times = pd.to_datetime(development["start_time"], utc=True)
    tune_mask = dev_times <= pd.Timestamp(tune_end)
    val_mask = (dev_times > pd.Timestamp(tune_end)) & (dev_times <= pd.Timestamp(end))
    tune = development.loc[tune_mask].copy()
    validation = development.loc[val_mask].copy()

    split = pd.DataFrame(
        [
            {
                "split": "tune",
                "n_rows": int(tune_mask.sum()),
                "n_matches": int(tune["match_id"].nunique()) if len(tune) else 0,
                "start": tune["start_time"].min() if len(tune) else pd.NaT,
                "end": tune["start_time"].max() if len(tune) else pd.NaT,
            },
            {
                "split": "validation",
                "n_rows": int(val_mask.sum()),
                "n_matches": int(validation["match_id"].nunique())
                if len(validation)
                else 0,
                "start": validation["start_time"].min() if len(validation) else pd.NaT,
                "end": validation["start_time"].max() if len(validation) else pd.NaT,
            },
        ]
    )
    coverage = pd.DataFrame([_coverage_row(development, spec) for spec in WINDOW_SPECS])
    cold_start = _cold_start_table(development)
    history_size = _history_size_table(development)
    estimator_comparison = pd.DataFrame(
        [
            _compare_estimators(
                development,
                left_name="expanding",
                right_name=spec.name,
                mean_suffix=mean_suffix,
                n_suffix=n_suffix,
                material_shift=material,
            )
            for spec in WINDOW_SPECS
            if spec.name != "expanding"
            for mean_suffix, n_suffix, material in (
                ("hero_share_at_position", "n", MATERIAL_USAGE_SHARE_SHIFT),
                ("elo_residual_mean", "elo_residual_n", MATERIAL_RESIDUAL_SHIFT),
                ("farming_mean", "farming_n", MATERIAL_FARMING_SHIFT),
                ("combat_mean", "combat_n", MATERIAL_COMBAT_SHIFT),
            )
        ]
    )
    residual_persistence = pd.concat(
        [
            _residual_persistence_table(tune, split="tune"),
            _residual_persistence_table(validation, split="validation"),
        ],
        ignore_index=True,
    )
    usage_persistence = _usage_block_persistence(development)
    version_transfer = _version_transfer_table(development)
    same_version = _same_version_persistence(development)
    requirement_drift = _requirement_drift_table(development)
    residual_grid_tune = _residual_shrinkage_grid(tune, split="tune")
    residual_k, residual_why = _select_residual_k(residual_grid_tune)
    residual_grid_val = _residual_shrinkage_grid(validation, split="validation")
    selected_window, window_why = _select_recent_window(residual_persistence, coverage)
    rtm = _regression_to_mean_table(development)
    sample_size = _sample_size_table(development)

    expanding_rmse = _metric(
        residual_persistence,
        "expanding",
        "rmse",
        split="validation",
        min_n=MIN_COMPARE_N,
    )
    usage_persistence_overall = (
        usage_persistence.loc[usage_persistence["split"] == "validation"]
        if not usage_persistence.empty and "split" in usage_persistence.columns
        else pd.DataFrame()
    )
    usage_rmse_expanding = _metric(
        usage_persistence_overall, "expanding", "rmse", split="validation"
    )
    usage_gate = _family_gate(
        name="usage",
        comparison=estimator_comparison,
        persistence=usage_persistence_overall
        if not usage_persistence_overall.empty
        else pd.DataFrame(
            {
                "estimator": ["expanding", selected_window],
                "split": ["validation", "validation"],
                "min_n": [MIN_COMPARE_N, MIN_COMPARE_N],
                "rmse": [usage_rmse_expanding, float("nan")],
                "pearson": [float("nan"), float("nan")],
            }
        ),
        expanding_rmse=usage_rmse_expanding,
        candidate=selected_window,
        variation_family="hero_share_at_position",
        min_variation_change=MATERIAL_USAGE_SHARE_SHIFT,
    )
    residual_gate = _family_gate(
        name="elo_residual",
        comparison=estimator_comparison,
        persistence=residual_persistence,
        expanding_rmse=expanding_rmse,
        candidate=selected_window,
        variation_family="elo_residual_mean",
        min_variation_change=MATERIAL_RESIDUAL_SHIFT,
    )
    # Current-version residual may be the better current-meta candidate.
    version_gate = _family_gate(
        name="elo_residual_current_version",
        comparison=estimator_comparison,
        persistence=residual_persistence,
        expanding_rmse=expanding_rmse,
        candidate="current_version",
        variation_family="elo_residual_mean",
        min_variation_change=MATERIAL_RESIDUAL_SHIFT,
    )
    if str(version_gate["grade"]) == "A" and str(residual_gate["grade"]) != "A":
        residual_gate = version_gate
        residual_gate["family"] = "elo_residual"
    drift_persistence = requirement_drift
    if "kind" in requirement_drift.columns:
        drift_persistence = requirement_drift.loc[
            requirement_drift["kind"] == "predict_next_requirement"
        ].copy()
        if "right" in drift_persistence.columns:
            drift_persistence = drift_persistence.rename(columns={"right": "estimator"})
            drift_persistence["split"] = "validation"
            drift_persistence["min_n"] = MIN_COMPARE_N
    farm_expanding = (
        _metric(drift_persistence, "expanding", "rmse")
        if not drift_persistence.empty
        else float("nan")
    )
    drift_gate = _family_gate(
        name="requirement_drift",
        comparison=estimator_comparison,
        persistence=drift_persistence
        if not drift_persistence.empty
        else pd.DataFrame(
            {
                "estimator": ["expanding", selected_window],
                "split": ["validation", "validation"],
                "min_n": [MIN_COMPARE_N, MIN_COMPARE_N],
                "rmse": [farm_expanding, float("nan")],
                "pearson": [float("nan"), float("nan")],
            }
        ),
        expanding_rmse=farm_expanding if np.isfinite(farm_expanding) else 0.0,
        candidate=selected_window,
        variation_family="farming_mean",
        min_variation_change=MATERIAL_FARMING_SHIFT,
    )

    classification = classify_slice24(
        usage_gate=usage_gate,
        residual_gate=residual_gate,
        drift_gate=drift_gate,
        selected_recent_window=selected_window,
    )
    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    recorded_frozen = tuple(classification.iloc[0]["frozen_components"])
    integrity = {
        "development_end": end.isoformat(),
        "tune_end": tune_end.isoformat(),
        "preferred_tune_end": PREFERRED_TUNE_END.isoformat(),
        "tune_end_matches_preferred": tune_end == PREFERRED_TUNE_END,
        "holdout_used_for_window_selection": False,
        "holdout_used_for_validation": False,
        "holdout_used_for_shrinkage": False,
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
        "hero_farm_k_is_2": FROZEN_HERO_FARM_SHRINKAGE_K == 2.0,
        "hero_combat_k_is_2": FROZEN_HERO_COMBAT_SHRINKAGE_K == 2.0,
        "slice22_state_columns_unchanged": list(SLICE22_STATE_COLUMNS)
        == [
            "hero_farming_prior_n",
            "hero_farming_prior_sum_b",
            "hero_farming_prior_mean_b",
            "hero_farming_shrunk_b",
            "hero_combat_prior_n",
            "hero_combat_prior_sum_c",
            "hero_combat_prior_mean_c",
            "hero_combat_shrunk_c",
        ],
        "slice23_fit_score_frozen": SLICE23_FIT_SCORE_FROZEN,
        "slice23_diagnostic_only": SLICE23_DIAGNOSTIC_ONLY,
        "player_hero_fit_created": False,
        "compatibility_score_revived": False,
        "current_position_resolved": False,
        "team_feature_created": False,
        "synergy_or_counter_created": False,
        "draft_probability_created": False,
        "win_model_run": False,
        "elo_changed": False,
        "raw_win_rate_primary": False,
        "composite_meta_score": False,
        "fallback_hierarchy_frozen": False,
        "residual_shrinkage_k_frozen": SLICE24_RESIDUAL_SHRINKAGE_K_FROZEN,
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "meta_in_feature_columns": any(
            name in FEATURE_COLUMNS
            for name in (
                *SLICE24_STATE_COLUMNS,
                *PLAYER_X_HERO_FIT_NAMES,
                ELO_RESIDUAL_COLUMN,
            )
        ),
        "meta_in_snapshot_columns": any(
            name in SNAPSHOT_COLUMNS for name in SLICE24_STATE_COLUMNS
        ),
        "meta_in_pre_draft_sql": any(
            name in PRE_DRAFT_SNAPSHOT_SQL for name in SLICE24_STATE_COLUMNS
        ),
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "match_player_box_score_field_count": len(MATCH_PLAYER_BOX_SCORE_COLUMNS),
        "n_holdout_excluded": len(holdout),
        "model_trained": False,
        "full_development_mean_fallback": False,
        "usage_gate": usage_gate,
        "residual_gate": residual_gate,
        "drift_gate": drift_gate,
        "computed_frozen_components": list(recorded_frozen),
        "recorded_research_classification": SLICE24_RESEARCH_CLASSIFICATION,
        "recorded_frozen_components": list(SLICE24_FROZEN_COMPONENTS),
        "diagnostic_only": SLICE24_DIAGNOSTIC_ONLY,
    }
    return Slice24DiagnosticReport(
        development_end=end,
        tune_end=tune_end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        selected_recent_window=selected_window,
        selected_recent_window_justification=window_why,
        residual_shrinkage_k=residual_k,
        residual_shrinkage_justification=residual_why,
        semantics=_semantics(),
        classification=classification,
        split=split,
        coverage=coverage,
        cold_start=cold_start,
        history_size=history_size,
        estimator_comparison=estimator_comparison,
        residual_persistence=residual_persistence,
        usage_persistence=usage_persistence,
        version_transfer=version_transfer,
        same_version_persistence=same_version,
        requirement_drift=requirement_drift,
        residual_shrinkage_tune=residual_grid_tune,
        residual_shrinkage_validation=residual_grid_val,
        regression_to_mean=rtm,
        sample_size=sample_size,
        integrity=integrity,
    )


def slice24_report_to_jsonable(report: Slice24DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 24 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "tune_end": report.tune_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "selected_recent_window": report.selected_recent_window,
        "selected_recent_window_justification": (
            report.selected_recent_window_justification
        ),
        "residual_shrinkage_k": report.residual_shrinkage_k,
        "residual_shrinkage_justification": report.residual_shrinkage_justification,
        "recorded_classification": SLICE24_RESEARCH_CLASSIFICATION,
        "recorded_frozen_components": list(SLICE24_FROZEN_COMPONENTS),
        "diagnostic_only": SLICE24_DIAGNOSTIC_ONLY,
        "residual_shrinkage_k_frozen": SLICE24_RESIDUAL_SHRINKAGE_K_FROZEN,
        "semantics": _jsonable_value(report.semantics),
        "classification": _jsonable_value(report.classification),
        "split": _jsonable_value(report.split),
        "coverage": _jsonable_value(report.coverage),
        "cold_start": _jsonable_value(report.cold_start),
        "history_size": _jsonable_value(report.history_size),
        "estimator_comparison": _jsonable_value(report.estimator_comparison),
        "residual_persistence": _jsonable_value(report.residual_persistence),
        "usage_persistence": _jsonable_value(report.usage_persistence),
        "version_transfer": _jsonable_value(report.version_transfer),
        "same_version_persistence": _jsonable_value(report.same_version_persistence),
        "requirement_drift": _jsonable_value(report.requirement_drift),
        "residual_shrinkage_tune": _jsonable_value(report.residual_shrinkage_tune),
        "residual_shrinkage_validation": _jsonable_value(
            report.residual_shrinkage_validation
        ),
        "regression_to_mean": _jsonable_value(report.regression_to_mean),
        "sample_size": _jsonable_value(report.sample_size),
        "integrity": _jsonable_value(report.integrity),
        "state_columns": list(SLICE24_STATE_COLUMNS),
    }
