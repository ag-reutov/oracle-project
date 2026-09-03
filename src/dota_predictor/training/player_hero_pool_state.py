"""Slice 25: causal Player × Position hero-pool state.

Research only. Availability / hero-pool identity — not player skill.
Does not add production features, does not train a win model, and does
not implement Slice 26 draft assignment / flex logic.

Question
--------
At time ``T``, conditional on player ``P`` playing explicit position
``R``, what distribution over heroes is supported by that player's
strictly prior history?

    (P, R, T) -> { n(P,R,H), pi(H | P,R,T) }

No win rate, Elo residual, farming B, combat C, Slice 23 compatibility,
or Slice 24 H×P outcome state enters the pool definition.

Temporal integrity
------------------
History for match ``M`` uses only rows with ``start_time < M.start_time``.
Equal timestamps are mutually blind. Causality is never ordered by
``match_id``. The current match's hero, observed position, result,
duration, and box score do not update that row's state.

Current observed explicit position is the **validation condition**
(which role pool to score) and is not an input to construction.
All five role pools are built independently from historical explicit
positions.

Explicit positions only
-----------------------
Only historical POSITION_1–5 contribute to P×R×H counts. NULL /
UNKNOWN / FILTERED / ALL and lobby ``slot_in_side`` never enter role
buckets. ``expected_position`` is not used to build counts.

Scoring vs state
----------------
Raw empirical shares may be zero for a first-time hero. Proper scoring
uses one **common causal candidate universe** ``C_T`` per evaluation row
(independent of the estimator) and one fixed mixture wrapper:

    q = (1 - epsilon) * p + epsilon * U(C_T)

with fixed a-priori ``SCORING_MIXTURE_EPSILON``. That wrapper is **not**
part of the frozen pool state and is not tuned against validation or
match outcomes. Every estimator produces a probability vector over the
same ``C_T``. Multiclass Brier is reported on the **raw** ``p`` over
``C_T`` (no mixture). Log-loss uses ``q``.

Candidate universe
------------------
``C_T`` is the set of heroes observed in the professional development
frame with ``start_time < T``, plus the realized hero when it is a
genuine first observation (added identically for every estimator).
Estimator-specific positive support is never the normalization universe.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from dota_predictor.data.canonical_schema import MATCH_PLAYER_BOX_SCORE_COLUMNS
from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.hero_meta import RECENT_WINDOW_DAYS
from dota_predictor.features.player_position import RECENT_POSITION_WINDOWS
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.features.team_elo import DEFAULT_ELO_CONFIG, EloConfig
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.hero_position_meta_state import (
    RECENT_WINDOW_DAYS_ALT,
    causal_previous_version_id,
)
from dota_predictor.training.player_farming_state import (
    HISTORY_N_BUCKETS,
    development_tune_end,
    history_n_bucket,
)
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    EXPLICIT_POSITION_NUMBERS,
    _jsonable_value,
    _numeric,
    build_player_performance_frame,
    explicit_position_mask,
    restrict_development,
)
from dota_predictor.training.metrics import bootstrap_mean_ci
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    utc_datetime,
)

__all__ = [
    "CLASSIFICATION_A",
    "CLASSIFICATION_B",
    "CLASSIFICATION_C",
    "HIERARCHICAL_K_GRID",
    "POOL_WINDOW_SPECS",
    "SCORING_MIXTURE_EPSILON",
    "SLICE25_BOOTSTRAP_RESAMPLES",
    "SLICE25_BOOTSTRAP_SEED",
    "SLICE25_DIAGNOSTIC_ONLY",
    "SLICE25_FROZEN_COMPONENTS",
    "SLICE25_RESEARCH_CLASSIFICATION",
    "SLICE25_STATE_COLUMNS",
    "Slice25DiagnosticReport",
    "WindowSpec",
    "attach_player_hero_pool_state",
    "classify_slice25",
    "effective_pool_size",
    "pool_entropy",
    "run_player_hero_pool_diagnostics",
    "score_distribution",
    "scoring_candidates",
    "select_hierarchical_k",
    "slice25_report_to_jsonable",
]


# Fixed a-priori mixture weight for *scoring only*. Not part of frozen state.
# q = (1 - eps) * p + eps * U(C_T). Never selected on validation or match
# outcomes.
SCORING_MIXTURE_EPSILON = 1e-3
SLICE25_BOOTSTRAP_RESAMPLES = 2000
SLICE25_BOOTSTRAP_SEED = 25

RECENT_WINDOW_DAYS_PRIMARY = RECENT_WINDOW_DAYS
# Same shrinkage grid family as Slice 14; k is chosen on tune next-hero
# log-loss, never on match outcomes.
HIERARCHICAL_K_GRID: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)

VERSION_ALL = "all"
VERSION_CURRENT = "current"
VERSION_CURRENT_PLUS_PREVIOUS = "current_plus_previous"

CLASSIFICATION_A = (
    "A — freeze role-specific player hero pool: expanding P×R×H provides "
    "confirmation-stable next-choice information beyond unconditioned P×H"
)
CLASSIFICATION_B = (
    "B — partial / suggestive: role conditioning or a refinement helps "
    "only in limited regimes; freeze only what is supported"
)
CLASSIFICATION_C = (
    "C — do not freeze: position conditioning fails to outperform "
    "generic P×H history or does not generalize"
)

# Recorded after corrected common-support development diagnostics.
# Classification C: under shared C_T + epsilon mixture scoring, expanding
# P×R×H does not beat unconditioned P×H on log-loss/Brier (match-level
# validation CI for LL(exp)-LL(uncond) is entirely positive). last_5 raises
# hit@1 but collapses LL/Brier/rank. Hierarchical backoff toward
# unconditioned is diagnostic only. Nothing is frozen.
SLICE25_RESEARCH_CLASSIFICATION = "C"
SLICE25_DIAGNOSTIC_ONLY = True
SLICE25_FROZEN_COMPONENTS: tuple[str, ...] = ()

HIT_KS: tuple[int, ...] = (1, 3, 5)
MIN_SCORED_ROWS = 50
MATERIAL_LOGLOSS_DELTA = 0.01
MATERIAL_HIT1_DELTA = 0.01
MIN_POSITIONS_FOR_A = 4
CROSS_POSITION_ATTR = "slice25_cross_position"
FREEZE_ELIGIBLE_WINDOWS: frozenset[str] = frozenset(
    {
        "last_5_at_role",
        "last_10_at_role",
        "last_20_at_role",
        "recent_90d",
        "current_version",
        "current_plus_previous",
    }
)


@dataclass(frozen=True)
class WindowSpec:
    """One causal history filter for candidate B. Not a production feature."""

    name: str
    appearance_window: int | None
    window_days: int | None
    version_mode: str
    justification: str


POOL_WINDOW_SPECS: tuple[WindowSpec, ...] = (
    WindowSpec(
        name="expanding",
        appearance_window=None,
        window_days=None,
        version_mode=VERSION_ALL,
        justification="primary candidate A: expanding role-conditioned history",
    ),
    *(
        WindowSpec(
            name=f"last_{n}_at_role",
            appearance_window=n,
            window_days=None,
            version_mode=VERSION_ALL,
            justification=(
                f"last {n} strictly prior appearances at R "
                f"(player_position.RECENT_POSITION_WINDOWS)"
            ),
        )
        for n in RECENT_POSITION_WINDOWS
    ),
    WindowSpec(
        name="recent_90d",
        appearance_window=None,
        window_days=RECENT_WINDOW_DAYS_PRIMARY,
        version_mode=VERSION_ALL,
        justification=(
            f"hero_meta recent window ({RECENT_WINDOW_DAYS_PRIMARY} days)"
        ),
    ),
    WindowSpec(
        name="recent_180d",
        appearance_window=None,
        window_days=RECENT_WINDOW_DAYS_ALT,
        version_mode=VERSION_ALL,
        justification=(
            "LAST_180D / Slice 24 robustness window; not an a-priori primary"
        ),
    ),
    WindowSpec(
        name="current_version",
        appearance_window=None,
        window_days=None,
        version_mode=VERSION_CURRENT,
        justification="same STRATZ game_version_id history",
    ),
    WindowSpec(
        name="current_plus_previous",
        appearance_window=None,
        window_days=None,
        version_mode=VERSION_CURRENT_PLUS_PREVIOUS,
        justification="current + immediately previous represented version",
    ),
)


def _window_shape_columns(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_n_role",
        f"{prefix}_breadth",
        f"{prefix}_top1_share",
        f"{prefix}_top3_share",
        f"{prefix}_entropy",
        f"{prefix}_effective_size",
        f"{prefix}_realized_n",
        f"{prefix}_realized_share",
    )


SLICE25_STATE_COLUMNS: tuple[str, ...] = (
    "pool_n_player_explicit",
    "pool_n_player_hero_explicit",
    "pool_uncond_realized_share",
    "pool_n_role",
    *(f"pool_n_at_position_{n}" for n in EXPLICIT_POSITION_NUMBERS),
    "pool_expanding_realized_n",
    "pool_expanding_realized_share",
    "pool_expanding_breadth",
    "pool_expanding_top1_share",
    "pool_expanding_top3_share",
    "pool_expanding_entropy",
    "pool_expanding_effective_size",
    "pool_days_since_realized_at_role",
    "pool_role_gap",
    "pool_last_hero_at_role",
    *(
        column
        for spec in POOL_WINDOW_SPECS
        if spec.name != "expanding"
        for column in _window_shape_columns(f"pool_{spec.name}")
    ),
)

BASELINE_ESTIMATORS: tuple[str, ...] = (
    "expanding",
    "unconditioned",
    "last_hero_at_role",
    "population",
    "uniform_at_role",
)


@dataclass(frozen=True)
class Slice25DiagnosticReport:
    development_end: datetime
    tune_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    selected_hierarchical_k: float
    selected_hierarchical_k_justification: str
    scoring_mixture_epsilon: float
    semantics: dict[str, object]
    classification: pd.DataFrame
    split: pd.DataFrame
    coverage: pd.DataFrame
    cold_start: pd.DataFrame
    next_choice: pd.DataFrame
    next_choice_by_position: pd.DataFrame
    next_choice_by_history: pd.DataFrame
    next_choice_by_cold_start: pd.DataFrame
    role_gap: pd.DataFrame
    recency: pd.DataFrame
    recency_questions: pd.DataFrame
    hierarchical: pd.DataFrame
    pool_shape: pd.DataFrame
    cross_position: pd.DataFrame
    calibration: pd.DataFrame
    primary_comparison: pd.DataFrame
    window_support: pd.DataFrame
    integrity: dict[str, object]


def pool_entropy(shares: np.ndarray) -> float:
    """Shannon entropy of a discrete share vector (nats)."""
    positive = shares[shares > 0.0]
    if positive.size == 0:
        return float("nan")
    return float(-np.sum(positive * np.log(positive)))


def effective_pool_size(entropy: float) -> float:
    """exp(entropy); 1 for a one-hero specialist."""
    if not np.isfinite(entropy):
        return float("nan")
    return float(np.exp(entropy))


def scoring_candidates(
    prior_heroes: frozenset[int] | set[int] | tuple[int, ...] | list[int],
    *,
    realized_hero: int,
) -> frozenset[int]:
    """Common causal candidate universe for one evaluation row.

    Starts from heroes observed in professional data strictly before ``T``
    and always includes the realized hero so a genuine first observation
    is handled identically for every estimator.
    """
    candidates = {int(hero) for hero in prior_heroes}
    candidates.add(int(realized_hero))
    return frozenset(candidates)


def score_distribution(
    mass: dict[int, float],
    *,
    realized_hero: int,
    candidates: frozenset[int] | set[int] | tuple[int, ...] | list[int],
    epsilon: float = SCORING_MIXTURE_EPSILON,
) -> dict[str, float]:
    """Common-support scoring wrapper for every estimator and baseline.

    ``p`` is the empirical distribution implied by ``mass``, placed on the
    shared candidate universe ``C_T`` (zeros off support). Log-loss uses

        q = (1 - epsilon) * p + epsilon * U(C_T)

    Multiclass Brier uses raw ``p`` over the same ``C_T``. Rank / hit@k
    also use raw ``p`` with deterministic hero-id tie breaks.

    This wrapper is **not** frozen pool state. ``epsilon`` is fixed a
    priori and must not be tuned on validation or wins.
    """
    if not (0.0 <= epsilon < 1.0):
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    realized = int(realized_hero)
    universe = scoring_candidates(candidates, realized_hero=realized)
    nan_row = {
        "p_realized": float("nan"),
        "log_loss": float("nan"),
        "brier": float("nan"),
        "rank": float("nan"),
        "hit_1": float("nan"),
        "hit_3": float("nan"),
        "hit_5": float("nan"),
        "n_candidates": float("nan"),
    }
    if not universe:
        return nan_row
    heroes = sorted(universe)
    n = len(heroes)
    raw = np.asarray([float(mass.get(hero, 0.0)) for hero in heroes], dtype=float)
    total = float(raw.sum())
    if total > 0.0:
        probs = raw / total
    else:
        probs = np.zeros(n, dtype=float)
    uniform = 1.0 / float(n)
    mixed = (1.0 - epsilon) * probs + epsilon * uniform
    realized_index = heroes.index(realized)
    p_star = float(probs[realized_index])
    q_star = float(mixed[realized_index])
    log_loss = float(-np.log(max(q_star, 1e-15)))
    target = np.zeros(n, dtype=float)
    target[realized_index] = 1.0
    brier = float(np.sum((probs - target) ** 2))
    rank = 1
    for index, hero in enumerate(heroes):
        if hero == realized:
            continue
        prob = float(probs[index])
        if prob > p_star or (prob == p_star and hero < realized):
            rank += 1
    return {
        "p_realized": q_star,
        "log_loss": log_loss,
        "brier": brier,
        "rank": float(rank),
        "hit_1": float(rank <= 1),
        "hit_3": float(rank <= 3),
        "hit_5": float(rank <= 5),
        "n_candidates": float(n),
    }


def _shape_from_counts(counts: dict[int, int]) -> dict[str, float]:
    n = int(sum(counts.values()))
    if n <= 0:
        return {
            "n_role": 0.0,
            "breadth": float("nan"),
            "top1_share": float("nan"),
            "top3_share": float("nan"),
            "entropy": float("nan"),
            "effective_size": float("nan"),
        }
    values = np.array(sorted(counts.values(), reverse=True), dtype=float)
    shares = values / float(n)
    entropy = pool_entropy(shares)
    return {
        "n_role": float(n),
        "breadth": float(len(counts)),
        "top1_share": float(shares[0]),
        "top3_share": float(shares[:3].sum()),
        "entropy": entropy,
        "effective_size": effective_pool_size(entropy),
    }


def _days_between(earlier: Any, later: Any) -> float:
    if earlier is None:
        return float("nan")
    try:
        if isinstance(earlier, float) and not np.isfinite(earlier):
            return float("nan")
        if pd.isna(earlier):
            return float("nan")
    except (TypeError, ValueError):
        pass
    start = pd.Timestamp(earlier)
    end = pd.Timestamp(later)
    if pd.isna(start) or pd.isna(end):
        return float("nan")
    return float((end - start).total_seconds() / 86_400.0)


def _cross_position_from_pools(
    pools: dict[int, _PlayerRolePool],
) -> pd.DataFrame:
    """Descriptive end-of-frame inventory. Not a causal-at-T feature."""
    role_breadths: list[int] = []
    jaccards: list[float] = []
    n_player_hero_pairs_multi = 0
    multi_heroes_any_player: set[int] = set()
    n_players = 0
    breadth_hist = {n: 0 for n in range(6)}
    for pool in pools.values():
        supports = {
            role: {hero for hero, count in heroes.items() if count > 0}
            for role, heroes in pool.role_hero.items()
        }
        supports = {
            role: heroes for role, heroes in supports.items() if heroes
        }
        if pool.n_explicit <= 0:
            continue
        n_players += 1
        breadth = len(supports)
        role_breadths.append(breadth)
        breadth_hist[min(breadth, 5)] += 1
        hero_roles: dict[int, int] = defaultdict(int)
        for heroes in supports.values():
            for hero in heroes:
                hero_roles[hero] += 1
        for hero, count in hero_roles.items():
            if count >= 2:
                n_player_hero_pairs_multi += 1
                multi_heroes_any_player.add(int(hero))
        role_sets = list(supports.values())
        for i, left in enumerate(role_sets):
            for right in role_sets[i + 1 :]:
                union = left | right
                if union:
                    jaccards.append(len(left & right) / len(union))
    return pd.DataFrame(
        [
            {
                "n_players_with_explicit_history": n_players,
                # Count of (player, hero) pairs where that player played the
                # hero at two or more explicit positions — not distinct heroes.
                "n_player_hero_pairs_at_2plus_positions": n_player_hero_pairs_multi,
                "n_distinct_heroes_at_2plus_positions_any_player": len(
                    multi_heroes_any_player
                ),
                "mean_role_breadth_per_player": (
                    float(np.mean(role_breadths)) if role_breadths else float("nan")
                ),
                "median_role_breadth_per_player": (
                    float(np.median(role_breadths)) if role_breadths else float("nan")
                ),
                "n_players_role_breadth_0": breadth_hist[0],
                "n_players_role_breadth_1": breadth_hist[1],
                "n_players_role_breadth_2": breadth_hist[2],
                "n_players_role_breadth_3": breadth_hist[3],
                "n_players_role_breadth_4": breadth_hist[4],
                "n_players_role_breadth_5": breadth_hist[5],
                "mean_jaccard_role_pool_overlap": (
                    float(np.mean(jaccards)) if jaccards else float("nan")
                ),
                "n_role_pairs_compared": len(jaccards),
                "assignment_built": False,
                "flex_score_built": False,
                "inventory_includes_all_development_rows": True,
            }
        ]
    )


_NS_PER_DAY = 86_400_000_000_000


class _PlayerRolePool:
    """Mutable sparse accumulators for one player's role pools."""

    def __init__(self) -> None:
        self.role_hero: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.role_n: dict[int, int] = defaultdict(int)
        self.hero_n: dict[int, int] = defaultdict(int)
        self.n_explicit = 0
        self.last_time: dict[int, dict[int, Any]] = defaultdict(dict)
        # Chronological (hero_id, start_time_ns, version_id) per role.
        self.role_history: dict[int, list[tuple[int, int, int | None]]] = defaultdict(
            list
        )

    def add(
        self,
        *,
        role: int,
        hero: int,
        start_time: Any,
        version_id: int | None,
    ) -> None:
        stamp = pd.Timestamp(start_time)
        time_ns = int(stamp.value)
        self.role_hero[role][hero] += 1
        self.role_n[role] += 1
        self.hero_n[hero] += 1
        self.n_explicit += 1
        self.last_time[role][hero] = stamp
        self.role_history[role].append((hero, time_ns, version_id))

    def expanding_counts(self, role: int) -> dict[int, int]:
        return dict(self.role_hero.get(role, {}))

    def uncond_counts(self) -> dict[int, int]:
        return dict(self.hero_n)

    def mass_by_role(self) -> dict[int, dict[int, int]]:
        return {
            int(role): dict(self.role_hero.get(role, {}))
            for role in EXPLICIT_POSITION_NUMBERS
        }

    def window_counts(
        self,
        role: int,
        *,
        spec: WindowSpec,
        current_time: Any,
        current_version: int | None,
        previous_version: int | None,
    ) -> dict[int, int]:
        if spec.name == "expanding":
            return self.expanding_counts(role)
        hist = self.role_history.get(role)
        if not hist:
            return {}
        start = 0
        end = len(hist)
        if spec.appearance_window is not None:
            start = max(start, end - int(spec.appearance_window))
        if spec.window_days is not None:
            cutoff_ns = int(pd.Timestamp(current_time).value) - int(
                spec.window_days
            ) * _NS_PER_DAY
            lo = start
            hi = end
            while lo < hi:
                mid = (lo + hi) // 2
                if hist[mid][1] < cutoff_ns:
                    lo = mid + 1
                else:
                    hi = mid
            start = lo
        if start >= end:
            return {}

        counts: dict[int, int] = defaultdict(int)
        if spec.version_mode == VERSION_CURRENT:
            if current_version is None:
                return {}
            for hero, _time_ns, version in hist[start:end]:
                if version == current_version:
                    counts[int(hero)] += 1
            return dict(counts)
        if spec.version_mode == VERSION_CURRENT_PLUS_PREVIOUS:
            allowed = {
                version
                for version in (current_version, previous_version)
                if version is not None
            }
            if not allowed:
                return {}
            for hero, _time_ns, version in hist[start:end]:
                if version in allowed:
                    counts[int(hero)] += 1
            return dict(counts)
        for hero, _time_ns, _version in hist[start:end]:
            counts[int(hero)] += 1
        return dict(counts)


def attach_player_hero_pool_state(
    frame: pd.DataFrame,
    *,
    store_mass_by_role: bool = True,
) -> pd.DataFrame:
    """Attach causal P×R×H pool state and windowed competitors.

    For every row, state uses only ``start_time <`` that row's time.
    Same-timestamp rows are mutually blind: the accumulator is updated
    only after each timestamp group is fully scored.

    Role pools are built from historical explicit positions only. The
    current row's observed position selects which role pool to expose
    for diagnostics / next-choice evaluation. ``pool_n_at_position_*``
    are the five independent expanding denominators and do not depend
    on the current observed position.

    ``store_mass_by_role`` keeps the opaque ``_pool_mass_by_role`` snapshot
    used by unit tests. Diagnostics can disable it; cross-position
    inventory still comes from end-state pools via frame attrs.
    """
    out = frame.copy()
    n_rows = len(out)
    if n_rows == 0:
        for column in SLICE25_STATE_COLUMNS:
            out[column] = pd.Series(dtype=float)
        out["pool_previous_version_id"] = pd.Series(dtype="Int64")
        out["_pool_mass_expanding"] = pd.Series(dtype=object)
        out["_pool_mass_uncond"] = pd.Series(dtype=object)
        out["_pool_mass_population"] = pd.Series(dtype=object)
        out["_pool_mass_by_role"] = pd.Series(dtype=object)
        out["_pool_last_hero_at_role"] = pd.Series(dtype=object)
        out["_pool_candidates_prior"] = pd.Series(dtype=object)
        for spec in POOL_WINDOW_SPECS:
            out[f"_pool_mass_{spec.name}"] = pd.Series(dtype=object)
        out.attrs[CROSS_POSITION_ATTR] = _cross_position_from_pools({})
        return out

    times = pd.to_datetime(out["start_time"], utc=True)
    players = _numeric(out["player_id"]).to_numpy(dtype=float)
    heroes = _numeric(out["hero_id"]).to_numpy(dtype=float)
    positions = _numeric(out["position_number"]).to_numpy(dtype=float)
    versions = (
        _numeric(out["game_version_id"]).to_numpy(dtype=float)
        if "game_version_id" in out.columns
        else np.full(n_rows, np.nan)
    )
    previous_version = causal_previous_version_id(out)
    prev_vals = _numeric(previous_version).to_numpy(dtype=float)

    buffers: dict[str, np.ndarray] = {
        "pool_n_player_explicit": np.zeros(n_rows, dtype=float),
        "pool_n_player_hero_explicit": np.zeros(n_rows, dtype=float),
        "pool_uncond_realized_share": np.full(n_rows, np.nan),
        "pool_n_role": np.full(n_rows, np.nan),
        "pool_expanding_realized_n": np.full(n_rows, np.nan),
        "pool_expanding_realized_share": np.full(n_rows, np.nan),
        "pool_expanding_breadth": np.full(n_rows, np.nan),
        "pool_expanding_top1_share": np.full(n_rows, np.nan),
        "pool_expanding_top3_share": np.full(n_rows, np.nan),
        "pool_expanding_entropy": np.full(n_rows, np.nan),
        "pool_expanding_effective_size": np.full(n_rows, np.nan),
        "pool_days_since_realized_at_role": np.full(n_rows, np.nan),
        "pool_role_gap": np.full(n_rows, np.nan),
        "pool_last_hero_at_role": np.full(n_rows, np.nan),
    }
    for role in EXPLICIT_POSITION_NUMBERS:
        buffers[f"pool_n_at_position_{role}"] = np.zeros(n_rows, dtype=float)
    window_suffixes = (
        "n_role",
        "breadth",
        "top1_share",
        "top3_share",
        "entropy",
        "effective_size",
        "realized_n",
        "realized_share",
    )
    for spec in POOL_WINDOW_SPECS:
        if spec.name == "expanding":
            continue
        for suffix in window_suffixes:
            buffers[f"pool_{spec.name}_{suffix}"] = np.full(n_rows, np.nan)

    expanding_mass: list[dict[int, int] | None] = [None] * n_rows
    uncond_mass: list[dict[int, int] | None] = [None] * n_rows
    population_mass: list[dict[int, int] | None] = [None] * n_rows
    last_hero_at_role: list[int | None] = [None] * n_rows
    mass_by_role: list[dict[int, dict[int, int]] | None] = [None] * n_rows
    window_mass: dict[str, list[dict[int, int] | None]] = {
        spec.name: [None] * n_rows for spec in POOL_WINDOW_SPECS
    }
    candidates_prior: list[frozenset[int]] = [frozenset()] * n_rows
    pop_role_hero: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    pools: dict[int, _PlayerRolePool] = {}
    seen_heroes: set[int] = set()

    order = np.argsort(times.to_numpy(), kind="mergesort")
    sorted_times = times.to_numpy()[order]
    cuts = np.r_[True, sorted_times[1:] != sorted_times[:-1]]
    starts = np.flatnonzero(cuts)
    bounds = np.r_[starts, len(order)]

    for group in range(len(starts)):
        lo = int(bounds[group])
        hi = int(bounds[group + 1])
        prior_snapshot = frozenset(seen_heroes)
        for loc in range(lo, hi):
            idx = int(order[loc])
            candidates_prior[idx] = prior_snapshot
            player = players[idx]
            if not np.isfinite(player):
                continue
            pid = int(player)
            pool = pools.get(pid)
            if pool is None:
                pool = _PlayerRolePool()
                pools[pid] = pool

            hero = heroes[idx]
            role = positions[idx]
            version = versions[idx]
            stamp = times.iloc[idx]
            cur_v = int(version) if np.isfinite(version) else None
            prev_v = int(prev_vals[idx]) if np.isfinite(prev_vals[idx]) else None

            buffers["pool_n_player_explicit"][idx] = float(pool.n_explicit)
            uncond = pool.uncond_counts()
            uncond_mass[idx] = uncond
            if store_mass_by_role:
                mass_by_role[idx] = pool.mass_by_role()
            for number in EXPLICIT_POSITION_NUMBERS:
                buffers[f"pool_n_at_position_{number}"][idx] = float(
                    pool.role_n.get(number, 0)
                )
            if np.isfinite(hero):
                hid = int(hero)
                n_ph = int(uncond.get(hid, 0))
                buffers["pool_n_player_hero_explicit"][idx] = float(n_ph)
                if pool.n_explicit > 0:
                    buffers["pool_uncond_realized_share"][idx] = (
                        float(n_ph) / float(pool.n_explicit)
                    )

            role_ok = bool(
                np.isfinite(role) and int(role) in EXPLICIT_POSITION_NUMBERS
            )
            if not role_ok:
                continue
            r = int(role)
            pop_counts = dict(pop_role_hero[r])
            population_mass[idx] = pop_counts
            exp_counts = pool.expanding_counts(r)
            expanding_mass[idx] = exp_counts
            window_mass["expanding"][idx] = exp_counts
            shape = _shape_from_counts(exp_counts)
            buffers["pool_n_role"][idx] = shape["n_role"]
            buffers["pool_expanding_breadth"][idx] = shape["breadth"]
            buffers["pool_expanding_top1_share"][idx] = shape["top1_share"]
            buffers["pool_expanding_top3_share"][idx] = shape["top3_share"]
            buffers["pool_expanding_entropy"][idx] = shape["entropy"]
            buffers["pool_expanding_effective_size"][idx] = shape["effective_size"]
            hist = pool.role_history.get(r)
            if hist:
                last_hero = int(hist[-1][0])
                last_hero_at_role[idx] = last_hero
                buffers["pool_last_hero_at_role"][idx] = float(last_hero)
            if np.isfinite(hero):
                hid = int(hero)
                n_prh = int(exp_counts.get(hid, 0))
                buffers["pool_expanding_realized_n"][idx] = float(n_prh)
                if shape["n_role"] > 0:
                    buffers["pool_expanding_realized_share"][idx] = (
                        float(n_prh) / shape["n_role"]
                    )
                last_t = pool.last_time.get(r, {}).get(hid)
                buffers["pool_days_since_realized_at_role"][idx] = _days_between(
                    last_t, stamp
                )
                n_ph = int(uncond.get(hid, 0))
                buffers["pool_role_gap"][idx] = float(n_ph > 0 and n_prh == 0)
            for spec in POOL_WINDOW_SPECS:
                if spec.name == "expanding":
                    continue
                counts = pool.window_counts(
                    r,
                    spec=spec,
                    current_time=stamp,
                    current_version=cur_v,
                    previous_version=prev_v,
                )
                window_mass[spec.name][idx] = counts
                wshape = _shape_from_counts(counts)
                prefix = f"pool_{spec.name}"
                buffers[f"{prefix}_n_role"][idx] = wshape["n_role"]
                buffers[f"{prefix}_breadth"][idx] = wshape["breadth"]
                buffers[f"{prefix}_top1_share"][idx] = wshape["top1_share"]
                buffers[f"{prefix}_top3_share"][idx] = wshape["top3_share"]
                buffers[f"{prefix}_entropy"][idx] = wshape["entropy"]
                buffers[f"{prefix}_effective_size"][idx] = wshape["effective_size"]
                if np.isfinite(hero):
                    hid = int(hero)
                    n_w = int(counts.get(hid, 0))
                    buffers[f"{prefix}_realized_n"][idx] = float(n_w)
                    if wshape["n_role"] > 0:
                        buffers[f"{prefix}_realized_share"][idx] = (
                            float(n_w) / wshape["n_role"]
                        )

        for loc in range(lo, hi):
            idx = int(order[loc])
            player = players[idx]
            hero = heroes[idx]
            role = positions[idx]
            version = versions[idx]
            if np.isfinite(hero):
                seen_heroes.add(int(hero))
            if not (
                np.isfinite(player)
                and np.isfinite(hero)
                and np.isfinite(role)
                and int(role) in EXPLICIT_POSITION_NUMBERS
            ):
                continue
            pid = int(player)
            hid = int(hero)
            r = int(role)
            cur_v = int(version) if np.isfinite(version) else None
            pool = pools.setdefault(pid, _PlayerRolePool())
            pool.add(
                role=r,
                hero=hid,
                start_time=times.iloc[idx],
                version_id=cur_v,
            )
            pop_role_hero[r][hid] += 1

    for name, values in buffers.items():
        out[name] = values
    out["pool_previous_version_id"] = previous_version
    out["_pool_mass_expanding"] = expanding_mass
    out["_pool_mass_uncond"] = uncond_mass
    out["_pool_mass_population"] = population_mass
    out["_pool_mass_by_role"] = mass_by_role
    out["_pool_last_hero_at_role"] = last_hero_at_role
    out["_pool_candidates_prior"] = candidates_prior
    for spec in POOL_WINDOW_SPECS:
        out[f"_pool_mass_{spec.name}"] = window_mass[spec.name]
    out.attrs[CROSS_POSITION_ATTR] = _cross_position_from_pools(pools)
    return out


def _hierarchical_mass(
    role_mass: dict[int, int] | None,
    uncond_mass: dict[int, int] | None,
    *,
    n_role: float,
    k: float,
) -> dict[int, float]:
    """Count-scale mixture so the shared scoring wrapper matches A/B.

    ``pi_C = lambda * pi_A + (1-lambda) * pi_uncond`` with
    ``lambda = n_role / (n_role + k)``. Returned weights are on the same
    additive scale as empirical counts: when ``lambda=1`` the weights equal
    the role counts, so scoring is identical to expanding A.
    """
    role = role_mass or {}
    uncond = uncond_mass or {}
    n_r = float(n_role) if np.isfinite(n_role) else 0.0
    n_u = float(sum(uncond.values()))
    if n_r <= 0 and n_u <= 0:
        return {}
    if n_r <= 0:
        return {int(hero): float(count) for hero, count in uncond.items()}
    lam = n_r / (n_r + k) if (n_r + k) > 0.0 else 0.0
    heroes = set(role) | set(uncond)
    mixed: dict[int, float] = {}
    for hero in heroes:
        role_count = float(role.get(hero, 0))
        uncond_count = float(uncond.get(hero, 0))
        uncond_scaled = (uncond_count / n_u * n_r) if n_u > 0 else 0.0
        weight = lam * role_count + (1.0 - lam) * uncond_scaled
        if weight > 0.0:
            mixed[int(hero)] = weight
    return mixed


def _mass_from_object(value: object) -> dict[int, int] | None:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return value  # already int-keyed counts from attach
    return None


def _estimator_mass_at(
    *,
    name: str,
    hero: float,
    n_role: float,
    n_player: float,
    expanding: object,
    uncond: object,
    population: object,
    last_hero: object,
    window_mass: object,
    window_n: float,
    hierarchical_k: float,
) -> tuple[dict[int, float], bool]:
    if not np.isfinite(hero):
        return {}, False
    if name == "expanding":
        mass = _mass_from_object(expanding)
        if mass is None or not np.isfinite(n_role) or n_role <= 0:
            return {}, False
        return {int(h): float(c) for h, c in mass.items()}, True
    if name == "unconditioned":
        mass = _mass_from_object(uncond)
        if mass is None or not np.isfinite(n_player) or n_player <= 0:
            return {}, False
        return {int(h): float(c) for h, c in mass.items()}, True
    if name == "last_hero_at_role":
        if last_hero is None or (
            isinstance(last_hero, float) and not np.isfinite(last_hero)
        ):
            return {}, False
        return {int(last_hero): 1.0}, True
    if name == "population":
        mass = _mass_from_object(population)
        if not mass:
            return {}, False
        return {int(h): float(c) for h, c in mass.items()}, True
    if name == "uniform_at_role":
        mass = _mass_from_object(expanding)
        if mass is None or not np.isfinite(n_role) or n_role <= 0:
            return {}, False
        return {int(h): 1.0 for h, c in mass.items() if c > 0}, True
    if name == "hierarchical" or str(name).startswith("hierarchical"):
        role_mass = _mass_from_object(expanding)
        uncond_mass = _mass_from_object(uncond)
        role_n = float(n_role) if np.isfinite(n_role) else 0.0
        player_n = float(n_player) if np.isfinite(n_player) else 0.0
        if (role_mass is None or role_n <= 0) and (
            uncond_mass is None or player_n <= 0
        ):
            return {}, False
        return (
            _hierarchical_mass(
                role_mass,
                uncond_mass,
                n_role=role_n,
                k=hierarchical_k,
            ),
            True,
        )
    mass = _mass_from_object(window_mass)
    if mass is None or not np.isfinite(window_n) or window_n <= 0:
        return {}, False
    return {int(h): float(c) for h, c in mass.items()}, True


def _empty_metrics(estimator: str, split: str, row_set: str) -> dict[str, object]:
    return {
        "estimator": estimator,
        "split": split,
        "row_set": row_set,
        "n_scored": 0,
        "n_undefined": 0,
        "log_loss": float("nan"),
        "brier": float("nan"),
        "mean_rank": float("nan"),
        "hit_1": float("nan"),
        "hit_3": float("nan"),
        "hit_5": float("nan"),
        "mean_p_realized": float("nan"),
    }


def _summarize_scores(
    *,
    estimator: str,
    split: str,
    row_set: str,
    scored: list[dict[str, float]],
    n_undefined: int,
) -> dict[str, object]:
    n_scored = len(scored)
    if n_scored == 0:
        row = _empty_metrics(estimator, split, row_set)
        row["n_undefined"] = n_undefined
        return row
    return {
        "estimator": estimator,
        "split": split,
        "row_set": row_set,
        "n_scored": n_scored,
        "n_undefined": n_undefined,
        "log_loss": float(np.mean([item["log_loss"] for item in scored])),
        "brier": float(np.mean([item["brier"] for item in scored])),
        "mean_rank": float(np.mean([item["rank"] for item in scored])),
        "hit_1": float(np.mean([item["hit_1"] for item in scored])),
        "hit_3": float(np.mean([item["hit_3"] for item in scored])),
        "hit_5": float(np.mean([item["hit_5"] for item in scored])),
        "mean_p_realized": float(np.mean([item["p_realized"] for item in scored])),
    }


def _work_frame(frame: pd.DataFrame, *, row_set: str) -> pd.DataFrame:
    eligible = frame.loc[explicit_position_mask(frame)]
    if row_set == "native":
        return eligible
    if row_set == "role_history":
        n_role = _numeric(eligible["pool_n_role"])
        return eligible.loc[n_role.fillna(0) > 0]
    raise ValueError(f"unknown row_set {row_set!r}")


def _window_n_column(name: str) -> str:
    if name == "expanding":
        return "pool_n_role"
    return f"pool_{name}_n_role"


def _row_candidates(value: object, *, realized_hero: int) -> frozenset[int]:
    if isinstance(value, frozenset):
        prior = value
    elif isinstance(value, (set, tuple, list)):
        prior = frozenset(int(hero) for hero in value)
    else:
        prior = frozenset()
    return scoring_candidates(prior, realized_hero=realized_hero)


def _score_frame(
    frame: pd.DataFrame,
    *,
    estimator: str,
    split: str,
    hierarchical_k: float,
    row_set: str = "native",
    epsilon: float = SCORING_MIXTURE_EPSILON,
) -> pd.DataFrame:
    work = _work_frame(frame, row_set=row_set)
    if work.empty:
        return pd.DataFrame([_empty_metrics(estimator, split, row_set)])
    heroes = _numeric(work["hero_id"]).to_numpy(dtype=float)
    n_role = _numeric(work["pool_n_role"]).to_numpy(dtype=float)
    n_player = _numeric(work["pool_n_player_explicit"]).to_numpy(dtype=float)
    expanding = work["_pool_mass_expanding"].to_numpy()
    uncond = work["_pool_mass_uncond"].to_numpy()
    population = work["_pool_mass_population"].to_numpy()
    last_hero = work["_pool_last_hero_at_role"].to_numpy()
    prior = work["_pool_candidates_prior"].to_numpy()
    if estimator in {
        "expanding",
        "unconditioned",
        "last_hero_at_role",
        "population",
        "uniform_at_role",
        "hierarchical",
    } or str(estimator).startswith("hierarchical"):
        window_mass = expanding
        window_n = n_role
    else:
        window_mass = work[f"_pool_mass_{estimator}"].to_numpy()
        window_n = _numeric(work[_window_n_column(estimator)]).to_numpy(dtype=float)
    scored: list[dict[str, float]] = []
    n_undefined = 0
    for i in range(len(work)):
        mass, defined = _estimator_mass_at(
            name=estimator,
            hero=heroes[i],
            n_role=n_role[i],
            n_player=n_player[i],
            expanding=expanding[i],
            uncond=uncond[i],
            population=population[i],
            last_hero=last_hero[i],
            window_mass=window_mass[i],
            window_n=float(window_n[i]) if np.isfinite(window_n[i]) else float("nan"),
            hierarchical_k=hierarchical_k,
        )
        if not defined or not np.isfinite(heroes[i]):
            n_undefined += 1
            continue
        hid = int(heroes[i])
        metrics = score_distribution(
            mass,
            realized_hero=hid,
            candidates=_row_candidates(prior[i], realized_hero=hid),
            epsilon=epsilon,
        )
        if not np.isfinite(metrics["log_loss"]):
            n_undefined += 1
            continue
        scored.append(metrics)
    return pd.DataFrame(
        [
            _summarize_scores(
                estimator=estimator,
                split=split,
                row_set=row_set,
                scored=scored,
                n_undefined=n_undefined,
            )
        ]
    )


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _estimator_mass_from_series(
    row: pd.Series,
    *,
    name: str,
    hierarchical_k: float,
) -> tuple[dict[int, float], bool]:
    """Series-based mass lookup used by the reference scorer only."""
    hero = _finite_float(row.get("hero_id"))
    if not np.isfinite(hero):
        return {}, False
    if name == "expanding" or name in {
        "unconditioned",
        "last_hero_at_role",
        "population",
        "uniform_at_role",
        "hierarchical",
    } or str(name).startswith("hierarchical"):
        window_mass = row.get("_pool_mass_expanding")
        window_n = _finite_float(row.get("pool_n_role"))
    else:
        window_mass = row.get(f"_pool_mass_{name}")
        window_n = _finite_float(row.get(_window_n_column(name)))
    return _estimator_mass_at(
        name=name,
        hero=hero,
        n_role=_finite_float(row.get("pool_n_role")),
        n_player=_finite_float(row.get("pool_n_player_explicit")),
        expanding=row.get("_pool_mass_expanding"),
        uncond=row.get("_pool_mass_uncond"),
        population=row.get("_pool_mass_population"),
        last_hero=row.get("_pool_last_hero_at_role"),
        window_mass=window_mass,
        window_n=window_n,
        hierarchical_k=hierarchical_k,
    )


def _score_frame_reference(
    frame: pd.DataFrame,
    *,
    estimator: str,
    split: str,
    hierarchical_k: float,
    row_set: str = "native",
    epsilon: float = SCORING_MIXTURE_EPSILON,
) -> pd.DataFrame:
    """Simple Series/record scorer kept for equivalence tests.

    Semantics must match ``_score_frame`` exactly: same causal masses,
    same common ``C_T`` / mixture wrapper, same defined/cold-start rules,
    same metrics.
    """
    work = _work_frame(frame, row_set=row_set)
    if work.empty:
        return pd.DataFrame([_empty_metrics(estimator, split, row_set)])
    scored: list[dict[str, float]] = []
    n_undefined = 0
    heroes = _numeric(work["hero_id"]).to_numpy(dtype=float)
    for i, record in enumerate(work.to_dict(orient="records")):
        if not np.isfinite(heroes[i]):
            n_undefined += 1
            continue
        mass, defined = _estimator_mass_from_series(
            pd.Series(record),
            name=estimator,
            hierarchical_k=hierarchical_k,
        )
        if not defined:
            n_undefined += 1
            continue
        hid = int(heroes[i])
        metrics = score_distribution(
            mass,
            realized_hero=hid,
            candidates=_row_candidates(
                record.get("_pool_candidates_prior"), realized_hero=hid
            ),
            epsilon=epsilon,
        )
        if not np.isfinite(metrics["log_loss"]):
            n_undefined += 1
            continue
        scored.append(metrics)
    return pd.DataFrame(
        [
            _summarize_scores(
                estimator=estimator,
                split=split,
                row_set=row_set,
                scored=scored,
                n_undefined=n_undefined,
            )
        ]
    )


def _paired_delta(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    split: str,
    hierarchical_k: float,
    epsilon: float = SCORING_MIXTURE_EPSILON,
    include_match_bootstrap: bool = False,
) -> dict[str, object]:
    """Score ``right`` vs ``left`` only on rows where both are defined."""
    work = _work_frame(frame, row_set="role_history")
    left_scores: list[dict[str, float]] = []
    right_scores: list[dict[str, float]] = []
    match_ids: list[int] = []
    n_left_only = 0
    n_right_only = 0
    n_neither = 0
    empty = {
        "split": split,
        "left": left,
        "right": right,
        "n_common": 0,
        "n_left_only": 0,
        "n_right_only": 0,
        "n_neither": 0,
        "left_log_loss": float("nan"),
        "right_log_loss": float("nan"),
        "delta_log_loss": float("nan"),
        "left_brier": float("nan"),
        "right_brier": float("nan"),
        "delta_brier": float("nan"),
        "left_hit_1": float("nan"),
        "right_hit_1": float("nan"),
        "delta_hit_1": float("nan"),
        "left_mean_rank": float("nan"),
        "right_mean_rank": float("nan"),
        "delta_mean_rank": float("nan"),
        "left_hit_3": float("nan"),
        "right_hit_3": float("nan"),
        "delta_hit_3": float("nan"),
        "left_hit_5": float("nan"),
        "right_hit_5": float("nan"),
        "delta_hit_5": float("nan"),
    }
    if work.empty:
        if include_match_bootstrap:
            empty.update(
                {
                    "n_matches": 0,
                    "delta_log_loss_ci95_lo": float("nan"),
                    "delta_log_loss_ci95_hi": float("nan"),
                    "delta_brier_ci95_lo": float("nan"),
                    "delta_brier_ci95_hi": float("nan"),
                    "bootstrap_resamples": SLICE25_BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": SLICE25_BOOTSTRAP_SEED,
                    "bootstrap_unit": "match",
                }
            )
        return empty
    heroes = _numeric(work["hero_id"]).to_numpy(dtype=float)
    n_role = _numeric(work["pool_n_role"]).to_numpy(dtype=float)
    n_player = _numeric(work["pool_n_player_explicit"]).to_numpy(dtype=float)
    expanding = work["_pool_mass_expanding"].to_numpy()
    uncond = work["_pool_mass_uncond"].to_numpy()
    population = work["_pool_mass_population"].to_numpy()
    last_hero = work["_pool_last_hero_at_role"].to_numpy()
    prior = work["_pool_candidates_prior"].to_numpy()
    matches = _numeric(work["match_id"]).to_numpy(dtype=float)

    def _arrays_for(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name in {
            "expanding",
            "unconditioned",
            "last_hero_at_role",
            "population",
            "uniform_at_role",
            "hierarchical",
        } or str(name).startswith("hierarchical"):
            return expanding, n_role
        return (
            work[f"_pool_mass_{name}"].to_numpy(),
            _numeric(work[_window_n_column(name)]).to_numpy(dtype=float),
        )

    left_window, left_n = _arrays_for(left)
    right_window, right_n = _arrays_for(right)
    for i in range(len(work)):
        if not np.isfinite(heroes[i]):
            n_neither += 1
            continue
        left_mass, left_ok = _estimator_mass_at(
            name=left,
            hero=heroes[i],
            n_role=n_role[i],
            n_player=n_player[i],
            expanding=expanding[i],
            uncond=uncond[i],
            population=population[i],
            last_hero=last_hero[i],
            window_mass=left_window[i],
            window_n=float(left_n[i]) if np.isfinite(left_n[i]) else float("nan"),
            hierarchical_k=hierarchical_k,
        )
        right_mass, right_ok = _estimator_mass_at(
            name=right,
            hero=heroes[i],
            n_role=n_role[i],
            n_player=n_player[i],
            expanding=expanding[i],
            uncond=uncond[i],
            population=population[i],
            last_hero=last_hero[i],
            window_mass=right_window[i],
            window_n=float(right_n[i]) if np.isfinite(right_n[i]) else float("nan"),
            hierarchical_k=hierarchical_k,
        )
        if left_ok and not right_ok:
            n_left_only += 1
            continue
        if right_ok and not left_ok:
            n_right_only += 1
            continue
        if not left_ok and not right_ok:
            n_neither += 1
            continue
        hid = int(heroes[i])
        cands = _row_candidates(prior[i], realized_hero=hid)
        left_scores.append(
            score_distribution(
                left_mass, realized_hero=hid, candidates=cands, epsilon=epsilon
            )
        )
        right_scores.append(
            score_distribution(
                right_mass, realized_hero=hid, candidates=cands, epsilon=epsilon
            )
        )
        match_ids.append(int(matches[i]) if np.isfinite(matches[i]) else -1)
    n_common = len(left_scores)

    def _mean(key: str, scores: list[dict[str, float]]) -> float:
        if not scores:
            return float("nan")
        return float(np.mean([item[key] for item in scores]))

    left_ll = _mean("log_loss", left_scores)
    right_ll = _mean("log_loss", right_scores)
    left_hit = _mean("hit_1", left_scores)
    right_hit = _mean("hit_1", right_scores)
    left_hit3 = _mean("hit_3", left_scores)
    right_hit3 = _mean("hit_3", right_scores)
    left_hit5 = _mean("hit_5", left_scores)
    right_hit5 = _mean("hit_5", right_scores)
    left_rank = _mean("rank", left_scores)
    right_rank = _mean("rank", right_scores)
    left_brier = _mean("brier", left_scores)
    right_brier = _mean("brier", right_scores)
    result: dict[str, object] = {
        "split": split,
        "left": left,
        "right": right,
        "n_common": n_common,
        "n_left_only": n_left_only,
        "n_right_only": n_right_only,
        "n_neither": n_neither,
        "left_log_loss": left_ll,
        "right_log_loss": right_ll,
        "delta_log_loss": (
            left_ll - right_ll
            if np.isfinite(left_ll) and np.isfinite(right_ll)
            else float("nan")
        ),
        "left_brier": left_brier,
        "right_brier": right_brier,
        "delta_brier": (
            left_brier - right_brier
            if np.isfinite(left_brier) and np.isfinite(right_brier)
            else float("nan")
        ),
        "left_hit_1": left_hit,
        "right_hit_1": right_hit,
        "delta_hit_1": (
            right_hit - left_hit
            if np.isfinite(left_hit) and np.isfinite(right_hit)
            else float("nan")
        ),
        "left_hit_3": left_hit3,
        "right_hit_3": right_hit3,
        "delta_hit_3": (
            right_hit3 - left_hit3
            if np.isfinite(left_hit3) and np.isfinite(right_hit3)
            else float("nan")
        ),
        "left_hit_5": left_hit5,
        "right_hit_5": right_hit5,
        "delta_hit_5": (
            right_hit5 - left_hit5
            if np.isfinite(left_hit5) and np.isfinite(right_hit5)
            else float("nan")
        ),
        "left_mean_rank": left_rank,
        "right_mean_rank": right_rank,
        "delta_mean_rank": (
            left_rank - right_rank
            if np.isfinite(left_rank) and np.isfinite(right_rank)
            else float("nan")
        ),
    }
    if include_match_bootstrap:
        if n_common == 0:
            match_ll = np.asarray([], dtype=float)
            match_brier = np.asarray([], dtype=float)
        else:
            paired = pd.DataFrame(
                {
                    "match_id": match_ids,
                    "ll_delta": [
                        float(left_scores[i]["log_loss"])
                        - float(right_scores[i]["log_loss"])
                        for i in range(n_common)
                    ],
                    "brier_delta": [
                        float(left_scores[i]["brier"])
                        - float(right_scores[i]["brier"])
                        for i in range(n_common)
                    ],
                }
            )
            grouped = paired.groupby("match_id", sort=False)
            match_ll = grouped["ll_delta"].mean().to_numpy(dtype=float)
            match_brier = grouped["brier_delta"].mean().to_numpy(dtype=float)
        ll_lo, ll_hi = bootstrap_mean_ci(
            match_ll,
            n_resamples=SLICE25_BOOTSTRAP_RESAMPLES,
            random_state=SLICE25_BOOTSTRAP_SEED,
        )
        brier_lo, brier_hi = bootstrap_mean_ci(
            match_brier,
            n_resamples=SLICE25_BOOTSTRAP_RESAMPLES,
            random_state=SLICE25_BOOTSTRAP_SEED + 1,
        )
        result.update(
            {
                "n_matches": int(match_ll.size),
                "delta_log_loss_ci95_lo": ll_lo,
                "delta_log_loss_ci95_hi": ll_hi,
                "delta_brier_ci95_lo": brier_lo,
                "delta_brier_ci95_hi": brier_hi,
                "bootstrap_resamples": SLICE25_BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": SLICE25_BOOTSTRAP_SEED,
                "bootstrap_unit": "match",
            }
        )
    return result


def select_hierarchical_k(
    tune: pd.DataFrame, *, epsilon: float = SCORING_MIXTURE_EPSILON
) -> tuple[float, str, pd.DataFrame]:
    """Choose k on tune role-history rows by next-hero log-loss only."""
    rows: list[dict[str, object]] = []
    best_k = HIERARCHICAL_K_GRID[0]
    best_ll = float("inf")
    for k in HIERARCHICAL_K_GRID:
        metrics = _score_frame(
            tune,
            estimator="hierarchical",
            split="tune",
            hierarchical_k=k,
            row_set="role_history",
            epsilon=epsilon,
        )
        ll = float(metrics.iloc[0]["log_loss"])
        n_scored = int(metrics.iloc[0]["n_scored"])
        rows.append({"k": k, "log_loss": ll, "n_scored": n_scored})
        if np.isfinite(ll) and n_scored >= MIN_SCORED_ROWS and ll < best_ll:
            best_ll = ll
            best_k = k
    justification = (
        "min tune next-hero log-loss on rows with n(P,R)>0 among "
        f"{list(HIERARCHICAL_K_GRID)}; selected k={best_k} (ll={best_ll:.6f}). "
        "Not chosen on validation or match outcomes."
    )
    return float(best_k), justification, pd.DataFrame(rows)


def _cold_start_label_from_values(
    *,
    n_player: float,
    n_role: float,
    n_prh: float,
    n_ph: float,
) -> str:
    if not np.isfinite(n_player) or n_player <= 0:
        return "new_player"
    if not np.isfinite(n_role) or n_role <= 0:
        return "known_player_zero_role"
    if n_prh <= 0 and n_ph > 0:
        return "hero_known_unseen_at_role"
    if n_prh <= 0:
        return "role_known_hero_unseen_at_role"
    return "in_pool"


def _cold_start_label(row: pd.Series) -> str:
    return _cold_start_label_from_values(
        n_player=float(row.get("pool_n_player_explicit") or 0.0),
        n_role=float(row.get("pool_n_role") or 0.0),
        n_prh=float(row.get("pool_expanding_realized_n") or 0.0),
        n_ph=float(row.get("pool_n_player_hero_explicit") or 0.0),
    )


def _cold_start_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[explicit_position_mask(frame)]
    n = len(eligible)
    if n == 0:
        return pd.DataFrame()
    n_player = _numeric(eligible["pool_n_player_explicit"])
    n_role = _numeric(eligible["pool_n_role"])
    n_ph = _numeric(eligible["pool_n_player_hero_explicit"])
    n_prh = _numeric(eligible["pool_expanding_realized_n"])
    version_n = (
        _numeric(eligible["pool_current_version_n_role"])
        if "pool_current_version_n_role" in eligible.columns
        else pd.Series(np.nan, index=eligible.index)
    )
    cases = [
        ("new_player", n_player.fillna(0) <= 0),
        (
            "known_player_zero_role",
            (n_player > 0) & (n_role.isna() | (n_role <= 0)),
        ),
        (
            "role_known_hero_unseen_at_role",
            (n_role > 0) & (n_prh.fillna(0) <= 0),
        ),
        (
            "hero_known_unseen_at_role",
            (n_ph > 0) & (n_prh.fillna(0) <= 0),
        ),
        (
            "current_version_empty_expanding_exists",
            (n_role > 0) & (version_n.fillna(0) <= 0),
        ),
    ]
    rows = [
        {
            "case": label,
            "n_rows": int(mask.sum()),
            "fraction_rows": float(mask.mean()) if n else float("nan"),
            "invents_information": False,
            "mutually_exclusive": False,
        }
        for label, mask in cases
    ]
    return pd.DataFrame(rows)


def _role_gap_table(frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
    eligible = frame.loc[explicit_position_mask(frame)]
    if eligible.empty:
        return pd.DataFrame(
            [
                {
                    "split": split,
                    "n_explicit": 0,
                    "n_hero_known": 0,
                    "n_role_gap": 0,
                    "fraction_role_gap_given_hero_known": float("nan"),
                    "fraction_role_gap_all_explicit": float("nan"),
                }
            ]
        )
    gap = _numeric(eligible["pool_role_gap"])
    known = _numeric(eligible["pool_n_player_hero_explicit"]) > 0
    n_known = int(known.sum())
    return pd.DataFrame(
        [
            {
                "split": split,
                "n_explicit": int(len(eligible)),
                "n_hero_known": n_known,
                "n_role_gap": int(((gap == 1.0) & known).sum()),
                "fraction_role_gap_given_hero_known": (
                    float(gap[known].mean()) if n_known else float("nan")
                ),
                "fraction_role_gap_all_explicit": float(gap.fillna(0).mean()),
            }
        ]
    )


def _by_group_table(
    frame: pd.DataFrame,
    *,
    estimator: str,
    split: str,
    hierarchical_k: float,
    group_name: str,
    group_values: list[object],
    mask_of,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = frame.loc[explicit_position_mask(frame)]
    for value in group_values:
        subset = eligible.loc[mask_of(eligible, value)]
        metrics = _score_frame(
            subset,
            estimator=estimator,
            split=split,
            hierarchical_k=hierarchical_k,
            row_set="native",
        )
        row = metrics.iloc[0].to_dict()
        row[group_name] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _pool_shape_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[
        explicit_position_mask(frame) & (_numeric(frame["pool_n_role"]).fillna(0) > 0)
    ]
    if eligible.empty:
        return pd.DataFrame()
    columns = [
        "pool_expanding_breadth",
        "pool_expanding_top1_share",
        "pool_expanding_top3_share",
        "pool_expanding_entropy",
        "pool_expanding_effective_size",
        "pool_n_role",
    ]
    rows = []
    for column in columns:
        values = _numeric(eligible[column]).dropna()
        rows.append(
            {
                "metric": column,
                "n": int(values.size),
                "mean": float(values.mean()) if len(values) else float("nan"),
                "median": float(values.median()) if len(values) else float("nan"),
                "p10": float(values.quantile(0.10)) if len(values) else float("nan"),
                "p90": float(values.quantile(0.90)) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Reliability of expanding empirical shares vs realized selection.

    Each explicit row with ``n(P,R)>0`` emits one Bernoulli trial per hero
    in the expanding pool, plus a share-0 trial if the realized hero is
    unseen at R. Bins compare mean predicted share to mean selection rate.
    """
    eligible = frame.loc[
        explicit_position_mask(frame) & (_numeric(frame["pool_n_role"]).fillna(0) > 0)
    ]
    if eligible.empty:
        return pd.DataFrame()
    predicted: list[float] = []
    realized: list[int] = []
    for record in eligible.to_dict(orient="records"):
        mass = _mass_from_object(record.get("_pool_mass_expanding")) or {}
        n_role = float(record.get("pool_n_role") or 0.0)
        if n_role <= 0:
            continue
        hero = record.get("hero_id")
        if hero is None or not np.isfinite(float(hero)):
            continue
        hid = int(hero)
        seen = False
        for other, count in mass.items():
            share = float(count) / n_role
            predicted.append(share)
            is_hit = int(int(other) == hid)
            realized.append(is_hit)
            if int(other) == hid:
                seen = True
        if not seen:
            predicted.append(0.0)
            realized.append(1)
    if not predicted:
        return pd.DataFrame()
    cal = pd.DataFrame({"predicted": predicted, "realized": realized})
    bins = pd.cut(
        cal["predicted"],
        bins=[-0.01, 0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.01],
        include_lowest=True,
    )
    rows = []
    for label, subset in cal.groupby(bins, observed=False):
        rows.append(
            {
                "share_bin": str(label),
                "n_trials": int(len(subset)),
                "mean_predicted_share": (
                    float(subset["predicted"].mean()) if len(subset) else float("nan")
                ),
                "mean_realized_frequency": (
                    float(subset["realized"].mean()) if len(subset) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _logloss_helps(candidate: float, baseline: float) -> bool:
    return (
        np.isfinite(candidate)
        and np.isfinite(baseline)
        and (baseline - candidate) >= MATERIAL_LOGLOSS_DELTA
    )


def _hit1_helps(candidate: float, baseline: float) -> bool:
    return (
        np.isfinite(candidate)
        and np.isfinite(baseline)
        and (candidate - baseline) >= MATERIAL_HIT1_DELTA
    )


def classify_slice25(
    *,
    role_conditioning_confirmed: bool,
    role_conditioning_partial: bool,
    window_confirmed: bool,
    window_name: str | None,
    hierarchical_confirmed: bool,
) -> pd.DataFrame:
    """Map next-choice gates onto Slice 25 A/B/C classification."""
    frozen: list[str] = []
    if role_conditioning_confirmed and not role_conditioning_partial:
        classification = "A"
        gate = CLASSIFICATION_A
        frozen.append("expanding_pxrxh")
        if window_confirmed and window_name:
            frozen.append(window_name)
        if hierarchical_confirmed:
            frozen.append("hierarchical_backoff")
        next_slice = (
            "Freeze expanding P×R×H availability state for Slice 26. "
            "Do not build assignment, flex scores, or permutation solvers "
            "in this slice."
        )
    elif (
        role_conditioning_confirmed
        or role_conditioning_partial
        or window_confirmed
        or hierarchical_confirmed
    ):
        classification = "B"
        gate = CLASSIFICATION_B
        if role_conditioning_confirmed or role_conditioning_partial:
            frozen.append("expanding_pxrxh")
        if window_confirmed and window_name:
            frozen.append(window_name)
        if hierarchical_confirmed:
            frozen.append("hierarchical_backoff")
        next_slice = (
            "Freeze only supported components. Treat unsupported windows "
            "/ hierarchical k as diagnostic. Do not build Slice 26 "
            "assignment from unfrozen refinements."
        )
    else:
        classification = "C"
        gate = CLASSIFICATION_C
        frozen = []
        next_slice = (
            "Do not freeze a role-specific player hero pool. Keep "
            "diagnostics as evidence against treating P×R×H as identity "
            "beyond P×H."
        )
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "gate": gate,
                "frozen_components": tuple(frozen),
                "role_conditioning_confirmed": role_conditioning_confirmed,
                "role_conditioning_partial": role_conditioning_partial,
                "window_confirmed": window_confirmed,
                "hierarchical_confirmed": hierarchical_confirmed,
                "next_slice": next_slice,
                "scoring_smoothing_in_state": False,
                "win_model_run": False,
                "slice26_built": False,
            }
        ]
    )


def _semantics() -> dict[str, object]:
    return {
        "key": "player_id × explicit position 1–5 × hero_id",
        "history_filter": "start_time < T; explicit historical positions only",
        "same_timestamp": "mutually blind",
        "causality_ordered_by_match_id": False,
        "current_hero_in_state": False,
        "current_observed_position_builds_counts": False,
        "current_observed_position_selects_eval_role": True,
        "expected_position_builds_counts": False,
        "win_rate_in_pool": False,
        "elo_residual_in_pool": False,
        "farming_b_in_pool": False,
        "combat_c_in_pool": False,
        "slice23_in_pool": False,
        "slice24_outcome_in_pool": False,
        "leave_player_out": False,
        "n_role_0_shares": "NULL",
        "unseen_hero_at_role_share": 0.0,
        "scoring_mixture_epsilon": SCORING_MIXTURE_EPSILON,
        "scoring_wrapper": (
            "q=(1-eps)*p + eps*U(C_T) over common causal candidate universe; "
            "raw multiclass Brier on p; not part of frozen state; not tuned "
            "on validation or wins"
        ),
        "scoring_wrapper_in_frozen_state": False,
        "candidate_universe": (
            "heroes observed in professional development rows with "
            "start_time < T, plus realized hero if first-ever"
        ),
        "unconditioned_denominator": "explicit-position player appearances",
        "windows": [
            {
                "name": spec.name,
                "appearance_window": spec.appearance_window,
                "window_days": spec.window_days,
                "version_mode": spec.version_mode,
                "justification": spec.justification,
            }
            for spec in POOL_WINDOW_SPECS
        ],
        "primary_candidate": "expanding",
        "production_features": False,
    }


def _integrity(frame: pd.DataFrame, store: FeatureDuckDBConnection) -> dict[str, object]:
    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    return {
        "feature_columns_unchanged": list(FEATURE_COLUMNS),
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "n_feature_columns": len(FEATURE_COLUMNS),
        "all_feature_columns_count": len(ALL_FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "snapshot_sql_defined": bool(PRE_DRAFT_SNAPSHOT_SQL),
        "snapshot_column_count": len(SNAPSHOT_COLUMNS),
        "box_score_columns": list(BOX_SCORE_COLUMNS),
        "match_player_box_score_columns": list(MATCH_PLAYER_BOX_SCORE_COLUMNS),
        "state_columns_not_in_feature_columns": all(
            column not in FEATURE_COLUMNS for column in SLICE25_STATE_COLUMNS
        ),
        "n_rows": int(len(frame)),
        "n_explicit_rows": int(explicit_position_mask(frame).sum()),
        "holdout_used_for_window_selection": False,
        "holdout_used_for_validation": False,
        "holdout_used_for_k_selection": False,
        "holdout_used_for_smoothing_selection": False,
        "holdout_used_for_classification": False,
        "smoothing_tuned": False,
        "stratz_called": False,
        "ingestion_modified": False,
        "schema_modified": False,
        "win_model_run": False,
        "draft_probability_added": False,
        "slice23_fit_used": False,
        "slice24_outcome_used": False,
        "synergy_or_counter_created": False,
        "slice26_assignment_built": False,
        "expected_position_used_to_build_counts": False,
        "box_scores_in_match_players_view": any(
            column in view_columns for column in MATCH_PLAYER_BOX_SCORE_COLUMNS
        ),
        "full_development_mean_fallback": False,
    }


def _lookup_metrics(
    table: pd.DataFrame, *, split: str, estimator: str, row_set: str
) -> dict[str, object]:
    subset = table.loc[
        (table["split"] == split)
        & (table["estimator"] == estimator)
        & (table["row_set"] == row_set)
    ]
    if subset.empty:
        return _empty_metrics(estimator, split, row_set)
    return subset.iloc[0].to_dict()


def _window_support_table(
    frame: pd.DataFrame, *, split: str, estimator: str = "last_5_at_role"
) -> pd.DataFrame:
    """Support-size diagnostics for a windowed estimator (not a freeze gate)."""
    work = _work_frame(frame, row_set="role_history")
    mass_col = f"_pool_mass_{estimator}"
    if work.empty or mass_col not in work.columns:
        return pd.DataFrame(
            [
                {
                    "split": split,
                    "estimator": estimator,
                    "n_rows": 0,
                    "fraction_realized_in_support": float("nan"),
                    "fraction_realized_unseen": float("nan"),
                    "mean_support_size": float("nan"),
                    "median_support_size": float("nan"),
                }
            ]
        )
    heroes = _numeric(work["hero_id"]).to_numpy(dtype=float)
    masses = work[mass_col].to_numpy()
    in_support = 0
    support_sizes: list[int] = []
    n = 0
    for i in range(len(work)):
        if not np.isfinite(heroes[i]):
            continue
        mass = _mass_from_object(masses[i]) or {}
        support = {int(hero) for hero, count in mass.items() if float(count) > 0.0}
        support_sizes.append(len(support))
        n += 1
        if int(heroes[i]) in support:
            in_support += 1
    fraction_in = float(in_support / n) if n else float("nan")
    return pd.DataFrame(
        [
            {
                "split": split,
                "estimator": estimator,
                "n_rows": n,
                "fraction_realized_in_support": fraction_in,
                "fraction_realized_unseen": (
                    1.0 - fraction_in if n else float("nan")
                ),
                "mean_support_size": (
                    float(np.mean(support_sizes)) if support_sizes else float("nan")
                ),
                "median_support_size": (
                    float(np.median(support_sizes)) if support_sizes else float("nan")
                ),
            }
        ]
    )


def _window_names() -> list[str]:
    return [spec.name for spec in POOL_WINDOW_SPECS if spec.name != "expanding"]


def _recency_questions(paired: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "last_N_at_role": ["last_5_at_role", "last_10_at_role", "last_20_at_role"],
        "recent_90d": ["recent_90d"],
        "recent_180d_robustness": ["recent_180d"],
        "current_version": ["current_version"],
        "current_plus_previous": ["current_plus_previous"],
    }
    rows: list[dict[str, object]] = []
    for question, names in groups.items():
        tune_any = False
        val_any = False
        best_name = None
        best_val_delta = float("-inf")
        n_common_val = 0
        for name in names:
            tune = paired.loc[(paired["split"] == "tune") & (paired["right"] == name)]
            val = paired.loc[
                (paired["split"] == "validation") & (paired["right"] == name)
            ]
            if tune.empty or val.empty:
                continue
            t_delta = float(tune.iloc[0]["delta_log_loss"])
            v_delta = float(val.iloc[0]["delta_log_loss"])
            t_n = int(tune.iloc[0]["n_common"])
            v_n = int(val.iloc[0]["n_common"])
            t_hit = float(tune.iloc[0]["delta_hit_1"])
            v_hit = float(val.iloc[0]["delta_hit_1"])
            tune_help = (
                t_n >= MIN_SCORED_ROWS
                and np.isfinite(t_delta)
                and t_delta >= MATERIAL_LOGLOSS_DELTA
            )
            val_help = (
                v_n >= MIN_SCORED_ROWS
                and np.isfinite(v_delta)
                and v_delta >= MATERIAL_LOGLOSS_DELTA
            )
            # Hit@1 may still rise for tiny-support windows while log-loss /
            # Brier collapse under common C_T; freeze gates require LL lift.
            _ = (t_hit, v_hit)
            tune_any = tune_any or tune_help
            val_any = val_any or val_help
            if np.isfinite(v_delta) and v_delta > best_val_delta:
                best_val_delta = v_delta
                best_name = name
                n_common_val = v_n
        confirmed = tune_any and val_any
        freeze_eligible = question != "recent_180d_robustness"
        rows.append(
            {
                "question": question,
                "best_estimator": best_name,
                "tune_improves": tune_any,
                "validation_improves": val_any,
                "confirmed": confirmed,
                "validation_delta_log_loss": (
                    best_val_delta if np.isfinite(best_val_delta) else float("nan")
                ),
                "n_common_validation": n_common_val,
                "freeze_eligible": freeze_eligible,
                "stable_across_positions_checked_separately": True,
            }
        )
    return pd.DataFrame(rows)


def run_player_hero_pool_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice25DiagnosticReport:
    """Development-only Slice 25 research. Does not train a win model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    development = attach_player_hero_pool_state(
        development, store_mass_by_role=False
    )

    tune_end = development_tune_end(development["start_time"], development_end=end)
    dev_times = pd.to_datetime(development["start_time"], utc=True)
    tune_mask = dev_times <= pd.Timestamp(tune_end)
    val_mask = (dev_times > pd.Timestamp(tune_end)) & (dev_times <= pd.Timestamp(end))
    tune = development.loc[tune_mask].copy()
    validation = development.loc[val_mask].copy()

    selected_k, k_why, k_grid = select_hierarchical_k(tune)
    estimators = [
        *BASELINE_ESTIMATORS,
        *_window_names(),
        "hierarchical",
    ]
    next_choice_parts: list[pd.DataFrame] = []
    for split_name, split_frame in (("tune", tune), ("validation", validation)):
        for row_set in ("native", "role_history"):
            for estimator in estimators:
                scored = _score_frame(
                    split_frame,
                    estimator=estimator,
                    split=split_name,
                    hierarchical_k=selected_k,
                    row_set=row_set,
                )
                if estimator == "hierarchical":
                    scored = scored.assign(estimator=f"hierarchical_k{selected_k}")
                next_choice_parts.append(scored)
    next_choice = pd.concat(next_choice_parts, ignore_index=True)

    paired_rows = []
    competitors = [
        "unconditioned",
        "last_hero_at_role",
        "population",
        "uniform_at_role",
        *_window_names(),
        "hierarchical",
    ]
    for split_name, split_frame in (("tune", tune), ("validation", validation)):
        for competitor in competitors:
            row = _paired_delta(
                split_frame,
                left="expanding",
                right=competitor,
                split=split_name,
                hierarchical_k=selected_k,
                include_match_bootstrap=(
                    split_name == "validation" and competitor == "unconditioned"
                ),
            )
            if competitor == "hierarchical":
                row["right"] = f"hierarchical_k{selected_k}"
            paired_rows.append(row)
    recency = pd.DataFrame(paired_rows)
    recency_questions = _recency_questions(recency)

    primary_rows = [
        _paired_delta(
            tune,
            left="expanding",
            right="unconditioned",
            split="tune",
            hierarchical_k=selected_k,
            include_match_bootstrap=True,
        ),
        _paired_delta(
            validation,
            left="expanding",
            right="unconditioned",
            split="validation",
            hierarchical_k=selected_k,
            include_match_bootstrap=True,
        ),
    ]
    primary_comparison = pd.DataFrame(primary_rows)
    window_support = pd.concat(
        [
            _window_support_table(tune, split="tune", estimator="last_5_at_role"),
            _window_support_table(
                validation, split="validation", estimator="last_5_at_role"
            ),
        ],
        ignore_index=True,
    )

    def _row(split: str, estimator: str) -> dict[str, object]:
        return _lookup_metrics(
            next_choice, split=split, estimator=estimator, row_set="role_history"
        )

    hier_name = f"hierarchical_k{selected_k}"
    exp_tune = _row("tune", "expanding")
    exp_val = _row("validation", "expanding")
    unc_tune = _row("tune", "unconditioned")
    unc_val = _row("validation", "unconditioned")

    def _role_helps(candidate: dict[str, object], baseline: dict[str, object]) -> bool:
        n_scored = int(candidate.get("n_scored") or 0)
        if n_scored < MIN_SCORED_ROWS:
            return False
        return _logloss_helps(
            float(candidate["log_loss"]), float(baseline["log_loss"])
        ) or _hit1_helps(float(candidate["hit_1"]), float(baseline["hit_1"]))

    role_tune = _role_helps(exp_tune, unc_tune)
    role_val = _role_helps(exp_val, unc_val)
    role_confirmed_point = role_tune and role_val
    val_primary = primary_comparison.loc[
        primary_comparison["split"] == "validation"
    ]
    ci_hi = (
        float(val_primary.iloc[0]["delta_log_loss_ci95_hi"])
        if not val_primary.empty
        else float("nan")
    )
    # delta = LL(expanding) - LL(unconditioned); negative favors role conditioning.
    role_ci_confirms = bool(np.isfinite(ci_hi) and ci_hi < 0.0)
    role_confirmed = role_confirmed_point and role_ci_confirms
    role_partial = role_confirmed_point and not role_ci_confirms

    position_help = 0
    position_checked = 0
    by_position_parts: list[pd.DataFrame] = []
    for split_name, split_frame in (("tune", tune), ("validation", validation)):
        for estimator in ("expanding", "unconditioned"):
            by_position_parts.append(
                _by_group_table(
                    split_frame,
                    estimator=estimator,
                    split=split_name,
                    hierarchical_k=selected_k,
                    group_name="position_number",
                    group_values=list(EXPLICIT_POSITION_NUMBERS),
                    mask_of=lambda frame, value: _numeric(frame["position_number"])
                    == float(value),
                )
            )
    next_choice_by_position = pd.concat(by_position_parts, ignore_index=True)
    for pos in EXPLICIT_POSITION_NUMBERS:
        exp_pos = next_choice_by_position.loc[
            (next_choice_by_position["split"] == "validation")
            & (next_choice_by_position["estimator"] == "expanding")
            & (next_choice_by_position["position_number"] == pos)
        ]
        unc_pos = next_choice_by_position.loc[
            (next_choice_by_position["split"] == "validation")
            & (next_choice_by_position["estimator"] == "unconditioned")
            & (next_choice_by_position["position_number"] == pos)
        ]
        if exp_pos.empty or unc_pos.empty:
            continue
        if int(exp_pos.iloc[0]["n_scored"]) < MIN_SCORED_ROWS:
            continue
        position_checked += 1
        if _logloss_helps(
            float(exp_pos.iloc[0]["log_loss"]), float(unc_pos.iloc[0]["log_loss"])
        ) or _hit1_helps(
            float(exp_pos.iloc[0]["hit_1"]), float(unc_pos.iloc[0]["hit_1"])
        ):
            position_help += 1
    role_partial = False
    if role_confirmed_point and not role_ci_confirms:
        # Point estimates help, but match-level validation CI includes 0.
        role_partial = True
    elif role_confirmed and position_checked > 0:
        role_partial = position_help < MIN_POSITIONS_FOR_A
    elif role_tune and not role_val:
        role_partial = True
    elif (not role_confirmed) and position_help > 0:
        role_partial = True

    history_parts: list[pd.DataFrame] = []
    bucket_labels = [label for label, _low, _high in HISTORY_N_BUCKETS]

    def _bucket_mask(frame: pd.DataFrame, label: object) -> pd.Series:
        n_role = _numeric(frame["pool_n_role"]).fillna(0.0)
        return n_role.map(lambda value: history_n_bucket(float(value))) == label

    for split_name, split_frame in (("tune", tune), ("validation", validation)):
        for estimator in ("expanding", "unconditioned"):
            history_parts.append(
                _by_group_table(
                    split_frame,
                    estimator=estimator,
                    split=split_name,
                    hierarchical_k=selected_k,
                    group_name="history_bucket",
                    group_values=bucket_labels,
                    mask_of=_bucket_mask,
                )
            )
    next_choice_by_history = pd.concat(history_parts, ignore_index=True)

    cold_labels = [
        "new_player",
        "known_player_zero_role",
        "hero_known_unseen_at_role",
        "role_known_hero_unseen_at_role",
        "in_pool",
    ]
    cold_parts: list[pd.DataFrame] = []
    for split_name, split_frame in (("tune", tune), ("validation", validation)):
        work = split_frame.loc[explicit_position_mask(split_frame)].copy()
        if work.empty:
            continue
        n_player = _numeric(work["pool_n_player_explicit"]).to_numpy(dtype=float)
        n_role = _numeric(work["pool_n_role"]).to_numpy(dtype=float)
        n_prh = _numeric(work["pool_expanding_realized_n"]).to_numpy(dtype=float)
        n_ph = _numeric(work["pool_n_player_hero_explicit"]).to_numpy(dtype=float)
        work["_cold"] = [
            _cold_start_label_from_values(
                n_player=n_player[i],
                n_role=n_role[i],
                n_prh=n_prh[i],
                n_ph=n_ph[i],
            )
            for i in range(len(work))
        ]
        for estimator in ("expanding", "unconditioned"):
            cold_parts.append(
                _by_group_table(
                    work,
                    estimator=estimator,
                    split=split_name,
                    hierarchical_k=selected_k,
                    group_name="cold_start_category",
                    group_values=cold_labels,
                    mask_of=lambda frame, value: frame["_cold"] == value,
                )
            )
    next_choice_by_cold_start = (
        pd.concat(cold_parts, ignore_index=True) if cold_parts else pd.DataFrame()
    )

    best_window_name: str | None = None
    window_confirmed = False
    freeze_rows = recency_questions.loc[recency_questions["freeze_eligible"]]
    confirmed_windows = freeze_rows.loc[freeze_rows["confirmed"]]
    if not confirmed_windows.empty:
        confirmed_windows = confirmed_windows.sort_values(
            "validation_delta_log_loss", ascending=False, kind="mergesort"
        )
        best = confirmed_windows.iloc[0]
        best_window_name = (
            str(best["best_estimator"]) if pd.notna(best["best_estimator"]) else None
        )
        window_confirmed = best_window_name is not None

    hier_tune_row = recency.loc[
        (recency["split"] == "tune") & (recency["right"] == hier_name)
    ]
    hier_val_row = recency.loc[
        (recency["split"] == "validation") & (recency["right"] == hier_name)
    ]
    hierarchical_confirmed = False
    if not hier_tune_row.empty and not hier_val_row.empty:
        hierarchical_confirmed = bool(
            int(hier_tune_row.iloc[0]["n_common"]) >= MIN_SCORED_ROWS
            and int(hier_val_row.iloc[0]["n_common"]) >= MIN_SCORED_ROWS
            and float(hier_tune_row.iloc[0]["delta_log_loss"]) >= MATERIAL_LOGLOSS_DELTA
            and float(hier_val_row.iloc[0]["delta_log_loss"]) >= MATERIAL_LOGLOSS_DELTA
        )

    # Refinements cannot create a freeze when expanding P×R×H fails the
    # primary common-support comparison against unconditioned P×H.
    # Hierarchical lift toward unconditioned is expected in that regime.
    if not (role_confirmed or role_partial):
        window_confirmed = False
        hierarchical_confirmed = False
        best_window_name = None

    overall_confirmed = (
        role_confirmed
        and position_checked > 0
        and position_help >= MIN_POSITIONS_FOR_A
    )
    partial = (not overall_confirmed) and (
        role_confirmed or role_partial or role_tune
    )
    classification = classify_slice25(
        role_conditioning_confirmed=overall_confirmed,
        role_conditioning_partial=partial,
        window_confirmed=window_confirmed,
        window_name=best_window_name,
        hierarchical_confirmed=hierarchical_confirmed,
    )

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
    explicit_dev = development.loc[explicit_position_mask(development)]
    coverage = pd.DataFrame(
        [
            {
                "unit": "development_explicit",
                "n_rows": int(len(explicit_dev)),
                "fraction_role_n_positive": float(
                    (_numeric(explicit_dev["pool_n_role"]).fillna(0) > 0).mean()
                )
                if len(explicit_dev)
                else float("nan"),
                "fraction_player_explicit_positive": float(
                    (_numeric(explicit_dev["pool_n_player_explicit"]) > 0).mean()
                )
                if len(explicit_dev)
                else float("nan"),
                "median_n_role": float(
                    _numeric(explicit_dev["pool_n_role"]).dropna().median()
                )
                if _numeric(explicit_dev["pool_n_role"]).notna().any()
                else float("nan"),
            }
        ]
    )
    hierarchical = pd.concat(
        [
            k_grid.assign(split="tune_grid"),
            next_choice.loc[
                next_choice["estimator"].isin(["expanding", hier_name])
            ].copy(),
        ],
        ignore_index=True,
        sort=False,
    )
    cross_position = development.attrs.get(CROSS_POSITION_ATTR)
    if not isinstance(cross_position, pd.DataFrame):
        cross_position = pd.DataFrame()

    return Slice25DiagnosticReport(
        development_end=end,
        tune_end=tune_end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=int(len(development)),
        n_holdout_excluded=int(len(holdout)),
        selected_hierarchical_k=float(selected_k),
        selected_hierarchical_k_justification=k_why,
        scoring_mixture_epsilon=float(SCORING_MIXTURE_EPSILON),
        semantics=_semantics(),
        classification=classification,
        split=split,
        coverage=coverage,
        cold_start=_cold_start_table(development),
        next_choice=next_choice,
        next_choice_by_position=next_choice_by_position,
        next_choice_by_history=next_choice_by_history,
        next_choice_by_cold_start=next_choice_by_cold_start,
        role_gap=pd.concat(
            [
                _role_gap_table(tune, split="tune"),
                _role_gap_table(validation, split="validation"),
                _role_gap_table(development, split="development"),
            ],
            ignore_index=True,
        ),
        recency=recency,
        recency_questions=recency_questions,
        hierarchical=hierarchical,
        pool_shape=_pool_shape_table(development),
        cross_position=cross_position,
        calibration=_calibration_table(development),
        primary_comparison=primary_comparison,
        window_support=window_support,
        integrity=_integrity(development, store),
    )


def slice25_report_to_jsonable(report: Slice25DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 25 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "tune_end": report.tune_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "selected_hierarchical_k": report.selected_hierarchical_k,
        "selected_hierarchical_k_justification": (
            report.selected_hierarchical_k_justification
        ),
        "scoring_mixture_epsilon": report.scoring_mixture_epsilon,
        "recorded_classification": SLICE25_RESEARCH_CLASSIFICATION,
        "recorded_frozen_components": list(SLICE25_FROZEN_COMPONENTS),
        "diagnostic_only": SLICE25_DIAGNOSTIC_ONLY,
        "semantics": _jsonable_value(report.semantics),
        "classification": _jsonable_value(report.classification),
        "split": _jsonable_value(report.split),
        "coverage": _jsonable_value(report.coverage),
        "cold_start": _jsonable_value(report.cold_start),
        "next_choice": _jsonable_value(report.next_choice),
        "next_choice_by_position": _jsonable_value(report.next_choice_by_position),
        "next_choice_by_history": _jsonable_value(report.next_choice_by_history),
        "next_choice_by_cold_start": _jsonable_value(report.next_choice_by_cold_start),
        "role_gap": _jsonable_value(report.role_gap),
        "recency": _jsonable_value(report.recency),
        "recency_questions": _jsonable_value(report.recency_questions),
        "hierarchical": _jsonable_value(report.hierarchical),
        "pool_shape": _jsonable_value(report.pool_shape),
        "cross_position": _jsonable_value(report.cross_position),
        "calibration": _jsonable_value(report.calibration),
        "primary_comparison": _jsonable_value(report.primary_comparison),
        "window_support": _jsonable_value(report.window_support),
        "integrity": _jsonable_value(report.integrity),
        "state_columns": list(SLICE25_STATE_COLUMNS),
    }
