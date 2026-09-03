"""Slice 17: combat / teamfight player-tendency *target* diagnostics.

Research only. This module does not persist a player rating, does not
write rolling player state, does not add production features, and does
not train a win model. Candidate columns never enter ``FEATURE_COLUMNS``.

Question
--------
Does the already-landed player-match box score contain a second
repeatable, interpretable player tendency — preferably combat /
teamfight contribution — that is meaningfully different from frozen
farming candidate B?

Population
----------
Matches with ``start_time <=`` the frozen Slice 9 development end
(``FROZEN_DEVELOPMENT_END``). Holdout / TI 2026 rows are excluded from
every summary. Box-score values are POST_MATCH observations of the
*current* appearance; they are the candidate *target*, not
prediction-time features.

Reuse
-----
Development cutoff, position residualization, duration residualization,
split-half, and Pearson helpers come from Slice 12
(``player_performance_target``). Frozen farming candidate B is
reproduced from the Slice 13 formula and compared, never rewritten.
This module does not reimplement Elo, does not change ``k=5``, and
does not call STRATZ.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from dota_predictor.data.canonical_schema import (
    MATCH_PLAYER_BOX_SCORE_COLUMNS,
    PLAYER_BOX_SCORE_FIELD_MAP,
)
from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS, SNAPSHOT_COLUMNS
from dota_predictor.features.team_elo import DEFAULT_ELO_CONFIG, EloConfig
from dota_predictor.training.farming_performance_target import (
    CANDIDATE_B as FROZEN_FARMING_B,
)
from dota_predictor.training.farming_performance_target import (
    FARMING_CANDIDATE_COLUMN_NAMES,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.player_farming_state import FROZEN_SHRINKAGE_K
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    CANDIDATE_COLUMN_NAMES,
    EXPLICIT_POSITION_NUMBERS,
    MIN_HALF_APPEARANCES,
    CandidateSpec,
    _jsonable_value,
    _numeric,
    _pearson,
    _spearman,
    _std,
    build_player_performance_frame,
    explicit_position_mask,
    first_half_second_half_correlation,
    ols_residual,
    per_minute,
    position_adjusted,
    position_duration_residual,
    position_r_squared,
    restrict_development,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    utc_datetime,
)

__all__ = [
    "COMBAT_A",
    "COMBAT_B",
    "COMBAT_C",
    "COMBAT_CANDIDATE_COLUMN_NAMES",
    "COMBAT_CANDIDATE_SPECS",
    "COMBAT_C_DURATION",
    "COMBAT_C_POSITION",
    "COMBAT_C_POSITION_DURATION",
    "COMBAT_D",
    "FIELD_CATALOG",
    "FROZEN_COMBAT_CANDIDATE",
    "FROZEN_FARMING_B_COLUMN",
    "GATE_A",
    "GATE_B",
    "GATE_C",
    "REQUIRED_TEAM_SIZE",
    "SKIPPED_CANDIDATES",
    "Slice17DiagnosticReport",
    "attach_combat_candidates",
    "attach_frozen_farming_b",
    "classify_slice17",
    "complete_side_mask",
    "consecutive_persistence",
    "deaths_per_30",
    "duration_residual",
    "frozen_farming_b_values",
    "hero_damage_per_min",
    "hero_damage_share",
    "kill_participation",
    "player_variance_decomposition",
    "run_combat_performance_target_diagnostics",
    "slice17_report_to_jsonable",
    "team_sum",
]


COMBAT_A = "hero_damage_per_min"
COMBAT_B = "kill_participation"
COMBAT_C = "hero_damage_share"
COMBAT_C_POSITION = "hero_damage_share_position_adj"
COMBAT_C_DURATION = "hero_damage_share_duration_resid"
COMBAT_C_POSITION_DURATION = "hero_damage_share_position_duration_resid"
COMBAT_D = "deaths_per_30"
FROZEN_FARMING_B_COLUMN = FROZEN_FARMING_B
# Methodological freeze of the combat *target definition* after Slice 17
# diagnostics. Historical state is Slice 18. Not a production feature.
FROZEN_COMBAT_CANDIDATE = COMBAT_C_POSITION
REQUIRED_TEAM_SIZE = 5
MIN_VARIANCE_PLAYER_N = 10
MIN_VARIANCE_PLAYERS = 10
DEATHS_PER_WINDOW_SECONDS = 1800.0

GATE_A = "A — freeze candidate C as a repeatable player tendency"
GATE_B = (
    "B — promising player-performance signal exists but target "
    "definition needs one more diagnostic slice"
)
GATE_C = (
    "C — existing landed combat data does not support a sufficiently "
    "distinct repeatable player tendency"
)

_REPEATABILITY_FLOOR = 0.10
_POSITION_R2_NEUTRAL = 0.05
_DURATION_CORR_NEUTRAL = 0.15
_WIN_CORR_DISGUISED = 0.50
_FARMING_REDUNDANT = 0.60
_FARMING_CAUTION = 0.40

COMBAT_CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name=COMBAT_A,
        family="damage_rate",
        formula="hero_damage / (duration_seconds / 60)",
        adjustment_variables=("duration_seconds",),
        hero_agnostic=True,
        source_fields=("hero_damage", "duration_seconds"),
    ),
    CandidateSpec(
        name=COMBAT_B,
        family="participation",
        formula=(
            "(kills + assists) / sum(kills | match, side); "
            "NULL if team kills = 0 or the five-player kill vector is incomplete"
        ),
        adjustment_variables=(),
        hero_agnostic=True,
        source_fields=("kills", "assists"),
    ),
    CandidateSpec(
        name=COMBAT_C,
        family="damage_share",
        formula=(
            "hero_damage / sum(hero_damage | match, side); "
            "NULL if team damage = 0 or the five-player damage vector is incomplete"
        ),
        adjustment_variables=(),
        hero_agnostic=True,
        source_fields=("hero_damage",),
    ),
    CandidateSpec(
        name=COMBAT_C_POSITION,
        family="damage_share",
        formula="hero_damage_share - mean(hero_damage_share | position 1–5)",
        adjustment_variables=("position",),
        hero_agnostic=True,
        source_fields=("hero_damage",),
    ),
    CandidateSpec(
        name=COMBAT_C_DURATION,
        family="damage_share",
        formula="OLS residual of hero_damage_share ~ intercept + duration_seconds",
        adjustment_variables=("duration_seconds",),
        hero_agnostic=True,
        source_fields=("hero_damage", "duration_seconds"),
    ),
    CandidateSpec(
        name=COMBAT_C_POSITION_DURATION,
        family="damage_share",
        formula=(
            "OLS residual of hero_damage_share ~ position dummies + duration_seconds"
        ),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=True,
        source_fields=("hero_damage", "duration_seconds"),
    ),
    CandidateSpec(
        name=COMBAT_D,
        family="survivability",
        formula="deaths * 1800 / duration_seconds  (deaths per 30 minutes)",
        adjustment_variables=("duration_seconds",),
        hero_agnostic=True,
        source_fields=("deaths", "duration_seconds"),
    ),
)
COMBAT_CANDIDATE_COLUMN_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in COMBAT_CANDIDATE_SPECS
)
RAW_CANDIDATE_NAMES: tuple[str, ...] = (COMBAT_A, COMBAT_B, COMBAT_C, COMBAT_D)
SHARE_FAMILY_NAMES: tuple[str, ...] = (
    COMBAT_C,
    COMBAT_C_POSITION,
    COMBAT_C_DURATION,
    COMBAT_C_POSITION_DURATION,
)

SKIPPED_CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "name": "hero_damage_per_networth",
        "reason": (
            "Damage efficiency is not causally defensible from landed fields. "
            "networth and gold_per_minute are economy/farming outcomes and are "
            "partly produced by combat (kills yield gold). Using them as a "
            "denominator would mix frozen farming candidate B into the target "
            "by construction."
        ),
    },
    {
        "name": "tower_damage_rate",
        "reason": (
            "tower_damage is an objective/push total, not teamfight contribution. "
            "It is inventoried but not a combat-tendency candidate."
        ),
    },
    {
        "name": "hero_healing_share",
        "reason": (
            "hero_healing is support-concentrated and often zero. Slice 12 "
            "already found it hero-dependent; it is not a general combat tendency."
        ),
    },
    {
        "name": "teamfight_participation_events",
        "reason": (
            "No landed fight timeline, IMP, award, or stats payload exists. "
            "Ingestion deliberately omits those STRATZ fields."
        ),
    },
    {
        "name": "combat_equal_weight",
        "reason": (
            "Slice 12 already evaluated an equal-weight kills/deaths/assists/"
            "damage composite. Slice 17 requires one coherent concept, not "
            "an arbitrary blend."
        ),
    },
)


@dataclass(frozen=True)
class FieldCatalogRow:
    field: str
    source: str
    units: str
    historically_usable: bool
    notes: str


FIELD_CATALOG: tuple[FieldCatalogRow, ...] = (
    FieldCatalogRow(
        field="kills",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.kills via PLAYER_BOX_SCORE_FIELD_MAP"
        ),
        units="integer kill count this match",
        historically_usable=True,
        notes="POST_MATCH. Null if omitted; zero is an observed zero.",
    ),
    FieldCatalogRow(
        field="deaths",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.deaths"
        ),
        units="integer death count this match",
        historically_usable=True,
        notes="POST_MATCH. Null if omitted; zero is an observed zero.",
    ),
    FieldCatalogRow(
        field="assists",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.assists"
        ),
        units="integer assist count this match",
        historically_usable=True,
        notes="POST_MATCH. Null if omitted; zero is an observed zero.",
    ),
    FieldCatalogRow(
        field="hero_damage",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.heroDamage"
        ),
        units="integer total hero damage dealt this match",
        historically_usable=True,
        notes="POST_MATCH. Not a per-minute rate. Null if omitted.",
    ),
    FieldCatalogRow(
        field="tower_damage",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.towerDamage"
        ),
        units="integer total tower damage this match",
        historically_usable=True,
        notes="POST_MATCH objective damage; inventoried, not a Slice 17 target.",
    ),
    FieldCatalogRow(
        field="hero_healing",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.heroHealing"
        ),
        units="integer total hero healing this match",
        historically_usable=True,
        notes="POST_MATCH. Often zero outside supports.",
    ),
    FieldCatalogRow(
        field="gold_per_minute",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.goldPerMinute"
        ),
        units="integer gold per minute this match",
        historically_usable=True,
        notes="POST_MATCH economy rate. Farming-adjacent; not a combat target.",
    ),
    FieldCatalogRow(
        field="experience_per_minute",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.experiencePerMinute"
        ),
        units="integer experience per minute this match",
        historically_usable=True,
        notes="POST_MATCH economy rate. Farming-adjacent; not a combat target.",
    ),
    FieldCatalogRow(
        field="num_last_hits",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.numLastHits"
        ),
        units="integer last-hit count this match",
        historically_usable=True,
        notes="POST_MATCH farming total. Frozen candidate B's source field.",
    ),
    FieldCatalogRow(
        field="num_denies",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.numDenies"
        ),
        units="integer deny count this match",
        historically_usable=True,
        notes="POST_MATCH farming-adjacent. Not used as a combat target.",
    ),
    FieldCatalogRow(
        field="networth",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.networth"
        ),
        units="integer end-of-match net worth",
        historically_usable=True,
        notes=(
            "POST_MATCH economy stock. Rejected as a damage-efficiency "
            "denominator (combat also produces gold)."
        ),
    ),
    FieldCatalogRow(
        field="level",
        source=(
            "match_players.parquet / Canonical MatchPlayerBoxScore; "
            "STRATZ MatchPlayerType.level"
        ),
        units="integer end-of-match hero level",
        historically_usable=True,
        notes="POST_MATCH mixed farming/combat/duration outcome. Ambiguous target.",
    ),
    FieldCatalogRow(
        field="duration_seconds",
        source="matches.parquet / CanonicalMatch.duration_seconds",
        units="integer match duration in seconds",
        historically_usable=True,
        notes="POST_MATCH. Required, non-null on completed canonical matches.",
    ),
    FieldCatalogRow(
        field="radiant_win",
        source="matches.parquet / CanonicalMatch.radiant_win",
        units="boolean Radiant victory",
        historically_usable=True,
        notes="POST_MATCH result. Mapped to team_won on the player row.",
    ),
    FieldCatalogRow(
        field="team_won",
        source="derived in build_player_performance_frame from side + radiant_win",
        units="0/1 whether this player's side won",
        historically_usable=True,
        notes="POST_MATCH. Used only as a confounder diagnostic, never as a target.",
    ),
    FieldCatalogRow(
        field="position",
        source="match_players.parquet; STRATZ MatchPlayerType.position",
        units="POSITION_1..5 / UNKNOWN / FILTERED / ALL / NULL",
        historically_usable=True,
        notes=(
            "POST_MATCH parse label of this match. Explicit 1–5 only; "
            "missing is not imputed."
        ),
    ),
    FieldCatalogRow(
        field="lane",
        source="match_players.parquet; STRATZ MatchPlayerType.lane",
        units="SAFE_LANE / MID_LANE / OFF_LANE / JUNGLE / UNKNOWN / NULL",
        historically_usable=True,
        notes="POST_MATCH. Inventoried; not a Slice 17 target input.",
    ),
    FieldCatalogRow(
        field="role",
        source="match_players.parquet; STRATZ MatchPlayerType.role",
        units="CORE / LIGHT_SUPPORT / HARD_SUPPORT / UNKNOWN / NULL",
        historically_usable=True,
        notes="POST_MATCH. Coarser than position; not used as the role control.",
    ),
    FieldCatalogRow(
        field="team_id",
        source="match_players.parquet via parent matches radiant_team_id/dire_team_id",
        units="integer team identity",
        historically_usable=True,
        notes="PRE_DRAFT identity. Side grouping uses (match_id, side).",
    ),
    FieldCatalogRow(
        field="side",
        source="match_players.parquet",
        units="RADIANT or DIRE",
        historically_usable=True,
        notes="PRE_DRAFT identity. Team-relative sums are grouped by match_id+side.",
    ),
    FieldCatalogRow(
        field="imp / award / stats / fight events",
        source="not landed; MATCH_SELECTION and MATCH_PLAYER_PERFORMANCE_QUERY omit them",
        units="absent",
        historically_usable=False,
        notes="Do not invent teamfight participation from fields we do not store.",
    ),
)


@dataclass(frozen=True)
class Slice17DiagnosticReport:
    development_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    n_missing_position: int
    n_explicit_position: int
    field_inventory: pd.DataFrame
    skipped_candidates: pd.DataFrame
    formulas: pd.DataFrame
    coverage: pd.DataFrame
    distributions: pd.DataFrame
    position_dependence: pd.DataFrame
    duration_dependence: pd.DataFrame
    result_relationship: pd.DataFrame
    winner_loser: pd.DataFrame
    farming_relationship: pd.DataFrame
    farming_within_position: pd.DataFrame
    split_half: pd.DataFrame
    split_half_by_position: pd.DataFrame
    consecutive_persistence: pd.DataFrame
    variance_decomposition: pd.DataFrame
    candidate_position_means: pd.DataFrame
    candidate_comparison: pd.DataFrame
    classification: pd.DataFrame
    integrity: dict[str, object]


def complete_side_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    """True where all five teammates have a non-null value of ``column``."""
    values = _numeric(frame[column])
    present = values.notna().astype(int)
    grouped = present.groupby([frame["match_id"], frame["side"]], sort=False)
    n_rows = grouped.transform("size")
    n_present = grouped.transform("sum")
    return (n_rows == REQUIRED_TEAM_SIZE) & (n_present == REQUIRED_TEAM_SIZE)


def team_sum(frame: pd.DataFrame, column: str) -> pd.Series:
    """Sum of ``column`` on (match_id, side). Null unless the five-player vector is complete."""
    values = _numeric(frame[column])
    complete = complete_side_mask(frame, column)
    summed = values.groupby([frame["match_id"], frame["side"]], sort=False).transform(
        "sum"
    )
    return summed.where(complete)


def hero_damage_per_min(frame: pd.DataFrame) -> pd.Series:
    """``hero_damage / match minutes``. Null duration or non-positive minutes stay null."""
    return per_minute(frame["hero_damage"], frame["duration_seconds"])


def hero_damage_share(frame: pd.DataFrame) -> pd.Series:
    """Player share of own-team hero damage. Null on incomplete or zero team damage."""
    damage = _numeric(frame["hero_damage"])
    total = team_sum(frame, "hero_damage")
    share = damage / total
    return share.where(total > 0.0)


def kill_participation(frame: pd.DataFrame) -> pd.Series:
    """``(kills + assists) / team_kills``. Null on incomplete or zero team kills."""
    kills = _numeric(frame["kills"])
    assists = _numeric(frame["assists"])
    involvement = kills + assists
    team_kills = team_sum(frame, "kills")
    participation = involvement / team_kills
    return participation.where(team_kills > 0.0)


def deaths_per_30(frame: pd.DataFrame) -> pd.Series:
    """Deaths scaled to a 30-minute match. Null if duration is missing or non-positive."""
    deaths = _numeric(frame["deaths"])
    duration = _numeric(frame["duration_seconds"])
    rate = deaths * DEATHS_PER_WINDOW_SECONDS / duration
    return rate.where(duration > 0.0)


def duration_residual(frame: pd.DataFrame, column: str) -> pd.Series:
    """OLS residual of ``column ~ intercept + duration_seconds``."""
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "duration_seconds": _numeric(frame["duration_seconds"]),
        },
        index=frame.index,
    )
    residual, _coef = ols_residual(_numeric(frame[column]), design)
    return residual


def _farming_residual_zscore(series: pd.Series) -> pd.Series:
    """Match Slice 13 candidate-B standardization without editing farming code."""
    values = _numeric(series)
    finite = values[values.notna()]
    if finite.empty:
        return values
    std = float(finite.std(ddof=0))
    if std <= 1e-12:
        return values - float(finite.mean())
    return (values - float(finite.mean())) / std


def frozen_farming_b_values(frame: pd.DataFrame) -> pd.Series:
    """Reproduce Slice 13 candidate B on ``frame``. Does not modify farming modules."""
    work = frame.copy()
    work["last_hits_per_minute"] = per_minute(
        work["num_last_hits"], work["duration_seconds"]
    )
    residual = position_duration_residual(work, "last_hits_per_minute")
    return _farming_residual_zscore(residual)


def attach_frozen_farming_b(frame: pd.DataFrame) -> pd.DataFrame:
    """Add frozen farming B as a diagnostic column. Leaves existing B untouched if present."""
    out = frame.copy()
    computed = frozen_farming_b_values(out)
    if FROZEN_FARMING_B_COLUMN in out.columns:
        existing = _numeric(out[FROZEN_FARMING_B_COLUMN])
        out[FROZEN_FARMING_B_COLUMN] = existing
        out["_recomputed_farming_b"] = computed
    else:
        out[FROZEN_FARMING_B_COLUMN] = computed
    return out


def attach_combat_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Slice 17 combat candidates. Does not impute missing position or farming B."""
    out = frame.copy()
    out[COMBAT_A] = hero_damage_per_min(out)
    out[COMBAT_B] = kill_participation(out)
    out[COMBAT_C] = hero_damage_share(out)
    out[COMBAT_C_POSITION] = position_adjusted(out, COMBAT_C)
    out[COMBAT_C_DURATION] = duration_residual(out, COMBAT_C)
    out[COMBAT_C_POSITION_DURATION] = position_duration_residual(out, COMBAT_C)
    out[COMBAT_D] = deaths_per_30(out)
    return out


def consecutive_persistence(frame: pd.DataFrame, column: str) -> dict[str, object]:
    """Same-player consecutive-appearance Pearson of ``column``.

    Chronological only. Equal timestamps are skipped rather than treated
    as consecutive.
    """
    values = _numeric(frame[column])
    work = frame.loc[:, ["player_id", "start_time"]].copy()
    work[column] = values
    work["start_time"] = pd.to_datetime(work["start_time"], utc=True)
    now: list[float] = []
    nxt: list[float] = []
    for _player_id, group in work.groupby("player_id", sort=False):
        ordered = group.sort_values("start_time", kind="mergesort")
        stamps = ordered["start_time"].to_numpy()
        vals = ordered[column].to_numpy(dtype=float)
        for i in range(len(ordered) - 1):
            if not (stamps[i] < stamps[i + 1]):
                continue
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]):
                now.append(float(vals[i]))
                nxt.append(float(vals[i + 1]))
    n_pairs = len(now)
    return {
        "candidate": column,
        "n_pairs": n_pairs,
        "pearson": _pearson(pd.Series(now), pd.Series(nxt))
        if n_pairs
        else float("nan"),
        "spearman": (
            _spearman(pd.Series(now), pd.Series(nxt)) if n_pairs else float("nan")
        ),
    }


def player_variance_decomposition(
    frame: pd.DataFrame,
    column: str,
    *,
    min_player_n: int = MIN_VARIANCE_PLAYER_N,
) -> dict[str, object]:
    """Between- vs within-player variance and an ICC-like ratio."""
    values = _numeric(frame[column])
    work = pd.DataFrame(
        {"player_id": frame["player_id"], "value": values},
        index=frame.index,
    ).dropna()
    empty = {
        "candidate": column,
        "n_players": 0,
        "n_appearances": 0,
        "min_player_n": min_player_n,
        "within_player_variance": float("nan"),
        "between_player_variance": float("nan"),
        "icc": float("nan"),
    }
    if work.empty:
        return empty
    stats = work.groupby("player_id")["value"].agg(n="size", mean="mean", var="var")
    eligible = stats.loc[stats["n"] >= min_player_n]
    n_players = len(eligible)
    n_appearances = int(eligible["n"].sum()) if n_players else 0
    if n_players < MIN_VARIANCE_PLAYERS:
        empty["n_players"] = n_players
        empty["n_appearances"] = n_appearances
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
    return {
        "candidate": column,
        "n_players": n_players,
        "n_appearances": n_appearances,
        "min_player_n": min_player_n,
        "within_player_variance": within,
        "between_player_variance": between,
        "between_raw_var_of_means": between_raw,
        "mean_sampling_variance": sampling,
        "icc": icc,
    }


def _distribution_row(
    values: pd.Series, *, candidate: str, subset: str
) -> dict[str, object]:
    finite = _numeric(values).dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return {
            "candidate": candidate,
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
        "candidate": candidate,
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


def _caveat_for(
    spec: CandidateSpec,
    *,
    position_r2: float,
    duration_r: float,
    win_r: float,
    farming_r: float,
    split_half: float,
) -> str:
    notes: list[str] = []
    if spec.name == COMBAT_A:
        notes.append("raw damage rate still mixes farm, duration, and role")
    if spec.name == COMBAT_B:
        notes.append("kill participation is a team-fight proxy, not fight timestamps")
    if spec.name == COMBAT_C:
        notes.append("share is relative to this team's output, not an absolute skill")
    if spec.name == COMBAT_D:
        notes.append("higher values mean more deaths (worse survivability)")
    if np.isfinite(position_r2) and position_r2 > _POSITION_R2_NEUTRAL:
        notes.append(f"position R²={position_r2:.3f}")
    if np.isfinite(duration_r) and abs(duration_r) > _DURATION_CORR_NEUTRAL:
        notes.append(f"duration r={duration_r:.3f}")
    if np.isfinite(win_r) and abs(win_r) >= _WIN_CORR_DISGUISED:
        notes.append("appears to be a winning-team statistic")
    elif np.isfinite(win_r) and abs(win_r) >= 0.25:
        notes.append(f"moderate result contamination r={win_r:.3f}")
    if np.isfinite(farming_r) and abs(farming_r) >= _FARMING_REDUNDANT:
        notes.append("redundant with farming B")
    elif np.isfinite(farming_r) and abs(farming_r) >= _FARMING_CAUTION:
        notes.append(f"overlaps farming B r={farming_r:.3f}")
    if np.isfinite(split_half) and split_half < _REPEATABILITY_FLOOR:
        notes.append("weak split-half repeatability")
    return "; ".join(notes) if notes else "none"


def classify_slice17(report: Slice17DiagnosticReport) -> pd.DataFrame:
    """Map Slice 17 tables onto the A / B / C decision gate.

    Gate A freezes *candidate C* (team-relative hero-damage share, or a
    simple position/duration adjustment of it). Other families cannot
    claim gate A even if they look statistically stronger.
    """
    comparison = report.candidate_comparison
    empty = pd.DataFrame(
        [
            {
                "classification": "C",
                "gate": GATE_C,
                "recommended_candidate": None,
                "recommended_formula": None,
                "rationale": "No candidate comparison rows were produced.",
                "next_slice": "Do not freeze a combat target.",
            }
        ]
    )
    if comparison.empty:
        return empty

    by_name = comparison.set_index("candidate")

    def _row(name: str) -> pd.Series | None:
        if name not in by_name.index:
            return None
        return by_name.loc[name]

    def _passes_freeze(row: pd.Series) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        split_half = float(row["split_half_pearson"])
        win_r = float(row["pearson_team_won"])
        farming_r = float(row["pearson_farming_b"])
        duration_r = float(row["pearson_duration"])
        pos_r2 = float(row["position_r2"])
        coverage = float(row["coverage_explicit"])
        if not np.isfinite(split_half) or split_half < _REPEATABILITY_FLOOR:
            reasons.append("split-half below floor")
        if np.isfinite(win_r) and abs(win_r) >= _WIN_CORR_DISGUISED:
            reasons.append("disguised win label")
        if np.isfinite(farming_r) and abs(farming_r) >= _FARMING_REDUNDANT:
            reasons.append("redundant with farming B")
        if np.isfinite(duration_r) and abs(duration_r) > _DURATION_CORR_NEUTRAL:
            reasons.append("duration still dominates")
        if np.isfinite(pos_r2) and pos_r2 > _POSITION_R2_NEUTRAL:
            reasons.append("position still dominates")
        if not np.isfinite(coverage) or coverage < 0.90:
            reasons.append("coverage below 90% of explicit-position rows")
        return (len(reasons) == 0), reasons

    ranked_share: list[tuple[str, pd.Series, bool, list[str]]] = []
    for name in SHARE_FAMILY_NAMES:
        row = _row(name)
        if row is None:
            continue
        ok, reasons = _passes_freeze(row)
        ranked_share.append((name, row, ok, reasons))

    freeze_ready = [item for item in ranked_share if item[2]]
    if freeze_ready:
        name, row, _ok, _reasons = freeze_ready[0]
        spec = next(spec for spec in COMBAT_CANDIDATE_SPECS if spec.name == name)
        return pd.DataFrame(
            [
                {
                    "classification": "A",
                    "gate": GATE_A,
                    "recommended_candidate": name,
                    "recommended_formula": spec.formula,
                    "rationale": (
                        "Candidate C (team-relative hero-damage share) is a "
                        "coherent combat-contribution concept with high coverage, "
                        f"split-half r={float(row['split_half_pearson']):.3f}, "
                        f"result r={float(row['pearson_team_won']):.3f}, "
                        f"farming-B r={float(row['pearson_farming_b']):.3f}. "
                        f"Frozen definition is {name}."
                    ),
                    "next_slice": (
                        "Slice 18 may build leakage-safe historical state. "
                        "Do not add FEATURE_COLUMNS or run a win model yet."
                    ),
                    "split_half_pearson": float(row["split_half_pearson"]),
                    "pearson_team_won": float(row["pearson_team_won"]),
                    "pearson_farming_b": float(row["pearson_farming_b"]),
                    "position_r2": float(row["position_r2"]),
                    "pearson_duration": float(row["pearson_duration"]),
                }
            ]
        )

    promising: list[str] = []
    blockers: list[str] = []
    for name, row, _ok, reasons in ranked_share:
        split_half = float(row["split_half_pearson"])
        if np.isfinite(split_half) and split_half >= _REPEATABILITY_FLOOR:
            promising.append(name)
            blockers.extend(f"{name}: {item}" for item in reasons)
    for name in (COMBAT_A, COMBAT_B, COMBAT_D):
        row = _row(name)
        if row is None:
            continue
        split_half = float(row["split_half_pearson"])
        if np.isfinite(split_half) and split_half >= _REPEATABILITY_FLOOR:
            promising.append(name)

    if promising:
        share_row = _row(COMBAT_C)
        extra = ""
        if share_row is not None:
            extra = (
                f" Raw share split-half r={float(share_row['split_half_pearson']):.3f}, "
                f"position R²={float(share_row['position_r2']):.3f}, "
                f"result r={float(share_row['pearson_team_won']):.3f}, "
                f"farming-B r={float(share_row['pearson_farming_b']):.3f}."
            )
        return pd.DataFrame(
            [
                {
                    "classification": "B",
                    "gate": GATE_B,
                    "recommended_candidate": COMBAT_C,
                    "recommended_formula": next(
                        spec.formula
                        for spec in COMBAT_CANDIDATE_SPECS
                        if spec.name == COMBAT_C
                    ),
                    "rationale": (
                        "A combat-oriented signal is visible ("
                        + ", ".join(promising)
                        + ") but candidate C is not yet freeze-clean. "
                        + ("; ".join(blockers) if blockers else "see comparison table.")
                        + extra
                    ),
                    "next_slice": (
                        "One more diagnostic slice on the share family "
                        "(hero confounding, team environment, or a single "
                        "frozen adjustment) before historical state."
                    ),
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "classification": "C",
                "gate": GATE_C,
                "recommended_candidate": None,
                "recommended_formula": None,
                "rationale": (
                    "Landed combat fields exist, but no candidate is both "
                    "repeatable and distinct from role, duration, current "
                    "result, and farming B."
                ),
                "next_slice": "Do not build combat historical state.",
            }
        ]
    )


def run_combat_performance_target_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice17DiagnosticReport:
    """Development-only Slice 17 combat-target research. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    with_combat = attach_combat_candidates(development)
    with_combat = attach_frozen_farming_b(with_combat)
    n_missing_position = int((~explicit_position_mask(with_combat)).sum())
    n_explicit = int(explicit_position_mask(with_combat).sum())
    eligible = with_combat.loc[explicit_position_mask(with_combat)]

    field_rows: list[dict[str, object]] = []
    n_all = len(with_combat)
    catalog_by_name = {row.field: row for row in FIELD_CATALOG}
    inventory_fields = [
        *MATCH_PLAYER_BOX_SCORE_COLUMNS,
        "duration_seconds",
        "team_won",
        "position",
        "lane",
        "role",
        "team_id",
        "side",
    ]
    for field in inventory_fields:
        series = (
            with_combat[field]
            if field in with_combat.columns
            else pd.Series(dtype=object)
        )
        if field in {
            "kills",
            "deaths",
            "assists",
            "hero_damage",
            "tower_damage",
            "hero_healing",
            "gold_per_minute",
            "experience_per_minute",
            "num_last_hits",
            "num_denies",
            "networth",
            "level",
            "duration_seconds",
            "team_won",
        }:
            values = _numeric(series)
            present = values.notna()
            zeros = int((values.loc[present] == 0).sum()) if present.any() else 0
        else:
            present = series.notna()
            zeros = 0
            values = series
        catalog = catalog_by_name.get(field)
        field_rows.append(
            {
                "field": field,
                "source": catalog.source if catalog else "player-performance frame",
                "coverage": float(present.mean()) if n_all else float("nan"),
                "null_rate": float((~present).mean()) if n_all else float("nan"),
                "n_non_null": int(present.sum()),
                "zero_count": zeros,
                "units": catalog.units if catalog else "",
                "historically_usable": (
                    catalog.historically_usable if catalog else True
                ),
                "notes": catalog.notes if catalog else "",
            }
        )
    absent = catalog_by_name["imp / award / stats / fight events"]
    field_rows.append(
        {
            "field": absent.field,
            "source": absent.source,
            "coverage": 0.0,
            "null_rate": 1.0,
            "n_non_null": 0,
            "zero_count": 0,
            "units": absent.units,
            "historically_usable": False,
            "notes": absent.notes,
        }
    )
    field_inventory = pd.DataFrame(field_rows)
    skipped_candidates = pd.DataFrame(list(SKIPPED_CANDIDATES))
    formulas = pd.DataFrame(
        [
            {
                "candidate": spec.name,
                "family": spec.family,
                "formula": spec.formula,
                "adjustment_variables": ", ".join(spec.adjustment_variables),
                "source_fields": ", ".join(spec.source_fields),
            }
            for spec in COMBAT_CANDIDATE_SPECS
        ]
    )

    coverage_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        values = _numeric(with_combat[spec.name])
        explicit_values = _numeric(eligible[spec.name])
        coverage_rows.append(
            {
                "candidate": spec.name,
                "n_non_null": int(values.notna().sum()),
                "null_rate": float(values.isna().mean()) if n_all else float("nan"),
                "n_explicit_non_null": int(explicit_values.notna().sum()),
                "coverage_explicit": (
                    float(explicit_values.notna().mean())
                    if n_explicit
                    else float("nan")
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    distribution_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        distribution_rows.append(
            _distribution_row(with_combat[spec.name], candidate=spec.name, subset="all")
        )
        distribution_rows.append(
            _distribution_row(
                eligible[spec.name], candidate=spec.name, subset="explicit_position"
            )
        )
    distributions = pd.DataFrame(distribution_rows)

    position_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        values = _numeric(with_combat[spec.name])
        row: dict[str, object] = {
            "candidate": spec.name,
            "position_r2": position_r_squared(with_combat, spec.name),
            "n_explicit": int(
                (explicit_position_mask(with_combat) & values.notna()).sum()
            ),
        }
        for number in EXPLICIT_POSITION_NUMBERS:
            subset = values.loc[
                explicit_position_mask(with_combat)
                & (with_combat["position_number"] == number)
            ]
            finite = subset.dropna().to_numpy(dtype=float)
            row[f"pos{number}_n"] = int(finite.size)
            row[f"pos{number}_mean"] = (
                float(finite.mean()) if finite.size else float("nan")
            )
            row[f"pos{number}_std"] = _std(finite)
        position_rows.append(row)
    position_dependence = pd.DataFrame(position_rows)

    duration_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        values = _numeric(with_combat[spec.name])
        duration_rows.append(
            {
                "candidate": spec.name,
                "pearson_duration": _pearson(values, with_combat["duration_seconds"]),
                "spearman_duration": _spearman(values, with_combat["duration_seconds"]),
                "slope_vs_duration": _duration_slope(values, with_combat),
            }
        )
    duration_dependence = pd.DataFrame(duration_rows)

    result_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        values = _numeric(with_combat[spec.name])
        result_rows.append(
            {
                "candidate": spec.name,
                "pearson_team_won": _pearson(values, with_combat["team_won"]),
                "spearman_team_won": _spearman(values, with_combat["team_won"]),
            }
        )
    result_relationship = pd.DataFrame(result_rows)

    winner_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        for won, label in ((1, "winners"), (0, "losers")):
            subset = with_combat.loc[with_combat["team_won"] == won, spec.name]
            winner_rows.append(
                _distribution_row(subset, candidate=spec.name, subset=label)
            )
    winner_loser = pd.DataFrame(winner_rows)

    farming_rows: list[dict[str, object]] = []
    farming_b = _numeric(with_combat[FROZEN_FARMING_B_COLUMN])
    for spec in COMBAT_CANDIDATE_SPECS:
        values = _numeric(with_combat[spec.name])
        farming_rows.append(
            {
                "candidate": spec.name,
                "pearson_farming_b": _pearson(values, farming_b),
                "spearman_farming_b": _spearman(values, farming_b),
                "n": int((_numeric(values).notna() & farming_b.notna()).sum()),
            }
        )
    farming_relationship = pd.DataFrame(farming_rows)

    within_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        values = _numeric(with_combat[spec.name])
        for number in EXPLICIT_POSITION_NUMBERS:
            mask = explicit_position_mask(with_combat) & (
                with_combat["position_number"] == number
            )
            within_rows.append(
                {
                    "candidate": spec.name,
                    "position_number": number,
                    "n": int((mask & values.notna() & farming_b.notna()).sum()),
                    "pearson_farming_b": _pearson(values[mask], farming_b[mask]),
                }
            )
    farming_within_position = pd.DataFrame(within_rows)

    half_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        half_rows.append(
            first_half_second_half_correlation(
                eligible, spec.name, min_each=MIN_HALF_APPEARANCES
            )
        )
    split_half = pd.DataFrame(half_rows)

    half_position_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        for number in EXPLICIT_POSITION_NUMBERS:
            subset = eligible.loc[eligible["position_number"] == number]
            stats = first_half_second_half_correlation(
                subset, spec.name, min_each=MIN_HALF_APPEARANCES
            )
            stats["position_number"] = number
            half_position_rows.append(stats)
    split_half_by_position = pd.DataFrame(half_position_rows)

    consecutive_rows = [
        consecutive_persistence(eligible, spec.name) for spec in COMBAT_CANDIDATE_SPECS
    ]
    consecutive_table = pd.DataFrame(consecutive_rows)

    variance_rows = [
        player_variance_decomposition(eligible, spec.name)
        for spec in COMBAT_CANDIDATE_SPECS
    ]
    variance_table = pd.DataFrame(variance_rows)

    position_means_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        values = _numeric(with_combat[spec.name])
        present = explicit_position_mask(with_combat) & values.notna()
        row = {
            "candidate": spec.name,
            "n_explicit_position": int(present.sum()),
            "position_r2": position_r_squared(with_combat, spec.name),
        }
        for number in EXPLICIT_POSITION_NUMBERS:
            subset = values.loc[present & (with_combat["position_number"] == number)]
            finite = subset.to_numpy(dtype=float)
            row[f"pos{number}_n"] = int(finite.size)
            row[f"pos{number}_mean"] = (
                float(finite.mean()) if finite.size else float("nan")
            )
            row[f"pos{number}_std"] = _std(finite)
        position_means_rows.append(row)
    candidate_position_means = pd.DataFrame(position_means_rows)

    half_by = (
        split_half.set_index("candidate") if not split_half.empty else pd.DataFrame()
    )
    consec_by = (
        consecutive_table.set_index("candidate")
        if not consecutive_table.empty
        else pd.DataFrame()
    )
    var_by = (
        variance_table.set_index("candidate")
        if not variance_table.empty
        else pd.DataFrame()
    )
    comparison_rows: list[dict[str, object]] = []
    for spec in COMBAT_CANDIDATE_SPECS:
        values = _numeric(eligible[spec.name])
        finite = values.dropna().to_numpy(dtype=float)
        pos_r2 = position_r_squared(with_combat, spec.name)
        duration_r = _pearson(values, eligible["duration_seconds"])
        win_r = _pearson(values, eligible["team_won"])
        farming_r = _pearson(values, _numeric(eligible[FROZEN_FARMING_B_COLUMN]))
        split_r = (
            float(half_by.loc[spec.name, "pearson"])
            if spec.name in half_by.index
            else float("nan")
        )
        consec_r = (
            float(consec_by.loc[spec.name, "pearson"])
            if spec.name in consec_by.index
            else float("nan")
        )
        icc = (
            float(var_by.loc[spec.name, "icc"])
            if spec.name in var_by.index
            else float("nan")
        )
        within = (
            float(var_by.loc[spec.name, "within_player_variance"])
            if spec.name in var_by.index
            else float("nan")
        )
        between = (
            float(var_by.loc[spec.name, "between_player_variance"])
            if spec.name in var_by.index
            else float("nan")
        )
        comparison_rows.append(
            {
                "candidate": spec.name,
                "family": spec.family,
                "coverage_explicit": (
                    float(values.notna().mean()) if n_explicit else float("nan")
                ),
                "n": int(finite.size),
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "std": _std(finite),
                "pearson_team_won": win_r,
                "pearson_farming_b": farming_r,
                "pearson_duration": duration_r,
                "position_r2": pos_r2,
                "split_half_n_players": (
                    int(half_by.loc[spec.name, "n_paired_players"])
                    if spec.name in half_by.index
                    else 0
                ),
                "split_half_min_each": MIN_HALF_APPEARANCES,
                "split_half_pearson": split_r,
                "split_half_spearman": (
                    float(half_by.loc[spec.name, "spearman"])
                    if spec.name in half_by.index
                    else float("nan")
                ),
                "consecutive_n_pairs": (
                    int(consec_by.loc[spec.name, "n_pairs"])
                    if spec.name in consec_by.index
                    else 0
                ),
                "consecutive_pearson": consec_r,
                "between_player_variance": between,
                "within_player_variance": within,
                "icc": icc,
                "major_caveat": _caveat_for(
                    spec,
                    position_r2=pos_r2,
                    duration_r=duration_r,
                    win_r=win_r,
                    farming_r=farming_r,
                    split_half=split_r,
                ),
            }
        )
    candidate_comparison = pd.DataFrame(comparison_rows)

    stratz_names = tuple(name for name, _canonical in PLAYER_BOX_SCORE_FIELD_MAP)
    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    integrity = {
        "development_end": end.isoformat(),
        "ti2026_used_for_target_definition": False,
        "holdout_used_for_selection": False,
        "stratz_called": False,
        "ingestion_modified": False,
        "schema_modified": False,
        "farming_code_modified": False,
        "frozen_farming_b": FROZEN_FARMING_B_COLUMN,
        "frozen_combat_candidate": FROZEN_COMBAT_CANDIDATE,
        "frozen_shrinkage_k": FROZEN_SHRINKAGE_K,
        "frozen_k_is_5": FROZEN_SHRINKAGE_K == 5.0,
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "candidate_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in COMBAT_CANDIDATE_COLUMN_NAMES
        ),
        "candidate_in_snapshot_columns": any(
            name in SNAPSHOT_COLUMNS for name in COMBAT_CANDIDATE_COLUMN_NAMES
        ),
        "farming_candidate_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in FARMING_CANDIDATE_COLUMN_NAMES
        ),
        "slice12_candidate_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in CANDIDATE_COLUMN_NAMES
        ),
        "candidate_in_all_feature_columns": any(
            name in ALL_FEATURE_COLUMNS for name in COMBAT_CANDIDATE_COLUMN_NAMES
        ),
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "landed_stratz_box_score_fields": list(stratz_names),
        "player_rating_persisted": False,
        "historical_state_built": False,
        "shrinkage_introduced": False,
        "model_trained": False,
        "missing_position_imputed": False,
        "damage_efficiency_invented": False,
    }

    report = Slice17DiagnosticReport(
        development_end=end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        n_missing_position=n_missing_position,
        n_explicit_position=n_explicit,
        field_inventory=field_inventory,
        skipped_candidates=skipped_candidates,
        formulas=formulas,
        coverage=coverage,
        distributions=distributions,
        position_dependence=position_dependence,
        duration_dependence=duration_dependence,
        result_relationship=result_relationship,
        winner_loser=winner_loser,
        farming_relationship=farming_relationship,
        farming_within_position=farming_within_position,
        split_half=split_half,
        split_half_by_position=split_half_by_position,
        consecutive_persistence=consecutive_table,
        variance_decomposition=variance_table,
        candidate_position_means=candidate_position_means,
        candidate_comparison=candidate_comparison,
        classification=pd.DataFrame(),
        integrity=integrity,
    )
    classification = classify_slice17(report)
    return Slice17DiagnosticReport(
        **{**report.__dict__, "classification": classification}
    )


def _duration_slope(values: pd.Series, frame: pd.DataFrame) -> float:
    y = _numeric(values)
    x = _numeric(frame["duration_seconds"])
    design = pd.DataFrame({"intercept": 1.0, "x": x}, index=y.index)
    _residual, coef = ols_residual(y, design)
    if coef.size < 2:
        return float("nan")
    return float(coef[1])


def slice17_report_to_jsonable(report: Slice17DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 17 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "n_missing_position": report.n_missing_position,
        "n_explicit_position": report.n_explicit_position,
        "field_inventory": _jsonable_value(report.field_inventory),
        "skipped_candidates": _jsonable_value(report.skipped_candidates),
        "formulas": _jsonable_value(report.formulas),
        "coverage": _jsonable_value(report.coverage),
        "distributions": _jsonable_value(report.distributions),
        "position_dependence": _jsonable_value(report.position_dependence),
        "duration_dependence": _jsonable_value(report.duration_dependence),
        "result_relationship": _jsonable_value(report.result_relationship),
        "winner_loser": _jsonable_value(report.winner_loser),
        "farming_relationship": _jsonable_value(report.farming_relationship),
        "farming_within_position": _jsonable_value(report.farming_within_position),
        "split_half": _jsonable_value(report.split_half),
        "split_half_by_position": _jsonable_value(report.split_half_by_position),
        "consecutive_persistence": _jsonable_value(report.consecutive_persistence),
        "variance_decomposition": _jsonable_value(report.variance_decomposition),
        "candidate_position_means": _jsonable_value(report.candidate_position_means),
        "candidate_comparison": _jsonable_value(report.candidate_comparison),
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
            for spec in COMBAT_CANDIDATE_SPECS
        ],
    }
