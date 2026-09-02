"""Slice 12: role-adjusted player-performance *target* diagnostics.

Research only. This module does not persist a player rating, does not
write rolling player state, does not add production features, and does
not train a win model. Candidate columns never enter ``FEATURE_COLUMNS``.

Question
--------
Can landed STRATZ box scores support a useful single-match individual
player-performance target after controlling for position and team
context?

Population
----------
Matches with ``start_time <=`` the frozen Slice 9 development end.
TI 2026 and later matches are excluded from every summary. Box-score
values are POST_MATCH observations of the *current* appearance; they
are the candidate *target*, not prediction-time features.

Elo
---
Team strength uses the existing leakage-safe pre-match Elo expected
win (``match_elo_expected_wins``). This module does not reimplement
Elo. The binary match result is never subtracted from a performance
metric as the primary adjustment.

Temporal integrity
------------------
Repeatability / prior-mean calculations use strictly earlier
appearances: ``H.start_time < M.start_time``. Same-timestamp matches
are mutually blind. History is never ordered by ``match_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dota_predictor.data.canonical_schema import (
    MATCH_PLAYER_BOX_SCORE_COLUMNS,
    MatchPlayerPosition,
)
from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.player_hero_elo import match_elo_expected_wins
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.features.team_elo import DEFAULT_ELO_CONFIG, EloConfig
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    utc_datetime,
)

__all__ = [
    "BOX_SCORE_COLUMNS",
    "CANDIDATE_COLUMN_NAMES",
    "CANDIDATE_SPECS",
    "EXPLICIT_POSITION_NUMBERS",
    "MIN_PRIOR_APPEARANCES",
    "RATE_LIKE_COLUMNS",
    "REPEATABILITY_PRIOR_THRESHOLDS",
    "TOTAL_LIKE_COLUMNS",
    "CandidateSpec",
    "Slice12DiagnosticReport",
    "attach_candidate_targets",
    "build_player_performance_frame",
    "candidate_position_means_table",
    "duration_dependence_table",
    "elo_residualized",
    "explicit_position_mask",
    "first_half_second_half_correlation",
    "hero_dependence_tables",
    "ols_residual",
    "parse_position_number",
    "per_minute",
    "position_adjusted",
    "position_dependence_table",
    "position_duration_residual",
    "position_r_squared",
    "position_standardized",
    "prior_player_history",
    "raw_field_diagnostics",
    "recommend_candidates",
    "redundancy_tables",
    "repeatability_by_min_prior",
    "restrict_development",
    "run_player_performance_target_diagnostics",
    "slice12_report_to_jsonable",
    "slope_coefficient",
    "winner_loser_by_position_table",
]


BOX_SCORE_COLUMNS: tuple[str, ...] = MATCH_PLAYER_BOX_SCORE_COLUMNS
RATE_LIKE_COLUMNS: tuple[str, ...] = (
    "gold_per_minute",
    "experience_per_minute",
)
TOTAL_LIKE_COLUMNS: tuple[str, ...] = tuple(
    column for column in BOX_SCORE_COLUMNS if column not in RATE_LIKE_COLUMNS
)
EXPLICIT_POSITION_NUMBERS: tuple[int, ...] = (1, 2, 3, 4, 5)
_POSITION_NUMBER_BY_LABEL: dict[str, int] = {
    MatchPlayerPosition.POSITION_1.value: 1,
    MatchPlayerPosition.POSITION_2.value: 2,
    MatchPlayerPosition.POSITION_3.value: 3,
    MatchPlayerPosition.POSITION_4.value: 4,
    MatchPlayerPosition.POSITION_5.value: 5,
}
SELECTED_QUANTILES: tuple[float, ...] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
)
REPEATABILITY_PRIOR_THRESHOLDS: tuple[int, ...] = (3, 5, 10, 20)
MIN_PRIOR_APPEARANCES = 10
MIN_HERO_APPEARANCES = 20
MIN_HALF_APPEARANCES = 3
PCA_PRIMITIVE_COLUMNS: tuple[str, ...] = (
    "gold_per_minute",
    "experience_per_minute",
    "num_last_hits",
    "networth",
    "kills",
    "deaths",
    "assists",
    "hero_damage",
    "tower_damage",
    "hero_healing",
)
_REPEATABILITY_FLOOR = 0.10
_POSITION_R2_NEUTRAL = 0.05
_DURATION_CORR_NEUTRAL = 0.15
_WIN_CORR_DISGUISED = 0.50


@dataclass(frozen=True)
class CandidateSpec:
    """One diagnostic target definition. Not a production feature."""

    name: str
    family: str
    formula: str
    adjustment_variables: tuple[str, ...]
    hero_agnostic: bool
    source_fields: tuple[str, ...]


CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="gpm_position_standardized",
        family="single",
        formula="(gold_per_minute - mean[position]) / sd[position]",
        adjustment_variables=("position",),
        hero_agnostic=True,
        source_fields=("gold_per_minute",),
    ),
    CandidateSpec(
        name="xpm_position_standardized",
        family="single",
        formula="(experience_per_minute - mean[position]) / sd[position]",
        adjustment_variables=("position",),
        hero_agnostic=True,
        source_fields=("experience_per_minute",),
    ),
    CandidateSpec(
        name="last_hits_per_min_position_standardized",
        family="single",
        formula=("((num_last_hits / minutes) - mean[position]) / sd[position]"),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=True,
        source_fields=("num_last_hits",),
    ),
    CandidateSpec(
        name="networth_position_duration_residual_z",
        family="single",
        formula=("z(OLS residual of networth ~ position + duration_seconds)"),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=True,
        source_fields=("networth",),
    ),
    CandidateSpec(
        name="kills_position_standardized",
        family="single",
        formula="(kills - mean[position]) / sd[position]",
        adjustment_variables=("position",),
        hero_agnostic=True,
        source_fields=("kills",),
    ),
    CandidateSpec(
        name="minus_deaths_position_standardized",
        family="single",
        formula="-1 * (deaths - mean[position]) / sd[position]",
        adjustment_variables=("position",),
        hero_agnostic=True,
        source_fields=("deaths",),
    ),
    CandidateSpec(
        name="assists_position_standardized",
        family="single",
        formula="(assists - mean[position]) / sd[position]",
        adjustment_variables=("position",),
        hero_agnostic=True,
        source_fields=("assists",),
    ),
    CandidateSpec(
        name="hero_damage_position_duration_residual_z",
        family="single",
        formula=("z(OLS residual of hero_damage ~ position + duration_seconds)"),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=False,
        source_fields=("hero_damage",),
    ),
    CandidateSpec(
        name="tower_damage_position_duration_residual_z",
        family="single",
        formula=("z(OLS residual of tower_damage ~ position + duration_seconds)"),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=False,
        source_fields=("tower_damage",),
    ),
    CandidateSpec(
        name="healing_position_duration_residual_z",
        family="single",
        formula=("z(OLS residual of hero_healing ~ position + duration_seconds)"),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=False,
        source_fields=("hero_healing",),
    ),
    CandidateSpec(
        name="economy_equal_weight",
        family="economy",
        formula=(
            "mean(gpm_z, xpm_z, last_hits_per_min_z, networth_pos_dur_z); "
            "equal weights, not fit to victory"
        ),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=True,
        source_fields=(
            "gold_per_minute",
            "experience_per_minute",
            "num_last_hits",
            "networth",
        ),
    ),
    CandidateSpec(
        name="combat_equal_weight",
        family="combat",
        formula=(
            "mean(kills_z, -deaths_z, assists_z, hero_damage_pos_dur_z); "
            "equal weights, not fit to victory"
        ),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=False,
        source_fields=("kills", "deaths", "assists", "hero_damage"),
    ),
    CandidateSpec(
        name="multidimensional_equal_weight",
        family="multidimensional",
        formula=(
            "mean(z(economy), z(combat), tower_pos_dur_z, healing_pos_dur_z); "
            "equal family weights, not fit to victory"
        ),
        adjustment_variables=("position", "duration_seconds"),
        hero_agnostic=False,
        source_fields=(
            "gold_per_minute",
            "experience_per_minute",
            "num_last_hits",
            "networth",
            "kills",
            "deaths",
            "assists",
            "hero_damage",
            "tower_damage",
            "hero_healing",
        ),
    ),
    CandidateSpec(
        name="gpm_position_elo_residual",
        family="single",
        formula=(
            "OLS residual of position-adjusted GPM ~ pre-match Elo "
            "expected win (intercept + estimated slope, not forced to 1)"
        ),
        adjustment_variables=("position", "elo_expected_win"),
        hero_agnostic=True,
        source_fields=("gold_per_minute",),
    ),
    CandidateSpec(
        name="economy_position_elo_residual",
        family="economy",
        formula=(
            "OLS residual of economy_equal_weight ~ pre-match Elo "
            "expected win (intercept + estimated slope, not forced to 1)"
        ),
        adjustment_variables=("position", "duration_seconds", "elo_expected_win"),
        hero_agnostic=True,
        source_fields=(
            "gold_per_minute",
            "experience_per_minute",
            "num_last_hits",
            "networth",
        ),
    ),
)
CANDIDATE_COLUMN_NAMES: tuple[str, ...] = tuple(spec.name for spec in CANDIDATE_SPECS)
_CANDIDATE_BY_NAME: dict[str, CandidateSpec] = {
    spec.name: spec for spec in CANDIDATE_SPECS
}


@dataclass(frozen=True)
class Slice12DiagnosticReport:
    development_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    n_missing_position: int
    coverage: pd.DataFrame
    raw_diagnostics: pd.DataFrame
    extremes: pd.DataFrame
    role_dependence: pd.DataFrame
    duration_dependence: pd.DataFrame
    outcome_contamination: pd.DataFrame
    hero_sample_sizes: pd.DataFrame
    hero_within_position: pd.DataFrame
    hero_variance: pd.DataFrame
    correlation_matrix: pd.DataFrame
    within_position_correlation_matrix: pd.DataFrame
    pca: pd.DataFrame
    winner_loser_by_position: pd.DataFrame
    candidate_position_means: pd.DataFrame
    candidate_quality: pd.DataFrame
    repeatability: pd.DataFrame
    first_half_second_half: pd.DataFrame
    patch_stability: pd.DataFrame
    falsification: pd.DataFrame
    recommendations: pd.DataFrame
    integrity: dict[str, object]


def parse_position_number(value: object) -> float:
    """Map an explicit Dota 1–5 label to an integer; otherwise NaN.

    Does not impute UNKNOWN / FILTERED / ALL / NULL.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return float("nan")
    if isinstance(value, (int, np.integer)) and int(value) in EXPLICIT_POSITION_NUMBERS:
        return int(value)
    label = str(value)
    mapped = _POSITION_NUMBER_BY_LABEL.get(label)
    if mapped is None:
        return float("nan")
    return mapped


def explicit_position_mask(frame: pd.DataFrame) -> pd.Series:
    """True where ``position_number`` is an explicit Dota position 1–5."""
    numbers = pd.to_numeric(frame["position_number"], errors="coerce")
    return numbers.isin(EXPLICIT_POSITION_NUMBERS)


def restrict_development(
    frame: pd.DataFrame, *, development_end: datetime | None = None
) -> pd.DataFrame:
    """Keep rows with ``start_time <=`` the frozen development end."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    stamp = pd.to_datetime(frame["start_time"], utc=True)
    return frame.loc[stamp <= pd.Timestamp(end)].copy()


def build_player_performance_frame(
    store: FeatureDuckDBConnection, *, elo_config: EloConfig = DEFAULT_ELO_CONFIG
) -> pd.DataFrame:
    """One row per player appearance with box scores and pre-match Elo.

    Elo expected win is joined from ``match_elo_expected_wins`` (the
    same pre-match sequential replay as Slice 10). Box-score columns
    are read from ``match_players.parquet`` rather than the production
    DuckDB ``match_players`` view, which deliberately omits POST_MATCH
    scalars.
    """
    matches = store.sql(
        f"""
        SELECT
            match_id,
            start_time,
            game_version_id,
            radiant_team_id,
            dire_team_id,
            radiant_win,
            duration_seconds
        FROM {MATCHES_VIEW}
        """
    ).df()
    expected = match_elo_expected_wins(matches, config=elo_config)
    quoted = "'" + str(store.config.match_players_path).replace("'", "''") + "'"
    players = store.sql(f"SELECT * FROM read_parquet({quoted})").df()
    frame = players.merge(matches, on="match_id", how="inner", validate="many_to_one")
    frame = frame.merge(expected, on="match_id", how="inner", validate="many_to_one")
    radiant = frame["side"].astype(str) == "RADIANT"
    frame["team_won"] = np.where(
        radiant,
        frame["radiant_win"].astype(bool).astype(int),
        (~frame["radiant_win"].astype(bool)).astype(int),
    )
    frame["elo_expected_win"] = np.where(
        radiant, frame["radiant_expected_win"], frame["dire_expected_win"]
    )
    frame["position_number"] = [
        parse_position_number(value) for value in frame["position"].to_numpy()
    ]
    duration = pd.to_numeric(frame["duration_seconds"], errors="coerce")
    frame["duration_minutes"] = duration / 60.0
    columns = [
        "match_id",
        "start_time",
        "game_version_id",
        "player_id",
        "hero_id",
        "team_id",
        "side",
        "slot_in_side",
        "position",
        "position_number",
        "lane",
        "role",
        "team_won",
        "elo_expected_win",
        "duration_seconds",
        "duration_minutes",
        *BOX_SCORE_COLUMNS,
    ]
    return frame.loc[:, columns].copy()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite_mask(*series: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=series[0].index)
    for item in series:
        values = _numeric(item).astype(float)
        mask &= np.isfinite(values.to_numpy())
    return mask


def _pearson(x: pd.Series, y: pd.Series) -> float:
    xv = _numeric(x)
    yv = _numeric(y)
    mask = _finite_mask(xv, yv)
    if int(mask.sum()) < 3:
        return float("nan")
    left = xv[mask].to_numpy(dtype=float)
    right = yv[mask].to_numpy(dtype=float)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    xv = _numeric(x)
    yv = _numeric(y)
    mask = _finite_mask(xv, yv)
    if int(mask.sum()) < 3:
        return float("nan")
    return _pearson(xv[mask].rank(), yv[mask].rank())


def _std(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return 0.0
    return float(values.std(ddof=1))


def per_minute(values: pd.Series, duration_seconds: pd.Series) -> pd.Series:
    """``values / (duration_seconds / 60)``. Null duration or non-positive minutes stay null."""
    minutes = _numeric(duration_seconds) / 60.0
    amount = _numeric(values)
    out = amount / minutes
    out = out.where(minutes > 0.0)
    return out


def position_adjusted(frame: pd.DataFrame, column: str) -> pd.Series:
    """``x - mean(x | explicit position)``. NULL position is excluded from the mean."""
    values = _numeric(frame[column])
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    eligible = explicit_position_mask(frame) & values.notna()
    if not bool(eligible.any()):
        return out
    means = values[eligible].groupby(frame.loc[eligible, "position_number"]).mean()
    mapped = frame.loc[eligible, "position_number"].map(means)
    out.loc[eligible] = values.loc[eligible] - mapped.to_numpy(dtype=float)
    return out


def position_standardized(frame: pd.DataFrame, column: str) -> pd.Series:
    """Within-position z-score. NULL position does not enter the moments."""
    values = _numeric(frame[column])
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    eligible = explicit_position_mask(frame) & values.notna()
    if not bool(eligible.any()):
        return out
    grouped = values[eligible].groupby(frame.loc[eligible, "position_number"])
    means = grouped.mean()
    stds = grouped.std(ddof=0)
    pos = frame.loc[eligible, "position_number"]
    centered = values.loc[eligible] - pos.map(means).to_numpy(dtype=float)
    scale = pos.map(stds).to_numpy(dtype=float)
    z = np.divide(
        centered.to_numpy(dtype=float),
        scale,
        out=np.zeros(int(eligible.sum()), dtype=float),
        where=scale > 0.0,
    )
    out.loc[eligible] = z
    return out


def position_r_squared(frame: pd.DataFrame, column: str) -> float:
    """ANOVA-style R² of ``column`` on explicit position 1–5."""
    values = _numeric(frame[column])
    eligible = explicit_position_mask(frame) & values.notna()
    y = values.loc[eligible].to_numpy(dtype=float)
    if y.size < 2:
        return float("nan")
    grand = float(y.mean())
    ss_tot = float(np.sum((y - grand) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_within = 0.0
    positions = frame.loc[eligible, "position_number"].to_numpy(dtype=float)
    for number in EXPLICIT_POSITION_NUMBERS:
        group = y[positions == number]
        if group.size == 0:
            continue
        ss_within += float(np.sum((group - group.mean()) ** 2))
    return 1.0 - ss_within / ss_tot


def ols_residual(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """Residual of ``y ~ X`` (X should already include an intercept column)."""
    yv = _numeric(y)
    residual = pd.Series(np.nan, index=y.index, dtype=float)
    mask = yv.notna() & X.notna().all(axis=1)
    n_params = int(X.shape[1])
    coef = np.full(n_params, np.nan, dtype=float)
    if int(mask.sum()) < n_params:
        return residual, coef
    design = X.loc[mask].to_numpy(dtype=float)
    target = yv.loc[mask].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual.loc[mask] = target - design @ coef
    return residual, coef


def _position_dummy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numbers = _numeric(frame["position_number"])
    intercept = pd.Series(1.0, index=frame.index, name="intercept", dtype=float)
    intercept = intercept.where(numbers.notna())
    dummies = pd.get_dummies(numbers, prefix="pos", drop_first=True, dtype=float)
    dummies = dummies.where(numbers.notna())
    return pd.concat([intercept, dummies], axis=1)


def position_duration_residual(frame: pd.DataFrame, column: str) -> pd.Series:
    """OLS residual of ``column ~ position dummies + duration_seconds``."""
    design = _position_dummy_frame(frame)
    design["duration_seconds"] = _numeric(frame["duration_seconds"])
    eligible = explicit_position_mask(frame)
    residual, _coef = ols_residual(
        _numeric(frame[column]).where(eligible), design.where(eligible)
    )
    return residual


def elo_residualized(values: pd.Series, elo_expected_win: pd.Series) -> pd.Series:
    """OLS residual of ``values ~ intercept + elo_expected_win``.

    The slope is estimated from the supplied rows. It is not forced to 1.
    """
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "elo_expected_win": _numeric(elo_expected_win),
        },
        index=values.index,
    )
    residual, _coef = ols_residual(values, design)
    return residual


def slope_coefficient(y: pd.Series, x: pd.Series) -> float:
    """OLS slope of ``y ~ intercept + x``."""
    design = pd.DataFrame(
        {"intercept": 1.0, "x": _numeric(x)},
        index=y.index,
    )
    _residual, coef = ols_residual(y, design)
    if coef.size < 2:
        return float("nan")
    return float(coef[1])


def _zscore(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    finite = values[values.notna()]
    if finite.empty:
        return values
    std = float(finite.std(ddof=0))
    if std == 0.0:
        return values - float(finite.mean())
    return (values - float(finite.mean())) / std


def attach_candidate_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add diagnostic candidate columns. Does not impute missing position."""
    out = frame.copy()
    out["gpm_position_standardized"] = position_standardized(out, "gold_per_minute")
    out["xpm_position_standardized"] = position_standardized(
        out, "experience_per_minute"
    )
    out["last_hits_per_minute"] = per_minute(
        out["num_last_hits"], out["duration_seconds"]
    )
    out["last_hits_per_min_position_standardized"] = position_standardized(
        out, "last_hits_per_minute"
    )
    out["networth_position_duration_residual"] = position_duration_residual(
        out, "networth"
    )
    out["networth_position_duration_residual_z"] = _zscore(
        out["networth_position_duration_residual"]
    )
    out["kills_position_standardized"] = position_standardized(out, "kills")
    out["minus_deaths_position_standardized"] = -position_standardized(out, "deaths")
    out["assists_position_standardized"] = position_standardized(out, "assists")
    out["hero_damage_position_duration_residual"] = position_duration_residual(
        out, "hero_damage"
    )
    out["hero_damage_position_duration_residual_z"] = _zscore(
        out["hero_damage_position_duration_residual"]
    )
    out["tower_damage_position_duration_residual"] = position_duration_residual(
        out, "tower_damage"
    )
    out["tower_damage_position_duration_residual_z"] = _zscore(
        out["tower_damage_position_duration_residual"]
    )
    out["healing_position_duration_residual"] = position_duration_residual(
        out, "hero_healing"
    )
    out["healing_position_duration_residual_z"] = _zscore(
        out["healing_position_duration_residual"]
    )
    economy_parts = out[
        [
            "gpm_position_standardized",
            "xpm_position_standardized",
            "last_hits_per_min_position_standardized",
            "networth_position_duration_residual_z",
        ]
    ]
    out["economy_equal_weight"] = economy_parts.mean(axis=1, skipna=False)
    combat_parts = out[
        [
            "kills_position_standardized",
            "minus_deaths_position_standardized",
            "assists_position_standardized",
            "hero_damage_position_duration_residual_z",
        ]
    ]
    out["combat_equal_weight"] = combat_parts.mean(axis=1, skipna=False)
    family = pd.DataFrame(
        {
            "economy": _zscore(out["economy_equal_weight"]),
            "combat": _zscore(out["combat_equal_weight"]),
            "tower": out["tower_damage_position_duration_residual_z"],
            "heal": out["healing_position_duration_residual_z"],
        },
        index=out.index,
    )
    out["multidimensional_equal_weight"] = family.mean(axis=1, skipna=False)
    out["gpm_position_elo_residual"] = elo_residualized(
        position_adjusted(out, "gold_per_minute"),
        out["elo_expected_win"],
    )
    out["economy_position_elo_residual"] = elo_residualized(
        out["economy_equal_weight"],
        out["elo_expected_win"],
    )
    return out


def raw_field_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Coverage, moments, and correlations for every raw box-score field."""
    rows: list[dict[str, object]] = []
    n_all = len(frame)
    for column in BOX_SCORE_COLUMNS:
        values = _numeric(frame[column])
        present = values.notna()
        finite = values.loc[present].to_numpy(dtype=float)
        zeros = int((values.loc[present] == 0).sum()) if present.any() else 0
        row: dict[str, object] = {
            "field": column,
            "n_non_null": int(present.sum()),
            "null_count": int((~present).sum()),
            "null_rate": float((~present).mean()) if n_all else float("nan"),
            "zero_count": zeros,
            "zero_rate_among_non_null": (
                zeros / int(present.sum()) if int(present.sum()) else float("nan")
            ),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "std": _std(finite),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "pearson_duration": _pearson(values, frame["duration_seconds"]),
            "spearman_duration": _spearman(values, frame["duration_seconds"]),
            "pearson_team_won": _pearson(values, frame["team_won"]),
            "spearman_team_won": _spearman(values, frame["team_won"]),
            "pearson_elo_expected_win": _pearson(values, frame["elo_expected_win"]),
            "spearman_elo_expected_win": _spearman(values, frame["elo_expected_win"]),
        }
        for quantile in SELECTED_QUANTILES:
            key = f"q{int(quantile * 100):02d}"
            row[key] = (
                float(np.quantile(finite, quantile)) if finite.size else float("nan")
            )
        for number in EXPLICIT_POSITION_NUMBERS:
            subset = values.loc[
                explicit_position_mask(frame) & (frame["position_number"] == number)
            ]
            finite_pos = subset.dropna().to_numpy(dtype=float)
            row[f"pos{number}_n"] = int(finite_pos.size)
            row[f"pos{number}_mean"] = (
                float(finite_pos.mean()) if finite_pos.size else float("nan")
            )
            row[f"pos{number}_median"] = (
                float(np.median(finite_pos)) if finite_pos.size else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _extreme_rows(frame: pd.DataFrame, *, n_each: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    identity = ["match_id", "player_id", "hero_id", "position", "position_number"]
    for column in BOX_SCORE_COLUMNS:
        values = _numeric(frame[column])
        ranked = frame.loc[values.notna(), identity + [column]].copy()
        if ranked.empty:
            continue
        ranked = ranked.sort_values(column, kind="mergesort")
        low = ranked.head(n_each).copy()
        high = ranked.tail(n_each).copy()
        low["field"] = column
        high["field"] = column
        low["extreme"] = "min"
        high["extreme"] = "max"
        low["value"] = _numeric(low[column])
        high["value"] = _numeric(high[column])
        rows.append(low.drop(columns=[column]))
        rows.append(high.drop(columns=[column]))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def position_dependence_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in BOX_SCORE_COLUMNS:
        values = _numeric(frame[column])
        eligible = explicit_position_mask(frame) & values.notna()
        row: dict[str, object] = {
            "field": column,
            "n_explicit_position": int(eligible.sum()),
            "n_missing_position": int((~explicit_position_mask(frame)).sum()),
            "position_r2": position_r_squared(frame, column),
        }
        for number in EXPLICIT_POSITION_NUMBERS:
            subset = values.loc[eligible & (frame["position_number"] == number)]
            finite = subset.to_numpy(dtype=float)
            row[f"pos{number}_mean"] = (
                float(finite.mean()) if finite.size else float("nan")
            )
            row[f"pos{number}_std"] = _std(finite)
        rows.append(row)
    return pd.DataFrame(rows)


def duration_dependence_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in BOX_SCORE_COLUMNS:
        values = _numeric(frame[column])
        per_min = per_minute(values, frame["duration_seconds"])
        pos_adj = position_adjusted(frame, column)
        pos_dur = position_duration_residual(frame, column)
        rate_pos = position_adjusted(
            frame.assign(_per_min=per_min),
            "_per_min",
        )
        rows.append(
            {
                "field": column,
                "kind": "rate" if column in RATE_LIKE_COLUMNS else "total",
                "pearson_raw_duration": _pearson(values, frame["duration_seconds"]),
                "spearman_raw_duration": _spearman(values, frame["duration_seconds"]),
                "pearson_per_minute_duration": _pearson(
                    per_min, frame["duration_seconds"]
                ),
                "pearson_position_adjusted_duration": _pearson(
                    pos_adj, frame["duration_seconds"]
                ),
                "pearson_position_duration_residual_duration": _pearson(
                    pos_dur, frame["duration_seconds"]
                ),
                "pearson_per_minute_position_adjusted_duration": _pearson(
                    rate_pos, frame["duration_seconds"]
                ),
                "position_r2_raw": position_r_squared(frame, column),
                "position_r2_per_minute": position_r_squared(
                    frame.assign(_per_min=per_min), "_per_min"
                ),
            }
        )
    return pd.DataFrame(rows)


def winner_loser_by_position_table(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for column in columns:
        gap = _winner_loser_gap(frame, column)
        gap.insert(0, "field", column)
        parts.append(gap)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _winner_loser_gap(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = _numeric(frame[column])
    eligible = explicit_position_mask(frame) & values.notna()
    for number in EXPLICIT_POSITION_NUMBERS:
        subset = frame.loc[eligible & (frame["position_number"] == number)]
        won = _numeric(subset.loc[subset["team_won"] == 1, column])
        lost = _numeric(subset.loc[subset["team_won"] == 0, column])
        rows.append(
            {
                "position_number": number,
                "n_winners": int(won.notna().sum()),
                "n_losers": int(lost.notna().sum()),
                "mean_winners": float(won.mean())
                if won.notna().any()
                else float("nan"),
                "mean_losers": float(lost.mean())
                if lost.notna().any()
                else float("nan"),
                "mean_gap": (
                    float(won.mean() - lost.mean())
                    if won.notna().any() and lost.notna().any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def candidate_position_means_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean of each candidate by explicit position after adjustment.

    Position-standardized targets should be near zero in every role.
    """
    rows: list[dict[str, object]] = []
    eligible = explicit_position_mask(frame)
    for spec in CANDIDATE_SPECS:
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


def outcome_contamination_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in CANDIDATE_SPECS:
        values = _numeric(frame[spec.name])
        pos_r2 = position_r_squared(frame.assign(_c=values), "_c")
        elo_beta = slope_coefficient(values, frame["elo_expected_win"])
        after_elo = elo_residualized(values, frame["elo_expected_win"])
        rows.append(
            {
                "candidate": spec.name,
                "family": spec.family,
                "n": int(values.notna().sum()),
                "pearson_team_won": _pearson(values, frame["team_won"]),
                "spearman_team_won": _spearman(values, frame["team_won"]),
                "pearson_elo_expected_win": _pearson(values, frame["elo_expected_win"]),
                "spearman_elo_expected_win": _spearman(
                    values, frame["elo_expected_win"]
                ),
                "elo_slope": elo_beta,
                "pearson_team_won_after_elo": _pearson(after_elo, frame["team_won"]),
                "pearson_duration": _pearson(values, frame["duration_seconds"]),
                "position_r2": pos_r2,
                "mean": float(values.mean()) if values.notna().any() else float("nan"),
                "std": _std(values.dropna().to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def hero_dependence_tables(
    frame: pd.DataFrame, *, min_hero_n: int = MIN_HERO_APPEARANCES
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = (
        frame.groupby("hero_id", dropna=False)
        .size()
        .rename("n_appearances")
        .reset_index()
        .sort_values("n_appearances", ascending=False, kind="mergesort")
    )
    within_rows: list[dict[str, object]] = []
    variance_rows: list[dict[str, object]] = []
    eligible = frame.loc[explicit_position_mask(frame)].copy()
    for spec in CANDIDATE_SPECS:
        values = _numeric(eligible[spec.name])
        work = eligible.loc[values.notna(), ["hero_id", "position_number", spec.name]]
        for number in EXPLICIT_POSITION_NUMBERS:
            subset = work.loc[work["position_number"] == number]
            hero_n = subset.groupby("hero_id").size()
            keep = hero_n[hero_n >= min_hero_n].index
            for hero_id in keep:
                hero_values = _numeric(
                    subset.loc[subset["hero_id"] == hero_id, spec.name]
                )
                within_rows.append(
                    {
                        "candidate": spec.name,
                        "position_number": int(number),
                        "hero_id": int(hero_id),
                        "n": int(hero_values.notna().sum()),
                        "mean": float(hero_values.mean()),
                    }
                )
        hero_counts_for_candidate = work.groupby("hero_id").size()
        n_heroes_ge_min = int((hero_counts_for_candidate >= min_hero_n).sum())
        if work.empty:
            variance_rows.append(
                {
                    "candidate": spec.name,
                    "hero_r2_overall": float("nan"),
                    "hero_r2_within_position": float("nan"),
                    "n_heroes_ge_min": 0,
                }
            )
            continue
        overall = _categorical_r2(work[spec.name], work["hero_id"])
        residual = position_adjusted(
            eligible.assign(_c=eligible[spec.name]),
            "_c",
        )
        residual_work = eligible.assign(_resid=residual, _hero=eligible["hero_id"])
        residual_work = residual_work.loc[residual_work["_resid"].notna()]
        within = _categorical_r2(residual_work["_resid"], residual_work["_hero"])
        variance_rows.append(
            {
                "candidate": spec.name,
                "hero_r2_overall": overall,
                "hero_r2_within_position": within,
                "n_heroes_ge_min": n_heroes_ge_min,
            }
        )
    return (
        counts,
        pd.DataFrame(within_rows),
        pd.DataFrame(variance_rows),
    )


def _categorical_r2(values: pd.Series, labels: pd.Series) -> float:
    y = _numeric(values)
    mask = y.notna() & labels.notna()
    yv = y.loc[mask].to_numpy(dtype=float)
    if yv.size < 2:
        return float("nan")
    grand = float(yv.mean())
    ss_tot = float(np.sum((yv - grand) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_within = 0.0
    for _, group in y.loc[mask].groupby(labels.loc[mask]):
        arr = group.to_numpy(dtype=float)
        ss_within += float(np.sum((arr - arr.mean()) ** 2))
    return 1.0 - ss_within / ss_tot


def redundancy_tables(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primitives = list(PCA_PRIMITIVE_COLUMNS)
    raw = frame.loc[explicit_position_mask(frame), primitives].apply(
        pd.to_numeric, errors="coerce"
    )
    corr = raw.corr(method="pearson")
    adjusted = pd.DataFrame(
        {column: position_adjusted(frame, column) for column in primitives},
        index=frame.index,
    )
    within = adjusted.loc[explicit_position_mask(frame)].corr(method="pearson")
    complete = adjusted.loc[explicit_position_mask(frame)].dropna()
    pca_rows: list[dict[str, object]] = []
    if len(complete) >= len(primitives):
        standardized = (complete - complete.mean()) / complete.std(ddof=0).replace(
            0.0, np.nan
        )
        standardized = standardized.dropna(axis=1)
        if standardized.shape[1] >= 2:
            pca = PCA()
            pca.fit(standardized.to_numpy(dtype=float))
            loadings = pd.DataFrame(
                pca.components_.T,
                index=list(standardized.columns),
                columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
            )
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                row: dict[str, object] = {
                    "component": f"PC{i + 1}",
                    "explained_variance_ratio": float(ratio),
                    "cumulative": float(pca.explained_variance_ratio_[: i + 1].sum()),
                }
                for column in loadings.index:
                    row[f"load_{column}"] = float(loadings.loc[column, f"PC{i + 1}"])
                pca_rows.append(row)
    return (
        corr.reset_index().rename(columns={"index": "field"}),
        within.reset_index().rename(columns={"index": "field"}),
        pd.DataFrame(pca_rows),
    )


def prior_player_history(
    frame: pd.DataFrame, column: str
) -> tuple[pd.Series, pd.Series]:
    """Strictly prior player mean of ``column``. Same-timestamp rows are excluded."""
    values = _numeric(frame[column])
    means = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0, index=frame.index, dtype=int)
    if frame.empty:
        return means, counts
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
        for i in range(len(ordered)):
            prior = (stamps < stamps[i]) & np.isfinite(vals)
            player_counts[i] = int(prior.sum())
            if player_counts[i] > 0:
                player_means[i] = float(vals[prior].mean())
        means.loc[ordered.index] = player_means
        counts.loc[ordered.index] = player_counts
    return means, counts


def repeatability_by_min_prior(
    frame: pd.DataFrame,
    column: str,
    *,
    thresholds: tuple[int, ...] = REPEATABILITY_PRIOR_THRESHOLDS,
) -> pd.DataFrame:
    prior_mean, prior_n = prior_player_history(frame, column)
    current = _numeric(frame[column])
    rows: list[dict[str, object]] = []
    for minimum in thresholds:
        mask = (prior_n >= minimum) & current.notna() & prior_mean.notna()
        rows.append(
            {
                "candidate": column,
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


def first_half_second_half_correlation(
    frame: pd.DataFrame,
    column: str,
    *,
    min_each: int = MIN_HALF_APPEARANCES,
) -> dict[str, object]:
    values = _numeric(frame[column])
    work = frame.loc[values.notna(), ["player_id", "start_time", column]].copy()
    work[column] = _numeric(work[column])
    work["start_time"] = pd.to_datetime(work["start_time"], utc=True)
    early_means: list[float] = []
    late_means: list[float] = []
    n_paired = 0
    for _player_id, group in work.groupby("player_id", sort=False):
        ordered = group.sort_values("start_time", kind="mergesort")
        n = len(ordered)
        split = n // 2
        if split < min_each or (n - split) < min_each:
            continue
        early = ordered.iloc[:split][column]
        late = ordered.iloc[split:][column]
        early_means.append(float(early.mean()))
        late_means.append(float(late.mean()))
        n_paired += 1
    return {
        "candidate": column,
        "n_paired_players": n_paired,
        "min_each_half": min_each,
        "pearson": _pearson(pd.Series(early_means), pd.Series(late_means))
        if n_paired
        else float("nan"),
        "spearman": _spearman(pd.Series(early_means), pd.Series(late_means))
        if n_paired
        else float("nan"),
    }


def _loo_team_match_residual(frame: pd.DataFrame, column: str) -> pd.Series:
    """Appearance minus the other four teammates' mean on the same match/side."""
    values = _numeric(frame[column])
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    work = frame.assign(_v=values)
    grouped = work.groupby(["match_id", "side"], sort=False)["_v"]
    total = grouped.transform("sum")
    count = grouped.transform("count")
    others = (total - values) / (count - 1)
    out.loc[values.notna() & (count > 1)] = (values - others).loc[
        values.notna() & (count > 1)
    ]
    return out


def _player_mean_vs_team_mean(frame: pd.DataFrame, column: str) -> dict[str, object]:
    values = _numeric(frame[column])
    work = frame.loc[values.notna(), ["player_id", "team_id", column]].copy()
    work[column] = _numeric(work[column])
    player_means = work.groupby("player_id")[column].mean()
    team_means = work.groupby("team_id")[column].mean()
    joined = work[["player_id", "team_id"]].drop_duplicates()
    joined = joined.merge(player_means.rename("player_mean"), on="player_id")
    joined = joined.merge(team_means.rename("team_mean"), on="team_id")
    return {
        "n_player_team_pairs": len(joined),
        "pearson_player_mean_vs_team_mean": _pearson(
            joined["player_mean"], joined["team_mean"]
        ),
    }


def recommend_candidates(
    quality: pd.DataFrame, *, max_recommend: int = 3
) -> pd.DataFrame:
    """Select at most a few defensible Slice 13 research candidates.

    Does not force a positive conclusion. A candidate must be
    approximately role-neutral and show chronological player
    repeatability above a modest floor. Victory correlation is reported
    as a warning, not used to choose weights.
    """
    if quality.empty:
        return pd.DataFrame()
    work = quality.copy()
    work["role_neutral"] = _numeric(work["position_r2"]).abs() <= _POSITION_R2_NEUTRAL
    work["repeatable"] = (
        _numeric(work["repeatability_pearson_min_prior_10"]) >= _REPEATABILITY_FLOOR
    )
    work["duration_ok"] = (
        _numeric(work["pearson_duration"]).abs() <= _DURATION_CORR_NEUTRAL
    )
    work["win_not_disguised"] = (
        _numeric(work["pearson_team_won"]).abs() < _WIN_CORR_DISGUISED
    )
    work["passes"] = (
        work["role_neutral"] & work["repeatable"] & work["win_not_disguised"]
    )
    ranked = work.sort_values(
        by=["passes", "repeatability_pearson_min_prior_10"],
        ascending=[False, False],
        kind="mergesort",
    )
    chosen = ranked.loc[ranked["passes"]].head(max_recommend)
    if chosen.empty:
        return pd.DataFrame(
            [
                {
                    "candidate": None,
                    "recommended": False,
                    "reason": (
                        "No candidate is sufficiently repeatable and "
                        "role-neutral to justify a player-performance rating."
                    ),
                    "formula": None,
                    "adjustment_variables": None,
                    "hero_agnostic": None,
                }
            ]
        )
    rows: list[dict[str, object]] = []
    for _, row in chosen.iterrows():
        spec = _CANDIDATE_BY_NAME[str(row["candidate"])]
        rows.append(
            {
                "candidate": spec.name,
                "recommended": True,
                "reason": "role-neutral with chronological player repeatability",
                "formula": spec.formula,
                "adjustment_variables": ", ".join(spec.adjustment_variables),
                "hero_agnostic": spec.hero_agnostic,
                "family": spec.family,
                "pearson_team_won": row["pearson_team_won"],
                "pearson_elo_expected_win": row["pearson_elo_expected_win"],
                "position_r2": row["position_r2"],
                "pearson_duration": row["pearson_duration"],
                "repeatability_pearson_min_prior_10": row[
                    "repeatability_pearson_min_prior_10"
                ],
                "n": row["n"],
                "major_weaknesses": row.get("major_weaknesses"),
            }
        )
    return pd.DataFrame(rows)


def _candidate_quality_table(
    contamination: pd.DataFrame,
    repeatability: pd.DataFrame,
    halves: pd.DataFrame,
    hero_variance: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_candidate_rep = {
        name: group for name, group in repeatability.groupby("candidate")
    }
    half_by = halves.set_index("candidate") if not halves.empty else pd.DataFrame()
    hero_by = (
        hero_variance.set_index("candidate")
        if not hero_variance.empty
        else pd.DataFrame()
    )
    for spec in CANDIDATE_SPECS:
        cont = contamination.loc[contamination["candidate"] == spec.name].iloc[0]
        rep = by_candidate_rep.get(spec.name)
        r10 = float("nan")
        n10 = 0
        if rep is not None:
            at10 = rep.loc[rep["min_prior_appearances"] == MIN_PRIOR_APPEARANCES]
            if not at10.empty:
                r10 = float(at10.iloc[0]["pearson"])
                n10 = int(at10.iloc[0]["n_rows"])
        half_r = (
            float(half_by.loc[spec.name, "pearson"])
            if spec.name in half_by.index
            else float("nan")
        )
        hero_r2 = (
            float(hero_by.loc[spec.name, "hero_r2_within_position"])
            if spec.name in hero_by.index
            else float("nan")
        )
        weaknesses: list[str] = []
        if abs(float(cont["position_r2"])) > _POSITION_R2_NEUTRAL and np.isfinite(
            float(cont["position_r2"])
        ):
            weaknesses.append("residual position dependence")
        if abs(
            float(cont["pearson_duration"])
        ) > _DURATION_CORR_NEUTRAL and np.isfinite(float(cont["pearson_duration"])):
            weaknesses.append("residual duration dependence")
        if abs(float(cont["pearson_team_won"])) >= _WIN_CORR_DISGUISED and np.isfinite(
            float(cont["pearson_team_won"])
        ):
            weaknesses.append("high victory correlation")
        if not (np.isfinite(r10) and r10 >= _REPEATABILITY_FLOOR):
            weaknesses.append("weak chronological repeatability")
        if np.isfinite(hero_r2) and hero_r2 >= 0.20:
            weaknesses.append("hero-kit dependence")
        rows.append(
            {
                "candidate": spec.name,
                "family": spec.family,
                "formula": spec.formula,
                "adjustment_variables": ", ".join(spec.adjustment_variables),
                "hero_agnostic": spec.hero_agnostic,
                "n": int(cont["n"]),
                "mean": cont["mean"],
                "position_r2": cont["position_r2"],
                "pearson_duration": cont["pearson_duration"],
                "pearson_team_won": cont["pearson_team_won"],
                "pearson_elo_expected_win": cont["pearson_elo_expected_win"],
                "elo_slope": cont["elo_slope"],
                "repeatability_pearson_min_prior_10": r10,
                "repeatability_n_min_prior_10": n10,
                "first_half_second_half_pearson": half_r,
                "hero_r2_within_position": hero_r2,
                "major_weaknesses": "; ".join(weaknesses) if weaknesses else "",
            }
        )
    return pd.DataFrame(rows)


def run_player_performance_target_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice12DiagnosticReport:
    """Development-only Slice 12 target research. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    development = attach_candidate_targets(development)
    n_missing_position = int((~explicit_position_mask(development)).sum())
    coverage = pd.DataFrame(
        [
            {
                "n_development_matches": int(development["match_id"].nunique()),
                "n_development_player_rows": len(development),
                "n_holdout_excluded": len(holdout),
                "n_missing_position": n_missing_position,
                "expected_player_rows_if_ten_per_match": int(
                    development["match_id"].nunique() * 10
                ),
            }
        ]
    )
    raw = raw_field_diagnostics(development)
    extremes = _extreme_rows(development)
    role = position_dependence_table(development)
    duration = duration_dependence_table(development)
    with_candidates = development
    contamination = outcome_contamination_table(with_candidates)
    candidate_position_means = candidate_position_means_table(with_candidates)
    hero_counts, hero_within, hero_var = hero_dependence_tables(with_candidates)
    corr, within_corr, pca = redundancy_tables(with_candidates)
    repeat_parts = [
        repeatability_by_min_prior(with_candidates, spec.name)
        for spec in CANDIDATE_SPECS
    ]
    repeatability = pd.concat(repeat_parts, ignore_index=True)
    halves = pd.DataFrame(
        [
            first_half_second_half_correlation(with_candidates, spec.name)
            for spec in CANDIDATE_SPECS
        ]
    )
    quality = _candidate_quality_table(contamination, repeatability, halves, hero_var)
    patch_rows: list[dict[str, object]] = []
    for version, group in with_candidates.groupby("game_version_id", dropna=False):
        for spec in CANDIDATE_SPECS:
            values = _numeric(group[spec.name]).dropna().to_numpy(dtype=float)
            patch_rows.append(
                {
                    "game_version_id": version,
                    "candidate": spec.name,
                    "n": int(values.size),
                    "mean": float(values.mean()) if values.size else float("nan"),
                    "std": _std(values),
                    "median": float(np.median(values)) if values.size else float("nan"),
                }
            )
    patch_stability = pd.DataFrame(patch_rows)

    falsification_rows: list[dict[str, object]] = []
    for spec in CANDIDATE_SPECS:
        values = _numeric(with_candidates[spec.name])
        hero_resid = values - values.groupby(with_candidates["hero_id"]).transform(
            "mean"
        )
        hero_frame = with_candidates.assign(_h=hero_resid)
        hero_rep = repeatability_by_min_prior(hero_frame, "_h")
        hero_r10 = hero_rep.loc[
            hero_rep["min_prior_appearances"] == MIN_PRIOR_APPEARANCES, "pearson"
        ]
        loo = _loo_team_match_residual(with_candidates, spec.name)
        loo_frame = with_candidates.assign(_t=loo)
        team_rep = repeatability_by_min_prior(loo_frame, "_t")
        team_r10 = team_rep.loc[
            team_rep["min_prior_appearances"] == MIN_PRIOR_APPEARANCES, "pearson"
        ]
        winners = with_candidates.loc[with_candidates["team_won"] == 1]
        losers = with_candidates.loc[with_candidates["team_won"] == 0]
        win_rep = repeatability_by_min_prior(winners, spec.name)
        lose_rep = repeatability_by_min_prior(losers, spec.name)
        player_team = _player_mean_vs_team_mean(with_candidates, spec.name)
        version_r: list[float] = []
        for _version, group in with_candidates.groupby("game_version_id", dropna=False):
            if len(group) < 50:
                continue
            version_rep = repeatability_by_min_prior(group, spec.name)
            at10 = version_rep.loc[
                version_rep["min_prior_appearances"] == MIN_PRIOR_APPEARANCES, "pearson"
            ]
            if not at10.empty and np.isfinite(float(at10.iloc[0])):
                version_r.append(float(at10.iloc[0]))
        falsification_rows.append(
            {
                "candidate": spec.name,
                "repeatability_after_hero_demean": (
                    float(hero_r10.iloc[0]) if not hero_r10.empty else float("nan")
                ),
                "repeatability_loo_teammate_residual": (
                    float(team_r10.iloc[0]) if not team_r10.empty else float("nan")
                ),
                "repeatability_winners_only": float(
                    win_rep.loc[
                        win_rep["min_prior_appearances"] == MIN_PRIOR_APPEARANCES,
                        "pearson",
                    ].iloc[0]
                )
                if not win_rep.empty
                else float("nan"),
                "repeatability_losers_only": float(
                    lose_rep.loc[
                        lose_rep["min_prior_appearances"] == MIN_PRIOR_APPEARANCES,
                        "pearson",
                    ].iloc[0]
                )
                if not lose_rep.empty
                else float("nan"),
                "mean_within_version_repeatability": (
                    float(np.mean(version_r)) if version_r else float("nan")
                ),
                **player_team,
            }
        )
    falsification = pd.DataFrame(falsification_rows)
    recommendations = recommend_candidates(quality)

    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    integrity = {
        "development_end": end.isoformat(),
        "ti2026_used_for_target_definition": False,
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "candidate_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in CANDIDATE_COLUMN_NAMES
        ),
        "candidate_in_all_feature_columns": any(
            name in ALL_FEATURE_COLUMNS for name in CANDIDATE_COLUMN_NAMES
        ),
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "elo_implementation": "match_elo_expected_wins",
        "null_distinct_from_zero": True,
        "missing_position_imputed": False,
        "current_match_box_score_used_as_feature": False,
        "win_outcome_subtracted_from_performance": False,
        "player_rating_persisted": False,
        "model_trained": False,
    }
    return Slice12DiagnosticReport(
        development_end=end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        n_missing_position=n_missing_position,
        coverage=coverage,
        raw_diagnostics=raw,
        extremes=extremes,
        role_dependence=role,
        duration_dependence=duration,
        outcome_contamination=contamination,
        hero_sample_sizes=hero_counts,
        hero_within_position=hero_within,
        hero_variance=hero_var,
        correlation_matrix=corr,
        within_position_correlation_matrix=within_corr,
        pca=pca,
        winner_loser_by_position=winner_loser_by_position_table(
            with_candidates,
            BOX_SCORE_COLUMNS + CANDIDATE_COLUMN_NAMES,
        ),
        candidate_position_means=candidate_position_means,
        candidate_quality=quality,
        repeatability=repeatability,
        first_half_second_half=halves,
        patch_stability=patch_stability,
        falsification=falsification,
        recommendations=recommendations,
        integrity=integrity,
    )


def _jsonable_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.DataFrame):
        return [
            {str(column): _jsonable_value(cell) for column, cell in row.items()}
            for row in value.to_dict(orient="records")
        ]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


def slice12_report_to_jsonable(report: Slice12DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 12 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "n_missing_position": report.n_missing_position,
        "coverage": _jsonable_value(report.coverage),
        "raw_diagnostics": _jsonable_value(report.raw_diagnostics),
        "extremes": _jsonable_value(report.extremes),
        "role_dependence": _jsonable_value(report.role_dependence),
        "duration_dependence": _jsonable_value(report.duration_dependence),
        "outcome_contamination": _jsonable_value(report.outcome_contamination),
        "hero_sample_sizes": _jsonable_value(report.hero_sample_sizes),
        "hero_within_position": _jsonable_value(report.hero_within_position),
        "hero_variance": _jsonable_value(report.hero_variance),
        "correlation_matrix": _jsonable_value(report.correlation_matrix),
        "within_position_correlation_matrix": _jsonable_value(
            report.within_position_correlation_matrix
        ),
        "pca": _jsonable_value(report.pca),
        "winner_loser_by_position": _jsonable_value(report.winner_loser_by_position),
        "candidate_position_means": _jsonable_value(report.candidate_position_means),
        "candidate_quality": _jsonable_value(report.candidate_quality),
        "repeatability": _jsonable_value(report.repeatability),
        "first_half_second_half": _jsonable_value(report.first_half_second_half),
        "patch_stability": _jsonable_value(report.patch_stability),
        "falsification": _jsonable_value(report.falsification),
        "recommendations": _jsonable_value(report.recommendations),
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
            for spec in CANDIDATE_SPECS
        ],
    }
