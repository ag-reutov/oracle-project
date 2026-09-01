"""Diagnostic summaries of walk-forward OOS deltas by patch maturity.

Joins existing walk-forward per-match predictions to descriptive
patch-age context. Does not retrain models, does not add columns to
any training feature set, and does not choose bins from metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dota_predictor.features.duckdb_layer import FeatureDuckDBConnection
from dota_predictor.features.patch_maturity import (
    CALENDAR_AGE_BIN_ORDER,
    PRIOR_MATCH_BIN_ORDER,
    assign_calendar_age_bin,
    assign_prior_match_bin,
)
from dota_predictor.features.player_hero import player_hero_sql
from dota_predictor.features.team_hero import team_hero_sql
from dota_predictor.training.feature_sets import POST_DRAFT_BLOCK_ABLATION_SPECS
from dota_predictor.training.walk_forward import ELO_BLOCK_SPEC_NAME

__all__ = [
    "DRAFT_BLOCK_SPEC_NAMES",
    "PatchMaturityDiagnosticReport",
    "attach_patch_maturity",
    "build_familiarity_mix",
    "run_patch_maturity_diagnostics",
]

DRAFT_BLOCK_SPEC_NAMES: tuple[str, ...] = tuple(
    spec.name
    for spec in POST_DRAFT_BLOCK_ABLATION_SPECS
    if spec.name != ELO_BLOCK_SPEC_NAME
)

_MIN_AUC_N = 30


def _finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(
        y.to_numpy(dtype=float)
    )
    return x.to_numpy(dtype=float)[mask], y.to_numpy(dtype=float)[mask]


def _pearson(x: pd.Series, y: pd.Series) -> float:
    xv, yv = _finite_pair(x, y)
    if xv.size < 3 or np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    xv, yv = _finite_pair(x, y)
    if xv.size < 3 or np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return float("nan")
    rx = pd.Series(xv).rank(method="average")
    ry = pd.Series(yv).rank(method="average")
    return _pearson(rx, ry)


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    n = int(values.size)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan")
    return mean, float(values.std(ddof=1) / np.sqrt(n))


def _auc_delta(y: pd.Series, p_spec: pd.Series, p_elo: pd.Series) -> float:
    labels = np.asarray(y, dtype=int)
    if labels.size < _MIN_AUC_N or np.unique(labels).size < 2:
        return float("nan")
    try:
        return float(
            roc_auc_score(labels, p_spec) - roc_auc_score(labels, p_elo)
        )
    except ValueError:
        return float("nan")


def _paired_bin_row(
    subset: pd.DataFrame, *, bin_name: str, spec_name: str, spec_label: str
) -> dict[str, object]:
    delta = subset["delta_vs_elo"].to_numpy(dtype=float)
    mean_delta, se_delta = _mean_se(delta)
    return {
        "bin": bin_name,
        "model": spec_name,
        "label": spec_label,
        "n": len(subset),
        "mean_delta_vs_elo": mean_delta,
        "se_delta_vs_elo": se_delta,
        "median_delta_vs_elo": (
            float(np.median(delta)) if len(delta) else float("nan")
        ),
        "auc_delta_vs_elo": _auc_delta(
            subset["y_true"], subset["p_spec"], subset["p_elo"]
        ),
    }


def attach_patch_maturity(
    oos_predictions: pd.DataFrame, maturity: pd.DataFrame
) -> pd.DataFrame:
    """Join walk-forward OOS rows to patch-age context and assign bins."""
    context = maturity[
        [
            "match_id",
            "days_since_game_version_start",
            "prior_matches_in_game_version",
            "game_version_name",
            "as_of_datetime",
        ]
    ]
    joined = oos_predictions.merge(
        context, on="match_id", how="left", validate="many_to_one"
    )
    missing_prior = joined["prior_matches_in_game_version"].isna()
    if bool(missing_prior.any()):
        raise ValueError(
            "OOS predictions are missing prior_matches_in_game_version "
            f"for {int(missing_prior.sum())} rows after the maturity join"
        )
    joined["calendar_age_bin"] = [
        assign_calendar_age_bin(
            None if pd.isna(value) else float(value)
        )
        for value in joined["days_since_game_version_start"]
    ]
    joined["prior_match_bin"] = [
        assign_prior_match_bin(int(value))
        for value in joined["prior_matches_in_game_version"]
    ]
    return joined


def _bin_summary(
    joined: pd.DataFrame, *, bin_column: str, bin_order: tuple[str, ...]
) -> pd.DataFrame:
    labels = {spec.name: spec.label for spec in POST_DRAFT_BLOCK_ABLATION_SPECS}
    rows: list[dict[str, object]] = []
    for spec_name in DRAFT_BLOCK_SPEC_NAMES:
        spec_rows = joined.loc[joined["model"] == spec_name]
        for bin_name in bin_order:
            subset = spec_rows.loc[spec_rows[bin_column] == bin_name]
            rows.append(
                _paired_bin_row(
                    subset,
                    bin_name=bin_name,
                    spec_name=spec_name,
                    spec_label=labels[spec_name],
                )
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["bin"] = pd.Categorical(
        frame["bin"], categories=list(bin_order), ordered=True
    )
    return frame.sort_values(["bin", "model"], kind="stable").reset_index(
        drop=True
    )


def _version_matrix(joined: pd.DataFrame) -> pd.DataFrame:
    labels = {spec.name: spec.label for spec in POST_DRAFT_BLOCK_ABLATION_SPECS}
    rows: list[dict[str, object]] = []
    versions = joined[["game_version_id", "game_version_name"]].drop_duplicates()
    versions = versions.sort_values("game_version_id")
    for _, version in versions.iterrows():
        version_id = version["game_version_id"]
        name = version["game_version_name"]
        for spec_name in DRAFT_BLOCK_SPEC_NAMES:
            subset = joined.loc[
                (joined["game_version_id"] == version_id)
                & (joined["model"] == spec_name)
            ]
            if subset.empty:
                continue
            row = _paired_bin_row(
                subset,
                bin_name=str(name) if pd.notna(name) else str(version_id),
                spec_name=spec_name,
                spec_label=labels[spec_name],
            )
            row["game_version_id"] = version_id
            row["game_version"] = name
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["game_version_id", "model"], kind="stable"
    ).reset_index(drop=True)


def _quantile_means(
    joined: pd.DataFrame, *, x_column: str, n_quantiles: int = 8
) -> pd.DataFrame:
    labels = {spec.name: spec.label for spec in POST_DRAFT_BLOCK_ABLATION_SPECS}
    rows: list[dict[str, object]] = []
    for spec_name in DRAFT_BLOCK_SPEC_NAMES:
        spec_rows = joined.loc[joined["model"] == spec_name].copy()
        x = spec_rows[x_column]
        valid = x.notna()
        spec_rows = spec_rows.loc[valid]
        if spec_rows.empty:
            continue
        try:
            spec_rows["quantile"] = pd.qcut(
                spec_rows[x_column],
                q=n_quantiles,
                duplicates="drop",
            )
        except ValueError:
            continue
        for interval, subset in spec_rows.groupby("quantile", observed=True):
            mean_delta, se_delta = _mean_se(
                subset["delta_vs_elo"].to_numpy(dtype=float)
            )
            left = float(interval.left)
            right = float(interval.right)
            rows.append(
                {
                    "model": spec_name,
                    "label": labels[spec_name],
                    "x_column": x_column,
                    "quantile_left": left,
                    "quantile_right": right,
                    "quantile_mid": (left + right) / 2.0,
                    "n": len(subset),
                    "mean_delta_vs_elo": mean_delta,
                    "se_delta_vs_elo": se_delta,
                }
            )
    return pd.DataFrame(rows)


def _correlations(joined: pd.DataFrame) -> pd.DataFrame:
    labels = {spec.name: spec.label for spec in POST_DRAFT_BLOCK_ABLATION_SPECS}
    rows: list[dict[str, object]] = []
    for spec_name in DRAFT_BLOCK_SPEC_NAMES:
        subset = joined.loc[joined["model"] == spec_name]
        for x_column, x_name in (
            ("days_since_game_version_start", "calendar_age"),
            ("prior_matches_in_game_version", "professional_maturity"),
        ):
            rows.append(
                {
                    "model": spec_name,
                    "label": labels[spec_name],
                    "x": x_name,
                    "n": int(subset[x_column].notna().sum()),
                    "pearson": _pearson(subset[x_column], subset["delta_vs_elo"]),
                    "spearman": _spearman(
                        subset[x_column], subset["delta_vs_elo"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_familiarity_mix(
    store: FeatureDuckDBConnection, *, catalog_registered: bool = True
) -> pd.DataFrame:
    """Match-level Radiant − Dire lifetime / same-version / recent-90d diffs.

    Aggregates existing Player × Hero and Team × Hero metrics. Diagnostic
    only — not a training feature.
    """
    player_sql = player_hero_sql(
        catalog_registered=catalog_registered, match_id=None
    )
    team_sql = team_hero_sql(
        catalog_registered=catalog_registered, match_id=None
    )
    sql = f"""
WITH player_hero AS (
    SELECT * FROM ({player_sql}) AS player_hero_inner
),
team_hero AS (
    SELECT * FROM ({team_sql}) AS team_hero_inner
),
player_side AS (
    SELECT
        match_id,
        side,
        AVG(prior_games_on_hero)::DOUBLE AS mean_prior_games,
        AVG(same_version_games_on_hero)::DOUBLE AS mean_same_version_games,
        AVG(recent_90d_games_on_hero)::DOUBLE AS mean_recent_90d_games
    FROM player_hero
    GROUP BY match_id, side
),
team_side AS (
    SELECT
        match_id,
        side,
        AVG(team_prior_games_with_hero)::DOUBLE AS mean_prior_games,
        AVG(same_version_team_games_with_hero)::DOUBLE
            AS mean_same_version_games,
        AVG(recent_90d_team_games_with_hero)::DOUBLE AS mean_recent_90d_games
    FROM team_hero
    GROUP BY match_id, side
)

SELECT
    rp.match_id,
    (rp.mean_prior_games - dp.mean_prior_games)
        AS player_prior_games_diff,
    (rp.mean_same_version_games - dp.mean_same_version_games)
        AS player_same_version_games_diff,
    (rp.mean_recent_90d_games - dp.mean_recent_90d_games)
        AS player_recent_90d_games_diff,
    (rt.mean_prior_games - dt.mean_prior_games)
        AS team_prior_games_diff,
    (rt.mean_same_version_games - dt.mean_same_version_games)
        AS team_same_version_games_diff,
    (rt.mean_recent_90d_games - dt.mean_recent_90d_games)
        AS team_recent_90d_games_diff
FROM player_side AS rp
INNER JOIN player_side AS dp
    ON rp.match_id = dp.match_id
    AND rp.side = 'RADIANT'
    AND dp.side = 'DIRE'
INNER JOIN team_side AS rt
    ON rt.match_id = rp.match_id AND rt.side = 'RADIANT'
INNER JOIN team_side AS dt
    ON dt.match_id = rp.match_id AND dt.side = 'DIRE'
"""
    return store.sql(sql).df()


def _carryover_rows(
    joined: pd.DataFrame,
    mix: pd.DataFrame,
    *,
    spec_name: str,
    spec_label: str,
    prefix: str,
) -> list[dict[str, object]]:
    subset = joined.loc[joined["model"] == spec_name].merge(
        mix, on="match_id", how="left", validate="many_to_one"
    )
    lifetime = subset[f"{prefix}_prior_games_diff"]
    same_version = subset[f"{prefix}_same_version_games_diff"]
    stale = lifetime - same_version
    subset = subset.assign(stale_games_diff=stale)
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all_oos", subset)]
    for bin_name in CALENDAR_AGE_BIN_ORDER:
        scoped = subset.loc[subset["calendar_age_bin"] == bin_name]
        if not scoped.empty:
            scopes.append((bin_name, scoped))
    for scope_name, scoped in scopes:
        rows.append(
            {
                "model": spec_name,
                "label": spec_label,
                "scope": scope_name,
                "n": len(scoped),
                "pearson_delta_vs_lifetime": _pearson(
                    scoped[f"{prefix}_prior_games_diff"],
                    scoped["delta_vs_elo"],
                ),
                "pearson_delta_vs_same_version": _pearson(
                    scoped[f"{prefix}_same_version_games_diff"],
                    scoped["delta_vs_elo"],
                ),
                "pearson_delta_vs_recent_90d": _pearson(
                    scoped[f"{prefix}_recent_90d_games_diff"],
                    scoped["delta_vs_elo"],
                ),
                "pearson_delta_vs_stale": _pearson(
                    scoped["stale_games_diff"], scoped["delta_vs_elo"]
                ),
                "mean_lifetime_diff": float(
                    scoped[f"{prefix}_prior_games_diff"].mean()
                ),
                "mean_same_version_diff": float(
                    scoped[f"{prefix}_same_version_games_diff"].mean()
                ),
                "mean_recent_90d_diff": float(
                    scoped[f"{prefix}_recent_90d_games_diff"].mean()
                ),
                "mean_stale_diff": float(scoped["stale_games_diff"].mean()),
                "mean_abs_lifetime_diff": float(
                    scoped[f"{prefix}_prior_games_diff"].abs().mean()
                ),
                "mean_abs_same_version_diff": float(
                    scoped[f"{prefix}_same_version_games_diff"].abs().mean()
                ),
                "mean_abs_stale_diff": float(
                    scoped["stale_games_diff"].abs().mean()
                ),
            }
        )
    return rows


@dataclass
class PatchMaturityDiagnosticReport:
    """All diagnostic tables for the patch-maturity study."""

    n_oos_matches: int
    sanity: pd.DataFrame
    calendar_bins: pd.DataFrame
    prior_match_bins: pd.DataFrame
    calendar_bin_counts: pd.DataFrame
    prior_match_bin_counts: pd.DataFrame
    version_matrix: pd.DataFrame
    correlations: pd.DataFrame
    calendar_quantile_means: pd.DataFrame
    prior_quantile_means: pd.DataFrame
    carryover: pd.DataFrame


def run_patch_maturity_diagnostics(
    oos_predictions: pd.DataFrame,
    maturity: pd.DataFrame,
    familiarity_mix: pd.DataFrame,
    sanity: pd.DataFrame,
) -> PatchMaturityDiagnosticReport:
    """Summarize existing OOS paired deltas by patch-age context."""
    joined = attach_patch_maturity(oos_predictions, maturity)
    one_row = joined.loc[joined["model"] == ELO_BLOCK_SPEC_NAME]
    n_oos = int(one_row["match_id"].nunique())

    def _counts(column: str, order: tuple[str, ...]) -> pd.DataFrame:
        counts = (
            one_row[column]
            .value_counts()
            .rename("n")
            .reindex(order)
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        counts.columns = ["bin", "n"]
        return counts

    labels = {spec.name: spec.label for spec in POST_DRAFT_BLOCK_ABLATION_SPECS}
    carryover_rows: list[dict[str, object]] = []
    carryover_rows.extend(
        _carryover_rows(
            joined,
            familiarity_mix,
            spec_name="logistic_elo_plus_player_hero",
            spec_label=labels["logistic_elo_plus_player_hero"],
            prefix="player",
        )
    )
    carryover_rows.extend(
        _carryover_rows(
            joined,
            familiarity_mix,
            spec_name="logistic_elo_plus_team_hero",
            spec_label=labels["logistic_elo_plus_team_hero"],
            prefix="team",
        )
    )

    return PatchMaturityDiagnosticReport(
        n_oos_matches=n_oos,
        sanity=sanity,
        calendar_bins=_bin_summary(
            joined,
            bin_column="calendar_age_bin",
            bin_order=CALENDAR_AGE_BIN_ORDER,
        ),
        prior_match_bins=_bin_summary(
            joined,
            bin_column="prior_match_bin",
            bin_order=PRIOR_MATCH_BIN_ORDER,
        ),
        calendar_bin_counts=_counts("calendar_age_bin", CALENDAR_AGE_BIN_ORDER),
        prior_match_bin_counts=_counts(
            "prior_match_bin", PRIOR_MATCH_BIN_ORDER
        ),
        version_matrix=_version_matrix(joined),
        correlations=_correlations(joined),
        calendar_quantile_means=_quantile_means(
            joined, x_column="days_since_game_version_start"
        ),
        prior_quantile_means=_quantile_means(
            joined, x_column="prior_matches_in_game_version"
        ),
        carryover=pd.DataFrame(carryover_rows),
    )
