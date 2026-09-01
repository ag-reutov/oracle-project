"""Development-only diagnostics for Slice 10 Elo-adjusted Player × Hero.

Describes residual strength, shrinkage, coverage, and the contrast with
Career P×H volume. Does not train a win model, does not add columns to
``FEATURE_COLUMNS``, and does not use TI 2026 to choose ``k`` or any
threshold.

The diagnostic population is matches with ``start_time <=`` the frozen
Slice 9 development end. Later matches, including TI 2026, are excluded
from every summary and from the method-of-moments ``k`` estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.player_hero import build_player_hero
from dota_predictor.features.player_hero_elo import (
    DEFAULT_SHRINKAGE_K,
    MATCH_ID_COLUMN,
    RESIDUAL_VARIANCE_PRIOR,
    TEAM_ELO_EXPECTED_VIEW,
    TRUE_EFFECT_SD_PRIOR,
    build_player_hero_elo,
    match_elo_expected_wins,
    shrinkage_weight,
)
from dota_predictor.features.player_hero_elo_comparison import (
    player_hero_elo_comparison_from_players,
)
from dota_predictor.features.team_elo import DEFAULT_ELO_CONFIG, EloConfig
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    holdout_mask,
    utc_datetime,
)

__all__ = [
    "MIN_GAMES_FOR_K_ESTIMATE",
    "ShrinkageKEstimate",
    "Slice10DiagnosticReport",
    "appearance_residuals",
    "estimate_shrinkage_k",
    "run_player_hero_elo_diagnostics",
    "slice10_report_to_jsonable",
]


MIN_GAMES_FOR_K_ESTIMATE = 8
_N_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1–2", 1, 2),
    ("3–4", 3, 4),
    ("5–9", 5, 9),
    ("10–19", 10, 19),
    ("20–39", 20, 39),
    ("40+", 40, None),
)


@dataclass(frozen=True)
class ShrinkageKEstimate:
    """Method-of-moments ``k`` from development completed cells only."""

    k: float
    residual_variance: float
    effect_variance: float
    n_cells: int
    n_appearances: int
    min_games_for_cell: int
    development_end: datetime
    used_for_state: bool


@dataclass(frozen=True)
class Slice10DiagnosticReport:
    shrinkage_k_frozen: float
    shrinkage_k_prior_note: str
    shrinkage_k_estimate: ShrinkageKEstimate
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    development_end: datetime
    residual_distribution: pd.DataFrame
    residual_by_n: pd.DataFrame
    high_volume_combinations: pd.DataFrame
    temporal_stability: pd.DataFrame
    volume_contrast: pd.DataFrame
    coverage: pd.DataFrame
    match_comparison_distribution: pd.DataFrame
    integrity: dict[str, object]


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _summary(values: np.ndarray) -> dict[str, object]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "frac_positive": float("nan"),
            "frac_negative": float("nan"),
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
        "frac_positive": float((values > 0).mean()),
        "frac_negative": float((values < 0).mean()),
    }


def _pearson(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    left = xv[mask].to_numpy(dtype=float)
    right = yv[mask].to_numpy(dtype=float)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def appearance_residuals(
    store: FeatureDuckDBConnection, *, elo_config: EloConfig = DEFAULT_ELO_CONFIG
) -> pd.DataFrame:
    """One row per completed player-match: ``y - e`` using pre-match Elo.

    ``y`` is that appearance's own result. This is diagnostic raw material
    for shrinkage-constant estimation and temporal stability, not a
    current-match feature.
    """
    matches = store.sql(
        f"""
        SELECT match_id, start_time, radiant_team_id, dire_team_id, radiant_win
        FROM {MATCHES_VIEW}
        """
    ).df()
    expected = match_elo_expected_wins(matches, config=elo_config)
    store.connection.register(TEAM_ELO_EXPECTED_VIEW, expected)
    return store.sql(
        f"""
        SELECT
            mp.match_id,
            mp.start_time,
            m.game_version_id,
            mp.player_id,
            mp.hero_id,
            mp.side,
            mp.team_id,
            CASE
                WHEN mp.side = 'RADIANT' THEN CAST(m.radiant_win AS INTEGER)
                ELSE CAST(NOT m.radiant_win AS INTEGER)
            END AS y,
            CASE
                WHEN mp.side = 'RADIANT' THEN e.radiant_expected_win
                ELSE e.dire_expected_win
            END AS elo_expected_win
        FROM {MATCH_PLAYERS_VIEW} mp
        JOIN {MATCHES_VIEW} m ON m.match_id = mp.match_id
        JOIN {TEAM_ELO_EXPECTED_VIEW} e ON e.match_id = mp.match_id
        """
    ).df()


def estimate_shrinkage_k(
    residuals: pd.DataFrame,
    *,
    development_end: datetime,
    min_games: int = MIN_GAMES_FOR_K_ESTIMATE,
) -> ShrinkageKEstimate:
    """Method-of-moments ``k`` on development completed player×hero cells.

    Uses only appearances with ``start_time <= development_end``. Cells
    with fewer than ``min_games`` appearances are excluded from the
    between-cell variance estimate. That floor is an estimation sample
    restriction, not a feature gate. TI 2026 must not be in this frame.
    """
    end = utc_datetime(development_end)
    stamp = pd.to_datetime(residuals["start_time"], utc=True)
    development = residuals.loc[stamp <= pd.Timestamp(end)].copy()
    development["residual"] = (
        pd.to_numeric(development["y"], errors="coerce")
        - pd.to_numeric(development["elo_expected_win"], errors="coerce")
    )
    expected = pd.to_numeric(development["elo_expected_win"], errors="coerce")
    bernoulli_var = expected * (1.0 - expected)
    cells = (
        development.groupby(["player_id", "hero_id"], sort=False)
        .agg(
            n=("residual", "size"),
            mean_residual=("residual", "mean"),
            mean_bernoulli_var=("elo_expected_win", lambda s: float(
                (pd.to_numeric(s, errors="coerce")
                 * (1.0 - pd.to_numeric(s, errors="coerce"))).mean()
            )),
        )
        .reset_index()
    )
    eligible = cells.loc[cells["n"] >= min_games]
    n_appearances = int(eligible["n"].sum()) if not eligible.empty else 0
    if len(eligible) < 8:
        return ShrinkageKEstimate(
            k=DEFAULT_SHRINKAGE_K,
            residual_variance=RESIDUAL_VARIANCE_PRIOR,
            effect_variance=TRUE_EFFECT_SD_PRIOR ** 2,
            n_cells=len(eligible),
            n_appearances=n_appearances,
            min_games_for_cell=min_games,
            development_end=end,
            used_for_state=False,
        )
    sigma2 = float(eligible["mean_bernoulli_var"].mean())
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        sigma2 = float(bernoulli_var.mean())
    sampling_var = (eligible["mean_bernoulli_var"] / eligible["n"]).to_numpy(
        dtype=float
    )
    mean_residual = eligible["mean_residual"].to_numpy(dtype=float)
    between = float(np.var(mean_residual, ddof=1))
    tau2 = max(0.0, between - float(np.mean(sampling_var)))
    k = DEFAULT_SHRINKAGE_K if tau2 <= 0.0 else float(sigma2 / tau2)
    return ShrinkageKEstimate(
        k=k,
        residual_variance=sigma2,
        effect_variance=tau2,
        n_cells=len(eligible),
        n_appearances=n_appearances,
        min_games_for_cell=min_games,
        development_end=end,
        used_for_state=False,
    )


def _assign_n_bin(value: float) -> str:
    n = int(value)
    for label, lo, hi in _N_BINS:
        if hi is None:
            if n >= lo:
                return label
        elif lo <= n <= hi:
            return label
    return "0"


def run_player_hero_elo_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice10DiagnosticReport:
    """Describe Slice 10 on the frozen development window only."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    residuals = appearance_residuals(store, elo_config=elo_config)
    k_hat = estimate_shrinkage_k(residuals, development_end=end)

    state = build_player_hero_elo(
        store, shrinkage_k=shrinkage_k, elo_config=elo_config
    ).to_frame()
    career = build_player_hero(store).to_frame()
    start_time = pd.to_datetime(state["start_time"], utc=True)
    later = holdout_mask(start_time, end)
    development = state.loc[~later].copy()
    n_holdout = int(later.sum())
    if development.empty:
        raise ValueError("no development rows for Slice 10 diagnostics")

    n_matches = int(development[MATCH_ID_COLUMN].nunique())
    mean_res = _finite(development["mean_outcome_residual_on_hero"])
    shrunk = _finite(development["shrunk_outcome_residual_on_hero"])
    residual_distribution = pd.DataFrame(
        [
            {"signal": "mean_outcome_residual (observed n>0)", **_summary(mean_res)},
            {"signal": "shrunk_outcome_residual (cold-start = 0)", **_summary(shrunk)},
            {
                "signal": "shrunk_outcome_residual | n>0",
                **_summary(
                    _finite(
                        development.loc[
                            pd.to_numeric(
                                development["prior_games_on_hero"], errors="coerce"
                            )
                            > 0,
                            "shrunk_outcome_residual_on_hero",
                        ]
                    )
                ),
            },
        ]
    )

    development["n_bin"] = [
        _assign_n_bin(value)
        for value in pd.to_numeric(development["prior_games_on_hero"], errors="coerce")
        .fillna(0)
        .to_numpy()
    ]
    by_n_rows: list[dict[str, object]] = []
    for label, _lo, _hi in _N_BINS:
        subset = development.loc[development["n_bin"] == label]
        raw = _finite(subset["mean_outcome_residual_on_hero"])
        sh = _finite(subset["shrunk_outcome_residual_on_hero"])
        games = pd.to_numeric(subset["prior_games_on_hero"], errors="coerce")
        by_n_rows.append(
            {
                "n_bin": label,
                "n_rows": len(subset),
                "mean_prior_games": float(games.mean()) if len(subset) else float("nan"),
                "mean_raw_residual": float(raw.mean()) if raw.size else float("nan"),
                "mean_abs_raw_residual": float(np.abs(raw).mean())
                if raw.size
                else float("nan"),
                "mean_shrunk_residual": float(sh.mean()) if sh.size else float("nan"),
                "mean_abs_shrunk_residual": float(np.abs(sh).mean())
                if sh.size
                else float("nan"),
                "mean_shrinkage_weight": float(
                    pd.to_numeric(
                        subset["shrinkage_weight_on_hero"], errors="coerce"
                    ).mean()
                )
                if len(subset)
                else float("nan"),
                "frac_positive_shrunk": float((sh > 0).mean()) if sh.size else float("nan"),
            }
        )
    residual_by_n = pd.DataFrame(by_n_rows)

    # Latest development appearance per player×hero, then highest volume.
    latest = (
        development.sort_values("start_time", kind="mergesort")
        .groupby(["player_id", "hero_id"], sort=False, as_index=False)
        .tail(1)
    )
    high_volume = (
        latest.sort_values("prior_games_on_hero", ascending=False, kind="mergesort")
        .head(25)
        .copy()
    )
    high_volume_combinations = high_volume[
        [
            "player_id",
            "hero_id",
            "hero_name",
            "prior_games_on_hero",
            "prior_wins_on_hero",
            "prior_elo_expected_wins_on_hero",
            "mean_outcome_residual_on_hero",
            "shrunk_outcome_residual_on_hero",
            "shrinkage_weight_on_hero",
        ]
    ].reset_index(drop=True)

    # Temporal stability: first vs second half of development appearances.
    dev_resid = residuals.loc[
        pd.to_datetime(residuals["start_time"], utc=True) <= pd.Timestamp(end)
    ].copy()
    dev_resid["residual"] = (
        pd.to_numeric(dev_resid["y"], errors="coerce")
        - pd.to_numeric(dev_resid["elo_expected_win"], errors="coerce")
    )
    midpoint = pd.to_datetime(dev_resid["start_time"], utc=True).quantile(0.5)
    early = dev_resid.loc[pd.to_datetime(dev_resid["start_time"], utc=True) <= midpoint]
    late = dev_resid.loc[pd.to_datetime(dev_resid["start_time"], utc=True) > midpoint]
    early_cells = (
        early.groupby(["player_id", "hero_id"])
        .agg(n_early=("residual", "size"), mean_early=("residual", "mean"))
        .reset_index()
    )
    late_cells = (
        late.groupby(["player_id", "hero_id"])
        .agg(n_late=("residual", "size"), mean_late=("residual", "mean"))
        .reset_index()
    )
    paired = early_cells.merge(late_cells, on=["player_id", "hero_id"], how="inner")
    stable = paired.loc[(paired["n_early"] >= 5) & (paired["n_late"] >= 5)]
    early_shrunk = [
        shrinkage_weight(n, k=shrinkage_k) * m
        for n, m in zip(stable["n_early"], stable["mean_early"], strict=True)
    ]
    late_shrunk = [
        shrinkage_weight(n, k=shrinkage_k) * m
        for n, m in zip(stable["n_late"], stable["mean_late"], strict=True)
    ]
    temporal_stability = pd.DataFrame(
        [
            {
                "n_paired_cells_n_ge_5": len(stable),
                "corr_raw_early_late": _pearson(
                    stable["mean_early"], stable["mean_late"]
                )
                if not stable.empty
                else float("nan"),
                "corr_shrunk_early_late": _pearson(
                    pd.Series(early_shrunk), pd.Series(late_shrunk)
                )
                if not stable.empty
                else float("nan"),
                "mean_early_raw": float(stable["mean_early"].mean())
                if not stable.empty
                else float("nan"),
                "mean_late_raw": float(stable["mean_late"].mean())
                if not stable.empty
                else float("nan"),
                "split_timestamp": pd.Timestamp(midpoint).isoformat(),
            }
        ]
    )

    career_dev = career.loc[
        ~holdout_mask(pd.to_datetime(career["start_time"], utc=True), end)
    ]
    merged = development.merge(
        career_dev[
            [
                MATCH_ID_COLUMN,
                "player_id",
                "hero_id",
                "prior_games_on_hero",
                "prior_win_rate_on_hero",
            ]
        ].rename(
            columns={
                "prior_games_on_hero": "career_prior_games",
                "prior_win_rate_on_hero": "career_prior_win_rate",
            }
        ),
        on=[MATCH_ID_COLUMN, "player_id", "hero_id"],
        how="inner",
        validate="one_to_one",
    )
    observed = merged.loc[
        pd.to_numeric(merged["prior_games_on_hero"], errors="coerce") > 0
    ]
    volume_contrast = pd.DataFrame(
        [
            {
                "n_joined_rows": len(merged),
                "corr_volume_vs_career_volume": _pearson(
                    merged["prior_games_on_hero"], merged["career_prior_games"]
                ),
                "max_abs_volume_minus_career": float(
                    (
                        pd.to_numeric(merged["prior_games_on_hero"], errors="coerce")
                        - pd.to_numeric(merged["career_prior_games"], errors="coerce")
                    )
                    .abs()
                    .max()
                ),
                "corr_volume_vs_raw_residual": _pearson(
                    observed["prior_games_on_hero"],
                    observed["mean_outcome_residual_on_hero"],
                ),
                "corr_volume_vs_shrunk_residual": _pearson(
                    merged["prior_games_on_hero"],
                    merged["shrunk_outcome_residual_on_hero"],
                ),
                "corr_volume_vs_abs_shrunk": _pearson(
                    merged["prior_games_on_hero"],
                    pd.to_numeric(
                        merged["shrunk_outcome_residual_on_hero"], errors="coerce"
                    ).abs(),
                ),
                "corr_career_win_rate_vs_raw_residual": _pearson(
                    observed["career_prior_win_rate"],
                    observed["mean_outcome_residual_on_hero"],
                ),
                "corr_career_win_rate_vs_shrunk_residual": _pearson(
                    observed["career_prior_win_rate"],
                    observed["shrunk_outcome_residual_on_hero"],
                ),
            }
        ]
    )

    games = pd.to_numeric(development["prior_games_on_hero"], errors="coerce")
    coverage = pd.DataFrame(
        [
            {
                "n_player_rows": len(development),
                "n_matches": n_matches,
                "pct_zero_prior": float((games == 0).mean() * 100.0),
                "pct_n_ge_1": float((games >= 1).mean() * 100.0),
                "pct_n_ge_5": float((games >= 5).mean() * 100.0),
                "pct_n_ge_10": float((games >= 10).mean() * 100.0),
                "pct_n_ge_20": float((games >= 20).mean() * 100.0),
                "mean_prior_games": float(games.mean()),
                "median_prior_games": float(games.median()),
                "pct_matches_any_cold_start": float(
                    development.groupby(MATCH_ID_COLUMN)["prior_games_on_hero"]
                    .apply(lambda s: (pd.to_numeric(s, errors="coerce") == 0).any())
                    .mean()
                    * 100.0
                ),
                "pct_matches_all_ten_n_ge_1": float(
                    development.groupby(MATCH_ID_COLUMN)["prior_games_on_hero"]
                    .apply(lambda s: (pd.to_numeric(s, errors="coerce") >= 1).all())
                    .mean()
                    * 100.0
                ),
            }
        ]
    )

    comparison = player_hero_elo_comparison_from_players(development)
    match_comparison_distribution = pd.DataFrame(
        [
            {
                "signal": "mean_player_hero_shrunk_residual_diff",
                **_summary(_finite(comparison["mean_player_hero_shrunk_residual_diff"])),
            },
            {
                "signal": "mean_player_hero_outcome_residual_diff",
                **_summary(
                    _finite(comparison["mean_player_hero_outcome_residual_diff"])
                ),
            },
            {
                "signal": "mean_player_hero_prior_games_diff",
                **_summary(_finite(comparison["mean_player_hero_prior_games_diff"])),
            },
        ]
    )

    games_match_career = bool(
        np.allclose(
            pd.to_numeric(merged["prior_games_on_hero"], errors="coerce").to_numpy(
                dtype=float
            ),
            pd.to_numeric(merged["career_prior_games"], errors="coerce").to_numpy(
                dtype=float
            ),
            equal_nan=True,
        )
    )
    integrity = {
        "shrinkage_k_frozen": float(shrinkage_k),
        "shrinkage_k_estimated_development": float(k_hat.k),
        "estimated_k_used_for_state": False,
        "holdout_player_rows_excluded": n_holdout,
        "ti2026_used_for_thresholds_or_k": False,
        "development_end": end.isoformat(),
        "prior_games_match_career_layer": games_match_career,
        "shrunk_zero_when_n_zero": bool(
            (
                development.loc[games == 0, "shrunk_outcome_residual_on_hero"]
                .fillna(0.0)
                == 0.0
            ).all()
        ),
        "mean_residual_null_when_n_zero": bool(
            development.loc[games == 0, "mean_outcome_residual_on_hero"].isna().all()
        ),
        "evidence_columns_are_not_strength": [
            "prior_games_on_hero",
            "shrinkage_weight_on_hero",
        ],
    }

    return Slice10DiagnosticReport(
        shrinkage_k_frozen=float(shrinkage_k),
        shrinkage_k_prior_note=(
            f"k = σ²/τ² with σ²={RESIDUAL_VARIANCE_PRIOR} and "
            f"τ={TRUE_EFFECT_SD_PRIOR} (8pp true residual SD); "
            f"DEFAULT_SHRINKAGE_K={DEFAULT_SHRINKAGE_K}. "
            "Not estimated from TI 2026."
        ),
        shrinkage_k_estimate=k_hat,
        n_development_matches=n_matches,
        n_development_player_rows=len(development),
        n_holdout_excluded=n_holdout,
        development_end=end,
        residual_distribution=residual_distribution,
        residual_by_n=residual_by_n,
        high_volume_combinations=high_volume_combinations,
        temporal_stability=temporal_stability,
        volume_contrast=volume_contrast,
        coverage=coverage,
        match_comparison_distribution=match_comparison_distribution,
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


def slice10_report_to_jsonable(report: Slice10DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 10 report."""
    estimate = report.shrinkage_k_estimate
    return {
        "shrinkage_k_frozen": report.shrinkage_k_frozen,
        "shrinkage_k_prior_note": report.shrinkage_k_prior_note,
        "shrinkage_k_estimate": {
            "k": _jsonable_value(estimate.k),
            "residual_variance": _jsonable_value(estimate.residual_variance),
            "effect_variance": _jsonable_value(estimate.effect_variance),
            "n_cells": estimate.n_cells,
            "n_appearances": estimate.n_appearances,
            "min_games_for_cell": estimate.min_games_for_cell,
            "development_end": estimate.development_end.isoformat(),
            "used_for_state": estimate.used_for_state,
        },
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "development_end": report.development_end.isoformat(),
        "residual_distribution": _jsonable_value(report.residual_distribution),
        "residual_by_n": _jsonable_value(report.residual_by_n),
        "high_volume_combinations": _jsonable_value(report.high_volume_combinations),
        "temporal_stability": _jsonable_value(report.temporal_stability),
        "volume_contrast": _jsonable_value(report.volume_contrast),
        "coverage": _jsonable_value(report.coverage),
        "match_comparison_distribution": _jsonable_value(
            report.match_comparison_distribution
        ),
        "integrity": _jsonable_value(report.integrity),
    }
