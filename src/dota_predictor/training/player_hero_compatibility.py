"""Slice 23: player × hero behavioral-compatibility diagnostics.

Research only. This module does not persist a fit score, does not add
production features, does not aggregate to team, and does not train a
win model. Compatibility columns never enter ``FEATURE_COLUMNS``.

Question
--------
After accounting for what the player is like (frozen player state) and
what the hero-role is like (frozen LPO hero×position requirement)
independently, does their *relationship* predict subsequent causal
farming/combat performance?

High P with high H is not automatically “good fit”. ``P ≈ H`` is only a
hypothesis. The incremental value of an interaction/mismatch term
beyond the additive baseline is the quantity of interest.

Frozen inputs
-------------
Player farming: Slice 14 ``farming_shrunk_b``, ``k=5``.
Player combat: Slice 18 ``combat_shrunk_c``, ``k=20``.
Hero farming: Slice 22 LPO ``hero_farming_shrunk_b``, ``k=2``.
Hero combat: Slice 22 LPO ``hero_combat_shrunk_c``, ``k=2``.
Targets: ``farming_causal_b`` and ``combat_causal_c``.

Tune / validation uses the established Slice 14/18/22 chronological
boundary. The frozen Slice 9 holdout is excluded from fitting, form
selection, scaling, and every diagnostic used to choose a form.

Research result
---------------
Classification **B — suggestive but unstable**. This slice is
diagnostic-only. It does not freeze a production fit score. Downstream
code must not treat compatibility terms as an accepted predictive
feature. See ``docs/research/slice_status.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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
    PLAYER_X_HERO_FIT_NAMES,
    SPECIALIST_TOP_SHARE,
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
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    HISTORY_N_BUCKETS,
    development_tune_end,
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
    "COMBAT_SPEC",
    "COMPATIBILITY_TERM_NAMES",
    "FARMING_SPEC",
    "MODEL_SPECS",
    "SLICE23_DIAGNOSTIC_COLUMNS",
    "SLICE23_DIAGNOSTIC_ONLY",
    "SLICE23_FIT_SCORE_FROZEN",
    "SLICE23_RESEARCH_CLASSIFICATION",
    "Slice23DiagnosticReport",
    "attach_compatibility_terms",
    "attach_player_hero_compatibility_terms",
    "classify_slice23",
    "compatibility_terms",
    "eligibility_mask",
    "permute_hero_requirement",
    "run_compatibility_diagnostics_on_frame",
    "run_player_hero_compatibility_diagnostics",
    "slice23_report_to_jsonable",
]


RNG_SEED = 23
N_PERMUTATIONS = 100
N_BOOTSTRAP = 200
N_SHAPE_BINS = 8
PATCH_MIN_VERSION_N = 50
RANK_TOL = 1e-8
SCALE_FLOOR = 1e-12
NOISE_RMSE = 1e-6
MATERIAL_RELATIVE_RMSE = 0.005
PERMUTATION_ALPHA = 0.05
MIN_CONFIRM_N = 50

COMPATIBILITY_TERM_NAMES: tuple[str, ...] = (
    "signed_gap",
    "abs_gap",
    "sq_gap",
    "interaction",
    "player_below_requirement",
    "player_above_requirement",
)
HERO_HISTORY_BUCKETS: tuple[tuple[str, int, int | None], ...] = tuple(
    item for item in HISTORY_N_BUCKETS if item[0] != "0"
)
UNIQUE_PLAYER_HISTORY_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1–2", 1, 2),
    ("3–5", 3, 5),
    ("6–10", 6, 10),
    (">10", 11, None),
)
WIN_LABEL_COLUMNS: tuple[str, ...] = (
    "team_won",
    "radiant_win",
    "dire_win",
    "elo_residual_win",
    "model_win_probability",
    "bookmaker_odds",
)

CLASSIFICATION_A = "A — compatibility interaction exists"
CLASSIFICATION_B = "B — suggestive but unstable"
CLASSIFICATION_C = "C — no useful compatibility interaction"
# Recorded development-window result. Diagnostic only: not a
# methodological freeze of a fit score and not a production feature.
SLICE23_RESEARCH_CLASSIFICATION = "B"
SLICE23_DIAGNOSTIC_ONLY = True
SLICE23_FIT_SCORE_FROZEN = False


@dataclass(frozen=True)
class DimensionSpec:
    """Farming or combat column mapping. Dimensions stay independent."""

    name: str
    target: str
    player_state: str
    player_prior_n: str
    hero_requirement: str
    hero_prior_n: str
    hero_unique_players: str
    hero_top_share: str
    hero_inclusive_mean: str
    hero_inclusive_n: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    predictors: tuple[str, ...]
    algebraically_redundant: bool = False
    selection_candidate: bool = False


FARMING_SPEC = DimensionSpec(
    name="farming",
    target=CAUSAL_B_COLUMN,
    player_state="farming_shrunk_b",
    player_prior_n="farming_prior_n",
    hero_requirement="hero_farming_shrunk_b",
    hero_prior_n="hero_farming_prior_n",
    hero_unique_players="hero_farming_unique_prior_players",
    hero_top_share="hero_farming_top_player_share",
    hero_inclusive_mean="hero_farming_inclusive_prior_mean_b",
    hero_inclusive_n="hero_farming_inclusive_prior_n",
)
COMBAT_SPEC = DimensionSpec(
    name="combat",
    target=CAUSAL_C_COLUMN,
    player_state="combat_shrunk_c",
    player_prior_n="combat_prior_n",
    hero_requirement="hero_combat_shrunk_c",
    hero_prior_n="hero_combat_prior_n",
    hero_unique_players="hero_combat_unique_prior_players",
    hero_top_share="hero_combat_top_player_share",
    hero_inclusive_mean="hero_combat_inclusive_prior_mean_c",
    hero_inclusive_n="hero_combat_inclusive_prior_n",
)
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("M0", "intercept", ()),
    ModelSpec("M1", "player", ("player_state",)),
    ModelSpec("M2", "hero", ("hero_requirement",)),
    ModelSpec("M3", "additive", ("player_state", "hero_requirement")),
    ModelSpec(
        "M4a",
        "signed_gap",
        ("player_state", "hero_requirement", "signed_gap"),
        algebraically_redundant=True,
    ),
    ModelSpec(
        "M4b",
        "abs_gap",
        ("player_state", "hero_requirement", "abs_gap"),
        selection_candidate=True,
    ),
    ModelSpec(
        "M4c",
        "sq_gap",
        ("player_state", "hero_requirement", "sq_gap"),
        selection_candidate=True,
    ),
    ModelSpec(
        "M4d",
        "interaction",
        ("player_state", "hero_requirement", "interaction"),
        selection_candidate=True,
    ),
    ModelSpec(
        "M4e",
        "directional_pair",
        (
            "player_state",
            "hero_requirement",
            "player_below_requirement",
            "player_above_requirement",
        ),
        algebraically_redundant=True,
    ),
    ModelSpec(
        "M4f",
        "piecewise_below",
        ("player_state", "hero_requirement", "player_below_requirement"),
        selection_candidate=True,
    ),
    ModelSpec(
        "M4g",
        "piecewise_above",
        ("player_state", "hero_requirement", "player_above_requirement"),
        selection_candidate=True,
    ),
)
SLICE23_DIAGNOSTIC_COLUMNS: tuple[str, ...] = tuple(
    f"{spec.name}_{term}"
    for spec in (FARMING_SPEC, COMBAT_SPEC)
    for term in COMPATIBILITY_TERM_NAMES
)


@dataclass(frozen=True)
class FittedOLS:
    """Tune-fitted OLS with train-only predictor standardization."""

    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    rank_unscaled: int
    expected_rank: int
    condition_number: float
    algebraically_redundant: bool


@dataclass(frozen=True)
class Slice23DiagnosticReport:
    development_end: datetime
    tune_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    farming_semantics: dict[str, object]
    combat_semantics: dict[str, object]
    classification: pd.DataFrame
    split: pd.DataFrame
    farming_coverage: pd.DataFrame
    combat_coverage: pd.DataFrame
    farming_marginal: pd.DataFrame
    combat_marginal: pd.DataFrame
    farming_comparison: pd.DataFrame
    combat_comparison: pd.DataFrame
    farming_residual: pd.DataFrame
    combat_residual: pd.DataFrame
    farming_shape_bins: pd.DataFrame
    combat_shape_bins: pd.DataFrame
    farming_asymmetry: pd.DataFrame
    combat_asymmetry: pd.DataFrame
    farming_position: pd.DataFrame
    combat_position: pd.DataFrame
    farming_hero_history: pd.DataFrame
    combat_hero_history: pd.DataFrame
    farming_player_history: pd.DataFrame
    combat_player_history: pd.DataFrame
    farming_unique_player: pd.DataFrame
    combat_unique_player: pd.DataFrame
    farming_specialist: pd.DataFrame
    combat_specialist: pd.DataFrame
    farming_patch: pd.DataFrame
    combat_patch: pd.DataFrame
    farming_permutation: pd.DataFrame
    combat_permutation: pd.DataFrame
    farming_collinearity: pd.DataFrame
    combat_collinearity: pd.DataFrame
    integrity: dict[str, object]


def compatibility_terms(player: pd.Series, hero: pd.Series) -> pd.DataFrame:
    """Diagnostic candidate variables from frozen ``P`` and ``H``.

    ``signed_gap = P - H`` is algebraically redundant with ``P`` and
    ``H`` inside an additive model. ``player_below + player_above``
    equals ``abs_gap``, and ``player_above - player_below`` equals
    ``signed_gap``.
    """
    p = _numeric(player).astype(float)
    h = _numeric(hero).astype(float)
    signed = p - h
    below = np.maximum(h - p, 0.0)
    above = np.maximum(p - h, 0.0)
    return pd.DataFrame(
        {
            "player_state": p,
            "hero_requirement": h,
            "signed_gap": signed,
            "abs_gap": signed.abs(),
            "sq_gap": signed**2,
            "interaction": p * h,
            "player_below_requirement": below,
            "player_above_requirement": above,
        },
        index=player.index,
    )


def attach_compatibility_terms(
    frame: pd.DataFrame, spec: DimensionSpec
) -> pd.DataFrame:
    """Add prefixed diagnostic terms for one dimension. Does not use inclusive H."""
    terms = compatibility_terms(frame[spec.player_state], frame[spec.hero_requirement])
    out = frame.copy()
    for name in COMPATIBILITY_TERM_NAMES:
        out[f"{spec.name}_{name}"] = terms[name]
    return out


def attach_player_hero_compatibility_terms(frame: pd.DataFrame) -> pd.DataFrame:
    """Add farming and combat diagnostic terms. Farming and combat stay separate."""
    out = attach_compatibility_terms(frame, FARMING_SPEC)
    return attach_compatibility_terms(out, COMBAT_SPEC)


def eligibility_mask(frame: pd.DataFrame, spec: DimensionSpec) -> pd.Series:
    """Eligible appearances for one dimension. No hero-only / inclusive fallback."""
    if spec.player_state not in frame.columns or spec.hero_requirement not in frame.columns:
        return pd.Series(False, index=frame.index)
    if spec.target not in frame.columns or spec.hero_prior_n not in frame.columns:
        return pd.Series(False, index=frame.index)
    position_ok = explicit_position_mask(frame)
    target_ok = np.isfinite(_numeric(frame[spec.target]).to_numpy(dtype=float))
    player_ok = np.isfinite(_numeric(frame[spec.player_state]).to_numpy(dtype=float))
    hero_ok = np.isfinite(_numeric(frame[spec.hero_requirement]).to_numpy(dtype=float))
    n_ok = pd.to_numeric(frame[spec.hero_prior_n], errors="coerce").fillna(0) >= 1
    return (
        position_ok
        & pd.Series(target_ok, index=frame.index)
        & pd.Series(player_ok, index=frame.index)
        & pd.Series(hero_ok, index=frame.index)
        & n_ok
    )


def coverage_table(frame: pd.DataFrame, spec: DimensionSpec) -> pd.DataFrame:
    """Exact eligibility coverage. Counts are not mutually exclusive exclusions."""
    n_rows = len(frame)
    position_ok = explicit_position_mask(frame)
    target_ok = pd.Series(False, index=frame.index)
    player_ok = pd.Series(False, index=frame.index)
    hero_ok = pd.Series(False, index=frame.index)
    n_ok = pd.Series(False, index=frame.index)
    if spec.target in frame.columns:
        target_ok = pd.Series(
            np.isfinite(_numeric(frame[spec.target]).to_numpy(dtype=float)),
            index=frame.index,
        )
    if spec.player_state in frame.columns:
        player_ok = pd.Series(
            np.isfinite(_numeric(frame[spec.player_state]).to_numpy(dtype=float)),
            index=frame.index,
        )
    if spec.hero_requirement in frame.columns:
        hero_ok = pd.Series(
            np.isfinite(_numeric(frame[spec.hero_requirement]).to_numpy(dtype=float)),
            index=frame.index,
        )
    if spec.hero_prior_n in frame.columns:
        n_ok = pd.to_numeric(frame[spec.hero_prior_n], errors="coerce").fillna(0) >= 1
    eligible = eligibility_mask(frame, spec)
    n_eligible = int(eligible.sum())
    return pd.DataFrame(
        [
            {
                "dimension": spec.name,
                "n_development_player_rows": n_rows,
                "n_explicit_position_1_5": int(position_ok.sum()),
                "n_finite_target": int(target_ok.sum()),
                "n_finite_player_state": int(player_ok.sum()),
                "n_hero_lpo_n_ge_1": int(n_ok.sum()),
                "n_finite_hero_state": int(hero_ok.sum()),
                "n_eligible": n_eligible,
                "eligible_share": (
                    float(n_eligible / n_rows) if n_rows else float("nan")
                ),
                "n_excluded_not_explicit_position": int((~position_ok).sum()),
                "n_excluded_missing_target": int((position_ok & ~target_ok).sum()),
                "n_excluded_missing_player_state": int(
                    (position_ok & target_ok & ~player_ok).sum()
                ),
                "n_excluded_hero_lpo_n_0": int(
                    (position_ok & target_ok & player_ok & ~n_ok).sum()
                ),
                "n_excluded_nonfinite_hero_state": int(
                    (position_ok & target_ok & player_ok & n_ok & ~hero_ok).sum()
                ),
            }
        ]
    )


def permute_hero_requirement(
    frame: pd.DataFrame,
    hero_column: str,
    *,
    rng: np.random.Generator,
    position_column: str = "position_number",
    version_column: str = "game_version_id",
) -> pd.DataFrame:
    """Shuffle ``H`` within position, preserving version when present.

    Player state and target rows are unchanged. Singleton strata are
    left untouched.
    """
    out = frame.copy()
    if hero_column not in out.columns or out.empty:
        return out
    position = _numeric(out[position_column]) if position_column in out.columns else None
    values = out[hero_column].to_numpy(copy=True)
    if position is None:
        keys = np.zeros(len(out), dtype=object)
    elif version_column in out.columns:
        version = out[version_column].astype("object").to_numpy()
        keys = np.empty(len(out), dtype=object)
        pos_vals = position.fillna(-1).to_numpy()
        for i, (pos, ver) in enumerate(zip(pos_vals, version, strict=True)):
            keys[i] = (pos, ver)
    else:
        keys = position.fillna(-1).to_numpy()
    grouped: dict[object, list[int]] = {}
    for i, key in enumerate(keys):
        grouped.setdefault(key, []).append(i)
    for idxs in grouped.values():
        if len(idxs) < 2:
            continue
        original = values[idxs].copy()
        values[idxs] = original[rng.permutation(len(idxs))]
    out[hero_column] = values
    return out


def _work_frame(frame: pd.DataFrame, spec: DimensionSpec) -> pd.DataFrame:
    terms = compatibility_terms(frame[spec.player_state], frame[spec.hero_requirement])
    work = terms.copy()
    work["target"] = _numeric(frame[spec.target]).astype(float)
    work["position_number"] = _numeric(frame["position_number"])
    if spec.hero_prior_n in frame.columns:
        work["hero_prior_n"] = pd.to_numeric(frame[spec.hero_prior_n], errors="coerce")
    else:
        work["hero_prior_n"] = np.nan
    if spec.player_prior_n in frame.columns:
        work["player_prior_n"] = pd.to_numeric(
            frame[spec.player_prior_n], errors="coerce"
        )
    else:
        work["player_prior_n"] = np.nan
    if spec.hero_unique_players in frame.columns:
        work["hero_unique_players"] = pd.to_numeric(
            frame[spec.hero_unique_players], errors="coerce"
        )
    else:
        work["hero_unique_players"] = np.nan
    if spec.hero_top_share in frame.columns:
        work["hero_top_share"] = _numeric(frame[spec.hero_top_share])
    else:
        work["hero_top_share"] = np.nan
    if "game_version_id" in frame.columns:
        work["game_version_id"] = frame["game_version_id"].to_numpy()
    else:
        work["game_version_id"] = np.nan
    if "start_time" in frame.columns:
        work["start_time"] = pd.to_datetime(frame["start_time"], utc=True)
    work["match_id"] = frame["match_id"] if "match_id" in frame.columns else np.nan
    work["player_id"] = frame["player_id"] if "player_id" in frame.columns else np.nan
    work["hero_id"] = frame["hero_id"] if "hero_id" in frame.columns else np.nan
    return work.reset_index(drop=True)


def _fit_ols(
    y: np.ndarray,
    x_frame: pd.DataFrame,
    spec: ModelSpec,
) -> FittedOLS:
    names = spec.predictors
    expected = 1 + len(names)
    n = int(y.size)
    if not names:
        intercept = float(y.mean()) if n else float("nan")
        return FittedOLS(
            names=(),
            mean=np.zeros(0, dtype=float),
            scale=np.ones(0, dtype=float),
            coef=np.array([intercept], dtype=float),
            rank_unscaled=1 if n else 0,
            expected_rank=1,
            condition_number=1.0,
            algebraically_redundant=False,
        )
    x = x_frame.loc[:, list(names)].to_numpy(dtype=float)
    unscaled = np.column_stack([np.ones(n), x])
    rank = int(np.linalg.matrix_rank(unscaled, tol=RANK_TOL)) if n else 0
    mean = x.mean(axis=0) if n else np.zeros(len(names), dtype=float)
    std = x.std(axis=0, ddof=0) if n else np.ones(len(names), dtype=float)
    scale = np.where(std < SCALE_FLOOR, 1.0, std)
    xs = (x - mean) / scale if n else x
    design = np.column_stack([np.ones(n), xs])
    if n == 0:
        coef = np.full(expected, np.nan, dtype=float)
        cond = float("nan")
    else:
        coef, _residuals, _rank, singular = np.linalg.lstsq(design, y, rcond=None)
        if singular.size and float(singular[-1]) > 0.0:
            cond = float(singular[0] / singular[-1])
        else:
            cond = float("inf")
    return FittedOLS(
        names=names,
        mean=mean.astype(float),
        scale=scale.astype(float),
        coef=np.asarray(coef, dtype=float),
        rank_unscaled=rank,
        expected_rank=expected,
        condition_number=cond,
        algebraically_redundant=spec.algebraically_redundant or rank < expected,
    )


def _predict_ols(fitted: FittedOLS, x_frame: pd.DataFrame) -> np.ndarray:
    n = len(x_frame)
    if not fitted.names:
        return np.full(n, float(fitted.coef[0]), dtype=float)
    x = x_frame.loc[:, list(fitted.names)].to_numpy(dtype=float)
    xs = (x - fitted.mean) / fitted.scale
    design = np.column_stack([np.ones(n), xs])
    return design @ fitted.coef


def _prediction_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    n = int(y.size)
    if n == 0:
        return {
            "n": 0.0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
        }
    err = y - yhat
    return {
        "n": float(n),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": _pearson(pd.Series(yhat), pd.Series(y)),
    }


def _delta(candidate: float, baseline: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(baseline):
        return float("nan")
    return float(candidate - baseline)


def _bootstrap_delta_rmse(
    y: np.ndarray,
    pred_m3: np.ndarray,
    pred_cand: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    n = int(y.size)
    if n == 0 or n_bootstrap <= 0:
        return {
            "delta_rmse_mean": float("nan"),
            "delta_rmse_p025": float("nan"),
            "delta_rmse_p975": float("nan"),
        }
    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        err_m3 = y[idx] - pred_m3[idx]
        err_c = y[idx] - pred_cand[idx]
        deltas[i] = float(np.sqrt(np.mean(err_c**2)) - np.sqrt(np.mean(err_m3**2)))
    return {
        "delta_rmse_mean": float(deltas.mean()),
        "delta_rmse_p025": float(np.quantile(deltas, 0.025)),
        "delta_rmse_p975": float(np.quantile(deltas, 0.975)),
    }


def _bucket_mask(values: pd.Series, low: int, high: int | None) -> pd.Series:
    n = pd.to_numeric(values, errors="coerce")
    if high is None:
        return n >= low
    return (n >= low) & (n <= high)


def _empty_metrics_row(**extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "n": 0,
        "rmse": float("nan"),
        "mae": float("nan"),
        "pearson": float("nan"),
        "delta_rmse": float("nan"),
        "delta_mae": float("nan"),
    }
    row.update(extra)
    return row


def _subset_comparison(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    extra: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not bool(mask.any()):
        for spec in MODEL_SPECS:
            rows.append(_empty_metrics_row(model=spec.name, **extra))
        return rows
    y_sub = y[mask]
    m3 = _prediction_metrics(y_sub, predictions["M3"][mask])
    for spec in MODEL_SPECS:
        metrics = _prediction_metrics(y_sub, predictions[spec.name][mask])
        rows.append(
            {
                "model": spec.name,
                **extra,
                "n": int(metrics["n"]),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "pearson": metrics["pearson"],
                "delta_rmse": _delta(metrics["rmse"], m3["rmse"]),
                "delta_mae": _delta(metrics["mae"], m3["mae"]),
            }
        )
    return rows


def _marginal_table(work: pd.DataFrame, split: str) -> pd.DataFrame:
    y = work["target"]
    p = work["player_state"]
    h = work["hero_requirement"]
    return pd.DataFrame(
        [
            {
                "split": split,
                "predictor": "player_state",
                "n": len(work),
                "pearson": _pearson(p, y),
                "spearman": _spearman(p, y),
                "slope": slope_coefficient(y, p),
            },
            {
                "split": split,
                "predictor": "hero_requirement",
                "n": len(work),
                "pearson": _pearson(h, y),
                "spearman": _spearman(h, y),
                "slope": slope_coefficient(y, h),
            },
        ]
    )


def _residual_table(
    work: pd.DataFrame,
    residual: np.ndarray,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    resid = pd.Series(residual, index=work.index)
    for name in COMPATIBILITY_TERM_NAMES:
        series = work[name]
        pearson = _pearson(series, resid)
        slope = slope_coefficient(resid, series)
        rows.append(
            {
                "split": split,
                "candidate": name,
                "n": len(work),
                "pearson": pearson,
                "residual_slope": slope,
            }
        )
    return pd.DataFrame(rows)


def _shape_edges(signed: pd.Series, n_bins: int) -> np.ndarray:
    finite = _numeric(signed)
    finite = finite[np.isfinite(finite.to_numpy())]
    if finite.empty:
        return np.array([-np.inf, np.inf], dtype=float)
    probs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(finite.to_numpy(dtype=float), probs))
    if edges.size < 2:
        value = float(edges[0]) if edges.size else 0.0
        return np.array([value - 1.0, value + 1.0], dtype=float)
    return edges.astype(float)


def _shape_bin_table(
    work: pd.DataFrame,
    residual: np.ndarray,
    edges: np.ndarray,
    split: str,
) -> pd.DataFrame:
    signed = _numeric(work["signed_gap"])
    bins = pd.cut(signed, bins=edges, include_lowest=True)
    labeled = work.assign(_bin=bins, _residual=residual)
    rows: list[dict[str, object]] = []
    for label, group in labeled.groupby("_bin", observed=False, dropna=False):
        rows.append(
            {
                "split": split,
                "bin": str(label),
                "n": len(group),
                "mean_player_state": float(group["player_state"].mean())
                if len(group)
                else float("nan"),
                "mean_hero_requirement": float(group["hero_requirement"].mean())
                if len(group)
                else float("nan"),
                "mean_signed_gap": float(group["signed_gap"].mean())
                if len(group)
                else float("nan"),
                "mean_target": float(group["target"].mean()) if len(group) else float("nan"),
                "mean_additive_residual": float(group["_residual"].mean())
                if len(group)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _asymmetry_table(
    work: pd.DataFrame, residual: np.ndarray, split: str
) -> pd.DataFrame:
    signed = _numeric(work["signed_gap"]).to_numpy(dtype=float)
    y = work["target"].to_numpy(dtype=float)
    p = work["player_state"].to_numpy(dtype=float)
    h = work["hero_requirement"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for label, mask in (
        ("player_below_requirement", signed < 0.0),
        ("aligned", signed == 0.0),
        ("player_above_requirement", signed > 0.0),
    ):
        n = int(mask.sum())
        rows.append(
            {
                "split": split,
                "region": label,
                "n": n,
                "mean_player_state": float(p[mask].mean()) if n else float("nan"),
                "mean_hero_requirement": float(h[mask].mean()) if n else float("nan"),
                "mean_target": float(y[mask].mean()) if n else float("nan"),
                "mean_additive_residual": float(residual[mask].mean())
                if n
                else float("nan"),
                "mean_abs_gap": float(np.abs(signed[mask]).mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _history_table(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    values: pd.Series,
    buckets: tuple[tuple[str, int, int | None], ...],
    *,
    kind: str,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, low, high in buckets:
        mask = _bucket_mask(values, low, high).to_numpy(dtype=bool)
        rows.extend(
            _subset_comparison(
                y,
                predictions,
                mask,
                extra={"split": split, "kind": kind, "bucket": label},
            )
        )
    return pd.DataFrame(rows)


def _position_table(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    positions: pd.Series,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        _subset_comparison(
            y,
            predictions,
            np.ones(len(y), dtype=bool),
            extra={"split": split, "position": "pooled"},
        )
    )
    pos = _numeric(positions)
    for number in EXPLICIT_POSITION_NUMBERS:
        mask = (pos == float(number)).to_numpy(dtype=bool)
        rows.extend(
            _subset_comparison(
                y,
                predictions,
                mask,
                extra={"split": split, "position": str(number)},
            )
        )
    return pd.DataFrame(rows)


def _specialist_table(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    share: pd.Series,
    split: str,
) -> pd.DataFrame:
    values = _numeric(share)
    rows: list[dict[str, object]] = []
    for label, mask in (
        (
            "specialist_top_share>=0.50",
            values.notna() & (values >= SPECIALIST_TOP_SHARE),
        ),
        (
            "diversified_top_share<0.50",
            values.notna() & (values < SPECIALIST_TOP_SHARE),
        ),
        ("top_share_missing", values.isna()),
    ):
        rows.extend(
            _subset_comparison(
                y,
                predictions,
                mask.to_numpy(dtype=bool),
                extra={"split": split, "subset": label},
            )
        )
    return pd.DataFrame(rows)


def _patch_table(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    versions: pd.Series,
    split: str,
    strongest: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pred_m3 = predictions["M3"]
    pred_s = predictions[strongest]
    grouped = pd.Series(np.arange(len(y)), index=versions.index).groupby(
        versions, dropna=False
    )
    version_n = {key: len(idx) for key, idx in grouped.groups.items()}
    populated = {
        key: n for key, n in version_n.items() if n >= PATCH_MIN_VERSION_N
    }
    for version, idx in grouped.groups.items():
        mask = np.zeros(len(y), dtype=bool)
        mask[np.asarray(list(idx), dtype=int)] = True
        n = int(mask.sum())
        m3 = _prediction_metrics(y[mask], pred_m3[mask])
        cand = _prediction_metrics(y[mask], pred_s[mask])
        resid = y[mask] - pred_m3[mask]
        rows.append(
            {
                "split": split,
                "game_version_id": version,
                "n": n,
                "populated": n >= PATCH_MIN_VERSION_N,
                "model": strongest,
                "rmse": cand["rmse"],
                "mae": cand["mae"],
                "delta_rmse": _delta(cand["rmse"], m3["rmse"]),
                "delta_mae": _delta(cand["mae"], m3["mae"]),
                "residual_pearson": _pearson(
                    pd.Series(predictions[strongest][mask] - pred_m3[mask]),
                    pd.Series(resid),
                )
                if n
                else float("nan"),
            }
        )
    if rows:
        deltas = [
            float(row["delta_rmse"])
            for row in rows
            if bool(row["populated"]) and np.isfinite(row["delta_rmse"])
        ]
        sign_flip = (
            any(d < 0.0 for d in deltas) and any(d > 0.0 for d in deltas)
            if deltas
            else False
        )
        largest = max(version_n.values()) if version_n else 0
        total = max(sum(version_n.values()), 1)
        domination = bool(largest > 0 and largest / total >= 0.80)
        for row in rows:
            row["sign_flip_across_populated_versions"] = sign_flip
            row["one_version_domination"] = domination
            row["n_populated_versions"] = len(populated)
    return pd.DataFrame(rows)


def _collinearity_table(fitted: dict[str, FittedOLS]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        model = fitted[spec.name]
        rows.append(
            {
                "model": spec.name,
                "label": spec.label,
                "n_predictors": len(spec.predictors),
                "expected_rank": model.expected_rank,
                "rank_unscaled": model.rank_unscaled,
                "rank_deficient": model.rank_unscaled < model.expected_rank,
                "algebraically_redundant": spec.algebraically_redundant,
                "condition_number": model.condition_number,
                "caveat": (
                    "signed_gap = P-H is spanned by P and H; coefficients "
                    "and apparent fit are not new information"
                    if spec.name == "M4a"
                    else (
                        "below-above are jointly spanned by abs_gap plus P-H; "
                        "predictions match M4b up to numerical error"
                        if spec.name == "M4e"
                        else ""
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_strongest(comparison: pd.DataFrame) -> str:
    tune = comparison.loc[
        (comparison["split"] == "tune")
        & (comparison["model"].isin([s.name for s in MODEL_SPECS if s.selection_candidate]))
    ]
    if tune.empty:
        return "M4b"
    order = tune.sort_values(["delta_rmse", "model"], kind="mergesort")
    return str(order.iloc[0]["model"])


def _dimension_gate(
    *,
    name: str,
    strongest: str,
    comparison: pd.DataFrame,
    position: pd.DataFrame,
    specialist: pd.DataFrame,
    patch: pd.DataFrame,
    permutation: pd.DataFrame,
    collinearity: pd.DataFrame,
) -> dict[str, object]:
    def _row(split: str, model: str) -> pd.Series | None:
        subset = comparison.loc[
            (comparison["split"] == split) & (comparison["model"] == model)
        ]
        if subset.empty:
            return None
        return subset.iloc[0]

    tune = _row("tune", strongest)
    val = _row("validation", strongest)
    m3_val = _row("validation", "M3")
    tune_delta = float(tune["delta_rmse"]) if tune is not None else float("nan")
    val_delta = float(val["delta_rmse"]) if val is not None else float("nan")
    val_n = int(val["n"]) if val is not None else 0
    m3_rmse = float(m3_val["rmse"]) if m3_val is not None else float("nan")
    material = (
        abs(MATERIAL_RELATIVE_RMSE * m3_rmse)
        if np.isfinite(m3_rmse)
        else NOISE_RMSE
    )
    material = max(material, NOISE_RMSE)
    val_ci_hi = (
        float(val["delta_rmse_p975"])
        if val is not None and "delta_rmse_p975" in val.index
        else float("nan")
    )
    perm_p = float("nan")
    if not permutation.empty:
        perm_p = float(permutation.iloc[0]["p_value"])
    pos_val = position.loc[
        (position["split"] == "validation")
        & (position["model"] == strongest)
        & (position["position"] != "pooled")
    ]
    improving_positions = pos_val.loc[
        pd.to_numeric(pos_val["delta_rmse"], errors="coerce") < 0.0
    ]
    n_improving_positions = len(improving_positions)
    specialist_val = specialist.loc[
        (specialist["split"] == "validation") & (specialist["model"] == strongest)
    ]
    diversified = specialist_val.loc[
        specialist_val["subset"] == "diversified_top_share<0.50"
    ]
    specialist_only = specialist_val.loc[
        specialist_val["subset"] == "specialist_top_share>=0.50"
    ]
    diversified_delta = (
        float(diversified.iloc[0]["delta_rmse"]) if not diversified.empty else float("nan")
    )
    specialist_delta = (
        float(specialist_only.iloc[0]["delta_rmse"])
        if not specialist_only.empty
        else float("nan")
    )
    specialist_driven = bool(
        np.isfinite(specialist_delta)
        and specialist_delta < 0.0
        and (not np.isfinite(diversified_delta) or diversified_delta >= 0.0)
    )
    patch_val = patch.loc[patch["split"] == "validation"] if not patch.empty else patch
    sign_flip = (
        bool(patch_val["sign_flip_across_populated_versions"].iloc[0])
        if not patch_val.empty
        else False
    )
    domination = (
        bool(patch_val["one_version_domination"].iloc[0]) if not patch_val.empty else False
    )
    collinear = False
    if not collinearity.empty:
        row = collinearity.loc[collinearity["model"] == strongest]
        if not row.empty:
            collinear = bool(row.iloc[0]["algebraically_redundant"])
    tune_improves = bool(np.isfinite(tune_delta) and tune_delta < -NOISE_RMSE)
    val_improves = bool(np.isfinite(val_delta) and val_delta < -material)
    ci_excludes_zero = bool(np.isfinite(val_ci_hi) and val_ci_hi < 0.0)
    perm_ok = bool(np.isfinite(perm_p) and perm_p <= PERMUTATION_ALPHA)
    pooled_coherent = n_improving_positions >= 2
    confirmed = (
        tune_improves
        and val_improves
        and val_n >= MIN_CONFIRM_N
        and (ci_excludes_zero or not np.isfinite(val_ci_hi))
        and perm_ok
        and pooled_coherent
        and not specialist_driven
        and not sign_flip
        and not domination
        and not collinear
    )
    unstable = tune_improves and not confirmed
    if confirmed:
        grade = "A"
        rationale = (
            f"{strongest} improves validation RMSE beyond additive M3 "
            f"(ΔRMSE={val_delta:.6g}), confirms on tune, permutation, "
            "and is not confined to one role/patch/specialist cell."
        )
    elif unstable:
        grade = "B"
        rationale = (
            f"{strongest} improves tune (ΔRMSE={tune_delta:.6g}) but "
            "validation, permutation, role, specialist, or patch checks "
            "are weak or inconsistent. Do not freeze a fit score."
        )
    else:
        grade = "C"
        rationale = (
            "P and H independently explain the target; no simple "
            "compatibility form adds repeatable predictive information."
        )
    return {
        "dimension": name,
        "grade": grade,
        "strongest_candidate": strongest,
        "tune_delta_rmse": tune_delta,
        "validation_delta_rmse": val_delta,
        "validation_n": val_n,
        "validation_delta_rmse_p975": val_ci_hi,
        "permutation_p_value": perm_p,
        "n_improving_positions": n_improving_positions,
        "specialist_driven": specialist_driven,
        "patch_sign_flip": sign_flip,
        "patch_domination": domination,
        "selected_on_tune_only": True,
        "rationale": rationale,
    }


def classify_slice23(
    *,
    farming_gate: dict[str, object],
    combat_gate: dict[str, object],
) -> pd.DataFrame:
    """Map per-dimension gates onto one Slice 23 classification."""
    farm_grade = str(farming_gate["grade"])
    combat_grade = str(combat_gate["grade"])
    if farm_grade == "A" or combat_grade == "A":
        classification = "A"
        gate = CLASSIFICATION_A
        next_slice = (
            "Slice 24 may formalize/freeze the confirmed interaction. "
            "This is not yet a production fit score."
        )
    elif farm_grade == "C" and combat_grade == "C":
        classification = "C"
        gate = CLASSIFICATION_C
        next_slice = (
            "Do not freeze a player×hero fit metric. Additive P+H is "
            "sufficient on the causal B/C layer."
        )
    else:
        classification = "B"
        gate = CLASSIFICATION_B
        next_slice = (
            "Diagnostic only. Do not freeze a production fit score and "
            "do not treat compatibility terms as an accepted predictive "
            "feature. Revisit only if a later slice has a sharper "
            "identification strategy."
        )
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "gate": gate,
                "farming_grade": farm_grade,
                "combat_grade": combat_grade,
                "farming_strongest": farming_gate["strongest_candidate"],
                "combat_strongest": combat_gate["strongest_candidate"],
                "farming_rationale": farming_gate["rationale"],
                "combat_rationale": combat_gate["rationale"],
                "next_slice": next_slice,
                "high_P_high_H_is_not_automatically_fit": True,
                "P_approx_H_is_only_a_hypothesis": True,
            }
        ]
    )


def _permutation_table(
    tune: pd.DataFrame,
    spec: DimensionSpec,
    strongest: str,
    observed_delta: float,
    observed_residual_pearson: float,
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> pd.DataFrame:
    model = next(item for item in MODEL_SPECS if item.name == strongest)
    deltas = np.full(max(n_permutations, 0), np.nan, dtype=float)
    pears = np.full(max(n_permutations, 0), np.nan, dtype=float)
    additive = next(item for item in MODEL_SPECS if item.name == "M3")
    term = model.predictors[-1] if model.predictors else "signed_gap"
    for i in range(n_permutations):
        shuffled = permute_hero_requirement(tune, spec.hero_requirement, rng=rng)
        work = _work_frame(shuffled, spec)
        y = work["target"].to_numpy(dtype=float)
        fitted_m3 = _fit_ols(y, work, additive)
        fitted_c = _fit_ols(y, work, model)
        pred_m3 = _predict_ols(fitted_m3, work)
        pred_c = _predict_ols(fitted_c, work)
        m3 = _prediction_metrics(y, pred_m3)
        cand = _prediction_metrics(y, pred_c)
        deltas[i] = _delta(cand["rmse"], m3["rmse"])
        residual = y - pred_m3
        pears[i] = (
            _pearson(work[term], pd.Series(residual))
            if term in work.columns
            else float("nan")
        )
    finite = deltas[np.isfinite(deltas)]
    if finite.size and np.isfinite(observed_delta):
        p_value = float(np.mean(finite <= observed_delta))
    else:
        p_value = float("nan")
    finite_p = pears[np.isfinite(pears)]
    if finite_p.size and np.isfinite(observed_residual_pearson):
        p_pearson = float(
            np.mean(np.abs(finite_p) >= abs(observed_residual_pearson))
        )
    else:
        p_pearson = float("nan")
    return pd.DataFrame(
        [
            {
                "model": strongest,
                "n_permutations": n_permutations,
                "observed_delta_rmse": observed_delta,
                "permutation_delta_rmse_mean": float(finite.mean())
                if finite.size
                else float("nan"),
                "permutation_delta_rmse_p025": float(np.quantile(finite, 0.025))
                if finite.size
                else float("nan"),
                "permutation_delta_rmse_p975": float(np.quantile(finite, 0.975))
                if finite.size
                else float("nan"),
                "p_value": p_value,
                "observed_residual_pearson": observed_residual_pearson,
                "permutation_residual_pearson_p_value": p_pearson,
                "strata": "position_number × game_version_id",
            }
        ]
    )


def _empty_work(frame: pd.DataFrame, spec: DimensionSpec) -> pd.DataFrame:
    if spec.player_state in frame.columns and spec.hero_requirement in frame.columns:
        return _work_frame(frame.iloc[0:0], spec)
    return pd.DataFrame(
        columns=[
            "player_state",
            "hero_requirement",
            "signed_gap",
            "abs_gap",
            "sq_gap",
            "interaction",
            "player_below_requirement",
            "player_above_requirement",
            "target",
            "position_number",
            "hero_prior_n",
            "player_prior_n",
            "hero_unique_players",
            "hero_top_share",
            "game_version_id",
        ]
    )


def _evaluate_dimension(
    frame: pd.DataFrame,
    spec: DimensionSpec,
    *,
    tune_mask: pd.Series,
    val_mask: pd.Series,
    rng: np.random.Generator,
    n_permutations: int,
    n_bootstrap: int,
    n_shape_bins: int,
) -> dict[str, object]:
    eligible = eligibility_mask(frame, spec)
    coverage = coverage_table(frame, spec)
    tune_rows = frame.loc[eligible & tune_mask]
    val_rows = frame.loc[eligible & val_mask]
    split = pd.DataFrame(
        [
            {
                "dimension": spec.name,
                "split": "tune",
                "n_eligible": len(tune_rows),
                "n_unique_matches": int(tune_rows["match_id"].nunique())
                if "match_id" in tune_rows.columns and len(tune_rows)
                else 0,
                "n_unique_players": int(tune_rows["player_id"].nunique())
                if "player_id" in tune_rows.columns and len(tune_rows)
                else 0,
            },
            {
                "dimension": spec.name,
                "split": "validation",
                "n_eligible": len(val_rows),
                "n_unique_matches": int(val_rows["match_id"].nunique())
                if "match_id" in val_rows.columns and len(val_rows)
                else 0,
                "n_unique_players": int(val_rows["player_id"].nunique())
                if "player_id" in val_rows.columns and len(val_rows)
                else 0,
            },
        ]
    )
    tune_work = _work_frame(tune_rows, spec) if len(tune_rows) else _empty_work(frame, spec)
    val_work = _work_frame(val_rows, spec) if len(val_rows) else _empty_work(frame, spec)
    y_tune = (
        tune_work["target"].to_numpy(dtype=float)
        if len(tune_work)
        else np.zeros(0, dtype=float)
    )
    y_val = (
        val_work["target"].to_numpy(dtype=float)
        if len(val_work)
        else np.zeros(0, dtype=float)
    )
    fitted: dict[str, FittedOLS] = {}
    for model in MODEL_SPECS:
        fitted[model.name] = _fit_ols(y_tune, tune_work, model)
    pred_tune = {name: _predict_ols(model, tune_work) for name, model in fitted.items()}
    pred_val = {name: _predict_ols(model, val_work) for name, model in fitted.items()}
    comparison_rows: list[dict[str, object]] = []
    m3_tune = _prediction_metrics(y_tune, pred_tune["M3"])
    m3_val = _prediction_metrics(y_val, pred_val["M3"])
    for model in MODEL_SPECS:
        for split_name, y, preds, baseline in (
            ("tune", y_tune, pred_tune, m3_tune),
            ("validation", y_val, pred_val, m3_val),
        ):
            metrics = _prediction_metrics(y, preds[model.name])
            row: dict[str, object] = {
                "model": model.name,
                "label": model.label,
                "split": split_name,
                "n": int(metrics["n"]),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "pearson": metrics["pearson"],
                "delta_rmse": _delta(metrics["rmse"], baseline["rmse"]),
                "delta_mae": _delta(metrics["mae"], baseline["mae"]),
                "algebraically_redundant": model.algebraically_redundant,
                "rank_unscaled": fitted[model.name].rank_unscaled,
                "expected_rank": fitted[model.name].expected_rank,
                "max_abs_pred_diff_vs_m3": (
                    float(np.max(np.abs(preds[model.name] - preds["M3"])))
                    if y.size
                    else float("nan")
                ),
            }
            if split_name == "validation" and model.selection_candidate:
                boot = _bootstrap_delta_rmse(
                    y, preds["M3"], preds[model.name], rng, n_bootstrap
                )
                row.update(boot)
            else:
                row["delta_rmse_mean"] = float("nan")
                row["delta_rmse_p025"] = float("nan")
                row["delta_rmse_p975"] = float("nan")
            comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    strongest = _select_strongest(comparison)
    residual_tune = y_tune - pred_tune["M3"]
    residual_val = y_val - pred_val["M3"]
    residual = pd.concat(
        [
            _residual_table(tune_work, residual_tune, "tune"),
            _residual_table(val_work, residual_val, "validation"),
        ],
        ignore_index=True,
    )
    edges = _shape_edges(tune_work["signed_gap"], n_shape_bins)
    shape = pd.concat(
        [
            _shape_bin_table(tune_work, residual_tune, edges, "tune"),
            _shape_bin_table(val_work, residual_val, edges, "validation"),
        ],
        ignore_index=True,
    )
    asymmetry = pd.concat(
        [
            _asymmetry_table(tune_work, residual_tune, "tune"),
            _asymmetry_table(val_work, residual_val, "validation"),
        ],
        ignore_index=True,
    )
    position = pd.concat(
        [
            _position_table(y_tune, pred_tune, tune_work["position_number"], "tune"),
            _position_table(y_val, pred_val, val_work["position_number"], "validation"),
        ],
        ignore_index=True,
    )
    hero_history = pd.concat(
        [
            _history_table(
                y_tune,
                pred_tune,
                tune_work["hero_prior_n"],
                HERO_HISTORY_BUCKETS,
                kind="hero_lpo_n",
                split="tune",
            ),
            _history_table(
                y_val,
                pred_val,
                val_work["hero_prior_n"],
                HERO_HISTORY_BUCKETS,
                kind="hero_lpo_n",
                split="validation",
            ),
        ],
        ignore_index=True,
    )
    player_history = pd.concat(
        [
            _history_table(
                y_tune,
                pred_tune,
                tune_work["player_prior_n"],
                HISTORY_N_BUCKETS,
                kind="player_prior_n",
                split="tune",
            ),
            _history_table(
                y_val,
                pred_val,
                val_work["player_prior_n"],
                HISTORY_N_BUCKETS,
                kind="player_prior_n",
                split="validation",
            ),
        ],
        ignore_index=True,
    )
    unique_player = pd.concat(
        [
            _history_table(
                y_tune,
                pred_tune,
                tune_work["hero_unique_players"],
                UNIQUE_PLAYER_HISTORY_BUCKETS,
                kind="hero_unique_prior_players",
                split="tune",
            ),
            _history_table(
                y_val,
                pred_val,
                val_work["hero_unique_players"],
                UNIQUE_PLAYER_HISTORY_BUCKETS,
                kind="hero_unique_prior_players",
                split="validation",
            ),
        ],
        ignore_index=True,
    )
    specialist = pd.concat(
        [
            _specialist_table(y_tune, pred_tune, tune_work["hero_top_share"], "tune"),
            _specialist_table(y_val, pred_val, val_work["hero_top_share"], "validation"),
        ],
        ignore_index=True,
    )
    patch = pd.concat(
        [
            _patch_table(
                y_tune, pred_tune, tune_work["game_version_id"], "tune", strongest
            ),
            _patch_table(
                y_val, pred_val, val_work["game_version_id"], "validation", strongest
            ),
        ],
        ignore_index=True,
    )
    collinearity = _collinearity_table(fitted)
    strongest_spec = next(item for item in MODEL_SPECS if item.name == strongest)
    strongest_term = (
        strongest_spec.predictors[-1] if strongest_spec.predictors else "signed_gap"
    )
    observed_delta = (
        float(
            comparison.loc[
                (comparison["split"] == "tune") & (comparison["model"] == strongest),
                "delta_rmse",
            ].iloc[0]
        )
        if not comparison.empty
        else float("nan")
    )
    observed_resid_r = (
        _pearson(
            tune_work[strongest_term],
            pd.Series(residual_tune, index=tune_work.index),
        )
        if len(tune_work) and strongest_term in tune_work.columns
        else float("nan")
    )
    permutation = _permutation_table(
        tune_rows,
        spec,
        strongest,
        observed_delta,
        observed_resid_r,
        rng=rng,
        n_permutations=n_permutations,
    )
    gate = _dimension_gate(
        name=spec.name,
        strongest=strongest,
        comparison=comparison,
        position=position,
        specialist=specialist,
        patch=patch,
        permutation=permutation,
        collinearity=collinearity,
    )
    marginal = pd.concat(
        [
            _marginal_table(tune_work, "tune"),
            _marginal_table(val_work, "validation"),
        ],
        ignore_index=True,
    )
    return {
        "coverage": coverage,
        "split": split,
        "marginal": marginal,
        "comparison": comparison,
        "residual": residual,
        "shape_bins": shape,
        "asymmetry": asymmetry,
        "position": position,
        "hero_history": hero_history,
        "player_history": player_history,
        "unique_player": unique_player,
        "specialist": specialist,
        "patch": patch,
        "permutation": permutation,
        "collinearity": collinearity,
        "gate": gate,
        "strongest": strongest,
    }


def _semantics(spec: DimensionSpec) -> dict[str, object]:
    return {
        "dimension": spec.name,
        "target": spec.target,
        "player_state": spec.player_state,
        "hero_requirement": spec.hero_requirement,
        "player_history": "strictly prior player appearances; same-timestamp blind",
        "hero_history": (
            "Slice 22 LPO: start_time < T; same hero_id; same explicit "
            "position; player_id != current player; n >= 1 required"
        ),
        "inclusive_hero_state_used": False,
        "hero_only_fallback": False,
        "current_player_hero_history_used": False,
        "development_wide_fallback": False,
        "high_P_high_H_is_not_automatically_fit": True,
        "P_approx_H_is_only_a_hypothesis": True,
        "fit_constructed": False,
        "team_aggregated": False,
        "win_labels_used": False,
        "current_position": (
            "diagnostic realized post-match position 1–5; not a PRE_DRAFT feature"
        ),
    }


def run_compatibility_diagnostics_on_frame(
    development: pd.DataFrame,
    *,
    development_end: datetime | None = None,
    n_holdout_excluded: int = 0,
    n_permutations: int = N_PERMUTATIONS,
    n_bootstrap: int = N_BOOTSTRAP,
    n_shape_bins: int = N_SHAPE_BINS,
    rng_seed: int = RNG_SEED,
) -> Slice23DiagnosticReport:
    """Run Slice 23 diagnostics on an already-restricted development frame."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    frame = development.copy()
    if "farming_shrunk_b" not in frame.columns or "combat_shrunk_c" not in frame.columns:
        frame = attach_hero_profile_observations(frame)
    if "hero_farming_shrunk_b" not in frame.columns:
        frame = attach_hero_requirement_state(
            frame,
            k_farm=FROZEN_HERO_FARM_SHRINKAGE_K,
            k_combat=FROZEN_HERO_COMBAT_SHRINKAGE_K,
        )
    frame = attach_player_hero_compatibility_terms(frame)
    tune_end = development_tune_end(frame["start_time"], development_end=end)
    times = pd.to_datetime(frame["start_time"], utc=True)
    tune_mask = times <= pd.Timestamp(tune_end)
    val_mask = (times > pd.Timestamp(tune_end)) & (times <= pd.Timestamp(end))
    rng = np.random.default_rng(rng_seed)
    farming = _evaluate_dimension(
        frame,
        FARMING_SPEC,
        tune_mask=tune_mask,
        val_mask=val_mask,
        rng=rng,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        n_shape_bins=n_shape_bins,
    )
    combat = _evaluate_dimension(
        frame,
        COMBAT_SPEC,
        tune_mask=tune_mask,
        val_mask=val_mask,
        rng=rng,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        n_shape_bins=n_shape_bins,
    )
    classification = classify_slice23(
        farming_gate=farming["gate"], combat_gate=combat["gate"]
    )
    split = pd.concat([farming["split"], combat["split"]], ignore_index=True)
    same_stamp_leak = bool(
        set(times[tune_mask].to_numpy()) & set(times[val_mask].to_numpy())
    )
    predictor_names = {name for spec in MODEL_SPECS for name in spec.predictors}
    win_in_predictors = bool(predictor_names.intersection(WIN_LABEL_COLUMNS))
    integrity = {
        "development_end": end.isoformat(),
        "tune_end": tune_end.isoformat(),
        "preferred_tune_end": PREFERRED_TUNE_END.isoformat(),
        "tune_end_matches_preferred": tune_end == PREFERRED_TUNE_END,
        "same_timestamp_groups_disjoint": not same_stamp_leak,
        "holdout_used_for_fitting": False,
        "holdout_used_for_form_selection": False,
        "holdout_used_for_scaling": False,
        "holdout_used_for_thresholds": False,
        "validation_used_for_form_fitting": False,
        "validation_used_for_scaling": False,
        "inclusive_hero_state_used": False,
        "hero_only_state_used": False,
        "current_player_hero_history_used": False,
        "development_wide_fallback": False,
        "farming_player_k_is_5": FROZEN_SHRINKAGE_K == 5.0,
        "combat_player_k_is_20": FROZEN_COMBAT_SHRINKAGE_K == 20.0,
        "hero_farm_k_is_2": FROZEN_HERO_FARM_SHRINKAGE_K == 2.0,
        "hero_combat_k_is_2": FROZEN_HERO_COMBAT_SHRINKAGE_K == 2.0,
        "farming_candidate_b_unchanged": FROZEN_CANDIDATE_B == CANDIDATE_B,
        "combat_candidate_c_unchanged": FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION,
        "stratz_called": False,
        "ingestion_modified": False,
        "schema_modified": False,
        "player_hero_fit_created": False,
        "current_position_resolved": False,
        "team_feature_created": False,
        "team_aggregation": False,
        "win_model_run": False,
        "win_labels_used_in_predictors": win_in_predictors,
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "compatibility_in_feature_columns": any(
            name in FEATURE_COLUMNS
            for name in (
                *SLICE23_DIAGNOSTIC_COLUMNS,
                *SLICE22_STATE_COLUMNS,
                *PLAYER_X_HERO_FIT_NAMES,
                CAUSAL_B_COLUMN,
                CAUSAL_C_COLUMN,
            )
        ),
        "compatibility_in_snapshot_columns": any(
            name in SNAPSHOT_COLUMNS for name in SLICE23_DIAGNOSTIC_COLUMNS
        ),
        "compatibility_in_pre_draft_sql": any(
            name in PRE_DRAFT_SNAPSHOT_SQL for name in SLICE23_DIAGNOSTIC_COLUMNS
        ),
        "n_holdout_excluded": n_holdout_excluded,
        "model_trained": False,
        "fit_score_frozen": SLICE23_FIT_SCORE_FROZEN,
        "diagnostic_only": SLICE23_DIAGNOSTIC_ONLY,
        "farming_gate": farming["gate"],
        "combat_gate": combat["gate"],
        "rng_seed": rng_seed,
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
    }
    return Slice23DiagnosticReport(
        development_end=end,
        tune_end=tune_end,
        n_development_matches=int(frame["match_id"].nunique())
        if "match_id" in frame.columns
        else 0,
        n_development_player_rows=len(frame),
        n_holdout_excluded=n_holdout_excluded,
        farming_semantics=_semantics(FARMING_SPEC),
        combat_semantics=_semantics(COMBAT_SPEC),
        classification=classification,
        split=split,
        farming_coverage=farming["coverage"],
        combat_coverage=combat["coverage"],
        farming_marginal=farming["marginal"],
        combat_marginal=combat["marginal"],
        farming_comparison=farming["comparison"],
        combat_comparison=combat["comparison"],
        farming_residual=farming["residual"],
        combat_residual=combat["residual"],
        farming_shape_bins=farming["shape_bins"],
        combat_shape_bins=combat["shape_bins"],
        farming_asymmetry=farming["asymmetry"],
        combat_asymmetry=combat["asymmetry"],
        farming_position=farming["position"],
        combat_position=combat["position"],
        farming_hero_history=farming["hero_history"],
        combat_hero_history=combat["hero_history"],
        farming_player_history=farming["player_history"],
        combat_player_history=combat["player_history"],
        farming_unique_player=farming["unique_player"],
        combat_unique_player=combat["unique_player"],
        farming_specialist=farming["specialist"],
        combat_specialist=combat["specialist"],
        farming_patch=farming["patch"],
        combat_patch=combat["patch"],
        farming_permutation=farming["permutation"],
        combat_permutation=combat["permutation"],
        farming_collinearity=farming["collinearity"],
        combat_collinearity=combat["collinearity"],
        integrity=integrity,
    )


def run_player_hero_compatibility_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
    n_permutations: int = N_PERMUTATIONS,
    n_bootstrap: int = N_BOOTSTRAP,
    n_shape_bins: int = N_SHAPE_BINS,
    rng_seed: int = RNG_SEED,
) -> Slice23DiagnosticReport:
    """Development-only Slice 23 compatibility research. Does not train a win model."""
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
        k_farm=FROZEN_HERO_FARM_SHRINKAGE_K,
        k_combat=FROZEN_HERO_COMBAT_SHRINKAGE_K,
    )
    report = run_compatibility_diagnostics_on_frame(
        development,
        development_end=end,
        n_holdout_excluded=len(holdout),
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        n_shape_bins=n_shape_bins,
        rng_seed=rng_seed,
    )
    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    integrity = dict(report.integrity)
    integrity["box_scores_in_feature_match_players_view"] = any(
        column in view_columns for column in BOX_SCORE_COLUMNS
    )
    integrity["match_player_box_score_field_count"] = len(MATCH_PLAYER_BOX_SCORE_COLUMNS)
    return Slice23DiagnosticReport(
        **{**report.__dict__, "integrity": integrity},
    )


def slice23_report_to_jsonable(report: Slice23DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 23 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "tune_end": report.tune_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "frozen_player_farm_k": FROZEN_SHRINKAGE_K,
        "frozen_player_combat_k": FROZEN_COMBAT_SHRINKAGE_K,
        "frozen_hero_farm_k": FROZEN_HERO_FARM_SHRINKAGE_K,
        "frozen_hero_combat_k": FROZEN_HERO_COMBAT_SHRINKAGE_K,
        "farming_target": FARMING_SPEC.target,
        "combat_target": COMBAT_SPEC.target,
        "farming_semantics": _jsonable_value(report.farming_semantics),
        "combat_semantics": _jsonable_value(report.combat_semantics),
        "classification": _jsonable_value(report.classification),
        "split": _jsonable_value(report.split),
        "farming_coverage": _jsonable_value(report.farming_coverage),
        "combat_coverage": _jsonable_value(report.combat_coverage),
        "farming_marginal": _jsonable_value(report.farming_marginal),
        "combat_marginal": _jsonable_value(report.combat_marginal),
        "farming_comparison": _jsonable_value(report.farming_comparison),
        "combat_comparison": _jsonable_value(report.combat_comparison),
        "farming_residual": _jsonable_value(report.farming_residual),
        "combat_residual": _jsonable_value(report.combat_residual),
        "farming_shape_bins": _jsonable_value(report.farming_shape_bins),
        "combat_shape_bins": _jsonable_value(report.combat_shape_bins),
        "farming_asymmetry": _jsonable_value(report.farming_asymmetry),
        "combat_asymmetry": _jsonable_value(report.combat_asymmetry),
        "farming_position": _jsonable_value(report.farming_position),
        "combat_position": _jsonable_value(report.combat_position),
        "farming_hero_history": _jsonable_value(report.farming_hero_history),
        "combat_hero_history": _jsonable_value(report.combat_hero_history),
        "farming_player_history": _jsonable_value(report.farming_player_history),
        "combat_player_history": _jsonable_value(report.combat_player_history),
        "farming_unique_player": _jsonable_value(report.farming_unique_player),
        "combat_unique_player": _jsonable_value(report.combat_unique_player),
        "farming_specialist": _jsonable_value(report.farming_specialist),
        "combat_specialist": _jsonable_value(report.combat_specialist),
        "farming_patch": _jsonable_value(report.farming_patch),
        "combat_patch": _jsonable_value(report.combat_patch),
        "farming_permutation": _jsonable_value(report.farming_permutation),
        "combat_permutation": _jsonable_value(report.combat_permutation),
        "farming_collinearity": _jsonable_value(report.farming_collinearity),
        "combat_collinearity": _jsonable_value(report.combat_collinearity),
        "integrity": _jsonable_value(report.integrity),
    }
