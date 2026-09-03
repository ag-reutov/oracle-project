"""Slice 20: walk-forward evaluation of frozen player combat vs Elo.

One incremental test. The candidate is the Slice 19 named spec
``logistic_elo_plus_player_combat`` (Elo + ``mean_combat_shrunk_c_diff``).
The reference is the frozen logistic Elo spec ``logistic_elo_only``.

Does not redesign candidate C, ``k``, player history, or the five-player
Radiant − Dire mean. Does not add ``prior_n``, raw combat means,
farming, or interactions. Does not change production ``FEATURE_COLUMNS``
or Slice 9 specs.

Holdout policy
--------------
The frozen Slice 9 holdout remains reserved until an explicit later
promotion of a production spec. Slice 9's one-shot scorer is locked to
Career Player × Hero. Later research slices (10–19) evaluate on the
development frame only. Slice 20 follows that policy: expanding-window
OOS on ``start_time <= FROZEN_DEVELOPMENT_END``. The holdout is not
scored and is not used to choose ``C`` or preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.player_combat_comparison import (
    COMBAT_CAUSAL_C_COLUMN,
    COMBAT_ROSTER_SIDE_SIZE,
    MATCH_ID_COLUMN,
    PLAYER_COMBAT_COMPARISON_EVIDENCE_COLUMNS,
    PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS,
    PLAYER_COMBAT_FEATURE_COLUMNS,
    PLAYER_COMBAT_REQUIRED_COLUMNS,
    PLAYER_COMBAT_STATE_FEATURE_COLUMNS,
)
from dota_predictor.features.player_farming_comparison import (
    PLAYER_FARMING_FEATURE_COLUMNS,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
    build_pre_draft_snapshot,
)
from dota_predictor.features.team_elo import (
    DEFAULT_ELO_CONFIG,
    TEAM_ELO_DELTA_COLUMN,
    TEAM_ELO_FEATURE_COLUMNS,
    EloConfig,
)
from dota_predictor.training.combat_performance_target import (
    COMBAT_C_POSITION,
    FROZEN_COMBAT_CANDIDATE,
)
from dota_predictor.training.dataset import ModelReadyDataset, TrainingDatasetError
from dota_predictor.training.evaluation import _fit_logistic
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_PLAYER_COMBAT_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
    SLICE15_CANDIDATE_SPEC,
    SLICE19_CANDIDATE_SPEC,
    SLICE19_CANDIDATE_SPEC_NAME,
    SLICE19_FROZEN_SPECS,
    SLICE19_REFERENCE_SPEC,
    SLICE19_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.logistic_model import (
    LogisticRegressionConfig,
    standardized_coefficients,
)
from dota_predictor.training.metrics import (
    bootstrap_mean_ci,
    evaluate_probabilities,
    per_sample_brier,
    per_sample_log_loss,
)
from dota_predictor.training.player_combat_comparison import (
    build_player_combat_comparison,
)
from dota_predictor.training.player_combat_state import (
    FROZEN_COMBAT_SHRINKAGE_K,
    SLICE18_STATE_COLUMNS,
)
from dota_predictor.training.player_farming_comparison import (
    build_player_farming_comparison,
)
from dota_predictor.training.player_farming_state import (
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
)
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    _jsonable_value,
    _numeric,
    _pearson,
    _spearman,
    _std,
    restrict_development,
)
from dota_predictor.training.preprocessing import MISSINGNESS_INDICATOR_SUFFIX
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    FROZEN_DEVELOPMENT_MATCH_COUNT,
    FROZEN_DEVELOPMENT_OOS_MATCH_COUNT,
    FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES,
    FROZEN_HOLDOUT_BOOTSTRAP_SEED,
    assert_development_frame_excludes_holdout,
    utc_datetime,
)
from dota_predictor.training.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG,
    ELO_BLOCK_SPEC_NAME,
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardReport,
    run_post_draft_walk_forward,
)

__all__ = [
    "ABS_COMBAT_BUCKET_COUNT",
    "CLASSIFICATION_A",
    "CLASSIFICATION_B",
    "CLASSIFICATION_C",
    "COMBAT_FEATURE_COLUMN",
    "FARMING_FEATURE_COLUMN",
    "HOLDOUT_POLICY",
    "PREDICTION_MOVE_THRESHOLDS",
    "SLICE16_FARMING_FROZEN_RESULT",
    "SLICE20_BOOTSTRAP_RESAMPLES",
    "SLICE20_BOOTSTRAP_SEED",
    "SLICE20_CANDIDATE_SPEC",
    "SLICE20_CANDIDATE_SPEC_NAME",
    "SLICE20_FROZEN_SPECS",
    "SLICE20_REFERENCE_SPEC",
    "SLICE20_REFERENCE_SPEC_NAME",
    "Slice20Assembly",
    "Slice20BenchmarkReport",
    "assign_abs_quantile_bucket",
    "assign_signed_combat_bucket",
    "build_slice20_model_ready_dataset",
    "classify_slice20",
    "run_slice20_player_combat_benchmark",
    "slice20_report_to_jsonable",
]


# Repository names. User-facing "logistic_elo" is ``logistic_elo_only``.
SLICE20_REFERENCE_SPEC_NAME = SLICE19_REFERENCE_SPEC_NAME
SLICE20_CANDIDATE_SPEC_NAME = SLICE19_CANDIDATE_SPEC_NAME
SLICE20_REFERENCE_SPEC = SLICE19_REFERENCE_SPEC
SLICE20_CANDIDATE_SPEC = SLICE19_CANDIDATE_SPEC
# Alias of the Slice 19 *benchmark/evaluation* spec. Held fixed so this
# walk-forward cannot retune columns. Does not mean combat is production.
SLICE20_FROZEN_SPECS = SLICE19_FROZEN_SPECS
COMBAT_FEATURE_COLUMN = PLAYER_COMBAT_FEATURE_COLUMNS[0]
FARMING_FEATURE_COLUMN = PLAYER_FARMING_FEATURE_COLUMNS[0]

SLICE20_BOOTSTRAP_RESAMPLES = FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES
SLICE20_BOOTSTRAP_SEED = FROZEN_HOLDOUT_BOOTSTRAP_SEED
ABS_COMBAT_BUCKET_COUNT = 4
PREDICTION_MOVE_THRESHOLDS: tuple[float, ...] = (0.01, 0.02, 0.05)

HOLDOUT_POLICY = (
    "development_oos_only: frozen Slice 9 holdout remains reserved until "
    "an explicit later promotion of a production spec. Slice 9's one-shot "
    "scorer is locked to Career Player × Hero. Combat's target, k, and "
    "comparison formula were methodologically frozen in Slices 17–19, and "
    "the Slice 19 evaluation spec is held fixed here, but this research "
    "candidate is not a promoted production spec, so Slice 20 scores "
    "development walk-forward OOS only "
    "(start_time <= FROZEN_DEVELOPMENT_END)."
)

CLASSIFICATION_A = "A — consistent incremental combat signal beyond Elo"
CLASSIFICATION_B = (
    "B — weak/mixed incremental combat signal; retain state but do not promote"
)
CLASSIFICATION_C = "C — no useful incremental combat win signal beyond Elo"

# Frozen Slice 16 farming result. Recorded as stated; Slice 20 does not
# re-score farming or run a farming-vs-combat promotion test.
SLICE16_FARMING_FROZEN_RESULT: dict[str, object] = {
    "pooled_delta_log_loss_approx": -0.00029,
    "ci_included_zero": True,
    "n_folds_improved": 2,
    "n_folds": 4,
    "coefficient": "stable_positive",
    "classification": (
        "B — weak/mixed incremental signal; retain state but do not promote"
    ),
}

_FORBIDDEN_MODEL_COLUMNS: tuple[str, ...] = (
    "combat_prior_n",
    "mean_combat_prior_n",
    "mean_combat_prior_n_diff",
    "min_combat_prior_n",
    "radiant_combat_prior_n_sum",
    "dire_combat_prior_n_sum",
    "radiant_combat_cold_start_count",
    "dire_combat_cold_start_count",
    "radiant_mean_combat_shrunk_c",
    "dire_mean_combat_shrunk_c",
    FARMING_FEATURE_COLUMN,
    "farming_prior_n",
    "hero_id",
    "position",
    "position_number",
    "hero_damage",
    "kills",
    "assists",
    "deaths",
    "num_last_hits",
    "duration_seconds",
    "networth",
    COMBAT_CAUSAL_C_COLUMN,
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class Slice20Assembly:
    """Elo + frozen combat comparison on the development frame."""

    dataset: ModelReadyDataset
    n_snapshot_matches: int
    n_combat_comparison_matches: int
    n_missing_combat_join: int
    n_holdout_excluded: int
    farming_by_match: pd.DataFrame


@dataclass
class Slice20BenchmarkReport:
    """Walk-forward paired evaluation of frozen combat vs logistic Elo."""

    development_end: datetime
    holdout_policy: str
    n_development_matches: int
    n_holdout_excluded: int
    n_oos: int
    frozen_combat_k: float
    frozen_farming_k: float
    assembly: Slice20Assembly
    walk_forward: WalkForwardReport
    fold_table: pd.DataFrame
    pooled: pd.DataFrame
    bootstrap: dict[str, object]
    coefficients: pd.DataFrame
    fold_sign_consistency: dict[str, object]
    combat_elo_pearson: float
    combat_elo_spearman: float
    combat_farming_pearson: float
    combat_farming_spearman: float
    combat_distribution: pd.DataFrame
    elo_by_combat_bucket: pd.DataFrame
    prediction_movement: dict[str, object]
    magnitude_buckets: pd.DataFrame
    directional_buckets: pd.DataFrame
    calibration: dict[str, object]
    slice16_comparison: dict[str, object]
    integrity: dict[str, object]
    classification: str
    classification_rationale: str


def build_slice20_model_ready_dataset(
    store: FeatureDuckDBConnection,
    *,
    elo_config: EloConfig | None = None,
    development_end: datetime | None = None,
) -> Slice20Assembly:
    """PRE_DRAFT Elo plus the frozen combat comparison, development only.

    Combat state is attached with ``development_end`` set so later
    holdout box scores cannot enter the position baseline or player
    history. The current match still never contributes hero damage,
    kills, assists, deaths, duration, position, hero, or result to
    ``mean_combat_shrunk_c_diff``. Farming is joined for diagnostics
    only and is not a model column.
    """
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    resolved_elo = elo_config if elo_config is not None else DEFAULT_ELO_CONFIG
    snapshot = build_pre_draft_snapshot(store, elo_config=resolved_elo)
    full = snapshot.to_frame()
    n_snapshot = int(full[MATCH_ID_COLUMN].nunique())
    stamp = pd.to_datetime(full["start_time"], utc=True)
    holdout_n = int((stamp > pd.Timestamp(end)).sum())
    development = restrict_development(full, development_end=end)
    comparison = build_player_combat_comparison(
        store,
        k=FROZEN_COMBAT_SHRINKAGE_K,
        elo_config=resolved_elo,
        development_end=end,
    )
    combat = comparison.loc[:, [MATCH_ID_COLUMN, *PLAYER_COMBAT_FEATURE_COLUMNS]]
    merged = development.merge(
        combat,
        on=MATCH_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    farming_comparison = build_player_farming_comparison(
        store, k=FROZEN_SHRINKAGE_K, elo_config=resolved_elo, development_end=end
    )
    farming = farming_comparison.loc[
        :, [MATCH_ID_COLUMN, *PLAYER_FARMING_FEATURE_COLUMNS]
    ]
    ordered = merged.sort_values(
        ["start_time", MATCH_ID_COLUMN], kind="stable"
    ).reset_index(drop=True)
    feature_columns = ELO_PLUS_PLAYER_COMBAT_COLUMNS
    missing = [name for name in feature_columns if name not in ordered.columns]
    if missing:
        raise TrainingDatasetError(
            f"Slice 20 assembly is missing required columns: {missing}"
        )
    overlap = set(FEATURE_COLUMNS) & set(PLAYER_COMBAT_FEATURE_COLUMNS)
    if overlap:
        raise TrainingDatasetError(
            "combat candidate columns must not appear in FEATURE_COLUMNS: "
            f"{sorted(overlap)}"
        )
    farming_in_x = [
        name for name in PLAYER_FARMING_FEATURE_COLUMNS if name in feature_columns
    ]
    if farming_in_x:
        raise TrainingDatasetError(
            f"Slice 20 feature matrix must not include farming columns: {farming_in_x}"
        )
    forbidden = [name for name in _FORBIDDEN_MODEL_COLUMNS if name in feature_columns]
    if forbidden:
        raise TrainingDatasetError(
            f"Slice 20 feature matrix must not include {forbidden}"
        )
    dataset = ModelReadyDataset(
        X=ordered[list(feature_columns)].copy(),
        y=ordered[TARGET_COLUMN].copy(),
        context=ordered[list(IDENTITY_COLUMNS)].copy(),
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        identity_columns=IDENTITY_COLUMNS,
    )
    assert_development_frame_excludes_holdout(dataset.context, development_end=end)
    present_ids = set(combat[MATCH_ID_COLUMN].tolist()) if len(combat) else set()
    n_missing_join = int((~ordered[MATCH_ID_COLUMN].isin(present_ids)).sum())
    return Slice20Assembly(
        dataset=dataset,
        n_snapshot_matches=n_snapshot,
        n_combat_comparison_matches=int(combat[MATCH_ID_COLUMN].nunique())
        if len(combat)
        else 0,
        n_missing_combat_join=n_missing_join,
        n_holdout_excluded=holdout_n,
        farming_by_match=farming.copy(),
    )


def _quantile_edges(values: np.ndarray, n_buckets: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    probs = np.linspace(0.0, 1.0, n_buckets + 1)
    edges = np.quantile(finite, probs)
    unique = np.unique(edges)
    if unique.size < 2:
        lo = float(unique[0])
        return np.array([lo, lo], dtype=float)
    return unique.astype(float)


def assign_abs_quantile_bucket(value: float, edges: np.ndarray) -> str:
    """Map ``|combat diff|`` onto precomputed quantile edges.

    Edges come from the OOS ``|mean_combat_shrunk_c_diff|`` distribution
    only. They are not chosen from prediction metrics.
    """
    if not np.isfinite(value):
        return "NULL"
    if edges.size < 2:
        return "Q1"
    n_bins = int(edges.size - 1)
    if edges[0] == edges[-1]:
        return "Q1"
    idx = int(np.searchsorted(edges, float(value), side="right") - 1)
    idx = min(max(idx, 0), n_bins - 1)
    return f"Q{idx + 1}"


def assign_signed_combat_bucket(value: float) -> str:
    """Fixed sign split. Cutoffs are not optimized from outcomes."""
    if not np.isfinite(value):
        return "NULL"
    if value < 0.0:
        return "combat_diff_lt_0"
    if value > 0.0:
        return "combat_diff_gt_0"
    return "combat_diff_eq_0"


def _paired_oos(walk_forward: WalkForwardReport) -> pd.DataFrame:
    oos = walk_forward.oos_predictions
    reference = oos.loc[oos["model"] == SLICE20_REFERENCE_SPEC_NAME].copy()
    candidate = oos.loc[oos["model"] == SLICE20_CANDIDATE_SPEC_NAME].copy()
    ref_ids = reference[MATCH_ID_COLUMN].to_numpy()
    cand_ids = candidate[MATCH_ID_COLUMN].to_numpy()
    if ref_ids.shape != cand_ids.shape or not np.array_equal(ref_ids, cand_ids):
        raise TrainingDatasetError(
            "reference and candidate OOS match_ids are not identical"
        )
    paired = candidate.rename(
        columns={
            "p_spec": "p_candidate",
            "sample_log_loss": "candidate_log_loss",
            "delta_vs_elo": "delta_log_loss",
        }
    )
    paired["p_reference"] = reference["p_spec"].to_numpy()
    paired["reference_log_loss"] = reference["sample_log_loss"].to_numpy()
    paired["prediction_delta"] = paired["p_candidate"] - paired["p_reference"]
    paired["reference_brier"] = per_sample_brier(
        paired["y_true"], paired["p_reference"]
    )
    paired["candidate_brier"] = per_sample_brier(
        paired["y_true"], paired["p_candidate"]
    )
    return paired.reset_index(drop=True)


def _attach_features(
    paired: pd.DataFrame,
    dataset: ModelReadyDataset,
    farming_by_match: pd.DataFrame,
) -> pd.DataFrame:
    features = dataset.context[[MATCH_ID_COLUMN]].copy()
    features[TEAM_ELO_DELTA_COLUMN] = dataset.X[TEAM_ELO_DELTA_COLUMN].to_numpy()
    features[COMBAT_FEATURE_COLUMN] = dataset.X[COMBAT_FEATURE_COLUMN].to_numpy()
    attached = paired.merge(
        features, on=MATCH_ID_COLUMN, how="left", validate="one_to_one"
    )
    farming = farming_by_match.loc[
        :, [MATCH_ID_COLUMN, *PLAYER_FARMING_FEATURE_COLUMNS]
    ]
    return attached.merge(
        farming, on=MATCH_ID_COLUMN, how="left", validate="one_to_one"
    )


def _combat_coefficient(table: pd.DataFrame) -> float:
    rows = table.loc[table["feature"] == COMBAT_FEATURE_COLUMN]
    if rows.empty:
        return float("nan")
    return float(rows["coefficient"].iloc[0])


def _fold_coefficients(
    folds: tuple[WalkForwardFold, ...], selected_c: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in folds:
        c_lookup = selected_c.loc[selected_c["fold_id"] == fold.fold_id]
        for spec in SLICE20_FROZEN_SPECS:
            c = float(c_lookup.loc[c_lookup["model"] == spec.name, "C"].iloc[0])
            model = _fit_logistic(
                fold.train,
                spec.feature_columns,
                config=LogisticRegressionConfig(C=c),
            )
            coef = standardized_coefficients(model)
            combat = _combat_coefficient(coef)
            train_combat = _numeric(fold.train.X[COMBAT_FEATURE_COLUMN])
            finite = train_combat.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": spec.name,
                    "C": c,
                    "combat_coefficient": combat
                    if spec.name == SLICE20_CANDIDATE_SPEC_NAME
                    else float("nan"),
                    "train_combat_mean": (
                        float(finite.mean()) if finite.size else float("nan")
                    ),
                    "train_combat_std": (_std(finite) if finite.size else float("nan")),
                    "n_train": len(fold.train),
                    "missingness_indicator_in_model": any(
                        str(name).endswith(MISSINGNESS_INDICATOR_SUFFIX)
                        for name in coef["feature"].tolist()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _period(partition_times: pd.Series) -> tuple[datetime, datetime]:
    return utc_datetime(partition_times.min()), utc_datetime(partition_times.max())


def _fold_table(
    walk_forward: WalkForwardReport,
    paired: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cand_coef = coefficients.loc[
        coefficients["model"] == SLICE20_CANDIDATE_SPEC_NAME
    ].set_index("fold_id")
    for fold in walk_forward.folds:
        subset = paired.loc[paired["fold_id"] == fold.fold_id]
        train_start, train_end = _period(fold.train.context["start_time"])
        val_start, val_end = _period(fold.validation.context["start_time"])
        test_start, test_end = _period(fold.test.context["start_time"])
        y = subset["y_true"]
        ref_metrics = evaluate_probabilities(y, subset["p_reference"])
        cand_metrics = evaluate_probabilities(y, subset["p_candidate"])
        delta = per_sample_log_loss(y, subset["p_candidate"]) - per_sample_log_loss(
            y, subset["p_reference"]
        )
        coef_row = cand_coef.loc[fold.fold_id]
        rows.append(
            {
                "fold": fold.fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": val_start,
                "validation_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
                "n_matches": len(subset),
                "reference_log_loss": ref_metrics.log_loss,
                "candidate_log_loss": cand_metrics.log_loss,
                "paired_delta_log_loss": float(delta.mean()),
                "reference_brier": ref_metrics.brier_score,
                "candidate_brier": cand_metrics.brier_score,
                "combat_coefficient": float(coef_row["combat_coefficient"]),
                "train_combat_mean": float(coef_row["train_combat_mean"]),
                "train_combat_std": float(coef_row["train_combat_std"]),
                "C_reference": float(
                    walk_forward.selected_C.loc[
                        (walk_forward.selected_C["fold_id"] == fold.fold_id)
                        & (
                            walk_forward.selected_C["model"]
                            == SLICE20_REFERENCE_SPEC_NAME
                        ),
                        "C",
                    ].iloc[0]
                ),
                "C_candidate": float(coef_row["C"]),
            }
        )
    return pd.DataFrame(rows)


def _pooled_row(paired: pd.DataFrame) -> pd.DataFrame:
    y = paired["y_true"]
    ref = evaluate_probabilities(y, paired["p_reference"])
    cand = evaluate_probabilities(y, paired["p_candidate"])
    delta = paired["candidate_log_loss"] - paired["reference_log_loss"]
    return pd.DataFrame(
        [
            {
                "n": len(paired),
                "reference_log_loss": ref.log_loss,
                "candidate_log_loss": cand.log_loss,
                "paired_delta_log_loss": float(delta.mean()),
                "reference_brier": ref.brier_score,
                "candidate_brier": cand.brier_score,
                "reference_ece": ref.expected_calibration_error,
                "candidate_ece": cand.expected_calibration_error,
                "reference_roc_auc": ref.roc_auc,
                "candidate_roc_auc": cand.roc_auc,
            }
        ]
    )


def _bootstrap_delta(delta: np.ndarray) -> dict[str, object]:
    observed = float(delta.mean()) if delta.size else float("nan")
    ci_lo, ci_hi = bootstrap_mean_ci(
        delta,
        n_resamples=SLICE20_BOOTSTRAP_RESAMPLES,
        random_state=SLICE20_BOOTSTRAP_SEED,
    )
    rng = np.random.default_rng(SLICE20_BOOTSTRAP_SEED)
    if delta.size == 0:
        frac_negative = float("nan")
        boot_mean = float("nan")
    else:
        draws = rng.choice(
            delta, size=(SLICE20_BOOTSTRAP_RESAMPLES, delta.size), replace=True
        )
        means = draws.mean(axis=1)
        frac_negative = float((means < 0.0).mean())
        boot_mean = float(means.mean())
    return {
        "observed_delta": observed,
        "bootstrap_mean_delta": boot_mean,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "frac_delta_negative": frac_negative,
        "n_resamples": SLICE20_BOOTSTRAP_RESAMPLES,
        "seed": SLICE20_BOOTSTRAP_SEED,
        "grouping": "oos_match",
    }


def _sign(value: float) -> int:
    if not np.isfinite(value) or value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def _fold_sign_consistency(fold_table: pd.DataFrame) -> dict[str, object]:
    deltas = fold_table["paired_delta_log_loss"].to_numpy(dtype=float)
    coefs = fold_table["combat_coefficient"].to_numpy(dtype=float)
    n_neg = int((deltas < 0.0).sum())
    n_pos = int((deltas > 0.0).sum())
    n_zero = int((deltas == 0.0).sum()) + int((~np.isfinite(deltas)).sum())
    coef_signs = [_sign(value) for value in coefs]
    n_pos_coef = coef_signs.count(1)
    n_neg_coef = coef_signs.count(-1)
    unique_nonzero = {sign for sign in coef_signs if sign != 0}
    return {
        "n_folds": len(fold_table),
        "n_folds_delta_negative": n_neg,
        "n_folds_delta_positive": n_pos,
        "n_folds_delta_zero": n_zero,
        "delta_by_fold": [float(value) for value in deltas],
        "coefficient_by_fold": [float(value) for value in coefs],
        "coefficient_sign_by_fold": coef_signs,
        "n_folds_coefficient_positive": n_pos_coef,
        "n_folds_coefficient_negative": n_neg_coef,
        "coefficient_sign_stable": len(unique_nonzero) <= 1,
    }


def _distribution(values: pd.Series, *, column: str) -> pd.DataFrame:
    raw = _numeric(values)
    finite = raw.to_numpy(dtype=float)
    n_null = int((~np.isfinite(finite)).sum())
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        stats = {
            "column": column,
            "n": 0,
            "n_null": n_null,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "mean_abs": float("nan"),
        }
        return pd.DataFrame([stats])
    return pd.DataFrame(
        [
            {
                "column": column,
                "n": int(finite.size),
                "n_null": n_null,
                "mean": float(finite.mean()),
                "std": _std(finite),
                "median": float(np.median(finite)),
                "p05": float(np.quantile(finite, 0.05)),
                "p25": float(np.quantile(finite, 0.25)),
                "p75": float(np.quantile(finite, 0.75)),
                "p95": float(np.quantile(finite, 0.95)),
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean_abs": float(np.mean(np.abs(finite))),
            }
        ]
    )


def _prediction_movement(delta: np.ndarray) -> dict[str, object]:
    abs_delta = np.abs(delta)
    finite = abs_delta[np.isfinite(abs_delta)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean_abs": float("nan"),
            "median_abs": float("nan"),
            "p90_abs": float("nan"),
            "p95_abs": float("nan"),
            "max_abs": float("nan"),
            "frac_ge_1pp": float("nan"),
            "frac_ge_2pp": float("nan"),
            "frac_ge_5pp": float("nan"),
        }
    return {
        "n": int(finite.size),
        "mean_abs": float(finite.mean()),
        "median_abs": float(np.median(finite)),
        "p90_abs": float(np.quantile(finite, 0.90)),
        "p95_abs": float(np.quantile(finite, 0.95)),
        "max_abs": float(finite.max()),
        "frac_ge_1pp": float((finite >= PREDICTION_MOVE_THRESHOLDS[0]).mean()),
        "frac_ge_2pp": float((finite >= PREDICTION_MOVE_THRESHOLDS[1]).mean()),
        "frac_ge_5pp": float((finite >= PREDICTION_MOVE_THRESHOLDS[2]).mean()),
    }


def _bucket_sort_key(label: str) -> int:
    if label.startswith("Q"):
        return int(label[1:])
    signed = {
        "combat_diff_lt_0": 0,
        "combat_diff_eq_0": 1,
        "combat_diff_gt_0": 2,
    }
    return signed.get(label, 99)


def _bucket_metrics(
    work: pd.DataFrame, *, bucket_column: str, mean_abs_column: str | None
) -> pd.DataFrame:
    labels = work[bucket_column].tolist()
    order = sorted(
        {label for label in labels if label != "NULL"},
        key=_bucket_sort_key,
    )
    if "NULL" in labels:
        order.append("NULL")
    rows: list[dict[str, object]] = []
    for label in order:
        subset = work.loc[work[bucket_column] == label]
        if subset.empty:
            continue
        y = subset["y_true"]
        ref_ll = float(per_sample_log_loss(y, subset["p_reference"]).mean())
        cand_ll = float(per_sample_log_loss(y, subset["p_candidate"]).mean())
        row: dict[str, object] = {
            "bucket": label,
            "n": len(subset),
            "reference_log_loss": ref_ll,
            "candidate_log_loss": cand_ll,
            "paired_delta_log_loss": cand_ll - ref_ll,
        }
        if mean_abs_column is not None:
            row["mean_abs_combat_diff"] = float(
                np.nanmean(subset[mean_abs_column].to_numpy(dtype=float))
            )
        else:
            combat = _numeric(subset[COMBAT_FEATURE_COLUMN]).to_numpy(dtype=float)
            finite = combat[np.isfinite(combat)]
            row["mean_combat_diff"] = (
                float(finite.mean()) if finite.size else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _magnitude_buckets(paired: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    abs_diff = np.abs(_numeric(paired[COMBAT_FEATURE_COLUMN]).to_numpy(dtype=float))
    edges = _quantile_edges(abs_diff, ABS_COMBAT_BUCKET_COUNT)
    work = paired.copy()
    work["abs_combat_diff"] = abs_diff
    work["abs_combat_bucket"] = [
        assign_abs_quantile_bucket(value, edges) for value in abs_diff
    ]
    return _bucket_metrics(
        work, bucket_column="abs_combat_bucket", mean_abs_column="abs_combat_diff"
    ), edges


def _directional_buckets(paired: pd.DataFrame) -> pd.DataFrame:
    combat = _numeric(paired[COMBAT_FEATURE_COLUMN]).to_numpy(dtype=float)
    work = paired.copy()
    work["signed_combat_bucket"] = [
        assign_signed_combat_bucket(value) for value in combat
    ]
    return _bucket_metrics(
        work, bucket_column="signed_combat_bucket", mean_abs_column=None
    )


def _elo_by_combat_bucket(paired: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    abs_diff = np.abs(_numeric(paired[COMBAT_FEATURE_COLUMN]).to_numpy(dtype=float))
    labels = [assign_abs_quantile_bucket(value, edges) for value in abs_diff]
    work = paired.copy()
    work["abs_combat_bucket"] = labels
    rows: list[dict[str, object]] = []
    order = sorted(
        {label for label in labels if label != "NULL"},
        key=lambda item: int(item[1:]) if item.startswith("Q") else 99,
    )
    for label in order:
        subset = work.loc[work["abs_combat_bucket"] == label]
        elo = _numeric(subset[TEAM_ELO_DELTA_COLUMN]).to_numpy(dtype=float)
        finite = elo[np.isfinite(elo)]
        rows.append(
            {
                "bucket": label,
                "n": len(subset),
                "mean_elo_diff": float(finite.mean()) if finite.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _calibration(paired: pd.DataFrame) -> dict[str, object]:
    ref = evaluate_probabilities(paired["y_true"], paired["p_reference"])
    cand = evaluate_probabilities(paired["y_true"], paired["p_candidate"])
    return {
        "reference_brier": ref.brier_score,
        "candidate_brier": cand.brier_score,
        "reference_ece": ref.expected_calibration_error,
        "candidate_ece": cand.expected_calibration_error,
        "reference_bins": ref.calibration_table,
        "candidate_bins": cand.calibration_table,
        "intercept_slope": None,
        "intercept_slope_available": False,
    }


def classify_slice20(
    *,
    pooled_delta: float,
    ci_low: float,
    ci_high: float,
    frac_delta_negative: float,
    n_folds_delta_negative: int,
    n_folds_delta_positive: int,
    n_folds: int,
    coefficient_sign_stable: bool,
    reference_brier: float,
    candidate_brier: float,
    reference_ece: float,
    candidate_ece: float,
    mean_abs_prediction_delta: float,
) -> tuple[str, str]:
    """Map frozen-protocol diagnostics onto A / B / C. Does not retune."""
    ci_excludes_zero_negative = (
        np.isfinite(ci_high) and np.isfinite(pooled_delta) and ci_high < 0.0
    )
    ci_excludes_zero_positive = (
        np.isfinite(ci_low) and np.isfinite(pooled_delta) and ci_low > 0.0
    )
    majority_negative = (
        n_folds >= 2
        and n_folds_delta_negative >= max(2, n_folds - 1)
        and n_folds_delta_positive == 0
    )
    mixed_folds = n_folds_delta_negative > 0 and n_folds_delta_positive > 0
    isolated_fold = n_folds >= 3 and n_folds_delta_negative == 1 and pooled_delta < 0.0
    brier_worse = (
        np.isfinite(candidate_brier)
        and np.isfinite(reference_brier)
        and candidate_brier > reference_brier + 1e-4
    )
    ece_much_worse = (
        np.isfinite(candidate_ece)
        and np.isfinite(reference_ece)
        and candidate_ece > reference_ece + 0.02
    )
    tiny_movement = (
        np.isfinite(mean_abs_prediction_delta) and mean_abs_prediction_delta < 0.002
    )

    if (
        np.isfinite(pooled_delta)
        and pooled_delta < 0.0
        and ci_excludes_zero_negative
        and majority_negative
        and not isolated_fold
        and coefficient_sign_stable
        and not brier_worse
        and not ece_much_worse
    ):
        return (
            CLASSIFICATION_A,
            (
                "Pooled paired Δ log loss is negative, the bootstrap CI "
                "excludes zero on the improvement side, folds agree, the "
                "combat coefficient sign is stable, and calibration does "
                "not degrade."
            ),
        )
    if (
        (np.isfinite(pooled_delta) and pooled_delta > 0.0 and ci_excludes_zero_positive)
        or (n_folds >= 2 and n_folds_delta_positive == n_folds)
        or (
            np.isfinite(pooled_delta)
            and pooled_delta >= 0.0
            and tiny_movement
            and (not np.isfinite(frac_delta_negative) or frac_delta_negative < 0.5)
        )
        or (brier_worse and ece_much_worse and pooled_delta >= 0.0)
    ):
        return (
            CLASSIFICATION_C,
            (
                "The candidate does not improve on Elo: pooled Δ is "
                "non-negative with evidence against improvement, folds do "
                "not support a gain, and/or prediction quality worsens."
            ),
        )
    reasons: list[str] = []
    if not np.isfinite(pooled_delta) or abs(pooled_delta) < 1e-5:
        reasons.append("pooled Δ is near zero")
    elif pooled_delta < 0.0 and not ci_excludes_zero_negative:
        reasons.append("pooled Δ is negative but the CI includes zero")
    elif pooled_delta > 0.0 and not ci_excludes_zero_positive:
        reasons.append("pooled Δ is positive but the CI includes zero")
    if mixed_folds:
        reasons.append("folds disagree on the sign of Δ")
    if isolated_fold:
        reasons.append("pooled improvement is concentrated in one fold")
    if not coefficient_sign_stable:
        reasons.append("combat coefficient sign is unstable")
    if brier_worse:
        reasons.append("candidate Brier is worse")
    if tiny_movement:
        reasons.append("prediction movement is tiny")
    if not reasons:
        reasons.append("the incremental effect is too small or uncertain to promote")
    return CLASSIFICATION_B, "; ".join(reasons) + "."


def _slice16_qualitative_comparison(
    *,
    pooled_delta: float,
    ci_low: float,
    ci_high: float,
    n_folds_delta_negative: int,
    n_folds: int,
    coefficient_sign_stable: bool,
    n_folds_coefficient_positive: int,
    mean_abs_prediction_delta: float,
    reference_ece: float,
    candidate_ece: float,
    classification: str,
) -> dict[str, object]:
    ci_includes_zero = (
        np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= 0.0 <= ci_high
    )
    return {
        "farming_pooled_delta_approx": SLICE16_FARMING_FROZEN_RESULT[
            "pooled_delta_log_loss_approx"
        ],
        "farming_ci_included_zero": SLICE16_FARMING_FROZEN_RESULT["ci_included_zero"],
        "farming_folds_improved": (
            f"{SLICE16_FARMING_FROZEN_RESULT['n_folds_improved']}/"
            f"{SLICE16_FARMING_FROZEN_RESULT['n_folds']}"
        ),
        "farming_coefficient": SLICE16_FARMING_FROZEN_RESULT["coefficient"],
        "farming_classification": SLICE16_FARMING_FROZEN_RESULT["classification"],
        "combat_pooled_delta": pooled_delta,
        "combat_ci_includes_zero": ci_includes_zero,
        "combat_folds_improved": f"{n_folds_delta_negative}/{n_folds}",
        "combat_coefficient_sign_stable": coefficient_sign_stable,
        "combat_n_folds_coefficient_positive": n_folds_coefficient_positive,
        "combat_mean_abs_prediction_delta": mean_abs_prediction_delta,
        "combat_ece_reference": reference_ece,
        "combat_ece_candidate": candidate_ece,
        "combat_classification": classification,
        "models_combined": False,
        "statistical_test_between_farming_and_combat": False,
        "promotion_contest": False,
    }


def _integrity(
    store: FeatureDuckDBConnection,
    assembly: Slice20Assembly,
    paired: pd.DataFrame,
) -> dict[str, object]:
    dataset = assembly.dataset
    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    ref_ids = paired[MATCH_ID_COLUMN].to_numpy()
    extra = set(SLICE20_CANDIDATE_SPEC.feature_columns) - set(
        SLICE20_REFERENCE_SPEC.feature_columns
    )
    return {
        "holdout_policy": HOLDOUT_POLICY,
        "development_end": FROZEN_DEVELOPMENT_END.isoformat(),
        "frozen_combat_candidate": FROZEN_COMBAT_CANDIDATE,
        "slice17_candidate_unchanged": FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION,
        "frozen_combat_shrinkage_k": FROZEN_COMBAT_SHRINKAGE_K,
        "combat_k_is_20": FROZEN_COMBAT_SHRINKAGE_K == 20.0,
        "k_re_searched": False,
        "alternative_combat_features_searched": False,
        "farming_candidate_b": FROZEN_CANDIDATE_B,
        "farming_candidate_b_unchanged": FROZEN_CANDIDATE_B == CANDIDATE_B,
        "farming_frozen_shrinkage_k": FROZEN_SHRINKAGE_K,
        "farming_k_is_5": FROZEN_SHRINKAGE_K == 5.0,
        "farming_in_candidate": FARMING_FEATURE_COLUMN
        in SLICE20_CANDIDATE_SPEC.feature_columns,
        "farming_spec_unchanged": list(SLICE15_CANDIDATE_SPEC.feature_columns)
        == list(ELO_ONLY_FEATURE_COLUMNS) + list(PLAYER_FARMING_FEATURE_COLUMNS),
        "holdout_scored": False,
        "holdout_used_for_c": False,
        "holdout_used_for_feature": False,
        "stratz_called": False,
        "ingestion_modified": False,
        "schema_modified": False,
        "reference_spec": SLICE20_REFERENCE_SPEC_NAME,
        "candidate_spec": SLICE20_CANDIDATE_SPEC_NAME,
        "reference_columns": list(SLICE20_REFERENCE_SPEC.feature_columns),
        "candidate_columns": list(SLICE20_CANDIDATE_SPEC.feature_columns),
        "reference_is_elo_only": list(SLICE20_REFERENCE_SPEC.feature_columns)
        == list(ELO_ONLY_FEATURE_COLUMNS)
        == list(TEAM_ELO_FEATURE_COLUMNS),
        "candidate_is_elo_plus_combat": list(SLICE20_CANDIDATE_SPEC.feature_columns)
        == list(ELO_PLUS_PLAYER_COMBAT_COLUMNS),
        "candidate_extra_columns": sorted(extra),
        "candidate_excludes_prior_n": all(
            "prior_n" not in name for name in SLICE20_CANDIDATE_SPEC.feature_columns
        ),
        "candidate_excludes_evidence": all(
            name not in SLICE20_CANDIDATE_SPEC.feature_columns
            for name in PLAYER_COMBAT_COMPARISON_EVIDENCE_COLUMNS
        ),
        "forbidden_columns_absent_from_x": all(
            name not in dataset.X.columns for name in _FORBIDDEN_MODEL_COLUMNS
        ),
        "farming_absent_from_x": FARMING_FEATURE_COLUMN not in dataset.X.columns,
        "hero_id_in_required_columns": "hero_id" in PLAYER_COMBAT_REQUIRED_COLUMNS,
        "position_in_required_columns": "position" in PLAYER_COMBAT_REQUIRED_COLUMNS,
        "hero_damage_in_required_columns": (
            "hero_damage" in PLAYER_COMBAT_REQUIRED_COLUMNS
        ),
        "kills_in_required_columns": "kills" in PLAYER_COMBAT_REQUIRED_COLUMNS,
        "causal_c_in_state_feature_columns": (
            COMBAT_CAUSAL_C_COLUMN in PLAYER_COMBAT_STATE_FEATURE_COLUMNS
        ),
        "causal_c_in_comparison_columns": (
            COMBAT_CAUSAL_C_COLUMN in PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS
        ),
        "state_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in SLICE18_STATE_COLUMNS
        ),
        "combat_in_feature_columns": COMBAT_FEATURE_COLUMN in FEATURE_COLUMNS,
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_names": [spec.name for spec in SLICE9_FROZEN_SPECS],
        "slice9_reference_unchanged": (
            SLICE9_REFERENCE_SPEC_NAME == ELO_BLOCK_SPEC_NAME
        ),
        "slice9_candidate_unchanged": (
            SLICE9_CANDIDATE_SPEC_NAME == "logistic_elo_plus_player_hero"
        ),
        "slice9_candidate_excludes_combat": COMBAT_FEATURE_COLUMN
        not in SLICE9_CANDIDATE_SPEC.feature_columns,
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "identical_oos_match_ids": True,
        "n_oos_match_ids": len(np.unique(ref_ids)),
        "population_matches_expected": (len(dataset) == FROZEN_DEVELOPMENT_MATCH_COUNT),
        "oos_count_matches_frozen_frame": (
            len(paired) == FROZEN_DEVELOPMENT_OOS_MATCH_COUNT
        ),
        "n_holdout_excluded": assembly.n_holdout_excluded,
        "n_missing_combat_join": assembly.n_missing_combat_join,
        "team_aggregation": "arithmetic_five_player_side_mean",
        "team_aggregation_n_players": COMBAT_ROSTER_SIDE_SIZE,
        "orientation": "radiant_minus_dire",
        "cold_start": 0.0,
        "walk_forward_config_n_blocks": DEFAULT_WALK_FORWARD_CONFIG.n_blocks,
    }


def run_slice20_player_combat_benchmark(
    store: FeatureDuckDBConnection,
    *,
    config: WalkForwardConfig | None = None,
    elo_config: EloConfig | None = None,
    development_end: datetime | None = None,
) -> Slice20BenchmarkReport:
    """Walk-forward paired test of frozen combat vs logistic Elo."""
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    assembly = build_slice20_model_ready_dataset(
        store, elo_config=elo_config, development_end=end
    )
    walk_forward = run_post_draft_walk_forward(
        assembly.dataset,
        config=resolved,
        specs=SLICE20_FROZEN_SPECS,
    )
    paired = _attach_features(
        _paired_oos(walk_forward), assembly.dataset, assembly.farming_by_match
    )
    coefficients = _fold_coefficients(walk_forward.folds, walk_forward.selected_C)
    fold_table = _fold_table(walk_forward, paired, coefficients)
    pooled = _pooled_row(paired)
    bootstrap = _bootstrap_delta(
        (paired["candidate_log_loss"] - paired["reference_log_loss"]).to_numpy(
            dtype=float
        )
    )
    consistency = _fold_sign_consistency(fold_table)
    combat_elo_pearson = _pearson(
        paired[COMBAT_FEATURE_COLUMN], paired[TEAM_ELO_DELTA_COLUMN]
    )
    combat_elo_spearman = _spearman(
        paired[COMBAT_FEATURE_COLUMN], paired[TEAM_ELO_DELTA_COLUMN]
    )
    combat_farming_pearson = _pearson(
        paired[COMBAT_FEATURE_COLUMN], paired[FARMING_FEATURE_COLUMN]
    )
    combat_farming_spearman = _spearman(
        paired[COMBAT_FEATURE_COLUMN], paired[FARMING_FEATURE_COLUMN]
    )
    combat_distribution = _distribution(
        paired[COMBAT_FEATURE_COLUMN], column=COMBAT_FEATURE_COLUMN
    )
    magnitude_buckets, edges = _magnitude_buckets(paired)
    elo_by_bucket = _elo_by_combat_bucket(paired, edges)
    directional_buckets = _directional_buckets(paired)
    movement = _prediction_movement(paired["prediction_delta"].to_numpy(dtype=float))
    calibration = _calibration(paired)
    classification, rationale = classify_slice20(
        pooled_delta=float(pooled.iloc[0]["paired_delta_log_loss"]),
        ci_low=float(bootstrap["ci95_low"]),
        ci_high=float(bootstrap["ci95_high"]),
        frac_delta_negative=float(bootstrap["frac_delta_negative"]),
        n_folds_delta_negative=int(consistency["n_folds_delta_negative"]),
        n_folds_delta_positive=int(consistency["n_folds_delta_positive"]),
        n_folds=int(consistency["n_folds"]),
        coefficient_sign_stable=bool(consistency["coefficient_sign_stable"]),
        reference_brier=float(pooled.iloc[0]["reference_brier"]),
        candidate_brier=float(pooled.iloc[0]["candidate_brier"]),
        reference_ece=float(pooled.iloc[0]["reference_ece"]),
        candidate_ece=float(pooled.iloc[0]["candidate_ece"]),
        mean_abs_prediction_delta=float(movement["mean_abs"]),
    )
    slice16_comparison = _slice16_qualitative_comparison(
        pooled_delta=float(pooled.iloc[0]["paired_delta_log_loss"]),
        ci_low=float(bootstrap["ci95_low"]),
        ci_high=float(bootstrap["ci95_high"]),
        n_folds_delta_negative=int(consistency["n_folds_delta_negative"]),
        n_folds=int(consistency["n_folds"]),
        coefficient_sign_stable=bool(consistency["coefficient_sign_stable"]),
        n_folds_coefficient_positive=int(consistency["n_folds_coefficient_positive"]),
        mean_abs_prediction_delta=float(movement["mean_abs"]),
        reference_ece=float(pooled.iloc[0]["reference_ece"]),
        candidate_ece=float(pooled.iloc[0]["candidate_ece"]),
        classification=classification,
    )
    integrity = _integrity(store, assembly, paired)
    integrity["abs_combat_bucket_edges"] = [float(edge) for edge in edges]
    return Slice20BenchmarkReport(
        development_end=end,
        holdout_policy=HOLDOUT_POLICY,
        n_development_matches=len(assembly.dataset),
        n_holdout_excluded=assembly.n_holdout_excluded,
        n_oos=len(paired),
        frozen_combat_k=FROZEN_COMBAT_SHRINKAGE_K,
        frozen_farming_k=FROZEN_SHRINKAGE_K,
        assembly=assembly,
        walk_forward=walk_forward,
        fold_table=fold_table,
        pooled=pooled,
        bootstrap=bootstrap,
        coefficients=coefficients,
        fold_sign_consistency=consistency,
        combat_elo_pearson=combat_elo_pearson,
        combat_elo_spearman=combat_elo_spearman,
        combat_farming_pearson=combat_farming_pearson,
        combat_farming_spearman=combat_farming_spearman,
        combat_distribution=combat_distribution,
        elo_by_combat_bucket=elo_by_bucket,
        prediction_movement=movement,
        magnitude_buckets=magnitude_buckets,
        directional_buckets=directional_buckets,
        calibration=calibration,
        slice16_comparison=slice16_comparison,
        integrity=integrity,
        classification=classification,
        classification_rationale=rationale,
    )


def slice20_report_to_jsonable(report: Slice20BenchmarkReport) -> dict[str, object]:
    """JSON-safe dump of the Slice 20 walk-forward report."""
    return {
        "development_end": report.development_end.isoformat(),
        "holdout_policy": report.holdout_policy,
        "n_development_matches": report.n_development_matches,
        "n_holdout_excluded": report.n_holdout_excluded,
        "n_oos": report.n_oos,
        "frozen_combat_k": report.frozen_combat_k,
        "frozen_farming_k": report.frozen_farming_k,
        "fold_table": _jsonable_value(report.fold_table),
        "pooled": _jsonable_value(report.pooled),
        "bootstrap": _jsonable_value(report.bootstrap),
        "coefficients": _jsonable_value(report.coefficients),
        "fold_sign_consistency": _jsonable_value(report.fold_sign_consistency),
        "combat_elo_pearson": _jsonable_value(report.combat_elo_pearson),
        "combat_elo_spearman": _jsonable_value(report.combat_elo_spearman),
        "combat_farming_pearson": _jsonable_value(report.combat_farming_pearson),
        "combat_farming_spearman": _jsonable_value(report.combat_farming_spearman),
        "combat_distribution": _jsonable_value(report.combat_distribution),
        "elo_by_combat_bucket": _jsonable_value(report.elo_by_combat_bucket),
        "prediction_movement": _jsonable_value(report.prediction_movement),
        "magnitude_buckets": _jsonable_value(report.magnitude_buckets),
        "directional_buckets": _jsonable_value(report.directional_buckets),
        "calibration": {
            "reference_brier": report.calibration["reference_brier"],
            "candidate_brier": report.calibration["candidate_brier"],
            "reference_ece": report.calibration["reference_ece"],
            "candidate_ece": report.calibration["candidate_ece"],
            "reference_bins": _jsonable_value(report.calibration["reference_bins"]),
            "candidate_bins": _jsonable_value(report.calibration["candidate_bins"]),
            "intercept_slope_available": report.calibration[
                "intercept_slope_available"
            ],
        },
        "slice16_comparison": _jsonable_value(report.slice16_comparison),
        "integrity": _jsonable_value(report.integrity),
        "classification": report.classification,
        "classification_rationale": report.classification_rationale,
        "n_missing_combat_join": report.assembly.n_missing_combat_join,
        "n_combat_comparison_matches": (report.assembly.n_combat_comparison_matches),
    }
