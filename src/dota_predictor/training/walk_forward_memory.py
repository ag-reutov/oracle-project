"""Walk-forward diagnostic comparing training-memory policies.

Reuses the existing expanding-window OOS fold boundaries. Each policy
fits the six predefined Elo + draft blocks on a restricted past and
scores the same evaluation matches. Incremental draft value is always
paired against Elo trained under the *same* memory policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dota_predictor.training.dataset import ModelReadyDataset
from dota_predictor.training.evaluation import (
    _fit_logistic,
    _select_regularization,
    evaluate_predictor,
)
from dota_predictor.training.feature_sets import POST_DRAFT_BLOCK_ABLATION_SPECS
from dota_predictor.training.logistic_model import (
    LogisticRegressionConfig,
    standardized_coefficients,
)
from dota_predictor.training.memory_policy import (
    MEMORY_POLICIES,
    MemoryRestrictedFold,
    restrict_fold_to_memory,
)
from dota_predictor.training.metrics import evaluate_probabilities, per_sample_log_loss
from dota_predictor.training.preprocessing import (
    MISSINGNESS_INDICATOR_SUFFIX,
    PreprocessingSpec,
)
from dota_predictor.training.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG,
    ELO_BLOCK_SPEC_NAME,
    WalkForwardConfig,
    _paired_stats,
    resolve_walk_forward_folds,
)

__all__ = [
    "COEFFICIENT_SPECS",
    "WalkForwardMemoryReport",
    "run_walk_forward_memory_diagnostics",
]

COEFFICIENT_SPECS: tuple[str, ...] = (
    "logistic_elo_plus_player_hero",
    "logistic_elo_plus_hero_meta",
)


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    n = values.size
    if n == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan")
    return mean, float(values.std(ddof=1))


@dataclass
class WalkForwardMemoryReport:
    """Memory-policy walk-forward diagnostics on one post-draft matrix."""

    config: WalkForwardConfig
    policies: tuple[str, ...]
    coverage: pd.DataFrame
    selected_C: pd.DataFrame
    fold_metrics: pd.DataFrame
    pooled_metrics: pd.DataFrame
    fold_stability: pd.DataFrame
    version_breakdown: pd.DataFrame
    coefficients: pd.DataFrame
    coefficient_summary: pd.DataFrame
    elo_baselines: pd.DataFrame
    oos_predictions: pd.DataFrame


def _try_restrict(
    dataset: ModelReadyDataset,
    fold,
    *,
    policy: str,
    train_fraction_of_past: float,
) -> MemoryRestrictedFold:
    return restrict_fold_to_memory(
        dataset,
        fold,
        policy=policy,
        train_fraction_of_past=train_fraction_of_past,
    )


def _pooled_for_policy(
    oos: pd.DataFrame, *, policy: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = {spec.name: spec for spec in POST_DRAFT_BLOCK_ABLATION_SPECS}
    subset_policy = oos.loc[oos["policy"] == policy]
    for spec_name, spec in specs.items():
        subset = subset_policy.loc[subset_policy["model"] == spec_name]
        if subset.empty:
            continue
        y = subset["y_true"]
        p = subset["p_spec"]
        p_elo = subset["p_elo"]
        paired = _paired_stats(y, p, p_elo)
        full = evaluate_probabilities(y, p)
        elo_metrics = evaluate_probabilities(y, p_elo)
        rows.append(
            {
                "policy": policy,
                "model": spec_name,
                "label": spec.label,
                "n_features": len(spec.feature_columns),
                "n": int(paired["n"]),
                "n_folds": int(subset["fold_id"].nunique()),
                "log_loss": full.log_loss,
                "elo_log_loss": elo_metrics.log_loss,
                "mean_delta_vs_elo": paired["mean_delta_vs_elo"],
                "se_delta_vs_elo": paired["se_delta_vs_elo"],
                "roc_auc": full.roc_auc,
                "elo_roc_auc": elo_metrics.roc_auc,
                "auc_delta_vs_elo": full.roc_auc - elo_metrics.roc_auc,
            }
        )
    return rows


def _version_for_policy(
    oos: pd.DataFrame, *, policy: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = {spec.name: spec for spec in POST_DRAFT_BLOCK_ABLATION_SPECS}
    subset_policy = oos.loc[oos["policy"] == policy]
    versions = sorted(
        v for v in subset_policy["game_version_id"].dropna().unique().tolist()
    )
    for version in versions:
        version_mask = subset_policy["game_version_id"] == version
        for spec_name, spec in specs.items():
            subset = subset_policy.loc[
                version_mask & (subset_policy["model"] == spec_name)
            ]
            if subset.empty:
                continue
            paired = _paired_stats(
                subset["y_true"], subset["p_spec"], subset["p_elo"]
            )
            rows.append(
                {
                    "policy": policy,
                    "game_version_id": version,
                    "model": spec_name,
                    "label": spec.label,
                    "n": int(paired["n"]),
                    "log_loss": paired["log_loss"],
                    "elo_log_loss": paired["elo_log_loss"],
                    "mean_delta_vs_elo": paired["mean_delta_vs_elo"],
                    "se_delta_vs_elo": paired["se_delta_vs_elo"],
                }
            )
    return rows


def _fold_stability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    draft = fold_metrics.loc[
        fold_metrics["model"] != ELO_BLOCK_SPEC_NAME
    ]
    for (policy, model), subset in draft.groupby(["policy", "model"], sort=False):
        deltas = subset["mean_delta_vs_elo"].to_numpy(dtype=float)
        mean_delta, std_delta = _mean_std(deltas)
        label = subset["label"].iloc[0]
        rows.append(
            {
                "policy": policy,
                "model": model,
                "label": label,
                "n_folds": len(subset),
                "mean_fold_delta_vs_elo": mean_delta,
                "std_fold_delta_vs_elo": std_delta,
                "n_folds_improved": int((deltas < 0).sum()),
                "n_folds_worsened": int((deltas > 0).sum()),
                "fold_deltas": tuple(float(value) for value in deltas),
            }
        )
    return pd.DataFrame(rows)


def _coefficient_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return coefficients
    working = coefficients.loc[
        ~coefficients["feature"].str.endswith(MISSINGNESS_INDICATOR_SUFFIX)
    ].copy()
    rows: list[dict[str, object]] = []
    grouped = working.groupby(
        ["policy", "model", "feature"], sort=False
    )
    for (policy, model, feature), subset in grouped:
        coef = subset["coefficient"].to_numpy(dtype=float)
        mean_coef, std_coef = _mean_std(coef)
        signs = np.sign(coef)
        signs = signs[signs != 0]
        if signs.size == 0:
            sign_consistency = float("nan")
            n_pos = 0
            n_neg = 0
        else:
            n_pos = int((signs > 0).sum())
            n_neg = int((signs < 0).sum())
            sign_consistency = max(n_pos, n_neg) / signs.size
        rows.append(
            {
                "policy": policy,
                "model": model,
                "label": subset["label"].iloc[0],
                "feature": feature,
                "n_folds": len(subset),
                "mean_coefficient": mean_coef,
                "std_coefficient": std_coef,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "sign_consistency": sign_consistency,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["policy", "model", "feature"], kind="stable"
    ).reset_index(drop=True)


def run_walk_forward_memory_diagnostics(
    dataset: ModelReadyDataset,
    *,
    config: WalkForwardConfig | None = None,
    policies: tuple[str, ...] = MEMORY_POLICIES,
) -> WalkForwardMemoryReport:
    """Fit each memory policy on the existing walk-forward OOS folds."""
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    specs = POST_DRAFT_BLOCK_ABLATION_SPECS
    spec_by_name = {spec.name: spec for spec in specs}
    if ELO_BLOCK_SPEC_NAME not in spec_by_name:
        raise ValueError(
            f"memory diagnostics require {ELO_BLOCK_SPEC_NAME!r} among "
            "the predefined block specs"
        )
    for policy in policies:
        if policy not in MEMORY_POLICIES:
            raise ValueError(f"unknown memory policy {policy!r}")

    preprocessing_spec = PreprocessingSpec()
    folds = resolve_walk_forward_folds(dataset, config=resolved)

    coverage_rows: list[dict[str, object]] = []
    selected_c_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []

    for policy in policies:
        for fold in folds:
            restricted = _try_restrict(
                dataset,
                fold,
                policy=policy,
                train_fraction_of_past=resolved.train_fraction_of_past,
            )
            coverage_rows.append(restricted.coverage)
            if restricted.skipped:
                continue
            assert restricted.train is not None
            assert restricted.validation is not None

            fitted = {}
            selected_c: dict[str, float] = {}
            fit_failed: str | None = None
            for spec in specs:
                try:
                    c, _reg = _select_regularization(
                        restricted.train,
                        restricted.validation,
                        spec.feature_columns,
                    )
                    model = _fit_logistic(
                        restricted.train,
                        spec.feature_columns,
                        config=LogisticRegressionConfig(
                            C=c, preprocessing=preprocessing_spec
                        ),
                    )
                except (ValueError, RuntimeError) as exc:
                    fit_failed = f"{spec.name}: {exc}"
                    break
                selected_c[spec.name] = c
                fitted[spec.name] = model

            if fit_failed is not None:
                coverage_rows[-1] = {
                    **restricted.coverage,
                    "skipped": True,
                    "skip_reason": (
                        "existing logistic machinery could not fit: "
                        f"{fit_failed}"
                    ),
                    "n_train": len(restricted.train),
                    "n_validation": len(restricted.validation),
                }
                continue

            for spec in specs:
                selected_c_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "policy": policy,
                        "model": spec.name,
                        "label": spec.label,
                        "C": selected_c[spec.name],
                        "n_train": len(restricted.train),
                        "n_validation": len(restricted.validation),
                        "n_test": len(restricted.test),
                    }
                )
                if spec.name in COEFFICIENT_SPECS:
                    coef = standardized_coefficients(fitted[spec.name]).copy()
                    coef.insert(0, "fold_id", fold.fold_id)
                    coef.insert(1, "policy", policy)
                    coef.insert(2, "model", spec.name)
                    coef.insert(3, "label", spec.label)
                    coefficient_frames.append(coef)

            evaluations = {
                spec.name: evaluate_predictor(
                    spec.name, restricted.test, fitted[spec.name]
                )
                for spec in specs
            }
            elo_preds = evaluations[ELO_BLOCK_SPEC_NAME].predictions
            elo_p = elo_preds.p_radiant_win.reset_index(drop=True)
            y = elo_preds.y_true.reset_index(drop=True)
            context = elo_preds.context.reset_index(drop=True)

            for spec in specs:
                evaluation = evaluations[spec.name]
                p = evaluation.predictions.p_radiant_win.reset_index(drop=True)
                paired = _paired_stats(y, p, elo_p)
                metrics = evaluation.metrics
                elo_metrics = evaluate_probabilities(y, elo_p)
                fold_metric_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "policy": policy,
                        "model": spec.name,
                        "label": spec.label,
                        "C": selected_c[spec.name],
                        "n_train": len(restricted.train),
                        "n_validation": len(restricted.validation),
                        "n_test": metrics.n_samples,
                        "log_loss": metrics.log_loss,
                        "elo_log_loss": elo_metrics.log_loss,
                        "mean_delta_vs_elo": paired["mean_delta_vs_elo"],
                        "se_delta_vs_elo": paired["se_delta_vs_elo"],
                        "roc_auc": metrics.roc_auc,
                        "elo_roc_auc": elo_metrics.roc_auc,
                        "auc_delta_vs_elo": metrics.roc_auc - elo_metrics.roc_auc,
                    }
                )
                spec_ll = per_sample_log_loss(y, p)
                elo_ll = per_sample_log_loss(y, elo_p)
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "fold_id": fold.fold_id,
                            "policy": policy,
                            "model": spec.name,
                            "label": spec.label,
                            "match_id": context["match_id"].to_numpy(),
                            "start_time": context["start_time"].to_numpy(),
                            "game_version_id": context["game_version_id"].to_numpy(),
                            "y_true": y.to_numpy(),
                            "p_spec": p.to_numpy(),
                            "p_elo": elo_p.to_numpy(),
                            "sample_log_loss": spec_ll,
                            "delta_vs_elo": spec_ll - elo_ll,
                        }
                    )
                )

    coverage = pd.DataFrame(coverage_rows)
    selected_C = pd.DataFrame(selected_c_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    oos = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    pooled_rows: list[dict[str, object]] = []
    version_rows: list[dict[str, object]] = []
    if not oos.empty:
        for policy in policies:
            pooled_rows.extend(_pooled_for_policy(oos, policy=policy))
            version_rows.extend(_version_for_policy(oos, policy=policy))
    pooled_metrics = pd.DataFrame(pooled_rows)
    version_breakdown = pd.DataFrame(version_rows)
    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame()
    )
    elo_baselines = pooled_metrics.loc[
        pooled_metrics["model"] == ELO_BLOCK_SPEC_NAME
    ].copy() if not pooled_metrics.empty else pd.DataFrame()

    return WalkForwardMemoryReport(
        config=resolved,
        policies=policies,
        coverage=coverage,
        selected_C=selected_C,
        fold_metrics=fold_metrics,
        pooled_metrics=pooled_metrics,
        fold_stability=_fold_stability(fold_metrics)
        if not fold_metrics.empty
        else pd.DataFrame(),
        version_breakdown=version_breakdown,
        coefficients=coefficients,
        coefficient_summary=_coefficient_summary(coefficients),
        elo_baselines=elo_baselines,
        oos_predictions=oos,
    )
