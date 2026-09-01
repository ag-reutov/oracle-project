"""CLI for the predefined post-draft block ablation.

Compares six logistic specs on one chronological split:

    Elo only
    Elo + Player × Hero
    Elo + Team × Hero
    Elo + Hero Meta
    Elo + Player × Hero + Team × Hero
    Elo + all three

Uses the same dataset assembly, split, TRAIN-only preprocessing, and
per-spec validation ``C`` selection as ``run_post_draft_benchmark``.

Usage:
    uv run python scripts/run_post_draft_block_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from dota_predictor.features import (
    connect,
    load_feature_store_config,
    load_reference_store_config,
    register_reference_views,
)
from dota_predictor.training import (
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    build_post_draft_model_ready_dataset,
    chronological_split,
    run_post_draft_block_ablation,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


def _metrics_row(
    name: str, label: str, n_features: int, selected_c: float, metrics: object
) -> dict[str, object]:
    return {
        "model": name,
        "label": label,
        "n_features": n_features,
        "C": selected_c,
        "log_loss": metrics.log_loss,
        "brier_score": metrics.brier_score,
        "accuracy_at_0.5": metrics.accuracy_at_0_5,
        "roc_auc": metrics.roc_auc,
        "ece": metrics.expected_calibration_error,
        "n": metrics.n_samples,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_project_env(root)
    config = load_feature_store_config(root=root)
    if not config.matches_path.is_file():
        print(f"Canonical matches file not found: {config.matches_path}", file=sys.stderr)
        return 1

    reference_config = load_reference_store_config(root=root)
    catalog_available = (
        reference_config.heroes_path.is_file()
        and reference_config.game_versions_path.is_file()
    )

    with connect(config) as store:
        if catalog_available:
            register_reference_views(store, reference_config)
        dataset = build_post_draft_model_ready_dataset(store)

    split = chronological_split(dataset)
    report = run_post_draft_block_ablation(split, include_test_evaluation=True)
    labels = {spec.name: spec.label for spec in POST_DRAFT_BLOCK_ABLATION_SPECS}

    print("=== Post-draft block ablation ===")
    print(
        "feature blocks: Player × Hero / Team × Hero / Hero Meta "
        "(full metric groups, no correlation subset)"
    )
    print(
        "specs: "
        + " | ".join(
            f"{spec.label} ({len(spec.feature_columns)})"
            for spec in POST_DRAFT_BLOCK_ABLATION_SPECS
        )
    )
    print(
        f"rows: train={len(split.train)}  "
        f"validation={len(split.validation)}  "
        f"test={len(split.test)}  "
        f"total={len(dataset)}"
    )
    print(
        f"split boundaries: train_end={split.boundaries.train_end}  "
        f"validation_end={split.boundaries.validation_end}"
    )
    print(
        "preprocessing: train-only median impute + missingness indicators + "
        "StandardScaler (same spec as post-draft benchmark)"
    )
    print("regularization: C selected independently per spec on validation")
    print()
    print("Regularization comparison (validation log loss):")
    print(report.regularization_comparison.to_string(index=False))
    print()
    print("Validation metrics:")
    val_rows = [
        _metrics_row(
            name,
            labels[name],
            report.n_features[name],
            report.selected_C[name],
            ev.metrics,
        )
        for name, ev in report.validation_evaluations.items()
    ]
    print(pd.DataFrame(val_rows).to_string(index=False))
    print()
    print("TEST metrics (frozen specs, single evaluation):")
    test_rows = [
        _metrics_row(
            name,
            labels[name],
            report.n_features[name],
            report.selected_C[name],
            ev.metrics,
        )
        for name, ev in report.test_evaluations.items()
    ]
    print(pd.DataFrame(test_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
