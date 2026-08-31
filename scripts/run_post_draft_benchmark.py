"""CLI for the first post-draft benchmark: Elo vs Elo + draft comparison.

Uses chronological train/validation/test splits. Includes every
Radiant − Dire draft-comparison metric; none are dropped from
descriptive correlations. Does not change PRE_DRAFT `FEATURE_COLUMNS`.

Usage:
    uv run python scripts/run_post_draft_benchmark.py
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
    DRAFT_COMPARISON_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_DRAFT_COMPARISON_COLUMNS,
    build_post_draft_model_ready_dataset,
    chronological_split,
    run_post_draft_benchmark,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


def _metrics_row(name: str, metrics: object) -> dict[str, object]:
    return {
        "model": name,
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
    report = run_post_draft_benchmark(split, include_test_evaluation=True)

    print("=== Post-draft benchmark: Elo vs Elo + draft comparison ===")
    print(
        "feature sets: "
        f"elo_only={len(ELO_ONLY_FEATURE_COLUMNS)}  "
        f"draft_comparison={len(DRAFT_COMPARISON_FEATURE_COLUMNS)} "
        f"(full metric set, no correlation subset)  "
        f"elo+draft={len(ELO_PLUS_DRAFT_COMPARISON_COLUMNS)}"
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
        f"selected C: logistic_elo_only={report.elo_logistic_C}  "
        f"logistic_elo_plus_draft_comparison={report.elo_plus_draft_C}"
    )
    print(
        "preprocessing: train-only median impute + missingness indicators + "
        "StandardScaler (same spec as Step 4B)"
    )
    print()
    print("Regularization comparison (validation log loss):")
    print(report.regularization_comparison.to_string(index=False))
    print()
    print("Validation metrics:")
    val_rows = [
        _metrics_row(name, ev.metrics)
        for name, ev in report.validation_evaluations.items()
    ]
    print(pd.DataFrame(val_rows).to_string(index=False))
    print()
    print("TEST metrics (frozen specs, single evaluation):")
    test_rows = [
        _metrics_row(name, ev.metrics)
        for name, ev in report.test_evaluations.items()
    ]
    print(pd.DataFrame(test_rows).to_string(index=False))
    print()
    print("Top standardized logistic coefficients (Elo + all draft diffs):")
    print(report.coefficients.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
