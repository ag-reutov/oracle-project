"""CLI for Step 4B baseline modeling on the current canonical dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from dota_predictor.features import build_pre_draft_snapshot, connect
from dota_predictor.training import (
    build_model_ready_dataset,
    chronological_split,
    run_step4b_benchmark,
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

    with connect() as store:
        dataset = build_model_ready_dataset(build_pre_draft_snapshot(store))
    split = chronological_split(dataset)
    report = run_step4b_benchmark(split, include_test_evaluation=True)

    print("=== Step 4B benchmark (provisional corpus) ===")
    print(f"selected regularization C: {report.selected_regularization_C}")
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
    print("Ablation (validation):")
    ablation_rows = [
        _metrics_row(name, ev.metrics)
        for name, ev in report.ablation_validation.items()
    ]
    print(pd.DataFrame(ablation_rows).to_string(index=False))
    print()
    print("Top standardized logistic coefficients (all features):")
    print(report.coefficients.head(15).to_string(index=False))
    print()
    print("Calibration bins (validation, logistic all features):")
    print(
        report.validation_evaluations[
            "logistic_regression_all_features"
        ].metrics.calibration_table.to_string(index=False)
    )
    if report.test_evaluations:
        print()
        print("TEST metrics (single frozen-spec evaluation):")
        test_rows = [
            _metrics_row(name, ev.metrics)
            for name, ev in report.test_evaluations.items()
        ]
        print(pd.DataFrame(test_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
