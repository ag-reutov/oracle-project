"""CLI for walk-forward temporal robustness of the six post-draft blocks.

Expanding windows on the same Elo + draft-comparison matrix as the
holdout block ablation. Each fold selects ``C`` on a trailing
validation slice of the past and evaluates once on the next time
block. Reports paired log-loss deltas vs logistic Elo (negative =
spec better) and OOS diagnostics by game version.

Usage:
    uv run python scripts/run_post_draft_walk_forward.py
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
    DEFAULT_WALK_FORWARD_CONFIG,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    build_post_draft_model_ready_dataset,
    run_post_draft_walk_forward,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 200)


def _version_names(root: Path) -> dict[int, str]:
    reference_config = load_reference_store_config(root=root)
    path = reference_config.game_versions_path
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path, columns=["game_version_id", "name"])
    return {
        int(version_id): str(name)
        for version_id, name in zip(frame["game_version_id"], frame["name"])
    }


def _attach_version_name(frame: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    if "game_version_id" not in frame.columns:
        return frame
    labeled = frame.copy()
    labeled.insert(
        1,
        "game_version",
        labeled["game_version_id"].map(
            lambda value: names.get(int(value), str(value)) if pd.notna(value) else ""
        ),
    )
    return labeled


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

    wf_config = DEFAULT_WALK_FORWARD_CONFIG
    report = run_post_draft_walk_forward(dataset, config=wf_config)
    names = _version_names(root)
    n_oos = int(
        report.pooled_metrics.loc[
            report.pooled_metrics["model"] == "logistic_elo_only", "n"
        ].iloc[0]
    )

    print("=== Post-draft walk-forward block robustness ===")
    print(
        "specs: "
        + " | ".join(
            f"{spec.label} ({len(spec.feature_columns)})"
            for spec in POST_DRAFT_BLOCK_ABLATION_SPECS
        )
    )
    print(
        f"blocks={wf_config.n_blocks}  folds={len(report.folds)}  "
        f"train_fraction_of_past={wf_config.train_fraction_of_past:.4f}  "
        f"rows={len(dataset)}  oos={n_oos}"
    )
    print(
        "preprocessing: train-only median impute + missingness indicators + "
        "StandardScaler (same spec as holdout block ablation)"
    )
    print(
        "regularization: C selected independently per spec on each fold's "
        "trailing validation slice; test block is frozen"
    )
    print("paired delta = spec log loss − Elo log loss (negative = spec better)")
    print()
    print("Fold windows:")
    window_rows = []
    for fold in report.folds:
        window_rows.append(
            {
                "fold": fold.fold_id,
                "n_train": len(fold.train),
                "n_validation": len(fold.validation),
                "n_test": len(fold.test),
                "train_end": fold.train_end,
                "validation_end": fold.validation_end,
                "test_end": fold.test_end,
            }
        )
    print(pd.DataFrame(window_rows).to_string(index=False))
    print()
    print("Selected C by fold:")
    print(
        report.selected_C[["fold_id", "label", "C", "n_train", "n_validation", "n_test"]]
        .to_string(index=False)
    )
    print()
    print("Per-fold test metrics and paired Δ vs Elo:")
    fold_view = report.fold_metrics[
        [
            "fold_id",
            "label",
            "C",
            "n_test",
            "log_loss",
            "mean_delta_vs_elo",
            "se_delta_vs_elo",
            "frac_better_than_elo",
            "roc_auc",
            "brier_score",
        ]
    ]
    print(fold_view.to_string(index=False))
    print()
    print("Pooled OOS (each match after the first block, once):")
    print(report.pooled_metrics.to_string(index=False))
    print()
    print("OOS match counts by fold × game version:")
    print(_attach_version_name(report.version_fold_counts, names).to_string(index=False))
    print()
    print("Paired Δ vs Elo by game version (pooled OOS):")
    version_view = report.version_breakdown[
        [
            "game_version_id",
            "label",
            "n",
            "log_loss",
            "elo_log_loss",
            "mean_delta_vs_elo",
            "se_delta_vs_elo",
            "frac_better_than_elo",
        ]
    ]
    print(_attach_version_name(version_view, names).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
