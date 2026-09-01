"""CLI for walk-forward training-memory diagnostics.

Reuses existing OOS fold boundaries. Each memory policy fits the six
predefined Elo + draft blocks on a restricted past and scores the same
evaluation matches. Incremental Δ is always vs Elo under the same
policy.

Usage:
    uv run python scripts/run_walk_forward_memory.py
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
    run_walk_forward_memory_diagnostics,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 72)


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


def _fmt_versions(values: object, names: dict[int, str]) -> str:
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return ""
    if isinstance(values, str):
        return values
    try:
        ids = list(values)
    except TypeError:
        return str(values)
    labels = []
    for version_id in ids:
        if pd.isna(version_id):
            continue
        key = int(version_id)
        labels.append(names.get(key, str(key)))
    return ", ".join(labels)


def _attach_version_name(frame: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    if frame.empty or "game_version_id" not in frame.columns:
        return frame
    labeled = frame.copy()
    labeled.insert(
        1,
        "game_version",
        labeled["game_version_id"].map(
            lambda value: names.get(int(value), str(value))
            if pd.notna(value)
            else ""
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
    report = run_walk_forward_memory_diagnostics(dataset, config=wf_config)
    names = _version_names(root)

    print("=== Walk-forward training-memory diagnostics ===")
    print(
        "OOS fold boundaries are the existing expanding walk-forward "
        "windows. Memory policies restrict only train/validation. "
        "Paired Δ = spec log loss − same-policy Elo (negative = spec better)."
    )
    print(
        "CURRENT_PLUS_PREVIOUS_VERSION current = last past game_version_id "
        "(not peeked from the evaluation window). Previous = immediately "
        "preceding represented version in past first-seen order."
    )
    print(
        "specs: "
        + " | ".join(
            f"{spec.label} ({len(spec.feature_columns)})"
            for spec in POST_DRAFT_BLOCK_ABLATION_SPECS
        )
    )
    print(
        f"policies: {' | '.join(report.policies)}  "
        f"blocks={wf_config.n_blocks}  rows={len(dataset)}"
    )
    print()

    coverage = report.coverage.copy()
    coverage["train_game_versions"] = coverage["train_game_versions"].map(
        lambda value: _fmt_versions(value, names)
    )
    coverage["validation_game_versions"] = coverage["validation_game_versions"].map(
        lambda value: _fmt_versions(value, names)
    )
    coverage["evaluation_game_versions"] = coverage["evaluation_game_versions"].map(
        lambda value: _fmt_versions(value, names)
    )
    coverage["current_version"] = coverage["current_version_id"].map(
        lambda value: names.get(int(value), str(value)) if pd.notna(value) else ""
    )
    coverage["previous_version"] = coverage["previous_version_id"].map(
        lambda value: names.get(int(value), str(value)) if pd.notna(value) else ""
    )
    print("--- 1. Training N by fold and memory policy ---")
    print(
        coverage[
            [
                "fold_id",
                "policy",
                "skipped",
                "skip_reason",
                "n_train",
                "n_validation",
                "n_evaluation",
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "evaluation_start",
                "evaluation_end",
                "train_game_versions",
                "evaluation_game_versions",
                "current_version",
                "previous_version",
            ]
        ].to_string(index=False)
    )
    skipped = coverage.loc[coverage["skipped"]]
    if skipped.empty:
        print("No fold/policy skipped.")
    else:
        print("SKIPPED fold/policy rows (memory window was not widened):")
        print(skipped[["fold_id", "policy", "skip_reason", "n_evaluation"]].to_string(index=False))
    print()
    print("Selected C by fold / policy / spec:")
    print(
        report.selected_C[
            ["fold_id", "policy", "label", "C", "n_train", "n_validation", "n_test"]
        ].to_string(index=False)
    )
    print()
    print("--- 6. Absolute Elo-only log loss by memory policy (pooled OOS) ---")
    print(report.elo_baselines.to_string(index=False))
    print()
    print("--- 2. Pooled OOS by memory policy × spec ---")
    print(
        "Δ is vs Elo trained under the SAME policy on the SAME OOS matches."
    )
    print(report.pooled_metrics.to_string(index=False))
    print()
    print("--- 3. Fold-level paired Δ vs same-policy Elo ---")
    print(report.fold_stability.to_string(index=False))
    print()
    print("Per-fold metrics:")
    print(
        report.fold_metrics[
            [
                "fold_id",
                "policy",
                "label",
                "C",
                "n_train",
                "n_test",
                "log_loss",
                "elo_log_loss",
                "mean_delta_vs_elo",
                "se_delta_vs_elo",
                "roc_auc",
                "auc_delta_vs_elo",
            ]
        ].to_string(index=False)
    )
    print()
    print("--- 4. Patch/version paired Δ by memory policy ---")
    print(_attach_version_name(report.version_breakdown, names).to_string(index=False))
    print()
    print("--- 5. Coefficient stability (Player × Hero, Hero Meta; no missingness indicators) ---")
    print(report.coefficient_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
