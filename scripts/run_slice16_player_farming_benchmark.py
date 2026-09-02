"""CLI for Slice 16: walk-forward evaluation of frozen player farming vs Elo.

Compares logistic_elo_only with logistic_elo_plus_player_farming on the
existing expanding-window development OOS folds. Does not redesign the
farming feature, does not score the frozen holdout, does not call STRATZ,
and does not change FEATURE_COLUMNS.

Usage:
    uv run python scripts/run_slice16_player_farming_benchmark.py
    uv run python scripts/run_slice16_player_farming_benchmark.py \\
        --output data/interim/slice16_player_farming_benchmark.json
"""

from __future__ import annotations

import argparse
import json
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
    FROZEN_DEVELOPMENT_END,
    FROZEN_SHRINKAGE_K,
    run_slice16_player_farming_benchmark,
    slice16_report_to_jsonable,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.8f}")
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 88)


def _print_table(title: str, frame: pd.DataFrame) -> None:
    print()
    print(f"--- {title} ---")
    if frame.empty:
        print("(empty)")
        return
    print(frame.to_string(index=False))


def _print_mapping(title: str, payload: dict[str, object]) -> None:
    print()
    print(f"--- {title} ---")
    for key, value in payload.items():
        print(f"{key}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Slice 16 walk-forward: frozen player farming vs logistic Elo. "
            "Development OOS only; does not score TI 2026 / frozen holdout."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice16_player_farming_benchmark.json"),
        help="JSON path under data/interim/ (gitignored).",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    load_project_env(root)
    config = load_feature_store_config(root=root)
    if not config.matches_path.is_file():
        print(
            f"Canonical matches file not found: {config.matches_path}",
            file=sys.stderr,
        )
        return 1

    reference_config = load_reference_store_config(root=root)
    with connect(config) as store:
        if (
            reference_config.heroes_path.is_file()
            and reference_config.game_versions_path.is_file()
        ):
            register_reference_views(store, reference_config)
        report = run_slice16_player_farming_benchmark(store)

    pooled = report.pooled.iloc[0]
    print("=== Slice 16: frozen player farming vs logistic Elo ===")
    print(
        "Reference logistic_elo_only vs candidate "
        "logistic_elo_plus_player_farming "
        f"(elo block + {report.integrity['candidate_columns'][-1]}). "
        "Frozen Slice 14 candidate B, k=5, five-player arithmetic mean, "
        "Radiant − Dire. Walk-forward folds are the existing expanding "
        "windows on the development frame. Frozen holdout is not scored."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"OOS matches={report.n_oos}  "
        f"holdout matches excluded={report.n_holdout_excluded}  "
        f"frozen_k={report.frozen_k:g}  "
        f"n_blocks={DEFAULT_WALK_FORWARD_CONFIG.n_blocks}  "
        f"development end={FROZEN_DEVELOPMENT_END.isoformat()}  "
        f"FROZEN_SHRINKAGE_K={FROZEN_SHRINKAGE_K:g}"
    )
    print(f"holdout policy: {report.holdout_policy}")
    _print_table("Per-fold benchmark", report.fold_table)
    _print_table("Pooled OOS", report.pooled)
    _print_mapping("Paired bootstrap (OOS matches)", report.bootstrap)
    _print_table("Fold coefficients (standardized)", report.coefficients)
    _print_mapping("Fold sign consistency", report.fold_sign_consistency)
    print()
    print(
        "Pearson corr(mean_farming_shrunk_b_diff, team_elo_delta) = "
        f"{report.farming_elo_correlation:.8f}"
    )
    _print_table("Farming-diff distribution (OOS)", report.farming_distribution)
    _print_table(
        "Mean Elo diff by |farming| quantile bucket",
        report.elo_by_farming_bucket,
    )
    _print_mapping("Prediction movement |p_candidate - p_reference|", report.prediction_movement)
    _print_table(
        "Paired log loss by |farming diff| quantile bucket",
        report.magnitude_buckets,
    )
    print()
    print("--- Calibration (existing framework; no intercept/slope) ---")
    print(
        f"reference Brier={report.calibration['reference_brier']:.8f}  "
        f"candidate Brier={report.calibration['candidate_brier']:.8f}"
    )
    print(
        f"reference ECE={report.calibration['reference_ece']:.8f}  "
        f"candidate ECE={report.calibration['candidate_ece']:.8f}"
    )
    _print_table("Reference calibration bins", report.calibration["reference_bins"])
    _print_table("Candidate calibration bins", report.calibration["candidate_bins"])
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        print(f"{key}: {value}")
    print()
    print(
        f"pooled Δ log loss={pooled['paired_delta_log_loss']:.8f}  "
        f"bootstrap 95% CI=({report.bootstrap['ci95_low']:.8f}, "
        f"{report.bootstrap['ci95_high']:.8f})  "
        f"frac Δ<0={report.bootstrap['frac_delta_negative']:.4f}"
    )
    print(f"classification: {report.classification}")
    print(f"rationale: {report.classification_rationale}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice16_report_to_jsonable(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
