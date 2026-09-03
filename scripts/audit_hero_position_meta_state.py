"""CLI for Slice 24 current-meta Hero × Position state diagnostics.

Research / state only. Does not add columns to FEATURE_COLUMNS, does not
train a win model, does not construct player×hero fit, does not revive
Slice 23 compatibility, and does not overwrite Slice 21/22.

Usage:
    uv run python scripts/audit_hero_position_meta_state.py
    uv run python scripts/audit_hero_position_meta_state.py \\
        --output data/interim/slice24_hero_position_meta_state.json
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
    FROZEN_DEVELOPMENT_END,
    run_hero_position_meta_diagnostics,
    slice24_report_to_jsonable,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 88)


def _print_table(title: str, frame: pd.DataFrame) -> None:
    print()
    print(f"--- {title} ---")
    if frame.empty:
        print("(empty)")
        return
    print(frame.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only Slice 24 current-meta H×P diagnostics. "
            "Does not evaluate TI 2026 and does not train a win model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice24_hero_position_meta_state.json"),
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
        report = run_hero_position_meta_diagnostics(store)

    print("=== Slice 24 current-meta Hero × Position state diagnostics ===")
    print(
        "Environmental H×P × time state only. Usage, Elo-adjusted residuals, "
        "and requirement drift are investigated separately. History is "
        "start_time < T, explicit positions 1–5, same-timestamp blind. "
        "Current-match results never enter state. Shrinkage, if any, is "
        "chosen by predicting the next H×P residual, not match outcomes. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        "Holdout / TI 2026 matches are excluded from every summary."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"player rows={report.n_development_player_rows}  "
        f"holdout rows excluded={report.n_holdout_excluded}  "
        f"tune_end={report.tune_end.isoformat()}  "
        f"recent_window={report.selected_recent_window}  "
        f"residual_k={report.residual_shrinkage_k:g}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    print()
    print(f"recent-window justification: {report.selected_recent_window_justification}")
    print(f"residual k justification: {report.residual_shrinkage_justification}")
    _print_table("Tune / validation split", report.split)
    _print_table("Coverage", report.coverage)
    _print_table("Cold start", report.cold_start)
    _print_table("History-size buckets", report.history_size)
    _print_table("Long-run vs recent/version", report.estimator_comparison)
    _print_table("Elo-residual persistence", report.residual_persistence)
    _print_table("Usage block persistence", report.usage_persistence)
    _print_table("Version transfer", report.version_transfer)
    _print_table("Same-version persistence", report.same_version_persistence)
    _print_table("Requirement drift", report.requirement_drift)
    _print_table("Residual shrinkage (tune)", report.residual_shrinkage_tune)
    _print_table(
        "Residual shrinkage (validation)", report.residual_shrinkage_validation
    )
    _print_table("Regression to the mean", report.regression_to_mean)
    _print_table("Sample-size dependence", report.sample_size)
    _print_table("Classification", report.classification)
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        if key in {"usage_gate", "residual_gate", "drift_gate"}:
            print(f"{key}:")
            for inner_key, inner_value in value.items():
                print(f"  {inner_key}: {inner_value}")
        else:
            print(f"{key}: {value}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice24_report_to_jsonable(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
