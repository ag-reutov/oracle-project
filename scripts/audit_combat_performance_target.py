"""CLI for Slice 17 combat-performance target diagnostics.

Target discovery only. Does not persist a player rating, does not add
columns to FEATURE_COLUMNS, does not modify frozen farming code, does
not call STRATZ, and does not train a win model.

Usage:
    uv run python scripts/audit_combat_performance_target.py
    uv run python scripts/audit_combat_performance_target.py \\
        --output data/interim/slice17_combat_performance_target.json
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
    run_combat_performance_target_diagnostics,
    slice17_report_to_jsonable,
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
            "Development-only Slice 17 combat-target diagnostics. "
            "Does not evaluate TI 2026 and does not train a win model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice17_combat_performance_target.json"),
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
        report = run_combat_performance_target_diagnostics(store)

    print("=== Slice 17 combat-performance target diagnostics ===")
    print(
        "Target discovery only. Candidates are POST_MATCH observations. "
        "Team-relative sums require a complete five-player vector. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        "Holdout / TI 2026 matches are excluded from every summary."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"player rows={report.n_development_player_rows}  "
        f"explicit position={report.n_explicit_position}  "
        f"missing position={report.n_missing_position}  "
        f"holdout rows excluded={report.n_holdout_excluded}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    _print_table("Landed field inventory", report.field_inventory)
    _print_table("Skipped candidates", report.skipped_candidates)
    _print_table("Formulas", report.formulas)
    _print_table("Candidate coverage", report.coverage)
    _print_table(
        "Distributions (explicit position)",
        report.distributions.loc[report.distributions["subset"] == "explicit_position"],
    )
    _print_table("Position dependence", report.position_dependence)
    _print_table("Duration dependence", report.duration_dependence)
    _print_table("Relationship with current result", report.result_relationship)
    _print_table("Winners vs losers", report.winner_loser)
    _print_table("Relationship with frozen farming B", report.farming_relationship)
    _print_table(
        "Farming B correlation within position", report.farming_within_position
    )
    _print_table("Split-half repeatability", report.split_half)
    _print_table("Split-half by position", report.split_half_by_position)
    _print_table("Consecutive-match persistence", report.consecutive_persistence)
    _print_table("Between/within player variance", report.variance_decomposition)
    _print_table("Candidate means by position", report.candidate_position_means)
    _print_table("Candidate comparison", report.candidate_comparison)
    _print_table("Classification", report.classification)
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        print(f"{key}: {value}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(slice17_report_to_jsonable(report), indent=2) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
