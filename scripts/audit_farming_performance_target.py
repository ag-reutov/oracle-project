"""CLI for Slice 13 farming-performance target diagnostics.

Target refinement and falsification only. Does not persist a player
rating, does not add columns to FEATURE_COLUMNS, does not modify Elo,
and does not call STRATZ.

Usage:
    uv run python scripts/audit_farming_performance_target.py
    uv run python scripts/audit_farming_performance_target.py \\
        --output data/interim/slice13_farming_performance_target.json
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
    run_farming_performance_target_diagnostics,
    slice13_report_to_jsonable,
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
            "Development-only Slice 13 farming-performance target diagnostics. "
            "Does not evaluate TI 2026 and does not train a win model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice13_farming_performance_target.json"),
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
        report = run_farming_performance_target_diagnostics(store)

    print("=== Slice 13 farming-performance target diagnostics ===")
    print(
        "Target refinement only. Candidate A is the unchanged Slice 12 "
        "position-standardized last-hit rate. Candidates B/C/D residualize "
        "duration and hero. "
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
    _print_table("Coverage", report.coverage)
    _print_table("Formulas", report.formulas)
    _print_table("Duration adjustment", report.duration_adjustment)
    _print_table(
        "Duration non-linearity (after linear residual)",
        report.duration_nonlinearity,
    )
    _print_table("Residual distribution", report.residual_distribution)
    _print_table("Hero adjustment", report.hero_adjustment)
    _print_table("Hero / hero×position pooling", report.hero_pooling)
    _print_table("Hero sample sizes (top 40)", report.hero_sample_sizes.head(40))
    _print_table(
        "Candidate means by position after adjustment",
        report.candidate_position_means,
    )
    _print_table("Candidate comparison", report.candidate_comparison)
    _print_table("Chronological repeatability", report.repeatability)
    _print_table("First-half vs second-half", report.first_half_second_half)
    _print_table("Team-switcher falsification", report.team_switcher)
    _print_table("Within-team centered persistence", report.within_team_centered)
    _print_table(
        "Hero-excluded vs ordinary repeatability",
        report.hero_excluded_repeatability,
    )
    _print_table("Winner vs loser repeatability", report.winner_loser)
    _print_table("Patch / version stability", report.patch_stability)
    _print_table("Within-version repeatability", report.patch_repeatability)
    _print_table("Cross-version repeatability", report.cross_version_repeatability)
    _print_table("Falsification summary", report.falsification)
    _print_table("Classification", report.classification)
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        print(f"{key}: {value}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice13_report_to_jsonable(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
