"""CLI for Slice 12 player-performance target diagnostics.

Target research only. Does not persist a player rating, does not add
columns to FEATURE_COLUMNS, and does not score or fetch TI 2026.

Usage:
    uv run python scripts/audit_player_performance_target.py
    uv run python scripts/audit_player_performance_target.py \\
        --output data/interim/slice12_player_performance_target.json
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
    run_player_performance_target_diagnostics,
    slice12_report_to_jsonable,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 72)


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
            "Development-only Slice 12 player-performance target diagnostics. "
            "Does not evaluate TI 2026 and does not train a win model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path (typically under data/interim/, gitignored).",
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
        report = run_player_performance_target_diagnostics(store)

    print("=== Slice 12 player-performance target diagnostics ===")
    print(
        "Target research only. Box scores are POST_MATCH observations. "
        "Position 1–5 adjustment does not impute missing position. "
        "Elo expected win is the existing pre-match team Elo probability. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        "TI 2026 / later matches are excluded from every summary."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"player rows={report.n_development_player_rows}  "
        f"missing position={report.n_missing_position}  "
        f"holdout rows excluded={report.n_holdout_excluded}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    _print_table("Coverage", report.coverage)
    _print_table("Raw box-score diagnostics", report.raw_diagnostics)
    _print_table("Role dependence (position 1–5 R²)", report.role_dependence)
    _print_table("Duration dependence", report.duration_dependence)
    _print_table("Outcome / Elo contamination", report.outcome_contamination)
    _print_table("Winners vs losers within position", report.winner_loser_by_position)
    _print_table(
        "Candidate means by position after adjustment",
        report.candidate_position_means,
    )
    _print_table("Hero sample sizes (top 40)", report.hero_sample_sizes.head(40))
    _print_table("Hero variance on candidates", report.hero_variance)
    _print_table(
        "Raw correlation matrix (explicit position)", report.correlation_matrix
    )
    _print_table(
        "Within-position residual correlation matrix",
        report.within_position_correlation_matrix,
    )
    _print_table("PCA of position-adjusted primitives", report.pca)
    _print_table("Candidate quality", report.candidate_quality)
    _print_table("Temporal repeatability", report.repeatability)
    _print_table("First-half vs second-half", report.first_half_second_half)
    _print_table("Patch / version stability", report.patch_stability)
    _print_table("Falsification", report.falsification)
    _print_table("Recommendations", report.recommendations)
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        print(f"{key}: {value}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(slice12_report_to_jsonable(report), indent=2) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
