"""CLI for Slice 25 causal Player × Position hero-pool diagnostics.

Research / state only. Does not add columns to FEATURE_COLUMNS, does not
train a win model, does not implement Slice 26 assignment / flex logic,
and does not use Slice 23 fit or Slice 24 H×P outcome state.

Usage:
    uv run python scripts/audit_player_hero_pool_state.py
    uv run python scripts/audit_player_hero_pool_state.py \\
        --output data/interim/slice25_player_hero_pool.json
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
    run_player_hero_pool_diagnostics,
    slice25_report_to_jsonable,
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
            "Development-only Slice 25 Player × Position hero-pool "
            "diagnostics. Next-choice scoring, not match outcomes. "
            "Does not evaluate TI 2026."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice25_player_hero_pool.json"),
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
        report = run_player_hero_pool_diagnostics(store)

    print("=== Slice 25 causal Player × Position hero-pool state ===")
    print(
        "Availability / hero-pool identity only. Counts and empirical "
        "shares; no win rate, Elo residual, farming B, combat C, Slice 23 "
        "fit, or Slice 24 H×P outcome state. History is start_time < T, "
        "explicit positions 1–5, same-timestamp blind. Current hero and "
        "observed position do not update state. Scoring uses a fixed "
        f"mixture epsilon={report.scoring_mixture_epsilon:g} over a common "
        "causal candidate universe C_T; that wrapper is not part of frozen "
        "state. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        "Holdout / TI 2026 matches are excluded from every summary."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"player rows={report.n_development_player_rows}  "
        f"holdout rows excluded={report.n_holdout_excluded}  "
        f"tune_end={report.tune_end.isoformat()}  "
        f"hierarchical_k={report.selected_hierarchical_k:g}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    print()
    print(f"hierarchical k justification: {report.selected_hierarchical_k_justification}")
    _print_table("Tune / validation split", report.split)
    _print_table("Coverage", report.coverage)
    _print_table("Cold start", report.cold_start)
    _print_table("Next-choice (role-history rows)", report.next_choice)
    _print_table("Next-choice by position", report.next_choice_by_position)
    _print_table("Next-choice by history n(P,R)", report.next_choice_by_history)
    _print_table("Next-choice by cold-start category", report.next_choice_by_cold_start)
    _print_table("Role-gap diagnostic", report.role_gap)
    _print_table("Paired recency vs expanding", report.recency)
    _print_table("Recency questions", report.recency_questions)
    _print_table("Hierarchical", report.hierarchical)
    _print_table("Pool shape", report.pool_shape)
    _print_table("Cross-position (descriptive, Slice 26 later)", report.cross_position)
    _print_table("Primary expanding vs unconditioned (match CI)", report.primary_comparison)
    _print_table("last_5 support diagnostics", report.window_support)
    _print_table("Calibration of expanding empirical share", report.calibration)
    _print_table("Classification", report.classification)
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        print(f"{key}: {value}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice25_report_to_jsonable(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
