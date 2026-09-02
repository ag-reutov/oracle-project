"""CLI for Slice 18 leakage-safe historical player combat state.

Causal candidate C + shrunk player history only. Does not persist a
player rating, does not add columns to FEATURE_COLUMNS, does not
modify Elo, farming k=5, or Slice 17's candidate definition, does not
call STRATZ, and does not train a win model.

Usage:
    uv run python scripts/audit_player_combat_state.py
    uv run python scripts/audit_player_combat_state.py \\
        --output data/interim/slice18_player_combat_state.json
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
    FROZEN_COMBAT_CANDIDATE,
    FROZEN_COMBAT_SHRINKAGE_K,
    FROZEN_DEVELOPMENT_END,
    FROZEN_SHRINKAGE_K,
    run_player_combat_state_diagnostics,
    slice18_report_to_jsonable,
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
            "Development-only Slice 18 player combat-state diagnostics. "
            "Does not evaluate TI 2026 and does not train a win model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice18_player_combat_state.json"),
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
        report = run_player_combat_state_diagnostics(store)

    print("=== Slice 18 leakage-safe historical player combat state ===")
    print(
        "Causal candidate C only. Position means use start_time < T. "
        "Player state uses the same player's prior C with "
        "start_time < M.start_time. Shrinkage is toward zero and is "
        "chosen by predicting the next causal C, not match outcomes. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        "Holdout / TI 2026 matches are excluded from every summary."
    )
    print(
        f"frozen candidate={FROZEN_COMBAT_CANDIDATE}  "
        f"development matches={report.n_development_matches}  "
        f"player rows={report.n_development_player_rows}  "
        f"holdout rows excluded={report.n_holdout_excluded}  "
        f"tune_end={report.tune_end.isoformat()}  "
        f"selected_k={report.selected_k:g}  "
        f"frozen_combat_k_constant={FROZEN_COMBAT_SHRINKAGE_K:g}  "
        f"farming_k={FROZEN_SHRINKAGE_K:g}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    print()
    print(f"selected k justification: {report.selected_k_justification}")
    _print_table("Coverage", report.coverage)
    _print_table("Position-baseline warm-up", report.position_baseline_warmup)
    _print_table("Tune / validation split", report.split)
    _print_table("Shrinkage grid (tune)", report.shrinkage_grid_tune)
    _print_table("Shrinkage grid (validation)", report.shrinkage_grid_validation)
    _print_table("Empirical-Bayes sanity check (tune)", report.empirical_bayes)
    _print_table("History-size buckets (tune)", report.history_bucket_tune)
    _print_table(
        "History-size buckets (validation)", report.history_bucket_validation
    )
    _print_table("State distribution", report.state_distribution)
    _print_table("Persistence (prior state vs next C)", report.persistence)
    _print_table("Consecutive-appearance persistence", report.consecutive_persistence)
    _print_table("First-half vs second-half", report.first_half_second_half)
    _print_table("Regression to the mean", report.regression_to_mean)
    _print_table("Relationship with frozen farming state", report.farming_relationship)
    _print_table("Classification", report.classification)
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        print(f"{key}: {value}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice18_report_to_jsonable(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
