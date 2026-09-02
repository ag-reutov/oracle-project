"""CLI for Slice 22 leakage-safe historical hero×position requirement states.

Leave-current-player-out hero-role requirements only. Does not persist a
hero rating, does not add columns to FEATURE_COLUMNS, does not construct
player×hero fit, does not aggregate to team, does not call STRATZ, and
does not train a win model.

Usage:
    uv run python scripts/audit_hero_requirement_state.py
    uv run python scripts/audit_hero_requirement_state.py \\
        --output data/interim/slice22_hero_requirement_state.json
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
    run_hero_requirement_state_diagnostics,
    slice22_report_to_jsonable,
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
            "Development-only Slice 22 hero-requirement-state diagnostics. "
            "Does not evaluate TI 2026 and does not train a win model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice22_hero_requirement_state.json"),
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
        report = run_hero_requirement_state_diagnostics(store)

    print("=== Slice 22 leakage-safe historical hero×position requirement states ===")
    print(
        "Leave-current-player-out causal hero×position requirements only. "
        "Farming target is farming_causal_b; combat target is combat_causal_c. "
        "History is start_time < T, same hero, same explicit position 1–5, "
        "player_id != current player. Shrinkage is toward zero and is chosen "
        "by predicting the next frozen target, not match outcomes. "
        "Current position is diagnostic and is not a PRE_DRAFT production input. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        "Holdout / TI 2026 matches are excluded from every summary."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"player rows={report.n_development_player_rows}  "
        f"holdout rows excluded={report.n_holdout_excluded}  "
        f"tune_end={report.tune_end.isoformat()}  "
        f"k_farm_hero={report.selected_k_farm:g}  "
        f"k_combat_hero={report.selected_k_combat:g}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    print()
    print(f"farming k justification: {report.selected_k_farm_justification}")
    print(f"combat k justification: {report.selected_k_combat_justification}")
    _print_table("Tune / validation split", report.split)
    _print_table("Farming coverage", report.farming_coverage)
    _print_table("Combat coverage", report.combat_coverage)
    _print_table("Farming inclusive vs LPO", report.farming_inclusive_vs_lpo)
    _print_table("Combat inclusive vs LPO", report.combat_inclusive_vs_lpo)
    _print_table("Farming shrinkage grid (tune)", report.farming_grid_tune)
    _print_table("Farming shrinkage grid (validation)", report.farming_grid_validation)
    _print_table("Combat shrinkage grid (tune)", report.combat_grid_tune)
    _print_table("Combat shrinkage grid (validation)", report.combat_grid_validation)
    _print_table("Farming empirical-Bayes", report.farming_empirical_bayes)
    _print_table("Combat empirical-Bayes", report.combat_empirical_bayes)
    _print_table("Farming history-size buckets (tune)", report.farming_history_bucket_tune)
    _print_table(
        "Farming history-size buckets (validation)",
        report.farming_history_bucket_validation,
    )
    _print_table("Combat history-size buckets (tune)", report.combat_history_bucket_tune)
    _print_table(
        "Combat history-size buckets (validation)",
        report.combat_history_bucket_validation,
    )
    _print_table("Farming unique-player buckets", report.farming_unique_player)
    _print_table("Combat unique-player buckets", report.combat_unique_player)
    _print_table("Farming specialist cells", report.farming_specialist)
    _print_table("Combat specialist cells", report.combat_specialist)
    _print_table("Farming state distribution", report.farming_state_distribution)
    _print_table("Combat state distribution", report.combat_state_distribution)
    _print_table("Farming persistence", report.farming_persistence)
    _print_table("Combat persistence", report.combat_persistence)
    _print_table("Farming split-half", report.farming_split_half)
    _print_table("Combat split-half", report.combat_split_half)
    _print_table("Farming adjacent temporal blocks", report.farming_temporal_blocks)
    _print_table("Combat adjacent temporal blocks", report.combat_temporal_blocks)
    _print_table("Farming patch/version error", report.farming_patch)
    _print_table("Combat patch/version error", report.combat_patch)
    _print_table("Farming regression to the mean", report.farming_regression_to_mean)
    _print_table("Combat regression to the mean", report.combat_regression_to_mean)
    _print_table("Relationship with frozen player states", report.player_state_relationship)
    _print_table("Farming vs combat hero requirement", report.cross_dimension)
    _print_table("Classification", report.classification)
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        if key in {"farming_gate", "combat_gate"}:
            print(f"{key}:")
            for inner_key, inner_value in value.items():
                print(f"  {inner_key}: {inner_value}")
        else:
            print(f"{key}: {value}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice22_report_to_jsonable(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
