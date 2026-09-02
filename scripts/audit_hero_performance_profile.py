"""CLI for Slice 21 hero resource / combat profile diagnostics.

Hero-state discovery only. Does not persist a hero rating, does not add
columns to FEATURE_COLUMNS, does not construct player×hero fit, does
not aggregate to team, does not call STRATZ, and does not train a win
model.

Usage:
    uv run python scripts/audit_hero_performance_profile.py
    uv run python scripts/audit_hero_performance_profile.py \\
        --output data/interim/slice21_hero_performance_profile.json
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
    run_hero_performance_profile_diagnostics,
    slice21_report_to_jsonable,
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
            "Development-only Slice 21 hero-profile diagnostics. "
            "Does not evaluate TI 2026 and does not train a win model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice21_hero_performance_profile.json"),
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
        report = run_hero_performance_profile_diagnostics(store)

    print("=== Slice 21 hero resource / combat profile diagnostics ===")
    print(
        "Hero-state discovery only. Profiles are requirement/role tendencies, "
        "not strength. Current position is used diagnostically and is not a "
        "PRE_DRAFT production input. "
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
    _print_table("Formulas", report.formulas)
    _print_table("Coverage by representation", report.coverage)
    _print_table("Coverage thresholds", report.coverage_thresholds)
    _print_table("Position dependence", report.position_dependence)
    _print_table("Flex-hero sample", report.flex_heroes)
    _print_table("Circularity / leave-player-out", report.circularity)
    _print_table("Player-demeaned hero identity", report.player_demean)
    _print_table("Split-half repeatability", report.split_half)
    _print_table("Chronological blocks", report.temporal_blocks)
    _print_table("Adjacent-block stability", report.adjacent_block_stability)
    _print_table("Between/within variance", report.variance_decomposition)
    _print_table("Shrinkage diagnosis (no k freeze)", report.shrinkage_diagnosis)
    _print_table("Patch sample counts", report.patch_stability)
    _print_table("Adjacent patch stability", report.adjacent_patch_stability)
    _print_table(
        "Same-hero same-position patch stability",
        report.same_hero_position_patch,
    )
    _print_table(
        "Relationship with frozen player state",
        report.player_state_relationship,
    )
    _print_table("Farming vs combat hero profiles", report.cross_dimension)
    _print_table("Farming candidate comparison", report.farming_comparison)
    _print_table("Combat candidate comparison", report.combat_comparison)
    _print_table("Classification", report.classification)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = slice21_report_to_jsonable(report)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
