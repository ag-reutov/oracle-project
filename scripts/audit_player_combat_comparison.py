"""CLI for Slice 19 leakage-safe pre-draft combat team comparison.

Frozen Slice 18 player state averaged per side, then Radiant − Dire.
Does not persist a player rating, does not add columns to FEATURE_COLUMNS,
does not modify Elo, farming k=5, or combat k=20, does not call STRATZ,
and does not train a win model.

Usage:
    uv run python scripts/audit_player_combat_comparison.py
    uv run python scripts/audit_player_combat_comparison.py \\
        --output data/interim/slice19_player_combat_comparison.json
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
    run_player_combat_comparison_diagnostics,
    slice19_report_to_jsonable,
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
            "Development-only Slice 19 player-combat team comparison. "
            "Does not evaluate TI 2026 and does not train a win model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice19_player_combat_comparison.json"),
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
        report = run_player_combat_comparison_diagnostics(store)

    print("=== Slice 19 leakage-safe pre-draft combat team comparison ===")
    print(
        "Frozen Slice 18 shrunk causal C, averaged over the five rostered "
        "players per side, then Radiant − Dire. Known at PRE_DRAFT from "
        "player identities and strictly earlier appearances. Current hero "
        "damage, kills, assists, deaths, position, hero, and result are "
        "not inputs. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        "Holdout / TI 2026 matches are excluded from every summary."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"player rows={report.n_development_player_rows}  "
        f"holdout rows excluded={report.n_holdout_excluded}  "
        f"frozen_combat_k={report.frozen_combat_k:g}  "
        f"frozen_farming_k={report.frozen_farming_k:g}  "
        f"frozen candidate={FROZEN_COMBAT_CANDIDATE}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}  "
        f"FROZEN_COMBAT_SHRINKAGE_K={FROZEN_COMBAT_SHRINKAGE_K:g}  "
        f"FROZEN_SHRINKAGE_K={FROZEN_SHRINKAGE_K:g}"
    )
    _print_table("Coverage", report.coverage)
    print()
    print("--- Roster integrity ---")
    for key, value in report.roster_integrity.items():
        print(f"{key}: {value}")
    _print_table(
        "Radiant cold-start count distribution",
        report.cold_start_radiant_distribution,
    )
    _print_table(
        "Dire cold-start count distribution", report.cold_start_dire_distribution
    )
    _print_table(
        "Player combat_prior_n at current appearances", report.prior_n_distribution
    )
    _print_table("Feature distribution", report.feature_distribution)
    _print_table("Relationship with farming comparison", report.farming_relationship)
    _print_table("Farming joint quartiles", report.farming_joint_quartiles)
    _print_table("Relationship with Elo", report.elo_relationship)
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        print(f"{key}: {value}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice19_report_to_jsonable(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
