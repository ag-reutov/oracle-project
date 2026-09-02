"""CLI for Slice 23 player × hero behavioral-compatibility diagnostics.

Does not persist a fit score, does not add columns to FEATURE_COLUMNS,
does not aggregate to team, does not call STRATZ, and does not train a
win model.

Usage:
    uv run python scripts/audit_player_hero_compatibility.py
    uv run python scripts/audit_player_hero_compatibility.py \\
        --output data/interim/slice23_player_hero_compatibility.json
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
    run_player_hero_compatibility_diagnostics,
    slice23_report_to_jsonable,
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


def _print_comparison(title: str, frame: pd.DataFrame) -> None:
    keep = [
        column
        for column in (
            "model",
            "label",
            "split",
            "n",
            "rmse",
            "mae",
            "pearson",
            "delta_rmse",
            "delta_mae",
            "delta_rmse_p025",
            "delta_rmse_p975",
            "algebraically_redundant",
            "max_abs_pred_diff_vs_m3",
        )
        if column in frame.columns
    ]
    _print_table(title, frame.loc[:, keep])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only Slice 23 player×hero compatibility diagnostics. "
            "Does not evaluate TI 2026 and does not train a win model."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice23_player_hero_compatibility.json"),
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
        report = run_player_hero_compatibility_diagnostics(store)

    print("=== Slice 23 player × hero behavioral compatibility ===")
    print(
        "Diagnostics only. Frozen Slice 14 farming player state (k=5) and "
        "Slice 18 combat player state (k=20) versus frozen Slice 22 LPO "
        "hero×position requirements (k=2 / k=2). Targets are farming_causal_b "
        "and combat_causal_c. The question is the incremental value of a "
        "P×H relationship beyond additive P+H. High P with high H is not "
        "automatically good fit. Holdout / TI 2026 matches are excluded "
        f"from every summary. Population: start_time <= "
        f"{report.development_end.isoformat()}."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"player rows={report.n_development_player_rows}  "
        f"holdout rows excluded={report.n_holdout_excluded}  "
        f"tune_end={report.tune_end.isoformat()}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    _print_table("Classification", report.classification)
    _print_table("Tune / validation split", report.split)
    _print_table("Farming eligible coverage", report.farming_coverage)
    _print_table("Combat eligible coverage", report.combat_coverage)
    _print_table("Farming marginal P/H → target", report.farming_marginal)
    _print_table("Combat marginal P/H → target", report.combat_marginal)
    _print_comparison("Farming candidate comparison", report.farming_comparison)
    _print_comparison("Combat candidate comparison", report.combat_comparison)
    _print_table("Farming residual diagnostics", report.farming_residual)
    _print_table("Combat residual diagnostics", report.combat_residual)
    _print_table("Farming signed-gap shape bins", report.farming_shape_bins)
    _print_table("Combat signed-gap shape bins", report.combat_shape_bins)
    _print_table("Farming below vs above requirement", report.farming_asymmetry)
    _print_table("Combat below vs above requirement", report.combat_asymmetry)
    strongest_f = str(report.classification.iloc[0]["farming_strongest"])
    strongest_c = str(report.classification.iloc[0]["combat_strongest"])
    farming_pos = report.farming_position
    combat_pos = report.combat_position
    if "model" in farming_pos.columns:
        farming_pos = farming_pos.loc[
            farming_pos["model"].isin(["M3", strongest_f])
        ]
    if "model" in combat_pos.columns:
        combat_pos = combat_pos.loc[combat_pos["model"].isin(["M3", strongest_c])]
    _print_table("Farming position robustness", farming_pos)
    _print_table("Combat position robustness", combat_pos)
    farming_hero = report.farming_hero_history
    combat_hero = report.combat_hero_history
    if "model" in farming_hero.columns:
        farming_hero = farming_hero.loc[
            farming_hero["model"].isin(["M3", strongest_f])
        ]
    if "model" in combat_hero.columns:
        combat_hero = combat_hero.loc[
            combat_hero["model"].isin(["M3", strongest_c])
        ]
    _print_table("Farming hero-history robustness", farming_hero)
    _print_table("Combat hero-history robustness", combat_hero)
    farming_player = report.farming_player_history
    combat_player = report.combat_player_history
    if "model" in farming_player.columns:
        farming_player = farming_player.loc[
            farming_player["model"].isin(["M3", strongest_f])
        ]
    if "model" in combat_player.columns:
        combat_player = combat_player.loc[
            combat_player["model"].isin(["M3", strongest_c])
        ]
    _print_table("Farming player-history robustness", farming_player)
    _print_table("Combat player-history robustness", combat_player)
    farming_spec = report.farming_specialist
    combat_spec = report.combat_specialist
    if "model" in farming_spec.columns:
        farming_spec = farming_spec.loc[
            farming_spec["model"].isin(["M3", strongest_f])
        ]
    if "model" in combat_spec.columns:
        combat_spec = combat_spec.loc[
            combat_spec["model"].isin(["M3", strongest_c])
        ]
    _print_table("Farming specialist robustness", farming_spec)
    _print_table("Combat specialist robustness", combat_spec)
    _print_table("Farming patch/version robustness", report.farming_patch)
    _print_table("Combat patch/version robustness", report.combat_patch)
    _print_table("Farming permutation/placebo", report.farming_permutation)
    _print_table("Combat permutation/placebo", report.combat_permutation)
    _print_table("Farming collinearity caveats", report.farming_collinearity)
    _print_table("Combat collinearity caveats", report.combat_collinearity)
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
        json.dumps(slice23_report_to_jsonable(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
