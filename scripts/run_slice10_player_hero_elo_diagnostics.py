"""CLI for Slice 10 Elo-adjusted Player × Hero diagnostics.

Feature-state and diagnostics only. Does not train a win model, does not
add columns to FEATURE_COLUMNS, and does not score TI 2026.

Usage:
    uv run python scripts/run_slice10_player_hero_elo_diagnostics.py
    uv run python scripts/run_slice10_player_hero_elo_diagnostics.py \\
        --output data/interim/slice10_player_hero_elo_diagnostics.json
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
    run_player_hero_elo_diagnostics,
    slice10_report_to_jsonable,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 48)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only Slice 10 Elo-adjusted Player × Hero diagnostics. "
            "Does not evaluate TI 2026."
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
        report = run_player_hero_elo_diagnostics(store)

    print("=== Slice 10 Elo-adjusted Player × Hero diagnostics ===")
    print(
        "Feature-state only. Residual = historical player×hero outcome minus "
        "that appearance's pre-match team Elo expected win. Volume is "
        "shrinkage evidence, not strength. Frozen k = "
        f"{report.shrinkage_k_frozen:.0f} from σ²/τ² prior; development "
        "MoM k is reported but not used for state. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        "TI 2026 / later matches are excluded from every summary."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"player rows={report.n_development_player_rows}  "
        f"holdout rows excluded={report.n_holdout_excluded}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    print(report.shrinkage_k_prior_note)
    estimate = report.shrinkage_k_estimate
    print(
        "development MoM k="
        f"{estimate.k:.3f}  sigma2={estimate.residual_variance:.4f}  "
        f"tau2={estimate.effect_variance:.6f}  cells={estimate.n_cells}  "
        f"appearances={estimate.n_appearances}  "
        f"min_games_for_cell={estimate.min_games_for_cell}  "
        f"used_for_state={estimate.used_for_state}"
    )
    print()
    print("--- Residual distribution ---")
    print(report.residual_distribution.to_string(index=False))
    print()
    print("--- Residual by prior-games bin (volume is evidence, not strength) ---")
    print(report.residual_by_n.to_string(index=False))
    print()
    print("--- Highest-volume player×hero cells (latest development appearance) ---")
    print(report.high_volume_combinations.to_string(index=False))
    print()
    print("--- Temporal stability (early vs late development half, n>=5 each) ---")
    print(report.temporal_stability.to_string(index=False))
    print()
    print("--- Contrast with Career P×H volume / win rate ---")
    print(report.volume_contrast.to_string(index=False))
    print()
    print("--- Coverage / cold-start ---")
    print(report.coverage.to_string(index=False))
    print()
    print("--- Match-level Radiant−Dire comparison (not FEATURE_COLUMNS) ---")
    print(report.match_comparison_distribution.to_string(index=False))
    print()
    print("--- Integrity ---")
    for key, value in report.integrity.items():
        print(f"{key}: {value}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(slice10_report_to_jsonable(report), indent=2) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
