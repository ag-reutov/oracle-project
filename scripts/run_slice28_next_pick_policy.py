"""CLI for Slice 28 causal next-pick draft-policy benchmark.

Predicts next successfully picked hero from frozen Slice 26 draft
prefixes vs historical drafting baselines. Development OOS only.

Usage:
    uv run python scripts/run_slice28_next_pick_policy.py
    uv run python scripts/run_slice28_next_pick_policy.py \\
        --output data/interim/slice28_next_pick_policy.json
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
    run_slice28_next_pick_policy_benchmark,
    slice28_report_to_jsonable,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.8f}")
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 100)


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
            "Slice 28 walk-forward: next successful pick hero from causal "
            "draft prefix vs historical baselines. Development OOS only."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice28_next_pick_policy.json"),
        help="JSON path under data/interim/ (gitignored).",
    )
    parser.add_argument(
        "--skip-logistic",
        action="store_true",
        help="Score frequency baselines only (skip multinomial candidates).",
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
        report = run_slice28_next_pick_policy_benchmark(
            store, run_logistic_candidates=not args.skip_logistic
        )

    print("=== Slice 28 next-pick draft-policy benchmark ===")
    print(
        "Multiclass next successful pick from frozen Slice 26 prefixes. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        f"Holdout excluded ({report.n_holdout_excluded}). "
        f"FEATURE_COLUMNS={report.integrity['feature_columns_length']}."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"decision_rows={report.n_decision_rows}  "
        f"oos_rows={report.n_oos_rows}  "
        f"oos_matches={report.n_oos_matches}  "
        f"first_obs_rate={report.first_observed_target_rate:.6f}  "
        f"pattern={report.prefix_value_pattern}  "
        f"classification={report.classification.split('—')[0].strip()}  "
        f"frozen end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    print()
    print(report.classification_rationale)
    print()
    print(f"frozen components: {list(report.frozen_components)}")
    print(report.terminal_notes)

    _print_table("Pooled metrics", report.pooled_metrics)
    _print_table(
        "Paired Δ log loss (candidate − baseline; negative = better)",
        report.paired_deltas,
    )
    _print_table("Pick-position breakdown", report.pick_position_breakdown)
    _print_table("Side breakdown", report.side_breakdown)
    _print_table("Prefix ablation", report.prefix_ablation)
    _print_table("Ban ablation", report.ban_ablation)
    _print_table("Team ablation", report.team_ablation)
    _print_table("Team-history coverage", report.team_history_breakdown)
    _print_table("Signature diagnostics", report.signature_breakdown)
    _print_table("Version diagnostics", report.version_breakdown)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice28_report_to_jsonable(report), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
