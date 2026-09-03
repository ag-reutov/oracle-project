"""CLI for Slice 27 incremental draft-value walk-forward benchmark.

Compares logistic Elo with Elo + side-aware checkpoint picks on
development OOS folds. Consumes frozen Slice 26 draft-state semantics.
Does not score the frozen holdout and does not change FEATURE_COLUMNS.

Usage:
    uv run python scripts/run_slice27_sequential_draft_benchmark.py
    uv run python scripts/run_slice27_sequential_draft_benchmark.py \\
        --output data/interim/slice27_sequential_draft_benchmark.json
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
    run_slice27_sequential_draft_benchmark,
    slice27_report_to_jsonable,
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
            "Slice 27 walk-forward: Elo vs Elo + checkpoint picks. "
            "Development OOS only; does not score TI 2026 / frozen holdout."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice27_sequential_draft_benchmark.json"),
        help="JSON path under data/interim/ (gitignored).",
    )
    parser.add_argument(
        "--skip-ban-ablation",
        action="store_true",
        help="Skip secondary Elo+picks vs Elo+picks+bans ablation.",
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
        report = run_slice27_sequential_draft_benchmark(
            store, run_ban_ablation=not args.skip_ban_ablation
        )

    print("=== Slice 27 incremental draft-value benchmark ===")
    print(
        "Win-outcome walk-forward on frozen Slice 26 checkpoints. "
        "Primary: Elo vs Elo + side-aware successful picks. "
        "Secondary: + successful bans before the same boundary. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        f"Holdout excluded ({report.n_holdout_excluded}). "
        f"FEATURE_COLUMNS={report.integrity['feature_columns_length']}."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"oos={report.n_oos}  "
        f"pattern={report.accumulation_pattern}  "
        f"classification={report.classification.split('—')[0].strip()}  "
        f"frozen end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    print()
    print(report.classification_rationale)
    print()
    print(f"frozen components: {list(report.frozen_components)}")
    print(report.terminal_notes)

    _print_table("Checkpoint coverage", report.checkpoint_coverage)
    _print_table(
        "Incremental draft-value curve (Δ LL = candidate − Elo)",
        report.checkpoint_curve[
            [
                "n_picks",
                "elo_log_loss",
                "candidate_log_loss",
                "delta_log_loss",
                "delta_ci_low",
                "delta_ci_high",
                "elo_brier",
                "candidate_brier",
                "elo_auc",
                "candidate_auc",
            ]
        ],
    )
    _print_table("Fold-by-fold Δ log loss", report.fold_deltas)
    _print_table("Accumulation", report.accumulation)
    _print_table("Side pick balance", report.side_pick_balance)
    _print_table("Ban ablation (Δ = picks+bans − picks)", report.ban_ablation)
    _print_table(
        "Signature diagnostics (top)",
        report.signature_diagnostics.head(8),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice27_report_to_jsonable(report), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
