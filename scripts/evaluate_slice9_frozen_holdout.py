"""CLI: one-shot Slice 9 frozen holdout evaluation.

Scores Elo-only vs unconditional Career Player × Hero on later matches.
Selects C and fits preprocessing on development TRAIN/VAL only. Does not
tune on the holdout, add Slice 8 gates, or make a production-model
decision.

A lock is written after the first successful run. Later invocations
reprint the recorded result and refuse to re-fit.

Usage:
    uv run python scripts/evaluate_slice9_frozen_holdout.py
"""

from __future__ import annotations

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
    DEFAULT_WALK_FORWARD_CONFIG,
    FROZEN_DEVELOPMENT_END,
    FROZEN_DEVELOPMENT_MATCH_COUNT,
    FROZEN_DEVELOPMENT_OOS_MATCH_COUNT,
    FROZEN_HOLDOUT_EXPECTED_LEAGUE_ID,
    FROZEN_HOLDOUT_EXPECTED_N,
    FrozenHoldoutAlreadyEvaluatedError,
    FrozenHoldoutEvaluation,
    evaluate_frozen_holdout,
    load_frozen_holdout_eval_dir,
    load_frozen_holdout_evaluation,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 200)


def _print_lock(payload: dict) -> None:
    print("=== Slice 9: recorded one-shot holdout evaluation ===")
    print("Already evaluated. Refusing to re-select C, refit, or redefine.")
    print(f"evaluated={payload.get('evaluated')}")
    print(f"production_decision={payload.get('production_decision')}")
    print()
    print("--- Frozen specification ---")
    print(f"reference={payload.get('reference_spec')}")
    print(f"candidate={payload.get('candidate_spec')}")
    print(f"C grid={payload.get('C_grid')}")
    print(f"selected C={payload.get('selected_C')}")
    print()
    print("--- Census ---")
    print(
        f"development N={payload.get('n_development')}  "
        f"OOS N={payload.get('n_development_oos')}  "
        f"holdout N={payload.get('n_holdout')}"
    )
    print(f"development_end={payload.get('development_end')}")
    print(
        f"holdout {payload.get('holdout_start')} -> {payload.get('holdout_end')}  "
        f"leagues={payload.get('holdout_league_ids')}"
    )
    print()
    print("--- Holdout metrics ---")
    ref = payload.get("reference_metrics") or {}
    cand = payload.get("candidate_metrics") or {}
    print(
        f"Elo    log_loss={ref.get('log_loss')}  brier={ref.get('brier_score')}  "
        f"AUC={ref.get('roc_auc')}  ECE={ref.get('ece')}"
    )
    print(
        f"Career log_loss={cand.get('log_loss')}  brier={cand.get('brier_score')}  "
        f"AUC={cand.get('roc_auc')}  ECE={cand.get('ece')}"
    )
    print(
        f"paired Δ log loss={payload.get('paired_delta_log_loss')}  "
        f"paired Δ Brier={payload.get('paired_delta_brier')}"
    )
    print(
        "n(candidate lower log loss)="
        f"{payload.get('n_candidate_better_log_loss')}  "
        f"mean Δ={payload.get('mean_paired_log_loss_diff')}  "
        f"median Δ={payload.get('median_paired_log_loss_diff')}"
    )
    print(f"bootstrap 95% CI Δ log loss={payload.get('bootstrap_delta_log_loss_ci95')}")


def _print_report(report: FrozenHoldoutEvaluation, output_dir: Path) -> None:
    protocol = report.protocol
    print("=== Slice 9: one-shot frozen holdout evaluation ===")
    print("No new features. No Slice 8 gates. Not a production-model decision.")
    print(f"evaluated={protocol.evaluated}")
    print("production_decision=None")
    print()
    print("--- Frozen specification ---")
    print(
        f"reference={protocol.reference_spec.name}  "
        f"({len(protocol.reference_spec.feature_columns)} columns)"
    )
    print(
        f"candidate={protocol.candidate_spec.name}  "
        f"({len(protocol.candidate_spec.feature_columns)} columns)"
    )
    print(f"C grid={protocol.regularization_candidates}")
    print(f"selected C={report.selected_C}")
    print("VAL log loss by C:")
    print(report.regularization_comparison.to_string(index=False))
    print()
    print("--- Census ---")
    print(
        f"development N={protocol.n_development}  "
        f"(census {FROZEN_DEVELOPMENT_MATCH_COUNT})  "
        f"OOS N={protocol.n_development_oos}  "
        f"(census {FROZEN_DEVELOPMENT_OOS_MATCH_COUNT})"
    )
    print(f"development_end={protocol.development_end.isoformat()}")
    print(f"recorded constant={FROZEN_DEVELOPMENT_END.isoformat()}")
    print(
        f"holdout N={protocol.holdout.n}  "
        f"(expected {FROZEN_HOLDOUT_EXPECTED_N})  "
        f"{protocol.holdout.start} -> {protocol.holdout.end}  "
        f"league={FROZEN_HOLDOUT_EXPECTED_LEAGUE_ID}"
    )
    print(
        f"n_train={len(report.split.train)}  "
        f"n_validation={len(report.split.validation)}"
    )
    print()
    print("--- Holdout metrics ---")
    ref = report.reference_metrics
    cand = report.candidate_metrics
    print(
        f"Elo    log_loss={ref.log_loss:.6f}  brier={ref.brier_score:.6f}  "
        f"AUC={ref.roc_auc:.6f}  ECE={ref.expected_calibration_error:.6f}"
    )
    print(
        f"Career log_loss={cand.log_loss:.6f}  brier={cand.brier_score:.6f}  "
        f"AUC={cand.roc_auc:.6f}  ECE={cand.expected_calibration_error:.6f}"
    )
    print(
        f"paired Δ log loss={report.paired_delta_log_loss:.6f}  "
        f"paired Δ Brier={report.paired_delta_brier:.6f}"
    )
    print(
        "n(candidate lower log loss)="
        f"{report.n_candidate_better_log_loss}/{cand.n_samples}  "
        f"mean Δ={report.mean_paired_log_loss_diff:.6f}  "
        f"median Δ={report.median_paired_log_loss_diff:.6f}"
    )
    lo, hi = report.bootstrap_delta_log_loss_ci95
    print(f"bootstrap 95% CI Δ log loss=[{lo:.6f}, {hi:.6f}]")
    print()
    print("--- Diagnostics (not for tuning) ---")
    print("early/middle/late TI chronology:")
    print(report.chronology.to_string(index=False))
    print()
    print("Radiant/Dire winner:")
    print(report.winner_side.to_string(index=False))
    print()
    print("Career Player × Hero evidence (development TRAIN tertiles):")
    print(report.career_evidence.to_string(index=False))
    print()
    print("--- Predictions ---")
    print(report.predictions.to_string(index=False))
    print()
    print(f"wrote lock and predictions to {output_dir}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_project_env(root)
    config = load_feature_store_config(root=root)
    if not config.matches_path.is_file():
        print(
            f"Canonical matches file not found: {config.matches_path}",
            file=sys.stderr,
        )
        return 1

    output_dir = load_frozen_holdout_eval_dir(root=root)
    reference_config = load_reference_store_config(root=root)
    catalog_available = (
        reference_config.heroes_path.is_file()
        and reference_config.game_versions_path.is_file()
    )

    try:
        with connect(config) as store:
            if catalog_available:
                register_reference_views(store, reference_config)
            report = evaluate_frozen_holdout(
                store,
                config=DEFAULT_WALK_FORWARD_CONFIG,
                require_recorded_census=True,
                output_dir=output_dir,
            )
    except FrozenHoldoutAlreadyEvaluatedError as exc:
        _print_lock(load_frozen_holdout_evaluation(exc.path.parent))
        print()
        print(f"lock={exc.path}")
        return 0

    _print_report(report, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
