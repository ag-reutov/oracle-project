"""CLI for Slice 8: exploratory walk-forward Career Player × Hero gating.

Uses the same expanding-window OOS folds as Slice 7. Does not change
fold boundaries, Elo, or production FEATURE_COLUMNS.

This experiment is exploratory: the same 2024–2026 OOS window already
informed the Slice 7 hypothesis.

Usage:
    uv run python scripts/run_slice8_player_hero_gating_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

from dota_predictor.features import (
    connect,
    load_feature_store_config,
    load_reference_store_config,
    register_reference_views,
)
from dota_predictor.training import (
    DEFAULT_WALK_FORWARD_CONFIG,
    SLICE8_META_PLAYER_HERO_SPECS,
    run_slice8_player_hero_gating_benchmark,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 80)


def _print_overall(overall: pd.DataFrame) -> None:
    view = overall[
        [
            "label",
            "n",
            "log_loss",
            "delta_vs_elo",
            "delta_vs_career",
            "brier_score",
            "roc_auc",
            "ece",
        ]
    ].rename(
        columns={
            "label": "Spec",
            "n": "N",
            "log_loss": "LogLoss",
            "delta_vs_elo": "Δ vs Elo",
            "delta_vs_career": "Δ vs Career",
            "brier_score": "Brier",
            "roc_auc": "AUC",
            "ece": "ECE",
        }
    )
    print(view.to_string(index=False))


def _print_folds(fold_metrics: pd.DataFrame) -> None:
    view = fold_metrics[
        [
            "fold_id",
            "label",
            "n_train",
            "n_validation",
            "n_test",
            "log_loss",
            "mean_delta_vs_elo",
            "delta_vs_career",
        ]
    ].rename(
        columns={
            "fold_id": "Fold",
            "label": "Spec",
            "n_train": "Train N",
            "n_validation": "Val N",
            "n_test": "Test N",
            "log_loss": "LogLoss",
            "mean_delta_vs_elo": "Δ vs Elo",
            "delta_vs_career": "Δ vs Career",
        }
    )
    print(view.to_string(index=False))


def _print_wide(frame: pd.DataFrame, *, title_cols: list[str]) -> None:
    rename = {
        "n": "N",
        "career_delta_vs_elo": "Career Δ vs Elo",
        "evidence_delta_vs_elo": "Evidence-ixn Δ vs Elo",
        "role_delta_vs_elo": "Role-ixn Δ vs Elo",
        "patch_delta_vs_elo": "Patch-ixn Δ vs Elo",
        "full_delta_vs_elo": "Full Δ vs Elo",
        "gate_delta_vs_elo": "Gate Δ vs Elo",
        "full_delta_vs_career": "Full Δ vs Career",
        "gate_delta_vs_career": "Gate Δ vs Career",
        "career_evidence_bin": "Career evidence",
        "compatibility_bin": "Compatibility",
        "cross_cell": "Career × compatibility",
        "maturity": "Maturity",
    }
    cols = title_cols + [
        "n",
        "career_delta_vs_elo",
        "full_delta_vs_elo",
        "gate_delta_vs_elo",
        "full_delta_vs_career",
        "gate_delta_vs_career",
        "evidence_delta_vs_elo",
        "role_delta_vs_elo",
        "patch_delta_vs_elo",
    ]
    present = [column for column in cols if column in frame.columns]
    print(frame[present].rename(columns=rename).to_string(index=False))


def _print_coefficients(coefficients: pd.DataFrame) -> None:
    if coefficients.empty:
        print("(none)")
        return
    view = coefficients.copy()
    view["abs_coef"] = view["coefficient"].abs()
    top = (
        view.sort_values(["fold_id", "model", "abs_coef"], ascending=[True, True, False])
        .groupby(["fold_id", "model"], sort=False)
        .head(12)
        .drop(columns=["abs_coef"])
    )
    print(top.to_string(index=False))


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

    reference_config = load_reference_store_config(root=root)
    catalog_available = (
        reference_config.heroes_path.is_file()
        and reference_config.game_versions_path.is_file()
    )

    started = perf_counter()
    wf_config = DEFAULT_WALK_FORWARD_CONFIG
    with connect(config) as store:
        if catalog_available:
            register_reference_views(store, reference_config)
        report = run_slice8_player_hero_gating_benchmark(store, config=wf_config)
    elapsed = perf_counter() - started

    folds = report.walk_forward.folds
    assembly = report.assembly

    print("=== Slice 8: exploratory Career Player × Hero gating ===")
    print(
        "EXPLORATORY: the 2024–2026 OOS window already informed the Slice 7 "
        "hypothesis. This is not independent confirmation."
    )
    print(
        "Paired Δ = spec log loss − reference log loss "
        "(negative = spec better than the reference)."
    )
    print(
        "References: Elo = logistic Elo-only; "
        "Career = existing Player × Hero comparison block (unchanged)."
    )
    print(
        "Interactions are row-wise Career-signal × context products. "
        "log1p(count) is the career/same-version magnitude transform. "
        "NULL rates stay NULL (not 0%). Train-only median impute + "
        "`{column}__was_missing` (same PreprocessingSpec)."
    )
    print(
        "Fold-selected gate: TRAIN feature quantiles define a small candidate "
        "grid; VAL log loss chooses one gate; C is then selected on VAL with "
        "the frozen gate; TEST is scored once."
    )
    print(
        "specs: "
        + " | ".join(
            f"{spec.label} ({len(spec.feature_columns)})"
            for spec in SLICE8_META_PLAYER_HERO_SPECS
        )
    )
    print(
        f"blocks={wf_config.n_blocks}  folds={len(folds)}  "
        f"train_fraction_of_past={wf_config.train_fraction_of_past:.4f}  "
        f"post_draft_rows={assembly.n_post_draft_matches}  "
        f"oos={report.n_oos}"
    )
    print(
        "Slice 7 identity: same post-draft match_id order and the same "
        "walk-forward TEST match_ids (asserted at run start)."
    )
    print(
        "Diagnostic bins: LOW/MEDIUM/HIGH tertiles from each fold's TRAIN "
        "features, applied to that fold's TEST. Patch maturity keeps the "
        "Slice 7 semantic 0–49 / 50–199 / 200+ cuts."
    )
    print(f"runtime: {elapsed:.1f}s")
    print()

    print("Fold windows:")
    window_rows = []
    for fold in folds:
        window_rows.append(
            {
                "fold": fold.fold_id,
                "n_train": len(fold.train),
                "n_validation": len(fold.validation),
                "n_test": len(fold.test),
                "train_end": fold.train_end,
                "validation_end": fold.validation_end,
                "test_end": fold.test_end,
            }
        )
    print(pd.DataFrame(window_rows).to_string(index=False))
    print()

    print("OVERALL")
    _print_overall(report.overall)
    print()
    print("FOLD-BY-FOLD")
    _print_folds(report.fold_metrics)
    print()
    print("FOLD-SELECTED GATES (TRAIN quantiles, VAL choice, TEST unused)")
    print(report.selected_gates.to_string(index=False))
    print()
    print("CAREER EVIDENCE (TRAIN tertiles per fold)")
    _print_wide(report.career_evidence, title_cols=["career_evidence_bin"])
    print()
    print("ROLE COMPATIBILITY (TRAIN tertiles per fold)")
    _print_wide(report.compatibility, title_cols=["compatibility_bin"])
    print()
    print("CAREER EVIDENCE × ROLE COMPATIBILITY")
    _print_wide(report.cross_cell, title_cols=["cross_cell"])
    print()
    print("PATCH MATURITY (strictly-prior same-version matches)")
    _print_wide(report.patch_maturity, title_cols=["maturity"])
    print()
    print("STANDARDIZED COEFFICIENTS (Career signal + context/interactions)")
    _print_coefficients(report.coefficients)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
