"""CLI for Slice 7: walk-forward ablation of meta-aware Player × Hero.

Uses the same expanding-window OOS folds as the Elo / Player × Hero
benchmarks. Does not change fold boundaries, Elo, or production
FEATURE_COLUMNS.

Usage:
    uv run python scripts/run_slice7_meta_player_hero_benchmark.py
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
    SLICE7_META_PLAYER_HERO_SPECS,
    run_slice7_meta_player_hero_benchmark,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 80)


def _version_names(root: Path) -> dict[int, str]:
    reference_config = load_reference_store_config(root=root)
    path = reference_config.game_versions_path
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path, columns=["game_version_id", "name"])
    return {
        int(version_id): str(name)
        for version_id, name in zip(frame["game_version_id"], frame["name"])
    }


def _attach_version_name(frame: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    if "game_version_id" not in frame.columns:
        return frame
    labeled = frame.copy()
    labeled.insert(
        1,
        "game_version",
        labeled["game_version_id"].map(
            lambda value: names.get(int(value), str(value)) if pd.notna(value) else ""
        ),
    )
    return labeled


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
        "career_delta": "Career Δ",
        "same_version_delta": "Same-version Δ",
        "recent20_delta": "Recent-20 Δ",
        "role_delta": "Role-block Δ",
        "combined_delta": "Combined Δ",
        "career_log_loss": "Career",
        "same_version_log_loss": "Same-version",
        "recent20_log_loss": "Recent-20",
        "role_log_loss": "Role",
        "combined_log_loss": "Combined",
        "game_version_id": "Patch",
        "low_n": "low-N",
        "maturity": "Maturity",
        "population": "Population",
        "career_sample_bucket": "Career games",
        "compatibility_bucket": "Compatibility bucket",
        "hero_group": "Hero group",
    }
    cols = title_cols + [
        "n",
        "career_delta",
        "same_version_delta",
        "recent20_delta",
        "role_delta",
        "combined_delta",
    ]
    present = [column for column in cols if column in frame.columns]
    print(frame[present].rename(columns=rename).to_string(index=False))


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
        report = run_slice7_meta_player_hero_benchmark(store, config=wf_config)
    elapsed = perf_counter() - started

    slice7_folds = report.walk_forward.folds
    names = _version_names(root)
    assembly = report.assembly

    print("=== Slice 7: meta-aware Player × Hero walk-forward ===")
    print(
        "Paired Δ = spec log loss − reference log loss "
        "(negative = spec better than the reference)."
    )
    print(
        "References: Elo = logistic Elo-only; "
        "Career = existing Player × Hero comparison block."
    )
    print(
        "Win-rate NULLs are not 0%. Train-only median impute + "
        "`{column}__was_missing` (same PreprocessingSpec as the existing "
        "walk-forward). Zero evidence stays distinguishable from observed "
        "0% via the count column (0, never NULL) plus the rate missingness "
        "indicator."
    )
    print(
        "specs: "
        + " | ".join(
            f"{spec.label} ({len(spec.feature_columns)})"
            for spec in SLICE7_META_PLAYER_HERO_SPECS
        )
    )
    print(
        f"blocks={wf_config.n_blocks}  folds={len(slice7_folds)}  "
        f"train_fraction_of_past={wf_config.train_fraction_of_past:.4f}  "
        f"post_draft_rows={assembly.n_post_draft_matches}  "
        f"oos={report.n_oos}"
    )
    print(
        f"Slice 6 comparison coverage: matches={assembly.n_slice6_comparison_matches}  "
        f"missing_vs_post_draft={assembly.n_missing_slice6_comparison}  "
        f"incomplete_rosters(<10 players)={assembly.n_incomplete_player_rows}"
    )
    if assembly.n_missing_slice6_comparison:
        print(
            "ROW LOSS: some post-draft matches have no Slice 6 Radiant/Dire "
            "comparison; they remain in the common OOS set with NULL Slice 6 "
            "diffs (not dropped)."
        )
    else:
        print("Common OOS population: full post-draft walk-forward match set.")
    print(
        "Fold boundaries are resolve_walk_forward_folds on the post-draft "
        "match set (same start_time / n_blocks); Slice 7 does not retune them."
    )

    print(
        f"Role-compatibility quartile edges (OOS feature distribution, no outcomes): "
        f"q25={report.compatibility_q25:.4f}  q75={report.compatibility_q75:.4f}"
    )
    print(
        "Patch maturity: opening 0–49 prior same-version matches; "
        "early 50–199; mature 200+ (collapsed from the existing 200–499 / 500+ "
        "cuts; not chosen from prediction metrics)."
    )
    print(
        "Role-shift: Slice 5 descriptive rule — consecutive same-version "
        "snapshots, modal position change with ≥8 explicit observations; "
        "contest-shifted if |contest-rate delta| ≥ 0.20. Match label: any "
        "role-shifted drafted hero → role_shifted, else any contest-shifted → "
        "contest_shifted, else all stable → stable."
    )
    print(
        "Career sample-size / patch-cold use match-level mean of the ten "
        "players' career / same-version Player × Hero games."
    )
    print(f"runtime: {elapsed:.1f}s")
    print()

    print("Fold windows:")
    window_rows = []
    for fold in slice7_folds:
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
    print("BY PATCH (Δ vs Elo; low-N = N < 50, descriptive only)")
    _print_wide(
        _attach_version_name(report.by_patch, names),
        title_cols=["game_version_id", "game_version", "low_n"],
    )
    print()
    print("PATCH MATURITY (log loss; Δ vs Elo in Career Δ / … columns)")
    _print_wide(report.patch_maturity, title_cols=["maturity"])
    print()
    print("PATCH-COLD CAREER HISTORY")
    _print_wide(report.patch_cold, title_cols=["population"])
    print()
    print("CAREER SAMPLE SIZE (match-level mean prior games on current hero)")
    _print_wide(report.career_sample, title_cols=["career_sample_bucket"])
    print()
    print("ROLE COMPATIBILITY (OOS quartiles of mean recent-role compatibility)")
    _print_wide(report.compatibility, title_cols=["compatibility_bucket"])
    print()
    print("STABLE-ROLE VS ROLE-SHIFTED VS CONTEST-SHIFTED")
    _print_wide(report.role_shift, title_cols=["hero_group"])
    print()
    print("COUNT VS PERFORMANCE (Δ vs Elo)")
    print(
        report.count_vs_performance.rename(
            columns={
                "block": "Block",
                "count_only_delta_vs_elo": "Count-only Δ",
                "count_plus_wr_delta_vs_elo": "Count+WR Δ",
                "count_only_delta_vs_career": "Count-only Δ vs Career",
                "count_plus_wr_delta_vs_career": "Count+WR Δ vs Career",
            }
        ).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
