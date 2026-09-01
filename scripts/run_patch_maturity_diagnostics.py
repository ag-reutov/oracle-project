"""CLI for patch-age / professional-match maturity diagnostics.

Reuses walk-forward OOS predictions (retrains the existing six blocks
only because those predictions are not persisted). Joins descriptive
calendar age and prior-match counts. Does not add features, change
Elo, retune bins, or select a winning block.

Usage:
    uv run python scripts/run_patch_maturity_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from dota_predictor.features import (
    build_patch_maturity,
    connect,
    load_feature_store_config,
    load_reference_store_config,
    patch_age_sanity_table,
    register_reference_views,
)
from dota_predictor.training import (
    DEFAULT_WALK_FORWARD_CONFIG,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    build_familiarity_mix,
    build_post_draft_model_ready_dataset,
    run_patch_maturity_diagnostics,
    run_post_draft_walk_forward,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 40)


def _sparse_note(counts: pd.DataFrame, *, min_n: int = 50) -> str:
    sparse = counts.loc[counts["n"] < min_n]
    if sparse.empty:
        return f"no bin has n < {min_n}"
    parts = [f"{row.bin} (n={int(row.n)})" for row in sparse.itertuples()]
    return "sparse bins (n < " + str(min_n) + "): " + "; ".join(parts)


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
    if not (
        reference_config.heroes_path.is_file()
        and reference_config.game_versions_path.is_file()
    ):
        print(
            "Reference catalog (heroes + game_versions) is required "
            "for patch-age diagnostics.",
            file=sys.stderr,
        )
        return 1

    with connect(config) as store:
        register_reference_views(store, reference_config)
        maturity = build_patch_maturity(store).to_frame()
        sanity = patch_age_sanity_table(maturity)
        dataset = build_post_draft_model_ready_dataset(store)
        familiarity_mix = build_familiarity_mix(store)

    wf_config = DEFAULT_WALK_FORWARD_CONFIG
    report = run_post_draft_walk_forward(dataset, config=wf_config)
    diagnostics = run_patch_maturity_diagnostics(
        report.oos_predictions,
        maturity,
        familiarity_mix,
        sanity,
    )

    n_corpus = len(maturity)
    n_negative = int((maturity["days_since_game_version_start"] < 0).sum())
    n_missing = int(maturity["days_since_game_version_start"].isna().sum())

    print("=== Patch-age / professional-match maturity diagnostics ===")
    print(
        "Descriptive study only. Calendar age = start_time − STRATZ "
        "as_of_datetime. Professional-match maturity = strictly prior "
        "dataset matches in the same game_version_id (same-timestamp "
        "peers excluded). Bins were predefined, not tuned on metrics."
    )
    print(
        "specs: "
        + " | ".join(
            f"{spec.label} ({len(spec.feature_columns)})"
            for spec in POST_DRAFT_BLOCK_ABLATION_SPECS
        )
    )
    print(
        f"corpus matches={n_corpus}  walk-forward OOS={diagnostics.n_oos_matches}  "
        f"folds={len(report.folds)}  "
        f"negative calendar age={n_negative}  missing as_of={n_missing}"
    )
    print(
        "paired delta = draft spec match log loss − Elo match log loss "
        "(negative = spec better). Accuracy is not the primary metric."
    )
    print()
    print("--- 1. Patch-age sanity by game version (full corpus) ---")
    print(diagnostics.sanity.to_string(index=False))
    flagged = diagnostics.sanity.loc[diagnostics.sanity["flagged"]]
    if flagged.empty:
        print("No negative calendar ages or missing as_of rows.")
    else:
        print(
            "FLAGGED versions (negative calendar age and/or missing as_of; "
            "source metadata not repaired):"
        )
        print(
            flagged[
                [
                    "game_version_id",
                    "name",
                    "n_negative_calendar_age",
                    "n_missing_as_of",
                    "min_days_since_game_version_start",
                ]
            ].to_string(index=False)
        )
    print()
    print("--- OOS sample sizes by calendar-age bin ---")
    print(diagnostics.calendar_bin_counts.to_string(index=False))
    print(_sparse_note(diagnostics.calendar_bin_counts))
    print()
    print("--- OOS sample sizes by professional-match maturity bin ---")
    print(diagnostics.prior_match_bin_counts.to_string(index=False))
    print(_sparse_note(diagnostics.prior_match_bin_counts))
    print()
    print("--- 2. Paired Δ log loss by calendar patch-age bin ---")
    print(diagnostics.calendar_bins.to_string(index=False))
    print()
    print("--- 3. Paired Δ log loss by professional-match maturity bin ---")
    print(diagnostics.prior_match_bins.to_string(index=False))
    print()
    print("--- 4. Block × game version (OOS) ---")
    print(diagnostics.version_matrix.to_string(index=False))
    print()
    print("--- 7. Continuous: Pearson / Spearman of per-match Δ vs age ---")
    print(diagnostics.correlations.to_string(index=False))
    print()
    print("--- Quantile-binned mean Δ vs calendar age ---")
    print(diagnostics.calendar_quantile_means.to_string(index=False))
    print()
    print("--- Quantile-binned mean Δ vs prior matches in version ---")
    print(diagnostics.prior_quantile_means.to_string(index=False))
    print()
    print(
        "--- 8. Carryover: lifetime / same-version / recent-90d "
        "(Player × Hero, Team × Hero) ---"
    )
    print(
        "stale = lifetime − same-version (Radiant−Dire mean games). "
        "Pearson is per-match Δ vs that signed mix. Existing metrics only."
    )
    print(diagnostics.carryover.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
