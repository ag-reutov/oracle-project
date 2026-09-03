"""CLI for Slice 26 causal sequential draft-state audit.

Research / dataset integrity only. Does not add columns to
FEATURE_COLUMNS, does not train a next-pick model, and does not
compute predictive draft metrics.

Usage:
    uv run python scripts/audit_sequential_draft_state.py
    uv run python scripts/audit_sequential_draft_state.py \\
        --output data/interim/slice26_sequential_draft_state.json
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
    run_sequential_draft_state_diagnostics,
    slice26_report_to_jsonable,
)
from dota_predictor.utils.env import load_project_env

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 88)


def _print_mapping(title: str, mapping: dict) -> None:
    print()
    print(f"--- {title} ---")
    if not mapping:
        print("(empty)")
        return
    frame = pd.DataFrame(
        [{"key": key, "value": value} for key, value in mapping.items()]
    )
    print(frame.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only Slice 26 sequential draft-state integrity "
            "audit. Reconstructability only — no next-pick prediction. "
            "Does not evaluate TI 2026 / holdout."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/slice26_sequential_draft_state.json"),
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
        report = run_sequential_draft_state_diagnostics(store)

    audit = report.audit
    print("=== Slice 26 causal sequential draft-state dataset ===")
    print(
        "Dataset integrity / reconstruction only. "
        f"Boundary convention: {report.boundary_convention} "
        "(S_(M,t) = state before event sequence t; events with "
        "sequence < t). No next-hero log-loss, accuracy, win model, "
        "or draft-value metrics. "
        f"Population: start_time <= {report.development_end.isoformat()}. "
        "Holdout / TI 2026 matches are excluded from every summary."
    )
    print(
        f"development matches={report.n_development_matches}  "
        f"holdout excluded={report.n_holdout_excluded}  "
        f"classification={report.recorded_classification}  "
        f"FEATURE_COLUMNS={report.feature_columns_length}  "
        f"frozen development end={FROZEN_DEVELOPMENT_END.isoformat()}"
    )
    print()
    print(f"frozen components: {list(report.frozen_components)}")
    print(f"recommended modeling categories: {list(report.recommended_modeling_categories)}")
    print()
    print(report.phase_format_notes)
    print()
    print(report.hero_catalog_notes)

    _print_mapping(
        "Coverage",
        {
            "matches_total": audit["matches_total"],
            "matches_with_any_draft_events": audit["matches_with_any_draft_events"],
            "matches_with_exactly_10_successful_picks": audit[
                "matches_with_exactly_10_successful_picks"
            ],
            "matches_with_fewer_than_10_successful_picks": audit[
                "matches_with_fewer_than_10_successful_picks"
            ],
            "matches_with_more_than_10_successful_picks": audit[
                "matches_with_more_than_10_successful_picks"
            ],
            "ordering_ok_rate": audit["ordering_ok_rate"],
            "field_usable_rate": audit["field_usable_rate"],
            "terminal_exact_rate": audit["terminal_exact_rate"],
            "well_formed_share": audit["well_formed_share"],
            "unsuccessful_draft_events": audit["unsuccessful_draft_events"],
            "duplicate_match_id_sequence_rows": audit[
                "duplicate_match_id_sequence_rows"
            ],
            "malformed_sequence_matches": audit["malformed_sequence_matches"],
        },
    )
    _print_mapping("Event count distribution", audit["event_count_distribution"])
    _print_mapping(
        "Successful ban count distribution",
        audit["successful_ban_count_distribution"],
    )
    _print_mapping("Match categories", audit["category_counts"])
    _print_mapping("Terminal status", audit["terminal_status_counts"])
    _print_mapping("Coverage by year (total)", audit["coverage_by_year"]["total"])
    _print_mapping(
        "Coverage by mapper_version", audit["coverage_by_mapper_version"]
    )
    print()
    print("--- Classification ---")
    print(report.classification.to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(slice26_report_to_jsonable(report), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
