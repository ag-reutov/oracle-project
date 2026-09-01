"""CLI: record the Slice 9 frozen-model temporal holdout protocol.

Reports holdout N and date range. Does not score the holdout, and does
not tune thresholds, gates, C, feature definitions, or preprocessing.

Usage:
    uv run python scripts/record_slice9_frozen_holdout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    record_frozen_holdout_protocol,
)
from dota_predictor.training.slice9_frozen_holdout import (
    INSPECTED_LATER_T1_MAIN_EVENT_END,
    INSPECTED_LATER_T1_MAIN_EVENT_LEAGUE_ID,
    INSPECTED_LATER_T1_MAIN_EVENT_N,
    INSPECTED_LATER_T1_MAIN_EVENT_NAME,
    INSPECTED_LATER_T1_MAIN_EVENT_START,
)
from dota_predictor.utils.env import load_project_env


def _fmt(value: object) -> str:
    if value is None:
        return "(none)"
    return str(value)


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

    wf_config = DEFAULT_WALK_FORWARD_CONFIG
    with connect(config) as store:
        if catalog_available:
            register_reference_views(store, reference_config)
        protocol = record_frozen_holdout_protocol(
            store,
            config=wf_config,
            require_recorded_census=True,
        )

    holdout = protocol.holdout
    print("=== Slice 9: frozen-model temporal holdout protocol ===")
    print("No new features. Holdout is not scored in this script.")
    print()
    print("--- Holdout inventory (later matches only) ---")
    print(f"canonical/model-ready N={holdout.n}")
    print(f"canonical/model-ready start={_fmt(holdout.start)}")
    print(f"canonical/model-ready end={_fmt(holdout.end)}")
    print(
        f"canonical matches-view later N={protocol.canonical_later.n}  "
        f"start={_fmt(protocol.canonical_later.start)}  "
        f"end={_fmt(protocol.canonical_later.end)}"
    )
    print(
        "inspected later T1 main event (not in corpus, not used for tuning): "
        f"{INSPECTED_LATER_T1_MAIN_EVENT_NAME} "
        f"league_id={INSPECTED_LATER_T1_MAIN_EVENT_LEAGUE_ID} "
        f"N={INSPECTED_LATER_T1_MAIN_EVENT_N} "
        f"start={INSPECTED_LATER_T1_MAIN_EVENT_START.isoformat()} "
        f"end={INSPECTED_LATER_T1_MAIN_EVENT_END.isoformat()}"
    )
    print()
    print("--- Recorded frozen specification ---")
    print(
        f"candidate={protocol.candidate_spec.name}  "
        f"({protocol.candidate_spec.label}, "
        f"{len(protocol.candidate_spec.feature_columns)} columns)"
    )
    print(
        f"reference={protocol.reference_spec.name}  "
        f"({protocol.reference_spec.label}, "
        f"{len(protocol.reference_spec.feature_columns)} columns)"
    )
    print(
        "preprocessing=train-only median impute + missingness indicators + "
        f"StandardScaler ({protocol.preprocessing_spec})"
    )
    print(f"C grid (VAL only)={protocol.regularization_candidates}")
    print("gates/interactions: not part of the frozen specification")
    print()
    print("--- Recorded development / OOS boundary ---")
    print(f"frozen development_end={protocol.development_end.isoformat()}")
    print(f"recorded constant={FROZEN_DEVELOPMENT_END.isoformat()}")
    print(
        f"development N={protocol.n_development}  "
        f"(census {FROZEN_DEVELOPMENT_MATCH_COUNT})  "
        f"start={protocol.development_start.isoformat()}"
    )
    print(
        f"development OOS N={protocol.n_development_oos}  "
        f"(census {FROZEN_DEVELOPMENT_OOS_MATCH_COUNT})"
    )
    print(f"slice 8 post-draft rows={protocol.n_slice8_post_draft_matches}")
    print(f"evaluated={protocol.evaluated}")
    print()
    print(
        "Holdout rule: start_time > development_end; equal timestamps at "
        "the boundary stay in development. Nested C / preprocessing fit "
        "on development TRAIN/VAL only. Later matches must not re-enter "
        "Slice 7/8 walk-forward."
    )
    if holdout.n == 0:
        print(
            "Scoring blocked: canonical holdout N=0. Ingest later T1/T2 "
            "main-event matches after the frozen boundary before evaluating."
        )
    else:
        print(
            "Canonical later matches are present. This script still does "
            "not score them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
