"""Rebuild the Slice 6 `research.team_strength_state` derived table.

Idempotent and deterministic: re-running reads the same canonical match
facts (the sole source of truth) and replaces the derived table within a
single transaction (truncate + bulk insert), so a rebuild never leaves a
partially-populated table and never accumulates duplicates. Records the
build provenance / staleness marker in `research.team_strength_build`
(source corpus snapshot, count/extrema diagnostics, and a deterministic
SHA-256 `source_fingerprint`) and prints a summary.

The Elo mathematics are exactly the production `features.team_elo`
definition (initial rating 1500.0, K-factor 32.0, expected-score formula,
chronological replay with equal-`start_time` mutual blindness). This script
does not tune Elo parameters.

Usage:
    uv run python scripts/rebuild_team_strength.py
    uv run python scripts/rebuild_team_strength.py --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dota_predictor.data.team_strength import (
    audit_team_strength,
    check_freshness,
    rebuild_team_strength_state,
)
from dota_predictor.storage.engine import get_engine
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the Slice 6 research.team_strength_state derived table."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print the full team-strength audit census.",
    )
    return parser


def _default(value: object) -> object:
    if isinstance(value, dict):
        return {k: _default(v) for k, v in value.items()}
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = _project_root()
    load_project_env(root)

    engine = get_engine()
    summary = rebuild_team_strength_state(engine)
    freshness = check_freshness(engine)

    print("=== Team strength rebuild ===")
    print(f"source_match_count: {summary['source_match_count']}")
    print(f"source_skipped_matches: {summary['source_skipped_matches']}")
    print(f"states_written: {summary['states_written']}")
    print(f"source_min_start_time: {summary['source_min_start_time']}")
    print(f"source_max_start_time: {summary['source_max_start_time']}")
    print(f"source_fingerprint: {summary['source_fingerprint']}")
    print(f"fresh (stored == current fingerprint): {freshness['fresh']}")

    if args.verbose:
        print()
        print("=== Audit census ===")
        audit = audit_team_strength(engine)
        print(json.dumps(audit, indent=2, sort_keys=True, default=_default))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())