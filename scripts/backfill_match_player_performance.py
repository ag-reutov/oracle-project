"""Backfill observed STRATZ post-match box-score scalars onto match players.

Persisted raw JSON does not contain these fields. This script refetches
only the approved MatchPlayerType scalars, merges them onto existing
player objects by steamAccountId, and updates `match_players` columns.

Restart-safe and idempotent. Does not replace unrelated raw or canonical
fields. Does not coerce missing values to zero. Does not fetch TI 2026
unless `--include-post-development` is set.

Usage:
    uv run python scripts/backfill_match_player_performance.py --limit 5
    uv run python scripts/backfill_match_player_performance.py --match-id 7570622406
    uv run python scripts/backfill_match_player_performance.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from dota_predictor.ingestion.client import StratzClient
from dota_predictor.ingestion.config import (
    MissingStratzTokenError,
    load_ingestion_config,
)
from dota_predictor.ingestion.player_performance_backfill import (
    run_player_performance_backfill,
    summarize_player_performance_coverage,
)
from dota_predictor.storage.engine import MissingDatabaseUrlError, get_engine
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_until(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("until must include a timezone")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refetch STRATZ player box-score scalars into canonical match_players."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N selected match ids (debug).",
    )
    parser.add_argument(
        "--match-id",
        type=int,
        action="append",
        dest="match_ids",
        help="Restrict to these canonical match ids (repeatable).",
    )
    parser.add_argument(
        "--until",
        type=_parse_until,
        default=None,
        help="Inclusive start_time cutoff (ISO-8601 with timezone). "
        f"Default: frozen development end {FROZEN_DEVELOPMENT_END.isoformat()}.",
    )
    parser.add_argument(
        "--include-post-development",
        action="store_true",
        help="Also refetch matches after the frozen development end (includes TI 2026).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N matches (0 to disable).",
    )
    args = parser.parse_args()

    load_project_env(_project_root())
    try:
        engine = get_engine()
    except MissingDatabaseUrlError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        config = load_ingestion_config()
    except MissingStratzTokenError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    until = args.until
    if until is None and not args.include_post_development:
        until = FROZEN_DEVELOPMENT_END
    if args.include_post_development and args.until is not None:
        print("cannot combine --include-post-development with --until", file=sys.stderr)
        return 2

    with StratzClient(config) as client:
        result = run_player_performance_backfill(
            engine,
            client,
            limit=args.limit,
            until=until,
            match_ids=args.match_ids,
            progress_every=args.progress_every,
        )

    print(f"canonical matches considered: {result.canonical_matches}")
    print(
        f"until: {until.isoformat() if until is not None else 'none (all canonical)'}"
    )
    print(f"raw payloads already patched: {result.already_patched}")
    print(f"matches refetched: {result.fetched}")
    print(f"fetch failures: {result.fetch_failures}")
    print(f"canonical player rows updated: {result.canonical_rows_updated}")
    print(f"canonical matches missing raw payload: {result.missing_raw}")
    with engine.connect() as conn:
        coverage = summarize_player_performance_coverage(conn, until=until)
    print(
        f"coverage window matches={coverage['matches']} "
        f"player_rows={coverage['player_rows']}"
    )
    for column, stats in coverage["columns"].items():
        rate = stats["null_rate"]
        rate_text = "n/a" if rate is None else f"{100 * rate:.2f}%"
        print(
            f"  {column}: nulls={stats['nulls']} ({rate_text}) "
            f"min={stats['min']} max={stats['max']}"
        )
    if result.fetch_failures or result.missing_raw:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
