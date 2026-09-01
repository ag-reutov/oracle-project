"""Backfill observed STRATZ position/lane/role onto canonical match players.

Persisted raw JSON does not contain these fields. This script refetches
only match-player parse labels, merges them onto existing player objects
by steamAccountId, and updates `match_players.position/lane/role`.

Restart-safe and idempotent. Does not replace unrelated raw or canonical
fields. Does not infer missing positions.

Usage:
    uv run python scripts/backfill_match_player_positions.py
    uv run python scripts/backfill_match_player_positions.py --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dota_predictor.ingestion.client import StratzClient
from dota_predictor.ingestion.config import (
    MissingStratzTokenError,
    load_ingestion_config,
)
from dota_predictor.ingestion.player_position_backfill import (
    run_player_position_backfill,
)
from dota_predictor.storage.engine import MissingDatabaseUrlError, get_engine
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refetch STRATZ position/lane/role into canonical match_players."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N canonical match ids (debug).",
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

    with StratzClient(config) as client:
        result = run_player_position_backfill(engine, client, limit=args.limit)

    print(f"canonical matches considered: {result.canonical_matches}")
    print(f"raw payloads already patched: {result.already_patched}")
    print(f"matches refetched: {result.fetched}")
    print(f"fetch failures: {result.fetch_failures}")
    print(f"canonical player rows updated: {result.canonical_rows_updated}")
    print(f"canonical matches missing raw payload: {result.missing_raw}")
    if result.fetch_failures or result.missing_raw:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
