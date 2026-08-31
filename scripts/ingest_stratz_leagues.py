"""CLI for historical STRATZ league ingestion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dota_predictor.ingestion.client import StratzClient
from dota_predictor.ingestion.config import (
    MissingStratzTokenError,
    load_ingestion_config,
)
from dota_predictor.ingestion.errors import LeagueNotAllowlistedError
from dota_predictor.ingestion.pipeline import ingest_league, ingest_leagues
from dota_predictor.storage.engine import MissingDatabaseUrlError, get_engine
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest historical STRATZ matches for allowlisted leagues."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--league-id",
        type=int,
        help="Ingest one explicitly selected allowlisted league.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Ingest all leagues currently present in ingestion_leagues. "
        "Each league uses `leagues.fetch_mode` (`league` or `match_ids`); "
        "catalog-null leagues must be `match_ids` so `--all` never calls "
        "STRATZ `league(id)` for them.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    root = _project_root()
    load_project_env(root)

    try:
        config = load_ingestion_config()
        engine = get_engine()
    except (MissingStratzTokenError, MissingDatabaseUrlError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with StratzClient(config) as client:
        try:
            if args.all:
                results = ingest_leagues(engine, client, page_size=config.page_size)
            else:
                results = [
                    ingest_league(engine, client, args.league_id, page_size=config.page_size)
                ]
        except LeagueNotAllowlistedError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    for result in results:
        print(
            f"league {result.league_id}: status={result.status} "
            f"fetch_complete={result.fetch_complete} "
            f"matches_seen={result.matches_seen_count} "
            f"canonical_failures={result.canonicalization_failures}"
            + (f" message={result.message!r}" if result.message else "")
        )
        if result.status == "ERROR":
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
