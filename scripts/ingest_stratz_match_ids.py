"""CLI: ingest one league by STRATZ match(id) after discovering match IDs.

Proof path for STRATZ catalog-null leagues (e.g. BLAST Slam IV 17419).
Does not use `league(id) { matches }`. Discovery (OpenDota / team.matches)
supplies IDs only; payloads always come from STRATZ `match(id)`.

Usage:
    uv run python scripts/ingest_stratz_match_ids.py --league-id 17419 \\
        --seed-team-id 9247354
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlalchemy import func, select

from dota_predictor.ingestion.client import StratzClient
from dota_predictor.ingestion.config import (
    MissingStratzTokenError,
    load_ingestion_config,
)
from dota_predictor.ingestion.discovery import discover_league_match_ids
from dota_predictor.ingestion.errors import LeagueNotRegisteredError
from dota_predictor.ingestion.pipeline import ingest_matches_by_id
from dota_predictor.storage.engine import MissingDatabaseUrlError, get_engine
from dota_predictor.storage.ingestion_writer import get_league_match_date_window
from dota_predictor.storage.schema import DRAFT_EVENTS, MATCH_PLAYERS, MATCHES
from dota_predictor.utils.env import load_project_env

# Team Falcons — known BLAST Slam IV participant (audit seed).
DEFAULT_SEED_TEAM_ID = 9247354
BLAST_SLAM_IV_LEAGUE_ID = 17419


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league-id",
        type=int,
        default=BLAST_SLAM_IV_LEAGUE_ID,
        help="League to ingest (default: BLAST Slam IV 17419).",
    )
    parser.add_argument(
        "--seed-team-id",
        type=int,
        action="append",
        dest="seed_team_ids",
        help="STRATZ team id to seed team.matches BFS (repeatable). "
        f"Default: {DEFAULT_SEED_TEAM_ID} (Team Falcons).",
    )
    parser.add_argument("--skip-opendota", action="store_true")
    parser.add_argument("--skip-team-walk", action="store_true")
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

    seed_team_ids = args.seed_team_ids or [DEFAULT_SEED_TEAM_ID]

    with engine.connect() as conn:
        window_start, window_end = get_league_match_date_window(conn, args.league_id)

    with StratzClient(config) as client:
        discovery = discover_league_match_ids(
            args.league_id,
            team_fetcher=None if args.skip_team_walk else client,
            seed_team_ids=() if args.skip_team_walk else seed_team_ids,
            skip_opendota=args.skip_opendota,
            skip_team_walk=args.skip_team_walk,
            window_start=window_start,
            window_end=window_end,
        )
        print("=== discovery ===")
        for note in discovery.notes:
            print(f"  {note}")
        print(f"  unique match ids for ingest: {len(discovery.match_ids)}")
        print(f"  ids: {list(discovery.match_ids)}")

        try:
            result = ingest_matches_by_id(
                engine,
                client,
                args.league_id,
                discovery.match_ids,
                window_start=window_start,
                window_end=window_end,
            )
        except LeagueNotRegisteredError as exc:
            print(str(exc), file=sys.stderr)
            print(
                "Load config/leagues.yaml first: "
                "uv run python scripts/load_league_registry.py",
                file=sys.stderr,
            )
            return 1

    print("=== ingest ===")
    print(
        f"  league {result.league_id}: status={result.status} "
        f"fetch_complete={result.fetch_complete} "
        f"unique_ids={result.match_ids_unique} "
        f"fetch_ok={result.fetch_successes} fetch_fail={result.fetch_failures} "
        f"league_mismatches={result.league_id_mismatches} "
        f"skipped_raw={result.skipped_already_raw} "
        f"skipped_out_of_window={result.skipped_out_of_window} "
        f"raw_before={result.raw_rows_before} raw_after={result.raw_row_count} "
        f"canonical={result.canonical_row_count} "
        f"already_canonical={result.canonical_already_current} "
        f"map_failures={result.canonicalization_failures}"
        + (f" message={result.message!r}" if result.message else "")
    )
    print(f"  min_start_time={result.min_start_time}")
    print(f"  max_start_time={result.max_start_time}")
    print(f"  all_canonical_league_id_match={result.all_canonical_league_ids_match}")

    with engine.connect() as conn:
        match_id_sub = select(MATCHES.c.match_id).where(
            MATCHES.c.league_id == args.league_id
        )
        player_counts = conn.execute(
            select(MATCH_PLAYERS.c.match_id, func.count())
            .where(MATCH_PLAYERS.c.match_id.in_(match_id_sub))
            .group_by(MATCH_PLAYERS.c.match_id)
        ).all()
        draft_counts = conn.execute(
            select(DRAFT_EVENTS.c.match_id, func.count())
            .where(DRAFT_EVENTS.c.match_id.in_(match_id_sub))
            .group_by(DRAFT_EVENTS.c.match_id)
        ).all()

    ten_players = sum(1 for _mid, n in player_counts if int(n) == 10)
    player_rows = sum(int(n) for _mid, n in player_counts)
    draft_rows = sum(int(n) for _mid, n in draft_counts)
    draft_ns = [int(n) for _mid, n in draft_counts]

    print("=== completeness ===")
    print(f"  match_players rows={player_rows} matches_with_10_players={ten_players}")
    print(f"  draft_events rows={draft_rows}")
    if draft_ns:
        print(f"  draft_events per match: min={min(draft_ns)} max={max(draft_ns)}")

    if result.status == "ERROR":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
