"""Sync `config/leagues.yaml` into the `leagues` and `ingestion_leagues` tables.

`config/leagues.yaml` is the version-controlled source of truth for league
curation (see that file's header comment). This script is the only
intended writer of `leagues`/`ingestion_leagues` -- do not edit those
tables by hand.

Behavior:

1. Upsert every entry in the YAML file into `leagues` (full registry,
   including `in_scope: false` rows, for audit purposes).
2. Insert any newly in-scope league into `ingestion_leagues` (the strict
   allowlist that raw/canonical/progress tables are gated on). `in_scope`
   does not select a fetch strategy; `fetch_mode` (`league` or
   `match_ids`) does.
3. If a league was previously in-scope and is no longer, attempt to
   remove it from `ingestion_leagues`. If matches have already been
   ingested for it, the `ON DELETE RESTRICT` foreign key will block this
   -- that failure is reported clearly rather than silently ignored,
   because un-scoping a league with existing data is a decision that
   needs an explicit follow-up (e.g. deleting those matches), not a
   silent cascade.

Usage:
    uv run python scripts/load_league_registry.py [--config config/leagues.yaml] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from sqlalchemy import Connection, Engine, select
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dota_predictor.storage.engine import get_engine
from dota_predictor.storage.schema import (
    INGESTION_LEAGUES,
    LEAGUES,
    LEAGUE_FETCH_MODES,
    LEAGUE_FETCH_MODE_LEAGUE,
    LIQUIPEDIA_TIERS,
)
from dota_predictor.utils.env import load_project_env

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "leagues.yaml"


def load_registry_entries(config_path: Path) -> list[dict]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    entries = raw.get("leagues") or []
    seen_ids: set[int] = set()
    for entry in entries:
        league_id = entry["league_id"]
        if league_id in seen_ids:
            raise ValueError(f"Duplicate league_id {league_id} in {config_path}")
        seen_ids.add(league_id)
        if entry["liquipedia_tier"] not in LIQUIPEDIA_TIERS:
            raise ValueError(
                f"league_id {league_id}: liquipedia_tier "
                f"{entry['liquipedia_tier']!r} not in {LIQUIPEDIA_TIERS}"
            )
        fetch_mode = entry.get("fetch_mode") or LEAGUE_FETCH_MODE_LEAGUE
        if fetch_mode not in LEAGUE_FETCH_MODES:
            raise ValueError(
                f"league_id {league_id}: fetch_mode {fetch_mode!r} not in "
                f"{LEAGUE_FETCH_MODES}"
            )
        entry["fetch_mode"] = fetch_mode
    return entries


def sync_leagues(conn: Connection, entries: list[dict]) -> None:
    for entry in entries:
        existing = conn.execute(
            select(LEAGUES.c.league_id).where(
                LEAGUES.c.league_id == entry["league_id"]
            )
        ).first()
        values = {
            "league_id": entry["league_id"],
            "name": entry["name"],
            "stratz_tier": entry.get("stratz_tier"),
            "liquipedia_tier": entry["liquipedia_tier"],
            "in_scope": bool(entry.get("in_scope", False)),
            "fetch_mode": entry.get("fetch_mode") or LEAGUE_FETCH_MODE_LEAGUE,
            "notes": entry.get("notes"),
            "source": entry.get("source"),
            "start_date": entry.get("start_date"),
            "end_date": entry.get("end_date"),
        }
        if existing is None:
            conn.execute(LEAGUES.insert().values(**values))
        else:
            conn.execute(
                LEAGUES.update()
                .where(LEAGUES.c.league_id == entry["league_id"])
                .values(**values)
            )


def sync_ingestion_leagues(conn: Connection, entries: list[dict]) -> list[int]:
    """Sync the allowlist. Returns league_ids that could not be removed."""
    target_in_scope = {e["league_id"] for e in entries if e.get("in_scope")}
    current_allowlisted = {
        row.league_id
        for row in conn.execute(select(INGESTION_LEAGUES.c.league_id))
    }

    to_add = target_in_scope - current_allowlisted
    for league_id in to_add:
        conn.execute(INGESTION_LEAGUES.insert().values(league_id=league_id))

    to_remove = current_allowlisted - target_in_scope
    blocked: list[int] = []
    for league_id in to_remove:
        savepoint = conn.begin_nested()
        try:
            conn.execute(
                INGESTION_LEAGUES.delete().where(
                    INGESTION_LEAGUES.c.league_id == league_id
                )
            )
            savepoint.commit()
        except IntegrityError:
            savepoint.rollback()
            blocked.append(league_id)
    return blocked


def run(engine: Engine, config_path: Path, *, dry_run: bool = False) -> None:
    entries = load_registry_entries(config_path)
    with engine.begin() as conn:
        sync_leagues(conn, entries)
        blocked = sync_ingestion_leagues(conn, entries)
        if dry_run:
            conn.rollback()
            print(f"[dry-run] would sync {len(entries)} leagues (no changes committed)")
            return
        print(f"Synced {len(entries)} leagues from {config_path}")
        if blocked:
            print(
                "WARNING: could not remove the following leagues from "
                "ingestion_leagues because matches already reference them "
                f"(un-scope manually if intended): {sorted(blocked)}",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_project_env(REPO_ROOT)
    engine = get_engine()
    run(engine, args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
