"""Sync `config/team_organizations.yaml` into the organization identity tables.

`config/team_organizations.yaml` is the version-controlled source of truth
for curated organization mappings (raw STRATZ `team_id` -> real-world
organization). This script is the only intended writer of `organizations`
and `team_organization_memberships` -- do not edit those tables by hand.

Behavior:

1. Validate the config (explicit organization_id, name, de-duplicated
   team_ids, no team id in two organizations).
2. Require every configured `team_id` to already exist in the `teams`
   registry -- a stale curation entry fails loudly, it is never silently
   skipped.
3. Upsert organizations and memberships (idempotent), and remove
   memberships whose team id is no longer in the config. Organizations
   are never deleted (they stay for audit, like `leagues`).

Usage:
    uv run python scripts/load_team_organizations.py [--config config/team_organizations.yaml] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import Engine, select

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dota_predictor.data.team_identity import (
    load_team_organizations_config,
    sync_team_organizations,
)
from dota_predictor.storage.engine import get_engine
from dota_predictor.storage.schema import TEAM_ORGANIZATION_MEMBERSHIPS
from dota_predictor.utils.env import load_project_env

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "team_organizations.yaml"


def run(engine: Engine, config_path: Path, *, dry_run: bool = False) -> dict:
    entries = load_team_organizations_config(config_path)
    with engine.begin() as conn:
        if dry_run:
            conn.rollback()
            return {"entries": len(entries), "dry_run": True}
        sync_team_organizations(conn, entries)
        mapped_team_ids = {
            int(row.team_id)
            for row in conn.execute(
                select(TEAM_ORGANIZATION_MEMBERSHIPS.c.team_id)
            ).all()
        }
        return {
            "entries": len(entries),
            "mapped_team_ids": len(mapped_team_ids),
            "dry_run": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_project_env(REPO_ROOT)
    engine = get_engine()
    summary = run(engine, args.config, dry_run=args.dry_run)
    if summary["dry_run"]:
        print(
            f"[dry-run] would sync {summary['entries']} organizations "
            "(no changes committed)"
        )
        return 0
    print(f"Synced {summary['entries']} organizations from {args.config}")
    print(f"mapped raw team ids: {summary['mapped_team_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())