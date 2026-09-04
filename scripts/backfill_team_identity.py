"""Backfill the derived team-identity tables from local canonical/raw data.

Populates `team_aliases` and `team_tags` deterministically from data that
already exists locally -- NO API calls (see `data.team_identity`):

* `team_aliases`: every name observed per `team_id` in the immutable
  `matches.*_team_name_observed` columns, with first/last observation time
  and match-side count.
* `team_tags`: every STRATZ `tag` observed per `team_id` in existing
  `stratz_raw_matches` payloads, with first/last observation time (the
  payload's match `startDateTime`, falling back to `fetched_at`) and
  payload count. Only team ids present in the `teams` registry are
  recorded; raw payloads that reference a team id not in the registry are
  reported (not silently dropped from the report).

Idempotent: re-running recomputes the same rows via upsert and never
creates duplicates. Safe to run after any future ingest.

Usage:
    uv run python scripts/backfill_team_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dota_predictor.data.team_identity import run_backfill
from dota_predictor.storage.engine import get_engine
from dota_predictor.utils.env import load_project_env


def main() -> int:
    load_project_env(REPO_ROOT)
    engine = get_engine()
    summary = run_backfill(engine)
    print("Team identity backfill complete")
    print(f"teams in registry: {summary['registry_team_ids']}")
    print(f"team_aliases rows: {summary['alias_rows']}")
    print(f"team_tags rows: {summary['tag_rows']}")
    skipped = summary["skipped_raw_team_ids"]
    if skipped:
        print(
            f"WARNING: {len(skipped)} raw team id(s) are not in the teams "
            f"registry; their tags were not recorded: {skipped[:10]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())