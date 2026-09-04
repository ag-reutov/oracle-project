"""Backfill / repair the canonical player-identity registry.

Re-asserts that every `player_id` referenced by canonical `match_players`
has a row in the `players` identity registry (see
`dota_predictor.data.player_identity.run_backfill`). This is the
population/repair path for the Slice 2 identity foundation:

* Idempotent and deterministic: re-running adds nothing once the registry
  is complete, and never deletes orphan registry ids (see `storage.schema`
  module docstring for why orphan registry rows are intentionally kept).
* Reads only `match_players`/`players` and writes only to `players` -- the
  identity backfill never touches `matches` / `match_players`, so it cannot
  affect any canonical match fact, feature, or label.

The player-universe summary (`first_seen_at`, `last_seen_at`,
`match_count`, `display_name`) is a pure derivation over canonical facts
and is computed fresh via the `research.players` view / `fetch_player_universe`
-- it is never cached in Postgres, so there is nothing to backfill for it.

Usage:
    uv run python scripts/backfill_player_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dota_predictor.data.player_identity import run_backfill
from dota_predictor.storage.engine import get_engine
from dota_predictor.utils.env import load_project_env


def main() -> int:
    load_project_env(REPO_ROOT)
    engine = get_engine()
    summary = run_backfill(engine)
    print("Player identity backfill complete")
    print(f"players in registry: {summary['registry_player_count']}")
    print(
        f"player ids referenced by match_players: {summary['referenced_player_count']}"
    )
    print(f"player ids added by this run: {summary['added_player_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
