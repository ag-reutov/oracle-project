"""Backfill `match_players.hero_id` from persisted STRATZ raw JSON.

Does not call STRATZ or OpenDota. Hero assignment comes from
`payload.players[].heroId` joined to canonical rows by
`(match_id, steamAccountId)`.

Intended as an idempotent companion to the Alembic migration that adds
the column. Safe to re-run: already-populated rows are overwritten with
the same source value.

Usage:
    uv run python scripts/backfill_match_player_heroes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

from dota_predictor.storage.engine import MissingDatabaseUrlError, get_engine
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    load_project_env(_project_root())
    try:
        engine = get_engine()
    except MissingDatabaseUrlError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                UPDATE match_players AS mp
                SET hero_id = src.hero_id
                FROM (
                    SELECT
                        r.match_id,
                        (player ->> 'steamAccountId')::bigint AS player_id,
                        (player ->> 'heroId')::integer AS hero_id
                    FROM stratz_raw_matches AS r,
                         jsonb_array_elements(r.payload -> 'players') AS player
                    WHERE jsonb_typeof(r.payload -> 'players') = 'array'
                ) AS src
                WHERE mp.match_id = src.match_id
                  AND mp.player_id = src.player_id
                """
            )
        )
        updated = result.rowcount
        remaining_null = conn.execute(
            text("SELECT COUNT(*) FROM match_players WHERE hero_id IS NULL")
        ).scalar_one()

    print(f"updated match_players rows: {updated}")
    print(f"remaining null hero_id: {remaining_null}")
    if remaining_null:
        print("Backfill left null hero_id values", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
