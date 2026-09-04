"""Apply event-level match classifications from config to the DB.

Reads `config/event_match_assignments.yaml` and populates the
`match_classifications` table: for each assignment (league_id + date
window), every canonical match in that league whose `start_time` falls
within the window is upserted with the assignment's event/tier. Matches
in the same league outside the window are untouched (they keep the
league's default `leagues.liquipedia_tier`).

The effective tier of a match is therefore
`coalesce(match_classifications.liquipedia_tier, leagues.liquipedia_tier)`.

Idempotent: re-running upserts the same rows and leaves unrelated rows
alone.

Usage:
    uv run python scripts/apply_event_classifications.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ASSIGNMENTS_PATH = REPO_ROOT / "config" / "event_match_assignments.yaml"

from dota_predictor.storage.engine import get_engine
from dota_predictor.storage.schema import MATCH_CLASSIFICATIONS, MATCHES
from dota_predictor.utils.env import load_project_env


def _load_assignments(config_path: Path = ASSIGNMENTS_PATH) -> list[dict]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return list(raw.get("assignments") or [])


def _window_bounds(start_date: object, end_date: object) -> tuple[datetime, datetime]:
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
    end = datetime.combine(end_date, datetime.max.time(), tzinfo=UTC)
    return start, end


def apply_assignments(
    engine: Engine, assignments: list[dict] | None = None
) -> dict[int, dict[str, object]]:
    """Upsert match classifications; return per-league summary."""
    resolved = assignments if assignments is not None else _load_assignments()
    report: dict[int, dict[str, object]] = {}

    for assignment in resolved:
        league_id = int(assignment["league_id"])
        event = assignment["liquipedia_event"]
        tier = assignment["liquipedia_tier"]
        source = assignment.get("source")
        start, end = _window_bounds(
            assignment["start_date"], assignment["end_date"]
        )
        with engine.connect() as conn:
            match_rows = conn.execute(
                select(MATCHES.c.match_id)
                .where(
                    MATCHES.c.league_id == league_id,
                    MATCHES.c.start_time >= start,
                    MATCHES.c.start_time <= end,
                )
                .order_by(MATCHES.c.match_id)
            ).all()
        match_ids = [int(row.match_id) for row in match_rows]

        with engine.begin() as conn:
            for match_id in match_ids:
                stmt = pg_insert(MATCH_CLASSIFICATIONS).values(
                    match_id=match_id,
                    liquipedia_event=event,
                    liquipedia_tier=tier,
                    source=source,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[MATCH_CLASSIFICATIONS.c.match_id],
                    set_={
                        "liquipedia_event": event,
                        "liquipedia_tier": tier,
                        "source": source,
                    },
                )
                conn.execute(stmt)

        report[league_id] = {
            "event": event,
            "tier": tier,
            "n_matches": len(match_ids),
            "window": f"{start.date()}..{end.date()}",
        }
    return report


def main() -> int:
    root = REPO_ROOT
    load_project_env(root)
    engine = get_engine()
    assignments = _load_assignments()
    if not assignments:
        print(f"No assignments found in {ASSIGNMENTS_PATH}", file=sys.stderr)
        return 1
    report = apply_assignments(engine, assignments)
    for league_id, info in sorted(report.items()):
        print(
            f"league {league_id}: {info['event']} -> {info['tier']} "
            f"({info['n_matches']} matches, window {info['window']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())