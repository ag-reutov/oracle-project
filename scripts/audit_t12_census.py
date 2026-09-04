"""Automated T1/T2 main-event census audit (Liquipedia vs registry vs DB).

Compares the intended Liquipedia Tier 1 / Tier 2 main-event universe
(2024-01-01 through a configurable end, default 2026-09-04) against the
project's registry (`config/leagues.yaml`) and the canonical warehouse
(`matches` / `stratz_raw_matches` / `match_classifications`).

The intended universe comes from `config/liquipedia_events_2024plus.yaml`
(Liquipedia T1/T2 events) plus `config/event_match_assignments.yaml`
(event-level matches inside shared STRATZ leagues, e.g. the ACL 2025
finals inside T3 league 17875).

Output sections:
  - intended in-window events  (with per-event resolution status)
  - missing registry events    (intended event with no in_scope league row)
  - unresolved STRATZ ids      (intended event with no STRATZ league_id)
  - events with zero matches   (intended event, in registry, no canonical rows)
  - raw-only matches           (in `stratz_raw_matches`, not `matches`)
  - tier / classification      (event assignments applied)
  - unexpected registry events (in DB registry, not in intended manifest)
  - coverage                   (canonical / intended, by tier and year)

Exit status is non-zero when any in-window intended T1/T2 event is
absent or unresolved: no resolvable STRATZ id, no in_scope registry row,
or zero canonical matches. A registry row marked COMPLETE with zero
matches never proves completeness on its own; the audit re-checks the
canonical `matches` count directly.

Reconciliation against OpenDota (independent Valve-match source) is a
soft, diagnostic layer: enable it with `--opendota`. OpenDota has known
enumeration gaps, so OpenDota-missing matches are reported but never
fail the audit.

Usage:
    uv run python scripts/audit_t12_census.py
    uv run python scripts/audit_t12_census.py --end 2026-09-04 --opendota
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import quote

import httpx
import yaml
from sqlalchemy import Engine, select

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_leagues_yaml_from_liquipedia import COVERED_BY, STRATZ_ID_MAP

from dota_predictor.storage.engine import get_engine
from dota_predictor.storage.schema import (
    INGESTION_LEAGUES,
    LEAGUES,
    MATCH_CLASSIFICATIONS,
    MATCHES,
    STRATZ_RAW_MATCHES,
)
from dota_predictor.utils.env import load_project_env

MANIFEST_PATH = REPO_ROOT / "config" / "liquipedia_events_2024plus.yaml"
LEAGUES_YAML_PATH = REPO_ROOT / "config" / "leagues.yaml"

DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = date(2026, 9, 4)

OPENDOTA_EXPLORER_URL = "https://api.opendota.com/api/explorer"
OPENDOTA_USER_AGENT = "dota-predictor-census-audit/0.1"


def _load_manifest() -> list[dict]:
    raw = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    return list(raw.get("events") or [])


def _load_registry() -> dict[int, dict]:
    raw = yaml.safe_load(LEAGUES_YAML_PATH.read_text(encoding="utf-8")) or {}
    return {int(e["league_id"]): e for e in (raw.get("leagues") or [])}


def _resolve_league_id(event: dict) -> int | None:
    if event.get("league_id"):
        return int(event["league_id"])
    name = event["liquipedia_name"]
    # Covered events (e.g. 1win Summer/Fall) resolve to the umbrella event.
    if event.get("covered_by"):
        name = event["covered_by"]
    if name in COVERED_BY:
        name = COVERED_BY[name]
    mapped = STRATZ_ID_MAP.get(name)
    return int(mapped[0]) if mapped else None


def _event_window(event: dict) -> tuple[date | None, date | None]:
    start = event.get("start_date")
    end = event.get("end_date")
    return (start, end) if start and end else (None, None)


def _in_window(event: dict, *, start: date, end: date) -> bool:
    """An event is in-window if its main-event window overlaps [start, end].

    Events without explicit dates use the calendar year as the window
    (Jan 1 .. Dec 31 of `year`), which is exact for the 2024-2026
    universe except for boundary-year events, which carry dates.
    """
    s, e = _event_window(event)
    if s is None:
        s = date(int(event["year"]), 1, 1)
    if e is None:
        e = date(int(event["year"]), 12, 31)
    return s <= end and e >= start


def _window_bounds(
    start: date | None, end: date | None
) -> tuple[datetime, datetime]:
    s = (
        datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        if start is not None
        else datetime.min.replace(tzinfo=UTC)
    )
    e = (
        datetime.combine(end, datetime.max.time(), tzinfo=UTC)
        if end is not None
        else datetime.max.replace(tzinfo=UTC)
    )
    return s, e


def _opendota_match_ids(
    league_id: int, *, start: date | None, end: date | None, client: httpx.Client
) -> set[int]:
    s, e = _window_bounds(start, end)
    clauses = [f"leagueid = {int(league_id)}"]
    if start is not None:
        clauses.append(f"start_time >= {int(s.timestamp())}")
    if end is not None:
        clauses.append(f"start_time <= {int(e.timestamp())}")
    sql = "SELECT match_id FROM matches WHERE " + " AND ".join(clauses)
    url = f"{OPENDOTA_EXPLORER_URL}?sql={quote(sql)}"
    resp = client.get(url)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("err") or payload.get("error"):
        raise RuntimeError(
            f"OpenDota explorer error for league {league_id}: "
            f"{payload.get('err') or payload.get('error')}"
        )
    return {int(row["match_id"]) for row in (payload.get("rows") or [])}


def _db_state(engine: Engine) -> dict[str, object]:
    with engine.connect() as conn:
        league_rows = conn.execute(
            select(
                LEAGUES.c.league_id,
                LEAGUES.c.name,
                LEAGUES.c.liquipedia_tier,
                LEAGUES.c.in_scope,
            )
        ).all()
        allowlisted = {
            int(r[0]) for r in conn.execute(select(INGESTION_LEAGUES.c.league_id))
        }
        match_rows = conn.execute(
            select(
                MATCHES.c.match_id,
                MATCHES.c.league_id,
                MATCHES.c.start_time,
                MATCHES.c.draft_complete,
            )
        ).all()
        raw_rows = conn.execute(
            select(STRATZ_RAW_MATCHES.c.match_id, STRATZ_RAW_MATCHES.c.league_id)
        ).all()
        class_rows = conn.execute(
            select(
                MATCH_CLASSIFICATIONS.c.match_id,
                MATCH_CLASSIFICATIONS.c.liquipedia_event,
                MATCH_CLASSIFICATIONS.c.liquipedia_tier,
            )
        ).all()
    return {
        "leagues": {int(r.league_id): r for r in league_rows},
        "allowlisted": allowlisted,
        "matches": match_rows,
        "raw": raw_rows,
        "classifications": {int(r.match_id): r for r in class_rows},
    }


def _event_canonical_ids(
    event: dict,
    *,
    league_id: int | None,
    db: dict[str, object],
) -> set[int]:
    """Canonical match ids belonging to this event (windowed or league-wide)."""
    if league_id is None:
        return set()
    start, end = _event_window(event)
    s, e = _window_bounds(start, end)
    ids = {
        int(r.match_id)
        for r in db["matches"]
        if int(r.league_id) == league_id and s <= r.start_time <= e
    }
    # Shared-league events additionally count explicit classifications.
    assigned = {
        int(r.match_id)
        for r in db["classifications"].values()
        if r.liquipedia_event == event["liquipedia_name"]
    }
    ids |= assigned
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument(
        "--opendota", action="store_true", help="Run OpenDota reconciliation (network)."
    )
    args = parser.parse_args()

    load_project_env(REPO_ROOT)
    engine = get_engine()

    events = _load_manifest()
    registry = _load_registry()
    db = _db_state(engine)
    db_leagues = db["leagues"]
    classifications = db["classifications"]

    intended = [e for e in events if e["liquipedia_tier"] in ("T1", "T2")]
    in_window = [e for e in intended if _in_window(e, start=args.start, end=args.end)]

    print("=== T1/T2 main-event census audit ===")
    print(f"window: {args.start} .. {args.end}")
    print(f"intended T1/T2 events: {len(intended)}  (in-window: {len(in_window)})")

    fail = False
    rows: list[tuple[str, dict, int | None, int, int, list[str]]] = []

    for event in in_window:
        name = event["liquipedia_name"]
        tier = event["liquipedia_tier"]
        league_id = _resolve_league_id(event)
        problems: list[str] = []

        if league_id is None:
            problems.append("unresolved STRATZ id")
        elif league_id not in registry or not registry[league_id].get("in_scope"):
            problems.append("missing/in_scope=false registry row")
        elif league_id not in db_leagues:
            problems.append("missing DB league row")
        elif league_id not in db["allowlisted"]:
            problems.append("not allowlisted")

        ids = _event_canonical_ids(event, league_id=league_id, db=db)
        if not ids:
            problems.append("zero canonical matches")

        # Raw-only matches: raw rows for the event's league that have no
        # canonical row. For windowed events this is approximate (raw
        # payload start times are not joined here), so it is diagnostic.
        raw_only: set[int] = set()
        if league_id is not None:
            canon_league = {
                int(r.match_id)
                for r in db["matches"]
                if int(r.league_id) == league_id
            }
            raw_only = {
                int(r.match_id)
                for r in db["raw"]
                if int(r.league_id) == league_id and r.match_id not in canon_league
            }

        status = "FAIL" if problems else "OK"
        if problems:
            fail = True
        rows.append((status, event, league_id, len(ids), len(raw_only), problems))

    print("\n--- intended in-window events ---")
    for status, event, league_id, n, raw_only_n, problems in rows:
        print(
            f"{status}\t{event['liquipedia_tier']}\t{event['liquipedia_name']}\t"
            f"league={league_id}\tmatches={n}"
            + (f"\traw_only={raw_only_n}" if raw_only_n else "")
            + (f"\t{' '.join(problems)}" if problems else "")
        )

    # OpenDota reconciliation (soft diagnostic).
    if args.opendota:
        print("\n--- OpenDota reconciliation (diagnostic) ---")
        client = httpx.Client(timeout=60, headers={"User-Agent": OPENDOTA_USER_AGENT})
        try:
            for event in in_window:
                name = event["liquipedia_name"]
                league_id = _resolve_league_id(event)
                if league_id is None:
                    continue
                start, end = _event_window(event)
                try:
                    od = _opendota_match_ids(
                        league_id, start=start, end=end, client=client
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name}: OpenDota error: {exc}")
                    time.sleep(0.6)
                    continue
                canon = _event_canonical_ids(event, league_id=league_id, db=db)
                missing = sorted(od - canon)
                extra = sorted(canon - od)
                if missing or extra:
                    print(
                        f"  {name}: OD={len(od)} DB={len(canon)} "
                        f"OD_only={len(missing)} DB_only={len(extra)}"
                    )
                    if missing:
                        print(f"    OD-only (missing from canonical): {missing[:20]}")
                    if extra:
                        print(f"    DB-only (OpenDota gap): {extra[:20]}")
                else:
                    print(f"  {name}: OD={len(od)} DB={len(canon)} match")
                time.sleep(0.4)
        finally:
            client.close()

    # Classification / tier report.
    print("\n--- tier / classification ---")
    if classifications:
        for event in in_window:
            assigned = [
                r
                for r in classifications.values()
                if r.liquipedia_event == event["liquipedia_name"]
            ]
            if assigned:
                print(
                    f"  {event['liquipedia_name']}: {len(assigned)} matches "
                    f"classified {assigned[0].liquipedia_tier} via event assignment"
                )
    else:
        print("  (no event-level match classifications)")

    # Registry events not in the intended manifest (possible reclassification).
    intended_ids = {_resolve_league_id(e) for e in in_window}
    intended_ids.discard(None)
    unexpected: list[tuple[int, str, str, int]] = []
    for lid, row in sorted(db_leagues.items()):
        if not row.in_scope or row.liquipedia_tier not in ("T1", "T2"):
            continue
        if lid in intended_ids:
            continue
        n = sum(1 for r in db["matches"] if int(r.league_id) == lid)
        if n:
            unexpected.append((lid, row.name, row.liquipedia_tier, n))
    if unexpected:
        print("\n--- registry events not in intended manifest (possible reclassification) ---")
        for lid, name, tier, n in unexpected:
            print(f"  league {lid} {name} [{tier}] {n} matches")

    # Coverage by tier and year.
    print("\n--- coverage ---")
    by_tier: Counter[str] = Counter()
    by_year: Counter[str] = Counter()
    for status, event, _lid, n, _raw_n, _problems in rows:
        by_tier[event["liquipedia_tier"]] += n
        by_year[str(event["year"])] += n
    total = sum(by_tier.values())
    print(f"in-window intended events: {len(in_window)}")
    print(f"canonical in-window matches (intended events): {total}")
    for tier in ("T1", "T2"):
        print(f"  {tier}: {by_tier[tier]} matches")
    for year in ("2024", "2025", "2026"):
        print(f"  {year}: {by_year[year]} matches")
    print(
        f"event coverage: {len(rows) - sum(1 for s, *_ in rows if s == 'FAIL')}"
        f"/{len(in_window)} in-window events resolved"
    )

    if fail:
        print("\nRESULT: FAIL (an intended in-window event is absent or unresolved)")
        return 1
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())