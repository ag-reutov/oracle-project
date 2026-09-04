"""Chronological Liquipedia T3 ingestion using the existing STRATZ pipeline.

Does not ingest T1/T2 leagues. Does not change fetch architecture:
``ingest_league`` / ``ingest_matches_by_id`` plus current rate limits.

Date window: 2024-01-01 inclusive through 2026-09-03 inclusive. Leagues
whose Valve ID also contains out-of-window matches are fetched via
``match_ids`` with an OpenDota ID list restricted to that window.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import yaml
from sqlalchemy import Engine, func, select

from dota_predictor.data.canonical_schema import EXPLICIT_DOTA_POSITIONS, DraftAction
from dota_predictor.ingestion.client import StratzClient
from dota_predictor.ingestion.config import (
    MissingStratzTokenError,
    load_ingestion_config,
)
from dota_predictor.ingestion.discovery import (
    OPENDOTA_EXPLORER_URL,
    OPENDOTA_USER_AGENT,
)
from dota_predictor.ingestion.errors import StratzPermanentError
from dota_predictor.ingestion.pipeline import ingest_league
from dota_predictor.storage.engine import MissingDatabaseUrlError, get_engine
from dota_predictor.storage.schema import (
    DRAFT_EVENTS,
    LEAGUE_FETCH_MODE_MATCH_IDS,
    LEAGUE_INGESTION_STATE,
    MATCH_INGESTION_ERRORS,
    MATCH_PLAYERS,
    MATCHES,
    STRATZ_RAW_MATCHES,
)
from dota_predictor.utils.env import load_project_env

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAGUES_YAML = REPO_ROOT / "config" / "leagues.yaml"
PROGRESS_PATH = REPO_ROOT / "data" / "interim" / "tier3_ingest_progress.json"
CANDIDATES_PATH = REPO_ROOT / "data" / "interim" / "tier3_candidates.json"

WINDOW_START = datetime(2024, 1, 1, tzinfo=UTC)
WINDOW_END_EXCLUSIVE = datetime(2026, 9, 4, tzinfo=UTC)
WINDOW_START_TS = int(WINDOW_START.timestamp())
WINDOW_END_TS = int(WINDOW_END_EXCLUSIVE.timestamp())

SKIPPED_ALREADY_T2 = {
    16427: "1win Series Dota 2 Punch — Valve ID already registered as T2 (1WIN SERIES DOTA 2); T3 label not applied",
    17622: "AsiaPro League Season 2 — Valve ID already registered as T2 (AsiaPro League S2); T3 label not applied",
}
SKIPPED_NO_LEAGUE_ID = {
    "Own Code 2024": "Liquipedia wikitext has no |leagueid=; no usable STRATZ/OpenDota league id",
}
# Valve IDs known to extend outside 2024-01-01..2026-09-03 (reused or still running).
KNOWN_WINDOW_IDS = {11845, 18865, 19944}

logger = logging.getLogger(__name__)


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)[:10]
    return date.fromisoformat(text)


def league_window(entry: dict[str, Any]) -> tuple[datetime, datetime]:
    """Clamp this registry row's start/end to the global T3 expansion window.

    Shared Valve IDs must not ingest the whole 2024–2026 history of that id.
    Umbrella rows already span their member events.
    """
    start = _as_date(entry.get("start_date"))
    end = _as_date(entry.get("end_date"))
    start_dt = (
        datetime(start.year, start.month, start.day, tzinfo=UTC) if start else WINDOW_START
    )
    end_excl = (
        datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
        if end
        else WINDOW_END_EXCLUSIVE
    )
    start_dt = max(start_dt, WINDOW_START)
    end_excl = min(end_excl, WINDOW_END_EXCLUSIVE)
    if end_excl <= start_dt:
        end_excl = start_dt + timedelta(days=1)
    return start_dt, end_excl


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def load_t3_entries() -> list[dict[str, Any]]:
    raw = yaml.safe_load(LEAGUES_YAML.read_text(encoding="utf-8")) or {}
    entries = [
        entry
        for entry in (raw.get("leagues") or [])
        if entry.get("liquipedia_tier") == "T3" and entry.get("in_scope")
    ]
    entries.sort(
        key=lambda e: (
            str(e.get("start_date") or "9999-12-31"),
            int(e["league_id"]),
        )
    )
    return entries


def opendota_explorer(sql: str, *, attempts: int = 2) -> list[dict[str, Any]]:
    url = f"{OPENDOTA_EXPLORER_URL}?sql={quote(sql)}"
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(
                headers={"User-Agent": OPENDOTA_USER_AGENT}, timeout=120.0
            ) as client:
                response = client.get(url)
                if response.status_code in {429, 502, 503, 504, 520, 522, 524}:
                    raise httpx.HTTPStatusError(
                        f"OpenDota {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_exc = exc
            sleep_s = min(30.0, 2.0 ** attempt)
            logger.warning("OpenDota explorer attempt %s failed (%s); retry in %.1ss", attempt, exc, sleep_s)
            time.sleep(sleep_s)
            continue
        if payload.get("err") or payload.get("error"):
            raise RuntimeError(
                f"OpenDota explorer error: {payload.get('err') or payload.get('error')}"
            )
        return list(payload.get("rows") or [])
    assert last_exc is not None
    raise last_exc


def opendota_stats_for_ids(league_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not league_ids:
        return {}
    out: dict[int, dict[str, Any]] = {}
    chunk_size = 15
    for offset in range(0, len(league_ids), chunk_size):
        chunk = league_ids[offset : offset + chunk_size]
        id_list = ", ".join(str(int(i)) for i in chunk)
        sql = (
            "SELECT leagueid, count(*) AS n, "
            "min(start_time) AS min_ts, max(start_time) AS max_ts, "
            f"count(*) FILTER (WHERE start_time >= {WINDOW_START_TS} "
            f"AND start_time < {WINDOW_END_TS}) AS in_window "
            f"FROM matches WHERE leagueid IN ({id_list}) GROUP BY leagueid"
        )
        rows = opendota_explorer(sql)
        for row in rows:
            league_id = int(row["leagueid"])
            min_ts = int(row["min_ts"]) if row["min_ts"] is not None else None
            max_ts = int(row["max_ts"]) if row["max_ts"] is not None else None
            out[league_id] = {
                "opendota_matches": int(row["n"]),
                "opendota_in_window": int(row["in_window"]),
                "opendota_min_ts": min_ts,
                "opendota_max_ts": max_ts,
                "opendota_min": (
                    datetime.fromtimestamp(min_ts, tz=UTC).isoformat() if min_ts else None
                ),
                "opendota_max": (
                    datetime.fromtimestamp(max_ts, tz=UTC).isoformat() if max_ts else None
                ),
                "has_pre_2024": bool(min_ts is not None and min_ts < WINDOW_START_TS),
                "has_after_cutoff": bool(max_ts is not None and max_ts >= WINDOW_END_TS),
            }
    return out


def opendota_window_match_ids(
    league_id: int, start_ts: int, end_ts: int
) -> tuple[int, ...]:
    sql = (
        "SELECT match_id FROM matches "
        f"WHERE leagueid = {int(league_id)} "
        f"AND start_time >= {int(start_ts)} "
        f"AND start_time < {int(end_ts)} "
        "ORDER BY match_id"
    )
    return tuple(int(row["match_id"]) for row in opendota_explorer(sql))


def collect_stratz_window_ids(
    client: StratzClient, league_id: int, start_ts: int, end_ts: int
) -> tuple[int, ...]:
    """Page STRATZ league(id).matches and keep IDs inside [start_ts, end_ts)."""
    skip = 0
    page_size = 100
    ids: list[int] = []
    while True:
        page = client.fetch_league_matches_page(league_id, skip=skip, take=page_size)
        if not page:
            break
        for match in page:
            match_id = match.get("id")
            ts = match.get("startDateTime")
            if match_id is None or ts is None:
                continue
            ts_i = int(ts)
            if start_ts <= ts_i < end_ts:
                ids.append(int(match_id))
        last_ts = page[-1].get("startDateTime")
        if len(page) < page_size:
            break
        if last_ts is not None and int(last_ts) < start_ts:
            break
        skip += page_size
    return tuple(dict.fromkeys(ids))


def resolve_window_match_ids(
    client: StratzClient, league_id: int, start_ts: int, end_ts: int
) -> tuple[int, ...]:
    try:
        ids = opendota_window_match_ids(league_id, start_ts, end_ts)
        if ids:
            return ids
    except (httpx.HTTPError, RuntimeError, json.JSONDecodeError, TimeoutError) as extra:
        logger.warning("OpenDota window IDs failed for %s: %s", league_id, extra)
    try:
        return collect_stratz_window_ids(client, league_id, start_ts, end_ts)
    except StratzPermanentError as extra:
        logger.warning("STRATZ window ID paging failed for %s: %s", league_id, extra)
        return ()


def trim_out_of_window_matches(
    engine: Engine,
    league_id: int,
    start_dt: datetime,
    end_excl: datetime,
) -> dict[str, Any]:
    """Remove canonical/raw rows outside this league row's date window."""
    with engine.begin() as conn:
        out_ids = [
            int(match_id)
            for match_id in conn.execute(
                select(MATCHES.c.match_id).where(
                    MATCHES.c.league_id == league_id,
                    (MATCHES.c.start_time < start_dt)
                    | (MATCHES.c.start_time >= end_excl),
                )
            ).scalars()
        ]
        if not out_ids:
            return {"trimmed": 0, "match_ids_sample": []}
        conn.execute(MATCHES.delete().where(MATCHES.c.match_id.in_(out_ids)))
        conn.execute(
            STRATZ_RAW_MATCHES.delete().where(STRATZ_RAW_MATCHES.c.match_id.in_(out_ids))
        )
    logger.warning(
        "Trimmed %s out-of-window matches from T3 league %s (sample %s)",
        len(out_ids),
        league_id,
        out_ids[:10],
    )
    return {"trimmed": len(out_ids), "match_ids_sample": out_ids[:20]}


def probe_stratz_catalog(client: StratzClient, league_id: int) -> str:
    try:
        page = client.fetch_league_matches_page(league_id, skip=0, take=1)
    except StratzPermanentError as exc:
        message = str(exc).lower()
        if "not found" in message:
            return "catalog_null"
        return f"error:{exc}"
    if not page:
        return "empty"
    return "ok"


def canonical_counts(engine: Engine, league_ids: list[int]) -> dict[int, int]:
    if not league_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            select(MATCHES.c.league_id, func.count())
            .where(MATCHES.c.league_id.in_(league_ids))
            .group_by(MATCHES.c.league_id)
        ).all()
    return {int(league_id): int(n) for league_id, n in rows}


def verify_league(engine: Engine, league_id: int) -> dict[str, Any]:
    with engine.connect() as conn:
        raw_n = int(
            conn.execute(
                select(func.count()).select_from(STRATZ_RAW_MATCHES).where(
                    STRATZ_RAW_MATCHES.c.league_id == league_id
                )
            ).scalar_one()
        )
        match_n = int(
            conn.execute(
                select(func.count()).select_from(MATCHES).where(
                    MATCHES.c.league_id == league_id
                )
            ).scalar_one()
        )
        player_n = int(
            conn.execute(
                select(func.count())
                .select_from(MATCH_PLAYERS)
                .join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
                .where(MATCHES.c.league_id == league_id)
            ).scalar_one()
        )
        draft_n = int(
            conn.execute(
                select(func.count())
                .select_from(DRAFT_EVENTS)
                .join(MATCHES, MATCHES.c.match_id == DRAFT_EVENTS.c.match_id)
                .where(MATCHES.c.league_id == league_id)
            ).scalar_one()
        )
        not_10_players = int(
            conn.execute(
                select(func.count()).select_from(
                    select(MATCH_PLAYERS.c.match_id)
                    .join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
                    .where(MATCHES.c.league_id == league_id)
                    .group_by(MATCH_PLAYERS.c.match_id)
                    .having(func.count() != 10)
                    .subquery()
                )
            ).scalar_one()
        )
        ten_pick_matches = int(
            conn.execute(
                select(func.count()).select_from(
                    select(DRAFT_EVENTS.c.match_id)
                    .join(MATCHES, MATCHES.c.match_id == DRAFT_EVENTS.c.match_id)
                    .where(
                        MATCHES.c.league_id == league_id,
                        DRAFT_EVENTS.c.action == DraftAction.PICK,
                    )
                    .group_by(DRAFT_EVENTS.c.match_id)
                    .having(func.count() == 10)
                    .subquery()
                )
            ).scalar_one()
        )
        not_10_picks = match_n - ten_pick_matches
        dup_match_ids = int(
            conn.execute(
                select(func.count()).select_from(
                    select(MATCHES.c.match_id)
                    .where(MATCHES.c.league_id == league_id)
                    .group_by(MATCHES.c.match_id)
                    .having(func.count() > 1)
                    .subquery()
                )
            ).scalar_one()
        )
        dup_player_rows = int(
            conn.execute(
                select(func.count()).select_from(
                    select(MATCH_PLAYERS.c.match_id, MATCH_PLAYERS.c.player_id)
                    .join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
                    .where(MATCHES.c.league_id == league_id)
                    .group_by(MATCH_PLAYERS.c.match_id, MATCH_PLAYERS.c.player_id)
                    .having(func.count() > 1)
                    .subquery()
                )
            ).scalar_one()
        )
        dup_draft_seq = int(
            conn.execute(
                select(func.count()).select_from(
                    select(DRAFT_EVENTS.c.match_id, DRAFT_EVENTS.c.sequence)
                    .join(MATCHES, MATCHES.c.match_id == DRAFT_EVENTS.c.match_id)
                    .where(MATCHES.c.league_id == league_id)
                    .group_by(DRAFT_EVENTS.c.match_id, DRAFT_EVENTS.c.sequence)
                    .having(func.count() > 1)
                    .subquery()
                )
            ).scalar_one()
        )
        null_hero = int(
            conn.execute(
                select(func.count())
                .select_from(MATCH_PLAYERS)
                .join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
                .where(MATCHES.c.league_id == league_id, MATCH_PLAYERS.c.hero_id.is_(None))
            ).scalar_one()
        )
        explicit_n = int(
            conn.execute(
                select(func.count())
                .select_from(MATCH_PLAYERS)
                .join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
                .where(
                    MATCHES.c.league_id == league_id,
                    MATCH_PLAYERS.c.position.in_(tuple(EXPLICIT_DOTA_POSITIONS)),
                )
            ).scalar_one()
        )
        gv_n = int(
            conn.execute(
                select(func.count()).select_from(MATCHES).where(
                    MATCHES.c.league_id == league_id,
                    MATCHES.c.game_version_id.is_not(None),
                )
            ).scalar_one()
        )
        min_start, max_start = conn.execute(
            select(func.min(MATCHES.c.start_time), func.max(MATCHES.c.start_time)).where(
                MATCHES.c.league_id == league_id
            )
        ).one()
        error_n = int(
            conn.execute(
                select(func.count()).select_from(MATCH_INGESTION_ERRORS).where(
                    MATCH_INGESTION_ERRORS.c.league_id == league_id
                )
            ).scalar_one()
        )
        state = conn.execute(
            select(
                LEAGUE_INGESTION_STATE.c.status,
                LEAGUE_INGESTION_STATE.c.matches_seen_count,
                LEAGUE_INGESTION_STATE.c.last_error,
            ).where(LEAGUE_INGESTION_STATE.c.league_id == league_id)
        ).first()

    pre_2024 = min_start is not None and min_start < WINDOW_START
    after_cutoff = max_start is not None and max_start >= WINDOW_END_EXCLUSIVE
    anomalies: list[str] = []
    if not_10_players:
        anomalies.append(f"matches_without_10_players={not_10_players}")
    if not_10_picks:
        anomalies.append(f"matches_without_10_picks={not_10_picks}")
    if dup_match_ids:
        anomalies.append(f"duplicate_match_ids={dup_match_ids}")
    if dup_player_rows:
        anomalies.append(f"duplicate_player_rows={dup_player_rows}")
    if dup_draft_seq:
        anomalies.append(f"duplicate_draft_sequences={dup_draft_seq}")
    if null_hero:
        anomalies.append(f"null_hero_id={null_hero}")
    if pre_2024:
        anomalies.append(f"min_start_before_window={min_start.isoformat()}")
    if after_cutoff:
        anomalies.append(f"max_start_after_window={max_start.isoformat()}")
    if raw_n != match_n:
        anomalies.append(f"raw_canonical_gap raw={raw_n} canonical={match_n}")

    return {
        "raw_matches": raw_n,
        "canonical_matches": match_n,
        "match_players": player_n,
        "draft_events": draft_n,
        "matches_without_10_players": not_10_players,
        "matches_without_10_picks": not_10_picks,
        "duplicate_match_ids": dup_match_ids,
        "duplicate_player_rows": dup_player_rows,
        "duplicate_draft_sequences": dup_draft_seq,
        "null_hero_id": null_hero,
        "explicit_position_rows": explicit_n,
        "explicit_position_rate": (explicit_n / player_n) if player_n else None,
        "game_version_matches": gv_n,
        "game_version_rate": (gv_n / match_n) if match_n else None,
        "min_start_time": min_start.isoformat() if min_start else None,
        "max_start_time": max_start.isoformat() if max_start else None,
        "ingestion_errors": error_n,
        "state_status": None if state is None else state.status,
        "state_matches_seen": None if state is None else int(state.matches_seen_count),
        "state_last_error": None if state is None else state.last_error,
        "anomalies": anomalies,
    }


def set_fetch_mode_match_ids(league_id: int, extra_note: str) -> bool:
    text = LEAGUES_YAML.read_text(encoding="utf-8")
    marker = f"  - league_id: {league_id}\n"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"league_id {league_id} not found in {LEAGUES_YAML}")
    nxt = text.find("\n  - league_id:", start + len(marker))
    end = nxt + 1 if nxt >= 0 else len(text)
    block = text[start:end]
    changed = False
    if "fetch_mode:" not in block:
        block = block.replace(
            "    in_scope: true\n",
            "    in_scope: true\n    fetch_mode: match_ids\n",
            1,
        )
        changed = True
    if extra_note and extra_note not in block:
        if "    notes:" in block:
            block = block.replace(
                '    notes: "',
                f'    notes: "{extra_note} ',
                1,
            )
        else:
            if block.endswith("\n"):
                block = block[:-1] + f'\n    notes: "{extra_note}"\n'
            else:
                block = block + f'\n    notes: "{extra_note}"'
        changed = True
    if changed:
        LEAGUES_YAML.write_text(text[:start] + block + text[end:], encoding="utf-8")
    return changed


def sync_registry() -> None:
    subprocess.check_call(
        [sys.executable, str(REPO_ROOT / "scripts" / "load_league_registry.py")],
        cwd=REPO_ROOT,
    )


def load_progress() -> dict[str, Any]:
    if not PROGRESS_PATH.is_file():
        return {"leagues": {}}
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def save_progress(progress: dict[str, Any]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2, default=_jsonable) + "\n")


def propose_action(
    *,
    league_id: int,
    catalog: str,
    stats: dict[str, Any] | None,
    already_canonical: int,
) -> str:
    if league_id in KNOWN_WINDOW_IDS or league_id < 15000:
        return "match_ids_date_window"
    if already_canonical:
        return "inspect_existing_then_ingest_missing"
    if stats and (stats.get("has_pre_2024") or stats.get("has_after_cutoff")):
        return "match_ids_date_window"
    if catalog == "catalog_null":
        return "match_ids_catalog_null"
    if catalog == "empty" and stats and int(stats.get("opendota_in_window") or 0) > 0:
        return "match_ids_opendota_window"
    if catalog.startswith("error:"):
        return "probe_failed_retry_league"
    return "league"


def audit(
    engine: Engine,
    client: StratzClient | None,
    *,
    probe_stratz: bool,
) -> dict[str, Any]:
    entries = load_t3_entries()
    ids = [int(e["league_id"]) for e in entries] + list(SKIPPED_ALREADY_T2)
    logger.info("OpenDota date-range audit for %s T3 league ids", len(ids))
    try:
        odota = opendota_stats_for_ids(ids)
    except (httpx.HTTPError, RuntimeError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("OpenDota audit unavailable (%s); continuing without it", exc)
        odota = {}
    db_counts = canonical_counts(engine, ids)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        league_id = int(entry["league_id"])
        stats = odota.get(league_id)
        catalog = "not_probed"
        if probe_stratz:
            if client is None:
                raise RuntimeError("STRATZ client required for catalog probe")
            catalog = probe_stratz_catalog(client, league_id)
            logger.info("STRATZ catalog %s -> %s", league_id, catalog)
        action = propose_action(
            league_id=league_id,
            catalog=catalog,
            stats=stats,
            already_canonical=int(db_counts.get(league_id, 0)),
        )
        start = entry.get("start_date")
        end = entry.get("end_date")
        rows.append(
            {
                "league_id": league_id,
                "name": entry["name"],
                "tier": "T3",
                "date_range": f"{_jsonable(start)} .. {_jsonable(end)}",
                "already_registered": True,
                "canonical_matches": int(db_counts.get(league_id, 0)),
                "fetch_mode": entry.get("fetch_mode") or "league",
                "proposed_action": action,
                "stratz_catalog": catalog,
                "opendota": stats or {"opendota_matches": 0, "opendota_in_window": 0},
            }
        )

    skipped = []
    for league_id, reason in SKIPPED_ALREADY_T2.items():
        skipped.append(
            {
                "league_id": league_id,
                "name": None,
                "tier": "T2 (kept)",
                "already_registered": True,
                "canonical_matches": int(db_counts.get(league_id, 0)) if league_id in db_counts else None,
                "fetch_mode": None,
                "proposed_action": "skip_already_t2",
                "reason": reason,
            }
        )
    for name, reason in SKIPPED_NO_LEAGUE_ID.items():
        skipped.append(
            {
                "league_id": None,
                "name": name,
                "tier": "T3",
                "already_registered": False,
                "canonical_matches": 0,
                "fetch_mode": None,
                "proposed_action": "skip_no_leagueid",
                "reason": reason,
            }
        )

    payload = {
        "window_start": WINDOW_START.isoformat(),
        "window_end_inclusive": "2026-09-03",
        "t3_registry_leagues": len(entries),
        "candidates": rows,
        "skipped": skipped,
        "action_counts": dict(Counter(row["proposed_action"] for row in rows)),
    }
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n")
    print(f"Wrote {CANDIDATES_PATH} ({len(rows)} T3 leagues, {len(skipped)} skipped)")
    return payload


def _needs_match_ids(action: str) -> bool:
    return action in {
        "match_ids_date_window",
        "match_ids_catalog_null",
        "match_ids_opendota_window",
    }


def ingest_one(
    engine: Engine,
    client: StratzClient,
    entry: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    league_id = int(entry["league_id"])
    action = candidate["proposed_action"]
    fetch_mode = entry.get("fetch_mode") or "league"
    start_dt, end_excl = league_window(entry)
    start_ts, end_ts = int(start_dt.timestamp()), int(end_excl.timestamp())
    match_ids: tuple[int, ...] | None = None
    note = None
    if _needs_match_ids(action) or fetch_mode == LEAGUE_FETCH_MODE_MATCH_IDS:
        match_ids = resolve_window_match_ids(client, league_id, start_ts, end_ts)
        note = (
            "Date-window match_ids ingest 2024-01-01..2026-09-03 "
            f"({len(match_ids)} ids)."
        )
        if set_fetch_mode_match_ids(league_id, note):
            sync_registry()
            fetch_mode = LEAGUE_FETCH_MODE_MATCH_IDS
        else:
            fetch_mode = LEAGUE_FETCH_MODE_MATCH_IDS

    logger.info(
        "Ingesting T3 %s %s via %s (action=%s, match_ids=%s)",
        league_id,
        entry["name"],
        fetch_mode,
        action,
        None if match_ids is None else len(match_ids),
    )
    result = ingest_league(
        engine,
        client,
        league_id,
        match_ids=match_ids,
    )
    if (
        result.status == "ERROR"
        and result.message
        and "not found" in result.message.lower()
        and match_ids is None
    ):
        logger.warning("Catalog-null fallback for %s: %s", league_id, result.message)
        match_ids = resolve_window_match_ids(client, league_id, start_ts, end_ts)
        note = (
            "STRATZ league(id) null; ingest via match(id) after windowed "
            f"IDs ({len(match_ids)} ids)."
        )
        set_fetch_mode_match_ids(league_id, note)
        sync_registry()
        result = ingest_league(engine, client, league_id, match_ids=match_ids)
        fetch_mode = LEAGUE_FETCH_MODE_MATCH_IDS

    trimmed = trim_out_of_window_matches(engine, league_id, start_dt, end_excl)
    integrity = verify_league(engine, league_id)
    record = {
        "league_id": league_id,
        "name": entry["name"],
        "action": action,
        "fetch_mode": fetch_mode,
        "match_ids_supplied": None if match_ids is None else len(match_ids),
        "status": result.status,
        "fetch_complete": result.fetch_complete,
        "matches_seen_count": result.matches_seen_count,
        "canonicalization_failures": result.canonicalization_failures,
        "message": result.message,
        "trimmed_out_of_window": trimmed,
        "integrity": integrity,
    }
    if integrity["anomalies"]:
        logger.warning("T3 %s anomalies: %s", league_id, "; ".join(integrity["anomalies"]))
    return record


def run_ingest(engine: Engine, client: StratzClient, audit_payload: dict[str, Any]) -> None:
    progress = load_progress()
    by_id = {row["league_id"]: row for row in audit_payload["candidates"]}
    for entry in load_t3_entries():
        league_id = int(entry["league_id"])
        key = str(league_id)
        prior = progress.get("leagues", {}).get(key) or {}
        if prior.get("integrity") is not None:
            logger.info("Skipping already-attempted T3 %s", league_id)
            continue
        record = ingest_one(engine, client, entry, by_id[league_id])
        progress.setdefault("leagues", {})[key] = record
        save_progress(progress)
        print(
            f"league {league_id}: status={record['status']} "
            f"canonical={record['integrity']['canonical_matches']} "
            f"raw={record['integrity']['raw_matches']} "
            f"anomalies={record['integrity']['anomalies'] or '-'}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--skip-stratz-probe",
        action="store_true",
        help="OpenDota/DB audit only; do not call STRATZ league(id) during audit.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    load_project_env(REPO_ROOT)
    try:
        engine = get_engine()
        config = load_ingestion_config()
    except (MissingDatabaseUrlError, MissingStratzTokenError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    probe = not args.skip_stratz_probe
    with StratzClient(config) as client:
        payload = audit(engine, client if probe else None, probe_stratz=probe)
        if args.audit_only:
            return 0
        run_ingest(engine, client, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
