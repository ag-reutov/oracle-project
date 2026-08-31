"""Database writers for STRATZ ingestion progress and raw payloads."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from dota_predictor.data.stratz_mapping import CANONICAL_MAPPER_VERSION
from dota_predictor.ingestion.cursor import CursorState, cursor_to_dict
from dota_predictor.ingestion.errors import LeagueNotRegisteredError
from dota_predictor.storage.schema import (
    INGESTION_LEAGUES,
    LEAGUE_FETCH_MODE_LEAGUE,
    LEAGUE_INGESTION_STATE,
    LEAGUES,
    MATCH_INGESTION_ERRORS,
    MATCHES,
    STRATZ_RAW_MATCHES,
)

__all__ = [
    "ensure_league_allowlisted",
    "ensure_league_ingestion_state",
    "get_canonical_mapper_version",
    "get_league_fetch_mode",
    "get_persisted_match_ids_for_league",
    "insert_match_ingestion_error",
    "is_league_allowlisted",
    "league_canonicalization_complete",
    "list_allowlisted_league_ids",
    "list_raw_matches_for_league",
    "load_cursor_state",
    "persist_raw_page",
    "update_league_ingestion_state",
]


def is_league_allowlisted(conn: Connection, league_id: int) -> bool:
    row = conn.execute(
        select(INGESTION_LEAGUES.c.league_id).where(
            INGESTION_LEAGUES.c.league_id == league_id
        )
    ).first()
    return row is not None


def ensure_league_allowlisted(conn: Connection, league_id: int) -> None:
    """Ensure `league_id` is in `ingestion_leagues` so raw/canonical FKs hold.

    The league must already exist in `leagues`. This is the seam used by
    match-id ingest for catalog-null STRATZ leagues that remain
    `in_scope: false` in the yaml (so `load_league_registry` will not
    allowlist them itself).
    """
    if is_league_allowlisted(conn, league_id):
        return
    registered = conn.execute(
        select(LEAGUES.c.league_id).where(LEAGUES.c.league_id == league_id)
    ).first()
    if registered is None:
        raise LeagueNotRegisteredError(
            f"League {league_id} is not in the leagues registry"
        )
    conn.execute(INGESTION_LEAGUES.insert().values(league_id=league_id))


def list_allowlisted_league_ids(conn: Connection) -> list[int]:
    rows = conn.execute(
        select(INGESTION_LEAGUES.c.league_id).order_by(INGESTION_LEAGUES.c.league_id)
    ).all()
    return [int(row.league_id) for row in rows]


def get_league_fetch_mode(conn: Connection, league_id: int) -> str:
    """Return the registry fetch_mode for a league (`league` or `match_ids`).

    Missing rows and null/blank values default to `league` so historical
    leagues keep `league(id)` pagination unless explicitly configured.
    """
    row = conn.execute(
        select(LEAGUES.c.fetch_mode).where(LEAGUES.c.league_id == league_id)
    ).first()
    if row is None or not row.fetch_mode:
        return LEAGUE_FETCH_MODE_LEAGUE
    return str(row.fetch_mode)


def ensure_league_ingestion_state(conn: Connection, league_id: int) -> None:
    existing = conn.execute(
        select(LEAGUE_INGESTION_STATE.c.league_id).where(
            LEAGUE_INGESTION_STATE.c.league_id == league_id
        )
    ).first()
    if existing is not None:
        return
    now = datetime.now(UTC)
    conn.execute(
        LEAGUE_INGESTION_STATE.insert().values(
            league_id=league_id,
            status="PENDING",
            matches_seen_count=0,
            cursor_state=cursor_to_dict(CursorState()),
            last_synced_at=None,
            error_count=0,
            last_error=None,
            updated_at=now,
        )
    )


def load_cursor_state(conn: Connection, league_id: int) -> CursorState:
    from dota_predictor.ingestion.cursor import cursor_from_dict

    row = conn.execute(
        select(LEAGUE_INGESTION_STATE.c.cursor_state).where(
            LEAGUE_INGESTION_STATE.c.league_id == league_id
        )
    ).first()
    if row is None:
        return CursorState()
    return cursor_from_dict(row.cursor_state)


def get_persisted_match_ids_for_league(conn: Connection, league_id: int) -> set[int]:
    rows = conn.execute(
        select(STRATZ_RAW_MATCHES.c.match_id).where(
            STRATZ_RAW_MATCHES.c.league_id == league_id
        )
    ).all()
    return {int(row.match_id) for row in rows}


def persist_raw_page(
    conn: Connection,
    *,
    league_id: int,
    matches: list[dict[str, Any]],
    fetched_at: datetime,
) -> None:
    for match in matches:
        match_id = int(match["id"])
        stmt = pg_insert(STRATZ_RAW_MATCHES).values(
            match_id=match_id,
            league_id=league_id,
            payload=match,
            fetched_at=fetched_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[STRATZ_RAW_MATCHES.c.match_id],
            set_={
                "league_id": league_id,
                "payload": match,
                "fetched_at": fetched_at,
            },
        )
        conn.execute(stmt)


def update_league_ingestion_state(
    conn: Connection,
    *,
    league_id: int,
    status: str,
    cursor: CursorState,
    matches_seen_count: int,
    last_error: str | None = None,
    increment_error_count: int = 0,
) -> None:
    now = datetime.now(UTC)
    values: dict[str, Any] = {
        "status": status,
        "cursor_state": cursor_to_dict(cursor),
        "matches_seen_count": matches_seen_count,
        "last_synced_at": now,
        "updated_at": now,
    }
    if last_error is not None:
        values["last_error"] = last_error
    if increment_error_count:
        current = conn.execute(
            select(LEAGUE_INGESTION_STATE.c.error_count).where(
                LEAGUE_INGESTION_STATE.c.league_id == league_id
            )
        ).scalar_one()
        values["error_count"] = int(current) + increment_error_count

    conn.execute(
        LEAGUE_INGESTION_STATE.update()
        .where(LEAGUE_INGESTION_STATE.c.league_id == league_id)
        .values(**values)
    )


def insert_match_ingestion_error(
    conn: Connection,
    *,
    match_id: int,
    league_id: int,
    stage: str,
    error_message: str,
    raw_payload_snapshot: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        MATCH_INGESTION_ERRORS.insert().values(
            match_id=match_id,
            league_id=league_id,
            stage=stage,
            error_message=error_message,
            raw_payload_snapshot=raw_payload_snapshot,
            occurred_at=datetime.now(UTC),
            resolved=False,
        )
    )


def list_raw_matches_for_league(conn: Connection, league_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        select(STRATZ_RAW_MATCHES.c.match_id, STRATZ_RAW_MATCHES.c.payload).where(
            STRATZ_RAW_MATCHES.c.league_id == league_id
        )
    ).all()
    return [{"match_id": int(row.match_id), "payload": row.payload} for row in rows]


def get_canonical_mapper_version(conn: Connection, match_id: int) -> int | None:
    row = conn.execute(
        select(MATCHES.c.mapper_version).where(MATCHES.c.match_id == match_id)
    ).first()
    if row is None:
        return None
    return int(row.mapper_version)


def league_canonicalization_complete(conn: Connection, league_id: int) -> bool:
    """True when every raw match has canonical row at current mapper version."""
    raw_ids = conn.execute(
        select(STRATZ_RAW_MATCHES.c.match_id).where(
            STRATZ_RAW_MATCHES.c.league_id == league_id
        )
    ).all()
    if not raw_ids:
        return True
    for row in raw_ids:
        match_id = int(row.match_id)
        version = get_canonical_mapper_version(conn, match_id)
        if version is None or version < CANONICAL_MAPPER_VERSION:
            return False
    return True


def get_matches_seen_count(conn: Connection, league_id: int) -> int:
    return int(
        conn.execute(
            select(LEAGUE_INGESTION_STATE.c.matches_seen_count).where(
                LEAGUE_INGESTION_STATE.c.league_id == league_id
            )
        ).scalar_one()
    )
