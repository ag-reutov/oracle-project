"""Merge STRATZ match-player box-score scalars into stored raw JSON.

Persisted raw payloads were fetched without these fields. This module
refetches only the approved post-match scalars, copies them onto existing
player objects by `steamAccountId`, and updates the matching
`match_players` columns.

It does not replace the rest of the raw payload, does not rewrite
`team_id`/`hero_id`/`player_id`/`side`/`slot_in_side`/position/lane/role,
and does not coerce missing values to zero.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Connection, func, select, update
from sqlalchemy.engine import Engine

from dota_predictor.data.canonical_schema import (
    MATCH_PLAYER_BOX_SCORE_COLUMNS,
    PLAYER_BOX_SCORE_FIELD_MAP,
    STRATZ_PLAYER_BOX_SCORE_FIELDS,
)
from dota_predictor.storage.schema import MATCH_PLAYERS, MATCHES, STRATZ_RAW_MATCHES

__all__ = [
    "PlayerPerformanceBackfillResult",
    "apply_player_performance_to_canonical",
    "canonical_match_ids_for_performance",
    "merge_performance_fields_into_payload",
    "raw_payload_has_performance_fields",
    "run_player_performance_backfill",
    "summarize_player_performance_coverage",
]


@dataclass(frozen=True)
class PlayerPerformanceBackfillResult:
    canonical_matches: int
    already_patched: int
    fetched: int
    fetch_failures: int
    canonical_rows_updated: int
    missing_raw: int


def raw_payload_has_performance_fields(payload: Mapping[str, Any]) -> bool:
    """True when every player object already has every box-score key.

    Keys may be null. Presence means this payload was already merged
    and a restart can skip the STRATZ refetch.
    """
    players = payload.get("players")
    if not isinstance(players, list) or not players:
        return False
    return all(
        isinstance(player, Mapping)
        and all(field in player for field in STRATZ_PLAYER_BOX_SCORE_FIELDS)
        for player in players
    )


def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected integer or null, got {value!r}")
    return int(value)


def merge_performance_fields_into_payload(
    payload: Mapping[str, Any],
    fetched_players: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Copy `payload` and set box-score scalars on players by steam id.

    Other player and match fields are left unchanged. Players present in
    the stored payload but absent from the fetch receive explicit null
    keys so restart-safety can detect that the match was processed.
    """
    merged = copy.deepcopy(dict(payload))
    by_player_id: dict[int, Mapping[str, Any]] = {}
    for player in fetched_players:
        steam_account_id = player.get("steamAccountId")
        if steam_account_id is None:
            continue
        by_player_id[int(steam_account_id)] = player

    stored_players = merged.get("players")
    if not isinstance(stored_players, list):
        raise TypeError("raw payload players is not a list")

    updated_players: list[Any] = []
    for player in stored_players:
        if not isinstance(player, dict):
            updated_players.append(player)
            continue
        steam_account_id = player.get("steamAccountId")
        fetched = (
            by_player_id.get(int(steam_account_id))
            if steam_account_id is not None
            else None
        )
        patched = dict(player)
        source = fetched or {}
        for field in STRATZ_PLAYER_BOX_SCORE_FIELDS:
            patched[field] = _normalize_optional_int(source.get(field))
        updated_players.append(patched)
    merged["players"] = updated_players
    return merged


def canonical_match_ids_for_performance(
    conn: Connection,
    *,
    until: datetime | None = None,
    match_ids: Sequence[int] | None = None,
) -> list[int]:
    stmt = select(MATCHES.c.match_id)
    if match_ids is not None:
        stmt = stmt.where(
            MATCHES.c.match_id.in_([int(match_id) for match_id in match_ids])
        )
    if until is not None:
        stmt = stmt.where(MATCHES.c.start_time <= until)
    stmt = stmt.order_by(MATCHES.c.match_id)
    rows = conn.execute(stmt).all()
    return [int(row.match_id) for row in rows]


def apply_player_performance_to_canonical(
    conn: Connection, *, match_id: int, payload: Mapping[str, Any]
) -> int:
    """Update only box-score columns on `match_players` for `match_id`."""
    players = payload.get("players")
    if not isinstance(players, list):
        return 0
    updated = 0
    for player in players:
        if not isinstance(player, Mapping):
            continue
        steam_account_id = player.get("steamAccountId")
        if steam_account_id is None:
            continue
        values = {
            canonical_name: _normalize_optional_int(player.get(stratz_name))
            for stratz_name, canonical_name in PLAYER_BOX_SCORE_FIELD_MAP
        }
        result = conn.execute(
            update(MATCH_PLAYERS)
            .where(
                MATCH_PLAYERS.c.match_id == match_id,
                MATCH_PLAYERS.c.player_id == int(steam_account_id),
            )
            .values(**values)
        )
        updated += int(result.rowcount or 0)
    return updated


def run_player_performance_backfill(
    engine: Engine,
    fetcher: Any,
    *,
    limit: int | None = None,
    until: datetime | None = None,
    match_ids: Sequence[int] | None = None,
    progress_every: int = 100,
) -> PlayerPerformanceBackfillResult:
    """Refetch missing box-score scalars and update canonical player rows.

    `fetcher` must implement `fetch_match_player_performance(match_id)`.
    Restart-safe: matches whose raw player objects already have every
    box-score key are not refetched. Canonical columns are still
    reapplied from the stored payload.
    """
    with engine.connect() as conn:
        selected_ids = canonical_match_ids_for_performance(
            conn, until=until, match_ids=match_ids
        )
    if limit is not None:
        selected_ids = selected_ids[: int(limit)]

    already_patched = 0
    fetched = 0
    fetch_failures = 0
    canonical_rows_updated = 0
    missing_raw = 0

    for index, match_id in enumerate(selected_ids, start=1):
        with engine.connect() as conn:
            raw_row = conn.execute(
                select(STRATZ_RAW_MATCHES.c.payload).where(
                    STRATZ_RAW_MATCHES.c.match_id == match_id
                )
            ).first()
        if raw_row is None:
            missing_raw += 1
            continue
        payload: dict[str, Any] = dict(raw_row.payload)
        persist_raw = False
        if not raw_payload_has_performance_fields(payload):
            fetched_match = fetcher.fetch_match_player_performance(match_id)
            fetched_players = (
                fetched_match.get("players") if fetched_match is not None else None
            )
            if not isinstance(fetched_players, list) or not fetched_players:
                fetch_failures += 1
                continue
            try:
                payload = merge_performance_fields_into_payload(
                    payload, fetched_players
                )
            except (TypeError, ValueError):
                fetch_failures += 1
                continue
            persist_raw = True
            fetched += 1
        else:
            already_patched += 1

        with engine.begin() as conn:
            if persist_raw:
                conn.execute(
                    update(STRATZ_RAW_MATCHES)
                    .where(STRATZ_RAW_MATCHES.c.match_id == match_id)
                    .values(payload=payload)
                )
            canonical_rows_updated += apply_player_performance_to_canonical(
                conn, match_id=match_id, payload=payload
            )

        if progress_every and index % progress_every == 0:
            print(
                f"player-performance backfill {index}/{len(selected_ids)} "
                f"fetched={fetched} already_patched={already_patched} "
                f"fetch_failures={fetch_failures}",
                flush=True,
            )

    return PlayerPerformanceBackfillResult(
        canonical_matches=len(selected_ids),
        already_patched=already_patched,
        fetched=fetched,
        fetch_failures=fetch_failures,
        canonical_rows_updated=canonical_rows_updated,
        missing_raw=missing_raw,
    )


def summarize_player_performance_coverage(
    conn: Connection, *, until: datetime | None = None
) -> dict[str, Any]:
    """Null rates and min/max for landed box-score columns.

    Diagnostics only. Does not interpret predictive value.
    """
    match_filters = []
    player_from = MATCH_PLAYERS.join(
        MATCHES, MATCH_PLAYERS.c.match_id == MATCHES.c.match_id
    )
    if until is not None:
        match_filters.append(MATCHES.c.start_time <= until)

    n_matches = int(
        conn.execute(
            select(func.count()).select_from(MATCHES).where(*match_filters)
        ).scalar_one()
    )
    n_players = int(
        conn.execute(
            select(func.count()).select_from(player_from).where(*match_filters)
        ).scalar_one()
    )
    columns: dict[str, dict[str, Any]] = {}
    for column_name in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        column = MATCH_PLAYERS.c[column_name]
        nulls = int(
            conn.execute(
                select(func.count())
                .select_from(player_from)
                .where(*match_filters, column.is_(None))
            ).scalar_one()
        )
        minimum, maximum = conn.execute(
            select(func.min(column), func.max(column))
            .select_from(player_from)
            .where(*match_filters)
        ).one()
        columns[column_name] = {
            "nulls": nulls,
            "null_rate": (nulls / n_players) if n_players else None,
            "min": minimum,
            "max": maximum,
        }
    return {
        "matches": n_matches,
        "player_rows": n_players,
        "columns": columns,
    }
