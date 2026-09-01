"""Merge STRATZ match-player position/lane/role into stored raw JSON.

Persisted raw payloads were fetched without these fields. This module
refetches only the player parse labels, copies `position`/`lane`/`role`
onto existing player objects by `steamAccountId`, and updates the
matching `match_players` columns.

It does not replace the rest of the raw payload, does not rewrite
`team_id`/`hero_id`/`player_id`/`side`/`slot_in_side`, and does not
infer missing values.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Connection, select, update
from sqlalchemy.engine import Engine

from dota_predictor.data.canonical_schema import (
    MatchLane,
    MatchPlayerPosition,
    MatchPlayerRole,
)
from dota_predictor.storage.schema import MATCH_PLAYERS, MATCHES, STRATZ_RAW_MATCHES

__all__ = [
    "POSITION_FIELD_KEYS",
    "PlayerPositionBackfillResult",
    "apply_player_positions_to_canonical",
    "canonical_match_ids",
    "merge_position_fields_into_payload",
    "raw_payload_has_position_fields",
    "run_player_position_backfill",
]

POSITION_FIELD_KEYS: tuple[str, str, str] = ("position", "lane", "role")

_POSITION_VALUES = frozenset(member.value for member in MatchPlayerPosition)
_LANE_VALUES = frozenset(member.value for member in MatchLane)
_ROLE_VALUES = frozenset(member.value for member in MatchPlayerRole)
_ALLOWED_BY_FIELD = {
    "position": _POSITION_VALUES,
    "lane": _LANE_VALUES,
    "role": _ROLE_VALUES,
}


@dataclass(frozen=True)
class PlayerPositionBackfillResult:
    canonical_matches: int
    already_patched: int
    fetched: int
    fetch_failures: int
    canonical_rows_updated: int
    missing_raw: int


def raw_payload_has_position_fields(payload: Mapping[str, Any]) -> bool:
    """True when every player object already has a `position` key.

    The key may be null. Presence means this payload was already merged
    and a restart can skip the STRATZ refetch.
    """
    players = payload.get("players")
    if not isinstance(players, list) or not players:
        return False
    return all(
        isinstance(player, Mapping) and "position" in player for player in players
    )


def _normalize_enum_value(field: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    allowed = _ALLOWED_BY_FIELD[field]
    if text not in allowed:
        raise ValueError(f"unsupported {field} value {value!r}")
    return text


def merge_position_fields_into_payload(
    payload: Mapping[str, Any],
    fetched_players: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Copy `payload` and set position/lane/role on players by steam id.

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
        for field in POSITION_FIELD_KEYS:
            patched[field] = _normalize_enum_value(field, source.get(field))
        updated_players.append(patched)
    merged["players"] = updated_players
    return merged


def canonical_match_ids(conn: Connection) -> list[int]:
    rows = conn.execute(
        select(MATCHES.c.match_id).order_by(MATCHES.c.match_id)
    ).all()
    return [int(row.match_id) for row in rows]


def apply_player_positions_to_canonical(
    conn: Connection, *, match_id: int, payload: Mapping[str, Any]
) -> int:
    """Update only position/lane/role on `match_players` for `match_id`."""
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
        result = conn.execute(
            update(MATCH_PLAYERS)
            .where(
                MATCH_PLAYERS.c.match_id == match_id,
                MATCH_PLAYERS.c.player_id == int(steam_account_id),
            )
            .values(
                position=_normalize_enum_value("position", player.get("position")),
                lane=_normalize_enum_value("lane", player.get("lane")),
                role=_normalize_enum_value("role", player.get("role")),
            )
        )
        updated += int(result.rowcount or 0)
    return updated


def run_player_position_backfill(
    engine: Engine,
    fetcher: Any,
    *,
    limit: int | None = None,
    progress_every: int = 100,
) -> PlayerPositionBackfillResult:
    """Refetch missing parse labels and update canonical player rows.

    `fetcher` must implement `fetch_match_player_positions(match_id)`.
    Restart-safe: matches whose raw player objects already have a
    `position` key are not refetched. Canonical columns are still
    reapplied from the stored payload.
    """
    with engine.connect() as conn:
        match_ids = canonical_match_ids(conn)
    if limit is not None:
        match_ids = match_ids[: int(limit)]

    already_patched = 0
    fetched = 0
    fetch_failures = 0
    canonical_rows_updated = 0
    missing_raw = 0

    for index, match_id in enumerate(match_ids, start=1):
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
        if not raw_payload_has_position_fields(payload):
            fetched_match = fetcher.fetch_match_player_positions(match_id)
            fetched_players = (
                fetched_match.get("players") if fetched_match is not None else None
            )
            if not isinstance(fetched_players, list) or not fetched_players:
                fetch_failures += 1
                continue
            try:
                payload = merge_position_fields_into_payload(payload, fetched_players)
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
            canonical_rows_updated += apply_player_positions_to_canonical(
                conn, match_id=match_id, payload=payload
            )

        if progress_every and index % progress_every == 0:
            print(
                f"player-position backfill {index}/{len(match_ids)} "
                f"fetched={fetched} already_patched={already_patched} "
                f"fetch_failures={fetch_failures}",
                flush=True,
            )

    return PlayerPositionBackfillResult(
        canonical_matches=len(match_ids),
        already_patched=already_patched,
        fetched=fetched,
        fetch_failures=fetch_failures,
        canonical_rows_updated=canonical_rows_updated,
        missing_raw=missing_raw,
    )
