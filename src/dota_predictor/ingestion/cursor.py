"""Pagination cursor state for league ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dota_predictor.ingestion.config import DEFAULT_PAGE_SIZE

__all__ = ["CursorState", "cursor_from_dict", "cursor_to_dict"]


@dataclass
class CursorState:
    next_skip: int = 0
    take: int = DEFAULT_PAGE_SIZE
    last_match_id: int | None = None
    last_start_date_time: int | None = None
    fetch_complete: bool = False


def cursor_from_dict(data: dict[str, Any] | None) -> CursorState:
    if not data:
        return CursorState()
    return CursorState(
        next_skip=int(data.get("next_skip", 0)),
        take=int(data.get("take", DEFAULT_PAGE_SIZE)),
        last_match_id=data.get("last_match_id"),
        last_start_date_time=data.get("last_start_date_time"),
        fetch_complete=bool(data.get("fetch_complete", False)),
    )


def cursor_to_dict(cursor: CursorState) -> dict[str, Any]:
    return {
        "next_skip": cursor.next_skip,
        "take": cursor.take,
        "last_match_id": cursor.last_match_id,
        "last_start_date_time": cursor.last_start_date_time,
        "fetch_complete": cursor.fetch_complete,
    }
