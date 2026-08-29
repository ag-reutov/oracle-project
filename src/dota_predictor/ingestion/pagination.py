"""Pagination helpers for offset-based STRATZ league match ingestion."""

from __future__ import annotations

from typing import Any

from dota_predictor.ingestion.cursor import CursorState
from dota_predictor.ingestion.errors import PaginationDriftError

__all__ = [
    "advance_cursor_after_page",
    "empty_page_is_terminal",
    "short_page_is_terminal",
    "verify_resume_anchor",
]


def short_page_is_terminal(page_size: int, rows_returned: int) -> bool:
    return 0 < rows_returned < page_size


def empty_page_is_terminal(rows_returned: int) -> bool:
    return rows_returned == 0


def advance_cursor_after_page(
    cursor: CursorState,
    matches: list[dict[str, Any]],
    *,
    page_size: int,
) -> CursorState:
    """Return updated cursor after successfully persisting a non-empty page."""
    if not matches:
        return cursor
    last = matches[-1]
    return CursorState(
        next_skip=cursor.next_skip + len(matches),
        take=page_size,
        last_match_id=int(last["id"]),
        last_start_date_time=last.get("startDateTime"),
        fetch_complete=short_page_is_terminal(page_size, len(matches)),
    )


def verify_resume_anchor(
    anchor_match: dict[str, Any] | None,
    cursor: CursorState,
) -> None:
    """Verify the resume anchor matches stored cursor boundary.

    Raises `PaginationDriftError` when the anchor does not match.
  """
    if cursor.next_skip <= 0 or cursor.fetch_complete:
        return
    if cursor.last_match_id is None:
        raise PaginationDriftError(
            "Cannot resume: next_skip > 0 but cursor_state.last_match_id is missing"
        )
    if anchor_match is None:
        raise PaginationDriftError(
            f"Resume anchor missing at skip={cursor.next_skip - 1}; "
            f"expected match id {cursor.last_match_id}"
        )
    anchor_id = int(anchor_match["id"])
    if anchor_id != int(cursor.last_match_id):
        raise PaginationDriftError(
            f"Pagination drift: anchor match id {anchor_id} != "
            f"stored last_match_id {cursor.last_match_id}"
        )
    if cursor.last_start_date_time is not None:
        anchor_start = anchor_match.get("startDateTime")
        if anchor_start is not None and int(anchor_start) != int(
            cursor.last_start_date_time
        ):
            raise PaginationDriftError(
                f"Pagination drift: anchor startDateTime {anchor_start} != "
                f"stored last_start_date_time {cursor.last_start_date_time}"
            )
