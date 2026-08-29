"""Unit tests for pagination helpers."""

from __future__ import annotations

import pytest

from dota_predictor.ingestion.cursor import CursorState
from dota_predictor.ingestion.errors import PaginationDriftError
from dota_predictor.ingestion.pagination import (
    advance_cursor_after_page,
    empty_page_is_terminal,
    short_page_is_terminal,
    verify_resume_anchor,
)


def test_short_page_is_terminal() -> None:
    assert short_page_is_terminal(100, 50) is True
    assert short_page_is_terminal(100, 100) is False
    assert short_page_is_terminal(100, 0) is False


def test_empty_page_is_terminal() -> None:
    assert empty_page_is_terminal(0) is True
    assert empty_page_is_terminal(5) is False


def test_advance_cursor_after_page_increments_by_rows_not_take() -> None:
    cursor = CursorState(next_skip=0, take=100)
    matches = [{"id": 3, "startDateTime": 30}, {"id": 2, "startDateTime": 20}]
    updated = advance_cursor_after_page(cursor, matches, page_size=100)
    assert updated.next_skip == 2
    assert updated.last_match_id == 2
    assert updated.last_start_date_time == 20
    assert updated.fetch_complete is True


def test_advance_cursor_full_page_not_complete() -> None:
    cursor = CursorState(next_skip=100, take=100)
    matches = [{"id": i, "startDateTime": i} for i in range(100)]
    updated = advance_cursor_after_page(cursor, matches, page_size=100)
    assert updated.next_skip == 200
    assert updated.fetch_complete is False


def test_verify_resume_anchor_success() -> None:
    cursor = CursorState(
        next_skip=200,
        last_match_id=42,
        last_start_date_time=1000,
    )
    verify_resume_anchor({"id": 42, "startDateTime": 1000}, cursor)


def test_verify_resume_anchor_mismatch_raises() -> None:
    cursor = CursorState(next_skip=200, last_match_id=42)
    with pytest.raises(PaginationDriftError, match="Pagination drift"):
        verify_resume_anchor({"id": 99, "startDateTime": 1}, cursor)


def test_verify_resume_anchor_missing_raises() -> None:
    cursor = CursorState(next_skip=200, last_match_id=42)
    with pytest.raises(PaginationDriftError, match="Resume anchor missing"):
        verify_resume_anchor(None, cursor)
