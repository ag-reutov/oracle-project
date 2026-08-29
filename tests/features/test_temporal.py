"""Tests for `features.temporal` (chronological eligibility helpers).

Pure, in-memory tests: no Parquet fixtures, no DuckDB connection.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dota_predictor.features.temporal import is_historical

EARLIER = datetime(2024, 1, 1, tzinfo=UTC)
LATER = datetime(2024, 6, 1, tzinfo=UTC)


def test_strictly_earlier_start_time_is_historical() -> None:
    assert (
        is_historical(historical_start_time=EARLIER, current_start_time=LATER) is True
    )


def test_equal_start_time_is_not_historical() -> None:
    """Equal timestamps must be rejected -- a tie is not historical
    information, per the strict `<` requirement."""
    assert (
        is_historical(historical_start_time=EARLIER, current_start_time=EARLIER)
        is False
    )


def test_later_start_time_is_not_historical() -> None:
    assert (
        is_historical(historical_start_time=LATER, current_start_time=EARLIER) is False
    )


def test_one_microsecond_before_is_historical() -> None:
    just_before = LATER - timedelta(microseconds=1)
    assert (
        is_historical(historical_start_time=just_before, current_start_time=LATER)
        is True
    )


def test_one_microsecond_after_is_not_historical() -> None:
    just_after = LATER + timedelta(microseconds=1)
    assert (
        is_historical(historical_start_time=just_after, current_start_time=LATER)
        is False
    )
