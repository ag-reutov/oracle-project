"""Tests for observed-position diagnostics. No inference or repair."""

from __future__ import annotations

from dota_predictor.data.canonical_schema import MatchPlayerPosition
from dota_predictor.data.player_position_diagnostics import (
    audit_side_positions,
    is_clean_unique_1_to_5,
)

UNIQUE = (
    MatchPlayerPosition.POSITION_1,
    MatchPlayerPosition.POSITION_2,
    MatchPlayerPosition.POSITION_3,
    MatchPlayerPosition.POSITION_4,
    MatchPlayerPosition.POSITION_5,
)


def test_clean_unique_1_to_5() -> None:
    shuffled = (
        MatchPlayerPosition.POSITION_5,
        MatchPlayerPosition.POSITION_1,
        MatchPlayerPosition.POSITION_3,
        MatchPlayerPosition.POSITION_2,
        MatchPlayerPosition.POSITION_4,
    )
    assert is_clean_unique_1_to_5(UNIQUE)
    assert is_clean_unique_1_to_5(shuffled)
    audit = audit_side_positions(shuffled)
    assert audit.is_clean_unique_1_to_5
    assert audit.duplicate_explicit == ()
    assert audit.missing_explicit == ()
    assert audit.null_count == 0
    assert audit.unknown_count == 0


def test_duplicate_and_missing_are_reported_not_repaired() -> None:
    values = (
        MatchPlayerPosition.POSITION_1,
        MatchPlayerPosition.POSITION_1,
        None,
        MatchPlayerPosition.UNKNOWN,
        MatchPlayerPosition.POSITION_3,
    )
    audit = audit_side_positions(values)
    assert not audit.is_clean_unique_1_to_5
    assert audit.duplicate_explicit == ("POSITION_1",)
    assert "POSITION_2" in audit.missing_explicit
    assert "POSITION_4" in audit.missing_explicit
    assert "POSITION_5" in audit.missing_explicit
    assert audit.null_count == 1
    assert audit.unknown_count == 1
    assert values[1] is MatchPlayerPosition.POSITION_1
