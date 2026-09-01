"""Diagnostics for observed match-level STRATZ position/lane/role.

These helpers classify source values. They do not infer, repair, or fill
missing positions. `slot_in_side` is never treated as a Dota position.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from dota_predictor.data.canonical_schema import (
    EXPLICIT_DOTA_POSITIONS,
    MatchPlayerPosition,
)

__all__ = [
    "EXPLICIT_POSITION_VALUES",
    "SidePositionAudit",
    "audit_side_positions",
    "explicit_position_values",
    "is_clean_unique_1_to_5",
]

EXPLICIT_POSITION_VALUES: frozenset[str] = frozenset(
    member.value for member in EXPLICIT_DOTA_POSITIONS
)


@dataclass(frozen=True, slots=True)
class SidePositionAudit:
    """Observed-position completeness for one match side (5 players)."""

    explicit_values: tuple[str, ...]
    unknown_count: int
    null_count: int
    other_count: int
    duplicate_explicit: tuple[str, ...]
    missing_explicit: tuple[str, ...]
    is_clean_unique_1_to_5: bool


def _as_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, MatchPlayerPosition):
        return value.value
    return str(value)


def explicit_position_values(values: Sequence[object]) -> tuple[str, ...]:
    """Return POSITION_1–5 entries only, preserving duplicates and order."""
    return tuple(
        parsed
        for parsed in (_as_value(value) for value in values)
        if parsed in EXPLICIT_POSITION_VALUES
    )


def is_clean_unique_1_to_5(values: Sequence[object]) -> bool:
    """True iff the five slots are a permutation of POSITION_1–5."""
    explicit = explicit_position_values(values)
    return len(values) == 5 and set(explicit) == EXPLICIT_POSITION_VALUES


def audit_side_positions(values: Sequence[object]) -> SidePositionAudit:
    """Classify one side's five observed position slots.

    Does not rewrite values. Duplicate/missing 1–5 assignments are
    reported, not repaired.
    """
    parsed = [_as_value(value) for value in values]
    unknown_count = sum(value == MatchPlayerPosition.UNKNOWN.value for value in parsed)
    null_count = sum(value is None for value in parsed)
    explicit = tuple(
        value for value in parsed if value in EXPLICIT_POSITION_VALUES
    )
    other_count = len(parsed) - unknown_count - null_count - len(explicit)
    counts = Counter(explicit)
    duplicate_explicit = tuple(
        sorted(value for value, count in counts.items() if count > 1)
    )
    missing_explicit = tuple(
        sorted(EXPLICIT_POSITION_VALUES - set(explicit))
    )
    return SidePositionAudit(
        explicit_values=explicit,
        unknown_count=unknown_count,
        null_count=null_count,
        other_count=other_count,
        duplicate_explicit=duplicate_explicit,
        missing_explicit=missing_explicit,
        is_clean_unique_1_to_5=is_clean_unique_1_to_5(values),
    )
