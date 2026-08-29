"""Defensive validation for fetched STRATZ match pages."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from dota_predictor.ingestion.errors import PageValidationError

__all__ = ["PageValidationResult", "validate_match_page"]

logger = logging.getLogger(__name__)


class PageValidationResult:
    """Outcome of page validation (raises on hard failures)."""

    def __init__(
        self,
        *,
        overlap_with_persisted: list[int],
        start_times_non_increasing: bool | None,
    ) -> None:
        self.overlap_with_persisted = overlap_with_persisted
        self.start_times_non_increasing = start_times_non_increasing


def validate_match_page(
    matches: list[dict[str, Any]],
    *,
    league_id: int,
    persisted_match_ids: set[int] | None = None,
) -> PageValidationResult:
    """Validate a fetched page before raw persistence.

    Raises `PageValidationError` on hard failures (duplicate IDs within page,
    wrong league). Overlap with already-persisted IDs and ordering diagnostics
    are logged but do not block persistence.
    """
    if not matches:
        return PageValidationResult(
            overlap_with_persisted=[],
            start_times_non_increasing=None,
        )

    match_ids = [int(m["id"]) for m in matches]
    id_counts = Counter(match_ids)
    duplicates = [mid for mid, count in id_counts.items() if count > 1]
    if duplicates:
        raise PageValidationError(
            f"Page contains duplicate match IDs: {sorted(duplicates)}"
        )

    wrong_league = [
        int(m["id"])
        for m in matches
        if m.get("leagueId") is not None and int(m["leagueId"]) != league_id
    ]
    if wrong_league:
        raise PageValidationError(
            f"Matches do not belong to league {league_id}: {sorted(wrong_league)}"
        )

    overlap: list[int] = []
    if persisted_match_ids:
        overlap = sorted(set(match_ids) & persisted_match_ids)
        if overlap:
            logger.warning(
                "Page overlap with already-persisted match IDs for league %s: %s",
                league_id,
                overlap,
            )

    start_times = [m.get("startDateTime") for m in matches]
    non_increasing: bool | None = None
    if all(value is not None for value in start_times):
        non_increasing = start_times == sorted(start_times, reverse=True)
        if not non_increasing:
            logger.warning(
                "Observed startDateTime not non-increasing for league %s page "
                "(diagnostic only)",
                league_id,
            )

    return PageValidationResult(
        overlap_with_persisted=overlap,
        start_times_non_increasing=non_increasing,
    )
