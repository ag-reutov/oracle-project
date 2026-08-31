"""Unit tests for page validation."""

from __future__ import annotations

import pytest

from dota_predictor.ingestion.errors import PageValidationError
from dota_predictor.ingestion.page_validation import validate_match_page


def test_rejects_duplicate_ids_within_page() -> None:
    matches = [
        {"id": 1, "leagueId": 10, "startDateTime": 100},
        {"id": 1, "leagueId": 10, "startDateTime": 90},
    ]
    with pytest.raises(PageValidationError, match="duplicate"):
        validate_match_page(matches, league_id=10)


def test_rejects_wrong_league() -> None:
    matches = [{"id": 1, "leagueId": 99, "startDateTime": 100}]
    with pytest.raises(PageValidationError, match="do not belong"):
        validate_match_page(matches, league_id=10)


def test_validate_match_belongs_to_league_rejects_mismatch() -> None:
    from dota_predictor.ingestion.page_validation import validate_match_belongs_to_league

    with pytest.raises(PageValidationError, match="expected 10"):
        validate_match_belongs_to_league({"id": 1, "leagueId": 99}, league_id=10)


def test_validate_match_belongs_to_league_rejects_missing_league_id() -> None:
    from dota_predictor.ingestion.page_validation import validate_match_belongs_to_league

    with pytest.raises(PageValidationError, match="expected 10"):
        validate_match_belongs_to_league({"id": 1, "leagueId": None}, league_id=10)


def test_overlap_is_logged_not_fatal() -> None:
    matches = [{"id": 1, "leagueId": 10, "startDateTime": 100}]
    result = validate_match_page(
        matches, league_id=10, persisted_match_ids={1, 2}
    )
    assert result.overlap_with_persisted == [1]
