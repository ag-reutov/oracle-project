"""Unit tests for match(id) GraphQL payload parsing."""

from __future__ import annotations

import pytest

from dota_predictor.ingestion.client import parse_match_query_payload
from dota_predictor.ingestion.errors import StratzPermanentError


def test_parse_match_query_payload_extracts_match() -> None:
    payload = {
        "data": {
            "match": {
                "id": 8550292837,
                "leagueId": 17419,
                "league": None,
                "didRadiantWin": False,
            }
        }
    }
    match = parse_match_query_payload(payload)
    assert match is not None
    assert match["id"] == 8550292837
    assert match["leagueId"] == 17419
    assert match["league"] is None


def test_parse_match_query_payload_null_match_returns_none() -> None:
    assert parse_match_query_payload({"data": {"match": None}}) is None


def test_parse_match_query_payload_missing_data_is_permanent() -> None:
    with pytest.raises(StratzPermanentError, match="missing data"):
        parse_match_query_payload({"data": None})


def test_parse_match_query_payload_malformed_match_is_permanent() -> None:
    with pytest.raises(StratzPermanentError, match="malformed"):
        parse_match_query_payload({"data": {"match": {"id": None}}})
