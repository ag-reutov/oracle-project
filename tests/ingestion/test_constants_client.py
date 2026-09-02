"""Unit tests for STRATZ constants query parsing and client fetch methods.

No live STRATZ calls: payloads are fixtures and `StratzClient._fetch_with_retry`
is stubbed.
"""

from __future__ import annotations

from typing import Any

import pytest

from dota_predictor.ingestion.client import (
    StratzClient,
    parse_game_versions_query_payload,
    parse_heroes_query_payload,
)
from dota_predictor.ingestion.config import IngestionConfig
from dota_predictor.ingestion.errors import StratzPermanentError
from dota_predictor.ingestion.queries import (
    GAME_VERSIONS_QUERY,
    HEROES_QUERY,
    MATCH_PLAYER_PERFORMANCE_QUERY,
    MATCH_PLAYER_POSITIONS_QUERY,
)

HEROES_PAYLOAD = {
    "data": {
        "constants": {
            "heroes": [
                {"id": 1, "displayName": "Anti-Mage"},
                {"id": 2, "displayName": "Axe"},
            ]
        }
    }
}

GAME_VERSIONS_PAYLOAD = {
    "data": {
        "constants": {
            "gameVersions": [
                {"id": 173, "name": "7.36", "asOfDateTime": 1716422400},
                {"id": 175, "name": "7.36c", "asOfDateTime": 1719187200},
            ]
        }
    }
}


def test_heroes_query_requests_only_identity_fields() -> None:
    assert "displayName" in HEROES_QUERY
    assert "id" in HEROES_QUERY
    for forbidden in (
        "gameVersionId",
        "aliases",
        "roles",
        "stats",
        "facets",
        "talents",
        "abilities",
        "lore",
        "shortName",
        "language",
    ):
        assert forbidden not in HEROES_QUERY


def test_game_versions_query_requests_id_name_as_of() -> None:
    assert "id" in GAME_VERSIONS_QUERY
    assert "name" in GAME_VERSIONS_QUERY
    assert "asOfDateTime" in GAME_VERSIONS_QUERY


def test_parse_heroes_query_payload_extracts_rows() -> None:
    rows = parse_heroes_query_payload(HEROES_PAYLOAD)
    assert rows == [
        {"id": 1, "displayName": "Anti-Mage"},
        {"id": 2, "displayName": "Axe"},
    ]


def test_parse_game_versions_query_payload_extracts_rows() -> None:
    rows = parse_game_versions_query_payload(GAME_VERSIONS_PAYLOAD)
    assert rows == [
        {"id": 173, "name": "7.36", "asOfDateTime": 1716422400},
        {"id": 175, "name": "7.36c", "asOfDateTime": 1719187200},
    ]


def test_parse_heroes_null_list_returns_empty() -> None:
    assert parse_heroes_query_payload({"data": {"constants": {"heroes": None}}}) == []


def test_parse_heroes_missing_data_is_permanent() -> None:
    with pytest.raises(StratzPermanentError, match="missing data"):
        parse_heroes_query_payload({"data": None})


def test_parse_game_versions_missing_constants_is_permanent() -> None:
    with pytest.raises(StratzPermanentError, match="missing constants"):
        parse_game_versions_query_payload({"data": {"constants": None}})


def test_parse_heroes_non_list_is_permanent() -> None:
    with pytest.raises(StratzPermanentError, match="non-list"):
        parse_heroes_query_payload({"data": {"constants": {"heroes": {"id": 1}}}})


def test_fetch_heroes_uses_heroes_query(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_fetch(
        self: StratzClient, query: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        captured["query"] = query
        captured["variables"] = variables
        return HEROES_PAYLOAD

    monkeypatch.setattr(StratzClient, "_fetch_with_retry", _fake_fetch)
    with StratzClient(IngestionConfig(stratz_api_token="test")) as client:
        rows = client.fetch_heroes()

    assert captured["query"] == HEROES_QUERY
    assert captured["variables"] == {}
    assert [row["displayName"] for row in rows] == ["Anti-Mage", "Axe"]


def test_fetch_game_versions_uses_game_versions_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_fetch(
        self: StratzClient, query: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        captured["query"] = query
        captured["variables"] = variables
        return GAME_VERSIONS_PAYLOAD

    monkeypatch.setattr(StratzClient, "_fetch_with_retry", _fake_fetch)
    with StratzClient(IngestionConfig(stratz_api_token="test")) as client:
        rows = client.fetch_game_versions()

    assert captured["query"] == GAME_VERSIONS_QUERY
    assert captured["variables"] == {}
    assert [row["id"] for row in rows] == [173, 175]


def test_fetch_match_player_positions_uses_lightweight_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_fetch(
        self: StratzClient, query: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        captured["query"] = query
        captured["variables"] = variables
        return {
            "data": {
                "match": {
                    "id": 1,
                    "players": [
                        {
                            "steamAccountId": 11,
                            "position": "POSITION_1",
                            "lane": "SAFE_LANE",
                            "role": "CORE",
                        }
                    ],
                }
            }
        }

    monkeypatch.setattr(StratzClient, "_fetch_with_retry", _fake_fetch)
    with StratzClient(IngestionConfig(stratz_api_token="test")) as client:
        match = client.fetch_match_player_positions(1)

    assert captured["query"] == MATCH_PLAYER_POSITIONS_QUERY
    assert captured["variables"] == {"id": 1}
    assert match is not None
    assert match["players"][0]["position"] == "POSITION_1"


def test_fetch_match_player_performance_uses_lightweight_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_fetch(
        self: StratzClient, query: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        captured["query"] = query
        captured["variables"] = variables
        return {
            "data": {
                "match": {
                    "id": 1,
                    "players": [
                        {
                            "steamAccountId": 11,
                            "kills": 0,
                            "deaths": 3,
                            "heroHealing": None,
                        }
                    ],
                }
            }
        }

    monkeypatch.setattr(StratzClient, "_fetch_with_retry", _fake_fetch)
    with StratzClient(IngestionConfig(stratz_api_token="test")) as client:
        match = client.fetch_match_player_performance(1)

    assert captured["query"] == MATCH_PLAYER_PERFORMANCE_QUERY
    assert captured["variables"] == {"id": 1}
    assert match is not None
    assert match["players"][0]["kills"] == 0
    assert match["players"][0]["heroHealing"] is None
