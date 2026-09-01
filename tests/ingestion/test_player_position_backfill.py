"""Tests for merging STRATZ position/lane/role into stored raw payloads."""

from __future__ import annotations

from dota_predictor.ingestion.player_position_backfill import (
    merge_position_fields_into_payload,
    raw_payload_has_position_fields,
)


def _payload() -> dict:
    return {
        "id": 1,
        "durationSeconds": 1800,
        "players": [
            {
                "steamAccountId": 11,
                "isRadiant": True,
                "playerSlot": 0,
                "heroId": 1,
            },
            {
                "steamAccountId": 12,
                "isRadiant": True,
                "playerSlot": 1,
                "heroId": 2,
            },
        ],
    }


def test_raw_payload_without_position_key_is_not_patched() -> None:
    assert raw_payload_has_position_fields(_payload()) is False


def test_merge_copies_only_position_lane_role_by_steam_id() -> None:
    payload = _payload()
    fetched = [
        {
            "steamAccountId": 12,
            "heroId": 99,
            "playerSlot": 9,
            "position": "POSITION_2",
            "lane": "MID_LANE",
            "role": "CORE",
        },
        {
            "steamAccountId": 11,
            "position": "UNKNOWN",
            "lane": None,
            "role": "HARD_SUPPORT",
        },
    ]
    merged = merge_position_fields_into_payload(payload, fetched)
    by_id = {player["steamAccountId"]: player for player in merged["players"]}
    assert by_id[11]["heroId"] == 1
    assert by_id[11]["playerSlot"] == 0
    assert by_id[11]["position"] == "UNKNOWN"
    assert by_id[11]["lane"] is None
    assert by_id[11]["role"] == "HARD_SUPPORT"
    assert by_id[12]["heroId"] == 2
    assert by_id[12]["position"] == "POSITION_2"
    assert payload["players"][0].get("position") is None
    assert raw_payload_has_position_fields(merged) is True


def test_merge_sets_null_keys_when_fetched_player_missing() -> None:
    payload = _payload()
    fetched = [
        {
            "steamAccountId": 11,
            "position": "POSITION_1",
            "lane": "SAFE_LANE",
            "role": "CORE",
        }
    ]
    merged = merge_position_fields_into_payload(payload, fetched)
    second = merged["players"][1]
    assert second["steamAccountId"] == 12
    assert second["heroId"] == 2
    assert second["position"] is None
    assert second["lane"] is None
    assert second["role"] is None
    assert "position" in second
