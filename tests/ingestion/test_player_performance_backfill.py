"""Tests for merging STRATZ player box-score scalars into stored raw payloads."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ingestion_helpers import (
    DIRE_IDS,
    RADIANT_IDS,
    build_stratz_match,
    requires_test_database,
    seed_ingestion_league,
)
from sqlalchemy import select

from dota_predictor.data.canonical_schema import STRATZ_PLAYER_BOX_SCORE_FIELDS
from dota_predictor.data.stratz_mapping import canonical_match_from_stratz
from dota_predictor.ingestion.player_performance_backfill import (
    merge_performance_fields_into_payload,
    raw_payload_has_performance_fields,
    run_player_performance_backfill,
)
from dota_predictor.storage.ingestion_writer import persist_raw_page
from dota_predictor.storage.schema import MATCH_PLAYERS, STRATZ_RAW_MATCHES
from dota_predictor.storage.writer import write_canonical_match


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
                "position": "POSITION_1",
            },
            {
                "steamAccountId": 12,
                "isRadiant": True,
                "playerSlot": 1,
                "heroId": 2,
                "position": "POSITION_2",
            },
        ],
    }


def test_raw_payload_without_kills_key_is_not_patched() -> None:
    assert raw_payload_has_performance_fields(_payload()) is False


def test_merge_copies_only_box_score_fields_by_steam_id() -> None:
    payload = _payload()
    fetched = [
        {
            "steamAccountId": 12,
            "heroId": 99,
            "playerSlot": 9,
            "kills": 0,
            "deaths": 4,
            "assists": 12,
            "goldPerMinute": 312,
            "experiencePerMinute": 400,
            "numLastHits": 50,
            "numDenies": 0,
            "networth": 8000,
            "heroDamage": 1000,
            "towerDamage": 0,
            "heroHealing": None,
            "level": 18,
        },
        {
            "steamAccountId": 11,
            "kills": 10,
            "deaths": 1,
            "assists": 5,
            "goldPerMinute": 700,
            "experiencePerMinute": 800,
            "numLastHits": 300,
            "numDenies": 12,
            "networth": 20000,
            "heroDamage": 30000,
            "towerDamage": 4000,
            "heroHealing": 0,
            "level": 25,
        },
    ]
    merged = merge_performance_fields_into_payload(payload, fetched)
    by_id = {player["steamAccountId"]: player for player in merged["players"]}
    assert by_id[11]["heroId"] == 1
    assert by_id[11]["playerSlot"] == 0
    assert by_id[11]["position"] == "POSITION_1"
    assert by_id[11]["kills"] == 10
    assert by_id[11]["heroHealing"] == 0
    assert by_id[12]["heroId"] == 2
    assert by_id[12]["kills"] == 0
    assert by_id[12]["numDenies"] == 0
    assert by_id[12]["heroHealing"] is None
    assert payload["players"][0].get("kills") is None
    assert raw_payload_has_performance_fields(merged) is True
    for field in STRATZ_PLAYER_BOX_SCORE_FIELDS:
        assert field in by_id[11]
        assert field in by_id[12]


def test_merge_sets_null_keys_when_fetched_player_missing() -> None:
    payload = _payload()
    fetched = [
        {
            "steamAccountId": 11,
            "kills": 1,
            "deaths": 0,
            "assists": 0,
            "goldPerMinute": 1,
            "experiencePerMinute": 1,
            "numLastHits": 1,
            "numDenies": 1,
            "networth": 1,
            "heroDamage": 1,
            "towerDamage": 1,
            "heroHealing": 1,
            "level": 1,
        }
    ]
    merged = merge_performance_fields_into_payload(payload, fetched)
    second = merged["players"][1]
    assert second["steamAccountId"] == 12
    assert second["heroId"] == 2
    assert second["position"] == "POSITION_2"
    for field in STRATZ_PLAYER_BOX_SCORE_FIELDS:
        assert field in second
        assert second[field] is None


class _StubPerformanceFetcher:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def fetch_match_player_performance(self, match_id: int) -> dict[str, Any]:
        self.calls += 1
        assert match_id == self.payload["id"]
        return self.payload


def _fetched_players_with_stats() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, player_id in enumerate(RADIANT_IDS + DIRE_IDS):
        rows.append(
            {
                "steamAccountId": player_id,
                "kills": 0 if index == 0 else 3,
                "deaths": 1,
                "assists": 5,
                "goldPerMinute": 400,
                "experiencePerMinute": 450,
                "numLastHits": 0 if index == 0 else 80,
                "numDenies": 0,
                "networth": 9000,
                "heroDamage": None if index == 1 else 12000,
                "towerDamage": 100,
                "heroHealing": 0,
                "level": 18,
            }
        )
    return rows


@requires_test_database
def test_backfill_enriches_existing_match_and_is_idempotent(engine) -> None:
    league_id = 42
    match_id = 4242
    raw = build_stratz_match(
        match_id=match_id,
        league_id=league_id,
        start_date_time=1_700_000_000,
    )
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=league_id)
        persist_raw_page(
            conn,
            league_id=league_id,
            matches=[raw],
            fetched_at=datetime.now(UTC),
        )
    write_canonical_match(engine, canonical_match_from_stratz(raw))

    with engine.connect() as conn:
        before = conn.execute(
            select(MATCH_PLAYERS.c.kills).where(MATCH_PLAYERS.c.match_id == match_id)
        ).all()
    assert all(row.kills is None for row in before)

    fetcher = _StubPerformanceFetcher(
        {"id": match_id, "players": _fetched_players_with_stats()}
    )
    first = run_player_performance_backfill(engine, fetcher, match_ids=[match_id])
    assert first.fetched == 1
    assert first.already_patched == 0
    assert first.fetch_failures == 0
    assert fetcher.calls == 1

    with engine.connect() as conn:
        rows = conn.execute(
            MATCH_PLAYERS.select()
            .where(MATCH_PLAYERS.c.match_id == match_id)
            .order_by(MATCH_PLAYERS.c.side, MATCH_PLAYERS.c.slot_in_side)
        ).all()
        payload = conn.execute(
            select(STRATZ_RAW_MATCHES.c.payload).where(
                STRATZ_RAW_MATCHES.c.match_id == match_id
            )
        ).scalar_one()
    by_player = {row.player_id: row for row in rows}
    assert by_player[RADIANT_IDS[0]].kills == 0
    assert by_player[RADIANT_IDS[0]].num_last_hits == 0
    assert by_player[RADIANT_IDS[0]].hero_healing == 0
    assert by_player[RADIANT_IDS[1]].hero_damage is None
    assert by_player[RADIANT_IDS[1]].kills == 3
    assert payload["players"][0]["steamAccountId"] == RADIANT_IDS[0]
    assert payload["players"][0]["kills"] == 0
    assert "position" not in payload["players"][0]
    assert len(rows) == 10

    second = run_player_performance_backfill(engine, fetcher, match_ids=[match_id])
    assert second.fetched == 0
    assert second.already_patched == 1
    assert fetcher.calls == 1
    with engine.connect() as conn:
        again = conn.execute(
            MATCH_PLAYERS.select().where(
                MATCH_PLAYERS.c.match_id == match_id,
                MATCH_PLAYERS.c.player_id == RADIANT_IDS[0],
            )
        ).one()
    assert again.kills == 0
    assert again.hero_healing == 0
