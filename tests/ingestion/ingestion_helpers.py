"""Shared helpers for ingestion tests."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import Connection

from dota_predictor.storage.schema import INGESTION_LEAGUES, LEAGUES

__all__ = ["build_stratz_match", "requires_test_database", "seed_ingestion_league"]

requires_test_database = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set; skipping DB-touching test",
)

RADIANT_IDS = [898754153, 137129583, 129958758, 157475523, 94296097]
DIRE_IDS = [10366616, 100058342, 898455820, 183719386, 25907144]


def seed_ingestion_league(
    conn: Connection,
    league_id: int,
    *,
    name: str = "Test League",
    fetch_mode: str = "league",
) -> None:
    conn.execute(
        LEAGUES.insert().values(
            league_id=league_id,
            name=name,
            liquipedia_tier="T1",
            in_scope=True,
            fetch_mode=fetch_mode,
        )
    )
    conn.execute(INGESTION_LEAGUES.insert().values(league_id=league_id))


def build_stratz_match(
    *,
    match_id: int,
    league_id: int,
    start_date_time: int,
) -> dict:
    pick_bans = []
    order = 0
    hero_id = 1
    for i in range(7):
        pick_bans.append(
            {
                "isRadiant": i % 2 == 0,
                "order": order,
                "bannedHeroId": hero_id,
                "playerIndex": 0,
                "wasBannedSuccessfully": True,
                "isPick": False,
                "isCaptain": None,
                "letter": None,
                "heroId": hero_id,
            }
        )
        order += 1
        hero_id += 1
    for _ in range(5):
        pick_bans.append(
            {
                "isRadiant": True,
                "order": order,
                "bannedHeroId": None,
                "playerIndex": 0,
                "wasBannedSuccessfully": None,
                "isPick": True,
                "isCaptain": None,
                "letter": None,
                "heroId": hero_id,
            }
        )
        order += 1
        hero_id += 1
        pick_bans.append(
            {
                "isRadiant": False,
                "order": order,
                "bannedHeroId": None,
                "playerIndex": 0,
                "wasBannedSuccessfully": None,
                "isPick": True,
                "isCaptain": None,
                "letter": None,
                "heroId": hero_id,
            }
        )
        order += 1
        hero_id += 1

    players = [
        {
            "isRadiant": True,
            "playerSlot": i,
            "steamAccount": {"id": pid},
            "heroId": 1,
            "steamAccountId": pid,
        }
        for i, pid in enumerate(RADIANT_IDS)
    ] + [
        {
            "isRadiant": False,
            "playerSlot": 128 + i,
            "steamAccount": {"id": pid},
            "heroId": 1,
            "steamAccountId": pid,
        }
        for i, pid in enumerate(DIRE_IDS)
    ]

    return {
        "id": match_id,
        "startDateTime": start_date_time,
        "endDateTime": start_date_time + 3600,
        "leagueId": league_id,
        "league": {"id": league_id, "name": None, "displayName": "Test League"},
        "seriesId": 1010717,
        "series": {
            "id": 1010717,
            "type": "BEST_OF_FIVE",
            "teamOneId": 8261500,
            "teamTwoId": 9247354,
            "teamOneWinCount": 2,
            "teamTwoWinCount": 3,
            "winningTeamId": 9247354,
        },
        "radiantTeamId": 8261500,
        "direTeamId": 9247354,
        "radiantTeam": {"id": 8261500, "tag": "XG", "name": "Xtreme Gaming"},
        "direTeam": {"id": 9247354, "tag": "FLCN", "name": "Team Falcons"},
        "didRadiantWin": False,
        "tournamentId": None,
        "tournamentRound": None,
        "players": players,
        "pickBans": pick_bans,
        "durationSeconds": 2734,
        "gameVersionId": 176,
    }
