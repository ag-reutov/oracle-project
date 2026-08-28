"""Tests for mapping raw STRATZ match payloads to `CanonicalMatch`.

The synthetic payload below mirrors the real shape captured in
`data/raw/stratz_probe_matches.json`, with `durationSeconds` and
`gameVersionId` added: the existing probe's field selection did not request
those two (see task report), so they are not present in the committed
sample even though they exist on STRATZ's `MatchType` per schema
introspection.
"""

from __future__ import annotations

import copy

import pytest

from dota_predictor.data.canonical_schema import CanonicalMatchError, DraftAction, Side
from dota_predictor.data.stratz_mapping import (
    canonical_match_from_stratz,
    draft_event_from_stratz_pick_ban,
)


def _pick_ban_row(
    *,
    order: int,
    is_pick: bool,
    is_radiant: bool,
    hero_id: int,
    was_banned_successfully: bool | None = None,
) -> dict:
    return {
        "isRadiant": is_radiant,
        "order": order,
        "bannedHeroId": None if is_pick else hero_id,
        "playerIndex": 0,
        "wasBannedSuccessfully": was_banned_successfully,
        "isPick": is_pick,
        "isCaptain": None,
        "letter": None,
        "heroId": hero_id,
    }


def _player_row(
    *, is_radiant: bool, player_slot: int, steam_account_id: int, hero_id: int
) -> dict:
    return {
        "isRadiant": is_radiant,
        "playerSlot": player_slot,
        "steamAccount": {"id": steam_account_id},
        "heroId": hero_id,
        "steamAccountId": steam_account_id,
    }


RADIANT_IDS = [898754153, 137129583, 129958758, 157475523, 94296097]
DIRE_IDS = [10366616, 100058342, 898455820, 183719386, 25907144]


def build_raw_match() -> dict:
    pick_bans = []
    order = 0
    hero_id = 1
    for _ in range(7):
        pick_bans.append(
            _pick_ban_row(
                order=order, is_pick=False, is_radiant=order % 2 == 0, hero_id=hero_id
            )
        )
        order += 1
        hero_id += 1
    for i in range(5):
        pick_bans.append(
            _pick_ban_row(order=order, is_pick=True, is_radiant=True, hero_id=hero_id)
        )
        order += 1
        hero_id += 1
        pick_bans.append(
            _pick_ban_row(order=order, is_pick=True, is_radiant=False, hero_id=hero_id)
        )
        order += 1
        hero_id += 1

    players = [
        _player_row(is_radiant=True, player_slot=i, steam_account_id=pid, hero_id=1)
        for i, pid in enumerate(RADIANT_IDS)
    ] + [
        _player_row(
            is_radiant=False, player_slot=128 + i, steam_account_id=pid, hero_id=1
        )
        for i, pid in enumerate(DIRE_IDS)
    ]

    return {
        "id": 8461956309,
        "startDateTime": 1757872818,
        "leagueId": 18324,
        "league": {"id": 18324, "name": None, "displayName": "The International 2025"},
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
        # Not part of the existing probe's field selection; added here to
        # exercise the mapping since these are required/optional canonical
        # fields per schema introspection.
        "durationSeconds": 2734,
        "gameVersionId": 176,
    }


def test_draft_event_from_stratz_pick_ban_maps_pick() -> None:
    row = _pick_ban_row(order=7, is_pick=True, is_radiant=False, hero_id=89)
    event = draft_event_from_stratz_pick_ban(row)
    assert event.sequence == 7
    assert event.action is DraftAction.PICK
    assert event.side is Side.DIRE
    assert event.hero_id == 89
    assert event.was_successful is None


def test_draft_event_from_stratz_pick_ban_maps_ban() -> None:
    row = _pick_ban_row(
        order=0,
        is_pick=False,
        is_radiant=False,
        hero_id=3,
        was_banned_successfully=True,
    )
    event = draft_event_from_stratz_pick_ban(row)
    assert event.action is DraftAction.BAN
    assert event.side is Side.DIRE
    assert event.hero_id == 3
    assert event.was_successful is True


def test_draft_event_falls_back_to_banned_hero_id() -> None:
    row = _pick_ban_row(order=0, is_pick=False, is_radiant=True, hero_id=42)
    row["heroId"] = None  # only bannedHeroId populated
    event = draft_event_from_stratz_pick_ban(row)
    assert event.hero_id == 42


def test_canonical_match_from_stratz_maps_realistic_payload() -> None:
    raw = build_raw_match()
    match = canonical_match_from_stratz(raw)

    assert match.match_id == 8461956309
    assert match.league_id == 18324
    assert match.league_name == "The International 2025"
    assert match.series_id == 1010717
    assert match.series_type == "BEST_OF_FIVE"
    assert match.game_number_in_series is None
    assert match.patch == 176
    assert match.radiant_team_id == 8261500
    assert match.radiant_team_name == "Xtreme Gaming"
    assert list(match.radiant_player_ids) == RADIANT_IDS
    assert match.dire_team_id == 9247354
    assert list(match.dire_player_ids) == DIRE_IDS
    assert match.radiant_win is False
    assert match.duration_seconds == 2734
    assert len(match.draft_events) == 17  # 7 bans + 10 picks in this fixture
    assert len(match.radiant_final_hero_ids) == 5
    assert len(match.dire_final_hero_ids) == 5


def test_canonical_match_from_stratz_raises_on_missing_duration() -> None:
    raw = build_raw_match()
    del raw["durationSeconds"]
    with pytest.raises(CanonicalMatchError, match="durationSeconds"):
        canonical_match_from_stratz(raw)


def test_canonical_match_from_stratz_raises_on_missing_league_id() -> None:
    raw = build_raw_match()
    raw["leagueId"] = None
    raw["league"] = None
    with pytest.raises(CanonicalMatchError, match="leagueId"):
        canonical_match_from_stratz(raw)


def test_canonical_match_from_stratz_is_deterministic() -> None:
    raw = build_raw_match()
    match_a = canonical_match_from_stratz(copy.deepcopy(raw))
    match_b = canonical_match_from_stratz(copy.deepcopy(raw))
    assert match_a == match_b
