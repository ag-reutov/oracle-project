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
import json
from pathlib import Path

import pytest

from dota_predictor.data.canonical_schema import (
    CanonicalMatchError,
    DraftAction,
    MatchLane,
    MatchPlayerBoxScore,
    MatchPlayerPosition,
    MatchPlayerRole,
    Side,
)
from dota_predictor.data.stratz_mapping import (
    canonical_match_from_stratz,
    draft_event_from_stratz_pick_ban,
)

ANOMALY_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "stratz_anomalies"
REJECTED_ANOMALY_FIXTURES_DIR = ANOMALY_FIXTURES_DIR / "rejected"


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
    *,
    is_radiant: bool,
    player_slot: int,
    steam_account_id: int,
    hero_id: int,
    position: str | None = None,
    lane: str | None = None,
    role: str | None = None,
) -> dict:
    row = {
        "isRadiant": is_radiant,
        "playerSlot": player_slot,
        "steamAccount": {"id": steam_account_id},
        "heroId": hero_id,
        "steamAccountId": steam_account_id,
    }
    if position is not None:
        row["position"] = position
    if lane is not None:
        row["lane"] = lane
    if role is not None:
        row["role"] = role
    return row


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
        _player_row(
            is_radiant=True,
            player_slot=i,
            steam_account_id=pid,
            hero_id=8 + i * 2,
        )
        for i, pid in enumerate(RADIANT_IDS)
    ] + [
        _player_row(
            is_radiant=False,
            player_slot=128 + i,
            steam_account_id=pid,
            hero_id=9 + i * 2,
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
    # Raw `order` (99) deliberately differs from the caller-supplied
    # canonical `sequence` (3) to prove the two are independent: the
    # mapped event's sequence reflects the caller's positional assignment,
    # not the row's own `order` value.
    row = _pick_ban_row(order=99, is_pick=True, is_radiant=False, hero_id=89)
    event = draft_event_from_stratz_pick_ban(row, sequence=3)
    assert event.sequence == 3
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
    event = draft_event_from_stratz_pick_ban(row, sequence=0)
    assert event.action is DraftAction.BAN
    assert event.side is Side.DIRE
    assert event.hero_id == 3
    assert event.was_successful is True


def test_draft_event_falls_back_to_banned_hero_id() -> None:
    row = _pick_ban_row(order=0, is_pick=False, is_radiant=True, hero_id=42)
    row["heroId"] = None  # only bannedHeroId populated
    event = draft_event_from_stratz_pick_ban(row, sequence=0)
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
    assert match.game_version_id == 176
    assert match.radiant_team_id == 8261500
    assert match.radiant_team_name_observed == "Xtreme Gaming"
    assert list(match.radiant_player_ids) == RADIANT_IDS
    assert list(match.radiant_hero_ids) == [8, 10, 12, 14, 16]
    assert match.dire_team_id == 9247354
    assert list(match.dire_player_ids) == DIRE_IDS
    assert list(match.dire_hero_ids) == [9, 11, 13, 15, 17]
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


def test_canonical_match_from_stratz_normalizes_nonzero_gapped_order_to_sequence() -> (
    None
):
    """Raw STRATZ orders like 10, 20, 30, ... must normalize to 0, 1, 2, ...

    Canonical `sequence` is a normalized position derived by sorting and
    enumerating, not a copy of the raw `order` value. This uses a
    strictly-increasing but non-zero-based, non-gap-free raw ordering to
    prove the mapper does not depend on `order` already being zero-based
    or gap-free.
    """
    raw = build_raw_match()
    original_hero_sequence = [row["heroId"] for row in raw["pickBans"]]
    for row in raw["pickBans"]:
        row["order"] = row["order"] * 10 + 10  # 0,1,2,... -> 10,20,30,...

    match = canonical_match_from_stratz(raw)

    sequences = [event.sequence for event in match.draft_events]
    assert sequences == list(range(len(match.draft_events)))
    # Relative order is preserved: normalizing must not reshuffle events.
    assert [event.hero_id for event in match.draft_events] == original_hero_sequence


def test_canonical_match_from_stratz_rejects_duplicate_source_order() -> None:
    raw = build_raw_match()
    # Force two rows to share the same raw STRATZ `order`, which makes the
    # source ordering ambiguous. The draft is unusable, so the match is
    # represented with an absent draft rather than being dropped.
    raw["pickBans"][1]["order"] = raw["pickBans"][0]["order"]
    match = canonical_match_from_stratz(raw)
    assert match.draft_complete is False
    assert match.draft_events == ()


def test_canonical_match_from_stratz_rejects_missing_order() -> None:
    raw = build_raw_match()
    del raw["pickBans"][0]["order"]
    match = canonical_match_from_stratz(raw)
    assert match.draft_complete is False
    assert match.draft_events == ()


def test_canonical_match_from_stratz_maps_player_hero_id_from_players_not_pick_order() -> (
    None
):
    raw = build_raw_match()
    radiant_picks = [
        row["heroId"] for row in raw["pickBans"] if row["isPick"] and row["isRadiant"]
    ]
    reversed_heroes = list(reversed(radiant_picks))
    radiant_players = [player for player in raw["players"] if player["isRadiant"]]
    for player, hero_id in zip(radiant_players, reversed_heroes, strict=True):
        player["heroId"] = hero_id

    match = canonical_match_from_stratz(raw)

    assert list(match.radiant_hero_ids) == reversed_heroes
    assert set(match.radiant_hero_ids) == set(radiant_picks)
    assert list(match.radiant_hero_ids) != list(match.radiant_final_hero_ids)


def test_canonical_match_from_stratz_raises_on_missing_player_hero_id() -> None:
    raw = build_raw_match()
    raw["players"][0]["heroId"] = None
    with pytest.raises(CanonicalMatchError, match="heroId"):
        canonical_match_from_stratz(raw)


@pytest.mark.parametrize(
    "fixture_path",
    sorted(ANOMALY_FIXTURES_DIR.glob("*.json"))
    if ANOMALY_FIXTURES_DIR.is_dir()
    else [],
    ids=lambda path: path.name,
)
def test_real_anomaly_fixtures_still_map_successfully(fixture_path: Path) -> None:
    """Real STRATZ evidence fixtures from the verification probe must keep
    mapping successfully across schema/mapper changes.

    These are genuine professional-match payloads preserved because they
    exercise real edge-case behavior (non-chronological `series.matches`,
    a `heroId`-null/`bannedHeroId`-fallback ban row, null
    `tournamentId`/`tournamentRound`, a shorter pre-7.30 draft) -- not
    because they are invalid. They must continue to map without error.
    """
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    match = canonical_match_from_stratz(raw)
    assert match.match_id == raw["id"]
    assert len(match.radiant_final_hero_ids) == 5
    assert len(match.dire_final_hero_ids) == 5


@pytest.mark.parametrize(
    "fixture_path",
    sorted(REJECTED_ANOMALY_FIXTURES_DIR.glob("*.json"))
    if REJECTED_ANOMALY_FIXTURES_DIR.is_dir()
    else [],
    ids=lambda path: path.name,
)
def test_rejected_anomaly_fixtures_map_without_a_draft(
    fixture_path: Path,
) -> None:
    """Real STRATZ payloads with `pickBans: null` now canonicalize.

    These are completed professional matches where STRATZ returns
    `pickBans: null` even though `players[].heroId` is populated. Final
    hero lineups alone cannot reconstruct draft order, so canonicalization
    must not fabricate draft events; instead the match is canonical with
    an absent draft (`draft_complete=False`, empty `draft_events`).
    """
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert raw.get("pickBans") in (None, [])
    match = canonical_match_from_stratz(raw)
    assert match.match_id == raw["id"]
    assert match.draft_complete is False
    assert match.draft_events == ()
    assert len(match.radiant_player_ids) == 5
    assert len(match.dire_player_ids) == 5
    assert len(match.radiant_hero_ids) == 5
    assert len(match.dire_hero_ids) == 5


def test_null_pick_bans_map_to_absent_draft() -> None:
    raw = build_raw_match()
    raw["pickBans"] = None
    match = canonical_match_from_stratz(raw)
    assert match.draft_complete is False
    assert match.draft_events == ()
    assert list(match.radiant_hero_ids) == [8, 10, 12, 14, 16]


def test_empty_pick_bans_map_to_absent_draft() -> None:
    raw = build_raw_match()
    raw["pickBans"] = []
    match = canonical_match_from_stratz(raw)
    assert match.draft_complete is False
    assert match.draft_events == ()


def test_malformed_pick_bans_map_to_absent_draft() -> None:
    raw = build_raw_match()
    raw["pickBans"][1]["order"] = raw["pickBans"][0]["order"]
    match = canonical_match_from_stratz(raw)
    assert match.draft_complete is False
    assert match.draft_events == ()


def test_partial_draft_map_to_absent_draft_not_fabricated() -> None:
    raw = build_raw_match()
    raw["pickBans"] = raw["pickBans"][:6]
    match = canonical_match_from_stratz(raw)
    assert match.draft_complete is False
    assert match.draft_events == ()


def test_complete_draft_maps_draft_complete_true() -> None:
    raw = build_raw_match()
    match = canonical_match_from_stratz(raw)
    assert match.draft_complete is True
    assert len(match.draft_events) == 17


def test_draft_less_match_still_requires_core_identity() -> None:
    raw = build_raw_match()
    raw["pickBans"] = None
    raw["players"][0]["heroId"] = None
    with pytest.raises(CanonicalMatchError, match="heroId"):
        canonical_match_from_stratz(raw)


def test_paired_matches_confirm_radiant_dire_side_swap_not_inverted() -> None:
    """Two real games of one series with Radiant/Dire swapped map correctly.

    `8461956309` (game 5) and `8461854486` (game 4) are the same real
    STRATZ `BEST_OF_FIVE` series (`seriesId` 1010717); Team Falcons is
    Dire in game 5 but Radiant in game 4, and vice versa for Xtreme
    Gaming. This proves the mapper derives side from each match's own
    `radiantTeamId`/`direTeamId`/`isRadiant` fields rather than from any
    assumption that a team keeps one side across a series, and that
    there is no accidental Radiant/Dire inversion in either direction.
    """
    game_five = json.loads(
        (
            ANOMALY_FIXTURES_DIR
            / "8461956309_ti2025_tournamentid_and_tournamentround_null.json"
        ).read_text(encoding="utf-8")
    )
    game_four = json.loads(
        (
            ANOMALY_FIXTURES_DIR
            / "8461854486_ti2025_radiant_dire_side_swap_within_series.json"
        ).read_text(encoding="utf-8")
    )

    match_five = canonical_match_from_stratz(game_five)
    match_four = canonical_match_from_stratz(game_four)

    assert match_five.series_id == match_four.series_id
    # Same two teams, but sides are swapped between the two games.
    assert match_five.radiant_team_id == match_four.dire_team_id
    assert match_five.dire_team_id == match_four.radiant_team_id
    # Same two rosters (as sets, since lineups can vary in playerSlot
    # order across games), also swapped.
    assert set(match_five.radiant_player_ids) == set(match_four.dire_player_ids)
    assert set(match_five.dire_player_ids) == set(match_four.radiant_player_ids)

    # Each raw player's `isVictory` must agree with `radiant_win` combined
    # with that player's `isRadiant`, in both games -- the ground-truth
    # check for "no accidental inversion".
    for raw_match, canonical in ((game_five, match_five), (game_four, match_four)):
        for player in raw_match["players"]:
            expected_victory = (
                canonical.radiant_win
                if player["isRadiant"]
                else not canonical.radiant_win
            )
            assert player["isVictory"] == expected_victory


def test_missing_position_fields_map_to_none_and_do_not_fail() -> None:
    match = canonical_match_from_stratz(build_raw_match())
    assert match.radiant_positions == (None, None, None, None, None)
    assert match.dire_lanes == (None, None, None, None, None)
    assert match.radiant_roles == (None, None, None, None, None)


def test_maps_match_level_position_lane_role_from_player_objects() -> None:
    raw = build_raw_match()
    radiant = [player for player in raw["players"] if player["isRadiant"]]
    labels = [
        ("POSITION_1", "SAFE_LANE", "CORE"),
        ("POSITION_2", "MID_LANE", "CORE"),
        ("POSITION_3", "OFF_LANE", "CORE"),
        ("POSITION_4", "OFF_LANE", "LIGHT_SUPPORT"),
        ("POSITION_5", "SAFE_LANE", "HARD_SUPPORT"),
    ]
    for player, (position, lane, role) in zip(radiant, labels, strict=True):
        player["position"] = position
        player["lane"] = lane
        player["role"] = role
    match = canonical_match_from_stratz(raw)
    assert match.radiant_positions == (
        MatchPlayerPosition.POSITION_1,
        MatchPlayerPosition.POSITION_2,
        MatchPlayerPosition.POSITION_3,
        MatchPlayerPosition.POSITION_4,
        MatchPlayerPosition.POSITION_5,
    )
    assert match.radiant_lanes[1] is MatchLane.MID_LANE
    assert match.radiant_roles[-1] is MatchPlayerRole.HARD_SUPPORT


def test_unknown_position_is_preserved_not_guessed_from_slot() -> None:
    raw = build_raw_match()
    raw["players"][0]["position"] = "UNKNOWN"
    raw["players"][0]["playerSlot"] = 0
    match = canonical_match_from_stratz(raw)
    assert match.radiant_positions[0] is MatchPlayerPosition.UNKNOWN


def test_pro_steam_account_position_is_ignored() -> None:
    raw = build_raw_match()
    raw["players"][0]["proSteamAccount"] = {
        "position": "POSITION_5",
        "teamId": 999,
    }
    match = canonical_match_from_stratz(raw)
    assert match.radiant_positions[0] is None


def test_unsupported_position_value_fails_closed() -> None:
    raw = build_raw_match()
    raw["players"][0]["position"] = "CARRY"
    with pytest.raises(CanonicalMatchError, match="MatchPlayerPosition"):
        canonical_match_from_stratz(raw)


def test_missing_box_score_fields_map_to_none_and_do_not_fail() -> None:
    match = canonical_match_from_stratz(build_raw_match())
    assert match.radiant_box_scores[0] == MatchPlayerBoxScore()
    assert match.dire_box_scores[4].kills is None
    assert match.radiant_player_ids == tuple(RADIANT_IDS)
    assert match.radiant_positions == (None, None, None, None, None)


def test_maps_box_score_scalars_preserving_zero_and_null() -> None:
    raw = build_raw_match()
    raw["players"][0]["kills"] = 0
    raw["players"][0]["deaths"] = 7
    raw["players"][0]["assists"] = 12
    raw["players"][0]["goldPerMinute"] = 250
    raw["players"][0]["experiencePerMinute"] = 300
    raw["players"][0]["numLastHits"] = 0
    raw["players"][0]["numDenies"] = 0
    raw["players"][0]["networth"] = 4500
    raw["players"][0]["heroDamage"] = 0
    raw["players"][0]["towerDamage"] = None
    raw["players"][0]["heroHealing"] = 1500
    raw["players"][0]["level"] = 12
    match = canonical_match_from_stratz(raw)
    score = match.radiant_box_scores[0]
    assert score.kills == 0
    assert score.deaths == 7
    assert score.num_last_hits == 0
    assert score.hero_damage == 0
    assert score.tower_damage is None
    assert score.hero_healing == 1500
    assert score.gold_per_minute == 250
    assert score.level == 12
    assert match.dire_box_scores[0].kills is None


def test_non_integer_box_score_fails_closed() -> None:
    raw = build_raw_match()
    raw["players"][0]["kills"] = 1.5
    with pytest.raises(CanonicalMatchError, match="kills"):
        canonical_match_from_stratz(raw)
