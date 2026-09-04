"""Tests for the canonical historical match schema.

Fixtures deliberately use different total draft-event counts across tests
to demonstrate that no validator assumes one fixed historical pick/ban
structure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from itertools import cycle

import pytest

from dota_predictor.data.canonical_schema import (
    FIELD_INFORMATION_AVAILABILITY,
    CanonicalMatch,
    CanonicalMatchError,
    DraftAction,
    DraftEvent,
    InformationAvailability,
    MatchLane,
    MatchPlayerBoxScore,
    MatchPlayerPosition,
    MatchPlayerRole,
    Side,
)

START_TIME = datetime(2025, 8, 28, 12, 0, 0, tzinfo=UTC)


def make_draft_events(num_bans: int) -> tuple[DraftEvent, ...]:
    """Build a structurally valid draft: `num_bans` bans then 10 picks.

    Hero ids are all distinct so no "actual action" ever repeats a hero.
    The exact interleaving of bans/picks is irrelevant to validation; only
    ordering, hero uniqueness, and pick counts per side are enforced.
    """
    sides = cycle([Side.RADIANT, Side.DIRE])
    events: list[tuple[DraftAction, Side, int]] = []
    hero_id = 1
    for _ in range(num_bans):
        events.append((DraftAction.BAN, next(sides), hero_id))
        hero_id += 1
    for _ in range(5):
        events.append((DraftAction.PICK, Side.RADIANT, hero_id))
        hero_id += 1
        events.append((DraftAction.PICK, Side.DIRE, hero_id))
        hero_id += 1
    return tuple(
        DraftEvent(sequence=i, action=action, side=side, hero_id=hid)
        for i, (action, side, hid) in enumerate(events)
    )


def make_match(*, num_bans: int = 14, **overrides: object) -> CanonicalMatch:
    fields: dict[str, object] = {
        "match_id": 8461956309,
        "start_time": START_TIME,
        "league_id": 18324,
        "league_name": "The International 2025",
        "series_id": 1010717,
        "series_type": "BEST_OF_FIVE",
        "game_number_in_series": None,
        "game_version_id": None,
        "radiant_team_id": 8261500,
        "radiant_team_name_observed": "Xtreme Gaming",
        "radiant_player_ids": (898754153, 137129583, 129958758, 157475523, 94296097),
        "dire_team_id": 9247354,
        "dire_team_name_observed": "Team Falcons",
        "dire_player_ids": (10366616, 100058342, 898455820, 183719386, 25907144),
        "draft_events": make_draft_events(num_bans),
        "radiant_win": False,
        "duration_seconds": 2400,
    }
    fields.update(overrides)
    if "radiant_hero_ids" not in fields:
        draft_events = fields["draft_events"]
        fields["radiant_hero_ids"] = tuple(
            event.hero_id
            for event in draft_events
            if event.action is DraftAction.PICK and event.side is Side.RADIANT
        )
    if "dire_hero_ids" not in fields:
        draft_events = fields["draft_events"]
        fields["dire_hero_ids"] = tuple(
            event.hero_id
            for event in draft_events
            if event.action is DraftAction.PICK and event.side is Side.DIRE
        )
    return CanonicalMatch(**fields)


def test_constructs_valid_canonical_match() -> None:
    match = make_match()
    assert match.draft_complete is True
    assert match.match_id == 8461956309
    assert match.radiant_win is False
    assert match.duration_seconds == 2400


def test_player_side_representation() -> None:
    match = make_match()
    assert len(match.radiant_player_ids) == 5
    assert len(match.dire_player_ids) == 5
    assert set(match.radiant_player_ids).isdisjoint(match.dire_player_ids)


def test_ordered_draft_event_representation() -> None:
    match = make_match(num_bans=14)
    sequences = [event.sequence for event in match.draft_events]
    assert sequences == list(range(len(match.draft_events)))
    assert len(match.draft_events) == 24  # 14 bans + 10 picks for this fixture


def test_draft_order_is_preserved_not_resorted() -> None:
    events = make_draft_events(num_bans=6)
    match = make_match(num_bans=6, draft_events=events)
    assert match.draft_events == events


@pytest.mark.parametrize(
    "field_name,expected",
    [
        ("match_id", InformationAvailability.PRE_DRAFT),
        ("radiant_team_id", InformationAvailability.PRE_DRAFT),
        ("radiant_player_ids", InformationAvailability.PRE_DRAFT),
        ("radiant_hero_ids", InformationAvailability.DRAFT),
        ("draft_events", InformationAvailability.DRAFT),
        ("radiant_win", InformationAvailability.POST_MATCH),
        ("duration_seconds", InformationAvailability.POST_MATCH),
        ("radiant_positions", InformationAvailability.POST_MATCH),
        ("radiant_lanes", InformationAvailability.POST_MATCH),
        ("radiant_roles", InformationAvailability.POST_MATCH),
        ("dire_positions", InformationAvailability.POST_MATCH),
        ("dire_lanes", InformationAvailability.POST_MATCH),
        ("dire_roles", InformationAvailability.POST_MATCH),
        ("radiant_box_scores", InformationAvailability.POST_MATCH),
        ("dire_box_scores", InformationAvailability.POST_MATCH),
    ],
)
def test_information_availability_classification(
    field_name: str, expected: InformationAvailability
) -> None:
    assert FIELD_INFORMATION_AVAILABILITY[field_name] is expected


def test_no_post_draft_classification_exists() -> None:
    values = set(InformationAvailability)
    assert values == {
        InformationAvailability.PRE_DRAFT,
        InformationAvailability.DRAFT,
        InformationAvailability.POST_MATCH,
    }


def test_rejects_wrong_player_count_per_side() -> None:
    with pytest.raises(CanonicalMatchError, match="exactly 5 players"):
        make_match(radiant_player_ids=(1, 2, 3, 4))


def test_rejects_duplicate_player_id_within_side() -> None:
    with pytest.raises(CanonicalMatchError, match="duplicate player ids"):
        make_match(radiant_player_ids=(1, 2, 3, 4, 4))


def test_rejects_player_on_both_sides() -> None:
    with pytest.raises(CanonicalMatchError, match="both radiant and dire"):
        make_match(
            dire_player_ids=(898754153, 100058342, 898455820, 183719386, 25907144)
        )


def test_rejects_duplicate_hero_across_actual_draft_actions() -> None:
    events = list(make_draft_events(num_bans=6))
    # Force a duplicate: make the last pick re-use the first ban's hero id.
    duplicated = DraftEvent(
        sequence=events[-1].sequence,
        action=DraftAction.PICK,
        side=events[-1].side,
        hero_id=events[0].hero_id,
    )
    events[-1] = duplicated
    with pytest.raises(CanonicalMatchError, match="more than one actual draft action"):
        make_match(num_bans=6, draft_events=tuple(events))


def test_allows_repeated_hero_after_unsuccessful_ban() -> None:
    events = list(make_draft_events(num_bans=6))
    failed_ban_hero_id = 999
    # A failed ban attempt does not remove the hero from the pool, so the
    # same hero may legitimately be picked later without violating
    # uniqueness of *actual* draft actions.
    events[0] = DraftEvent(
        sequence=0,
        action=DraftAction.BAN,
        side=events[0].side,
        hero_id=failed_ban_hero_id,
        was_successful=False,
    )
    events[-1] = DraftEvent(
        sequence=events[-1].sequence,
        action=DraftAction.PICK,
        side=events[-1].side,
        hero_id=failed_ban_hero_id,
    )
    match = make_match(num_bans=6, draft_events=tuple(events))
    assert failed_ban_hero_id in match.radiant_final_hero_ids or (
        failed_ban_hero_id in match.dire_final_hero_ids
    )


def test_rejects_malformed_non_contiguous_draft_sequence() -> None:
    events = list(make_draft_events(num_bans=6))
    events[3] = DraftEvent(
        sequence=99,
        action=events[3].action,
        side=events[3].side,
        hero_id=events[3].hero_id,
    )
    with pytest.raises(CanonicalMatchError, match="sequence == position"):
        make_match(num_bans=6, draft_events=tuple(events))


def test_rejects_duplicate_draft_sequence() -> None:
    events = list(make_draft_events(num_bans=6))
    events[1] = DraftEvent(
        sequence=0,
        action=events[1].action,
        side=events[1].side,
        hero_id=events[1].hero_id,
    )
    with pytest.raises(CanonicalMatchError, match="sequence == position"):
        make_match(num_bans=6, draft_events=tuple(events))


def test_supports_drafts_with_different_valid_event_counts() -> None:
    match_a = make_match(num_bans=6)  # 16 total events
    match_b = make_match(num_bans=14)  # 24 total events
    assert len(match_a.draft_events) == 16
    assert len(match_b.draft_events) == 24
    assert len(match_a.radiant_final_hero_ids) == 5
    assert len(match_b.radiant_final_hero_ids) == 5


def test_player_hero_ids_match_successful_pick_set_not_order() -> None:
    match = make_match(num_bans=4)
    assert set(match.radiant_hero_ids) == set(match.radiant_final_hero_ids)
    assert set(match.dire_hero_ids) == set(match.dire_final_hero_ids)
    reversed_radiant = tuple(reversed(match.radiant_final_hero_ids))
    if reversed_radiant != match.radiant_final_hero_ids:
        reordered = make_match(num_bans=4, radiant_hero_ids=reversed_radiant)
        assert reordered.radiant_hero_ids == reversed_radiant
        assert set(reordered.radiant_hero_ids) == set(reordered.radiant_final_hero_ids)


def test_rejects_duplicate_hero_id_within_side() -> None:
    match = make_match(num_bans=4)
    duplicated = (match.radiant_hero_ids[0],) * 5
    with pytest.raises(CanonicalMatchError, match="duplicate hero ids"):
        make_match(num_bans=4, radiant_hero_ids=duplicated)


def test_rejects_player_hero_set_mismatch_with_picks() -> None:
    match = make_match(num_bans=4)
    wrong = tuple(hero_id + 1000 for hero_id in match.radiant_hero_ids)
    with pytest.raises(CanonicalMatchError, match="does not match successful PICK set"):
        make_match(num_bans=4, radiant_hero_ids=wrong)


def test_short_draft_zero_ban_still_requires_five_heroes_per_side() -> None:
    match = make_match(num_bans=0)
    assert len(match.draft_events) == 10
    assert len(match.radiant_hero_ids) == 5
    assert len(match.dire_hero_ids) == 5
    assert set(match.radiant_hero_ids) == set(match.radiant_final_hero_ids)
    assert set(match.dire_hero_ids) == set(match.dire_final_hero_ids)


def test_rejects_wrong_number_of_final_picks_per_side() -> None:
    events = list(make_draft_events(num_bans=4))
    # Turn one Dire pick into a ban so Dire ends up with only 4 picks.
    for i, event in enumerate(events):
        if event.action is DraftAction.PICK and event.side is Side.DIRE:
            events[i] = DraftEvent(
                sequence=event.sequence,
                action=DraftAction.BAN,
                side=event.side,
                hero_id=9001,
            )
            break
    with pytest.raises(CanonicalMatchError, match="exactly 5 actual dire picks"):
        make_match(num_bans=4, draft_events=tuple(events))


def test_rejects_unsupported_draft_action_type() -> None:
    with pytest.raises(CanonicalMatchError, match="unsupported draft action"):
        DraftEvent(sequence=0, action="SWAP", side=Side.RADIANT, hero_id=1)  # type: ignore[arg-type]


def test_rejects_invalid_acting_side() -> None:
    with pytest.raises(CanonicalMatchError, match="invalid acting side"):
        DraftEvent(sequence=0, action=DraftAction.BAN, side="NEUTRAL", hero_id=1)  # type: ignore[arg-type]


def test_rejects_non_positive_hero_id() -> None:
    with pytest.raises(CanonicalMatchError, match="hero_id must be"):
        DraftEvent(sequence=0, action=DraftAction.BAN, side=Side.RADIANT, hero_id=0)


def test_rejects_invalid_duration() -> None:
    with pytest.raises(CanonicalMatchError, match="duration_seconds must be"):
        make_match(duration_seconds=0)


def test_rejects_non_utc_start_time() -> None:
    naive = datetime(2025, 8, 28, 12, 0, 0)  # noqa: DTZ001 - intentionally naive, for this test
    with pytest.raises(CanonicalMatchError, match="UTC"):
        make_match(start_time=naive)

    offset_aware = datetime(2025, 8, 28, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    with pytest.raises(CanonicalMatchError, match="UTC"):
        make_match(start_time=offset_aware)


def test_rejects_invalid_required_identifiers() -> None:
    with pytest.raises(CanonicalMatchError, match="match_id"):
        make_match(match_id=0)
    with pytest.raises(CanonicalMatchError, match="league_id"):
        make_match(league_id=-1)
    with pytest.raises(
        CanonicalMatchError, match="radiant_team_id and dire_team_id must differ"
    ):
        make_match(dire_team_id=8261500)


def test_final_hero_ids_derived_from_draft_events_in_pick_order() -> None:
    events = make_draft_events(num_bans=2)
    match = make_match(num_bans=2, draft_events=events)
    expected_radiant = tuple(
        event.hero_id
        for event in events
        if event.side is Side.RADIANT and event.action is DraftAction.PICK
    )
    expected_dire = tuple(
        event.hero_id
        for event in events
        if event.side is Side.DIRE and event.action is DraftAction.PICK
    )
    assert match.radiant_final_hero_ids == expected_radiant
    assert match.dire_final_hero_ids == expected_dire
    assert len(match.radiant_final_hero_ids) == 5
    assert len(match.dire_final_hero_ids) == 5


UNIQUE_POSITIONS = (
    MatchPlayerPosition.POSITION_1,
    MatchPlayerPosition.POSITION_2,
    MatchPlayerPosition.POSITION_3,
    MatchPlayerPosition.POSITION_4,
    MatchPlayerPosition.POSITION_5,
)
UNIQUE_LANES = (
    MatchLane.SAFE_LANE,
    MatchLane.MID_LANE,
    MatchLane.OFF_LANE,
    MatchLane.OFF_LANE,
    MatchLane.SAFE_LANE,
)
UNIQUE_ROLES = (
    MatchPlayerRole.CORE,
    MatchPlayerRole.CORE,
    MatchPlayerRole.CORE,
    MatchPlayerRole.LIGHT_SUPPORT,
    MatchPlayerRole.HARD_SUPPORT,
)


def test_missing_position_metadata_does_not_fail_canonicalization() -> None:
    match = make_match()
    assert match.radiant_positions == (None, None, None, None, None)
    assert match.dire_lanes == (None, None, None, None, None)
    assert match.radiant_roles == (None, None, None, None, None)


def test_unknown_position_is_preserved_and_not_coerced() -> None:
    positions = (
        MatchPlayerPosition.POSITION_1,
        MatchPlayerPosition.UNKNOWN,
        MatchPlayerPosition.POSITION_3,
        MatchPlayerPosition.POSITION_4,
        MatchPlayerPosition.POSITION_5,
    )
    match = make_match(radiant_positions=positions)
    assert match.radiant_positions[1] is MatchPlayerPosition.UNKNOWN
    assert match.radiant_positions[1] is not None


def test_unique_explicit_1_to_5_assignment_is_valid() -> None:
    match = make_match(
        radiant_positions=UNIQUE_POSITIONS,
        radiant_lanes=UNIQUE_LANES,
        radiant_roles=UNIQUE_ROLES,
        dire_positions=UNIQUE_POSITIONS,
        dire_lanes=UNIQUE_LANES,
        dire_roles=UNIQUE_ROLES,
    )
    assert match.radiant_positions == UNIQUE_POSITIONS
    assert match.dire_positions == UNIQUE_POSITIONS


def test_duplicate_and_missing_positions_are_not_repaired() -> None:
    duplicates = (
        MatchPlayerPosition.POSITION_1,
        MatchPlayerPosition.POSITION_1,
        MatchPlayerPosition.POSITION_3,
        None,
        MatchPlayerPosition.UNKNOWN,
    )
    match = make_match(radiant_positions=duplicates, dire_positions=duplicates)
    assert match.radiant_positions == duplicates
    assert match.dire_positions.count(MatchPlayerPosition.POSITION_1) == 2


def test_rejects_wrong_position_tuple_length() -> None:
    with pytest.raises(CanonicalMatchError, match="exactly 5 entries"):
        make_match(radiant_positions=(MatchPlayerPosition.POSITION_1,))


def test_missing_box_scores_do_not_fail_canonicalization() -> None:
    match = make_match()
    assert match.radiant_box_scores[0].kills is None
    assert match.dire_box_scores[4].networth is None


def test_zero_box_score_is_preserved() -> None:
    zeros = MatchPlayerBoxScore(kills=0, deaths=0, assists=0, level=1)
    match = make_match(radiant_box_scores=(zeros,) + (MatchPlayerBoxScore(),) * 4)
    assert match.radiant_box_scores[0].kills == 0
    assert match.radiant_box_scores[0].deaths == 0
    assert match.radiant_box_scores[1].kills is None


def test_rejects_wrong_box_score_tuple_length() -> None:
    with pytest.raises(CanonicalMatchError, match="exactly 5 entries"):
        make_match(radiant_box_scores=(MatchPlayerBoxScore(),))


def test_draft_complete_false_allows_absent_draft() -> None:
    heroes = (101, 102, 103, 104, 105)
    match = make_match(
        draft_events=(), draft_complete=False,
        radiant_hero_ids=heroes, dire_hero_ids=heroes,
    )
    assert match.draft_complete is False
    assert match.draft_events == ()
    assert len(match.radiant_hero_ids) == 5
    assert len(match.dire_hero_ids) == 5


def test_draft_complete_true_requires_nonempty_draft() -> None:
    with pytest.raises(CanonicalMatchError, match="non-empty draft_events"):
        make_match(draft_events=(), draft_complete=True)


def test_draft_complete_false_rejects_partial_draft_events() -> None:
    with pytest.raises(CanonicalMatchError, match="draft_events to be empty"):
        make_match(draft_events=make_draft_events(4), draft_complete=False)


def test_draft_complete_false_skips_pick_set_crosscheck() -> None:
    # With no draft, the player hero_id set is not required to match a
    # successful PICK set (there is none). It must still be 5 distinct
    # positive ints.
    heroes = (101, 102, 103, 104, 105)
    match = make_match(
        draft_events=(), draft_complete=False,
        radiant_hero_ids=heroes, dire_hero_ids=heroes,
    )
    assert list(match.radiant_hero_ids) == list(heroes)


def test_draft_complete_false_still_requires_distinct_side_heroes() -> None:
    with pytest.raises(CanonicalMatchError, match="duplicate hero ids"):
        make_match(
            draft_events=(), draft_complete=False,
            radiant_hero_ids=(101, 101, 103, 104, 105),
        )
