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
        "patch": None,
        "radiant_team_id": 8261500,
        "radiant_team_name": "Xtreme Gaming",
        "radiant_player_ids": (898754153, 137129583, 129958758, 157475523, 94296097),
        "dire_team_id": 9247354,
        "dire_team_name": "Team Falcons",
        "dire_player_ids": (10366616, 100058342, 898455820, 183719386, 25907144),
        "draft_events": make_draft_events(num_bans),
        "radiant_win": False,
        "duration_seconds": 2400,
    }
    fields.update(overrides)
    return CanonicalMatch(**fields)


def test_constructs_valid_canonical_match() -> None:
    match = make_match()
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
        ("draft_events", InformationAvailability.DRAFT),
        ("radiant_win", InformationAvailability.POST_MATCH),
        ("duration_seconds", InformationAvailability.POST_MATCH),
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
