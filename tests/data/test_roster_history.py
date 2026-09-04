"""Tests for the pure observed-roster-history derivation logic (Slice 4).

These are database-free tests of `dota_predictor.data.roster_history`:
deterministic observed player-team spell derivation (order by
`(start_time, match_id, team_id)`, new spell only on team change, A -> B -> A
as three spells, time gaps not splitting spells, one-match spells retained)
and the explicit lineup-cardinality classification. Unresolved identities
are never fabricated: the spell derivation raises rather than silently
forming a spell from a NULL player/team id. DB-touching behavior (canonical
fact collection, the research views, and the audit) is covered separately in
`tests/research/test_roster_views.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dota_predictor.data.roster_history import (
    LineupSummary,
    classify_lineup,
    derive_observed_spells,
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)
T4 = datetime(2024, 4, 1, tzinfo=UTC)


def _spells(observations):
    return derive_observed_spells(observations)


# --- observed spell derivation -------------------------------------------------


def test_player_staying_on_one_team_yields_one_spell():
    spells = _spells(
        [
            (10, 100, 1, T1),
            (10, 100, 2, T2),
            (10, 100, 3, T3),
        ]
    )
    assert len(spells) == 1
    spell = spells[0]
    assert (spell.player_id, spell.team_id, spell.spell_index) == (10, 100, 1)
    assert spell.observed_match_count == 3
    assert spell.first_seen_at == T1
    assert spell.first_match_id == 1
    assert spell.last_seen_at == T3
    assert spell.last_match_id == 3


def test_player_moving_a_to_b_yields_two_spells():
    spells = _spells(
        [
            (10, 100, 1, T1),
            (10, 100, 2, T2),
            (10, 200, 3, T3),
        ]
    )
    assert [(s.team_id, s.spell_index, s.observed_match_count) for s in spells] == [
        (100, 1, 2),
        (200, 2, 1),
    ]


def test_player_moving_a_to_b_to_a_yields_three_spells():
    spells = _spells(
        [
            (10, 100, 1, T1),
            (10, 100, 2, T2),
            (10, 200, 3, T3),
            (10, 100, 4, T4),
        ]
    )
    assert [(s.team_id, s.spell_index, s.observed_match_count) for s in spells] == [
        (100, 1, 2),
        (200, 2, 1),
        (100, 3, 1),
    ]
    # The third spell is a NEW spell (spell_index 3), never merged back.
    assert spells[-1].first_match_id == 4
    assert spells[-1].last_match_id == 4


def test_multiple_matches_for_same_team_remain_one_spell():
    spells = _spells(
        [
            (10, 100, 5, T1),
            (10, 100, 7, T2),
            (10, 100, 9, T3),
            (10, 100, 11, T4),
        ]
    )
    assert len(spells) == 1
    assert spells[0].observed_match_count == 4
    assert spells[0].first_match_id == 5
    assert spells[0].last_match_id == 11


def test_long_inactivity_without_other_team_does_not_split_spell():
    """A gap of many months with no intervening team observation must not
    create a new spell: spells change because another team is observed, not
    because an arbitrary number of days elapsed."""
    far_future = datetime(2024, 12, 31, tzinfo=UTC)
    spells = _spells(
        [
            (10, 100, 1, T1),
            (10, 100, 2, far_future),
        ]
    )
    assert len(spells) == 1
    assert spells[0].observed_match_count == 2
    assert spells[0].first_seen_at == T1
    assert spells[0].last_seen_at == far_future


def test_one_match_intermediate_team_is_retained():
    spells = _spells(
        [
            (10, 100, 1, T1),
            (10, 300, 2, T2),
            (10, 100, 3, T3),
        ]
    )
    assert [(s.team_id, s.spell_index, s.observed_match_count) for s in spells] == [
        (100, 1, 1),
        (300, 2, 1),
        (100, 3, 1),
    ]


def test_equal_timestamps_ordered_deterministically_by_match_id():
    """Observations sharing a `start_time` are ordered by `match_id` (then
    `team_id`), independent of input order, so the resulting spells are
    deterministic."""
    forward = [
        (10, 100, 1, T1),
        (10, 200, 2, T1),
        (10, 100, 3, T2),
    ]
    backward = list(reversed(forward))
    fwd = _spells(forward)
    bwd = _spells(backward)
    assert fwd == bwd
    # match_id 1 (team 100) sorts before match_id 2 (team 200), so the
    # player's first observed team is 100 and the switch to 200 is spell 2.
    assert [(s.team_id, s.spell_index) for s in fwd] == [(100, 1), (200, 2), (100, 3)]


def test_spell_derivation_is_deterministic_regardless_of_input_order():
    observations = [
        (10, 100, 1, T1),
        (10, 200, 2, T2),
        (20, 100, 3, T3),
        (10, 100, 4, T4),
    ]
    assert _spells(observations) == _spells(list(reversed(observations)))


def test_spells_sorted_by_player_then_spell_index():
    spells = _spells(
        [
            (20, 100, 1, T1),
            (10, 100, 2, T2),
            (20, 200, 3, T3),
            (10, 100, 4, T4),
        ]
    )
    # Player 10: one spell (team 100, matches 2 and 4). Player 20: two spells
    # (team 100, then team 200). Output is sorted by (player_id, spell_index).
    assert [(s.player_id, s.spell_index) for s in spells] == [(10, 1), (20, 1), (20, 2)]


def test_empty_input_yields_no_spells():
    assert _spells([]) == []


def test_null_player_id_raises_not_fabricated():
    with pytest.raises(ValueError):
        _spells([(None, 100, 1, T1)])


def test_null_team_id_raises_not_fabricated():
    with pytest.raises(ValueError):
        _spells([(10, None, 1, T1)])


def test_duplicate_match_both_sides_is_deterministic_not_merged():
    """A degenerate observation with the same player in the same match on
    two teams is ordered deterministically by `team_id` and produces two
    spells -- it is never silently merged into one membership."""
    spells = _spells(
        [
            (10, 100, 1, T1),
            (10, 200, 1, T1),
        ]
    )
    assert [(s.team_id, s.spell_index, s.observed_match_count) for s in spells] == [
        (100, 1, 1),
        (200, 2, 1),
    ]


# --- lineup cardinality classification ------------------------------------------


def test_lineup_exactly_five_resolved():
    lineup = classify_lineup([1, 2, 3, 4, 5])
    assert isinstance(lineup, LineupSummary)
    assert lineup.n_players == 5
    assert lineup.n_resolved_players == 5
    assert lineup.n_null_player_ids == 0
    assert lineup.n_distinct_players == 5
    assert lineup.has_exactly_five
    assert lineup.is_complete_five
    assert not lineup.has_fewer_than_five
    assert not lineup.has_more_than_five
    assert not lineup.has_duplicate_players
    assert lineup.lineup_player_ids == (1, 2, 3, 4, 5)
    assert lineup.lineup_key == "1,2,3,4,5"


def test_lineup_fewer_than_five_detected():
    lineup = classify_lineup([1, 2, 3])
    assert lineup.n_resolved_players == 3
    assert lineup.has_fewer_than_five
    assert not lineup.has_exactly_five
    assert not lineup.is_complete_five
    assert lineup.lineup_key == "1,2,3"


def test_lineup_more_than_five_detected():
    lineup = classify_lineup([1, 2, 3, 4, 5, 6])
    assert lineup.n_resolved_players == 6
    assert lineup.has_more_than_five
    assert not lineup.has_exactly_five


def test_lineup_duplicate_players_detected():
    lineup = classify_lineup([1, 2, 2, 3, 4])
    assert lineup.n_resolved_players == 5
    assert lineup.n_distinct_players == 4
    assert lineup.has_duplicate_players
    assert not lineup.is_complete_five
    assert lineup.lineup_player_ids == (1, 2, 3, 4)
    assert lineup.lineup_key == "1,2,3,4"


def test_lineup_null_player_ids_detected():
    lineup = classify_lineup([1, 2, 3, None, None])
    assert lineup.n_players == 5
    assert lineup.n_resolved_players == 3
    assert lineup.n_null_player_ids == 2
    assert lineup.has_fewer_than_five
    assert not lineup.is_complete_five


def test_lineup_sorted_deterministic_regardless_of_input_order():
    a = classify_lineup([3, 1, 5, 2, 4])
    b = classify_lineup([4, 2, 5, 1, 3])
    assert a.lineup_key == b.lineup_key == "1,2,3,4,5"
    assert a.lineup_player_ids == b.lineup_player_ids


def test_lineup_with_no_resolved_ids_has_null_key():
    lineup = classify_lineup([None, None])
    assert lineup.n_resolved_players == 0
    assert lineup.lineup_key is None
    assert lineup.lineup_player_ids == ()