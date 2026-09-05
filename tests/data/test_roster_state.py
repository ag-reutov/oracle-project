"""Tests for the pure historical-roster-state derivation logic (Slice 5).

Database-free tests of `dota_predictor.data.roster_state`: the strictly
causal team-match (`TeamRosterState`) and player-team (`PlayerTeamState`)
state derivations, equal-timestamp policy, future-deletion invariance, and
the "current match never enters its own prior counts" rule.

Observation tuples are `(player_id, team_id, match_id, start_time)` --
the same shape `roster_history.collect_player_team_observations` produces
from the canonical warehouse. Unresolved identities are never fabricated:
the derivations raise on a NULL player/team id. DB-touching behavior (the
research views and the SQL-vs-Python cross-check) is covered separately in
`tests/research/test_roster_state_views.py` and
`tests/features/test_roster_continuity_cross_check.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dota_predictor.data.roster_state import (
    PlayerTeamState,
    TeamRosterState,
    check_future_deletion_invariant,
    derive_player_team_state,
    derive_team_roster_state,
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)
T4 = datetime(2024, 4, 1, tzinfo=UTC)
LONG_GAP = datetime(2025, 12, 1, tzinfo=UTC)

TEAM_A, TEAM_B = 100, 200
P1, P2, P3, P4, P5, P6, P7, P8 = 1, 2, 3, 4, 5, 6, 7, 8
ROSTER_A = (P1, P2, P3, P4, P5)


def observations(*matches: tuple[int, int, datetime, tuple[int, ...]]) -> list[tuple[int, int, int, datetime]]:
    """Build observation tuples from (match_id, team_id, start_time, players)."""
    rows: list[tuple[int, int, int, datetime]] = []
    for match_id, team_id, start_time, players in matches:
        for player_id in players:
            rows.append((player_id, team_id, match_id, start_time))
    return rows


def team_states_for(rows, *, team_id: int | None = None) -> dict[int, TeamRosterState]:
    states = derive_team_roster_state(rows)
    return {
        s.match_id: s
        for s in states
        if team_id is None or s.team_id == team_id
    }


def player_state(rows, *, player_id: int, team_id: int, match_id: int) -> PlayerTeamState:
    for s in derive_player_team_state(rows):
        if (s.player_id, s.team_id, s.match_id) == (player_id, team_id, match_id):
            return s
    raise AssertionError(
        f"no player-team state for player={player_id} team={team_id} match={match_id}"
    )


# --- 1. first team match has no previous lineup ---------------------------------


def test_first_team_match_has_no_previous_lineup():
    rows = observations((1, TEAM_A, T1, ROSTER_A))
    state = team_states_for(rows)[1]
    assert state.previous_match_id is None
    assert state.previous_match_at is None
    assert state.previous_lineup_player_ids is None
    assert state.players_retained_from_previous_match is None
    assert state.players_changed_from_previous_match is None
    assert state.same_lineup_as_previous_match is None
    assert state.days_since_team_previous_match is None
    assert state.prior_exact_lineup_match_count == 0


# --- 2-4. retained/changed for complete fives -----------------------------------


def test_same_five_players_retained_equals_five():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, ROSTER_A),
    )
    state = team_states_for(rows)[2]
    assert state.players_retained_from_previous_match == 5
    assert state.players_changed_from_previous_match == 0
    assert state.same_lineup_as_previous_match is True
    assert state.previous_match_id == 1


def test_one_replacement_retained_equals_four():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, (P1, P2, P3, P4, P6)),
    )
    state = team_states_for(rows)[2]
    assert state.players_retained_from_previous_match == 4
    assert state.players_changed_from_previous_match == 1
    assert state.same_lineup_as_previous_match is False


def test_entirely_different_lineup_retained_equals_zero():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, (11, 12, 13, 14, 15)),
    )
    state = team_states_for(rows)[2]
    assert state.players_retained_from_previous_match == 0
    assert state.players_changed_from_previous_match == 5
    assert state.same_lineup_as_previous_match is False


# --- 5-6. prior exact-lineup experience ------------------------------------------


def test_exact_lineup_first_appearance_prior_count_zero():
    rows = observations((1, TEAM_A, T1, ROSTER_A))
    state = team_states_for(rows)[1]
    assert state.prior_exact_lineup_match_count == 0
    assert state.last_exact_lineup_match_id is None
    assert state.last_exact_lineup_at is None


def test_exact_lineup_repeated_increases_prior_count_from_prior_only():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, (P1, P2, P3, P4, P6)),  # different lineup
        (3, TEAM_A, T3, ROSTER_A),  # repeat of match 1
        (4, TEAM_A, T4, ROSTER_A),  # repeat again
    )
    states = team_states_for(rows)
    assert states[1].prior_exact_lineup_match_count == 0
    assert states[2].prior_exact_lineup_match_count == 0  # (P1..P4,P6) never seen
    assert states[3].prior_exact_lineup_match_count == 1
    assert states[3].last_exact_lineup_match_id == 1
    assert states[3].last_exact_lineup_at == T1
    assert states[4].prior_exact_lineup_match_count == 2
    assert states[4].last_exact_lineup_match_id == 3
    assert states[4].last_exact_lineup_at == T3


def test_prior_exact_lineup_is_team_scoped():
    """An identical five on a different team is not prior exact-lineup
    experience for this team."""
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_B, T2, ROSTER_A),
        (3, TEAM_A, T3, ROSTER_A),
    )
    state = team_states_for(rows, team_id=TEAM_A)[3]
    assert state.prior_exact_lineup_match_count == 1  # only match 1 for Team A


# --- 7-9. player-team classification flags ----------------------------------------


def test_player_first_appearance_for_team():
    rows = observations((1, TEAM_A, T1, ROSTER_A))
    state = player_state(rows, player_id=P1, team_id=TEAM_A, match_id=1)
    assert state.prior_team_match_count == 0
    assert state.is_first_observed_match_for_team is True
    assert state.is_returning_to_team is False
    assert state.is_continuing_with_team is False
    assert state.previous_observed_team_id is None
    assert state.previous_observed_match_id is None
    assert state.consecutive_prior_team_appearances == 0
    assert state.days_since_player_previous_match is None
    assert state.days_since_player_previous_team_match is None


def test_player_continuing_with_team():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, ROSTER_A),
    )
    state = player_state(rows, player_id=P1, team_id=TEAM_A, match_id=2)
    assert state.prior_team_match_count == 1
    assert state.first_prior_team_match_at == T1
    assert state.last_prior_team_match_at == T1
    assert state.previous_observed_team_id == TEAM_A
    assert state.previous_observed_match_id == 1
    assert state.is_first_observed_match_for_team is False
    assert state.is_continuing_with_team is True
    assert state.is_returning_to_team is False
    assert state.consecutive_prior_team_appearances == 1


def test_player_returning_to_previous_team_a_to_b_to_a():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_B, T2, (P1, P6, P7, P8, 9)),
        (3, TEAM_A, T3, ROSTER_A),
    )
    state = player_state(rows, player_id=P1, team_id=TEAM_A, match_id=3)
    assert state.prior_team_match_count == 1
    assert state.previous_observed_team_id == TEAM_B
    assert state.previous_observed_match_id == 2
    assert state.is_first_observed_match_for_team is False
    assert state.is_returning_to_team is True
    assert state.is_continuing_with_team is False
    assert state.consecutive_prior_team_appearances == 0


def test_long_gap_without_other_team_remains_continuing():
    """A long inactivity gap with no intervening team observation does not
    change the continuing classification (no semantic gap interpretation)."""
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, LONG_GAP, ROSTER_A),
    )
    state = player_state(rows, player_id=P1, team_id=TEAM_A, match_id=2)
    assert state.previous_observed_team_id == TEAM_A
    assert state.is_continuing_with_team is True
    assert state.is_returning_to_team is False
    assert state.is_first_observed_match_for_team is False
    assert state.days_since_player_previous_match is not None
    assert state.days_since_player_previous_match > 300


# --- 10. team composition counts ---------------------------------------------------


def test_team_composition_counts_reconcile_to_five():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_B, T2, (P1, P6, P7, P8, 9)),
        (3, TEAM_A, T3, ROSTER_A),
    )
    state = team_states_for(rows, team_id=TEAM_A)[3]
    assert state.continuing_player_count == 4
    assert state.first_observed_for_team_count == 0
    assert state.returning_player_count == 1
    assert (
        state.continuing_player_count
        + state.first_observed_for_team_count
        + state.returning_player_count
        == 5
    )


def test_player_with_many_teams_still_continuing_when_previous_is_current():
    """A player who has appeared for several teams is still a continuing
    player when their immediately previous observed team is the current
    team (A -> B -> A -> A)."""
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_B, T2, (P1, P6, P7, P8, 9)),
        (3, TEAM_A, T3, ROSTER_A),
        (4, TEAM_A, T4, ROSTER_A),
    )
    state = player_state(rows, player_id=P1, team_id=TEAM_A, match_id=4)
    assert state.previous_observed_team_id == TEAM_A
    assert state.is_continuing_with_team is True
    assert state.is_returning_to_team is False
    assert state.prior_team_match_count == 2


# --- 11. equal timestamps never create causal precedence ---------------------------


def test_equal_timestamps_do_not_create_team_previous_match():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T1, (P1, P2, P3, P4, P6)),
        (3, TEAM_A, T2, ROSTER_A),
    )
    states = team_states_for(rows, team_id=TEAM_A)
    # Neither match 1 nor match 2 is prior to the other (equal start_time).
    assert states[1].previous_match_id is None
    assert states[2].previous_match_id is None
    assert states[1].prior_exact_lineup_match_count == 0
    assert states[2].prior_exact_lineup_match_count == 0
    # A strictly later match sees the deterministic most-recent prior
    # (match 2 by match_id tie-break among strictly prior rows).
    assert states[3].previous_match_id == 2
    assert states[3].players_retained_from_previous_match == 4


def test_equal_timestamps_do_not_create_player_previous_evidence():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_B, T1, (P1, P6, P7, P8, 9)),
        (3, TEAM_A, T2, ROSTER_A),
    )
    # For match 2 (same time as match 1), match 1 is NOT the previous
    # observed match, so the player is first-observed for Team B, not
    # returning.
    state = player_state(rows, player_id=P1, team_id=TEAM_B, match_id=2)
    assert state.previous_observed_team_id is None
    assert state.prior_team_match_count == 0
    assert state.is_first_observed_match_for_team is True
    assert state.is_returning_to_team is False
    # For the strictly later match 3, previous observed is match 2 (Team B).
    later = player_state(rows, player_id=P1, team_id=TEAM_A, match_id=3)
    assert later.previous_observed_team_id == TEAM_B
    assert later.is_returning_to_team is True


def test_equal_timestamp_derivation_is_deterministic_regardless_of_input_order():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T1, (P1, P2, P3, P4, P6)),
        (3, TEAM_A, T2, ROSTER_A),
    )
    assert derive_team_roster_state(rows) == derive_team_roster_state(
        list(reversed(rows))
    )
    assert derive_player_team_state(rows) == derive_player_team_state(
        list(reversed(rows))
    )


# --- 12-13. future deletion + current match never enters own prior counts ----------


def test_future_deletion_does_not_alter_historical_state():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, (P1, P2, P3, P4, P6)),
        (3, TEAM_A, T3, ROSTER_A),
        (4, TEAM_B, T4, (P1, P6, P7, P8, 9)),
        (5, TEAM_A, T4, ROSTER_A),  # same time as 4, different team
    )
    result = check_future_deletion_invariant(rows)
    assert result["team_state_violations"] == []
    assert result["player_state_violations"] == []
    assert result["matches_checked"] == 5


def test_future_deletion_sample_also_holds():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, (P1, P2, P3, P4, P6)),
        (3, TEAM_A, T3, ROSTER_A),
        (4, TEAM_A, T4, ROSTER_A),
    )
    result = check_future_deletion_invariant(rows, max_checks=2)
    assert result["matches_checked"] == 2
    assert result["team_state_violations"] == []
    assert result["player_state_violations"] == []


def test_current_match_never_enters_its_own_prior_counts():
    """A single-match corpus proves the current match contributes nothing to
    its own prior counts: every prior count is zero / NULL."""
    rows = observations((1, TEAM_A, T1, ROSTER_A))
    state = team_states_for(rows)[1]
    assert state.prior_exact_lineup_match_count == 0
    assert state.previous_match_id is None
    for player_id in ROSTER_A:
        ps = player_state(rows, player_id=player_id, team_id=TEAM_A, match_id=1)
        assert ps.prior_team_match_count == 0
        assert ps.is_first_observed_match_for_team is True
        assert ps.consecutive_prior_team_appearances == 0


def test_future_player_roster_change_does_not_alter_earlier_retained():
    """Deleting a future match that changes a player's team must not change
    an earlier team's retained count or an earlier player's flags."""
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, (P1, P2, P3, P4, P6)),
        (3, TEAM_B, T3, (P1, P6, P7, P8, 9)),
    )
    result = check_future_deletion_invariant(rows)
    assert result["team_state_violations"] == []
    assert result["player_state_violations"] == []


# --- 14. malformed/incomplete lineups stay explicit ---------------------------------


def test_incomplete_lineup_remains_explicit():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, (P1, P2, P3)),  # only three resolved players
    )
    state = team_states_for(rows)[2]
    assert state.is_complete_five is False
    assert state.n_resolved_players == 3
    assert state.lineup_player_ids == (P1, P2, P3)
    assert state.players_retained_from_previous_match is None
    assert state.players_changed_from_previous_match is None
    assert state.same_lineup_as_previous_match is None
    assert state.prior_exact_lineup_match_count is None
    # Composition counts still cover the resolved players only.
    assert (
        state.continuing_player_count
        + state.first_observed_for_team_count
        + state.returning_player_count
        == 3
    )


def test_previous_incomplete_lineup_blocks_retained():
    rows = observations(
        (1, TEAM_A, T1, (P1, P2, P3)),  # malformed prior
        (2, TEAM_A, T2, ROSTER_A),
    )
    state = team_states_for(rows)[2]
    assert state.previous_match_id == 1
    assert state.previous_is_complete_five is False
    assert state.players_retained_from_previous_match is None
    assert state.same_lineup_as_previous_match is None


# --- spell-so-far is causal, never the eventual spell ---------------------------------


def test_consecutive_prior_appearances_never_leaks_future_spell():
    """A player's spell-so-far before a match counts only immediately
    preceding same-team observations; it does not know the spell will
    continue or end later."""
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, ROSTER_A),
        (3, TEAM_A, T3, ROSTER_A),
        (4, TEAM_B, T4, (P1, P6, P7, P8, 9)),
    )
    assert (
        player_state(rows, player_id=P1, team_id=TEAM_A, match_id=1).consecutive_prior_team_appearances
        == 0
    )
    assert (
        player_state(rows, player_id=P1, team_id=TEAM_A, match_id=2).consecutive_prior_team_appearances
        == 1
    )
    assert (
        player_state(rows, player_id=P1, team_id=TEAM_A, match_id=3).consecutive_prior_team_appearances
        == 2
    )
    # The eventual spell length for Team A is 3 (matches 1-3); match 3's
    # pre-match spell-so-far is 2, never the eventual 3.
    assert (
        player_state(rows, player_id=P1, team_id=TEAM_A, match_id=3).consecutive_prior_team_appearances
        != 3
    )


# --- timing fields ---------------------------------------------------------------------


def test_days_since_team_previous_match():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_A, T2, ROSTER_A),
    )
    state = team_states_for(rows)[2]
    assert state.days_since_team_previous_match == pytest.approx(31.0, abs=0.01)


def test_days_since_player_previous_match_tracks_any_prior_match():
    rows = observations(
        (1, TEAM_A, T1, ROSTER_A),
        (2, TEAM_B, T2, (P1, P6, P7, P8, 9)),
        (3, TEAM_A, T3, ROSTER_A),
    )
    state = player_state(rows, player_id=P1, team_id=TEAM_A, match_id=3)
    # Previous observed match is Team B at T2.
    assert state.days_since_player_previous_match == pytest.approx(
        (T3 - T2).total_seconds() / 86400.0, abs=0.01
    )
    # Previous team match (Team A) was at T1.
    assert state.days_since_player_previous_team_match == pytest.approx(
        (T3 - T1).total_seconds() / 86400.0, abs=0.01
    )


# --- unresolved identities are never fabricated ------------------------------------------


def test_null_player_id_raises():
    with pytest.raises(ValueError):
        derive_team_roster_state([(None, TEAM_A, 1, T1)])
    with pytest.raises(ValueError):
        derive_player_team_state([(None, TEAM_A, 1, T1)])


def test_null_team_id_raises():
    with pytest.raises(ValueError):
        derive_team_roster_state([(P1, None, 1, T1)])
    with pytest.raises(ValueError):
        derive_player_team_state([(P1, None, 1, T1)])


def test_generator_input_is_consumed_once_not_exhausted():
    """The derivations must materialize the observation iterable once, so a
    single-pass generator input yields the same result as a list."""
    rows = observations((1, TEAM_A, T1, ROSTER_A), (2, TEAM_A, T2, ROSTER_A))
    assert derive_team_roster_state(iter(rows)) == derive_team_roster_state(rows)
    assert derive_player_team_state(iter(rows)) == derive_player_team_state(rows)