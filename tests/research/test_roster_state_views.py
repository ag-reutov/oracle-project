"""Semantic tests for the Slice 5 roster-state research views.

Seeds canonical relational rows and asserts:

* `research.player_team_state` -- one row per (player_id, team_id,
  match_id) with strictly-causal prior-team counts, previous-observed
  match (any team), and the mutually exclusive first/returning/continuing
  flags.
* `research.team_roster_state` -- one row per (match_id, team_id) with
  the previous observed team match, retained/changed/same-lineup fields
  (complete fives only), and prior exact-lineup experience.
* Equal timestamps never become causal precedence.
* Future deletion at the relational level leaves historical state
  unchanged.
* The SQL views agree with the pure Python derivation
  (`dota_predictor.data.roster_state`).
* The pre-existing research/canonical views keep working alongside the
  new ones.

The views are created from the same SQL the Alembic migration applies
(`dota_predictor.research.views`).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from research_helpers import seed_league
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dota_predictor.data.roster_history import collect_player_team_observations
from dota_predictor.data.roster_state import (
    derive_player_team_state,
    derive_team_roster_state,
)
from dota_predictor.storage.schema import MATCH_PLAYERS, MATCHES, PLAYERS, TEAMS

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set; skipping DB-touching test",
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)
T4 = datetime(2024, 4, 1, tzinfo=UTC)

TEAM_A, TEAM_B = 8261500, 9247354
P1, P2, P3, P4, P5 = 898754153, 137129583, 129958758, 157475523, 94296097
P6, P7, P8, P9, P10 = 10366616, 100058342, 898455820, 183719386, 25907144
P11 = 555555555
ROSTER_A = (P1, P2, P3, P4, P5)


def _rows(engine, sql: str, **params) -> list[sa.Row]:
    with engine.connect() as conn:
        return conn.execute(sa.text(sql), params).all()


def seed_team_match(
    conn,
    *,
    match_id: int,
    league_id: int,
    start_time,
    team_rosters: list[tuple[int, tuple[int, ...]]],
) -> None:
    """Seed one match between two teams with explicit rosters.

    `team_rosters` is `[(team_id, (p1, p2, p3, p4, p5)), ...]` (exactly two
    entries). Registers teams/players idempotently.
    """
    for team_id, _roster in team_rosters:
        conn.execute(
            pg_insert(TEAMS).values(team_id=team_id).on_conflict_do_nothing()
        )
        for player_id in _roster:
            conn.execute(
                pg_insert(PLAYERS).values(player_id=player_id).on_conflict_do_nothing()
            )
    radiant_team, dire_team = team_rosters[0][0], team_rosters[1][0]
    conn.execute(
        MATCHES.insert().values(
            match_id=match_id,
            league_id=league_id,
            start_time=start_time,
            league_name="Test League",
            game_version_id=178,
            radiant_team_id=radiant_team,
            radiant_team_name_observed="Radiant Test",
            dire_team_id=dire_team,
            dire_team_name_observed="Dire Test",
            radiant_win=True,
            duration_seconds=2400,
            draft_complete=False,
            mapper_version=1,
            canonicalized_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    for side, (team_id, roster) in enumerate(team_rosters):
        for slot, player_id in enumerate(roster):
            conn.execute(
                MATCH_PLAYERS.insert().values(
                    match_id=match_id,
                    side="RADIANT" if side == 0 else "DIRE",
                    slot_in_side=slot,
                    player_id=player_id,
                    hero_id=100 + slot,
                )
            )


def _python_states(engine):
    with engine.connect() as conn:
        observations, _n, _t = collect_player_team_observations(conn)
    team = derive_team_roster_state(observations)
    player = derive_player_team_state(observations)
    return team, player


def _approx(value, expected, tolerance=0.001):
    if value is None or expected is None:
        assert value is None and expected is None
        return
    assert abs(float(value) - float(expected)) <= tolerance


# --- basic grains and identity -------------------------------------------------------


def test_views_expose_one_row_per_team_match(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3001, name="Test League", tier="T1")
        seed_team_match(
            conn, match_id=1, league_id=3001, start_time=T1,
            team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
        )
    rows = _rows(engine, "SELECT match_id, team_id FROM research.team_roster_state")
    assert len(rows) == 2
    assert sorted(r.team_id for r in rows) == sorted([TEAM_A, TEAM_B])
    # No previous match for a first team match.
    assert _rows(
        engine,
        "SELECT previous_match_id, players_retained_from_previous_match, "
        "prior_exact_lineup_match_count FROM research.team_roster_state "
        "WHERE team_id = :t",
        t=TEAM_A,
    ) == [(None, None, 0)]


def test_player_team_state_one_row_per_lineup_player(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3002, name="Test League", tier="T1")
        seed_team_match(
            conn, match_id=11, league_id=3002, start_time=T1,
            team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
        )
    rows = _rows(
        engine,
        "SELECT player_id, team_id FROM research.player_team_state WHERE match_id = 11 "
        "ORDER BY team_id, player_id",
    )
    assert len(rows) == 10
    assert all(r[1] == TEAM_A for r in rows[:5])
    assert all(r[1] == TEAM_B for r in rows[5:])
    # First-observed flags on a brand-new player.
    first_flags = _rows(
        engine,
        "SELECT count(*) FROM research.player_team_state "
        "WHERE match_id = 11 AND is_first_observed_match_for_team",
    )[0][0]
    assert first_flags == 10


# --- SQL vs Python agreement -----------------------------------------------------------


def _seed_roster_state_corpus(conn) -> None:
    seed_league(conn, 3003, name="Test League", tier="T1")
    # Team A: roster A (match 21), roster A with P5 -> P11 (match 22),
    # roster A without P1 (match 23), roster A again (match 24).
    # Player P1: Team A (21, 22), Team B (23), Team A (24) => A -> B -> A.
    seed_team_match(
        conn, match_id=21, league_id=3003, start_time=T1,
        team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
    )
    seed_team_match(
        conn, match_id=22, league_id=3003, start_time=T2,
        team_rosters=[(TEAM_A, (P1, P2, P3, P4, P11)), (TEAM_B, (P6, P7, P8, P9, P10))],
    )
    seed_team_match(
        conn, match_id=23, league_id=3003, start_time=T3,
        team_rosters=[(TEAM_A, (P2, P3, P4, P5, P11)), (TEAM_B, (P1, P7, P8, P9, P10))],
    )
    seed_team_match(
        conn, match_id=24, league_id=3003, start_time=T4,
        team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
    )


def test_team_roster_state_sql_agrees_with_python(engine) -> None:
    with engine.begin() as conn:
        _seed_roster_state_corpus(conn)

    python_team, _player = _python_states(engine)
    sql_rows = _rows(
        engine,
        "SELECT match_id, team_id, lineup_player_ids, previous_match_id, "
        "players_retained_from_previous_match, players_changed_from_previous_match, "
        "same_lineup_as_previous_match, prior_exact_lineup_match_count, "
        "last_exact_lineup_match_id, continuing_player_count, "
        "first_observed_for_team_count, returning_player_count, "
        "days_since_team_previous_match "
        "FROM research.team_roster_state ORDER BY team_id, match_id",
    )
    assert len(python_team) == len(sql_rows)
    by_key = {(s.team_id, s.match_id): s for s in python_team}
    for row in sql_rows:
        state = by_key[(row.team_id, row.match_id)]
        assert list(row.lineup_player_ids) == list(state.lineup_player_ids)
        assert row.previous_match_id == state.previous_match_id
        assert row.players_retained_from_previous_match == state.players_retained_from_previous_match
        assert row.players_changed_from_previous_match == state.players_changed_from_previous_match
        assert row.same_lineup_as_previous_match == state.same_lineup_as_previous_match
        assert row.prior_exact_lineup_match_count == state.prior_exact_lineup_match_count
        assert row.last_exact_lineup_match_id == state.last_exact_lineup_match_id
        assert row.continuing_player_count == state.continuing_player_count
        assert row.first_observed_for_team_count == state.first_observed_for_team_count
        assert row.returning_player_count == state.returning_player_count
        _approx(row.days_since_team_previous_match, state.days_since_team_previous_match)


def test_player_team_state_sql_agrees_with_python(engine) -> None:
    with engine.begin() as conn:
        _seed_roster_state_corpus(conn)

    _team, python_player = _python_states(engine)
    sql_rows = _rows(
        engine,
        "SELECT player_id, team_id, match_id, prior_team_match_count, "
        "previous_observed_team_id, previous_observed_match_id, "
        "is_first_observed_match_for_team, is_returning_to_team, "
        "is_continuing_with_team, consecutive_prior_team_appearances, "
        "days_since_player_previous_match, days_since_player_previous_team_match "
        "FROM research.player_team_state ORDER BY player_id, team_id, match_id",
    )
    assert len(python_player) == len(sql_rows)
    by_key = {(s.player_id, s.team_id, s.match_id): s for s in python_player}
    for row in sql_rows:
        state = by_key[(row.player_id, row.team_id, row.match_id)]
        assert row.prior_team_match_count == state.prior_team_match_count
        assert row.previous_observed_team_id == state.previous_observed_team_id
        assert row.previous_observed_match_id == state.previous_observed_match_id
        assert row.is_first_observed_match_for_team == state.is_first_observed_match_for_team
        assert row.is_returning_to_team == state.is_returning_to_team
        assert row.is_continuing_with_team == state.is_continuing_with_team
        assert row.consecutive_prior_team_appearances == state.consecutive_prior_team_appearances
        _approx(row.days_since_player_previous_match, state.days_since_player_previous_match)
        _approx(
            row.days_since_player_previous_team_match,
            state.days_since_player_previous_team_match,
        )


def test_retained_and_returning_semantics_in_sql(engine) -> None:
    with engine.begin() as conn:
        _seed_roster_state_corpus(conn)

    team_a_match2 = _rows(
        engine,
        "SELECT players_retained_from_previous_match, same_lineup_as_previous_match "
        "FROM research.team_roster_state WHERE team_id = :t AND match_id = 22",
        t=TEAM_A,
    )[0]
    assert team_a_match2.players_retained_from_previous_match == 4
    assert team_a_match2.same_lineup_as_previous_match is False

    # P1 returns to Team A in match 24 after representing Team B in match 23.
    returning = _rows(
        engine,
        "SELECT previous_observed_team_id, is_returning_to_team, "
        "is_continuing_with_team, is_first_observed_match_for_team "
        "FROM research.player_team_state WHERE player_id = :p AND team_id = :t AND match_id = 24",
        p=P1, t=TEAM_A,
    )[0]
    assert returning.previous_observed_team_id == TEAM_B
    assert returning.is_returning_to_team is True
    assert returning.is_continuing_with_team is False
    assert returning.is_first_observed_match_for_team is False


# --- equal timestamps -------------------------------------------------------------------


def test_sql_equal_timestamps_do_not_create_previous_match(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3004, name="Test League", tier="T1")
        seed_team_match(
            conn, match_id=31, league_id=3004, start_time=T1,
            team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
        )
        # Same team, same start_time, different match_id, different roster.
        seed_team_match(
            conn, match_id=32, league_id=3004, start_time=T1,
            team_rosters=[(TEAM_A, (P1, P2, P3, P4, P11)), (TEAM_B, (P6, P7, P8, P9, P10))],
        )

    for match_id in (31, 32):
        row = _rows(
            engine,
            "SELECT previous_match_id, players_retained_from_previous_match, "
            "prior_exact_lineup_match_count "
            "FROM research.team_roster_state WHERE team_id = :t AND match_id = :m",
            t=TEAM_A, m=match_id,
        )[0]
        assert row.previous_match_id is None
        assert row.players_retained_from_previous_match is None
        assert row.prior_exact_lineup_match_count == 0

    # Player P1: match 31 (Team A) and match 32 (Team A) are simultaneous;
    # neither is the other's previous observed match.
    both = _rows(
        engine,
        "SELECT match_id, previous_observed_match_id, previous_observed_team_id "
        "FROM research.player_team_state WHERE player_id = :p AND team_id = :t "
        "ORDER BY match_id",
        p=P1, t=TEAM_A,
    )
    assert [(r.match_id, r.previous_observed_match_id) for r in both] == [
        (31, None),
        (32, None),
    ]


def test_sql_equal_timestamps_previous_tie_break_is_presentation_only(engine) -> None:
    """A strictly later match's most-recent prior is deterministic (largest
    match_id among the equal-time strictly-prior group), but that choice
    never makes an equal-time match itself a prior of its peer."""
    with engine.begin() as conn:
        seed_league(conn, 3005, name="Test League", tier="T1")
        seed_team_match(
            conn, match_id=41, league_id=3005, start_time=T1,
            team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
        )
        seed_team_match(
            conn, match_id=42, league_id=3005, start_time=T1,
            team_rosters=[(TEAM_A, (P1, P2, P3, P4, P11)), (TEAM_B, (P6, P7, P8, P9, P10))],
        )
        seed_team_match(
            conn, match_id=43, league_id=3005, start_time=T2,
            team_rosters=[(TEAM_A, (P1, P2, P3, P4, P11)), (TEAM_B, (P6, P7, P8, P9, P10))],
        )

    later = _rows(
        engine,
        "SELECT previous_match_id, players_retained_from_previous_match "
        "FROM research.team_roster_state WHERE team_id = :t AND match_id = 43",
        t=TEAM_A,
    )[0]
    # Most recent strictly prior is match 42 (tie-break by match_id DESC).
    assert later.previous_match_id == 42
    assert later.players_retained_from_previous_match == 5


# --- future deletion invariance (relational level) ---------------------------------------


def test_future_deletion_leaves_historical_state_unchanged(engine) -> None:
    with engine.begin() as conn:
        _seed_roster_state_corpus(conn)
        # Future matches that change Team A's roster and P1's team.
        seed_team_match(
            conn, match_id=51, league_id=3003, start_time=T4,
            team_rosters=[(TEAM_A, (11, 12, 13, 14, 15)), (TEAM_B, (P6, P7, P8, P9, P10))],
        )
        seed_team_match(
            conn, match_id=52, league_id=3003, start_time=T4,
            team_rosters=[(TEAM_B, (P1, P7, P8, P9, 10)), (TEAM_A, (P6, P2, P3, P4, P5))],
        )

    before_team = _rows(
        engine,
        "SELECT match_id, team_id, players_retained_from_previous_match, "
        "prior_exact_lineup_match_count, continuing_player_count, "
        "returning_player_count FROM research.team_roster_state "
        "WHERE start_time <= :cutoff ORDER BY team_id, match_id",
        cutoff=T3,
    )
    before_player = _rows(
        engine,
        "SELECT player_id, team_id, match_id, prior_team_match_count, "
        "is_first_observed_match_for_team, is_returning_to_team, "
        "previous_observed_team_id FROM research.player_team_state "
        "WHERE start_time <= :cutoff ORDER BY player_id, team_id, match_id",
        cutoff=T3,
    )

    # Delete all observations after T3 (strictly future matches) at the
    # relational level; the cascade removes their match_players rows.
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM matches WHERE start_time > :cutoff"), {"cutoff": T3})

    after_team = _rows(
        engine,
        "SELECT match_id, team_id, players_retained_from_previous_match, "
        "prior_exact_lineup_match_count, continuing_player_count, "
        "returning_player_count FROM research.team_roster_state ORDER BY team_id, match_id",
    )
    after_player = _rows(
        engine,
        "SELECT player_id, team_id, match_id, prior_team_match_count, "
        "is_first_observed_match_for_team, is_returning_to_team, "
        "previous_observed_team_id FROM research.player_team_state "
        "ORDER BY player_id, team_id, match_id",
    )

    # Future matches must have been present before deletion (fixture sanity:
    # 5 matches x 2 teams = 10 rows before, 6 rows remain after deleting the
    # two future matches).
    assert len(before_team) == 6
    assert _rows(engine, "SELECT count(*) FROM research.team_roster_state")[0][0] == 6
    assert before_team == after_team
    assert before_player == after_player


# --- incomplete lineups remain explicit ----------------------------------------------------


def test_sql_incomplete_lineup_remains_explicit(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3006, name="Test League", tier="T1")
        seed_team_match(
            conn, match_id=61, league_id=3006, start_time=T1,
            team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
        )
        # Team A plays with only four players in match 62.
        seed_team_match(
            conn, match_id=62, league_id=3006, start_time=T2,
            team_rosters=[(TEAM_A, (P1, P2, P3, P4)), (TEAM_B, (P6, P7, P8, P9, P10))],
        )

    row = _rows(
        engine,
        "SELECT is_complete_five, n_resolved_players, "
        "players_retained_from_previous_match, prior_exact_lineup_match_count "
        "FROM research.team_roster_state WHERE team_id = :t AND match_id = 62",
        t=TEAM_A,
    )[0]
    assert row.is_complete_five is False
    assert row.n_resolved_players == 4
    assert row.players_retained_from_previous_match is None
    assert row.prior_exact_lineup_match_count is None


# --- compatibility ---------------------------------------------------------------------------


def test_existing_research_views_remain_compatible(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3007, name="Test League", tier="T1")
        seed_team_match(
            conn, match_id=71, league_id=3007, start_time=T1,
            team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
        )
        seed_team_match(
            conn, match_id=72, league_id=3007, start_time=T2,
            team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
        )

    assert _rows(engine, "SELECT count(*) FROM research.matches")[0][0] == 2
    assert _rows(engine, "SELECT count(*) FROM research.player_matches")[0][0] == 20
    assert _rows(engine, "SELECT count(*) FROM research.team_match_lineups")[0][0] == 4
    assert _rows(engine, "SELECT count(*) FROM research.player_team_spells")[0][0] == 10
    assert _rows(engine, "SELECT count(*) FROM research.team_roster_state")[0][0] == 4
    assert _rows(engine, "SELECT count(*) FROM research.player_team_state")[0][0] == 20

    # The Slice 5 team roster state must resolve to the same teams as the
    # match view (cross-view consistency).
    lineup_teams = sorted(
        r[0]
        for r in _rows(
            engine,
            "SELECT DISTINCT team_id FROM research.team_roster_state WHERE match_id = 71",
        )
    )
    match = _rows(
        engine,
        "SELECT radiant_team_id, dire_team_id FROM research.matches WHERE match_id = 71",
    )[0]
    assert lineup_teams == sorted([match.radiant_team_id, match.dire_team_id])


# --- unresolved identities ----------------------------------------------------------------


def test_player_on_both_sides_is_structurally_impossible(engine) -> None:
    """A player cannot appear for both teams in one match: the
    (match_id, player_id) unique constraint rejects it, so no player-team
    state row can silently straddle both sides of one match."""
    with engine.begin() as conn:
        seed_league(conn, 3008, name="Test League", tier="T1")
        seed_team_match(
            conn, match_id=81, league_id=3008, start_time=T1,
            team_rosters=[(TEAM_A, ROSTER_A), (TEAM_B, (P6, P7, P8, P9, P10))],
        )
        with pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                MATCH_PLAYERS.insert().values(
                    match_id=81, side="DIRE", slot_in_side=0,
                    player_id=P1, hero_id=1,
                )
            )

    count = _rows(
        engine,
        "SELECT count(*) FROM ("
        "SELECT match_id, player_id FROM research.player_team_state "
        "GROUP BY match_id, player_id HAVING count(*) > 1) c",
    )[0][0]
    assert count == 0