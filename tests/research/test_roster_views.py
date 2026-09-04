"""Semantic tests for the Slice 4 roster-history research views.

Seeds canonical relational rows and asserts:

* `research.team_match_lineups` -- one row per (match_id, team_id), the
  sorted/deterministic lineup identity, and the explicit cardinality audit
  (exactly-five, fewer-than-five, duplicate, null flags). Malformed
  lineups are flagged, never forced into a five-player shape.
* `research.player_team_spells` -- deterministic observed spells: one team
  -> one spell, A -> B -> A -> three spells, one-match intermediate teams
  retained, long inactivity not splitting a spell, equal timestamps ordered
  by match id. The view is cross-checked against the pure Python derivation
  (`dota_predictor.data.roster_history.derive_observed_spells`).
* Invariants from the Slice 4 spec: a player cannot silently appear for
  both teams in one match (schema rejects it), canonical team/player joins
  resolve, and the pre-existing research/canonical views keep working
  alongside the new ones.

The views are created from the same SQL the Alembic migration applies
(`dota_predictor.research.views`).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from research_helpers import (
    DIRE_PLAYER_BASE_IDS,
    DIRE_TEAM,
    PLAYER_BASE_IDS,
    RADIANT_TEAM,
    seed_league,
    seed_match,
)
from sqlalchemy.exc import IntegrityError

from dota_predictor.data.roster_history import (
    collect_player_team_observations,
    derive_observed_spells,
)
from dota_predictor.storage.schema import MATCH_PLAYERS, MATCHES

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set; skipping DB-touching test",
)

UTC2024 = datetime(2024, 1, 1, tzinfo=UTC)
UTC2025 = datetime(2025, 1, 1, tzinfo=UTC)
RAD_TEAM = RADIANT_TEAM
DIR_TEAM = DIRE_TEAM
RAD_PLAYER = PLAYER_BASE_IDS[0]
DIR_PLAYER = DIRE_PLAYER_BASE_IDS[0]


def _rows(engine, sql: str, **params) -> list[sa.Row]:
    with engine.connect() as conn:
        return conn.execute(sa.text(sql), params).all()


def _seed_custom_match(conn, *, match_id: int, league_id: int, start_time, radiant_players, dire_players) -> None:
    """Seed a match with explicit per-side player lists (beyond the helper's
    fixed radiant/dire rosters), using the caller's open transaction."""
    conn.execute(
        MATCHES.insert().values(
            match_id=match_id,
            league_id=league_id,
            start_time=start_time,
            league_name="Test League",
            game_version_id=178,
            radiant_team_id=RAD_TEAM,
            radiant_team_name_observed="Radiant Test",
            dire_team_id=DIR_TEAM,
            dire_team_name_observed="Dire Test",
            radiant_win=True,
            duration_seconds=2400,
            draft_complete=False,
            mapper_version=1,
            canonicalized_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    for slot, player_id in enumerate(radiant_players):
        conn.execute(
            MATCH_PLAYERS.insert().values(
                match_id=match_id, side="RADIANT", slot_in_side=slot,
                player_id=player_id, hero_id=100 + slot,
            )
        )
    for slot, player_id in enumerate(dire_players):
        conn.execute(
            MATCH_PLAYERS.insert().values(
                match_id=match_id, side="DIRE", slot_in_side=slot,
                player_id=player_id, hero_id=200 + slot,
            )
        )


# --- team_match_lineups ---------------------------------------------------------


def test_lineup_view_one_row_per_team_match_and_complete_five(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 2001, name="Test League", tier="T1")
        seed_match(conn, match_id=1, league_id=2001, start_time=UTC2024)
        seed_match(conn, match_id=2, league_id=2001, start_time=UTC2024)

    rows = _rows(
        engine,
        "SELECT match_id, team_id, n_players, n_resolved_players, "
        "n_distinct_players, n_null_player_ids, has_duplicate_players, "
        "has_fewer_than_five, has_more_than_five, has_exactly_five, "
        "is_complete_five, lineup_key, team_is_match_team "
        "FROM research.team_match_lineups ORDER BY match_id, team_id",
    )
    assert len(rows) == 4  # two matches x two teams
    for row in rows:
        assert row.n_players == 5
        assert row.n_resolved_players == 5
        assert row.n_distinct_players == 5
        assert row.n_null_player_ids == 0
        assert row.has_duplicate_players is False
        assert row.has_fewer_than_five is False
        assert row.has_more_than_five is False
        assert row.has_exactly_five is True
        assert row.is_complete_five is True
        assert row.team_is_match_team is True
        assert row.lineup_key == ",".join(
            str(p) for p in sorted(PLAYER_BASE_IDS if row.team_id == RAD_TEAM else DIRE_PLAYER_BASE_IDS)
        )


def test_lineup_key_is_deterministic_sorted_identity(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 2002, name="Test League", tier="T1")
        seed_match(conn, match_id=11, league_id=2002, start_time=UTC2024)

    row = _rows(
        engine,
        "SELECT team_id, lineup_player_ids, lineup_key FROM research.team_match_lineups "
        "WHERE match_id = 11 AND team_id = :t",
        t=RAD_TEAM,
    )[0]
    assert list(row.lineup_player_ids) == sorted(PLAYER_BASE_IDS)
    assert row.lineup_key == ",".join(str(p) for p in sorted(PLAYER_BASE_IDS))


def test_lineup_view_flags_fewer_than_five(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 2003, name="Test League", tier="T1")
        seed_match(conn, match_id=21, league_id=2003, start_time=UTC2024)
        conn.execute(
            MATCH_PLAYERS.delete().where(
                sa.and_(
                    MATCH_PLAYERS.c.match_id == 21,
                    MATCH_PLAYERS.c.side == "DIRE",
                    MATCH_PLAYERS.c.slot_in_side == 4,
                )
            )
        )

    row = _rows(
        engine,
        "SELECT n_players, n_resolved_players, has_fewer_than_five, "
        "has_exactly_five, is_complete_five "
        "FROM research.team_match_lineups WHERE match_id = 21 AND team_id = :t",
        t=DIR_TEAM,
    )[0]
    assert row.n_players == 4
    assert row.n_resolved_players == 4
    assert row.has_fewer_than_five is True
    assert row.has_exactly_five is False
    assert row.is_complete_five is False


# --- player_team_spells ---------------------------------------------------------


def test_spell_view_same_team_is_one_spell(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 2004, name="Test League", tier="T1")
        seed_match(conn, match_id=31, league_id=2004, start_time=UTC2024)
        seed_match(conn, match_id=32, league_id=2004, start_time=UTC2025)

    rows = _rows(
        engine,
        "SELECT team_id, spell_index, observed_match_count, first_seen_at, "
        "last_seen_at, first_match_id, last_match_id "
        "FROM research.player_team_spells WHERE player_id = :p",
        p=RAD_PLAYER,
    )
    assert len(rows) == 1
    assert rows[0].team_id == RAD_TEAM
    assert rows[0].spell_index == 1
    assert rows[0].observed_match_count == 2
    assert rows[0].first_seen_at == UTC2024
    assert rows[0].last_seen_at == UTC2025
    assert rows[0].first_match_id == 31
    assert rows[0].last_match_id == 32


def test_spell_view_a_b_a_is_three_spells(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 2005, name="Test League", tier="T1")
        seed_match(conn, match_id=41, league_id=2005, start_time=UTC2024)
        seed_match(conn, match_id=42, league_id=2005, start_time=datetime(2024, 1, 5, tzinfo=UTC))
        _seed_custom_match(
            conn,
            match_id=43,
            league_id=2005,
            start_time=datetime(2024, 2, 1, tzinfo=UTC),
            radiant_players=[DIR_PLAYER] + PLAYER_BASE_IDS[1:],
            dire_players=[RAD_PLAYER] + DIRE_PLAYER_BASE_IDS[1:],
        )
        seed_match(conn, match_id=44, league_id=2005, start_time=datetime(2024, 3, 1, tzinfo=UTC))

    rows = _rows(
        engine,
        "SELECT team_id, spell_index, observed_match_count, first_match_id, last_match_id "
        "FROM research.player_team_spells WHERE player_id = :p ORDER BY spell_index",
        p=RAD_PLAYER,
    )
    assert [(r.team_id, r.spell_index, r.observed_match_count) for r in rows] == [
        (RAD_TEAM, 1, 2),
        (DIR_TEAM, 2, 1),
        (RAD_TEAM, 3, 1),
    ]
    assert rows[0].last_match_id == 42
    assert rows[1].first_match_id == 43
    assert rows[2].first_match_id == 44


def test_spell_view_one_match_intermediate_team_retained(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 2006, name="Test League", tier="T1")
        seed_match(conn, match_id=51, league_id=2006, start_time=UTC2024)
        _seed_custom_match(
            conn,
            match_id=52,
            league_id=2006,
            start_time=datetime(2024, 2, 1, tzinfo=UTC),
            radiant_players=[DIR_PLAYER] + PLAYER_BASE_IDS[1:],
            dire_players=[RAD_PLAYER] + DIRE_PLAYER_BASE_IDS[1:],
        )
        seed_match(conn, match_id=53, league_id=2006, start_time=datetime(2024, 3, 1, tzinfo=UTC))

    rows = _rows(
        engine,
        "SELECT team_id, spell_index, observed_match_count "
        "FROM research.player_team_spells WHERE player_id = :p ORDER BY spell_index",
        p=RAD_PLAYER,
    )
    assert [(r.team_id, r.spell_index, r.observed_match_count) for r in rows] == [
        (RAD_TEAM, 1, 1),
        (DIR_TEAM, 2, 1),
        (RAD_TEAM, 3, 1),
    ]


def test_spell_view_sql_matches_python_derivation(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 2007, name="Test League", tier="T1")
        seed_match(conn, match_id=61, league_id=2007, start_time=UTC2024)
        seed_match(conn, match_id=62, league_id=2007, start_time=UTC2024)
        _seed_custom_match(
            conn,
            match_id=63,
            league_id=2007,
            start_time=datetime(2024, 2, 1, tzinfo=UTC),
            radiant_players=[DIR_PLAYER] + PLAYER_BASE_IDS[1:],
            dire_players=[RAD_PLAYER] + DIRE_PLAYER_BASE_IDS[1:],
        )
        seed_match(conn, match_id=64, league_id=2007, start_time=datetime(2024, 3, 1, tzinfo=UTC))

    with engine.connect() as conn:
        observations, _null_p, _null_t = collect_player_team_observations(conn)
    python_spells = derive_observed_spells(observations)
    sql_rows = _rows(
        engine,
        "SELECT player_id, team_id, spell_index, observed_match_count, "
        "first_seen_at, first_match_id, last_seen_at, last_match_id "
        "FROM research.player_team_spells ORDER BY player_id, spell_index",
    )
    assert len(python_spells) == len(sql_rows)
    for p, s in zip(python_spells, sql_rows, strict=True):
        assert (p.player_id, p.team_id, p.spell_index, p.observed_match_count,
                p.first_match_id, p.last_match_id) == (
                s.player_id, s.team_id, s.spell_index, s.observed_match_count,
                s.first_match_id, s.last_match_id)
        assert p.first_seen_at == s.first_seen_at
        assert p.last_seen_at == s.last_seen_at


# --- spec invariants ------------------------------------------------------------


def test_player_cannot_appear_for_both_teams_in_one_match(engine) -> None:
    """A player observed for both teams in the same match is structurally
    impossible: the (match_id, player_id) unique constraint rejects it, so
    no observed spell can silently straddle both sides of one match."""
    with engine.begin() as conn:
        seed_league(conn, 2008, name="Test League", tier="T1")
        seed_match(conn, match_id=71, league_id=2008, start_time=UTC2024)
        with pytest.raises(IntegrityError):
            conn.execute(
                MATCH_PLAYERS.insert().values(
                    match_id=71, side="DIRE", slot_in_side=0,
                    player_id=RAD_PLAYER, hero_id=1,
                )
            )

    count = _rows(
        engine,
        "SELECT count(*) FROM ("
        "SELECT match_id, player_id FROM research.player_matches WHERE side = 'RADIANT' "
        "INTERSECT "
        "SELECT match_id, player_id FROM research.player_matches WHERE side = 'DIRE') c",
    )[0][0]
    assert count == 0


def test_canonical_team_and_player_joins_resolve(engine) -> None:
    """Every research roster fact joins cleanly to the canonical team/player
    registries (foreign keys guarantee this; the views expose no orphaned
    or dropped rows)."""
    with engine.begin() as conn:
        seed_league(conn, 2009, name="Test League", tier="T1")
        seed_match(conn, match_id=81, league_id=2009, start_time=UTC2024)

    player_matches = _rows(engine, "SELECT count(*) FROM research.player_matches")[0][0]
    lineup_rows = _rows(engine, "SELECT count(*) FROM research.team_match_lineups")[0][0]
    spell_rows = _rows(engine, "SELECT count(*) FROM research.player_team_spells")[0][0]
    assert player_matches == 10
    assert lineup_rows == 2
    assert spell_rows == 10  # one spell per distinct player

    unresolved_team = _rows(
        engine,
        "SELECT count(*) FROM research.team_match_lineups lu "
        "LEFT JOIN teams t USING (team_id) WHERE t.team_id IS NULL",
    )[0][0]
    unresolved_player = _rows(
        engine,
        "SELECT count(*) FROM research.player_team_spells s "
        "LEFT JOIN players p USING (player_id) WHERE p.player_id IS NULL",
    )[0][0]
    assert unresolved_team == 0
    assert unresolved_player == 0


def test_existing_research_views_remain_compatible(engine) -> None:
    """Adding the roster views must not disturb the pre-existing research
    layer: matches / player_matches / players / population views keep their
    semantics and row counts alongside the new views."""
    with engine.begin() as conn:
        seed_league(conn, 2010, name="Test League", tier="T1")
        seed_match(conn, match_id=91, league_id=2010, start_time=UTC2024)
        seed_match(conn, match_id=92, league_id=2010, start_time=datetime(2024, 6, 1, tzinfo=UTC))

    assert _rows(engine, "SELECT count(*) FROM research.matches")[0][0] == 2
    assert _rows(engine, "SELECT count(*) FROM research.player_matches")[0][0] == 20
    assert _rows(engine, "SELECT count(*) FROM research.t12_matches")[0][0] == 2
    assert _rows(engine, "SELECT count(DISTINCT player_id) FROM research.players")[0][0] == 10
    assert _rows(engine, "SELECT count(*) FROM research.team_match_lineups")[0][0] == 4
    assert _rows(engine, "SELECT count(*) FROM research.player_team_spells")[0][0] == 10

    # A roster fact for match 91 must resolve to the same teams the match
    # view reports (cross-view consistency).
    row = _rows(
        engine,
        "SELECT m.radiant_team_id, m.dire_team_id FROM research.matches m "
        "WHERE m.match_id = 91",
    )[0]
    lineup_teams = sorted(
        r[0]
        for r in _rows(
            engine,
            "SELECT DISTINCT team_id FROM research.team_match_lineups WHERE match_id = 91",
        )
    )
    assert lineup_teams == sorted([row.radiant_team_id, row.dire_team_id])