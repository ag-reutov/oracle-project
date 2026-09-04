"""Schema-level tests: DB CHECK constraints and league-scope FK gating.

These prove the DB genuinely enforces the invariants described in
`storage.schema`, not just the Python dataclass layer -- run only against
a real Postgres (`TEST_DATABASE_URL`), since SQLite does not enforce the
same CHECK/FK semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from helpers import requires_test_database, seed_ingestion_league
from sqlalchemy.exc import IntegrityError

from dota_predictor.storage.schema import (
    INGESTION_LEAGUES,
    LEAGUE_INGESTION_STATE,
    LEAGUES,
    MATCH_PLAYERS,
    MATCHES,
    PLAYERS,
    STRATZ_RAW_MATCHES,
    TEAMS,
)

pytestmark = requires_test_database


def _seed_teams(conn, *team_ids: int) -> None:
    conn.execute(TEAMS.insert(), [{"team_id": tid} for tid in team_ids])


def _seed_players(conn, *player_ids: int) -> None:
    conn.execute(PLAYERS.insert(), [{"player_id": pid} for pid in player_ids])


def test_leagues_rejects_invalid_liquipedia_tier(engine):
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            LEAGUES.insert().values(
                league_id=1,
                name="Bad tier league",
                liquipedia_tier="NOT_A_REAL_TIER",
                in_scope=True,
            )
        )


def test_leagues_accepts_t3_liquipedia_tier(engine):
    with engine.begin() as conn:
        conn.execute(
            LEAGUES.insert().values(
                league_id=900001,
                name="Tier 3 Example",
                liquipedia_tier="T3",
                in_scope=True,
                notes="Tier 3 expansion test",
            )
        )
        row = conn.execute(
            LEAGUES.select().where(LEAGUES.c.league_id == 900001)
        ).one()
        assert row.liquipedia_tier == "T3"


def test_leagues_rejects_invalid_fetch_mode(engine):
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            LEAGUES.insert().values(
                league_id=3,
                name="Bad fetch mode",
                liquipedia_tier="T1",
                in_scope=True,
                fetch_mode="not_a_mode",
            )
        )


def test_excluded_league_is_not_a_valid_fk_target_for_matches(engine):
    """The core fix from this revision: a FK to `leagues` alone would let
    an excluded league be a valid target. `ingestion_leagues` must not
    contain it, so writes referencing it fail."""
    with engine.begin() as conn:
        conn.execute(
            LEAGUES.insert().values(
                league_id=2,
                name="Excluded qualifier",
                liquipedia_tier="EXCLUDED",
                in_scope=False,
            )
        )
        # Deliberately NOT inserted into ingestion_leagues.
        _seed_teams(conn, 1, 2)

    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            MATCHES.insert().values(
                match_id=100,
                league_id=2,
                start_time=datetime(2024, 1, 1, tzinfo=UTC),
                radiant_team_id=1,
                dire_team_id=2,
                radiant_win=True,
                duration_seconds=1800,
                mapper_version=1,
                canonicalized_at=datetime.now(UTC),
            )
        )


def test_ingestion_leagues_league_id_must_exist_in_leagues(engine):
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(INGESTION_LEAGUES.insert().values(league_id=999999))


def test_allowlisted_league_permits_raw_match_write(engine):
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=3)
        conn.execute(
            STRATZ_RAW_MATCHES.insert().values(
                match_id=200,
                league_id=3,
                payload={"ok": True},
                fetched_at=datetime.now(UTC),
            )
        )
    with engine.connect() as conn:
        row = conn.execute(
            STRATZ_RAW_MATCHES.select().where(STRATZ_RAW_MATCHES.c.match_id == 200)
        ).one()
        assert row.league_id == 3


def test_match_players_slot_in_side_out_of_range_rejected(engine):
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=4)
        _seed_teams(conn, 1, 2)
        _seed_players(conn, 1)
        conn.execute(
            MATCHES.insert().values(
                match_id=300,
                league_id=4,
                start_time=datetime(2024, 1, 1, tzinfo=UTC),
                radiant_team_id=1,
                dire_team_id=2,
                radiant_win=True,
                duration_seconds=1800,
                mapper_version=1,
                canonicalized_at=datetime.now(UTC),
            )
        )

    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            MATCH_PLAYERS.insert().values(
                match_id=300,
                side="RADIANT",
                slot_in_side=5,  # valid range is 0-4
                player_id=1,
                hero_id=10,
            )
        )


def test_match_players_rejects_non_positive_hero_id(engine):
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=16)
        _seed_teams(conn, 1, 2)
        _seed_players(conn, 1)
        conn.execute(
            MATCHES.insert().values(
                match_id=301,
                league_id=16,
                start_time=datetime(2024, 1, 1, tzinfo=UTC),
                radiant_team_id=1,
                dire_team_id=2,
                radiant_win=True,
                duration_seconds=1800,
                mapper_version=1,
                canonicalized_at=datetime.now(UTC),
            )
        )

    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            MATCH_PLAYERS.insert().values(
                match_id=301,
                side="RADIANT",
                slot_in_side=0,
                player_id=1,
                hero_id=0,
            )
        )


def test_league_ingestion_state_rejects_invalid_status(engine):
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=5)

    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            LEAGUE_INGESTION_STATE.insert().values(
                league_id=5,
                status="NOT_A_REAL_STATUS",
                updated_at=datetime.now(UTC),
            )
        )


def test_matches_rejects_equal_radiant_dire_team(engine):
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=6)
        _seed_teams(conn, 42)

    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            MATCHES.insert().values(
                match_id=400,
                league_id=6,
                start_time=datetime(2024, 1, 1, tzinfo=UTC),
                radiant_team_id=42,
                dire_team_id=42,
                radiant_win=True,
                duration_seconds=1800,
                mapper_version=1,
                canonicalized_at=datetime.now(UTC),
            )
        )


def test_matches_rejects_unregistered_team_id(engine):
    """A `team_id` that has never been written to `teams` cannot be
    referenced by `matches` -- the FK is the enforcement point for "every
    team referenced by matches must exist in teams" (see `storage.schema`
    module docstring)."""
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=7)
        _seed_teams(conn, 500)  # only the dire side is registered

    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            MATCHES.insert().values(
                match_id=500,
                league_id=7,
                start_time=datetime(2024, 1, 1, tzinfo=UTC),
                radiant_team_id=999,  # never registered in teams
                dire_team_id=500,
                radiant_win=True,
                duration_seconds=1800,
                mapper_version=1,
                canonicalized_at=datetime.now(UTC),
            )
        )


def test_match_players_rejects_unregistered_player_id(engine):
    """A `player_id` that has never been written to `players` cannot be
    referenced by `match_players`."""
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=8)
        _seed_teams(conn, 1, 2)
        conn.execute(
            MATCHES.insert().values(
                match_id=600,
                league_id=8,
                start_time=datetime(2024, 1, 1, tzinfo=UTC),
                radiant_team_id=1,
                dire_team_id=2,
                radiant_win=True,
                duration_seconds=1800,
                mapper_version=1,
                canonicalized_at=datetime.now(UTC),
            )
        )

    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            MATCH_PLAYERS.insert().values(
                match_id=600,
                side="RADIANT",
                slot_in_side=0,
                player_id=42,  # never registered in players
                hero_id=10,
            )
        )


def test_teams_and_players_permit_unreferenced_rows(engine):
    """`teams`/`players` are identity registries, not analytical entity
    tables gated on current usage: a row that is not (or is no longer)
    referenced by any `matches`/`match_players` row is explicitly allowed
    to exist -- see `storage.schema` module docstring. There is no cleanup
    trigger/constraint that would reject or remove it."""
    with engine.begin() as conn:
        _seed_teams(conn, 111)
        _seed_players(conn, 222)

    with engine.connect() as conn:
        team_row = conn.execute(
            TEAMS.select().where(TEAMS.c.team_id == 111)
        ).one()
        assert team_row.team_id == 111
        player_row = conn.execute(
            PLAYERS.select().where(PLAYERS.c.player_id == 222)
        ).one()
        assert player_row.player_id == 222
