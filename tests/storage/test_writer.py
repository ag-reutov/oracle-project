"""Round-trip tests for `storage.writer.write_canonical_match`.

Covers the two properties this revision specifically required:
* a full write round-trips through `matches`/`match_players`/`draft_events`;
* reprocessing a match with a smaller draft leaves no stale child rows
  (this is exactly what delete+insert replacement is for; a naive upsert
  would fail this test).
"""

from __future__ import annotations

from dataclasses import replace

from helpers import build_canonical_match, requires_test_database, seed_ingestion_league

from dota_predictor.data.canonical_schema import (
    DraftAction,
    MatchLane,
    MatchPlayerPosition,
    MatchPlayerRole,
    Side,
)
from dota_predictor.data.stratz_mapping import CANONICAL_MAPPER_VERSION
from dota_predictor.storage.schema import (
    DRAFT_EVENTS,
    MATCH_PLAYERS,
    MATCHES,
    PLAYERS,
    TEAMS,
)
from dota_predictor.storage.writer import write_canonical_match

pytestmark = requires_test_database


def test_write_canonical_match_round_trip(engine):
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=10)

    match = build_canonical_match(match_id=1000, league_id=10, num_bans=6)
    write_canonical_match(engine, match)

    with engine.connect() as conn:
        match_row = conn.execute(
            MATCHES.select().where(MATCHES.c.match_id == 1000)
        ).one()
        assert match_row.league_id == 10
        assert match_row.radiant_win == match.radiant_win
        assert match_row.duration_seconds == match.duration_seconds
        assert match_row.mapper_version == CANONICAL_MAPPER_VERSION
        assert (
            match_row.radiant_team_name_observed
            == match.radiant_team_name_observed
        )
        assert match_row.dire_team_name_observed == match.dire_team_name_observed

        team_ids = {
            row.team_id
            for row in conn.execute(
                TEAMS.select().where(
                    TEAMS.c.team_id.in_(
                        [match.radiant_team_id, match.dire_team_id]
                    )
                )
            ).all()
        }
        assert team_ids == {match.radiant_team_id, match.dire_team_id}

        player_id_rows = {
            row.player_id
            for row in conn.execute(
                PLAYERS.select().where(
                    PLAYERS.c.player_id.in_(
                        match.radiant_player_ids + match.dire_player_ids
                    )
                )
            ).all()
        }
        assert player_id_rows == set(
            match.radiant_player_ids + match.dire_player_ids
        )

        player_rows = conn.execute(
            MATCH_PLAYERS.select()
            .where(MATCH_PLAYERS.c.match_id == 1000)
            .order_by(MATCH_PLAYERS.c.side, MATCH_PLAYERS.c.slot_in_side)
        ).all()
        assert len(player_rows) == 10
        radiant_ids = tuple(
            r.player_id for r in player_rows if r.side == Side.RADIANT
        )
        dire_ids = tuple(r.player_id for r in player_rows if r.side == Side.DIRE)
        assert radiant_ids == match.radiant_player_ids
        assert dire_ids == match.dire_player_ids
        radiant_heroes = tuple(
            r.hero_id for r in player_rows if r.side == Side.RADIANT
        )
        dire_heroes = tuple(r.hero_id for r in player_rows if r.side == Side.DIRE)
        assert radiant_heroes == match.radiant_hero_ids
        assert dire_heroes == match.dire_hero_ids
        assert all(row.position is None for row in player_rows)
        assert all(row.lane is None for row in player_rows)
        assert all(row.role is None for row in player_rows)

        draft_rows = conn.execute(
            DRAFT_EVENTS.select()
            .where(DRAFT_EVENTS.c.match_id == 1000)
            .order_by(DRAFT_EVENTS.c.sequence)
        ).all()
        assert len(draft_rows) == len(match.draft_events)
        for row, event in zip(draft_rows, match.draft_events, strict=True):
            assert row.action == event.action
            assert row.side == event.side
            assert row.hero_id == event.hero_id


def test_write_canonical_match_round_trips_observed_position_metadata(engine):
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=16)

    positions = (
        MatchPlayerPosition.POSITION_1,
        MatchPlayerPosition.POSITION_2,
        MatchPlayerPosition.POSITION_3,
        MatchPlayerPosition.POSITION_4,
        MatchPlayerPosition.UNKNOWN,
    )
    lanes = (
        MatchLane.SAFE_LANE,
        MatchLane.MID_LANE,
        MatchLane.OFF_LANE,
        MatchLane.OFF_LANE,
        None,
    )
    roles = (
        MatchPlayerRole.CORE,
        MatchPlayerRole.CORE,
        MatchPlayerRole.CORE,
        MatchPlayerRole.LIGHT_SUPPORT,
        MatchPlayerRole.HARD_SUPPORT,
    )
    match = replace(
        build_canonical_match(match_id=1010, league_id=16, num_bans=4),
        radiant_positions=positions,
        radiant_lanes=lanes,
        radiant_roles=roles,
    )
    write_canonical_match(engine, match)

    with engine.connect() as conn:
        radiant_rows = conn.execute(
            MATCH_PLAYERS.select()
            .where(MATCH_PLAYERS.c.match_id == 1010, MATCH_PLAYERS.c.side == Side.RADIANT)
            .order_by(MATCH_PLAYERS.c.slot_in_side)
        ).all()
        dire_rows = conn.execute(
            MATCH_PLAYERS.select().where(
                MATCH_PLAYERS.c.match_id == 1010, MATCH_PLAYERS.c.side == Side.DIRE
            )
        ).all()
    assert tuple(row.position for row in radiant_rows) == positions
    assert tuple(row.lane for row in radiant_rows) == lanes
    assert tuple(row.role for row in radiant_rows) == roles
    assert all(row.position is None for row in dire_rows)



def test_write_canonical_match_is_upsert_for_matches_row(engine):
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=11)

    match_v1 = build_canonical_match(
        match_id=1001, league_id=11, num_bans=4, radiant_win=True
    )
    write_canonical_match(engine, match_v1)

    match_v2 = build_canonical_match(
        match_id=1001, league_id=11, num_bans=4, radiant_win=False
    )
    write_canonical_match(engine, match_v2)

    with engine.connect() as conn:
        rows = conn.execute(
            MATCHES.select().where(MATCHES.c.match_id == 1001)
        ).all()
        assert len(rows) == 1
        assert rows[0].radiant_win is False
        player_rows = conn.execute(
            MATCH_PLAYERS.select().where(MATCH_PLAYERS.c.match_id == 1001)
        ).all()
        assert {row.hero_id for row in player_rows} == set(
            match_v2.radiant_hero_ids + match_v2.dire_hero_ids
        )


def test_reprocessing_with_fewer_draft_events_leaves_no_stale_rows(engine):
    """The core property this revision required: delete+insert replacement,
    not upsert, for match_players/draft_events."""
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=12)

    large_match = build_canonical_match(match_id=1002, league_id=12, num_bans=14)
    write_canonical_match(engine, large_match)

    with engine.connect() as conn:
        count_before = conn.execute(
            DRAFT_EVENTS.select().where(DRAFT_EVENTS.c.match_id == 1002)
        ).all()
        assert len(count_before) == 24  # 14 bans + 10 picks

    small_match = build_canonical_match(match_id=1002, league_id=12, num_bans=2)
    write_canonical_match(engine, small_match)

    with engine.connect() as conn:
        draft_rows = conn.execute(
            DRAFT_EVENTS.select().where(DRAFT_EVENTS.c.match_id == 1002)
        ).all()
        assert len(draft_rows) == 12  # 2 bans + 10 picks -- no stale rows from the first write
        max_sequence = max(row.sequence for row in draft_rows)
        assert max_sequence == 11  # no leftover high-sequence rows from the 24-event write

        ban_rows = [r for r in draft_rows if r.action == DraftAction.BAN]
        assert len(ban_rows) == 2


def test_write_canonical_match_reuses_existing_team_and_player_identity_rows(engine):
    """Writing two matches that share a `team_id`/`player_id` must not
    error and must not duplicate identity rows -- the writer's identity
    inserts are `ON CONFLICT DO NOTHING`, not an upsert (see
    `storage.writer` module docstring)."""
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=13)

    match_a = build_canonical_match(match_id=1003, league_id=13, num_bans=4)
    write_canonical_match(engine, match_a)

    # Second match reuses the same radiant_team_id/dire_team_id and all
    # ten player_ids as match_a (build_canonical_match uses fixed ids).
    match_b = build_canonical_match(match_id=1004, league_id=13, num_bans=4)
    write_canonical_match(engine, match_b)  # must not raise

    with engine.connect() as conn:
        team_rows = conn.execute(
            TEAMS.select().where(
                TEAMS.c.team_id.in_([match_a.radiant_team_id, match_a.dire_team_id])
            )
        ).all()
        assert len(team_rows) == 2  # no duplicates despite two writes

        player_rows = conn.execute(
            PLAYERS.select().where(
                PLAYERS.c.player_id.in_(
                    match_a.radiant_player_ids + match_a.dire_player_ids
                )
            )
        ).all()
        assert len(player_rows) == 10  # no duplicates despite two writes


def test_write_canonical_match_preserves_per_match_observed_team_names(engine):
    """Two matches with the same `team_id` but different reported names
    each keep their own `*_team_name_observed` value -- this is an
    immutable per-match fact, not an entity-level field that a later
    write could overwrite (see `canonical_schema.CanonicalMatch` and
    `storage.schema` module docstrings)."""
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=14)

    match_2023 = build_canonical_match(match_id=1005, league_id=14, num_bans=4)
    write_canonical_match(engine, match_2023)

    from dataclasses import replace

    match_2024 = replace(
        match_2023,
        match_id=1006,
        radiant_team_name_observed="Renamed Radiant Org",
    )
    write_canonical_match(engine, match_2024)

    with engine.connect() as conn:
        rows = {
            row.match_id: row.radiant_team_name_observed
            for row in conn.execute(
                MATCHES.select().where(MATCHES.c.match_id.in_([1005, 1006]))
            ).all()
        }
    assert rows[1005] == "Radiant Team"
    assert rows[1006] == "Renamed Radiant Org"


def test_write_canonical_match_does_not_delete_orphaned_identity_rows(engine):
    """If a reprocessed match no longer references a `team_id`/`player_id`
    it previously referenced, the now-unreferenced `teams`/`players` row
    is left in place -- there is deliberately no cleanup/deletion logic
    (see `storage.schema` module docstring)."""
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=15)

    from dataclasses import replace

    original = build_canonical_match(match_id=1007, league_id=15, num_bans=4)
    write_canonical_match(engine, original)

    reprocessed = replace(original, dire_team_id=999999)
    write_canonical_match(engine, reprocessed)

    with engine.connect() as conn:
        # The old dire_team_id (200, from build_canonical_match) is no
        # longer referenced by any match, but its identity row persists.
        orphaned = conn.execute(
            TEAMS.select().where(TEAMS.c.team_id == 200)
        ).one_or_none()
        assert orphaned is not None

        match_row = conn.execute(
            MATCHES.select().where(MATCHES.c.match_id == 1007)
        ).one()
        assert match_row.dire_team_id == 999999
