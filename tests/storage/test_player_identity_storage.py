"""DB-touching tests for the player-identity foundation (Slice 2).

Cover the Slice 2 invariants against a real (dedicated test) database:
one canonical player per valid source `player_id`, referential integrity,
idempotent deterministic registry backfill, orphan detection (registry ids
not referenced by any match are reported, never deleted), and holdout
protection (the identity layer never touches `matches` / `match_players`).

Skipped unless `TEST_DATABASE_URL` is set (see `tests/storage/conftest.py`).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from helpers import build_canonical_match, requires_test_database
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dota_predictor.data.player_identity import (
    audit_player_identity,
    derive_player_summaries,
    fetch_player_universe,
    list_player_ids_in_registry,
    run_backfill,
    sync_player_registry,
)
from dota_predictor.storage.schema import (
    INGESTION_LEAGUES,
    LEAGUES,
    MATCH_PLAYERS,
    MATCHES,
    PLAYERS,
)
from dota_predictor.storage.writer import write_canonical_match

pytestmark = requires_test_database


def _seed_league(engine, league_id: int) -> None:
    """Idempotent league allowlist seeding (safe across multiple matches)."""
    with engine.begin() as conn:
        conn.execute(
            pg_insert(LEAGUES)
            .values(
                league_id=league_id,
                name="Test League",
                liquipedia_tier="T1",
                in_scope=True,
            )
            .on_conflict_do_nothing(index_elements=[LEAGUES.c.league_id])
        )
        conn.execute(
            pg_insert(INGESTION_LEAGUES)
            .values(league_id=league_id)
            .on_conflict_do_nothing(index_elements=[INGESTION_LEAGUES.c.league_id])
        )


def _write_match(engine, *, match_id: int, league_id: int, **overrides):
    _seed_league(engine, league_id=league_id)
    match = replace(
        build_canonical_match(match_id=match_id, league_id=league_id),
        **overrides,
    )
    write_canonical_match(engine, match)
    return match


def test_backfill_ensures_registry_covers_every_referenced_player_id(engine):
    _write_match(engine, match_id=8000, league_id=80)
    _write_match(engine, match_id=8001, league_id=80)
    summary = run_backfill(engine)
    with engine.connect() as conn:
        registry = list_player_ids_in_registry(conn)
        referenced = {
            int(pid)
            for pid in conn.execute(
                select(MATCH_PLAYERS.c.player_id).distinct()
            ).scalars()
        }
    assert referenced <= registry
    assert summary["referenced_player_count"] == len(referenced)
    # build_canonical_match uses the same 10 player ids for every match, so
    # the writer already registered them; the backfill adds nothing.
    assert summary["added_player_ids"] == 0


def test_backfill_is_idempotent(engine):
    _write_match(engine, match_id=8010, league_id=81)
    _write_match(engine, match_id=8011, league_id=81)
    first = run_backfill(engine)
    second = run_backfill(engine)
    assert first == second
    with engine.connect() as conn:
        count = conn.execute(select(PLAYERS.c.player_id)).all()
    assert len(count) == first["registry_player_count"]


def test_sync_player_registry_adds_missing_ids(engine):
    """The repair path inserts unknown player ids into the registry without
    disturbing existing rows."""
    _write_match(engine, match_id=8020, league_id=82)
    with engine.begin() as conn:
        added = sync_player_registry(conn, {1, 2, 999})
    assert added == 1  # 1/2 already registered by the writer
    with engine.connect() as conn:
        registry = list_player_ids_in_registry(conn)
    assert 999 in registry


def test_backfill_never_touches_matches_or_match_players(engine):
    """The identity backfill writes only to the `players` registry, so no
    canonical match fact or player-observation row can change."""
    match = _write_match(engine, match_id=8030, league_id=83)
    with engine.connect() as conn:
        before = dict(
            conn.execute(
                select(
                    MATCHES.c.radiant_win,
                    MATCHES.c.duration_seconds,
                    MATCHES.c.mapper_version,
                    MATCHES.c.start_time,
                ).where(MATCHES.c.match_id == match.match_id)
            )
            .one()
            ._mapping
        )
        mp_before = conn.execute(
            select(MATCH_PLAYERS.c.match_id, MATCH_PLAYERS.c.player_id)
            .where(MATCH_PLAYERS.c.match_id == match.match_id)
            .order_by(MATCH_PLAYERS.c.side, MATCH_PLAYERS.c.slot_in_side)
        ).all()
    run_backfill(engine)
    run_backfill(engine)
    with engine.connect() as conn:
        after = dict(
            conn.execute(
                select(
                    MATCHES.c.radiant_win,
                    MATCHES.c.duration_seconds,
                    MATCHES.c.mapper_version,
                    MATCHES.c.start_time,
                ).where(MATCHES.c.match_id == match.match_id)
            )
            .one()
            ._mapping
        )
        mp_after = conn.execute(
            select(MATCH_PLAYERS.c.match_id, MATCH_PLAYERS.c.player_id)
            .where(MATCH_PLAYERS.c.match_id == match.match_id)
            .order_by(MATCH_PLAYERS.c.side, MATCH_PLAYERS.c.slot_in_side)
        ).all()
    assert before == after
    assert mp_before == mp_after


def test_orphan_registry_ids_reported_and_preserved(engine):
    """Registry ids no longer referenced by any match are deliberately kept
    (no cleanup) and remain measurable via the registry/list query."""
    _write_match(engine, match_id=8040, league_id=84)
    with engine.begin() as conn:
        sync_player_registry(conn, {5555555})
    with engine.connect() as conn:
        before = list_player_ids_in_registry(conn)
    run_backfill(engine)
    with engine.connect() as conn:
        after = list_player_ids_in_registry(conn)
    assert 5555555 in before
    assert before == after


def test_fetch_player_universe_matches_pure_derivation(engine):
    _write_match(
        engine,
        match_id=8050,
        league_id=85,
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
    )
    _write_match(
        engine,
        match_id=8051,
        league_id=85,
        start_time=datetime(2024, 3, 1, tzinfo=UTC),
    )
    with engine.connect() as conn:
        universe = {u.player_id: u for u in fetch_player_universe(conn)}
        rows = conn.execute(
            select(
                MATCH_PLAYERS.c.player_id,
                MATCHES.c.start_time,
            ).join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
        ).all()
    observations = [(int(r.player_id), r.start_time) for r in rows]
    expected = {s.player_id: s for s in derive_player_summaries(observations)}
    assert set(universe.keys()) == set(expected.keys())
    for player_id, summary in expected.items():
        identity = universe[player_id]
        assert identity.first_seen_at == summary.first_seen_at
        assert identity.last_seen_at == summary.last_seen_at
        assert identity.match_count == summary.match_count
        assert identity.display_name is None
    assert len(universe) == 10


def test_repeated_appearances_do_not_create_duplicate_identities(engine):
    """The same player in many matches still yields exactly one universe row."""
    for match_id in range(9000, 9010):
        _write_match(engine, match_id=match_id, league_id=90)
    with engine.connect() as conn:
        universe = fetch_player_universe(conn)
    player_ids = [u.player_id for u in universe]
    assert len(player_ids) == len(set(player_ids)) == 10
    assert all(u.match_count == 10 for u in universe)


def test_audit_report_is_deterministic_and_reports_integrity(engine):
    """The identity audit measures the guarantees deterministically: every
    referenced player id resolves (no unresolved/null/invalid ids), and
    re-running reproduces the same report."""
    _write_match(engine, match_id=9100, league_id=91)
    _write_match(engine, match_id=9101, league_id=91)
    first = audit_player_identity(engine)
    second = audit_player_identity(engine)
    assert first == second
    assert first["distinct_valid_player_ids"] == 10
    assert first["unresolved_player_ids"] == 0
    assert first["null_player_id_rows"] == 0
    assert first["invalid_player_id_rows"] == 0
    assert first["integrity_violations"] == []
    assert first["canonical_player_count"] == 10
    assert first["matches_per_player"] == {"min": 2, "median": 2, "max": 2, "count": 10}
