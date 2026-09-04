"""DB-touching tests for the team-identity layer (Slice 1).

Cover the Slice 1 invariants against a real (dedicated test) database:
historical preservation, no silent merge, explicit organization grouping,
deterministic alias/tag derivation, referential integrity, idempotent
backfill, and holdout protection (the identity layer never touches
`matches`).

Skipped unless `TEST_DATABASE_URL` is set (see `tests/storage/conftest.py`).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from helpers import build_canonical_match, requires_test_database
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dota_predictor.data.team_identity import (
    derive_team_aliases,
    run_backfill,
    sync_team_organizations,
)
from dota_predictor.storage.schema import (
    INGESTION_LEAGUES,
    LEAGUES,
    MATCHES,
    ORGANIZATIONS,
    STRATZ_RAW_MATCHES,
    TEAM_ALIASES,
    TEAM_ORGANIZATION_MEMBERSHIPS,
    TEAM_TAGS,
    TEAMS,
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
        build_canonical_match(
            match_id=match_id, league_id=league_id, num_bans=4
        ),
        **overrides,
    )
    write_canonical_match(engine, match)
    return match


def _seed_raw_payload(
    engine,
    *,
    match_id: int,
    league_id: int,
    radiant_id: int,
    radiant_name: str | None,
    radiant_tag: str | None,
    dire_id: int,
    dire_name: str | None,
    dire_tag: str | None,
    start_unix: int,
) -> None:
    with engine.begin() as conn:
        payload = {
            "id": match_id,
            "startDateTime": start_unix,
            "radiantTeam": {"id": radiant_id, "name": radiant_name, "tag": radiant_tag},
            "direTeam": {"id": dire_id, "name": dire_name, "tag": dire_tag},
        }
        conn.execute(
            STRATZ_RAW_MATCHES.insert().values(
                match_id=match_id,
                league_id=league_id,
                payload=payload,
                fetched_at=datetime.fromtimestamp(start_unix, tz=UTC),
            )
        )


def _alias_rows(engine) -> list[tuple[int, str, int]]:
    with engine.connect() as conn:
        return [
            (int(r.team_id), str(r.name), int(r.observation_count))
            for r in conn.execute(
                select(TEAM_ALIASES.c.team_id, TEAM_ALIASES.c.name, TEAM_ALIASES.c.observation_count).order_by(
                    TEAM_ALIASES.c.team_id, TEAM_ALIASES.c.name
                )
            ).all()
        ]


def test_backfill_preserves_historical_match_facts(engine):
    """Rebuilding identity must not alter observed-name match facts."""
    match = _write_match(
        engine,
        match_id=7000,
        league_id=70,
        radiant_team_name_observed="Virtus.pro",
        dire_team_name_observed="Team Spirit",
        start_time=datetime(2024, 5, 1, tzinfo=UTC),
    )
    with engine.connect() as conn:
        before = dict(
            conn.execute(
                select(
                    MATCHES.c.radiant_team_name_observed,
                    MATCHES.c.dire_team_name_observed,
                    MATCHES.c.radiant_team_id,
                    MATCHES.c.dire_team_id,
                    MATCHES.c.mapper_version,
                    MATCHES.c.radiant_win,
                ).where(MATCHES.c.match_id == match.match_id)
            ).one()._mapping
        )

    run_backfill(engine)
    run_backfill(engine)

    with engine.connect() as conn:
        after = dict(
            conn.execute(
                select(
                    MATCHES.c.radiant_team_name_observed,
                    MATCHES.c.dire_team_name_observed,
                    MATCHES.c.radiant_team_id,
                    MATCHES.c.dire_team_id,
                    MATCHES.c.mapper_version,
                    MATCHES.c.radiant_win,
                ).where(MATCHES.c.match_id == match.match_id)
            ).one()._mapping
        )
    assert before == after


def test_no_silent_merge_same_name_two_team_ids_two_source_teams(engine):
    """Team ID 100 and team ID 200 both named 'Example Team' must remain
    two source-team identities with two alias rows, not one merged team."""
    _write_match(
        engine,
        match_id=7001,
        league_id=71,
        radiant_team_id=100,
        radiant_team_name_observed="Example Team",
        dire_team_id=101,
        dire_team_name_observed="Other Team",
    )
    _write_match(
        engine,
        match_id=7002,
        league_id=71,
        radiant_team_id=200,
        radiant_team_name_observed="Example Team",
        dire_team_id=201,
        dire_team_name_observed="Other Team",
    )
    run_backfill(engine)
    assert _alias_rows(engine) == [
        (100, "Example Team", 1),
        (101, "Other Team", 1),
        (200, "Example Team", 1),
        (201, "Other Team", 1),
    ]
    # Both source teams remain distinct in the registry.
    with engine.connect() as conn:
        team_ids = {
            int(r.team_id)
            for r in conn.execute(select(TEAMS.c.team_id).where(TEAMS.c.team_id.in_([100, 200])))
        }
    assert team_ids == {100, 200}


def test_alias_derivation_records_periods_and_rename_history(engine):
    """A team renamed over time produces one row per name with periods."""
    _write_match(
        engine,
        match_id=7010,
        league_id=72,
        radiant_team_id=300,
        radiant_team_name_observed="Old Name",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
    )
    _write_match(
        engine,
        match_id=7011,
        league_id=72,
        radiant_team_id=300,
        radiant_team_name_observed="Old Name",
        start_time=datetime(2024, 3, 1, tzinfo=UTC),
    )
    _write_match(
        engine,
        match_id=7012,
        league_id=72,
        radiant_team_id=300,
        radiant_team_name_observed="New Name",
        start_time=datetime(2024, 5, 1, tzinfo=UTC),
    )
    run_backfill(engine)
    with engine.connect() as conn:
        rows = {
            (int(r.team_id), str(r.name)): (
                r.first_seen_at,
                r.last_seen_at,
                int(r.observation_count),
            )
            for r in conn.execute(
                select(
                    TEAM_ALIASES.c.team_id,
                    TEAM_ALIASES.c.name,
                    TEAM_ALIASES.c.first_seen_at,
                    TEAM_ALIASES.c.last_seen_at,
                    TEAM_ALIASES.c.observation_count,
                )
            ).all()
        }
    assert rows[(300, "Old Name")] == (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 3, 1, tzinfo=UTC),
        2,
    )
    assert rows[(300, "New Name")] == (
        datetime(2024, 5, 1, tzinfo=UTC),
        datetime(2024, 5, 1, tzinfo=UTC),
        1,
    )


def test_backfill_records_tags_from_raw_payloads(engine):
    _write_match(engine, match_id=7020, league_id=73)
    with engine.begin() as conn:
        # match 7020 uses radiant 100 / dire 200 (build_canonical_match).
        # Insert a raw payload whose radiant tag is 'VP' and whose dire tag
        # is missing.
        conn.execute(
            STRATZ_RAW_MATCHES.insert().values(
                match_id=7020,
                league_id=73,
                payload={
                    "id": 7020,
                    "startDateTime": 1704067200,
                    "radiantTeam": {"id": 100, "name": "Radiant Team", "tag": "VP"},
                    "direTeam": {"id": 200, "name": "Dire Team", "tag": None},
                },
                fetched_at=datetime(2024, 5, 1, tzinfo=UTC),
            )
        )
    summary = run_backfill(engine)
    with engine.connect() as conn:
        tags = {
            (int(r.team_id), str(r.tag)): int(r.observation_count)
            for r in conn.execute(
                select(TEAM_TAGS.c.team_id, TEAM_TAGS.c.tag, TEAM_TAGS.c.observation_count)
            ).all()
        }
    assert tags == {(100, "VP"): 1}
    assert summary["skipped_raw_team_ids"] == []


def test_backfill_skips_raw_team_ids_outside_registry(engine):
    """Raw payloads referencing team ids not in the `teams` registry must
    not create team_tags rows (referential integrity of the identity
    layer) and must be reported."""
    _write_match(engine, match_id=7021, league_id=74)
    _seed_raw_payload(
        engine,
        match_id=7021,
        league_id=74,
        radiant_id=100,
        radiant_name="Radiant Team",
        radiant_tag="VP",
        dire_id=999999,
        dire_name="Ghost",
        dire_tag="GHOST",
        start_unix=1704067200,
    )
    summary = run_backfill(engine)
    assert summary["skipped_raw_team_ids"] == [999999]
    with engine.connect() as conn:
        tag_team_ids = {int(r.team_id) for r in conn.execute(select(TEAM_TAGS.c.team_id))}
        alias_team_ids = {
            int(r.team_id) for r in conn.execute(select(TEAM_ALIASES.c.team_id))
        }
    assert tag_team_ids == {100}
    assert alias_team_ids == {100, 200}


def test_backfill_is_idempotent(engine):
    _write_match(engine, match_id=7030, league_id=75)
    _write_match(engine, match_id=7031, league_id=75)
    _seed_raw_payload(
        engine,
        match_id=7030,
        league_id=75,
        radiant_id=100,
        radiant_name="Radiant Team",
        radiant_tag="VP",
        dire_id=200,
        dire_name="Dire Team",
        dire_tag="DT",
        start_unix=1704067200,
    )
    first = run_backfill(engine)
    second = run_backfill(engine)
    assert first["alias_rows"] == second["alias_rows"]
    assert first["tag_rows"] == second["tag_rows"]
    with engine.connect() as conn:
        alias_count = conn.execute(select(TEAM_ALIASES.c.team_id)).all()
        tag_count = conn.execute(select(TEAM_TAGS.c.team_id)).all()
        alias_pks = {
            (int(r.team_id), str(r.name)) for r in conn.execute(select(TEAM_ALIASES.c.team_id, TEAM_ALIASES.c.name))
        }
        tag_pks = {
            (int(r.team_id), str(r.tag)) for r in conn.execute(select(TEAM_TAGS.c.team_id, TEAM_TAGS.c.tag))
        }
    assert len(alias_count) == first["alias_rows"]
    assert len(tag_count) == first["tag_rows"]
    assert len(alias_pks) == len(alias_count)
    assert len(tag_pks) == len(tag_count)


def test_referential_integrity_aliases_and_tags_resolve_to_registry(engine):
    _write_match(engine, match_id=7040, league_id=76)
    _seed_raw_payload(
        engine,
        match_id=7040,
        league_id=76,
        radiant_id=100,
        radiant_name="Radiant Team",
        radiant_tag="VP",
        dire_id=200,
        dire_name="Dire Team",
        dire_tag="DT",
        start_unix=1704067200,
    )
    run_backfill(engine)
    with engine.connect() as conn:
        registry = {
            int(r.team_id) for r in conn.execute(select(TEAMS.c.team_id))
        }
        alias_ids = {
            int(r.team_id) for r in conn.execute(select(TEAM_ALIASES.c.team_id))
        }
        tag_ids = {int(r.team_id) for r in conn.execute(select(TEAM_TAGS.c.team_id))}
        # Every team referenced by canonical matches resolves to the raw
        # teams registry (FK already guarantees this; assert it holds).
        referenced = {
            int(r.radiant_team_id)
            for r in conn.execute(
                select(MATCHES.c.radiant_team_id).where(MATCHES.c.radiant_team_id == 100)
            )
        } | {
            int(r.dire_team_id)
            for r in conn.execute(
                select(MATCHES.c.dire_team_id).where(MATCHES.c.dire_team_id == 200)
            )
        }
    assert referenced <= registry
    assert alias_ids <= registry
    assert tag_ids <= registry


def test_explicit_organization_grouping_keeps_source_identities_distinct(engine):
    """IDs 100 and 200 explicitly mapped to organization X: source
    identities stay distinct, organization lookup returns the same group,
    and matches still contain the original team ids."""
    _write_match(engine, match_id=7050, league_id=77, radiant_team_id=100, dire_team_id=200)
    entries = [
        {
            "organization_id": 1,
            "name": "Test Org",
            "team_ids": [100, 200],
            "reason": "explicit test mapping",
            "source": "test",
        }
    ]
    with engine.begin() as conn:
        sync_team_organizations(conn, entries)

    with engine.connect() as conn:
        memberships = {
            int(r.team_id): int(r.organization_id)
            for r in conn.execute(select(TEAM_ORGANIZATION_MEMBERSHIPS.c.team_id, TEAM_ORGANIZATION_MEMBERSHIPS.c.organization_id))
        }
        org_names = {
            int(r.organization_id): str(r.name)
            for r in conn.execute(select(ORGANIZATIONS.c.organization_id, ORGANIZATIONS.c.name))
        }
        match_teams = conn.execute(
            select(MATCHES.c.radiant_team_id, MATCHES.c.dire_team_id).where(MATCHES.c.match_id == 7050)
        ).one()
    assert memberships == {100: 1, 200: 1}
    assert org_names == {1: "Test Org"}
    # Matches keep the original raw team ids.
    assert (match_teams.radiant_team_id, match_teams.dire_team_id) == (100, 200)


def test_organization_loader_rejects_team_not_in_registry(engine):
    entries = [
        {
            "organization_id": 2,
            "name": "Ghost Org",
            "team_ids": [123456789],
            "reason": "test",
            "source": "test",
        }
    ]
    with engine.begin() as conn, pytest.raises(ValueError, match="not in the teams registry"):
        sync_team_organizations(conn, entries)


def test_organization_loader_is_convergent(engine):
    _write_match(engine, match_id=7060, league_id=78, radiant_team_id=100, dire_team_id=200)
    with engine.begin() as conn:
        sync_team_organizations(
            conn,
            [
                {
                    "organization_id": 3,
                    "name": "Org A",
                    "team_ids": [100, 200],
                    "source": "test",
                }
            ],
        )
    # Re-run with team 200 removed from the mapping: its membership must
    # disappear (convergent sync), while 100 stays mapped.
    with engine.begin() as conn:
        sync_team_organizations(
            conn,
            [
                {
                    "organization_id": 3,
                    "name": "Org A",
                    "team_ids": [100],
                    "source": "test",
                }
            ],
        )
    with engine.connect() as conn:
        memberships = {
            int(r.team_id)
            for r in conn.execute(select(TEAM_ORGANIZATION_MEMBERSHIPS.c.team_id))
        }
    assert memberships == {100}


def test_holdout_protection_identity_layer_never_touches_matches(engine):
    """No predictive feature, Elo state, frozen state, label, or evaluation
    result can change as part of the identity layer: the backfill and the
    org loader write only to the identity tables, never to `matches`."""
    _write_match(
        engine,
        match_id=7070,
        league_id=79,
        radiant_team_id=100,
        dire_team_id=200,
    )
    _seed_raw_payload(
        engine,
        match_id=7070,
        league_id=79,
        radiant_id=100,
        radiant_name="Radiant Team",
        radiant_tag="VP",
        dire_id=200,
        dire_name="Dire Team",
        dire_tag="DT",
        start_unix=1704067200,
    )
    with engine.connect() as conn:
        before = [
            dict(r._mapping)
            for r in conn.execute(
                select(
                    MATCHES.c.match_id,
                    MATCHES.c.league_id,
                    MATCHES.c.radiant_team_id,
                    MATCHES.c.dire_team_id,
                    MATCHES.c.radiant_team_name_observed,
                    MATCHES.c.dire_team_name_observed,
                    MATCHES.c.radiant_win,
                    MATCHES.c.duration_seconds,
                    MATCHES.c.mapper_version,
                ).order_by(MATCHES.c.match_id)
            ).all()
        ]
    run_backfill(engine)
    with engine.begin() as conn:
        sync_team_organizations(
            conn,
            [
                {
                    "organization_id": 9,
                    "name": "Holdout Org",
                    "team_ids": [100, 200],
                    "source": "test",
                }
            ],
        )
    with engine.connect() as conn:
        after = [
            dict(r._mapping)
            for r in conn.execute(
                select(
                    MATCHES.c.match_id,
                    MATCHES.c.league_id,
                    MATCHES.c.radiant_team_id,
                    MATCHES.c.dire_team_id,
                    MATCHES.c.radiant_team_name_observed,
                    MATCHES.c.dire_team_name_observed,
                    MATCHES.c.radiant_win,
                    MATCHES.c.duration_seconds,
                    MATCHES.c.mapper_version,
                ).order_by(MATCHES.c.match_id)
            ).all()
        ]
    assert before == after


def test_derive_team_aliases_consistency_with_db_backfill(engine):
    """The pure derivation over collected observations matches what the DB
    stores after `run_backfill`."""
    _write_match(engine, match_id=7080, league_id=80, radiant_team_id=100, dire_team_id=200)
    run_backfill(engine)
    with engine.connect() as conn:
        observations = []
        for r in conn.execute(
            select(
                MATCHES.c.radiant_team_id,
                MATCHES.c.radiant_team_name_observed,
                MATCHES.c.dire_team_id,
                MATCHES.c.dire_team_name_observed,
                MATCHES.c.start_time,
            )
        ).all():
            observations.append((r.radiant_team_id, r.radiant_team_name_observed, r.start_time))
            observations.append((r.dire_team_id, r.dire_team_name_observed, r.start_time))
    expected = {
        (a.team_id, a.name): a.observation_count for a in derive_team_aliases(observations)
    }
    assert set(_alias_rows(engine)) == {
        (team_id, name, count)
        for (team_id, name), count in expected.items()
    }