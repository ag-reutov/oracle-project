"""DB-touching tests for the reference-entity census audit (Slice 3).

Cover the Slice 3 invariants against a real (dedicated test) database:
the reference-entity census resolves every referenced hero / league /
game-version id against the canonical reference layer, unresolved ids are
reported (never silently invented), provenance/source-vs-curated
distinctions are preserved, and re-running the audit is deterministic.

Skipped unless `TEST_DATABASE_URL` is set (see `tests/storage/conftest.py`).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from helpers import build_canonical_match, requires_test_database
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dota_predictor.data.reference_identity import (
    audit_reference_entities,
    fetch_league_identities,
)
from dota_predictor.datasets.reference_export import build_reference_dataset
from dota_predictor.storage.schema import (
    DRAFT_EVENTS,
    INGESTION_LEAGUES,
    LEAGUES,
    MATCHES,
)
from dota_predictor.storage.writer import write_canonical_match

pytestmark = requires_test_database

RETRIEVED = datetime(2026, 1, 1, tzinfo=UTC)


def _seed_league(engine, league_id: int, *, name: str = "Test League", tier: str = "T1"):
    with engine.begin() as conn:
        conn.execute(
            pg_insert(LEAGUES)
            .values(
                league_id=league_id,
                name=name,
                stratz_tier="PROFESSIONAL",
                liquipedia_tier=tier,
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


def _write_reference(tmp_path: Path) -> tuple[Path, Path]:
    build_reference_dataset(
        tmp_path,
        heroes=[
            {"id": 1, "displayName": "Anti-Mage", "shortName": "antimage", "aliases": ["am"]},
            {"id": 2, "displayName": "Axe", "shortName": "axe", "aliases": []},
        ],
        game_versions=[
            {"id": 173, "name": "7.36", "asOfDateTime": 1716422400},
            {"id": 175, "name": "7.36c", "asOfDateTime": 1719187200},
        ],
        retrieved_at=RETRIEVED,
    )
    return tmp_path / "heroes.parquet", tmp_path / "game_versions.parquet"


def test_audit_resolves_all_referenced_reference_ids(engine, tmp_path: Path) -> None:
    heroes_path, versions_path = _write_reference(tmp_path)
    # build_canonical_match uses hero ids 1..14 and game version 173; the
    # reference catalog covers hero ids 1,2 and versions 173,175. Every
    # referenced id outside the catalog is reported as unresolved, never
    # silently invented, and the referenced league always resolves.
    _write_match(engine, match_id=8000, league_id=80, game_version_id=173)

    report = audit_reference_entities(
        engine, heroes_path=heroes_path, game_versions_path=versions_path
    )

    assert report["league_ids_referenced"] == 1
    assert report["league_ids_resolved"] == 1
    assert report["league_ids_unresolved"] == 0
    assert report["hero_ids_referenced"] == 14
    assert report["hero_ids_resolved"] == 2
    assert report["hero_ids_unresolved"] == 12
    assert report["game_version_ids_referenced"] == 1
    assert report["game_version_ids_resolved"] == 1
    assert report["game_version_ids_unresolved"] == 0
    assert report["null_game_version_matches"] == 0
    assert report["regions"]["status"] == "deferred"


def test_audit_is_deterministic_and_read_only(engine, tmp_path: Path) -> None:
    heroes_path, versions_path = _write_reference(tmp_path)
    _write_match(engine, match_id=8010, league_id=81)

    first = audit_reference_entities(
        engine, heroes_path=heroes_path, game_versions_path=versions_path
    )
    second = audit_reference_entities(
        engine, heroes_path=heroes_path, game_versions_path=versions_path
    )
    assert first == second
    with engine.connect() as conn:
        assert conn.execute(select(MATCHES.c.match_id)).scalars().all() == [8010]
        assert (
            conn.execute(select(DRAFT_EVENTS.c.match_id).distinct()).scalars().all()
            == [8010]
        )


def test_fetch_league_identities_preserves_source_vs_curated_tier(
    engine,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            pg_insert(LEAGUES)
            .values(
                league_id=16169,
                name="BetBoom Dacha Dubai 2024",
                stratz_tier="PROFESSIONAL",
                liquipedia_tier="T1",
                in_scope=True,
            )
            .on_conflict_do_nothing(index_elements=[LEAGUES.c.league_id])
        )
        conn.execute(
            pg_insert(INGESTION_LEAGUES)
            .values(league_id=16169)
            .on_conflict_do_nothing(index_elements=[INGESTION_LEAGUES.c.league_id])
        )

    with engine.connect() as conn:
        identities = fetch_league_identities(conn)
    by_id = {identity.league_id: identity for identity in identities}
    assert 16169 in by_id
    league = by_id[16169]
    assert league.name == "BetBoom Dacha Dubai 2024"
    assert league.stratz_tier == "PROFESSIONAL"  # source-provided
    assert league.liquipedia_tier == "T1"  # curated
    assert league.in_scope is True


def test_audit_reports_unresolved_league_id(engine, tmp_path: Path) -> None:
    heroes_path, versions_path = _write_reference(tmp_path)
    # A match referencing a league not present in the registry is impossible
    # by FK, so unresolved leagues must stay at zero for a well-formed corpus.
    _write_match(engine, match_id=8020, league_id=82)
    report = audit_reference_entities(
        engine, heroes_path=heroes_path, game_versions_path=versions_path
    )
    assert report["league_ids_unresolved"] == 0
    assert report["league_ids_referenced"] == 1