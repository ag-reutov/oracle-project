"""Semantic tests for the `research.leagues` canonical league/event view (Slice 3).

Seeds canonical league rows and asserts the reference-entity league view:
one row per curated `league_id`, the canonical curated name, the explicit
source-vs-curated tier distinction (`stratz_tier` vs `liquipedia_tier`),
provenance columns, and that every referenced league resolves. The view is
created from the same SQL the Alembic migration applies
(`dota_predictor.research.views`).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from research_helpers import seed_match

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set; skipping DB-touching test",
)

UTC2024 = datetime(2024, 6, 1, tzinfo=UTC)


def _rows(engine, sql: str) -> list[sa.Row]:
    with engine.connect() as conn:
        return conn.execute(sa.text(sql)).all()


def _seed(conn, *, league_id: int, name: str, tier: str, stratz_tier: str | None = None):
    conn.execute(
        sa.text(
            "INSERT INTO leagues (league_id, name, stratz_tier, liquipedia_tier, "
            "in_scope, fetch_mode, source) "
            "VALUES (:league_id, :name, :stratz_tier, :tier, TRUE, 'league', 'test')"
        ),
        {"league_id": league_id, "name": name, "stratz_tier": stratz_tier, "tier": tier},
    )
    conn.execute(
        sa.text("INSERT INTO ingestion_leagues (league_id) VALUES (:league_id)"),
        {"league_id": league_id},
    )


def test_leagues_view_exposes_curated_identity_with_source_tier(engine) -> None:
    with engine.begin() as conn:
        _seed(
            conn,
            league_id=16169,
            name="BetBoom Dacha Dubai 2024",
            tier="T1",
            stratz_tier="PROFESSIONAL",
        )
        _seed(
            conn,
            league_id=19269,
            name="DreamLeague Season 28",
            tier="T1",
            stratz_tier="PROFESSIONAL",
        )
        seed_match(conn, match_id=1, league_id=16169, start_time=UTC2024)

    rows = _rows(engine, "SELECT * FROM research.leagues ORDER BY league_id")
    by_id = {int(r.league_id): r for r in rows}
    assert set(by_id.keys()) == {16169, 19269}

    r = by_id[16169]
    assert r.league_name == "BetBoom Dacha Dubai 2024"
    # Source vs curated tier are distinct columns and never conflated.
    assert r.stratz_tier == "PROFESSIONAL"
    assert r.liquipedia_tier == "T1"
    assert r.in_scope is True
    assert r.curation_source == "test"
    assert r.curated_at is not None


def test_leagues_view_preserves_curated_tier_independent_of_stratz(engine) -> None:
    """Our T1/T2 classification is a curated property, not an intrinsic STRATZ
    one: the same STRATZ PROFESSIONAL tier can carry different curated tiers."""
    with engine.begin() as conn:
        _seed(conn, league_id=100, name="T1 Event", tier="T1", stratz_tier="PROFESSIONAL")
        _seed(conn, league_id=101, name="T3 Event", tier="T3", stratz_tier="PROFESSIONAL")

    rows = {
        int(r.league_id): (r.stratz_tier, r.liquipedia_tier)
        for r in _rows(engine, "SELECT league_id, stratz_tier, liquipedia_tier "
                               "FROM research.leagues")
    }
    assert rows[100] == ("PROFESSIONAL", "T1")
    assert rows[101] == ("PROFESSIONAL", "T3")


def test_leagues_view_is_a_plain_projection_of_registry(engine) -> None:
    """research.leagues duplicates no storage: it mirrors the curated
    `leagues` registry exactly, so in-scope + out-of-scope rows both appear."""
    with engine.begin() as conn:
        _seed(conn, league_id=200, name="In Scope", tier="T2")
        conn.execute(
            sa.text(
                "INSERT INTO leagues (league_id, name, stratz_tier, liquipedia_tier, "
                "in_scope, fetch_mode) "
                "VALUES (201, 'Out Of Scope', NULL, 'QUALIFIER', FALSE, 'league')"
            )
        )

    rows = _rows(engine, "SELECT league_id, in_scope FROM research.leagues ORDER BY league_id")
    assert [(int(r.league_id), r.in_scope) for r in rows] == [
        (200, True),
        (201, False),
    ]