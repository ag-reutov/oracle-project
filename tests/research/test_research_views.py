"""Semantic behavior tests for the `research` analytical views.

These seed canonical relational rows (leagues, matches, match_players,
draft_events, match_classifications) into the test database and assert on
the research views, which are created from the same SQL the Alembic
migration applies (`dota_predictor.research.views`).

The warehouse's total match count is intentionally NOT pinned: these tests
assert semantic behavior, and the corpus is allowed to grow.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from research_helpers import classify_match, seed_league, seed_match
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set; skipping DB-touching test",
)

UTC2024 = datetime(2024, 6, 1, tzinfo=UTC)
UTC2025 = datetime(2025, 6, 1, tzinfo=UTC)
UTC2023 = datetime(2023, 6, 1, tzinfo=UTC)


def _query(engine, sql: str) -> list[sa.Row]:
    with engine.connect() as conn:
        return conn.execute(sa.text(sql)).all()


def _count(engine, sql: str) -> int:
    return int(_query(engine, sql)[0][0])


def test_match_level_tier_override_wins_over_league_default(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 10, name="Shared League", tier="T2")
        seed_match(conn, match_id=1, league_id=10, start_time=UTC2024)
        seed_match(conn, match_id=2, league_id=10, start_time=UTC2024)
        classify_match(conn, 2, event="T3 Sub Event", tier="T3")

    rows = _query(
        engine,
        "SELECT match_id, effective_tier, effective_event, classification_source "
        "FROM research.matches ORDER BY match_id",
    )
    assert rows[0].effective_tier == "T2"
    assert rows[0].effective_event == "Shared League"
    assert rows[0].classification_source == "league default"
    assert rows[1].effective_tier == "T3"
    assert rows[1].effective_event == "T3 Sub Event"
    assert rows[1].classification_source == "match-level override"


def test_unclassified_matches_inherit_league_default_tier(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 20, name="T1 League", tier="T1")
        seed_match(conn, match_id=3, league_id=20, start_time=UTC2024)
        seed_match(conn, match_id=4, league_id=20, start_time=UTC2024)

    rows = _query(
        engine,
        "SELECT match_id, effective_tier, default_tier, classification_source "
        "FROM research.matches ORDER BY match_id",
    )
    assert [r.effective_tier for r in rows] == ["T1", "T1"]
    assert [r.default_tier for r in rows] == ["T1", "T1"]
    assert all(r.classification_source == "league default" for r in rows)


def test_acl_split_classifies_shared_league_matches(engine) -> None:
    """A T2 event embedded in a default-T3 league resolves T2 for the
    classified matches and T3 for the rest (the ACL 2025 / league 17875
    pattern)."""
    with engine.begin() as conn:
        seed_league(conn, 17875, name="ACL X ESL Challenger China", tier="T3")
        seed_match(conn, match_id=11, league_id=17875, start_time=UTC2025)
        seed_match(conn, match_id=12, league_id=17875, start_time=UTC2025)
        seed_match(conn, match_id=13, league_id=17875, start_time=UTC2025)
        classify_match(conn, 12, event="Asian Champions League 2025", tier="T2")

    by_tier = dict(
        _query(
            engine,
            "SELECT effective_tier, count(*) FROM research.matches "
            "WHERE league_id = 17875 GROUP BY effective_tier ORDER BY effective_tier",
        )
    )
    assert by_tier == {"T2": 1, "T3": 2}

    in_t12 = sorted(
        r[0]
        for r in _query(
            engine,
            "SELECT match_id FROM research.t12_matches WHERE league_id = 17875 "
            "ORDER BY match_id",
        )
    )
    assert in_t12 == [12]


def test_umbrella_t3_subevent_excluded_from_t12(engine) -> None:
    """A T3 sub-event inside a default-T2 umbrella must not leak into the
    T1/T2 corpus (the 1win Punch / league 16427 pattern)."""
    with engine.begin() as conn:
        seed_league(conn, 16427, name="1WIN SERIES DOTA 2", tier="T2")
        seed_match(conn, match_id=21, league_id=16427, start_time=UTC2024)
        seed_match(conn, match_id=22, league_id=16427, start_time=UTC2024)
        classify_match(conn, 22, event="1win Series Dota 2 Punch", tier="T3")

    assert (
        _count(
            engine, "SELECT count(*) FROM research.t12_matches WHERE league_id = 16427"
        )
        == 1
    )
    assert (
        _count(engine, "SELECT count(*) FROM research.t12_matches WHERE match_id = 22")
        == 0
    )
    assert (
        _count(
            engine, "SELECT count(*) FROM research.pro_matches WHERE league_id = 16427"
        )
        == 2
    )
    assert (
        _count(
            engine,
            "SELECT count(*) FROM research.matches WHERE league_id = 16427 AND effective_tier = 'T3'",
        )
        == 1
    )


def test_population_views_use_effective_classification(engine) -> None:
    """`research.t12_matches` must use the effective tier, not the league
    default, so a T2-classified match inside a T3 league is included."""
    with engine.begin() as conn:
        seed_league(conn, 17875, name="ACL X ESL Challenger China", tier="T3")
        seed_match(conn, match_id=31, league_id=17875, start_time=UTC2025)
        classify_match(conn, 31, event="Asian Champions League 2025", tier="T2")

    assert (
        _count(engine, "SELECT count(*) FROM research.t12_matches WHERE match_id = 31")
        == 1
    )
    assert (
        _count(
            engine, "SELECT count(*) FROM research.t12_matches WHERE league_id = 17875"
        )
        == 1
    )


def test_qualifier_league_cannot_enter_canonical_matches(engine) -> None:
    """A qualifier (in_scope=false, not allowlisted) league can never produce
    canonical matches, so it can never appear in the research views."""
    with engine.begin() as conn:
        seed_league(conn, 17299, name="Qualifier", tier="QUALIFIER", in_scope=False)
        with pytest.raises(IntegrityError):
            seed_match(conn, match_id=41, league_id=17299, start_time=UTC2024)

    assert (
        _count(engine, "SELECT count(*) FROM research.matches WHERE league_id = 17299")
        == 0
    )
    assert (
        _count(
            engine, "SELECT count(*) FROM research.t12_matches WHERE league_id = 17299"
        )
        == 0
    )


def test_draft_incomplete_remains_in_general_research_population(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 30, name="T1 League", tier="T1")
        seed_match(
            conn,
            match_id=51,
            league_id=30,
            start_time=UTC2025,
            draft_complete=False,
            with_draft=False,
        )

    assert (
        _count(engine, "SELECT count(*) FROM research.matches WHERE match_id = 51") == 1
    )
    assert (
        _count(
            engine, "SELECT count(*) FROM research.player_matches WHERE match_id = 51"
        )
        == 10
    )
    assert (
        _count(engine, "SELECT count(*) FROM research.t12_matches WHERE match_id = 51")
        == 1
    )


def test_draft_incomplete_excluded_from_draft_population(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 31, name="T1 League", tier="T1")
        seed_match(
            conn,
            match_id=52,
            league_id=31,
            start_time=UTC2025,
            draft_complete=False,
            with_draft=False,
        )
        seed_match(conn, match_id=53, league_id=31, start_time=UTC2025)

    assert (
        _count(
            engine,
            "SELECT count(*) FROM research.t12_draft_matches WHERE match_id = 52",
        )
        == 0
    )
    assert (
        _count(
            engine,
            "SELECT count(*) FROM research.t12_draft_matches WHERE match_id = 53",
        )
        == 1
    )
    assert (
        _count(engine, "SELECT count(*) FROM research.draft_events WHERE match_id = 52")
        == 0
    )


def test_research_match_ids_unique_and_one_row_per_match(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 40, name="League", tier="T2")
        for match_id in (61, 62, 63, 64):
            seed_match(conn, match_id=match_id, league_id=40, start_time=UTC2024)

    assert _count(engine, "SELECT count(*) FROM research.matches") == 4
    assert _count(engine, "SELECT count(DISTINCT match_id) FROM research.matches") == 4
    assert (
        _count(
            engine,
            "SELECT count(*) FROM (SELECT match_id FROM research.matches "
            "GROUP BY match_id HAVING count(*) > 1) d",
        )
        == 0
    )


def test_every_match_has_ten_player_rows_and_player_win(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 50, name="League", tier="T2")
        seed_match(
            conn, match_id=71, league_id=50, start_time=UTC2024, radiant_win=True
        )
        seed_match(
            conn, match_id=72, league_id=50, start_time=UTC2024, radiant_win=False
        )

    for match_id in (71, 72):
        assert (
            _count(
                engine,
                f"SELECT count(*) FROM research.player_matches WHERE match_id = {match_id}",
            )
            == 10
        )

    radiant_wins = _count(
        engine,
        "SELECT count(*) FROM research.player_matches "
        "WHERE match_id = 71 AND side = 'RADIANT' AND player_win",
    )
    dire_wins = _count(
        engine,
        "SELECT count(*) FROM research.player_matches "
        "WHERE match_id = 71 AND side = 'DIRE' AND NOT player_win",
    )
    assert radiant_wins == 5
    assert dire_wins == 5

    radiant_losses = _count(
        engine,
        "SELECT count(*) FROM research.player_matches "
        "WHERE match_id = 72 AND side = 'RADIANT' AND NOT player_win",
    )
    assert radiant_losses == 5


def test_draft_events_grain_matches_canonical(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 60, name="League", tier="T2")
        seed_match(conn, match_id=81, league_id=60, start_time=UTC2024)

    assert (
        _count(engine, "SELECT count(*) FROM research.draft_events WHERE match_id = 81")
        == 24
    )
    assert (
        _count(
            engine,
            "SELECT count(*) FROM (SELECT match_id, sequence FROM research.draft_events "
            "GROUP BY match_id, sequence HAVING count(*) > 1) d",
        )
        == 0
    )


def test_umbrella_league_exact_event_identity(engine) -> None:
    """All matches in a shared umbrella league resolve to their exact
    Liquipedia event name/tier when each event window is classified (the
    league-16427 Spring/Summer/Fall/Punch pattern), never the umbrella
    name."""
    with engine.begin() as conn:
        seed_league(conn, 16427, name="1WIN SERIES DOTA 2", tier="T2")
        events = [
            (2024, 3, 16, "1win Series Dota 2 Spring", "T2"),
            (2024, 6, 27, "1win Series Dota 2 Summer", "T2"),
            (2024, 11, 19, "1win Series Dota 2 Fall", "T2"),
            (2024, 12, 21, "1win Series Dota 2 Punch", "T3"),
        ]
        match_id = 100
        for year, month, day, event, tier in events:
            seed_match(
                conn,
                match_id=match_id,
                league_id=16427,
                start_time=datetime(year, month, day, tzinfo=UTC),
            )
            classify_match(conn, match_id, event=event, tier=tier)
            match_id += 1

    rows = _query(
        engine,
        "SELECT effective_event, effective_tier, classification_source "
        "FROM research.matches WHERE league_id = 16427 ORDER BY effective_event",
    )
    assert [(r.effective_event, r.effective_tier) for r in rows] == [
        ("1win Series Dota 2 Fall", "T2"),
        ("1win Series Dota 2 Punch", "T3"),
        ("1win Series Dota 2 Spring", "T2"),
        ("1win Series Dota 2 Summer", "T2"),
    ]
    assert all(r.classification_source == "match-level override" for r in rows)
    assert (
        _count(
            engine, "SELECT count(*) FROM research.t12_matches WHERE league_id = 16427"
        )
        == 3
    )
    assert (
        _count(
            engine, "SELECT count(*) FROM research.pro_matches WHERE league_id = 16427"
        )
        == 4
    )


def test_t12_excludes_pre_2024_matches(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 70, name="Old League", tier="T1")
        seed_match(conn, match_id=91, league_id=70, start_time=UTC2023)

    assert (
        _count(engine, "SELECT count(*) FROM research.matches WHERE match_id = 91") == 1
    )
    assert (
        _count(engine, "SELECT count(*) FROM research.t12_matches WHERE match_id = 91")
        == 0
    )
    row = _query(
        engine,
        "SELECT year, is_t12_main_event FROM research.matches WHERE match_id = 91",
    )[0]
    assert row.year == 2023
    assert row.is_t12_main_event is False
