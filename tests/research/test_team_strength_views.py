"""Semantic tests for the Slice 6 team-strength research views / derived table.

Seeds canonical relational rows and asserts:

* `research.team_strength_state` -- one row per (team_id, match_id) with
  `elo_pre` entering the match, `elo_post` bookkeeping, and a strictly-prior
  descriptive record; equal timestamps never become causal precedence.
* `research.raw_team_elo_latest` -- latest raw post-match Elo state per
  canonical `team_id` (one row per team_id, NO `rank` column, NOT a current
  global Dota ranking), joined to display/identity metadata; canonical team
  ids are never merged by organization; corpus as-of time is exposed.
* The idempotent transactional rebuild matches the pure Python derivation.
* Future deletion at the relational level leaves historical state unchanged.
* The pre-existing research/canonical views keep working alongside the new
  ones.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from research_helpers import seed_league, seed_match

from dota_predictor.data.team_strength import (
    audit_activity_distribution,
    audit_elo_population,
    audit_identity_fragmentation,
    audit_raw_elo_latest,
    audit_team_strength,
    check_freshness,
    rebuild_team_strength_state,
)
from dota_predictor.storage.schema import (
    MATCHES,
    ORGANIZATIONS,
    TEAM_ORGANIZATION_MEMBERSHIPS,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set; skipping DB-touching test",
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)


def _rows(engine, sql: str, **params) -> list[sa.Row]:
    with engine.connect() as conn:
        return conn.execute(sa.text(sql), params).all()


def _one(engine, sql: str, **params) -> sa.Row:
    rows = _rows(engine, sql, **params)
    assert len(rows) == 1
    return rows[0]


def _seed_two_matches(engine, league_id: int = 3001) -> None:
    with engine.begin() as conn:
        seed_league(conn, league_id, name="Test League", tier="T1")
        seed_match(
            conn, match_id=1, league_id=league_id, start_time=T1, radiant_win=True
        )
        seed_match(
            conn, match_id=2, league_id=league_id, start_time=T2, radiant_win=False
        )


# --- derived-table grain and rebuild -----------------------------------------


def test_rebuild_populates_one_row_per_team_match(engine) -> None:
    _seed_two_matches(engine)
    summary = rebuild_team_strength_state(engine)
    assert summary["source_match_count"] == 2
    assert summary["states_written"] == 4

    rows = _rows(engine, "SELECT match_id, team_id FROM research.team_strength_state")
    assert len(rows) == 4
    assert sorted({r.team_id for r in rows}) == sorted([8261500, 9247354])
    assert _rows(
        engine,
        "SELECT count(*) FROM (SELECT team_id, match_id FROM "
        "research.team_strength_state GROUP BY team_id, match_id "
        "HAVING count(*) > 1) d",
    )[0][0] == 0

    build = _one(engine, "SELECT source_match_count, rows_written, elo_initial_rating, elo_k_factor FROM research.team_strength_build")
    assert build.source_match_count == 2
    assert build.rows_written == 4
    assert build.elo_initial_rating == 1500.0
    assert build.elo_k_factor == 32.0


def test_rebuild_is_idempotent(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)
    first = _rows(
        engine,
        "SELECT match_id, team_id, elo_pre, elo_post, prior_match_count "
        "FROM research.team_strength_state ORDER BY team_id, match_id",
    )
    rebuild_team_strength_state(engine)
    second = _rows(
        engine,
        "SELECT match_id, team_id, elo_pre, elo_post, prior_match_count "
        "FROM research.team_strength_state ORDER BY team_id, match_id",
    )
    assert first == second
    assert _rows(engine, "SELECT count(*) FROM research.team_strength_build")[0][0] == 1


def test_elo_pre_and_post_timing_semantics(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)

    # First match: both teams start at 1500 (elo_pre); elo_post reflects the
    # result of match 1 only.
    radiant_m1 = _one(
        engine,
        "SELECT elo_pre, elo_post FROM research.team_strength_state "
        "WHERE match_id = 1 AND side = 'RADIANT'",
    )
    assert radiant_m1.elo_pre == 1500.0
    assert radiant_m1.elo_post == pytest.approx(1516.0)

    # Second match: the current result (a DIRE win) must not leak into the
    # same row's elo_pre; elo_post reflects it.
    dire_m2 = _one(
        engine,
        "SELECT elo_pre, elo_post FROM research.team_strength_state "
        "WHERE match_id = 2 AND side = 'DIRE'",
    )
    assert dire_m2.elo_pre == pytest.approx(1484.0)
    assert dire_m2.elo_post == pytest.approx(1501.4695015289756)

    # The prior record for match 2's dire team counts match 1 (a loss) only.
    dire_prior = _one(
        engine,
        "SELECT prior_match_count, prior_win_count, prior_loss_count, "
        "previous_match_id FROM research.team_strength_state "
        "WHERE match_id = 2 AND side = 'DIRE'",
    )
    assert (dire_prior.prior_match_count, dire_prior.prior_win_count, dire_prior.prior_loss_count) == (
        1,
        0,
        1,
    )
    assert dire_prior.previous_match_id == 1


def test_equal_timestamps_do_not_create_causal_precedence(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3002, name="Test League", tier="T1")
        seed_match(conn, match_id=11, league_id=3002, start_time=T1, radiant_win=True)
        seed_match(conn, match_id=12, league_id=3002, start_time=T1, radiant_win=False)

    rebuild_team_strength_state(engine)
    for match_id in (11, 12):
        row = _one(
            engine,
            "SELECT elo_pre, prior_match_count, previous_match_id, is_first_observed_match "
            "FROM research.team_strength_state WHERE match_id = :m AND team_id = 8261500",
            m=match_id,
        )
        assert row.elo_pre == 1500.0
        assert row.prior_match_count == 0
        assert row.previous_match_id is None
        assert row.is_first_observed_match is True


# --- raw latest-Elo state view -------------------------------------------------


def test_raw_team_elo_latest_reflects_terminal_post_match_rating(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)

    rows = _rows(
        engine,
        "SELECT team_id, rating, observed_match_count, wins, losses, last_match_id "
        "FROM research.raw_team_elo_latest",
    )
    assert len(rows) == 2
    by_team = {r.team_id: r for r in rows}
    assert by_team[8261500].rating == pytest.approx(1498.5304984710244)
    assert by_team[9247354].rating == pytest.approx(1501.4695015289756)
    assert by_team[8261500].observed_match_count == 2
    assert by_team[8261500].wins == 1
    assert by_team[8261500].losses == 1
    assert by_team[8261500].last_match_id == 2


def test_raw_team_elo_latest_has_no_rank_column(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)
    columns = [
        r[0]
        for r in _rows(
            engine,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'research' AND table_name = 'raw_team_elo_latest'",
        )
    ]
    assert "rank" not in columns
    assert "team_id" in columns
    assert "rating" in columns


def test_no_team_rankings_view_exists(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)
    assert _rows(
        engine, "SELECT to_regclass('research.team_rankings')"
    )[0][0] is None
    assert _rows(
        engine, "SELECT count(*) FROM research.raw_team_elo_latest"
    )[0][0] == 2


def test_raw_team_elo_latest_one_row_per_team_id(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3006, name="Test League", tier="T1")
        seed_match(conn, match_id=51, league_id=3006, start_time=T1, radiant_win=True)
        seed_match(conn, match_id=52, league_id=3006, start_time=T2, radiant_win=False)
    rebuild_team_strength_state(engine)
    assert _rows(
        engine,
        "SELECT count(*) FROM (SELECT team_id FROM research.raw_team_elo_latest "
        "GROUP BY team_id HAVING count(*) > 1) d",
    )[0][0] == 0
    assert _rows(engine, "SELECT count(*) FROM research.raw_team_elo_latest")[0][0] == 2


def test_raw_team_elo_latest_expose_corpus_as_of_time_and_activity(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)

    row = _one(engine, "SELECT as_of_at, days_since_last_match_as_of_corpus_end FROM research.raw_team_elo_latest WHERE team_id = 8261500")
    assert row.as_of_at == T2
    assert row.days_since_last_match_as_of_corpus_end == pytest.approx(0.0)


def test_raw_team_elo_latest_expose_display_name(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)

    names = _rows(engine, "SELECT team_id, team_name FROM research.raw_team_elo_latest")
    by_team = {r.team_id: r.team_name for r in names}
    assert by_team[8261500] == "Radiant Test"
    assert by_team[9247354] == "Dire Test"


def test_raw_team_elo_latest_terminal_rating_for_tied_last_group(engine) -> None:
    """When a team's final observed matches share a start_time, the latest
    Elo rating must be the batched terminal rating (pre-group + all group
    deltas), not one individual match's post rating."""
    with engine.begin() as conn:
        seed_league(conn, 3005, name="Test League", tier="T1")
        # Two simultaneous matches for the same team.
        seed_match(conn, match_id=31, league_id=3005, start_time=T1, radiant_win=True)
        seed_match(conn, match_id=32, league_id=3005, start_time=T1, radiant_win=True)

    rebuild_team_strength_state(engine)
    row = _one(
        engine,
        "SELECT rating, observed_match_count FROM research.raw_team_elo_latest "
        "WHERE team_id = 8261500",
    )
    # 1500 + 2 * (32 * 0.5).
    assert row.rating == pytest.approx(1532.0)
    assert row.observed_match_count == 2


def test_raw_team_elo_latest_does_not_merge_teams_by_organization(engine) -> None:
    _seed_two_matches(engine)
    with engine.begin() as conn:
        conn.execute(
            ORGANIZATIONS.insert().values(organization_id=7, name="Same Org")
        )
        conn.execute(
            TEAM_ORGANIZATION_MEMBERSHIPS.insert().values(
                team_id=8261500, organization_id=7, source="test"
            )
        )
        conn.execute(
            TEAM_ORGANIZATION_MEMBERSHIPS.insert().values(
                team_id=9247354, organization_id=7, source="test"
            )
        )
    rebuild_team_strength_state(engine)

    rows = _rows(
        engine,
        "SELECT team_id, organization_id, organization_name FROM research.raw_team_elo_latest ORDER BY team_id",
    )
    assert len(rows) == 2
    assert all(r.organization_id == 7 for r in rows)
    assert all(r.organization_name == "Same Org" for r in rows)
    # Two separate team_id rows, never merged.
    assert {r.team_id for r in rows} == {8261500, 9247354}


# --- future deletion invariance (relational level) ---------------------------


def test_future_deletion_leaves_historical_state_unchanged(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3003, name="Test League", tier="T1")
        seed_match(conn, match_id=21, league_id=3003, start_time=T1, radiant_win=True)
        seed_match(conn, match_id=22, league_id=3003, start_time=T2, radiant_win=False)
        seed_match(conn, match_id=23, league_id=3003, start_time=T3, radiant_win=True)

    rebuild_team_strength_state(engine)
    before = _rows(
        engine,
        "SELECT match_id, team_id, elo_pre, elo_post, prior_match_count "
        "FROM research.team_strength_state WHERE start_time <= :cutoff "
        "ORDER BY team_id, match_id",
        cutoff=T2,
    )

    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM matches WHERE start_time > :cutoff"), {"cutoff": T2})

    rebuild_team_strength_state(engine)
    after = _rows(
        engine,
        "SELECT match_id, team_id, elo_pre, elo_post, prior_match_count "
        "FROM research.team_strength_state ORDER BY team_id, match_id",
    )
    assert before == after


def test_future_roster_changes_do_not_affect_team_strength(engine) -> None:
    """Team strength depends only on match history/results, never on Slice 5
    roster information: deleting every `match_players` row (i.e. every future
    roster observation) must leave the historical team-strength state
    unchanged."""
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)
    before = _rows(
        engine,
        "SELECT match_id, team_id, elo_pre, elo_post, prior_match_count, "
        "prior_win_count, prior_loss_count FROM research.team_strength_state "
        "ORDER BY team_id, match_id",
    )

    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM match_players"))

    rebuild_team_strength_state(engine)
    after = _rows(
        engine,
        "SELECT match_id, team_id, elo_pre, elo_post, prior_match_count, "
        "prior_win_count, prior_loss_count FROM research.team_strength_state "
        "ORDER BY team_id, match_id",
    )
    assert before == after


# --- compatibility -------------------------------------------------------------


def test_existing_research_views_remain_compatible(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)

    assert _rows(engine, "SELECT count(*) FROM research.matches")[0][0] == 2
    assert _rows(engine, "SELECT count(*) FROM research.player_matches")[0][0] == 20
    assert _rows(engine, "SELECT count(*) FROM research.team_match_lineups")[0][0] == 4
    assert _rows(engine, "SELECT count(*) FROM research.team_roster_state")[0][0] == 4
    assert _rows(engine, "SELECT count(*) FROM research.team_strength_state")[0][0] == 4
    assert _rows(engine, "SELECT count(*) FROM research.raw_team_elo_latest")[0][0] == 2


# --- freshness / staleness detection -------------------------------------------


def test_freshness_is_true_after_rebuild(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)
    freshness = check_freshness(engine)
    assert freshness["fresh"] is True
    assert len(freshness["stored"]["source_fingerprint"]) == 64


def test_freshness_detects_old_result_correction_with_unchanged_count_and_extrema(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3010, name="Test League", tier="T1")
        seed_match(conn, match_id=1, league_id=3010, start_time=T1, radiant_win=True)
        seed_match(conn, match_id=2, league_id=3010, start_time=T2, radiant_win=False)
        seed_match(conn, match_id=3, league_id=3010, start_time=T3, radiant_win=True)
    rebuild_team_strength_state(engine)
    assert check_freshness(engine)["fresh"] is True

    # Correct an OLD result (match 1) to a loss. Match count, min start_time
    # and max start_time are all unchanged, so only the fingerprint catches it.
    with engine.begin() as conn:
        conn.execute(
            MATCHES.update()
            .where(MATCHES.c.match_id == 1)
            .values(radiant_win=False)
        )

    freshness = check_freshness(engine)
    assert freshness["fresh"] is False
    assert freshness["current"]["source_match_count"] == freshness["stored"]["source_match_count"]
    assert (
        freshness["current"]["source_max_start_time"]
        == freshness["stored"]["source_max_start_time"]
    )
    assert (
        freshness["current"]["source_fingerprint"]
        != freshness["stored"]["source_fingerprint"]
    )


# --- activity / population / fragmentation diagnostics --------------------------


def test_activity_distribution_reports_buckets_without_filtering(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)
    dist = audit_activity_distribution(engine)
    assert dist["teams_rated"] == 2
    assert dist["corpus_as_of_at"] == T2
    assert dist["days_since_last_match_buckets"]["le_30_days"] == 2
    assert sum(dist["days_since_last_match_buckets"].values()) == 2


def test_elo_population_reports_tier_composition(engine) -> None:
    with engine.begin() as conn:
        seed_league(conn, 3011, name="T3 League", tier="T3")
        seed_match(conn, match_id=41, league_id=3011, start_time=T1)
    pop = audit_elo_population(engine)
    assert pop["total_elo_matches"] == 1
    assert pop["by_category"]["T3"]["matches"] == 1
    assert pop["by_category"]["T3"]["share_of_elo_updates"] == pytest.approx(1.0)
    assert pop["teams_predominantly_t3"]["count"] == 2


def test_identity_fragmentation_finds_same_organization_pair(engine) -> None:
    _seed_two_matches(engine)
    with engine.begin() as conn:
        conn.execute(
            ORGANIZATIONS.insert().values(organization_id=7, name="Same Org")
        )
        conn.execute(
            TEAM_ORGANIZATION_MEMBERSHIPS.insert().values(
                team_id=8261500, organization_id=7, source="test"
            )
        )
        conn.execute(
            TEAM_ORGANIZATION_MEMBERSHIPS.insert().values(
                team_id=9247354, organization_id=7, source="test"
            )
        )
    frag = audit_identity_fragmentation(engine)
    assert frag["candidate_pair_count"] >= 1
    matching = [
        c
        for c in frag["candidate_pairs"]
        if set(c["team_ids"]) == {8261500, 9247354}
    ]
    assert any("same curated organization" in " ".join(c["signals"]) for c in matching)


# --- default audit contains no leaderboard -------------------------------------


def test_default_audit_contains_no_leaderboard(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)
    audit = audit_team_strength(engine)
    dumped = json.dumps(audit, default=str)
    assert "top_20" not in dumped
    assert "raw_team_id_elo_top_20" not in dumped
    assert '"rank"' not in dumped
    # Latest Elo is reported only as a distribution.
    assert "rating_min" in audit["latest_ratings"]
    assert "rating_median" in audit["latest_ratings"]
    assert "rating_max" in audit["latest_ratings"]


def test_opt_in_raw_elo_has_no_rank(engine) -> None:
    _seed_two_matches(engine)
    rebuild_team_strength_state(engine)
    out = audit_raw_elo_latest(engine)
    assert len(out["raw_latest_team_id_elo"]) == 2
    assert all("rank" not in row for row in out["raw_latest_team_id_elo"])
    assert "DEBUGGING/DIAGNOSTIC" in out["note"]
