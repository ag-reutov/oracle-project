"""Tests for the descriptive hero-meta layer (`features.hero_meta`).

Small deterministic fixtures with hand-calculated expected values.
Does not go through PRE_DRAFT snapshot SQL, Elo, or training assembly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from hero_meta_helpers import (
    draft_and_player_rows,
    hero_meta_frame,
    match_row,
)

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.hero_meta import (
    HERO_META_COLUMNS,
    HERO_META_METRIC_COLUMNS,
    RECENT_WINDOW_DAYS,
    hero_meta_sql,
    rank_hero_meta,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS

T0 = datetime(2024, 1, 1, tzinfo=UTC)
T1 = datetime(2024, 1, 2, tzinfo=UTC)
T2 = datetime(2024, 1, 3, tzinfo=UTC)

VERSION_A = 10
VERSION_B = 11

# Match ids are deliberately unordered vs start_time.
M1, M2, M3 = 4001, 1002, 3003

RADIANT_DEFAULT = (1, 2, 3, 4, 5)
DIRE_DEFAULT = (6, 7, 8, 9, 10)

CATALOG_HEROES = [
    {"id": hero_id, "displayName": f"Hero {hero_id}"}
    for hero_id in (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        20, 21, 22, 50, 99,
    )
]


def _row(frame: pd.DataFrame, match_id: int, hero_id: int) -> pd.Series:
    subset = frame[(frame["match_id"] == match_id) & (frame["hero_id"] == hero_id)]
    assert len(subset) == 1, (
        f"expected one row for ({match_id}, {hero_id}), got {len(subset)}"
    )
    return subset.iloc[0]


def _assemble(
    tmp_path: Path,
    specs: list[dict],
    *,
    heroes: list[dict] | None = CATALOG_HEROES,
    match_id: int | None = None,
) -> pd.DataFrame:
    matches: list[dict] = []
    players: list[dict] = []
    drafts: list[dict] = []
    for spec in specs:
        matches.append(
            match_row(
                spec["match_id"],
                start_time=spec["start_time"],
                radiant_win=spec["radiant_win"],
                game_version_id=spec["game_version_id"],
            )
        )
        draft_rows, player_rows = draft_and_player_rows(
            spec["match_id"],
            radiant_picks=spec.get("radiant_picks", RADIANT_DEFAULT),
            dire_picks=spec.get("dire_picks", DIRE_DEFAULT),
            successful_bans=spec.get("successful_bans", ()),
            unsuccessful_bans=spec.get("unsuccessful_bans", ()),
        )
        drafts.extend(draft_rows)
        players.extend(player_rows)
    return hero_meta_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        match_id=match_id,
    )


# --- SQL / contract guards ------------------------------------------------


def test_sql_uses_strict_less_than_never_less_or_equal() -> None:
    sql = hero_meta_sql(catalog_registered=True)
    assert "start_time <=" not in sql
    assert "EXCLUDE GROUP" in sql
    assert sql.count("EXCLUDE GROUP") == 2


def test_sql_encodes_same_version_and_90_day_window() -> None:
    sql = hero_meta_sql(catalog_registered=True)
    assert "PARTITION BY game_version_id, hero_id" in sql
    assert f"INTERVAL {RECENT_WINDOW_DAYS} DAY" in sql
    assert "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in sql


def test_sql_uses_successful_draft_action_rule() -> None:
    sql = hero_meta_sql(catalog_registered=True)
    assert "was_successful IS DISTINCT FROM FALSE" in sql
    assert "GROUP BY match_id, hero_id" in sql


def test_hero_meta_is_not_part_of_training_or_pre_draft_snapshot() -> None:
    assert set(HERO_META_METRIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(HERO_META_METRIC_COLUMNS).isdisjoint(SNAPSHOT_COLUMNS)
    assert set(HERO_META_METRIC_COLUMNS).isdisjoint(ALL_FEATURE_COLUMNS)
    assert "hero_meta" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "same_version_contest_rate" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged_by_this_layer() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 3
    assert REFERENCE_SCHEMA_VERSION == 1


# --- rate correctness (hand-calculated) -----------------------------------


def _rate_fixture_specs() -> list[dict]:
    """Two historical maps plus one evaluation map, all version A.

    M1 (T0, Radiant win):
      unsuccessful ban of 50; successful bans 20, 21;
      Radiant 1-5; Dire 6-10.
    M2 (T1, Dire win):
      successful bans 20, 22;
      Radiant 1,11-14; Dire 6,15-18.
    M3 (T2) is the evaluation point -- its own draft must not count.
    """
    return [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "successful_bans": (("RADIANT", 20), ("DIRE", 21)),
            "unsuccessful_bans": (("RADIANT", 50),),
        },
        {
            "match_id": M2,
            "start_time": T1,
            "radiant_win": False,
            "game_version_id": VERSION_A,
            "radiant_picks": (1, 11, 12, 13, 14),
            "dire_picks": (6, 15, 16, 17, 18),
            "successful_bans": (("RADIANT", 20), ("DIRE", 22)),
        },
        {
            "match_id": M3,
            "start_time": T2,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_picks": (1, 2, 3, 4, 5),
            "dire_picks": (6, 7, 8, 9, 10),
            "successful_bans": (("RADIANT", 20),),
        },
    ]


def test_pick_ban_contest_and_win_rates(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs(), match_id=M3)
    hero1 = _row(frame, M3, 1)
    assert hero1["same_version_prior_matches"] == 2
    # Hero 1 picked in M1 (Radiant win) and M2 (Radiant loss).
    assert hero1["same_version_prior_picks"] == 2
    assert hero1["same_version_prior_bans"] == 0
    assert hero1["same_version_prior_contests"] == 2
    assert hero1["same_version_pick_rate"] == pytest.approx(1.0)
    assert hero1["same_version_ban_rate"] == pytest.approx(0.0)
    assert hero1["same_version_contest_rate"] == pytest.approx(1.0)
    assert hero1["same_version_prior_wins"] == 1
    assert hero1["same_version_prior_losses"] == 1
    assert hero1["same_version_win_rate"] == pytest.approx(0.5)


def test_radiant_hero_win_and_dire_hero_win(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs(), match_id=M3)
    # Hero 5: Radiant pick in M1 only (Radiant won). Not picked in M2.
    hero5 = _row(frame, M3, 5)
    assert hero5["same_version_prior_picks"] == 1
    assert hero5["same_version_prior_wins"] == 1
    assert hero5["same_version_prior_losses"] == 0
    assert hero5["same_version_win_rate"] == pytest.approx(1.0)

    # Hero 6: Dire pick in M1 (Dire lost) and M2 (Dire won).
    hero6 = _row(frame, M3, 6)
    assert hero6["same_version_prior_picks"] == 2
    assert hero6["same_version_prior_wins"] == 1
    assert hero6["same_version_prior_losses"] == 1
    assert hero6["same_version_win_rate"] == pytest.approx(0.5)

    # Hero 10: Dire pick in M1 only (Dire lost).
    hero10 = _row(frame, M3, 10)
    assert hero10["same_version_prior_picks"] == 1
    assert hero10["same_version_prior_wins"] == 0
    assert hero10["same_version_prior_losses"] == 1
    assert hero10["same_version_win_rate"] == pytest.approx(0.0)


def test_unsuccessful_ban_excluded(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs(), match_id=M3)
    hero50 = _row(frame, M3, 50)
    assert hero50["same_version_prior_matches"] == 2
    assert hero50["same_version_prior_picks"] == 0
    assert hero50["same_version_prior_bans"] == 0
    assert hero50["same_version_prior_contests"] == 0
    assert hero50["same_version_ban_rate"] == pytest.approx(0.0)
    assert hero50["same_version_contest_rate"] == pytest.approx(0.0)
    assert pd.isna(hero50["same_version_win_rate"])


def test_hero_not_picked_has_no_win_loss_contribution(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs(), match_id=M3)
    # Hero 21 was successfully banned in M1, never picked.
    hero21 = _row(frame, M3, 21)
    assert hero21["same_version_prior_bans"] == 1
    assert hero21["same_version_prior_picks"] == 0
    assert hero21["same_version_prior_contests"] == 1
    assert hero21["same_version_prior_wins"] == 0
    assert hero21["same_version_prior_losses"] == 0
    assert pd.isna(hero21["same_version_win_rate"])
    assert hero21["same_version_ban_rate"] == pytest.approx(0.5)
    assert hero21["same_version_contest_rate"] == pytest.approx(0.5)


def test_never_seen_catalog_hero_has_zero_counts_and_zero_rates(
    tmp_path: Path,
) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs(), match_id=M3)
    hero99 = _row(frame, M3, 99)
    assert hero99["same_version_prior_matches"] == 2
    assert hero99["same_version_prior_picks"] == 0
    assert hero99["same_version_prior_bans"] == 0
    assert hero99["same_version_prior_contests"] == 0
    assert hero99["same_version_prior_wins"] == 0
    assert hero99["same_version_prior_losses"] == 0
    assert hero99["same_version_pick_rate"] == pytest.approx(0.0)
    assert hero99["hero_name"] == "Hero 99"
    assert pd.isna(hero99["same_version_win_rate"])


def test_zero_history_null_rates_on_first_match(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs(), match_id=M1)
    hero1 = _row(frame, M1, 1)
    assert hero1["same_version_prior_matches"] == 0
    assert hero1["same_version_prior_picks"] == 0
    assert hero1["same_version_prior_bans"] == 0
    assert hero1["same_version_prior_contests"] == 0
    assert hero1["same_version_prior_wins"] == 0
    assert hero1["same_version_prior_losses"] == 0
    assert pd.isna(hero1["same_version_pick_rate"])
    assert pd.isna(hero1["same_version_ban_rate"])
    assert pd.isna(hero1["same_version_contest_rate"])
    assert pd.isna(hero1["same_version_win_rate"])
    assert pd.isna(hero1["recent_90d_pick_rate"])
    assert pd.isna(hero1["recent_90d_win_rate"])


def test_current_match_excluded_from_its_own_counts(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs(), match_id=M2)
    hero1 = _row(frame, M2, 1)
    assert hero1["same_version_prior_matches"] == 1
    assert hero1["same_version_prior_picks"] == 1
    assert hero1["same_version_prior_wins"] == 1


def test_future_matches_excluded(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs(), match_id=M2)
    hero11 = _row(frame, M2, 11)
    assert hero11["same_version_prior_picks"] == 0
    assert hero11["same_version_pick_rate"] == pytest.approx(0.0)


def test_wins_plus_losses_equal_picks(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs())
    assert (
        frame["same_version_prior_wins"] + frame["same_version_prior_losses"]
        == frame["same_version_prior_picks"]
    ).all()
    assert (
        frame["recent_90d_prior_wins"] + frame["recent_90d_prior_losses"]
        == frame["recent_90d_prior_picks"]
    ).all()


def test_output_columns_and_catalog_grain(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs())
    assert list(frame.columns) == list(HERO_META_COLUMNS)
    assert set(frame["match_id"].unique()) == {M1, M2, M3}
    assert len(frame) == 3 * len(CATALOG_HEROES)
    assert set(frame["hero_id"].unique()) == {h["id"] for h in CATALOG_HEROES}


# --- same-timestamp leakage -----------------------------------------------


def test_identical_timestamps_are_mutually_blind(tmp_path: Path) -> None:
    """Matches sharing start_time must not contribute to each other.

    M0 at T0 picks hero 1. MA and MB share T1: both pick hero 2. Neither
    may see the other's pick of 2; both may see M0's pick of 1.
    """
    same_time = T1
    specs = [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_picks": (1, 3, 4, 5, 11),
            "dire_picks": (6, 7, 8, 9, 10),
        },
        {
            "match_id": M2,
            "start_time": same_time,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_picks": (2, 3, 4, 5, 11),
            "dire_picks": (6, 7, 8, 9, 10),
        },
        {
            "match_id": M3,
            "start_time": same_time,
            "radiant_win": False,
            "game_version_id": VERSION_A,
            "radiant_picks": (2, 12, 13, 14, 15),
            "dire_picks": (16, 17, 18, 7, 8),
        },
    ]
    frame = _assemble(tmp_path, specs)

    for match_id in (M2, M3):
        hero1 = _row(frame, match_id, 1)
        assert hero1["same_version_prior_matches"] == 1
        assert hero1["same_version_prior_picks"] == 1
        hero2 = _row(frame, match_id, 2)
        assert hero2["same_version_prior_picks"] == 0
        assert hero2["recent_90d_prior_picks"] == 0
        assert hero2["same_version_pick_rate"] == pytest.approx(0.0)

    hero2_at_m1 = _row(frame, M1, 2)
    assert hero2_at_m1["same_version_prior_matches"] == 0
    assert hero2_at_m1["same_version_prior_picks"] == 0
    assert pd.isna(hero2_at_m1["same_version_pick_rate"])


# --- patch transition -----------------------------------------------------


def test_first_match_of_new_version_has_zero_same_version_history(
    tmp_path: Path,
) -> None:
    """Hero 1 is contested in every version-A match. First version-B match
    has zero same-version history, but recent-90d still sees version A.
    """
    specs = [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_picks": (1, 2, 3, 4, 5),
            "dire_picks": (6, 7, 8, 9, 10),
            "successful_bans": (("RADIANT", 20),),
        },
        {
            "match_id": M2,
            "start_time": T1,
            "radiant_win": False,
            "game_version_id": VERSION_A,
            "radiant_picks": (11, 12, 13, 14, 15),
            "dire_picks": (6, 7, 8, 9, 10),
            "successful_bans": (("DIRE", 1),),
        },
        {
            "match_id": M3,
            "start_time": T2,
            "radiant_win": True,
            "game_version_id": VERSION_B,
            "radiant_picks": (1, 2, 3, 4, 5),
            "dire_picks": (6, 7, 8, 9, 10),
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M3)
    hero1 = _row(frame, M3, 1)

    assert hero1["same_version_prior_matches"] == 0
    assert hero1["same_version_prior_picks"] == 0
    assert hero1["same_version_prior_bans"] == 0
    assert hero1["same_version_prior_contests"] == 0
    assert pd.isna(hero1["same_version_pick_rate"])
    assert pd.isna(hero1["same_version_contest_rate"])
    assert pd.isna(hero1["same_version_win_rate"])

    assert hero1["recent_90d_prior_matches"] == 2
    assert hero1["recent_90d_prior_picks"] == 1
    assert hero1["recent_90d_prior_bans"] == 1
    assert hero1["recent_90d_prior_contests"] == 2
    assert hero1["recent_90d_contest_rate"] == pytest.approx(1.0)
    assert hero1["recent_90d_win_rate"] == pytest.approx(1.0)


def test_same_version_filtering_ignores_other_versions(tmp_path: Path) -> None:
    specs = [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_picks": (1, 2, 3, 4, 5),
            "dire_picks": (6, 7, 8, 9, 10),
        },
        {
            "match_id": M2,
            "start_time": T1,
            "radiant_win": True,
            "game_version_id": VERSION_B,
            "radiant_picks": (1, 2, 3, 4, 5),
            "dire_picks": (6, 7, 8, 9, 10),
        },
        {
            "match_id": M3,
            "start_time": T2,
            "radiant_win": True,
            "game_version_id": VERSION_B,
            "radiant_picks": (11, 12, 13, 14, 15),
            "dire_picks": (6, 7, 8, 9, 10),
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M3)
    hero1 = _row(frame, M3, 1)
    assert hero1["same_version_prior_matches"] == 1
    assert hero1["same_version_prior_picks"] == 1
    assert hero1["recent_90d_prior_matches"] == 2
    assert hero1["recent_90d_prior_picks"] == 2


# --- 90-day lower boundary ------------------------------------------------


def test_90_day_window_includes_exact_lower_bound(tmp_path: Path) -> None:
    t_old = T0
    t_exact = T0 + timedelta(days=RECENT_WINDOW_DAYS)
    specs = [
        {
            "match_id": M1,
            "start_time": t_old,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_picks": (1, 2, 3, 4, 5),
            "dire_picks": (6, 7, 8, 9, 10),
        },
        {
            "match_id": M2,
            "start_time": t_exact,
            "radiant_win": True,
            "game_version_id": VERSION_B,
            "radiant_picks": (11, 12, 13, 14, 15),
            "dire_picks": (6, 7, 8, 9, 10),
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    hero1 = _row(frame, M2, 1)
    assert hero1["recent_90d_prior_matches"] == 1
    assert hero1["recent_90d_prior_picks"] == 1
    assert hero1["same_version_prior_matches"] == 0
    assert pd.isna(hero1["same_version_pick_rate"])


def test_90_day_window_excludes_older_than_lower_bound(tmp_path: Path) -> None:
    t_old = T0
    t_after = T0 + timedelta(days=RECENT_WINDOW_DAYS, microseconds=1)
    specs = [
        {
            "match_id": M1,
            "start_time": t_old,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_picks": (1, 2, 3, 4, 5),
            "dire_picks": (6, 7, 8, 9, 10),
        },
        {
            "match_id": M2,
            "start_time": t_after,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_picks": (11, 12, 13, 14, 15),
            "dire_picks": (6, 7, 8, 9, 10),
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    hero1 = _row(frame, M2, 1)
    assert hero1["recent_90d_prior_matches"] == 0
    assert hero1["recent_90d_prior_picks"] == 0
    assert pd.isna(hero1["recent_90d_pick_rate"])
    assert hero1["same_version_prior_matches"] == 1
    assert hero1["same_version_prior_picks"] == 1


# --- universe without catalog; ranking ------------------------------------


def test_fact_derived_universe_when_catalog_is_absent(tmp_path: Path) -> None:
    specs = [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
        {
            "match_id": M2,
            "start_time": T1,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
    ]
    frame = _assemble(tmp_path, specs, heroes=None, match_id=M2)
    assert 99 not in set(frame["hero_id"])
    assert set(RADIANT_DEFAULT + DIRE_DEFAULT).issubset(set(frame["hero_id"]))
    assert frame["hero_name"].isna().all()
    hero1 = _row(frame, M2, 1)
    assert hero1["same_version_prior_picks"] == 1


def test_rank_hero_meta_orders_by_contest_rate(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _rate_fixture_specs(), match_id=M3)
    ranked = rank_hero_meta(frame)
    contest = ranked["same_version_contest_rate"]
    finite = contest.dropna()
    assert finite.tolist() == sorted(finite.tolist(), reverse=True)
    pos1 = ranked.index[ranked["hero_id"] == 1][0]
    pos99 = ranked.index[ranked["hero_id"] == 99][0]
    assert pos1 < pos99
