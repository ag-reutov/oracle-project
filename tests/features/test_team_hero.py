"""Tests for the descriptive team × hero layer (`features.team_hero`).

Small deterministic fixtures with hand-calculated expected values.
Does not go through PRE_DRAFT snapshot SQL, Elo, or training assembly.
History is keyed by `team_id`; roster changes do not reset it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from team_hero_helpers import (
    draft_and_player_rows,
    match_row,
    team_hero_frame,
)

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.features.team_hero import (
    RECENT_WINDOW_DAYS,
    TEAM_HERO_COLUMNS,
    TEAM_HERO_METRIC_COLUMNS,
    team_hero_sql,
)
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS

T0 = datetime(2024, 1, 1, tzinfo=UTC)
T1 = datetime(2024, 1, 2, tzinfo=UTC)
T2 = datetime(2024, 1, 3, tzinfo=UTC)

VERSION_A = 10
VERSION_B = 11

# Match ids are deliberately unordered vs start_time.
M1, M2, M3 = 4001, 1002, 3003

P1, P2, P3, P4, P5 = 1, 2, 3, 4, 5
P6, P7, P8, P9, P10 = 6, 7, 8, 9, 10
P11, P12, P13, P14, P15 = 11, 12, 13, 14, 15
P16, P17, P18, P19, P20 = 16, 17, 18, 19, 20

RADIANT_PLAYERS = (P1, P2, P3, P4, P5)
DIRE_PLAYERS = (P6, P7, P8, P9, P10)
RADIANT_HEROES = (1, 2, 3, 4, 5)
DIRE_HEROES = (6, 7, 8, 9, 10)

TEAM_A, TEAM_B, TEAM_C, TEAM_D = 100, 200, 300, 400

RATE_AND_SHARE_COLUMNS = (
    "team_prior_win_rate_with_hero",
    "same_version_team_win_rate_with_hero",
    "recent_90d_team_win_rate_with_hero",
    "team_hero_share",
    "recent_90d_team_hero_share",
)

CATALOG_HEROES = [
    {"id": hero_id, "displayName": f"Hero {hero_id}"}
    for hero_id in range(1, 23)
]


def _row(frame: pd.DataFrame, match_id: int, team_id: int, hero_id: int) -> pd.Series:
    subset = frame[
        (frame["match_id"] == match_id)
        & (frame["team_id"] == team_id)
        & (frame["hero_id"] == hero_id)
    ]
    assert len(subset) == 1, (
        f"expected one row for ({match_id}, {team_id}, {hero_id}), "
        f"got {len(subset)}"
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
                radiant_team_id=spec.get("radiant_team_id", TEAM_A),
                dire_team_id=spec.get("dire_team_id", TEAM_B),
            )
        )
        draft_rows, player_rows = draft_and_player_rows(
            spec["match_id"],
            radiant_player_ids=spec.get("radiant_players", RADIANT_PLAYERS),
            dire_player_ids=spec.get("dire_players", DIRE_PLAYERS),
            radiant_hero_ids=spec.get("radiant_heroes", RADIANT_HEROES),
            dire_hero_ids=spec.get("dire_heroes", DIRE_HEROES),
        )
        drafts.extend(draft_rows)
        players.extend(player_rows)
    return team_hero_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        match_id=match_id,
    )


def _three_match_specs() -> list[dict]:
    """Two historical maps plus one evaluation map, all version A.

    M1 (T0, Radiant win): default heroes. Team A Radiant hero 1 wins;
    Team B Dire hero 6 loses.
    M2 (T1, Dire win): Team A still Radiant; hero 2 is replaced by 11.
    Team B Dire hero 6 wins. M3 (T2) is the evaluation point -- its own
    draft/result must not count.
    """
    return [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
        {
            "match_id": M2,
            "start_time": T1,
            "radiant_win": False,
            "game_version_id": VERSION_A,
            "radiant_heroes": (1, 11, 3, 4, 5),
            "dire_heroes": (6, 7, 8, 9, 10),
        },
        {
            "match_id": M3,
            "start_time": T2,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_heroes": (1, 11, 3, 4, 5),
            "dire_heroes": (6, 7, 8, 9, 10),
        },
    ]


# --- SQL / contract guards ------------------------------------------------


def test_sql_uses_window_range_exclude_group_not_self_join() -> None:
    sql = team_hero_sql(catalog_registered=True)
    assert "start_time <=" not in sql
    assert "EXCLUDE GROUP" in sql
    assert sql.count("EXCLUDE GROUP") == 5
    assert "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in sql
    assert f"INTERVAL {RECENT_WINDOW_DAYS} DAY" in sql
    assert "h.start_time < c.start_time" not in sql


def test_sql_partitions_by_team_and_team_hero_not_player_or_slot() -> None:
    sql = team_hero_sql(catalog_registered=True)
    assert "PARTITION BY team_id" in sql
    assert "PARTITION BY team_id, hero_id" in sql
    assert "PARTITION BY team_id, hero_id, game_version_id" in sql
    assert "PARTITION BY player_id" not in sql
    for clause in sql.split("PARTITION BY")[1:]:
        header = clause.split("ORDER BY")[0]
        assert "slot_in_side" not in header
        assert "player_id" not in header


def test_sql_never_orders_history_by_match_id() -> None:
    sql = team_hero_sql(catalog_registered=True)
    assert "ORDER BY start_time" in sql
    assert "ORDER BY match_id" not in sql
    assert "ORDER BY start_time, match_id" not in sql


def test_sql_does_not_encode_positions_lanes_roles_or_elo() -> None:
    sql = team_hero_sql(catalog_registered=True).lower()
    for forbidden in (
        "position",
        "lane",
        "role",
        "synergy",
        "counter",
        "elo",
        "meta_fit",
        "player_id",
        "slot_in_side",
    ):
        assert forbidden not in sql


def test_team_hero_is_not_part_of_training_or_pre_draft_snapshot() -> None:
    assert set(TEAM_HERO_METRIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(TEAM_HERO_METRIC_COLUMNS).isdisjoint(SNAPSHOT_COLUMNS)
    assert set(TEAM_HERO_METRIC_COLUMNS).isdisjoint(ALL_FEATURE_COLUMNS)
    assert "team_hero" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "team_prior_games_with_hero" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "team_hero_share" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged_by_this_layer() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 5
    assert REFERENCE_SCHEMA_VERSION == 1


# --- current / future exclusion -------------------------------------------


def test_current_match_excluded_from_its_own_counts(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    team_a = _row(frame, M2, TEAM_A, 1)
    assert team_a["team_prior_games_with_hero"] == 1
    assert team_a["team_prior_wins_with_hero"] == 1
    assert team_a["team_prior_losses_with_hero"] == 0
    assert team_a["team_prior_games"] == 1


def test_future_matches_excluded(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    team_a = _row(frame, M2, TEAM_A, 11)
    # Team A first drafted hero 11 in M2; M3 also has hero 11 and must
    # not count.
    assert team_a["team_prior_games_with_hero"] == 0
    assert team_a["team_prior_games"] == 1
    assert pd.isna(team_a["team_prior_win_rate_with_hero"])


# --- identical-timestamp leakage ------------------------------------------


def test_identical_timestamps_are_mutually_blind(tmp_path: Path) -> None:
    """Matches sharing start_time must not contribute to each other.

    Team A plays hero 1 at T0. At T1, Team A appears in both M2 and M3
    on hero 1 with opposite outcomes. Each T1 row may see T0, never the
    peer.
    """
    same_time = T1
    specs = [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
        {
            "match_id": M2,
            "start_time": same_time,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "dire_team_id": TEAM_C,
            "dire_players": (P11, P12, P13, P14, P15),
            "dire_heroes": (16, 17, 18, 19, 20),
        },
        {
            "match_id": M3,
            "start_time": same_time,
            "radiant_win": False,
            "game_version_id": VERSION_A,
            "dire_team_id": TEAM_D,
            "dire_players": (P16, P17, P18, P19, P20),
            "dire_heroes": (16, 17, 18, 19, 20),
        },
    ]
    frame = _assemble(tmp_path, specs)

    for match_id in (M2, M3):
        team_a = _row(frame, match_id, TEAM_A, 1)
        assert team_a["team_prior_games_with_hero"] == 1
        assert team_a["team_prior_wins_with_hero"] == 1
        assert team_a["team_prior_losses_with_hero"] == 0
        assert team_a["team_prior_games"] == 1
        assert team_a["recent_90d_team_games_with_hero"] == 1

    team_a_at_m1 = _row(frame, M1, TEAM_A, 1)
    assert team_a_at_m1["team_prior_games_with_hero"] == 0
    assert team_a_at_m1["team_prior_games"] == 0
    assert pd.isna(team_a_at_m1["team_prior_win_rate_with_hero"])


# --- patch transition -----------------------------------------------------


def test_first_match_of_new_version_has_zero_same_version_history(
    tmp_path: Path,
) -> None:
    """Team A plays hero 1 in every version-A match. First version-B
    match has zero same-version hero history, but all-time and 90d still
    see A.
    """
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
            "radiant_win": False,
            "game_version_id": VERSION_A,
        },
        {
            "match_id": M3,
            "start_time": T2,
            "radiant_win": True,
            "game_version_id": VERSION_B,
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M3)
    team_a = _row(frame, M3, TEAM_A, 1)

    assert team_a["same_version_team_games_with_hero"] == 0
    assert team_a["same_version_team_wins_with_hero"] == 0
    assert pd.isna(team_a["same_version_team_win_rate_with_hero"])

    assert team_a["team_prior_games_with_hero"] == 2
    assert team_a["team_prior_wins_with_hero"] == 1
    assert team_a["team_prior_losses_with_hero"] == 1
    assert team_a["team_prior_win_rate_with_hero"] == pytest.approx(0.5)
    assert team_a["recent_90d_team_games_with_hero"] == 2
    assert team_a["recent_90d_team_wins_with_hero"] == 1
    assert team_a["recent_90d_team_win_rate_with_hero"] == pytest.approx(0.5)
    assert team_a["recent_90d_team_games"] == 2
    assert team_a["recent_90d_team_hero_share"] == pytest.approx(1.0)


def test_same_version_filtering_ignores_other_versions(tmp_path: Path) -> None:
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
            "game_version_id": VERSION_B,
        },
        {
            "match_id": M3,
            "start_time": T2,
            "radiant_win": True,
            "game_version_id": VERSION_B,
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M3)
    team_a = _row(frame, M3, TEAM_A, 1)
    assert team_a["same_version_team_games_with_hero"] == 1
    assert team_a["same_version_team_wins_with_hero"] == 1
    assert team_a["team_prior_games_with_hero"] == 2
    assert team_a["recent_90d_team_games_with_hero"] == 2


def test_90d_history_survives_patch_transition(tmp_path: Path) -> None:
    """90-day team×hero counts keep version-A games after the patch."""
    specs = [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
        {
            "match_id": M2,
            "start_time": T2,
            "radiant_win": False,
            "game_version_id": VERSION_B,
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    team_a = _row(frame, M2, TEAM_A, 1)
    assert team_a["same_version_team_games_with_hero"] == 0
    assert pd.isna(team_a["same_version_team_win_rate_with_hero"])
    assert team_a["recent_90d_team_games_with_hero"] == 1
    assert team_a["recent_90d_team_wins_with_hero"] == 1
    assert team_a["recent_90d_team_games"] == 1
    assert team_a["recent_90d_team_hero_share"] == pytest.approx(1.0)
    assert team_a["team_prior_games_with_hero"] == 1


# --- Radiant and Dire wins ------------------------------------------------


def test_radiant_wins_count_for_radiant_team(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M3)
    team_a = _row(frame, M3, TEAM_A, 1)
    # Radiant hero 1: M1 win, M2 loss.
    assert team_a["side"] == "RADIANT"
    assert team_a["team_prior_games_with_hero"] == 2
    assert team_a["team_prior_wins_with_hero"] == 1
    assert team_a["team_prior_losses_with_hero"] == 1
    assert team_a["team_prior_win_rate_with_hero"] == pytest.approx(0.5)


def test_dire_wins_count_for_dire_team(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M3)
    team_b = _row(frame, M3, TEAM_B, 6)
    # Dire hero 6: M1 loss (Radiant won), M2 win (Dire won).
    assert team_b["side"] == "DIRE"
    assert team_b["team_prior_games_with_hero"] == 2
    assert team_b["team_prior_wins_with_hero"] == 1
    assert team_b["team_prior_losses_with_hero"] == 1
    assert team_b["team_prior_win_rate_with_hero"] == pytest.approx(0.5)

    team_b_hero_10 = _row(frame, M3, TEAM_B, 10)
    assert team_b_hero_10["team_prior_wins_with_hero"] == 1
    assert team_b_hero_10["team_prior_losses_with_hero"] == 1


def test_wins_plus_losses_equal_games_with_hero(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs())
    assert (
        frame["team_prior_wins_with_hero"] + frame["team_prior_losses_with_hero"]
        == frame["team_prior_games_with_hero"]
    ).all()


# --- zero-history NULL semantics ------------------------------------------


def test_zero_history_null_rates_and_shares_on_first_match(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M1)
    team_a = _row(frame, M1, TEAM_A, 1)
    assert team_a["team_prior_games_with_hero"] == 0
    assert team_a["team_prior_wins_with_hero"] == 0
    assert team_a["team_prior_losses_with_hero"] == 0
    assert team_a["team_prior_games"] == 0
    assert team_a["same_version_team_games_with_hero"] == 0
    assert team_a["recent_90d_team_games_with_hero"] == 0
    assert team_a["recent_90d_team_games"] == 0
    assert pd.isna(team_a["team_prior_win_rate_with_hero"])
    assert pd.isna(team_a["same_version_team_win_rate_with_hero"])
    assert pd.isna(team_a["recent_90d_team_win_rate_with_hero"])
    assert pd.isna(team_a["team_hero_share"])
    assert pd.isna(team_a["recent_90d_team_hero_share"])
    assert pd.isna(team_a["days_since_team_played_hero"])


# --- team playing different heroes ----------------------------------------


def test_team_playing_different_heroes_does_not_count_other_hero_games(
    tmp_path: Path,
) -> None:
    """Team A plays hero 1, then hero 11. Hero-11 metrics ignore the
    hero-1 game; team-level games still count it.
    """
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
            "radiant_heroes": (11, 2, 3, 4, 5),
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    team_a = _row(frame, M2, TEAM_A, 11)
    assert team_a["team_prior_games_with_hero"] == 0
    assert team_a["team_prior_wins_with_hero"] == 0
    assert team_a["team_prior_games"] == 1
    assert team_a["team_hero_share"] == pytest.approx(0.0)
    assert pd.isna(team_a["team_prior_win_rate_with_hero"])
    assert pd.isna(team_a["days_since_team_played_hero"])


# --- hero-share correctness -----------------------------------------------


def test_hero_share_is_games_with_hero_over_team_games(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M3)
    team_a_h1 = _row(frame, M3, TEAM_A, 1)
    # Two prior games, both on hero 1.
    assert team_a_h1["team_prior_games_with_hero"] == 2
    assert team_a_h1["team_prior_games"] == 2
    assert team_a_h1["team_hero_share"] == pytest.approx(1.0)
    assert team_a_h1["recent_90d_team_games_with_hero"] == 2
    assert team_a_h1["recent_90d_team_games"] == 2
    assert team_a_h1["recent_90d_team_hero_share"] == pytest.approx(1.0)

    team_a_h11 = _row(frame, M3, TEAM_A, 11)
    # M1 did not draft hero 11; M2 did. Current is hero 11.
    assert team_a_h11["team_prior_games_with_hero"] == 1
    assert team_a_h11["team_prior_games"] == 2
    assert team_a_h11["team_hero_share"] == pytest.approx(0.5)
    assert team_a_h11["recent_90d_team_hero_share"] == pytest.approx(0.5)


# --- 90-day expiry --------------------------------------------------------


def test_90_day_window_includes_exact_lower_bound(tmp_path: Path) -> None:
    t_old = T0
    t_exact = T0 + timedelta(days=RECENT_WINDOW_DAYS)
    specs = [
        {
            "match_id": M1,
            "start_time": t_old,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
        {
            "match_id": M2,
            "start_time": t_exact,
            "radiant_win": True,
            "game_version_id": VERSION_B,
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    team_a = _row(frame, M2, TEAM_A, 1)
    assert team_a["recent_90d_team_games"] == 1
    assert team_a["recent_90d_team_games_with_hero"] == 1
    assert team_a["recent_90d_team_hero_share"] == pytest.approx(1.0)
    assert team_a["same_version_team_games_with_hero"] == 0
    assert pd.isna(team_a["same_version_team_win_rate_with_hero"])
    assert team_a["team_prior_games_with_hero"] == 1


def test_90_day_window_excludes_older_than_lower_bound(tmp_path: Path) -> None:
    t_old = T0
    t_after = T0 + timedelta(days=RECENT_WINDOW_DAYS, microseconds=1)
    specs = [
        {
            "match_id": M1,
            "start_time": t_old,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
        {
            "match_id": M2,
            "start_time": t_after,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    team_a = _row(frame, M2, TEAM_A, 1)
    assert team_a["recent_90d_team_games"] == 0
    assert team_a["recent_90d_team_games_with_hero"] == 0
    assert pd.isna(team_a["recent_90d_team_win_rate_with_hero"])
    assert pd.isna(team_a["recent_90d_team_hero_share"])
    assert team_a["team_prior_games_with_hero"] == 1
    assert team_a["team_prior_games"] == 1
    assert team_a["team_hero_share"] == pytest.approx(1.0)
    assert team_a["same_version_team_games_with_hero"] == 1


# --- days-since-last-played -----------------------------------------------


def test_days_since_team_played_hero(tmp_path: Path) -> None:
    t_later = T0 + timedelta(days=12)
    specs = [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
        {
            "match_id": M2,
            "start_time": T0 + timedelta(days=5),
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_heroes": (11, 2, 3, 4, 5),
        },
        {
            "match_id": M3,
            "start_time": t_later,
            "radiant_win": True,
            "game_version_id": VERSION_A,
        },
    ]
    frame = _assemble(tmp_path, specs)
    team_a = _row(frame, M3, TEAM_A, 1)
    # Last played hero 1 at T0, not the intervening hero-11 game.
    assert team_a["days_since_team_played_hero"] == pytest.approx(12.0)
    assert team_a["team_prior_games_with_hero"] == 1
    assert team_a["team_prior_games"] == 2
    assert team_a["team_hero_share"] == pytest.approx(0.5)

    team_a_at_m2 = _row(frame, M2, TEAM_A, 11)
    assert pd.isna(team_a_at_m2["days_since_team_played_hero"])


# --- roster changes do not reset team history -----------------------------


def test_team_roster_changes_do_not_reset_team_hero_history(tmp_path: Path) -> None:
    """Team A swaps its five players between M1 and M2. History follows
    team_id, not the roster.
    """
    specs = [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_team_id": TEAM_A,
            "dire_team_id": TEAM_B,
            "radiant_players": RADIANT_PLAYERS,
        },
        {
            "match_id": M2,
            "start_time": T1,
            "radiant_win": False,
            "game_version_id": VERSION_A,
            "radiant_team_id": TEAM_A,
            "dire_team_id": TEAM_C,
            "radiant_players": (P11, P12, P13, P14, P15),
            "dire_players": (P16, P17, P18, P19, P20),
            "dire_heroes": (16, 17, 18, 19, 20),
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    team_a = _row(frame, M2, TEAM_A, 1)
    assert team_a["side"] == "RADIANT"
    assert team_a["team_prior_games_with_hero"] == 1
    assert team_a["team_prior_wins_with_hero"] == 1
    assert team_a["team_prior_losses_with_hero"] == 0
    assert team_a["team_prior_games"] == 1
    assert team_a["team_prior_win_rate_with_hero"] == pytest.approx(1.0)
    assert team_a["team_hero_share"] == pytest.approx(1.0)
    assert team_a["days_since_team_played_hero"] == pytest.approx(1.0)


# --- invariants -----------------------------------------------------------


def test_games_with_hero_at_most_team_prior_games(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs())
    assert (
        frame["team_prior_games_with_hero"] <= frame["team_prior_games"]
    ).all()
    assert (
        frame["recent_90d_team_games_with_hero"] <= frame["recent_90d_team_games"]
    ).all()
    assert (
        frame["same_version_team_games_with_hero"] <= frame["team_prior_games"]
    ).all()


def test_all_rates_and_shares_in_unit_interval(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs())
    for column in RATE_AND_SHARE_COLUMNS:
        values = frame[column].dropna()
        assert (values >= 0.0).all(), column
        assert (values <= 1.0).all(), column


# --- grain / catalog ------------------------------------------------------


def test_output_columns_and_ten_rows_per_match(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs())
    assert list(frame.columns) == list(TEAM_HERO_COLUMNS)
    assert set(frame["match_id"].unique()) == {M1, M2, M3}
    assert len(frame) == 30
    assert frame.groupby("match_id").size().eq(10).all()
    assert "player_id" not in frame.columns
    team_a = _row(frame, M3, TEAM_A, 1)
    assert team_a["hero_name"] == "Hero 1"


def test_hero_name_null_when_catalog_is_absent(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), heroes=None, match_id=M3)
    assert frame["hero_name"].isna().all()
    assert len(frame) == 10
