"""Tests for the descriptive player × hero layer (`features.player_hero`).

Small deterministic fixtures with hand-calculated expected values.
Does not go through PRE_DRAFT snapshot SQL, Elo, or training assembly.
`slot_in_side` is lobby order only and is never treated as position 1-5.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from player_hero_helpers import (
    draft_and_player_rows,
    match_row,
    player_hero_frame,
)

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.player_hero import (
    PLAYER_HERO_COLUMNS,
    PLAYER_HERO_METRIC_COLUMNS,
    RECENT_WINDOW_DAYS,
    player_hero_sql,
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

P1, P2, P3, P4, P5 = 1, 2, 3, 4, 5
P6, P7, P8, P9, P10 = 6, 7, 8, 9, 10
# Extra players for same-timestamp dual-match fixtures.
P11, P12, P13, P14, P15 = 11, 12, 13, 14, 15
P16, P17, P18, P19, P20 = 16, 17, 18, 19, 20

RADIANT_PLAYERS = (P1, P2, P3, P4, P5)
DIRE_PLAYERS = (P6, P7, P8, P9, P10)
RADIANT_HEROES = (1, 2, 3, 4, 5)
DIRE_HEROES = (6, 7, 8, 9, 10)

TEAM_A, TEAM_B, TEAM_C, TEAM_D = 100, 200, 300, 400

CATALOG_HEROES = [
    {"id": hero_id, "displayName": f"Hero {hero_id}"}
    for hero_id in range(1, 23)
]


def _row(frame: pd.DataFrame, match_id: int, player_id: int) -> pd.Series:
    subset = frame[
        (frame["match_id"] == match_id) & (frame["player_id"] == player_id)
    ]
    assert len(subset) == 1, (
        f"expected one row for ({match_id}, {player_id}), got {len(subset)}"
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
    return player_hero_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        match_id=match_id,
    )


def _three_match_specs() -> list[dict]:
    """Two historical maps plus one evaluation map, all version A.

    M1 (T0, Radiant win): default heroes. P1 Radiant hero 1 wins; P6 Dire
    hero 6 loses.
    M2 (T1, Dire win): P1 still Radiant hero 1 (now a loss); P6 Dire hero 6
    wins. P2 switches to hero 11.
    M3 (T2) is the evaluation point -- its own draft/result must not count.
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
    sql = player_hero_sql(catalog_registered=True)
    assert "start_time <=" not in sql
    assert "EXCLUDE GROUP" in sql
    assert sql.count("EXCLUDE GROUP") == 5
    assert "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in sql
    assert f"INTERVAL {RECENT_WINDOW_DAYS} DAY" in sql
    assert "h.start_time < c.start_time" not in sql


def test_sql_partitions_by_player_and_player_hero_not_slot() -> None:
    sql = player_hero_sql(catalog_registered=True)
    assert "PARTITION BY player_id" in sql
    assert "PARTITION BY player_id, hero_id" in sql
    assert "PARTITION BY player_id, hero_id, game_version_id" in sql
    for clause in sql.split("PARTITION BY")[1:]:
        assert "slot_in_side" not in clause.split("ORDER BY")[0]


def test_sql_never_orders_history_by_match_id() -> None:
    sql = player_hero_sql(catalog_registered=True)
    assert "ORDER BY start_time" in sql
    assert "ORDER BY match_id" not in sql
    assert "ORDER BY start_time, match_id" not in sql


def test_sql_does_not_encode_positions_lanes_roles_or_elo() -> None:
    sql = player_hero_sql(catalog_registered=True).lower()
    for forbidden in (
        "position",
        "lane",
        "role",
        "synergy",
        "counter",
        "elo",
        "meta_fit",
    ):
        assert forbidden not in sql


def test_player_hero_is_not_part_of_training_or_pre_draft_snapshot() -> None:
    assert set(PLAYER_HERO_METRIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(PLAYER_HERO_METRIC_COLUMNS).isdisjoint(SNAPSHOT_COLUMNS)
    assert set(PLAYER_HERO_METRIC_COLUMNS).isdisjoint(ALL_FEATURE_COLUMNS)
    assert "player_hero" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "prior_games_on_hero" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "prior_hero_share" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged_by_this_layer() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 5
    assert REFERENCE_SCHEMA_VERSION == 1


# --- current / future exclusion -------------------------------------------


def test_current_match_excluded_from_its_own_counts(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    p1 = _row(frame, M2, P1)
    assert p1["hero_id"] == 1
    assert p1["prior_games_on_hero"] == 1
    assert p1["prior_wins_on_hero"] == 1
    assert p1["prior_losses_on_hero"] == 0
    assert p1["prior_player_games"] == 1


def test_future_matches_excluded(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    p2 = _row(frame, M2, P2)
    # P2 first played hero 11 in M2; M3 also has P2 on 11 and must not count.
    assert p2["hero_id"] == 11
    assert p2["prior_games_on_hero"] == 0
    assert p2["prior_player_games"] == 1
    assert pd.isna(p2["prior_win_rate_on_hero"])


# --- identical-timestamp leakage ------------------------------------------


def test_identical_timestamps_are_mutually_blind(tmp_path: Path) -> None:
    """Matches sharing start_time must not contribute to each other.

    P1 plays hero 1 at T0. At T1, P1 appears in both M2 and M3 on hero 1
    with opposite outcomes. Each T1 row may see T0, never the peer.
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
            "dire_players": (P11, P12, P13, P14, P15),
            "dire_heroes": (16, 17, 18, 19, 20),
        },
        {
            "match_id": M3,
            "start_time": same_time,
            "radiant_win": False,
            "game_version_id": VERSION_A,
            "dire_players": (P16, P17, P18, P19, P20),
            "dire_heroes": (16, 17, 18, 19, 20),
        },
    ]
    frame = _assemble(tmp_path, specs)

    for match_id in (M2, M3):
        p1 = _row(frame, match_id, P1)
        assert p1["hero_id"] == 1
        assert p1["prior_games_on_hero"] == 1
        assert p1["prior_wins_on_hero"] == 1
        assert p1["prior_losses_on_hero"] == 0
        assert p1["prior_player_games"] == 1
        assert p1["recent_90d_games_on_hero"] == 1

    p1_at_m1 = _row(frame, M1, P1)
    assert p1_at_m1["prior_games_on_hero"] == 0
    assert p1_at_m1["prior_player_games"] == 0
    assert pd.isna(p1_at_m1["prior_win_rate_on_hero"])


# --- patch transition -----------------------------------------------------


def test_first_match_of_new_version_has_zero_same_version_history(
    tmp_path: Path,
) -> None:
    """P1 plays hero 1 in every version-A match. First version-B match has
    zero same-version hero history, but all-time and 90d still see A.
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
    p1 = _row(frame, M3, P1)

    assert p1["same_version_games_on_hero"] == 0
    assert p1["same_version_wins_on_hero"] == 0
    assert pd.isna(p1["same_version_win_rate_on_hero"])

    assert p1["prior_games_on_hero"] == 2
    assert p1["prior_wins_on_hero"] == 1
    assert p1["prior_losses_on_hero"] == 1
    assert p1["prior_win_rate_on_hero"] == pytest.approx(0.5)
    assert p1["recent_90d_games_on_hero"] == 2
    assert p1["recent_90d_wins_on_hero"] == 1
    assert p1["recent_90d_win_rate_on_hero"] == pytest.approx(0.5)


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
    p1 = _row(frame, M3, P1)
    assert p1["same_version_games_on_hero"] == 1
    assert p1["same_version_wins_on_hero"] == 1
    assert p1["prior_games_on_hero"] == 2
    assert p1["recent_90d_games_on_hero"] == 2


# --- Radiant and Dire wins ------------------------------------------------


def test_radiant_and_dire_wins_and_losses(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M3)
    p1 = _row(frame, M3, P1)
    # Radiant hero 1: M1 win, M2 loss.
    assert p1["side"] == "RADIANT"
    assert p1["hero_id"] == 1
    assert p1["prior_games_on_hero"] == 2
    assert p1["prior_wins_on_hero"] == 1
    assert p1["prior_losses_on_hero"] == 1
    assert p1["prior_win_rate_on_hero"] == pytest.approx(0.5)

    p6 = _row(frame, M3, P6)
    # Dire hero 6: M1 loss (Radiant won), M2 win (Dire won).
    assert p6["side"] == "DIRE"
    assert p6["hero_id"] == 6
    assert p6["prior_games_on_hero"] == 2
    assert p6["prior_wins_on_hero"] == 1
    assert p6["prior_losses_on_hero"] == 1
    assert p6["prior_win_rate_on_hero"] == pytest.approx(0.5)

    p5 = _row(frame, M3, P5)
    # Radiant hero 5 in M1 (win) and M2 (loss).
    assert p5["prior_wins_on_hero"] == 1
    assert p5["prior_losses_on_hero"] == 1

    p10 = _row(frame, M3, P10)
    assert p10["prior_wins_on_hero"] == 1
    assert p10["prior_losses_on_hero"] == 1


def test_wins_plus_losses_equal_games_on_hero(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs())
    assert (
        frame["prior_wins_on_hero"] + frame["prior_losses_on_hero"]
        == frame["prior_games_on_hero"]
    ).all()


# --- zero-history NULL semantics ------------------------------------------


def test_zero_history_null_rates_and_shares_on_first_match(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M1)
    p1 = _row(frame, M1, P1)
    assert p1["prior_games_on_hero"] == 0
    assert p1["prior_wins_on_hero"] == 0
    assert p1["prior_losses_on_hero"] == 0
    assert p1["prior_player_games"] == 0
    assert p1["same_version_games_on_hero"] == 0
    assert p1["recent_90d_games_on_hero"] == 0
    assert p1["recent_90d_player_games"] == 0
    assert pd.isna(p1["prior_win_rate_on_hero"])
    assert pd.isna(p1["same_version_win_rate_on_hero"])
    assert pd.isna(p1["recent_90d_win_rate_on_hero"])
    assert pd.isna(p1["prior_hero_share"])
    assert pd.isna(p1["recent_90d_hero_share"])
    assert pd.isna(p1["days_since_last_played_hero"])


# --- player switching heroes ----------------------------------------------


def test_player_switching_heroes_does_not_count_other_hero_games(
    tmp_path: Path,
) -> None:
    """P1 plays hero 1, then hero 11. Hero-11 metrics ignore the hero-1 game;
    player-level games still count it.
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
    p1 = _row(frame, M2, P1)
    assert p1["hero_id"] == 11
    assert p1["prior_games_on_hero"] == 0
    assert p1["prior_wins_on_hero"] == 0
    assert p1["prior_player_games"] == 1
    assert p1["prior_hero_share"] == pytest.approx(0.0)
    assert pd.isna(p1["prior_win_rate_on_hero"])
    assert pd.isna(p1["days_since_last_played_hero"])


# --- hero-share correctness -----------------------------------------------


def test_hero_share_is_games_on_hero_over_player_games(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M3)
    p1 = _row(frame, M3, P1)
    # Two prior games, both on hero 1.
    assert p1["prior_games_on_hero"] == 2
    assert p1["prior_player_games"] == 2
    assert p1["prior_hero_share"] == pytest.approx(1.0)
    assert p1["recent_90d_games_on_hero"] == 2
    assert p1["recent_90d_player_games"] == 2
    assert p1["recent_90d_hero_share"] == pytest.approx(1.0)

    p2 = _row(frame, M3, P2)
    # M1 on hero 2, M2 on hero 11; current is hero 11.
    assert p2["hero_id"] == 11
    assert p2["prior_games_on_hero"] == 1
    assert p2["prior_player_games"] == 2
    assert p2["prior_hero_share"] == pytest.approx(0.5)
    assert p2["recent_90d_hero_share"] == pytest.approx(0.5)


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
    p1 = _row(frame, M2, P1)
    assert p1["recent_90d_player_games"] == 1
    assert p1["recent_90d_games_on_hero"] == 1
    assert p1["recent_90d_hero_share"] == pytest.approx(1.0)
    assert p1["same_version_games_on_hero"] == 0
    assert pd.isna(p1["same_version_win_rate_on_hero"])
    assert p1["prior_games_on_hero"] == 1


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
    p1 = _row(frame, M2, P1)
    assert p1["recent_90d_player_games"] == 0
    assert p1["recent_90d_games_on_hero"] == 0
    assert pd.isna(p1["recent_90d_win_rate_on_hero"])
    assert pd.isna(p1["recent_90d_hero_share"])
    assert p1["prior_games_on_hero"] == 1
    assert p1["prior_player_games"] == 1
    assert p1["prior_hero_share"] == pytest.approx(1.0)
    assert p1["same_version_games_on_hero"] == 1


# --- days-since-last-played -----------------------------------------------


def test_days_since_last_played_hero(tmp_path: Path) -> None:
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
    p1 = _row(frame, M3, P1)
    # Last played hero 1 at T0, not the intervening hero-11 game.
    assert p1["hero_id"] == 1
    assert p1["days_since_last_played_hero"] == pytest.approx(12.0)
    assert p1["prior_games_on_hero"] == 1
    assert p1["prior_player_games"] == 2
    assert p1["prior_hero_share"] == pytest.approx(0.5)

    p1_at_m2 = _row(frame, M2, P1)
    assert p1_at_m2["hero_id"] == 11
    assert pd.isna(p1_at_m2["days_since_last_played_hero"])


# --- player changing teams ------------------------------------------------


def test_player_changing_teams_does_not_reset_player_hero_history(
    tmp_path: Path,
) -> None:
    """P1 moves from Team A (Radiant) to Team D (Dire). History follows
    player_id, not team_id.
    """
    specs = [
        {
            "match_id": M1,
            "start_time": T0,
            "radiant_win": True,
            "game_version_id": VERSION_A,
            "radiant_team_id": TEAM_A,
            "dire_team_id": TEAM_B,
        },
        {
            "match_id": M2,
            "start_time": T1,
            "radiant_win": False,
            "game_version_id": VERSION_A,
            "radiant_team_id": TEAM_C,
            "dire_team_id": TEAM_D,
            "radiant_players": (P11, P12, P13, P14, P15),
            "dire_players": (P1, P7, P8, P9, P10),
            "radiant_heroes": (11, 12, 13, 14, 15),
            "dire_heroes": (1, 7, 8, 9, 10),
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    p1 = _row(frame, M2, P1)
    assert p1["team_id"] == TEAM_D
    assert p1["side"] == "DIRE"
    assert p1["hero_id"] == 1
    assert p1["prior_games_on_hero"] == 1
    assert p1["prior_wins_on_hero"] == 1
    assert p1["prior_losses_on_hero"] == 0
    assert p1["prior_player_games"] == 1
    assert p1["prior_win_rate_on_hero"] == pytest.approx(1.0)
    assert p1["prior_hero_share"] == pytest.approx(1.0)
    assert p1["days_since_last_played_hero"] == pytest.approx(1.0)


# --- grain / catalog ------------------------------------------------------


def test_output_columns_and_ten_rows_per_match(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs())
    assert list(frame.columns) == list(PLAYER_HERO_COLUMNS)
    assert set(frame["match_id"].unique()) == {M1, M2, M3}
    assert len(frame) == 30
    assert frame.groupby("match_id").size().eq(10).all()
    p1 = _row(frame, M3, P1)
    assert p1["hero_name"] == "Hero 1"
    assert p1["slot_in_side"] == 0


def test_hero_name_null_when_catalog_is_absent(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), heroes=None, match_id=M3)
    assert frame["hero_name"].isna().all()
    assert len(frame) == 10
