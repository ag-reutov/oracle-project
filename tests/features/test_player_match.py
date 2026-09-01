"""Tests for player-match fact and strictly-prior player state.

Small deterministic fixtures with hand-calculated expected values.
Does not go through PRE_DRAFT snapshot SQL, Elo, or training assembly.
`slot_in_side` is lobby order only and is never treated as position 1-5.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from hero_meta_helpers import match_row
from player_match_helpers import (
    draft_and_player_rows,
    player_match_frame,
    player_state_frame,
)

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.player_match import (
    PLAYER_MATCH_COLUMNS,
    PLAYER_STATE_COLUMNS,
    PLAYER_STATE_METRIC_COLUMNS,
    player_match_sql,
    player_state_sql,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.features.temporal import STRICT_PRIOR_RANGE_SQL
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS

T0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
T_SAME_DAY_LATER = datetime(2024, 1, 1, 18, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
T2 = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)

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
SERIES_1, SERIES_2 = 501, 502


def _row(frame: pd.DataFrame, match_id: int, player_id: int) -> pd.Series:
    subset = frame[
        (frame["match_id"] == match_id) & (frame["player_id"] == player_id)
    ]
    assert len(subset) == 1, (
        f"expected one row for ({match_id}, {player_id}), got {len(subset)}"
    )
    return subset.iloc[0]


def _assemble_state(
    tmp_path: Path,
    specs: list[dict],
    *,
    match_id: int | None = None,
) -> pd.DataFrame:
    matches, players, drafts = _tables(specs)
    return player_state_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        match_id=match_id,
    )


def _assemble_match(tmp_path: Path, specs: list[dict]) -> pd.DataFrame:
    matches, players, drafts = _tables(specs)
    return player_match_frame(
        tmp_path, matches=matches, players=players, drafts=drafts
    )


def _tables(
    specs: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    matches: list[dict] = []
    players: list[dict] = []
    drafts: list[dict] = []
    for spec in specs:
        row = match_row(
            spec["match_id"],
            start_time=spec["start_time"],
            radiant_win=spec["radiant_win"],
            game_version_id=spec["game_version_id"],
            radiant_team_id=spec.get("radiant_team_id", TEAM_A),
            dire_team_id=spec.get("dire_team_id", TEAM_B),
        )
        row["series_id"] = spec.get("series_id", SERIES_1)
        matches.append(row)
        draft_rows, player_rows = draft_and_player_rows(
            spec["match_id"],
            radiant_player_ids=spec.get("radiant_players", RADIANT_PLAYERS),
            dire_player_ids=spec.get("dire_players", DIRE_PLAYERS),
            radiant_hero_ids=spec.get("radiant_heroes", RADIANT_HEROES),
            dire_hero_ids=spec.get("dire_heroes", DIRE_HEROES),
        )
        drafts.extend(draft_rows)
        players.extend(player_rows)
    return matches, players, drafts


# --- SQL / contract guards ------------------------------------------------


def test_player_match_sql_projects_requested_columns_only() -> None:
    sql = player_match_sql()
    assert "radiant_win" in sql
    assert "duration_seconds" not in sql
    assert "mapper_version" not in sql
    for column in PLAYER_MATCH_COLUMNS:
        assert column in sql


def test_state_sql_uses_shared_exclude_group_windows() -> None:
    sql = player_state_sql()
    assert STRICT_PRIOR_RANGE_SQL in sql
    assert sql.count("EXCLUDE GROUP") == 2
    assert "start_time <=" not in sql
    assert "h.start_time < c.start_time" not in sql
    assert "PARTITION BY player_id ORDER BY start_time" in sql
    assert "PARTITION BY player_id, game_version_id ORDER BY start_time" in sql


def test_state_sql_never_orders_history_by_match_id() -> None:
    sql = player_state_sql()
    assert "ORDER BY start_time" in sql
    assert "ORDER BY match_id" not in sql
    assert "ORDER BY start_time, match_id" not in sql


def test_state_sql_does_not_partition_by_slot_or_observed_position() -> None:
    sql = player_state_sql()
    for clause in sql.split("PARTITION BY")[1:]:
        partition = clause.split("ORDER BY")[0]
        for forbidden in ("slot_in_side", "position", "lane", "role"):
            assert forbidden not in partition
    lowered = sql.lower()
    for forbidden in ("elo", "synergy", "counter", "prior_position", "modal_position"):
        assert forbidden not in lowered


def test_player_match_sql_exposes_observed_position_separately_from_slot() -> None:
    sql = player_match_sql()
    assert "mp.slot_in_side" in sql
    assert "mp.position" in sql
    assert "mp.lane" in sql
    assert "mp.role" in sql


def test_observed_position_is_not_a_training_or_pre_draft_feature() -> None:
    for column in ("position", "lane", "role"):
        assert column not in FEATURE_COLUMNS
        assert column not in SNAPSHOT_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
        assert column not in PLAYER_STATE_METRIC_COLUMNS


def test_player_state_is_not_part_of_training_or_pre_draft_snapshot() -> None:
    assert set(PLAYER_STATE_METRIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(PLAYER_STATE_METRIC_COLUMNS).isdisjoint(SNAPSHOT_COLUMNS)
    assert set(PLAYER_STATE_METRIC_COLUMNS).isdisjoint(ALL_FEATURE_COLUMNS)
    assert "prior_games" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "player_state_sql" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged_by_this_layer() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 3
    assert REFERENCE_SCHEMA_VERSION == 1


# --- fact grain -----------------------------------------------------------


def test_player_match_grain_is_one_row_per_match_player(tmp_path: Path) -> None:
    frame = _assemble_match(
        tmp_path,
        [
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
        ],
    )
    assert list(frame.columns) == list(PLAYER_MATCH_COLUMNS)
    assert len(frame) == 20
    assert frame["match_id"].nunique() == 2
    assert (frame.groupby("match_id").size() == 10).all()
    assert frame.duplicated(["match_id", "player_id"]).sum() == 0
    p1 = _row(frame, M1, P1)
    assert p1["team_id"] == TEAM_A
    assert p1["side"] == "RADIANT"
    assert p1["hero_id"] == 1
    assert bool(p1["won"]) is True
    assert p1["slot_in_side"] == 0
    assert pd.isna(p1["position"])
    assert pd.isna(p1["lane"])
    assert pd.isna(p1["role"])
    p6 = _row(frame, M1, P6)
    assert p6["team_id"] == TEAM_B
    assert bool(p6["won"]) is False


# --- first appearance -----------------------------------------------------


def test_first_appearance_has_zero_history_and_null_recency(tmp_path: Path) -> None:
    frame = _assemble_state(
        tmp_path,
        [
            {
                "match_id": M1,
                "start_time": T0,
                "radiant_win": True,
                "game_version_id": VERSION_A,
            }
        ],
    )
    p1 = _row(frame, M1, P1)
    assert list(frame.columns) == list(PLAYER_STATE_COLUMNS)
    assert p1["prior_games"] == 0
    assert p1["prior_wins"] == 0
    assert pd.isna(p1["prior_win_rate"])
    assert pd.isna(p1["previous_match_start_time"])
    assert pd.isna(p1["days_since_previous_match"])
    assert p1["prior_unique_heroes"] == 0
    assert p1["version_prior_games"] == 0
    assert p1["version_prior_wins"] == 0
    assert pd.isna(p1["version_prior_win_rate"])
    assert p1["version_prior_unique_heroes"] == 0
    assert bool(p1["won"]) is True


# --- sequential matches ---------------------------------------------------


def test_later_match_sees_earlier_match(tmp_path: Path) -> None:
    frame = _assemble_state(
        tmp_path,
        [
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
        ],
    )
    p1_m2 = _row(frame, M2, P1)
    assert p1_m2["prior_games"] == 1
    assert p1_m2["prior_wins"] == 1
    assert p1_m2["prior_win_rate"] == pytest.approx(1.0)
    assert p1_m2["previous_match_start_time"] == T0
    assert p1_m2["days_since_previous_match"] == pytest.approx(1.0)
    assert p1_m2["prior_unique_heroes"] == 1
    assert bool(p1_m2["won"]) is False

    p6_m2 = _row(frame, M2, P6)
    assert p6_m2["prior_games"] == 1
    assert p6_m2["prior_wins"] == 0
    assert p6_m2["prior_win_rate"] == pytest.approx(0.0)


def test_current_and_future_matches_excluded(tmp_path: Path) -> None:
    frame = _assemble_state(
        tmp_path,
        [
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
                "game_version_id": VERSION_A,
            },
        ],
        match_id=M2,
    )
    assert set(frame["match_id"].unique()) == {M2}
    p1 = _row(frame, M2, P1)
    assert p1["prior_games"] == 1
    assert p1["prior_wins"] == 1


# --- same-day sequential --------------------------------------------------


def test_same_day_later_match_may_use_earlier(tmp_path: Path) -> None:
    frame = _assemble_state(
        tmp_path,
        [
            {
                "match_id": M1,
                "start_time": T0,
                "radiant_win": True,
                "game_version_id": VERSION_A,
            },
            {
                "match_id": M2,
                "start_time": T_SAME_DAY_LATER,
                "radiant_win": False,
                "game_version_id": VERSION_A,
            },
        ],
    )
    p1 = _row(frame, M2, P1)
    assert T0.date() == T_SAME_DAY_LATER.date()
    assert p1["prior_games"] == 1
    assert p1["prior_wins"] == 1
    assert p1["days_since_previous_match"] == pytest.approx(0.25)


# --- exact timestamp ties -------------------------------------------------


def test_identical_timestamps_are_mutually_blind(tmp_path: Path) -> None:
    """A player in two matches at the same start_time must not see the peer."""
    frame = _assemble_state(
        tmp_path,
        [
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
                "dire_players": (P11, P12, P13, P14, P15),
                "dire_heroes": (16, 17, 18, 19, 20),
            },
            {
                "match_id": M3,
                "start_time": T1,
                "radiant_win": False,
                "game_version_id": VERSION_A,
                "radiant_heroes": (11, 12, 13, 14, 15),
                "dire_players": (P16, P17, P18, P19, P20),
                "dire_heroes": (16, 17, 18, 19, 20),
            },
        ],
    )
    for match_id in (M2, M3):
        p1 = _row(frame, match_id, P1)
        assert p1["prior_games"] == 1
        assert p1["prior_wins"] == 1
        assert p1["prior_unique_heroes"] == 1
        assert p1["previous_match_start_time"] == T0

    p1_m1 = _row(frame, M1, P1)
    assert p1_m1["prior_games"] == 0
    assert pd.isna(p1_m1["previous_match_start_time"])


# --- series ---------------------------------------------------------------


def test_later_series_map_sees_earlier_map_only(tmp_path: Path) -> None:
    frame = _assemble_state(
        tmp_path,
        [
            {
                "match_id": M1,
                "start_time": T0,
                "radiant_win": True,
                "game_version_id": VERSION_A,
                "series_id": SERIES_1,
            },
            {
                "match_id": M2,
                "start_time": T1,
                "radiant_win": False,
                "game_version_id": VERSION_A,
                "series_id": SERIES_1,
            },
        ],
    )
    p1_map2 = _row(frame, M2, P1)
    assert p1_map2["series_id"] == SERIES_1
    assert p1_map2["prior_games"] == 1
    assert p1_map2["prior_wins"] == 1
    sql = player_state_sql()
    assert "winningteamid" not in sql.lower()
    assert "winning_team" not in sql.lower()


# --- transfer -------------------------------------------------------------


def test_player_history_follows_player_id_across_teams(tmp_path: Path) -> None:
    frame = _assemble_state(
        tmp_path,
        [
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
                "radiant_players": (P1, P11, P12, P13, P14),
                "dire_players": (P15, P16, P17, P18, P19),
                "radiant_heroes": (11, 12, 13, 14, 15),
                "dire_heroes": (16, 17, 18, 19, 20),
            },
        ],
    )
    p1_m2 = _row(frame, M2, P1)
    assert p1_m2["team_id"] == TEAM_C
    assert p1_m2["prior_games"] == 1
    assert p1_m2["prior_wins"] == 1
    assert p1_m2["prior_unique_heroes"] == 1


# --- patch / version ------------------------------------------------------


def test_career_continues_across_versions_same_version_resets(tmp_path: Path) -> None:
    frame = _assemble_state(
        tmp_path,
        [
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
                "game_version_id": VERSION_B,
            },
        ],
    )
    p1 = _row(frame, M2, P1)
    assert p1["game_version_id"] == VERSION_B
    assert p1["prior_games"] == 1
    assert p1["prior_wins"] == 1
    assert p1["prior_unique_heroes"] == 1
    assert p1["version_prior_games"] == 0
    assert p1["version_prior_wins"] == 0
    assert pd.isna(p1["version_prior_win_rate"])
    assert p1["version_prior_unique_heroes"] == 0


# --- unique heroes --------------------------------------------------------


def test_current_hero_excluded_from_prior_unique_heroes(tmp_path: Path) -> None:
    frame = _assemble_state(
        tmp_path,
        [
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
            },
            {
                "match_id": M3,
                "start_time": T2,
                "radiant_win": True,
                "game_version_id": VERSION_A,
                "radiant_heroes": (11, 12, 3, 4, 5),
            },
        ],
    )
    p1_m1 = _row(frame, M1, P1)
    assert p1_m1["hero_id"] == 1
    assert p1_m1["prior_unique_heroes"] == 0

    p1_m2 = _row(frame, M2, P1)
    assert p1_m2["hero_id"] == 1
    assert p1_m2["prior_unique_heroes"] == 1

    p2_m2 = _row(frame, M2, P2)
    assert p2_m2["hero_id"] == 11
    assert p2_m2["prior_unique_heroes"] == 1

    p1_m3 = _row(frame, M3, P1)
    assert p1_m3["hero_id"] == 11
    assert p1_m3["prior_unique_heroes"] == 1
    assert p1_m3["prior_games"] == 2
    assert p1_m3["version_prior_unique_heroes"] == 1


def test_match_id_is_not_a_time_proxy(tmp_path: Path) -> None:
    """Later start_time with a smaller match_id still sees the earlier game."""
    early_id, late_id = 9000, 100
    frame = _assemble_state(
        tmp_path,
        [
            {
                "match_id": early_id,
                "start_time": T0,
                "radiant_win": True,
                "game_version_id": VERSION_A,
            },
            {
                "match_id": late_id,
                "start_time": T1,
                "radiant_win": False,
                "game_version_id": VERSION_A,
            },
        ],
    )
    p1 = _row(frame, late_id, P1)
    assert late_id < early_id
    assert p1["prior_games"] == 1
    assert p1["previous_match_start_time"] == T0


def test_current_match_position_is_fact_only_history_stays_strict_prior(
    tmp_path: Path,
) -> None:
    """Observed position on match M is not a current-match feature.

    It is stored on the fact row. `prior_*` still uses only
    `H.start_time < M.start_time` and does not partition or score by
    position.
    """
    matches, players, drafts = _tables(
        [
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
        ]
    )
    for player in players:
        if player["match_id"] == M1 and player["player_id"] == P1:
            player["position"] = "POSITION_5"
            player["lane"] = "SAFE_LANE"
            player["role"] = "HARD_SUPPORT"
        if player["match_id"] == M2 and player["player_id"] == P1:
            player["position"] = "POSITION_1"
            player["lane"] = "SAFE_LANE"
            player["role"] = "CORE"

    frame = player_state_frame(
        tmp_path, matches=matches, players=players, drafts=drafts
    )
    m1 = _row(frame, M1, P1)
    m2 = _row(frame, M2, P1)
    assert m1["position"] == "POSITION_5"
    assert m2["position"] == "POSITION_1"
    assert int(m1["slot_in_side"]) == 0
    assert int(m2["slot_in_side"]) == 0
    assert m1["position"] != f"POSITION_{int(m1['slot_in_side']) + 1}"
    assert m2["prior_games"] == 1
    assert m2["prior_wins"] == 1
    assert "prior_position" not in frame.columns
    assert list(frame.columns) == list(PLAYER_STATE_COLUMNS)

