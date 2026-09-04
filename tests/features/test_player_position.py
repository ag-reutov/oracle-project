"""Tests for leakage-safe historical player × position state.

Deterministic fixtures with hand-calculated expected values.
Does not go through PRE_DRAFT snapshot SQL, Elo, or training assembly.
Does not infer or fill missing positions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from hero_meta_helpers import match_row
from player_position_helpers import (
    assign_positions,
    draft_and_player_rows,
    player_position_state_frame,
)

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.player_position import (
    PLAYER_POSITION_STATE_COLUMNS,
    PLAYER_POSITION_STATE_METRIC_COLUMNS,
    player_position_state_sql,
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
T3 = datetime(2024, 1, 4, 12, 0, tzinfo=UTC)

VERSION_A = 10
VERSION_B = 11

M1, M2, M3, M4 = 4001, 1002, 3003, 5004

P1, P2, P3, P4, P5 = 1, 2, 3, 4, 5
P6, P7, P8, P9, P10 = 6, 7, 8, 9, 10
P11, P12, P13, P14, P15 = 11, 12, 13, 14, 15
P16, P17, P18, P19, P20 = 16, 17, 18, 19, 20

RADIANT_PLAYERS = (P1, P2, P3, P4, P5)
DIRE_PLAYERS = (P6, P7, P8, P9, P10)
RADIANT_HEROES = (1, 2, 3, 4, 5)
DIRE_HEROES = (6, 7, 8, 9, 10)

TEAM_A, TEAM_B, TEAM_C, TEAM_D = 100, 200, 300, 400
SERIES_1 = 501


def _row(frame: pd.DataFrame, match_id: int, player_id: int) -> pd.Series:
    subset = frame[
        (frame["match_id"] == match_id) & (frame["player_id"] == player_id)
    ]
    assert len(subset) == 1
    return subset.iloc[0]


def _tables(
    specs: list[dict],
    *,
    positions: dict[tuple[int, int], str | None] | None = None,
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
    if positions:
        assign_positions(players, positions)
    return matches, players, drafts


def _assemble(
    tmp_path: Path,
    specs: list[dict],
    *,
    positions: dict[tuple[int, int], str | None] | None = None,
    match_id: int | None = None,
) -> pd.DataFrame:
    matches, players, drafts = _tables(specs, positions=positions)
    return player_position_state_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        match_id=match_id,
    )


def test_sql_uses_shared_strict_prior_windows() -> None:
    sql = player_position_state_sql()
    assert STRICT_PRIOR_RANGE_SQL in sql
    assert sql.count("EXCLUDE GROUP") == 2
    assert "ORDER BY start_time" in sql
    assert "ORDER BY match_id" not in sql
    assert "PARTITION BY player_id ORDER BY start_time" in sql
    assert "PARTITION BY player_id, game_version_id ORDER BY start_time" in sql
    for clause in sql.split("PARTITION BY")[1:]:
        partition = clause.split("ORDER BY")[0]
        assert "slot_in_side" not in partition
        assert "position" not in partition


def test_historical_metrics_are_not_training_or_snapshot_features() -> None:
    for column in PLAYER_POSITION_STATE_METRIC_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in SNAPSHOT_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
    assert "prior_games_position_1" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "player_position_state" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 5
    assert REFERENCE_SCHEMA_VERSION == 2


def test_first_career_match_has_no_position_history(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {
                "match_id": M1,
                "start_time": T0,
                "radiant_win": True,
                "game_version_id": VERSION_A,
            }
        ],
        positions={(M1, P1): "POSITION_3"},
    )
    assert list(frame.columns) == list(PLAYER_POSITION_STATE_COLUMNS)
    row = _row(frame, M1, P1)
    assert row["position"] == "POSITION_3"
    assert row["prior_games"] == 0
    assert row["prior_explicit_position_games"] == 0
    assert row["prior_games_position_3"] == 0
    assert pd.isna(row["prior_win_rate_position_3"])
    assert pd.isna(row["prior_share_position_3"])
    assert pd.isna(row["historical_modal_position"])
    assert row["historical_distinct_positions"] == 0
    assert pd.isna(row["previous_explicit_position"])
    assert pd.isna(row["days_since_previous_explicit_position"])
    assert row["recent_5_explicit_games"] == 0
    assert pd.isna(row["recent_position_stability"])


def test_sequential_same_position_increments_counts(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={
            (M1, P1): "POSITION_3",
            (M2, P1): "POSITION_3",
            (M3, P1): "POSITION_3",
        },
    )
    m2 = _row(frame, M2, P1)
    assert m2["prior_games_position_3"] == 1
    assert m2["prior_wins_position_3"] == 1
    assert m2["prior_win_rate_position_3"] == pytest.approx(1.0)
    assert m2["prior_share_position_3"] == pytest.approx(1.0)
    assert m2["previous_start_time_position_3"] == T0
    assert m2["days_since_position_3"] == pytest.approx(1.0)
    assert m2["historical_modal_position"] == "POSITION_3"
    assert m2["previous_explicit_position"] == "POSITION_3"

    m3 = _row(frame, M3, P1)
    assert m3["prior_games"] == 2
    assert m3["prior_games_position_3"] == 2
    assert m3["prior_wins_position_3"] == 1
    assert m3["prior_win_rate_position_3"] == pytest.approx(0.5)
    assert m3["prior_games_position_1"] == 0
    assert m3["version_prior_games_position_3"] == 2


def test_current_observed_position_is_not_used_as_history(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={(M1, P1): "POSITION_5", (M2, P1): "POSITION_1"},
    )
    m2 = _row(frame, M2, P1)
    assert m2["position"] == "POSITION_1"
    assert m2["prior_games_position_1"] == 0
    assert m2["prior_games_position_5"] == 1
    assert m2["previous_explicit_position"] == "POSITION_5"


def test_position_switch_preserves_both_histories(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={
            (M1, P1): "POSITION_3",
            (M2, P1): "POSITION_1",
            (M3, P1): "POSITION_1",
        },
    )
    m1 = _row(frame, M1, P1)
    assert m1["prior_games_position_3"] == 0
    m3 = _row(frame, M3, P1)
    assert m3["prior_games_position_3"] == 1
    assert m3["prior_games_position_1"] == 1
    assert m3["prior_explicit_position_games"] == 2
    assert m3["prior_share_position_3"] == pytest.approx(0.5)
    assert pd.isna(m3["historical_modal_position"])
    assert m3["historical_distinct_positions"] == 2
    assert m3["previous_explicit_position"] == "POSITION_1"
    assert m3["prior_games_same_as_previous_position"] == 1


def test_return_to_previous_position(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={
            (M1, P1): "POSITION_3",
            (M2, P1): "POSITION_1",
            (M3, P1): "POSITION_3",
        },
    )
    m3 = _row(frame, M3, P1)
    assert m3["prior_games_position_3"] == 1
    assert m3["previous_start_time_position_3"] == T0
    assert m3["days_since_position_3"] == pytest.approx(2.0)
    assert m3["previous_explicit_position"] == "POSITION_1"
    assert m3["days_since_position_1"] == pytest.approx(1.0)


def test_identical_timestamps_are_mutually_blind(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={
            (M1, P1): "POSITION_3",
            (M2, P1): "POSITION_1",
            (M3, P1): "POSITION_5",
        },
    )
    for match_id in (M2, M3):
        row = _row(frame, match_id, P1)
        assert row["prior_games"] == 1
        assert row["prior_games_position_3"] == 1
        assert row["prior_games_position_1"] == 0
        assert row["prior_games_position_5"] == 0
        assert row["previous_explicit_position"] == "POSITION_3"
        assert row["recent_5_games_position_1"] == 0
        assert row["recent_5_games_position_5"] == 0


def test_same_day_later_match_may_use_earlier(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={(M1, P1): "POSITION_4", (M2, P1): "POSITION_4"},
    )
    row = _row(frame, M2, P1)
    assert row["prior_games_position_4"] == 1
    assert row["days_since_position_4"] == pytest.approx(0.25)


def test_later_series_map_sees_earlier_map(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={(M1, P1): "POSITION_2", (M2, P1): "POSITION_2"},
    )
    row = _row(frame, M2, P1)
    assert row["series_id"] == SERIES_1
    assert row["prior_games_position_2"] == 1


def test_history_follows_player_id_across_teams(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={(M1, P1): "POSITION_2", (M2, P1): "POSITION_2"},
    )
    row = _row(frame, M2, P1)
    assert row["team_id"] == TEAM_C
    assert row["prior_games_position_2"] == 1


def test_career_continues_across_versions_same_version_resets(
    tmp_path: Path,
) -> None:
    frame = _assemble(
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
        positions={(M1, P1): "POSITION_5", (M2, P1): "POSITION_5"},
    )
    row = _row(frame, M2, P1)
    assert row["prior_games_position_5"] == 1
    assert row["version_prior_games_position_5"] == 0
    assert pd.isna(row["version_prior_win_rate_position_5"])


def test_null_position_counts_as_match_not_as_position(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={(M1, P1): "POSITION_1", (M2, P1): None, (M3, P1): "POSITION_1"},
    )
    m2 = _row(frame, M2, P1)
    assert pd.isna(m2["position"])
    assert m2["prior_games"] == 1
    assert m2["prior_games_position_1"] == 1
    assert m2["prior_explicit_position_games"] == 1

    m3 = _row(frame, M3, P1)
    assert m3["prior_games"] == 2
    assert m3["prior_games_position_1"] == 1
    assert m3["prior_explicit_position_games"] == 1
    assert m3["prior_share_position_1"] == pytest.approx(1.0)
    assert m3["recent_5_explicit_games"] == 1
    assert m3["recent_5_games_position_1"] == 1


def test_unknown_position_is_not_an_explicit_position(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={(M1, P1): "UNKNOWN", (M2, P1): "POSITION_1"},
    )
    row = _row(frame, M2, P1)
    assert row["prior_games"] == 1
    assert row["prior_explicit_position_games"] == 0
    assert row["prior_games_position_1"] == 0
    assert pd.isna(row["previous_explicit_position"])


def test_recent_windows_exclude_current_and_future_rows(tmp_path: Path) -> None:
    specs = []
    positions: dict[tuple[int, int], str | None] = {}
    # 12 matches so last-10 at the final row is fully populated.
    for i in range(12):
        match_id = 8000 + i
        specs.append(
            {
                "match_id": match_id,
                "start_time": T0 + timedelta(days=i),
                "radiant_win": i % 2 == 0,
                "game_version_id": VERSION_A,
            }
        )
        # First 7 pos 4, last 5 pos 1 (including current).
        positions[(match_id, P1)] = "POSITION_4" if i < 7 else "POSITION_1"

    frame = _assemble(tmp_path, specs, positions=positions, match_id=8011)
    row = _row(frame, 8011, P1)
    assert set(frame["match_id"].unique()) == {8011}
    assert row["position"] == "POSITION_1"
    # Last 5 prior matches are 8006..8010: one pos 4 then four pos 1.
    assert row["recent_5_games_position_4"] == 1
    assert row["recent_5_games_position_1"] == 4
    assert row["recent_5_explicit_games"] == 5
    assert row["recent_5_modal_position"] == "POSITION_1"
    assert row["recent_5_modal_position_share"] == pytest.approx(0.8)
    # Current match 8011's POSITION_1 is not in the window.
    assert row["prior_games_position_1"] == 4
    assert row["recent_10_games_position_4"] == 6
    assert row["recent_10_games_position_1"] == 4
    assert row["recent_position_stability"] == pytest.approx(0.6)


def test_modal_tie_is_null(tmp_path: Path) -> None:
    frame = _assemble(
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
        positions={
            (M1, P1): "POSITION_3",
            (M2, P1): "POSITION_1",
            (M3, P1): "POSITION_2",
        },
    )
    row = _row(frame, M3, P1)
    assert row["prior_games_position_1"] == 1
    assert row["prior_games_position_3"] == 1
    assert pd.isna(row["historical_modal_position"])
    assert pd.isna(row["historical_modal_position_share"])
