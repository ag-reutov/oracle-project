"""Tests for leakage-safe meta-relevant Player × Hero state (Slice 6).

Deterministic fixtures with hand-calculated expected values.
Does not go through PRE_DRAFT snapshot SQL, Elo, or training assembly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from hero_meta_helpers import match_row
from player_hero_meta_helpers import (
    assign_positions,
    draft_and_player_rows,
    player_hero_meta_frame,
    write_hero_meta_store,
)

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.duckdb_layer import connect
from dota_predictor.features.expected_position import (
    EXPECTED_POSITION_COLUMNS,
    EXPECTED_POSITION_METHODS,
)
from dota_predictor.features.hero_meta import HERO_META_METRIC_COLUMNS
from dota_predictor.features.hero_state import (
    HERO_STATE_METRIC_COLUMNS,
    build_hero_state,
    hero_state_sql,
)
from dota_predictor.features.player_hero import PLAYER_HERO_METRIC_COLUMNS
from dota_predictor.features.player_hero_meta import (
    PLAYER_HERO_META_COLUMNS,
    PLAYER_HERO_META_METRIC_COLUMNS,
    PREFERRED_HERO_META_WINDOW,
    RECENT_PLAYER_HERO_MATCH_WINDOWS,
    build_player_hero_meta,
    player_hero_meta_sql,
    role_compatibility,
    summarize_player_hero_meta,
)
from dota_predictor.features.player_hero_position import (
    PLAYER_HERO_POSITION_METRIC_COLUMNS,
)
from dota_predictor.features.player_match import PLAYER_STATE_METRIC_COLUMNS
from dota_predictor.features.player_position import (
    EXPLICIT_POSITION_LABELS,
    PLAYER_POSITION_STATE_METRIC_COLUMNS,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.features.temporal import STRICT_PRIOR_RANGE_SQL
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS

T0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
T_TIE = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
T2 = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)

VERSION_A = 10
VERSION_B = 11
M1, M2, M3 = 4001, 1002, 3003
P1, P2, P3, P4, P5 = 1, 2, 3, 4, 5
P6, P7, P8, P9, P10 = 6, 7, 8, 9, 10
P11, P12, P13, P14, P15 = 11, 12, 13, 14, 15
P16, P17, P18, P19, P20 = 16, 17, 18, 19, 20
RADIANT = (P1, P2, P3, P4, P5)
DIRE = (P6, P7, P8, P9, P10)
RADIANT_HEROES = (1, 2, 3, 4, 5)
DIRE_HEROES = (6, 7, 8, 9, 10)
TEAM_A, TEAM_B = 100, 200


def _row(frame: pd.DataFrame, match_id: int, player_id: int) -> pd.Series:
    subset = frame[(frame["match_id"] == match_id) & (frame["player_id"] == player_id)]
    assert len(subset) == 1
    return subset.iloc[0]


def _unique_side_positions(
    match_id: int, players: tuple[int, ...]
) -> dict[tuple[int, int], str]:
    return {
        (match_id, player_id): label
        for player_id, label in zip(players, EXPLICIT_POSITION_LABELS, strict=True)
    }


def _assemble(
    tmp_path: Path,
    specs: list[dict],
    *,
    positions: dict[tuple[int, int], str | None] | None = None,
    method: str = "previous",
    match_id: int | None = None,
    heroes: list[dict] | None = None,
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
                game_version_id=spec.get("game_version_id", VERSION_A),
                radiant_team_id=spec.get("radiant_team_id", TEAM_A),
                dire_team_id=spec.get("dire_team_id", TEAM_B),
            )
        )
        draft_rows, player_rows = draft_and_player_rows(
            spec["match_id"],
            radiant_player_ids=spec.get("radiant_players", RADIANT),
            dire_player_ids=spec.get("dire_players", DIRE),
            radiant_hero_ids=spec.get("radiant_heroes", RADIANT_HEROES),
            dire_hero_ids=spec.get("dire_heroes", DIRE_HEROES),
        )
        drafts.extend(draft_rows)
        players.extend(player_rows)
    if positions:
        assign_positions(players, positions)
    return player_hero_meta_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        method=method,
        heroes=heroes,
        match_id=match_id,
    )


# --- SQL / contract guards ------------------------------------------------


def test_sql_uses_strict_prior_and_never_less_or_equal() -> None:
    sql = player_hero_meta_sql(catalog_registered=False)
    assert "start_time <=" not in sql
    assert STRICT_PRIOR_RANGE_SQL in sql
    assert "EXCLUDE GROUP" in sql
    assert "ORDER BY match_id" not in sql
    assert "expected_position" not in sql


def test_sql_recent_windows_are_player_appearances_not_last_n_on_hero() -> None:
    sql = player_hero_meta_sql(catalog_registered=False)
    assert "list(struct_pack(hero_id := hero_id, was_win := was_win))" in sql
    assert "PARTITION BY player_id" in sql
    assert "PARTITION BY player_id, hero_id" in sql
    assert "PARTITION BY player_id, hero_id, game_version_id" in sql
    for window in RECENT_PLAYER_HERO_MATCH_WINDOWS:
        assert f"GREATEST(len(w.prior_appearances) - {window - 1}, 1)" in sql
    # Trailing lists are over the player window, not a hero-id partition.
    assert "ROWS BETWEEN" not in sql


def test_slice_6_not_in_win_model_or_pre_draft_snapshot() -> None:
    for column in PLAYER_HERO_META_METRIC_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in SNAPSHOT_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
    assert "player_hero_meta" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "player_hero_recent_role_compatibility" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "player_hero_recent_20_matches" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 3
    assert REFERENCE_SCHEMA_VERSION == 1


def test_slices_0_to_5_remain_unchanged() -> None:
    assert PLAYER_STATE_METRIC_COLUMNS == (
        "prior_games",
        "prior_wins",
        "prior_win_rate",
        "previous_match_start_time",
        "days_since_previous_match",
        "prior_unique_heroes",
        "version_prior_games",
        "version_prior_wins",
        "version_prior_win_rate",
        "version_prior_unique_heroes",
    )
    assert PLAYER_HERO_METRIC_COLUMNS == (
        "prior_games_on_hero",
        "prior_wins_on_hero",
        "prior_losses_on_hero",
        "prior_win_rate_on_hero",
        "same_version_games_on_hero",
        "same_version_wins_on_hero",
        "same_version_win_rate_on_hero",
        "recent_90d_games_on_hero",
        "recent_90d_wins_on_hero",
        "recent_90d_win_rate_on_hero",
        "prior_player_games",
        "prior_hero_share",
        "recent_90d_player_games",
        "recent_90d_hero_share",
        "days_since_last_played_hero",
    )
    assert "historical_modal_position" in PLAYER_POSITION_STATE_METRIC_COLUMNS
    assert "expected_position" in EXPECTED_POSITION_COLUMNS
    assert EXPECTED_POSITION_METHODS == (
        "previous",
        "recent_5",
        "recent_10",
        "recent_20",
        "career",
        "same_version",
        "hierarchical",
    )
    assert PLAYER_HERO_POSITION_METRIC_COLUMNS[0] == "prior_games_on_hero"
    assert "prior_games_on_hero_at_expected_position" in PLAYER_HERO_POSITION_METRIC_COLUMNS
    assert HERO_META_METRIC_COLUMNS[0] == "same_version_prior_matches"
    assert HERO_STATE_METRIC_COLUMNS[0] == "hero_prior_matches"
    assert "hero_recent_50_contest_rate" in HERO_STATE_METRIC_COLUMNS
    old_sql = hero_state_sql(catalog_registered=True)
    assert "player_hero_recent_role_compatibility" not in old_sql
    assert PREFERRED_HERO_META_WINDOW == 50


# --- leakage / temporal ---------------------------------------------------


def test_first_match_has_zero_hero_history(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [{"match_id": M1, "start_time": T0, "radiant_win": True}],
        positions=_unique_side_positions(M1, RADIANT)
        | _unique_side_positions(M1, DIRE),
    )
    assert list(frame.columns) == list(PLAYER_HERO_META_COLUMNS)
    row = _row(frame, M1, P1)
    assert row["hero_id"] == 1
    assert row["prior_games_on_hero"] == 0
    assert row["player_hero_recent_20_matches"] == 0
    assert row["player_hero_same_version_matches"] == 0
    assert row["player_hero_position_explicit_games"] == 0
    assert pd.isna(row["prior_win_rate_on_hero"])
    assert pd.isna(row["player_hero_recent_20_win_rate"])
    assert pd.isna(row["player_hero_same_version_win_rate"])
    assert pd.isna(row["player_hero_position_1_share"])
    assert pd.isna(row["player_hero_recent_role_compatibility"])


def test_strictly_earlier_rows_only(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        _unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE)
    )
    positions.update(
        _unique_side_positions(M3, RADIANT) | _unique_side_positions(M3, DIRE)
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
            {"match_id": M3, "start_time": T2, "radiant_win": True},
        ],
        positions=positions,
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["prior_games_on_hero"] == 1
    assert row["prior_wins_on_hero"] == 1
    assert row["prior_win_rate_on_hero"] == pytest.approx(1.0)
    assert row["player_hero_recent_20_matches"] == 1
    assert row["player_hero_recent_20_wins"] == 1


def test_identical_timestamps_are_mutually_blind(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        {
            (M2, P1): "POSITION_1",
            (M3, P1): "POSITION_1",
        }
    )
    positions.update(
        {
            (M2, pid): label
            for pid, label in zip((P11, P12, P13, P14, P15), EXPLICIT_POSITION_LABELS)
        }
    )
    positions.update(
        {
            (M3, pid): label
            for pid, label in zip((P16, P17, P18, P19, P20), EXPLICIT_POSITION_LABELS)
        }
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {
                "match_id": M2,
                "start_time": T_TIE,
                "radiant_win": True,
                "dire_players": (P11, P12, P13, P14, P15),
                "dire_heroes": (16, 17, 18, 19, 20),
            },
            {
                "match_id": M3,
                "start_time": T_TIE,
                "radiant_win": False,
                "dire_players": (P16, P17, P18, P19, P20),
                "dire_heroes": (16, 17, 18, 19, 20),
            },
        ],
        positions=positions,
        method="previous",
    )
    for match_id in (M2, M3):
        row = _row(frame, match_id, P1)
        assert row["prior_games_on_hero"] == 1
        assert row["player_hero_recent_20_matches"] == 1
        assert row["player_hero_same_version_matches"] == 1


def test_later_series_map_may_use_earlier_map(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        _unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE)
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["prior_games_on_hero"] == 1
    assert row["player_hero_recent_20_matches"] == 1
    assert row["prior_wins_on_hero"] == 1


def test_current_result_never_enters_current_row(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        _unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE)
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": True},
        ],
        positions=positions,
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["prior_games_on_hero"] == 1
    assert row["prior_wins_on_hero"] == 1
    assert row["player_hero_recent_20_wins"] == 1
    assert row["prior_win_rate_on_hero"] == pytest.approx(1.0)


def test_current_observed_position_never_enters_shares(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        {
            (M2, P1): "POSITION_5",
            (M2, P2): "POSITION_2",
            (M2, P3): "POSITION_3",
            (M2, P4): "POSITION_4",
            (M2, P5): "POSITION_1",
        }
    )
    positions.update(_unique_side_positions(M2, DIRE))
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        method="previous",
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["observed_position"] == "POSITION_5"
    assert row["expected_position"] == "POSITION_1"
    assert row["player_hero_position_1_share"] == pytest.approx(1.0)
    assert row["player_hero_position_5_share"] == pytest.approx(0.0)
    assert row["player_hero_share_at_expected_position"] == pytest.approx(1.0)


def test_historical_observed_positions_populate_shares(tmp_path: Path) -> None:
    positions = {
        (M1, P1): "POSITION_1",
        (M1, P2): "POSITION_2",
        (M1, P3): "POSITION_3",
        (M1, P4): "POSITION_4",
        (M1, P5): "POSITION_5",
        (M2, P1): "POSITION_5",
        (M2, P2): "POSITION_1",
        (M2, P3): "POSITION_2",
        (M2, P4): "POSITION_3",
        (M2, P5): "POSITION_4",
    }
    positions.update(_unique_side_positions(M1, DIRE))
    positions.update(_unique_side_positions(M2, DIRE))
    positions.update(
        _unique_side_positions(M3, RADIANT) | _unique_side_positions(M3, DIRE)
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
            {"match_id": M3, "start_time": T2, "radiant_win": True},
        ],
        positions=positions,
        method="previous",
        match_id=M3,
    )
    row = _row(frame, M3, P1)
    assert row["prior_games_on_hero"] == 2
    assert row["player_hero_position_explicit_games"] == 2
    assert row["player_hero_position_1_share"] == pytest.approx(0.5)
    assert row["player_hero_position_5_share"] == pytest.approx(0.5)
    assert row["expected_position"] == "POSITION_5"
    assert row["player_hero_share_at_expected_position"] == pytest.approx(0.5)


def test_historical_null_positions_do_not_populate_shares(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, DIRE)
    positions.update(
        {
            (M1, P1): None,
            (M1, P2): "POSITION_2",
            (M1, P3): "POSITION_3",
            (M1, P4): "POSITION_4",
            (M1, P5): "POSITION_5",
        }
    )
    positions.update(
        _unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE)
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["prior_games_on_hero"] == 1
    assert row["player_hero_position_explicit_games"] == 0
    assert pd.isna(row["player_hero_position_1_share"])
    assert pd.isna(row["player_hero_share_at_expected_position"])
    assert pd.isna(row["player_hero_recent_role_compatibility"])


def test_unknown_historical_position_does_not_populate_shares(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, DIRE)
    positions.update(
        {
            (M1, P1): "UNKNOWN",
            (M1, P2): "POSITION_2",
            (M1, P3): "POSITION_3",
            (M1, P4): "POSITION_4",
            (M1, P5): "POSITION_5",
        }
    )
    positions.update(
        _unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE)
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["player_hero_position_explicit_games"] == 0
    assert pd.isna(row["player_hero_position_1_share"])


# --- recent / same-version ------------------------------------------------


def test_recent_window_counts_hero_games_inside_player_appearances(
    tmp_path: Path,
) -> None:
    """Career on this hero can exceed the trailing-20 player-appearance count."""
    n_prior = 21
    specs = []
    positions: dict[tuple[int, int], str | None] = {}
    for i in range(n_prior + 1):
        match_id = 10 + i
        start = T0 + timedelta(days=i)
        on_eval_hero = i == 0 or i == n_prior
        specs.append(
            {
                "match_id": match_id,
                "start_time": start,
                "radiant_win": True,
                "radiant_heroes": RADIANT_HEROES if on_eval_hero else (11, 2, 3, 4, 5),
            }
        )
        positions.update(
            _unique_side_positions(match_id, RADIANT)
            | _unique_side_positions(match_id, DIRE)
        )
    eval_id = 10 + n_prior
    frame = _assemble(tmp_path, specs, positions=positions, match_id=eval_id)
    row = _row(frame, eval_id, P1)
    assert row["hero_id"] == 1
    assert row["prior_games_on_hero"] == 1
    assert row["prior_player_games"] == n_prior
    assert row["player_hero_recent_20_matches"] == 0
    assert pd.isna(row["player_hero_recent_20_win_rate"])
    assert row["player_hero_recent_50_matches"] == 1
    assert row["player_hero_recent_50_wins"] == 1
    assert row["player_hero_recent_50_win_rate"] == pytest.approx(1.0)
    assert row["player_hero_recent_100_matches"] == 1


def test_zero_recent_games_yield_null_win_rate(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        _unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE)
    )
    frame = _assemble(
        tmp_path,
        [
            {
                "match_id": M1,
                "start_time": T0,
                "radiant_win": True,
                "radiant_heroes": (11, 2, 3, 4, 5),
            },
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["hero_id"] == 1
    assert row["prior_games_on_hero"] == 0
    assert row["player_hero_recent_20_matches"] == 0
    assert pd.isna(row["player_hero_recent_20_win_rate"])
    assert pd.isna(row["prior_win_rate_on_hero"])


def test_same_version_resets_at_version_boundary(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        _unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE)
    )
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
        positions=positions,
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["prior_games_on_hero"] == 1
    assert row["player_hero_same_version_matches"] == 0
    assert row["player_hero_same_version_wins"] == 0
    assert pd.isna(row["player_hero_same_version_win_rate"])
    assert pd.isna(row["player_hero_same_version_role_compatibility"])
    assert row["player_hero_recent_20_matches"] == 1


def test_no_future_same_version_matches_leak_backward(tmp_path: Path) -> None:
    t3 = T0 + timedelta(days=2)
    t4 = T0 + timedelta(days=3)
    positions: dict[tuple[int, int], str | None] = {}
    specs = []
    for match_id, start, version in (
        (M1, T0, VERSION_A),
        (M2, T1, VERSION_B),
        (M3, t3, VERSION_B),
        (4004, t4, VERSION_B),
    ):
        specs.append(
            {
                "match_id": match_id,
                "start_time": start,
                "radiant_win": True,
                "game_version_id": version,
            }
        )
        positions.update(
            _unique_side_positions(match_id, RADIANT)
            | _unique_side_positions(match_id, DIRE)
        )
    frame = _assemble(tmp_path, specs, positions=positions, match_id=M2)
    row = _row(frame, M2, P1)
    assert row["player_hero_same_version_matches"] == 0
    frame_later = _assemble(tmp_path, specs, positions=positions, match_id=M3)
    later = _row(frame_later, M3, P1)
    assert later["player_hero_same_version_matches"] == 1
    assert later["prior_games_on_hero"] == 2


# --- expected position / compatibility ------------------------------------


def test_expected_position_selects_current_role_share(tmp_path: Path) -> None:
    positions = {
        (M1, P1): "POSITION_1",
        (M1, P2): "POSITION_2",
        (M1, P3): "POSITION_3",
        (M1, P4): "POSITION_4",
        (M1, P5): "POSITION_5",
        (M2, P1): "POSITION_5",
        (M2, P2): "POSITION_2",
        (M2, P3): "POSITION_3",
        (M2, P4): "POSITION_4",
        (M2, P5): "POSITION_1",
    }
    positions.update(_unique_side_positions(M1, DIRE))
    positions.update(_unique_side_positions(M2, DIRE))
    positions.update(
        _unique_side_positions(M3, RADIANT) | _unique_side_positions(M3, DIRE)
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
            {"match_id": M3, "start_time": T2, "radiant_win": True},
        ],
        positions=positions,
        method="previous",
        match_id=M3,
    )
    row = _row(frame, M3, P1)
    assert row["expected_position"] == "POSITION_5"
    assert row["player_hero_share_at_expected_position"] == pytest.approx(0.5)
    assert row["hero_position_share_at_expected_position"] == pytest.approx(
        row["hero_meta_share_at_expected_position"]
    )
    # Hero 1 was played once at pos 1 (M1) and once at pos 5 (M2) before M3.
    assert row["hero_recent_50_position_5_share"] == pytest.approx(0.5)
    assert row["hero_position_share_at_expected_position"] == pytest.approx(0.5)


def test_role_compatibility_on_controlled_fixture(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        _unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE)
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        method="previous",
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["player_hero_position_1_share"] == pytest.approx(1.0)
    assert row["hero_recent_50_position_1_share"] == pytest.approx(1.0)
    assert row["player_hero_recent_role_compatibility"] == pytest.approx(1.0)
    assert row["player_hero_same_version_role_compatibility"] == pytest.approx(1.0)

    player = pd.DataFrame(
        {
            "player_hero_position_1_share": [1.0, 1.0, 0.5],
            "player_hero_position_2_share": [0.0, 0.0, 0.0],
            "player_hero_position_3_share": [0.0, 0.0, 0.0],
            "player_hero_position_4_share": [0.0, 0.0, 0.0],
            "player_hero_position_5_share": [0.0, 0.0, 0.5],
        }
    )
    meta_aligned = pd.DataFrame(
        {
            "hero_recent_50_position_1_share": [1.0, 0.0, 1.0],
            "hero_recent_50_position_2_share": [0.0, 0.0, 0.0],
            "hero_recent_50_position_3_share": [0.0, 0.0, 0.0],
            "hero_recent_50_position_4_share": [0.0, 0.0, 0.0],
            "hero_recent_50_position_5_share": [0.0, 1.0, 0.0],
        }
    )
    values = role_compatibility(
        player,
        meta_aligned,
        player_explicit=pd.Series([4, 4, 4]),
        meta_explicit=pd.Series([8, 8, 8]),
    )
    assert values.tolist() == pytest.approx([1.0, 0.0, 0.5])


def test_compatibility_null_when_either_distribution_has_no_evidence() -> None:
    player = pd.DataFrame(
        {f"p{p}": [1.0, float("nan")] for p in range(5)}
    )
    meta = pd.DataFrame(
        {f"m{p}": [float("nan"), 1.0] for p in range(5)}
    )
    values = role_compatibility(
        player,
        meta,
        player_explicit=pd.Series([3, 0]),
        meta_explicit=pd.Series([0, 5]),
    )
    assert pd.isna(values.iloc[0])
    assert pd.isna(values.iloc[1])


def test_current_hero_meta_uses_only_slice_5_historical_state(
    tmp_path: Path,
) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        _unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE)
    )
    matches = [
        match_row(M1, start_time=T0, radiant_win=True, game_version_id=VERSION_A),
        match_row(M2, start_time=T1, radiant_win=False, game_version_id=VERSION_A),
    ]
    drafts: list[dict] = []
    players: list[dict] = []
    for spec_id in (M1, M2):
        draft_rows, player_rows = draft_and_player_rows(
            spec_id,
            radiant_player_ids=RADIANT,
            dire_player_ids=DIRE,
            radiant_hero_ids=RADIANT_HEROES,
            dire_hero_ids=DIRE_HEROES,
        )
        drafts.extend(draft_rows)
        players.extend(player_rows)
    assign_positions(players, positions)
    feature_config, _ref = write_hero_meta_store(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=None,
        game_versions=None,
    )
    with connect(feature_config) as store:
        frame = build_player_hero_meta(store, method="previous", match_id=M2).to_frame()
        hero = build_hero_state(store, match_id=M2).to_frame()
    hero1 = hero[(hero["match_id"] == M2) & (hero["hero_id"] == 1)].iloc[0]
    row = _row(frame, M2, P1)
    assert row["hero_recent_50_contest_rate"] == pytest.approx(
        hero1["hero_recent_50_contest_rate"]
    )
    assert row["hero_recent_50_pick_rate"] == pytest.approx(hero1["hero_recent_50_pick_rate"])
    assert row["hero_same_version_contest_rate"] == pytest.approx(
        hero1["hero_same_version_contest_rate"]
    )
    assert hero1["hero_pick_count"] == 1
    assert hero1["hero_recent_50_pick_count"] == 1


def test_summarize_reports_career_and_compatibility_coverage() -> None:
    frame = pd.DataFrame(
        {
            "game_version_id": [10, 10, 11],
            "expected_position": ["POSITION_1", "POSITION_1", "POSITION_2"],
            "prior_games_on_hero": [4, 0, 20],
            "player_hero_recent_20_matches": [1, 0, 0],
            "player_hero_recent_50_matches": [2, 0, 3],
            "player_hero_recent_100_matches": [4, 0, 8],
            "player_hero_same_version_matches": [1, 0, 0],
            "player_hero_position_explicit_games": [3, 0, 10],
            "hero_recent_50_position_explicit_count": [8, 0, 12],
            "player_hero_recent_role_compatibility": [0.8, None, 0.1],
        }
    )
    summary = summarize_player_hero_meta(frame)
    overall = summary[summary["scope"] == "overall"].iloc[0]
    assert overall["career_coverage"] == pytest.approx(2 / 3)
    assert overall["recent_20_coverage"] == pytest.approx(1 / 3)
    assert overall["same_version_coverage"] == pytest.approx(1 / 3)
    assert overall["role_compatibility_coverage"] == pytest.approx(2 / 3)
