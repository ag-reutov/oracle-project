"""Tests for leakage-safe Player × Hero × expected-position state."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from hero_meta_helpers import match_row
from player_hero_position_helpers import (
    assign_positions,
    draft_and_player_rows,
    player_hero_position_frame,
)

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.player_hero_position import (
    PLAYER_HERO_POSITION_COLUMNS,
    PLAYER_HERO_POSITION_METRIC_COLUMNS,
    player_hero_position_sql,
    summarize_player_hero_position,
)
from dota_predictor.features.player_position import EXPLICIT_POSITION_LABELS
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
    return player_hero_position_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        method=method,
        match_id=match_id,
    )


def test_not_in_win_model_or_pre_draft_snapshot() -> None:
    for column in PLAYER_HERO_POSITION_METRIC_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in SNAPSHOT_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
    assert "player_hero_position" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "at_expected_position" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 3
    assert REFERENCE_SCHEMA_VERSION == 1


def test_sql_is_strict_prior_and_does_not_use_expected_or_slot() -> None:
    sql = player_hero_position_sql(catalog_registered=False)
    assert "expected_position" not in sql
    assert STRICT_PRIOR_RANGE_SQL in sql
    assert sql.count("EXCLUDE GROUP") == 5
    assert "ORDER BY match_id" not in sql
    assert "FILTER (WHERE position =" in sql
    for clause in sql.split("PARTITION BY")[1:]:
        header = clause.split("ORDER BY")[0]
        assert "slot_in_side" not in header
        assert "position" not in header


def test_first_match_has_zero_hero_history(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [{"match_id": M1, "start_time": T0, "radiant_win": True}],
        positions=_unique_side_positions(M1, RADIANT)
        | _unique_side_positions(M1, DIRE),
    )
    assert list(frame.columns) == list(PLAYER_HERO_POSITION_COLUMNS)
    row = _row(frame, M1, P1)
    assert row["hero_id"] == 1
    assert row["prior_games_on_hero"] == 0
    assert row["prior_games_on_hero_at_expected_position"] == 0
    assert pd.isna(row["prior_win_rate_on_hero"])
    assert pd.isna(row["prior_win_rate_on_hero_at_expected_position"])


def test_unconditioned_counts_hero_played_at_any_position(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, DIRE)
    positions.update(
        {
            (M1, P1): "POSITION_5",
            (M1, P2): "POSITION_1",
            (M1, P3): "POSITION_2",
            (M1, P4): "POSITION_3",
            (M1, P5): "POSITION_4",
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
        method="previous",
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["expected_position"] == "POSITION_5"
    assert row["hero_id"] == 1
    assert row["prior_games_on_hero"] == 1
    assert row["prior_wins_on_hero"] == 1
    assert row["prior_games_on_hero_at_expected_position"] == 1
    assert row["prior_position_share_on_hero"] == pytest.approx(1.0)


def test_conditioning_ignores_hero_history_at_other_positions(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, DIRE)
    positions.update(
        {
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
    )
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
    assert row["hero_id"] == 1
    assert row["prior_games_on_hero"] == 2
    assert row["prior_games_on_hero_at_expected_position"] == 1
    assert row["prior_wins_on_hero_at_expected_position"] == 0
    assert row["prior_position_share_on_hero"] == pytest.approx(0.5)


def test_current_observed_position_does_not_select_the_bucket(tmp_path: Path) -> None:
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
    assert row["prior_games_on_hero_at_expected_position"] == 1
    assert row["prior_wins_on_hero_at_expected_position"] == 1


def test_null_historical_position_does_not_enter_conditioned_counts(
    tmp_path: Path,
) -> None:
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
        method="previous",
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["prior_games_on_hero"] == 1
    assert row["prior_games_on_hero_at_expected_position"] == 0
    assert pd.isna(row["prior_win_rate_on_hero_at_expected_position"])


def test_timestamp_tie_is_mutually_blind(tmp_path: Path) -> None:
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
        assert row["prior_games_on_hero_at_expected_position"] == 1


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
        method="previous",
        match_id=M2,
    )
    assert _row(frame, M2, P1)["prior_games_on_hero_at_expected_position"] == 1


def test_same_version_resets_conditioned_and_unconditioned(tmp_path: Path) -> None:
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
        method="previous",
        match_id=M2,
    )
    row = _row(frame, M2, P1)
    assert row["prior_games_on_hero"] == 1
    assert row["same_version_games_on_hero"] == 0
    assert row["same_version_games_on_hero_at_expected_position"] == 0
    assert row["prior_games_on_hero_at_expected_position"] == 1


def test_summarize_conditioned_coverage_cannot_exceed_unconditioned() -> None:
    frame = pd.DataFrame(
        {
            "game_version_id": [10, 10, 11],
            "expected_position": ["POSITION_1", "POSITION_1", "POSITION_2"],
            "prior_games_on_hero": [2, 0, 4],
            "prior_games_on_hero_at_expected_position": [1, 0, 0],
            "prior_win_rate_on_hero": [0.5, None, 1.0],
            "prior_win_rate_on_hero_at_expected_position": [0.0, None, None],
        }
    )
    summary = summarize_player_hero_position(frame)
    overall = summary[summary["scope"] == "overall"].iloc[0]
    assert overall["unconditioned_coverage"] == pytest.approx(2 / 3)
    assert overall["conditioned_coverage"] == pytest.approx(1 / 3)
    assert overall["played_hero_not_at_expected_position"] == pytest.approx(1 / 3)
