"""Tests for leakage-safe Elo-adjusted Player × Hero state (Slice 10).

Does not go through PRE_DRAFT snapshot SQL or training assembly.
Slice 10 is not a production feature.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from player_hero_elo_helpers import (
    draft_and_player_rows,
    match_row,
    player_hero_elo_frame,
)

from dota_predictor.features.player_hero_elo import (
    DEFAULT_SHRINKAGE_K,
    EVIDENCE_METRIC_COLUMNS,
    PLAYER_HERO_ELO_METRIC_COLUMNS,
    STRENGTH_METRIC_COLUMNS,
    player_hero_elo_sql,
    shrinkage_weight,
    shrunk_residual,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.features.team_elo import EloConfig, expected_score
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    PLAYER_HERO_COMPARISON_COLUMNS,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_FROZEN_SPECS,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)
T1 = datetime(2024, 1, 2, tzinfo=UTC)
T2 = datetime(2024, 1, 3, tzinfo=UTC)

VERSION_A = 10
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
    shrinkage_k: float | None = None,
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
    return player_hero_elo_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        match_id=match_id,
        shrinkage_k=shrinkage_k,
    )


def _three_match_specs() -> list[dict]:
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


def test_sql_uses_window_range_exclude_group_not_self_join() -> None:
    sql = player_hero_elo_sql(catalog_registered=True)
    assert "start_time <=" not in sql
    assert sql.count("EXCLUDE GROUP") == 1
    assert "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in sql
    assert "h.start_time < c.start_time" not in sql


def test_sql_partitions_by_player_hero_not_slot() -> None:
    sql = player_hero_elo_sql(catalog_registered=True)
    assert "PARTITION BY player_id, hero_id" in sql
    for clause in sql.split("PARTITION BY")[1:]:
        assert "slot_in_side" not in clause.split("ORDER BY")[0]


def test_sql_never_orders_history_by_match_id() -> None:
    sql = player_hero_elo_sql(catalog_registered=True)
    assert "ORDER BY start_time" in sql
    assert "ORDER BY match_id" not in sql
    assert "ORDER BY start_time, match_id" not in sql


def test_slice10_is_not_part_of_training_or_pre_draft_snapshot() -> None:
    assert set(PLAYER_HERO_ELO_METRIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(PLAYER_HERO_ELO_METRIC_COLUMNS).isdisjoint(SNAPSHOT_COLUMNS)
    assert set(PLAYER_HERO_ELO_METRIC_COLUMNS).isdisjoint(ALL_FEATURE_COLUMNS)
    assert set(PLAYER_HERO_ELO_METRIC_COLUMNS).isdisjoint(ELO_PLUS_PLAYER_HERO_COLUMNS)
    assert set(PLAYER_HERO_ELO_METRIC_COLUMNS).isdisjoint(PLAYER_HERO_COMPARISON_COLUMNS)
    assert "player_hero_elo" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "shrunk_outcome_residual" not in PRE_DRAFT_SNAPSHOT_SQL
    for spec in SLICE9_FROZEN_SPECS:
        assert set(PLAYER_HERO_ELO_METRIC_COLUMNS).isdisjoint(spec.feature_columns)
    assert set(PLAYER_HERO_ELO_METRIC_COLUMNS).isdisjoint(
        SLICE9_CANDIDATE_SPEC.feature_columns
    )


def test_volume_is_evidence_not_strength() -> None:
    assert "prior_games_on_hero" in EVIDENCE_METRIC_COLUMNS
    assert "prior_games_on_hero" not in STRENGTH_METRIC_COLUMNS
    assert "shrinkage_weight_on_hero" in EVIDENCE_METRIC_COLUMNS
    assert "shrunk_outcome_residual_on_hero" in STRENGTH_METRIC_COLUMNS


def test_shrinkage_formula_and_cold_start() -> None:
    assert DEFAULT_SHRINKAGE_K == 40.0
    assert shrinkage_weight(0) == 0.0
    assert shrinkage_weight(40) == pytest.approx(0.5)
    assert shrunk_residual(None, 0) == 0.0
    assert shrunk_residual(0.20, 0) == 0.0
    assert shrunk_residual(0.20, 40) == pytest.approx(0.10)
    assert shrunk_residual(float("nan"), 10) == 0.0


def test_current_match_excluded_from_its_own_residual(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    p1 = _row(frame, M2, P1)
    assert p1["hero_id"] == 1
    assert p1["prior_games_on_hero"] == 1
    assert p1["prior_wins_on_hero"] == 1.0
    # M1: equal 1500 Elo, Radiant win. Residual contribution is 1 - 0.5.
    assert p1["prior_elo_expected_wins_on_hero"] == pytest.approx(0.5)
    assert p1["prior_wins_minus_expected_on_hero"] == pytest.approx(0.5)
    assert p1["mean_outcome_residual_on_hero"] == pytest.approx(0.5)
    assert p1["shrunk_outcome_residual_on_hero"] == pytest.approx(
        (1.0 / (1.0 + DEFAULT_SHRINKAGE_K)) * 0.5
    )


def test_future_matches_excluded(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    p2 = _row(frame, M2, P2)
    assert p2["hero_id"] == 11
    assert p2["prior_games_on_hero"] == 0
    assert pd.isna(p2["mean_outcome_residual_on_hero"])
    assert p2["shrunk_outcome_residual_on_hero"] == 0.0
    assert p2["shrinkage_weight_on_hero"] == 0.0


def test_n_zero_mean_is_null_and_shrunk_is_zero(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M1)
    p1 = _row(frame, M1, P1)
    assert p1["prior_games_on_hero"] == 0
    assert p1["prior_wins_on_hero"] == 0.0
    assert p1["prior_elo_expected_wins_on_hero"] == 0.0
    assert p1["prior_wins_minus_expected_on_hero"] == 0.0
    assert pd.isna(p1["mean_outcome_residual_on_hero"])
    assert p1["shrunk_outcome_residual_on_hero"] == 0.0


def test_textbook_elo_expected_and_post_match_rating(tmp_path: Path) -> None:
    """M1: 1500 vs 1500, expected 0.5, Radiant win → 1516 / 1484.

    M2 (same teams, later): P1's prior residual uses M1's pre-match 0.5,
    never M2's own expected win or M1's post-match rating as a feature.
    """
    config = EloConfig()
    gain = config.k_factor * 0.5
    assert gain == pytest.approx(16.0)
    m1_expected = expected_score(1500.0, 1500.0)
    assert m1_expected == pytest.approx(0.5)
    m2_expected = expected_score(1500.0 + gain, 1500.0 - gain)
    assert m2_expected == pytest.approx(expected_score(1516.0, 1484.0))

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
            },
        ],
    )
    p1_m1 = _row(frame, M1, P1)
    p1_m2 = _row(frame, M2, P1)
    assert p1_m1["prior_games_on_hero"] == 0
    assert pd.isna(p1_m1["mean_outcome_residual_on_hero"])
    assert p1_m2["prior_games_on_hero"] == 1
    assert p1_m2["prior_elo_expected_wins_on_hero"] == pytest.approx(m1_expected)
    assert p1_m2["mean_outcome_residual_on_hero"] == pytest.approx(1.0 - m1_expected)
    # M2's own (higher) expected win must not enter P1's M2 prior sum.
    assert p1_m2["prior_elo_expected_wins_on_hero"] != pytest.approx(m2_expected)


def test_identical_timestamps_are_mutually_blind_for_residual_and_elo(
    tmp_path: Path,
) -> None:
    """Same-timestamp peers must not contribute P×H history or Elo updates.

    P1 plays hero 1 at T0 (Radiant win, 1500 vs 1500). At T1, P1 appears
    in both M2 and M3 on hero 1 with opposite outcomes. Each T1 row may
    see T0 only. Both T1 Elo expected wins share the post-T0 ratings.
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
            "dire_team_id": TEAM_C,
        },
        {
            "match_id": M3,
            "start_time": same_time,
            "radiant_win": False,
            "game_version_id": VERSION_A,
            "dire_players": (P16, P17, P18, P19, P20),
            "dire_heroes": (16, 17, 18, 19, 20),
            "dire_team_id": TEAM_D,
        },
    ]
    frame = _assemble(tmp_path, specs)
    for match_id in (M2, M3):
        p1 = _row(frame, match_id, P1)
        assert p1["prior_games_on_hero"] == 1
        assert p1["prior_wins_on_hero"] == 1.0
        assert p1["prior_elo_expected_wins_on_hero"] == pytest.approx(0.5)
        assert p1["mean_outcome_residual_on_hero"] == pytest.approx(0.5)

    p1_at_m1 = _row(frame, M1, P1)
    assert p1_at_m1["prior_games_on_hero"] == 0
    assert pd.isna(p1_at_m1["mean_outcome_residual_on_hero"])
    assert p1_at_m1["shrunk_outcome_residual_on_hero"] == 0.0
