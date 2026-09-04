"""Tests for the Radiant − Dire draft comparison (`features.draft_comparison`).

Small deterministic fixtures with hand-calculated expected values.
Does not go through PRE_DRAFT snapshot SQL, Elo, or training assembly.
`slot_in_side` is lobby order only and is never treated as position 1-5.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from draft_comparison_helpers import (
    draft_comparison_frame,
    draft_comparison_layers,
)
from draft_profile_helpers import draft_and_player_rows, match_row

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.draft_comparison import (
    DRAFT_COMPARISON_COLUMNS,
    DRAFT_COMPARISON_METRIC_COLUMNS,
    draft_comparison_from_profile,
    draft_comparison_sql,
)
from dota_predictor.features.draft_profile import (
    DRAFT_PROFILE_METRIC_COLUMNS,
    draft_profile_sql,
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

UNDESIRABLE_COUNT_METRICS = (
    "players_with_zero_prior_games_on_hero",
    "players_with_zero_recent_90d_games_on_hero",
    "heroes_never_played_by_team",
    "heroes_not_played_by_team_recent_90d",
)

RATE_METRICS = (
    "mean_player_prior_hero_share",
    "mean_player_recent_90d_hero_share",
    "mean_team_hero_share",
    "mean_team_recent_90d_hero_share",
    "mean_same_version_contest_rate",
    "min_same_version_contest_rate",
    "mean_recent_90d_contest_rate",
    "min_recent_90d_contest_rate",
    "mean_same_version_pick_rate",
    "mean_same_version_ban_rate",
    "mean_recent_90d_pick_rate",
    "mean_recent_90d_ban_rate",
)

SAME_VERSION_META_METRICS = (
    "mean_same_version_contest_rate",
    "min_same_version_contest_rate",
    "mean_same_version_pick_rate",
    "mean_same_version_ban_rate",
)

CATALOG_HEROES = [
    {"id": hero_id, "displayName": f"Hero {hero_id}"}
    for hero_id in range(1, 23)
]


def _row(frame: pd.DataFrame, match_id: int) -> pd.Series:
    subset = frame[frame["match_id"] == match_id]
    assert len(subset) == 1, (
        f"expected one comparison row for {match_id}, got {len(subset)}"
    )
    return subset.iloc[0]


def _side(profile: pd.DataFrame, match_id: int, side: str) -> pd.Series:
    subset = profile[(profile["match_id"] == match_id) & (profile["side"] == side)]
    assert len(subset) == 1, (
        f"expected one profile row for ({match_id}, {side}), got {len(subset)}"
    )
    return subset.iloc[0]


def _assemble(
    tmp_path: Path,
    specs: list[dict],
    *,
    heroes: list[dict] | None = CATALOG_HEROES,
    match_id: int | None = None,
    layers: bool = False,
):
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
    kwargs = {
        "tmp_path": tmp_path,
        "matches": matches,
        "players": players,
        "drafts": drafts,
        "heroes": heroes,
        "match_id": match_id,
    }
    if layers:
        return draft_comparison_layers(**kwargs)
    return draft_comparison_frame(**kwargs)


def _three_match_specs(*, m3_version: int = VERSION_A) -> list[dict]:
    """Two historical maps plus one evaluation map.

    M1 (T0, Radiant win): default heroes.
    M2 (T1, Dire win): Radiant replaces hero 2 with 11.
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
        },
        {
            "match_id": M3,
            "start_time": T2,
            "radiant_win": True,
            "game_version_id": m3_version,
            "radiant_heroes": (1, 11, 3, 4, 5),
        },
    ]


def _manual_diff(radiant: pd.Series, dire: pd.Series, metric: str):
    left = radiant[metric]
    right = dire[metric]
    if pd.isna(left) or pd.isna(right):
        return None
    return left - right


# --- SQL / contract guards ------------------------------------------------


def test_sql_wraps_profile_once_and_subtracts_radiant_minus_dire() -> None:
    sql = draft_comparison_sql(catalog_registered=True)
    profile_sql = draft_profile_sql(catalog_registered=True)
    assert "AS MATERIALIZED" in sql
    assert sql.count(profile_sql.strip()) == 1
    outer = sql.split("AS profile_inner")[-1]
    assert "COALESCE" not in outer
    assert "AVG(" not in outer
    assert "SUM(" not in outer
    assert "EXCLUDE GROUP" not in outer
    assert "RANGE " not in outer
    assert "radiant_win" not in outer
    for metric in DRAFT_PROFILE_METRIC_COLUMNS:
        assert f"radiant.{metric} - dire.{metric}" in outer
        assert f"AS {metric}_diff" in outer


def test_sql_does_not_add_history_or_judgment() -> None:
    sql = draft_comparison_sql(catalog_registered=True)
    assert sql.count("EXCLUDE GROUP") == draft_profile_sql().count("EXCLUDE GROUP")
    assert sql.count("radiant_win") == draft_profile_sql().count("radiant_win")
    outer = sql.split("AS profile_inner")[-1].lower()
    for forbidden in ("position", "lane", "role", "synergy", "counter", "elo", "score"):
        assert forbidden not in outer


def test_comparison_is_not_part_of_training_or_pre_draft_snapshot() -> None:
    assert set(DRAFT_COMPARISON_METRIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(DRAFT_COMPARISON_METRIC_COLUMNS).isdisjoint(SNAPSHOT_COLUMNS)
    assert set(DRAFT_COMPARISON_METRIC_COLUMNS).isdisjoint(ALL_FEATURE_COLUMNS)
    assert "draft_comparison" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "mean_player_prior_games_on_hero_diff" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "radiant_win" not in DRAFT_COMPARISON_COLUMNS


def test_schema_versions_unchanged_by_this_layer() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 5
    assert REFERENCE_SCHEMA_VERSION == 2


def test_metric_columns_are_exactly_profile_metrics_with_diff_suffix() -> None:
    expected = tuple(f"{metric}_diff" for metric in DRAFT_PROFILE_METRIC_COLUMNS)
    assert DRAFT_COMPARISON_METRIC_COLUMNS == expected
    assert "radiant_win" not in DRAFT_COMPARISON_COLUMNS
    assert "side" not in DRAFT_COMPARISON_COLUMNS


# --- grain and context ----------------------------------------------------


def test_exactly_one_row_per_match(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs())
    assert list(frame.columns) == list(DRAFT_COMPARISON_COLUMNS)
    assert set(frame["match_id"]) == {M1, M2, M3}
    assert len(frame) == 3
    assert frame["match_id"].is_unique
    assert "radiant_win" not in frame.columns
    assert "side" not in frame.columns


def test_team_ids_and_match_context_are_carried_through(tmp_path: Path) -> None:
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
            "dire_players": RADIANT_PLAYERS,
            "radiant_heroes": (11, 12, 13, 14, 15),
            "dire_heroes": RADIANT_HEROES,
        },
    ]
    comparison, profile = _assemble(tmp_path, specs, layers=True)
    row = _row(comparison, M2)
    radiant = _side(profile, M2, "RADIANT")
    dire = _side(profile, M2, "DIRE")
    assert row["radiant_team_id"] == TEAM_C
    assert row["dire_team_id"] == TEAM_D
    assert row["radiant_team_id"] == radiant["team_id"]
    assert row["dire_team_id"] == dire["team_id"]
    assert row["start_time"] == radiant["start_time"]
    assert row["game_version_id"] == VERSION_A
    assert "radiant_win" not in comparison.columns


# --- orientation ----------------------------------------------------------


def test_positive_diff_when_radiant_is_higher(tmp_path: Path) -> None:
    """Dire drafts one new hero at M2; Radiant repeats M1. Games 1.0 − 0.8."""
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
            "dire_heroes": (6, 11, 8, 9, 10),
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    row = _row(frame, M2)
    assert row["mean_player_prior_games_on_hero_diff"] == pytest.approx(0.2)
    assert row["mean_player_prior_games_on_hero_diff"] > 0
    assert row["players_with_zero_prior_games_on_hero_diff"] == -1


def test_negative_diff_when_dire_is_higher(tmp_path: Path) -> None:
    """Radiant drafts one new hero at M2; Dire repeats M1. Games 0.8 − 1.0."""
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    row = _row(frame, M2)
    assert row["mean_player_prior_games_on_hero_diff"] == pytest.approx(-0.2)
    assert row["mean_player_prior_games_on_hero_diff"] < 0
    assert row["mean_same_version_contest_rate_diff"] == pytest.approx(-0.2)


def test_zero_diff_when_sides_match(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M1)
    row = _row(frame, M1)
    assert row["mean_player_prior_games_on_hero_diff"] == pytest.approx(0.0)
    assert row["min_player_prior_games_on_hero_diff"] == 0
    assert row["players_with_zero_prior_games_on_hero_diff"] == 0
    assert row["heroes_never_played_by_team_diff"] == 0
    assert row["mean_team_prior_games_with_hero_diff"] == pytest.approx(0.0)


def test_undesirable_count_fields_are_not_sign_flipped(tmp_path: Path) -> None:
    """Radiant has one zero-history hero at M2; Dire has none.

    Raw Radiant − Dire is +1. Flipping 'bad is negative' would yield -1.
    """
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    row = _row(frame, M2)
    for metric in UNDESIRABLE_COUNT_METRICS:
        assert row[f"{metric}_diff"] == 1, metric


# --- NULL semantics -------------------------------------------------------


def test_null_on_one_side_yields_null_diff(tmp_path: Path) -> None:
    """Radiant repeats M1 (shares defined); Dire is a brand-new roster."""
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
            "dire_team_id": TEAM_C,
            "dire_players": (P11, P12, P13, P14, P15),
            "dire_heroes": (11, 12, 13, 14, 15),
        },
    ]
    comparison, profile = _assemble(tmp_path, specs, match_id=M2, layers=True)
    radiant = _side(profile, M2, "RADIANT")
    dire = _side(profile, M2, "DIRE")
    row = _row(comparison, M2)
    assert pd.notna(radiant["mean_player_prior_hero_share"])
    assert pd.isna(dire["mean_player_prior_hero_share"])
    assert pd.isna(row["mean_player_prior_hero_share_diff"])
    assert pd.notna(radiant["mean_team_hero_share"])
    assert pd.isna(dire["mean_team_hero_share"])
    assert pd.isna(row["mean_team_hero_share_diff"])
    assert row["mean_player_prior_games_on_hero_diff"] == pytest.approx(1.0)


def test_null_on_both_sides_yields_null_diff(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M1)
    row = _row(frame, M1)
    for metric in RATE_METRICS:
        assert pd.isna(row[f"{metric}_diff"]), metric


def test_patch_opener_same_version_meta_diffs_remain_null(tmp_path: Path) -> None:
    comparison, profile = _assemble(
        tmp_path,
        _three_match_specs(m3_version=VERSION_B),
        match_id=M3,
        layers=True,
    )
    row = _row(comparison, M3)
    radiant = _side(profile, M3, "RADIANT")
    dire = _side(profile, M3, "DIRE")
    for metric in SAME_VERSION_META_METRICS:
        assert pd.isna(radiant[metric]), metric
        assert pd.isna(dire[metric]), metric
        assert pd.isna(row[f"{metric}_diff"]), metric
    assert pd.notna(row["mean_recent_90d_contest_rate_diff"])
    assert row["mean_recent_90d_contest_rate_diff"] == pytest.approx(
        radiant["mean_recent_90d_contest_rate"] - dire["mean_recent_90d_contest_rate"]
    )


# --- equivalence and swap -------------------------------------------------


def test_equivalence_to_manual_subtraction_of_profile_rows(tmp_path: Path) -> None:
    comparison, profile = _assemble(tmp_path, _three_match_specs(), layers=True)
    from_profile = draft_comparison_from_profile(profile)
    left = comparison.sort_values("match_id").reset_index(drop=True)
    right = from_profile.sort_values("match_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left[list(DRAFT_COMPARISON_COLUMNS)],
        right[list(DRAFT_COMPARISON_COLUMNS)],
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    for match_id in (M1, M2, M3):
        row = _row(comparison, match_id)
        radiant = _side(profile, match_id, "RADIANT")
        dire = _side(profile, match_id, "DIRE")
        assert row["radiant_team_id"] == radiant["team_id"]
        assert row["dire_team_id"] == dire["team_id"]
        for metric in DRAFT_PROFILE_METRIC_COLUMNS:
            expected = _manual_diff(radiant, dire, metric)
            actual = row[f"{metric}_diff"]
            if expected is None:
                assert pd.isna(actual), metric
            else:
                assert actual == pytest.approx(expected), metric


def test_swapping_radiant_and_dire_rows_negates_non_null_diffs(
    tmp_path: Path,
) -> None:
    comparison, profile = _assemble(tmp_path, _three_match_specs(), layers=True)
    swapped_profile = profile.copy()
    swapped_profile["side"] = swapped_profile["side"].map(
        {"RADIANT": "DIRE", "DIRE": "RADIANT"}
    )
    swapped = draft_comparison_from_profile(swapped_profile)
    left = comparison.sort_values("match_id").reset_index(drop=True)
    right = swapped.sort_values("match_id").reset_index(drop=True)
    assert (right["radiant_team_id"].to_numpy() == left["dire_team_id"].to_numpy()).all()
    assert (right["dire_team_id"].to_numpy() == left["radiant_team_id"].to_numpy()).all()
    for column in DRAFT_COMPARISON_METRIC_COLUMNS:
        original = left[column].to_numpy(dtype=float)
        flipped = right[column].to_numpy(dtype=float)
        both_null = np.isnan(original) & np.isnan(flipped)
        zero = (~np.isnan(original)) & (original == 0.0)
        nonzero = (~np.isnan(original)) & (original != 0.0)
        assert np.isnan(flipped[both_null]).all(), column
        assert (flipped[zero] == 0.0).all(), column
        np.testing.assert_allclose(
            flipped[nonzero], -original[nonzero], equal_nan=False
        )
        assert (np.sign(flipped[nonzero]) == -np.sign(original[nonzero])).all(), column


# --- inherited temporal integrity -----------------------------------------


def test_identical_timestamps_remain_mutually_blind(tmp_path: Path) -> None:
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
    comparison, profile = _assemble(tmp_path, specs, layers=True)
    for match_id in (M2, M3):
        row = _row(comparison, match_id)
        radiant = _side(profile, match_id, "RADIANT")
        dire = _side(profile, match_id, "DIRE")
        assert radiant["mean_player_prior_games_on_hero"] == pytest.approx(1.0)
        assert dire["mean_player_prior_games_on_hero"] == pytest.approx(0.0)
        assert row["mean_player_prior_games_on_hero_diff"] == pytest.approx(1.0)
        assert pd.isna(row["mean_player_prior_hero_share_diff"])
    first = _row(comparison, M1)
    assert first["mean_player_prior_games_on_hero_diff"] == pytest.approx(0.0)
    assert pd.isna(first["mean_same_version_contest_rate_diff"])
