"""Tests for the side-level draft profile (`features.draft_profile`).

Small deterministic fixtures with hand-calculated expected values.
Does not go through PRE_DRAFT snapshot SQL, Elo, or training assembly.
`slot_in_side` is lobby order only and is never treated as position 1-5.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from draft_profile_helpers import (
    draft_and_player_rows,
    draft_profile_frame,
    draft_profile_layers,
    match_row,
)

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.draft_profile import (
    DRAFT_PROFILE_COLUMNS,
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

RATE_COLUMNS = (
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

CATALOG_HEROES = [
    {"id": hero_id, "displayName": f"Hero {hero_id}"}
    for hero_id in range(1, 23)
]


def _side(frame: pd.DataFrame, match_id: int, side: str) -> pd.Series:
    subset = frame[(frame["match_id"] == match_id) & (frame["side"] == side)]
    assert len(subset) == 1, (
        f"expected one row for ({match_id}, {side}), got {len(subset)}"
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
        return draft_profile_layers(**kwargs)
    return draft_profile_frame(**kwargs)


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


# --- SQL / contract guards ------------------------------------------------


def test_sql_composes_existing_layers_and_aggregates_by_side() -> None:
    sql = draft_profile_sql(catalog_registered=True)
    assert "AVG(" in sql
    assert "GROUP BY" in sql
    assert "prior_games_on_hero" in sql
    assert "team_prior_games_with_hero" in sql
    assert "same_version_contest_rate" in sql
    assert "EXCLUDE GROUP" in sql
    assert "p.side" in sql.split("GROUP BY")[-1]


def test_sql_does_not_use_slot_as_position_or_current_outcome() -> None:
    sql = draft_profile_sql(catalog_registered=True)
    grouped = sql.split("GROUP BY")[-1]
    assert "slot_in_side" not in grouped
    assert "radiant_win" not in grouped
    assert "player_id" not in grouped
    for forbidden in ("position", "lane", "role", "synergy", "counter", "elo"):
        assert forbidden not in grouped.lower()


def test_draft_profile_is_not_part_of_training_or_pre_draft_snapshot() -> None:
    assert set(DRAFT_PROFILE_METRIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(DRAFT_PROFILE_METRIC_COLUMNS).isdisjoint(SNAPSHOT_COLUMNS)
    assert set(DRAFT_PROFILE_METRIC_COLUMNS).isdisjoint(ALL_FEATURE_COLUMNS)
    assert "draft_profile" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "mean_player_prior_games_on_hero" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged_by_this_layer() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 5
    assert REFERENCE_SCHEMA_VERSION == 2


# --- grain ----------------------------------------------------------------


def test_exactly_two_rows_per_match(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs())
    assert list(frame.columns) == list(DRAFT_PROFILE_COLUMNS)
    assert set(frame["match_id"].unique()) == {M1, M2, M3}
    assert len(frame) == 6
    assert frame.groupby("match_id").size().eq(2).all()
    assert set(frame["side"]) == {"RADIANT", "DIRE"}


def test_five_players_and_heroes_per_side(tmp_path: Path) -> None:
    profile, player_hero, team_hero, _ = _assemble(
        tmp_path, _three_match_specs(), match_id=M2, layers=True
    )
    assert len(profile) == 2
    assert player_hero.groupby("side").size().eq(5).all()
    assert team_hero.groupby("side").size().eq(5).all()
    assert player_hero.groupby("side")["hero_id"].nunique().eq(5).all()
    assert player_hero.groupby("side")["player_id"].nunique().eq(5).all()


# --- player assignment vs team aggregation --------------------------------


def test_player_hero_assignment_uses_the_player_on_that_hero(tmp_path: Path) -> None:
    """P1 (hero 1 in M1) and P2 (hero 2 in M1) swap those two heroes in M2.

    Player means must see two zeros (neither player has prior games on the
    swapped hero). Team means still see both heroes as previously played.
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
            "radiant_heroes": (2, 1, 3, 4, 5),
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    radiant = _side(frame, M2, "RADIANT")
    assert radiant["mean_player_prior_games_on_hero"] == pytest.approx(0.6)
    assert radiant["min_player_prior_games_on_hero"] == 0
    assert radiant["players_with_zero_prior_games_on_hero"] == 2
    assert radiant["mean_team_prior_games_with_hero"] == pytest.approx(1.0)
    assert radiant["min_team_prior_games_with_hero"] == 1
    assert radiant["heroes_never_played_by_team"] == 0


def test_team_hero_aggregation_uses_current_team(tmp_path: Path) -> None:
    """Same five players move from Team A to Team D. Player history
    continues; team history starts over.
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
            "dire_players": RADIANT_PLAYERS,
            "radiant_heroes": (11, 12, 13, 14, 15),
            "dire_heroes": RADIANT_HEROES,
        },
    ]
    frame = _assemble(tmp_path, specs, match_id=M2)
    dire = _side(frame, M2, "DIRE")
    assert dire["team_id"] == TEAM_D
    assert dire["mean_player_prior_games_on_hero"] == pytest.approx(1.0)
    assert dire["players_with_zero_prior_games_on_hero"] == 0
    assert dire["mean_team_prior_games_with_hero"] == pytest.approx(0.0)
    assert dire["heroes_never_played_by_team"] == 5
    assert pd.isna(dire["mean_team_hero_share"])


# --- zero-history and NULL propagation ------------------------------------


def test_zero_history_counts_and_null_rates_on_first_match(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M1)
    for side in ("RADIANT", "DIRE"):
        row = _side(frame, M1, side)
        assert row["mean_player_prior_games_on_hero"] == pytest.approx(0.0)
        assert row["min_player_prior_games_on_hero"] == 0
        assert row["players_with_zero_prior_games_on_hero"] == 5
        assert row["players_with_zero_recent_90d_games_on_hero"] == 5
        assert row["mean_team_prior_games_with_hero"] == pytest.approx(0.0)
        assert row["heroes_never_played_by_team"] == 5
        assert row["heroes_not_played_by_team_recent_90d"] == 5
        assert pd.isna(row["mean_player_prior_hero_share"])
        assert pd.isna(row["mean_player_recent_90d_hero_share"])
        assert pd.isna(row["mean_team_hero_share"])
        assert pd.isna(row["mean_team_recent_90d_hero_share"])
        assert pd.isna(row["mean_same_version_contest_rate"])
        assert pd.isna(row["min_same_version_contest_rate"])
        assert pd.isna(row["mean_recent_90d_contest_rate"])
        assert pd.isna(row["min_recent_90d_contest_rate"])
        assert pd.isna(row["mean_same_version_pick_rate"])
        assert pd.isna(row["mean_same_version_ban_rate"])
        assert pd.isna(row["mean_recent_90d_pick_rate"])
        assert pd.isna(row["mean_recent_90d_ban_rate"])


def test_null_rates_are_not_replaced_with_zero(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M1)
    radiant = _side(frame, M1, "RADIANT")
    for column in RATE_COLUMNS:
        assert pd.isna(radiant[column]), column


# --- identical-timestamp leakage ------------------------------------------


def test_identical_timestamps_are_mutually_blind(tmp_path: Path) -> None:
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
        radiant = _side(frame, match_id, "RADIANT")
        assert radiant["mean_player_prior_games_on_hero"] == pytest.approx(1.0)
        assert radiant["players_with_zero_prior_games_on_hero"] == 0
        assert radiant["mean_team_prior_games_with_hero"] == pytest.approx(1.0)
        assert radiant["mean_same_version_pick_rate"] == pytest.approx(1.0)

    first = _side(frame, M1, "RADIANT")
    assert first["mean_player_prior_games_on_hero"] == pytest.approx(0.0)
    assert pd.isna(first["mean_same_version_contest_rate"])


# --- patch opener ---------------------------------------------------------


def test_patch_opener_null_same_version_meta_keeps_recent_90d(
    tmp_path: Path,
) -> None:
    frame = _assemble(
        tmp_path, _three_match_specs(m3_version=VERSION_B), match_id=M3
    )
    radiant = _side(frame, M3, "RADIANT")
    assert pd.isna(radiant["mean_same_version_contest_rate"])
    assert pd.isna(radiant["min_same_version_contest_rate"])
    assert pd.isna(radiant["mean_same_version_pick_rate"])
    assert pd.isna(radiant["mean_same_version_ban_rate"])
    # Hero 11 was picked in 1 of 2 prior matches; 1,3,4,5 in both.
    assert radiant["mean_recent_90d_contest_rate"] == pytest.approx(0.9)
    assert radiant["min_recent_90d_contest_rate"] == pytest.approx(0.5)
    assert radiant["mean_recent_90d_pick_rate"] == pytest.approx(0.9)
    assert radiant["mean_recent_90d_ban_rate"] == pytest.approx(0.0)


# --- Radiant / Dire independence ------------------------------------------


def test_radiant_and_dire_profiles_are_independent(tmp_path: Path) -> None:
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    radiant = _side(frame, M2, "RADIANT")
    dire = _side(frame, M2, "DIRE")
    assert radiant["team_id"] == TEAM_A
    assert dire["team_id"] == TEAM_B
    assert radiant["mean_player_prior_games_on_hero"] == pytest.approx(0.8)
    assert radiant["players_with_zero_prior_games_on_hero"] == 1
    assert dire["mean_player_prior_games_on_hero"] == pytest.approx(1.0)
    assert dire["players_with_zero_prior_games_on_hero"] == 0
    assert dire["min_player_prior_games_on_hero"] == 1


# --- hand-computable aggregation ------------------------------------------


def test_hand_computed_radiant_profile_at_m2(tmp_path: Path) -> None:
    """M2 Radiant: P2 is on new hero 11; the other four heroes repeat M1.

    Player/team games: (1, 0, 1, 1, 1) → mean 0.8, min 0, one zero.
    Shares: (1, 0, 1, 1, 1) → mean 0.8.
    Meta (1 prior match, no bans): contest/pick (1, 0, 1, 1, 1) → 0.8 / min 0.
    """
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M2)
    radiant = _side(frame, M2, "RADIANT")
    assert radiant["mean_player_prior_games_on_hero"] == pytest.approx(0.8)
    assert radiant["min_player_prior_games_on_hero"] == 0
    assert radiant["mean_player_recent_90d_games_on_hero"] == pytest.approx(0.8)
    assert radiant["mean_player_prior_hero_share"] == pytest.approx(0.8)
    assert radiant["mean_player_recent_90d_hero_share"] == pytest.approx(0.8)
    assert radiant["players_with_zero_prior_games_on_hero"] == 1
    assert radiant["players_with_zero_recent_90d_games_on_hero"] == 1
    assert radiant["mean_team_prior_games_with_hero"] == pytest.approx(0.8)
    assert radiant["min_team_prior_games_with_hero"] == 0
    assert radiant["mean_team_recent_90d_games_with_hero"] == pytest.approx(0.8)
    assert radiant["mean_team_hero_share"] == pytest.approx(0.8)
    assert radiant["mean_team_recent_90d_hero_share"] == pytest.approx(0.8)
    assert radiant["heroes_never_played_by_team"] == 1
    assert radiant["heroes_not_played_by_team_recent_90d"] == 1
    assert radiant["mean_same_version_contest_rate"] == pytest.approx(0.8)
    assert radiant["min_same_version_contest_rate"] == pytest.approx(0.0)
    assert radiant["mean_recent_90d_contest_rate"] == pytest.approx(0.8)
    assert radiant["min_recent_90d_contest_rate"] == pytest.approx(0.0)
    assert radiant["mean_same_version_pick_rate"] == pytest.approx(0.8)
    assert radiant["mean_same_version_ban_rate"] == pytest.approx(0.0)
    assert radiant["mean_recent_90d_pick_rate"] == pytest.approx(0.8)
    assert radiant["mean_recent_90d_ban_rate"] == pytest.approx(0.0)


def test_hand_computed_radiant_profile_at_m3(tmp_path: Path) -> None:
    """M3 Radiant after two version-A maps: games (2, 1, 2, 2, 2) → 1.8.

    Shares: P2 is 1/2 on hero 11; others 1.0 → 0.9.
    Meta: hero 11 contested in 1/2 matches, others 2/2 → 0.9 / min 0.5.
    """
    frame = _assemble(tmp_path, _three_match_specs(), match_id=M3)
    radiant = _side(frame, M3, "RADIANT")
    assert radiant["mean_player_prior_games_on_hero"] == pytest.approx(1.8)
    assert radiant["min_player_prior_games_on_hero"] == 1
    assert radiant["players_with_zero_prior_games_on_hero"] == 0
    assert radiant["mean_player_prior_hero_share"] == pytest.approx(0.9)
    assert radiant["mean_team_prior_games_with_hero"] == pytest.approx(1.8)
    assert radiant["mean_team_hero_share"] == pytest.approx(0.9)
    assert radiant["heroes_never_played_by_team"] == 0
    assert radiant["mean_same_version_contest_rate"] == pytest.approx(0.9)
    assert radiant["min_same_version_contest_rate"] == pytest.approx(0.5)
    assert radiant["mean_same_version_pick_rate"] == pytest.approx(0.9)


def test_profile_matches_nanmean_of_underlying_layers(tmp_path: Path) -> None:
    profile, player_hero, team_hero, hero_meta = _assemble(
        tmp_path, _three_match_specs(), match_id=M2, layers=True
    )
    radiant_players = player_hero[player_hero["side"] == "RADIANT"]
    radiant_team = team_hero[team_hero["side"] == "RADIANT"]
    drafted = set(radiant_players["hero_id"])
    radiant_meta = hero_meta[hero_meta["hero_id"].isin(drafted)]
    row = _side(profile, M2, "RADIANT")
    assert row["mean_player_prior_games_on_hero"] == pytest.approx(
        float(radiant_players["prior_games_on_hero"].mean())
    )
    assert row["min_player_prior_games_on_hero"] == int(
        radiant_players["prior_games_on_hero"].min()
    )
    assert row["mean_team_prior_games_with_hero"] == pytest.approx(
        float(radiant_team["team_prior_games_with_hero"].mean())
    )
    assert row["mean_same_version_contest_rate"] == pytest.approx(
        float(radiant_meta["same_version_contest_rate"].mean())
    )
    assert row["players_with_zero_prior_games_on_hero"] == int(
        (radiant_players["prior_games_on_hero"] == 0).sum()
    )
