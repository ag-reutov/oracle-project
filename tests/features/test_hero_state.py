"""Tests for leakage-safe expanding hero meta state (`features.hero_state`).

Deterministic fixtures with hand-calculated expected values.
Does not go through PRE_DRAFT snapshot SQL, Elo, or training assembly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from hero_meta_helpers import draft_and_player_rows, match_row
from hero_state_helpers import assign_positions, hero_state_frame

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.expected_position import (
    EXPECTED_POSITION_COLUMNS,
    EXPECTED_POSITION_METHODS,
)
from dota_predictor.features.hero_meta import HERO_META_METRIC_COLUMNS, hero_meta_sql
from dota_predictor.features.hero_state import (
    HERO_STATE_COLUMNS,
    HERO_STATE_METRIC_COLUMNS,
    RECENT_HERO_MATCH_WINDOWS,
    hero_state_sql,
    summarize_hero_state,
)
from dota_predictor.features.player_hero import PLAYER_HERO_METRIC_COLUMNS
from dota_predictor.features.player_match import PLAYER_STATE_METRIC_COLUMNS
from dota_predictor.features.player_position import PLAYER_POSITION_STATE_METRIC_COLUMNS
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
SERIES_1 = 501

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


def _unique_side_positions(
    match_id: int, players: tuple[int, ...], labels: tuple[str, ...]
) -> dict[tuple[int, int], str]:
    return {
        (match_id, player_id): label
        for player_id, label in zip(players, labels, strict=True)
    }


def _assemble(
    tmp_path: Path,
    specs: list[dict],
    *,
    heroes: list[dict] | None = CATALOG_HEROES,
    positions: dict[tuple[int, int], str | None] | None = None,
    match_id: int | None = None,
) -> pd.DataFrame:
    matches: list[dict] = []
    players: list[dict] = []
    drafts: list[dict] = []
    for spec in specs:
        row = match_row(
            spec["match_id"],
            start_time=spec["start_time"],
            radiant_win=spec["radiant_win"],
            game_version_id=spec.get("game_version_id", VERSION_A),
        )
        row["series_id"] = spec.get("series_id", SERIES_1)
        matches.append(row)
        draft_rows, player_rows = draft_and_player_rows(
            spec["match_id"],
            radiant_picks=spec.get("radiant_picks", RADIANT_DEFAULT),
            dire_picks=spec.get("dire_picks", DIRE_DEFAULT),
            successful_bans=spec.get("successful_bans", ()),
            unsuccessful_bans=spec.get("unsuccessful_bans", ()),
        )
        drafts.extend(draft_rows)
        players.extend(player_rows)
    if positions:
        assign_positions(players, positions)
    return hero_state_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=heroes,
        match_id=match_id,
    )


def _radiant_player_ids(match_id: int) -> tuple[int, ...]:
    return tuple(match_id * 100 + slot + 1 for slot in range(5))


def _dire_player_ids(match_id: int) -> tuple[int, ...]:
    return tuple(match_id * 100 + slot + 6 for slot in range(5))


def _default_positions(*match_ids: int) -> dict[tuple[int, int], str]:
    labels = (
        "POSITION_1",
        "POSITION_2",
        "POSITION_3",
        "POSITION_4",
        "POSITION_5",
    )
    assigned: dict[tuple[int, int], str] = {}
    for match_id in match_ids:
        assigned.update(
            _unique_side_positions(match_id, _radiant_player_ids(match_id), labels)
        )
        assigned.update(
            _unique_side_positions(match_id, _dire_player_ids(match_id), labels)
        )
    return assigned


# --- SQL / contract guards ------------------------------------------------


def test_sql_uses_strict_prior_and_never_less_or_equal() -> None:
    sql = hero_state_sql(catalog_registered=True)
    assert "start_time <=" not in sql
    assert STRICT_PRIOR_RANGE_SQL in sql
    assert "EXCLUDE GROUP" in sql
    assert "ORDER BY match_id" not in sql
    assert "expected_position" not in sql


def test_sql_encodes_career_recent_and_same_version_windows() -> None:
    sql = hero_state_sql(catalog_registered=True)
    assert "PARTITION BY hero_id" in sql
    assert "PARTITION BY game_version_id, hero_id" in sql
    for window in RECENT_HERO_MATCH_WINDOWS:
        assert f"ROWS BETWEEN {window} PRECEDING AND CURRENT ROW EXCLUDE GROUP" in sql


def test_sql_uses_successful_draft_action_rule() -> None:
    sql = hero_state_sql(catalog_registered=True)
    assert "was_successful IS DISTINCT FROM FALSE" in sql
    assert "GROUP BY match_id, hero_id" in sql


def test_slice_5_not_in_win_model_or_pre_draft_snapshot() -> None:
    for column in HERO_STATE_METRIC_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in SNAPSHOT_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
    assert "hero_state" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "hero_contest_rate" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "hero_prior_win_rate" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 5
    assert REFERENCE_SCHEMA_VERSION == 1


def test_prior_slices_and_hero_meta_remain_unchanged() -> None:
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
    assert PLAYER_HERO_METRIC_COLUMNS[0] == "prior_games_on_hero"
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
    assert HERO_META_METRIC_COLUMNS == (
        "same_version_prior_matches",
        "same_version_prior_picks",
        "same_version_prior_bans",
        "same_version_prior_contests",
        "same_version_pick_rate",
        "same_version_ban_rate",
        "same_version_contest_rate",
        "same_version_prior_wins",
        "same_version_prior_losses",
        "same_version_win_rate",
        "recent_90d_prior_matches",
        "recent_90d_prior_picks",
        "recent_90d_prior_bans",
        "recent_90d_prior_contests",
        "recent_90d_pick_rate",
        "recent_90d_ban_rate",
        "recent_90d_contest_rate",
        "recent_90d_prior_wins",
        "recent_90d_prior_losses",
        "recent_90d_win_rate",
    )
    assert set(HERO_STATE_METRIC_COLUMNS).isdisjoint(PLAYER_STATE_METRIC_COLUMNS)
    assert set(HERO_STATE_METRIC_COLUMNS).isdisjoint(PLAYER_HERO_METRIC_COLUMNS)
    old_sql = hero_meta_sql(catalog_registered=True)
    assert old_sql.count("EXCLUDE GROUP") == 2
    assert "hero_prior_matches" not in old_sql
    assert "hero_position_1_share" not in old_sql


# --- cold start / leakage -------------------------------------------------


def test_first_historical_appearance_is_cold_start(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [{"match_id": M1, "start_time": T0, "radiant_win": True}],
        positions=_default_positions(M1),
        match_id=M1,
    )
    assert list(frame.columns) == list(HERO_STATE_COLUMNS)
    hero1 = _row(frame, M1, 1)
    assert hero1["hero_prior_matches"] == 0
    assert hero1["hero_pick_count"] == 0
    assert hero1["hero_ban_count"] == 0
    assert hero1["hero_contest_count"] == 0
    assert hero1["hero_prior_wins"] == 0
    assert hero1["hero_position_1_count"] == 0
    assert pd.isna(hero1["hero_pick_rate"])
    assert pd.isna(hero1["hero_ban_rate"])
    assert pd.isna(hero1["hero_contest_rate"])
    assert pd.isna(hero1["hero_prior_win_rate"])
    assert pd.isna(hero1["hero_days_since_last_pick"])
    assert pd.isna(hero1["hero_position_1_share"])
    assert pd.isna(hero1["hero_recent_20_pick_rate"])
    assert pd.isna(hero1["hero_same_version_win_rate"])


def test_strictly_earlier_rows_only(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
            {"match_id": M3, "start_time": T2, "radiant_win": True},
        ],
        positions=_default_positions(M1, M2, M3),
        match_id=M2,
    )
    hero1 = _row(frame, M2, 1)
    assert hero1["hero_prior_matches"] == 1
    assert hero1["hero_pick_count"] == 1
    hero11 = _row(frame, M2, 11)
    assert hero11["hero_pick_count"] == 0
    assert hero11["hero_pick_rate"] == pytest.approx(0.0)


def test_identical_timestamps_are_mutually_blind(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {
                "match_id": M1,
                "start_time": T0,
                "radiant_win": True,
                "radiant_picks": (1, 3, 4, 5, 11),
            },
            {
                "match_id": M2,
                "start_time": T_TIE,
                "radiant_win": True,
                "radiant_picks": (2, 3, 4, 5, 11),
            },
            {
                "match_id": M3,
                "start_time": T_TIE,
                "radiant_win": False,
                "radiant_picks": (2, 12, 13, 14, 15),
                "dire_picks": (16, 17, 18, 7, 8),
            },
        ],
        positions=_default_positions(M1, M2, M3),
    )
    for match_id in (M2, M3):
        hero1 = _row(frame, match_id, 1)
        assert hero1["hero_prior_matches"] == 1
        assert hero1["hero_pick_count"] == 1
        hero2 = _row(frame, match_id, 2)
        assert hero2["hero_pick_count"] == 0
        assert hero2["hero_recent_20_pick_count"] == 0
        assert hero2["hero_pick_rate"] == pytest.approx(0.0)


def test_later_series_map_may_use_earlier_map(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {
                "match_id": M1,
                "start_time": T0,
                "radiant_win": True,
                "series_id": SERIES_1,
            },
            {
                "match_id": M2,
                "start_time": T1,
                "radiant_win": False,
                "series_id": SERIES_1,
            },
        ],
        positions=_default_positions(M1, M2),
        match_id=M2,
    )
    hero1 = _row(frame, M2, 1)
    assert hero1["hero_pick_count"] == 1
    assert hero1["hero_prior_wins"] == 1
    assert hero1["hero_days_since_last_pick"] == pytest.approx(1.0)


def test_current_outcome_never_enters_current_hero_state(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": True},
        ],
        positions=_default_positions(M1, M2),
        match_id=M2,
    )
    hero1 = _row(frame, M2, 1)
    assert hero1["hero_prior_wins"] == 1
    assert hero1["hero_prior_losses"] == 0
    assert hero1["hero_pick_count"] == 1
    assert hero1["hero_prior_win_rate"] == pytest.approx(1.0)


def test_current_observed_position_never_enters_current_hero_state(
    tmp_path: Path,
) -> None:
    positions = _default_positions(M1)
    positions.update(
        {
            (M2, _radiant_player_ids(M2)[0]): "POSITION_5",
            (M2, _radiant_player_ids(M2)[1]): "POSITION_1",
            (M2, _radiant_player_ids(M2)[2]): "POSITION_2",
            (M2, _radiant_player_ids(M2)[3]): "POSITION_3",
            (M2, _radiant_player_ids(M2)[4]): "POSITION_4",
        }
    )
    positions.update(
        _unique_side_positions(
            M2,
            _dire_player_ids(M2),
            (
                "POSITION_1",
                "POSITION_2",
                "POSITION_3",
                "POSITION_4",
                "POSITION_5",
            ),
        )
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
    hero1 = _row(frame, M2, 1)
    assert hero1["hero_position_1_count"] == 1
    assert hero1["hero_position_5_count"] == 0
    assert hero1["hero_position_1_share"] == pytest.approx(1.0)


def test_null_historical_positions_do_not_increment_counts(tmp_path: Path) -> None:
    positions = _default_positions(M1)
    positions[(M1, _radiant_player_ids(M1)[0])] = None
    positions.update(_default_positions(M2))
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        match_id=M2,
    )
    hero1 = _row(frame, M2, 1)
    assert hero1["hero_pick_count"] == 1
    assert hero1["hero_position_explicit_count"] == 0
    assert hero1["hero_position_1_count"] == 0
    assert pd.isna(hero1["hero_position_1_share"])


def test_unknown_historical_position_does_not_increment_counts(tmp_path: Path) -> None:
    positions = _default_positions(M1)
    positions[(M1, _radiant_player_ids(M1)[0])] = "UNKNOWN"
    positions.update(_default_positions(M2))
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        match_id=M2,
    )
    hero1 = _row(frame, M2, 1)
    assert hero1["hero_position_explicit_count"] == 0
    assert pd.isna(hero1["hero_position_1_share"])


def test_position_shares_sum_when_evidence_exists(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=_default_positions(M1, M2),
        match_id=M2,
    )
    hero1 = _row(frame, M2, 1)
    shares = [
        hero1[f"hero_position_{position}_share"] for position in range(1, 6)
    ]
    assert hero1["hero_position_explicit_count"] == 1
    assert sum(shares) == pytest.approx(1.0)
    never = _row(frame, M2, 99)
    assert never["hero_position_explicit_count"] == 0
    assert pd.isna(never["hero_position_1_share"])


def test_draft_pick_ban_use_only_prior_drafts(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {
                "match_id": M1,
                "start_time": T0,
                "radiant_win": True,
                "successful_bans": (("RADIANT", 20), ("DIRE", 21)),
                "unsuccessful_bans": (("RADIANT", 50),),
            },
            {
                "match_id": M2,
                "start_time": T1,
                "radiant_win": False,
                "successful_bans": (("RADIANT", 20),),
            },
        ],
        positions=_default_positions(M1, M2),
        match_id=M2,
    )
    hero20 = _row(frame, M2, 20)
    assert hero20["hero_ban_count"] == 1
    assert hero20["hero_contest_count"] == 1
    assert hero20["hero_pick_count"] == 0
    assert hero20["hero_ban_rate"] == pytest.approx(1.0)
    hero50 = _row(frame, M2, 50)
    assert hero50["hero_ban_count"] == 0
    assert hero50["hero_contest_count"] == 0
    hero21 = _row(frame, M2, 21)
    assert hero21["hero_ban_count"] == 1
    assert pd.isna(hero21["hero_prior_win_rate"])


def test_expanding_history_does_not_reset_at_patch(tmp_path: Path) -> None:
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
        positions=_default_positions(M1, M2),
        match_id=M2,
    )
    hero1 = _row(frame, M2, 1)
    assert hero1["hero_prior_matches"] == 1
    assert hero1["hero_pick_count"] == 1
    assert hero1["hero_same_version_prior_matches"] == 0
    assert hero1["hero_same_version_pick_count"] == 0
    assert pd.isna(hero1["hero_same_version_pick_rate"])
    assert hero1["hero_recent_20_matches"] == 1
    assert hero1["hero_recent_20_pick_count"] == 1


def test_days_since_last_pick_is_null_until_a_prior_pick(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {
                "match_id": M1,
                "start_time": T0,
                "radiant_win": True,
                "successful_bans": (("DIRE", 99),),
            },
            {"match_id": M2, "start_time": T1, "radiant_win": True},
        ],
        positions=_default_positions(M1, M2),
        match_id=M2,
    )
    never_picked = _row(frame, M2, 99)
    assert never_picked["hero_ban_count"] == 1
    assert never_picked["hero_pick_count"] == 0
    assert pd.isna(never_picked["hero_days_since_last_pick"])
    assert pd.isna(never_picked["hero_prior_win_rate"])
    hero1 = _row(frame, M2, 1)
    assert hero1["hero_days_since_last_pick"] == pytest.approx(1.0)


def test_wins_plus_losses_equal_picks(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
            {"match_id": M3, "start_time": T2, "radiant_win": True},
        ],
        positions=_default_positions(M1, M2, M3),
    )
    assert (
        frame["hero_prior_wins"] + frame["hero_prior_losses"]
        == frame["hero_pick_count"]
    ).all()


def test_recent_window_includes_only_trailing_prior_matches(tmp_path: Path) -> None:
    t_old = T0
    specs = [
        {
            "match_id": 10 + i,
            "start_time": t_old + timedelta(days=i),
            "radiant_win": i % 2 == 0,
            "radiant_picks": (1, 2, 3, 4, 5) if i < 3 else (11, 12, 13, 14, 15),
            "dire_picks": (6, 7, 8, 9, 10),
        }
        for i in range(5)
    ]
    frame = _assemble(
        tmp_path,
        specs,
        positions=_default_positions(*(10 + i for i in range(5))),
        match_id=14,
    )
    hero1 = _row(frame, 14, 1)
    assert hero1["hero_prior_matches"] == 4
    assert hero1["hero_pick_count"] == 3
    assert hero1["hero_recent_20_matches"] == 4
    assert hero1["hero_recent_20_pick_count"] == 3


def test_output_columns_and_catalog_grain(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": True},
        ],
        positions=_default_positions(M1, M2),
    )
    assert list(frame.columns) == list(HERO_STATE_COLUMNS)
    assert set(frame["match_id"].unique()) == {M1, M2}
    assert len(frame) == 2 * len(CATALOG_HEROES)


def test_summarize_reports_cold_start_rate() -> None:
    frame = pd.DataFrame(
        {
            "game_version_id": [10, 10, 11],
            "hero_pick_count": [2, 0, 4],
            "hero_prior_matches": [5, 0, 8],
            "hero_position_explicit_count": [2, 0, 1],
            "hero_pick_rate": [0.4, None, 0.5],
            "hero_ban_rate": [0.1, None, 0.2],
            "hero_contest_rate": [0.5, None, 0.6],
            "hero_prior_win_rate": [0.5, None, 0.25],
            "hero_position_1_share": [1.0, None, 0.0],
            "hero_position_2_share": [0.0, None, 1.0],
            "hero_position_3_share": [0.0, None, 0.0],
            "hero_position_4_share": [0.0, None, 0.0],
            "hero_position_5_share": [0.0, None, 0.0],
        }
    )
    summary = summarize_hero_state(frame)
    coverage = summary[
        (summary["scope"] == "overall") & (summary["stat"] == "coverage")
    ].iloc[0]
    assert coverage["cold_start_no_picks"] == pytest.approx(1 / 3)
    assert coverage["prior_pick_coverage"] == pytest.approx(2 / 3)
