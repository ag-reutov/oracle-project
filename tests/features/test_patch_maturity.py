"""Tests for descriptive patch-age / professional-match maturity.

Derivation only: calendar days vs STRATZ as_of, strictly-prior match
counts, same-timestamp blindness, and predefined bins. Not a training
feature. Does not go through PRE_DRAFT snapshot SQL or Elo.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from hero_meta_helpers import (
    draft_and_player_rows,
    match_row,
    write_hero_meta_store,
)

from dota_predictor.features.duckdb_layer import connect, register_reference_views
from dota_predictor.features.patch_maturity import (
    MISSING_CALENDAR_AGE_BIN,
    NEGATIVE_CALENDAR_AGE_BIN,
    PATCH_MATURITY_COLUMNS,
    PATCH_MATURITY_METRIC_COLUMNS,
    assign_calendar_age_bin,
    assign_prior_match_bin,
    build_patch_maturity,
    patch_age_sanity_table,
    patch_maturity_sql,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS
from dota_predictor.training.patch_maturity_diagnostics import attach_patch_maturity

T0 = datetime(2024, 1, 1, tzinfo=UTC)
T0_UNIX = int(T0.timestamp())

VERSION_A = 10
VERSION_B = 11

CATALOG_HEROES = [{"id": hero_id, "displayName": f"Hero {hero_id}"} for hero_id in range(1, 23)]


def _players_and_drafts(match_id: int, *, offset: int) -> tuple[list, list]:
    radiant = tuple(range(offset, offset + 5))
    dire = tuple(range(offset + 5, offset + 10))
    drafts, players = draft_and_player_rows(
        match_id, radiant_picks=radiant, dire_picks=dire
    )
    return drafts, players


def _maturity_frame(
    tmp_path: Path,
    *,
    matches: list[dict],
    game_versions: list[dict],
) -> pd.DataFrame:
    drafts: list[dict] = []
    players: list[dict] = []
    for i, match in enumerate(matches):
        d, p = _players_and_drafts(int(match["match_id"]), offset=1 + i * 10)
        drafts.extend(d)
        players.extend(p)
    feature_config, reference_config = write_hero_meta_store(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        heroes=CATALOG_HEROES,
        game_versions=game_versions,
    )
    assert reference_config is not None
    with connect(feature_config) as store:
        register_reference_views(store, reference_config)
        return build_patch_maturity(store).to_frame()


def _row(frame: pd.DataFrame, match_id: int) -> pd.Series:
    subset = frame[frame["match_id"] == match_id]
    assert len(subset) == 1
    return subset.iloc[0]


def test_patch_maturity_is_not_part_of_training_or_pre_draft_snapshot() -> None:
    assert set(PATCH_MATURITY_METRIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(PATCH_MATURITY_METRIC_COLUMNS).isdisjoint(SNAPSHOT_COLUMNS)
    assert set(PATCH_MATURITY_METRIC_COLUMNS).isdisjoint(ALL_FEATURE_COLUMNS)
    assert "patch_maturity" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "days_since_game_version_start" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "prior_matches_in_game_version" not in PRE_DRAFT_SNAPSHOT_SQL


def test_sql_partitions_prior_matches_by_game_version() -> None:
    sql = patch_maturity_sql()
    assert "PARTITION BY m.game_version_id" in sql
    assert "EXCLUDE GROUP" in sql
    assert "ORDER BY m.start_time" in sql
    assert "ORDER BY match_id" not in sql


def test_calendar_age_is_days_since_as_of(tmp_path: Path) -> None:
    matches = [
        match_row(
            1,
            start_time=T0 + timedelta(days=10, hours=12),
            radiant_win=True,
            game_version_id=VERSION_A,
        )
    ]
    frame = _maturity_frame(
        tmp_path,
        matches=matches,
        game_versions=[
            {"id": VERSION_A, "name": "7.00", "asOfDateTime": T0_UNIX},
        ],
    )
    assert list(frame.columns) == list(PATCH_MATURITY_COLUMNS)
    row = _row(frame, 1)
    assert row["days_since_game_version_start"] == pytest.approx(10.5)
    assert row["prior_matches_in_game_version"] == 0
    assert row["game_version_name"] == "7.00"


def test_first_match_in_version_has_zero_prior_matches(tmp_path: Path) -> None:
    matches = [
        match_row(
            1, start_time=T0, radiant_win=True, game_version_id=VERSION_A
        ),
        match_row(
            2,
            start_time=T0 + timedelta(days=1),
            radiant_win=False,
            game_version_id=VERSION_A,
        ),
    ]
    frame = _maturity_frame(
        tmp_path,
        matches=matches,
        game_versions=[
            {"id": VERSION_A, "name": "7.00", "asOfDateTime": T0_UNIX},
        ],
    )
    assert _row(frame, 1)["prior_matches_in_game_version"] == 0
    assert _row(frame, 2)["prior_matches_in_game_version"] == 1


def test_same_timestamp_peers_are_excluded_from_prior_count(tmp_path: Path) -> None:
    matches = [
        match_row(
            1, start_time=T0, radiant_win=True, game_version_id=VERSION_A
        ),
        match_row(
            2, start_time=T0, radiant_win=False, game_version_id=VERSION_A
        ),
        match_row(
            3,
            start_time=T0 + timedelta(hours=1),
            radiant_win=True,
            game_version_id=VERSION_A,
        ),
    ]
    frame = _maturity_frame(
        tmp_path,
        matches=matches,
        game_versions=[
            {"id": VERSION_A, "name": "7.00", "asOfDateTime": T0_UNIX},
        ],
    )
    assert _row(frame, 1)["prior_matches_in_game_version"] == 0
    assert _row(frame, 2)["prior_matches_in_game_version"] == 0
    assert _row(frame, 3)["prior_matches_in_game_version"] == 2


def test_version_transition_resets_prior_match_maturity(tmp_path: Path) -> None:
    matches = [
        match_row(
            1, start_time=T0, radiant_win=True, game_version_id=VERSION_A
        ),
        match_row(
            2,
            start_time=T0 + timedelta(days=2),
            radiant_win=False,
            game_version_id=VERSION_A,
        ),
        match_row(
            3,
            start_time=T0 + timedelta(days=3),
            radiant_win=True,
            game_version_id=VERSION_B,
        ),
    ]
    frame = _maturity_frame(
        tmp_path,
        matches=matches,
        game_versions=[
            {"id": VERSION_A, "name": "7.00", "asOfDateTime": T0_UNIX},
            {
                "id": VERSION_B,
                "name": "7.01",
                "asOfDateTime": T0_UNIX + 3 * 86400,
            },
        ],
    )
    assert _row(frame, 2)["prior_matches_in_game_version"] == 1
    assert _row(frame, 3)["prior_matches_in_game_version"] == 0
    assert _row(frame, 3)["game_version_id"] == VERSION_B


def test_later_matches_do_not_leak_into_earlier_prior_counts(
    tmp_path: Path,
) -> None:
    matches = [
        match_row(
            1, start_time=T0, radiant_win=True, game_version_id=VERSION_A
        ),
        match_row(
            2,
            start_time=T0 + timedelta(days=5),
            radiant_win=False,
            game_version_id=VERSION_A,
        ),
        match_row(
            3,
            start_time=T0 + timedelta(days=9),
            radiant_win=True,
            game_version_id=VERSION_A,
        ),
    ]
    frame = _maturity_frame(
        tmp_path,
        matches=matches,
        game_versions=[
            {"id": VERSION_A, "name": "7.00", "asOfDateTime": T0_UNIX},
        ],
    )
    assert _row(frame, 1)["prior_matches_in_game_version"] == 0
    assert _row(frame, 2)["prior_matches_in_game_version"] == 1
    assert _row(frame, 3)["prior_matches_in_game_version"] == 2


def test_negative_calendar_age_is_preserved_not_repaired(tmp_path: Path) -> None:
    matches = [
        match_row(
            1, start_time=T0, radiant_win=True, game_version_id=VERSION_A
        )
    ]
    later_as_of = T0_UNIX + 5 * 86400
    frame = _maturity_frame(
        tmp_path,
        matches=matches,
        game_versions=[
            {"id": VERSION_A, "name": "7.00", "asOfDateTime": later_as_of},
        ],
    )
    days = float(_row(frame, 1)["days_since_game_version_start"])
    assert days == pytest.approx(-5.0)
    sanity = patch_age_sanity_table(frame)
    assert sanity.iloc[0]["n_negative_calendar_age"] == 1
    assert bool(sanity.iloc[0]["flagged"]) is True
    assert days < 0


def test_calendar_and_prior_match_bins_are_deterministic() -> None:
    assert assign_calendar_age_bin(None) == MISSING_CALENDAR_AGE_BIN
    assert assign_calendar_age_bin(float("nan")) == MISSING_CALENDAR_AGE_BIN
    assert assign_calendar_age_bin(-0.01) == NEGATIVE_CALENDAR_AGE_BIN
    assert assign_calendar_age_bin(0.0) == "0–7 days"
    assert assign_calendar_age_bin(7.99) == "0–7 days"
    assert assign_calendar_age_bin(8.0) == "8–21 days"
    assert assign_calendar_age_bin(21.9) == "8–21 days"
    assert assign_calendar_age_bin(22.0) == "22–45 days"
    assert assign_calendar_age_bin(45.9) == "22–45 days"
    assert assign_calendar_age_bin(46.0) == "46+ days"
    assert assign_prior_match_bin(0) == "0–49 prior matches"
    assert assign_prior_match_bin(49) == "0–49 prior matches"
    assert assign_prior_match_bin(50) == "50–199"
    assert assign_prior_match_bin(199) == "50–199"
    assert assign_prior_match_bin(200) == "200–499"
    assert assign_prior_match_bin(499) == "200–499"
    assert assign_prior_match_bin(500) == "500+"
    with pytest.raises(ValueError, match="negative"):
        assign_prior_match_bin(-1)


def test_attach_patch_maturity_assigns_predefined_bins() -> None:
    oos = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "model": ["logistic_elo_only"] * 4,
            "delta_vs_elo": [0.0, 0.0, 0.0, 0.0],
        }
    )
    maturity = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "days_since_game_version_start": [-1.0, 7.99, 22.0, 46.0],
            "prior_matches_in_game_version": [0, 49, 200, 500],
            "game_version_name": ["7.00"] * 4,
            "as_of_datetime": [T0] * 4,
        }
    )
    joined = attach_patch_maturity(oos, maturity)
    assert list(joined["calendar_age_bin"]) == [
        NEGATIVE_CALENDAR_AGE_BIN,
        "0–7 days",
        "22–45 days",
        "46+ days",
    ]
    assert list(joined["prior_match_bin"]) == [
        "0–49 prior matches",
        "0–49 prior matches",
        "200–499",
        "500+",
    ]


def test_attach_patch_maturity_rejects_matches_without_prior_count() -> None:
    oos = pd.DataFrame({"match_id": [1, 2], "model": ["logistic_elo_only"] * 2})
    maturity = pd.DataFrame(
        {
            "match_id": [1],
            "days_since_game_version_start": [1.0],
            "prior_matches_in_game_version": [0],
            "game_version_name": ["7.00"],
            "as_of_datetime": [T0],
        }
    )
    with pytest.raises(ValueError, match="prior_matches_in_game_version"):
        attach_patch_maturity(oos, maturity)
