"""Tests for team Elo state extraction and leaderboard helpers.

These cover `compute_team_elo_state` / ranking / active-team filtering --
not the Elo update mathematics, which `test_team_elo.py` already owns.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from dota_predictor.features.team_elo import (
    CURRENT_ELO_COLUMN,
    DEFAULT_ACTIVE_DAYS,
    ELO_COLUMN,
    TEAM_ELO_FEATURE_COLUMNS,
    TEAM_ELO_STATE_COLUMNS,
    TEAM_ID_COLUMN,
    EloConfig,
    active_team_elo_cutoff,
    compute_team_elo_features,
    compute_team_elo_state,
    expected_score,
    filter_active_team_elo,
    rank_team_elo_state,
    team_elo_trajectories,
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)
T_OLD = datetime(2023, 1, 1, tzinfo=UTC)
T_MAX = datetime(2024, 6, 1, tzinfo=UTC)

TEAM_A, TEAM_B, TEAM_C, TEAM_D = 1, 2, 3, 4


def _match(
    match_id: int,
    *,
    start_time: datetime,
    radiant_team_id: int,
    dire_team_id: int,
    radiant_win: bool,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "start_time": start_time,
        "radiant_team_id": radiant_team_id,
        "dire_team_id": dire_team_id,
        "radiant_win": radiant_win,
    }


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _five_match_frame() -> tuple[pd.DataFrame, EloConfig]:
    config = EloConfig()
    rows = [
        _match(
            10,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        _match(
            20,
            start_time=T1,
            radiant_team_id=TEAM_C,
            dire_team_id=TEAM_D,
            radiant_win=False,
        ),
        _match(
            30,
            start_time=T2,
            radiant_team_id=TEAM_B,
            dire_team_id=TEAM_C,
            radiant_win=True,
        ),
        _match(
            40,
            start_time=T2,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_D,
            radiant_win=False,
        ),
        _match(
            50,
            start_time=T3,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
    ]
    return _frame(rows), config


def _final_ratings_from_feature_snapshots(
    matches: pd.DataFrame, features: pd.DataFrame, *, config: EloConfig
) -> dict[int, float]:
    """Independent reconstruction: sum per-match deltas implied by the
    pre-match feature snapshots. Same-timestamp groups already snapshot
    the pre-group rating, so this sum equals the batched group update.
    """
    joined = matches.merge(features, on="match_id")
    deltas: dict[int, float] = {}
    for row in joined.itertuples(index=False):
        actual = 1.0 if bool(row.radiant_win) else 0.0
        change = config.k_factor * (
            actual - expected_score(float(row.radiant_team_elo), float(row.dire_team_elo))
        )
        radiant_id = int(row.radiant_team_id)
        dire_id = int(row.dire_team_id)
        deltas[radiant_id] = deltas.get(radiant_id, 0.0) + change
        deltas[dire_id] = deltas.get(dire_id, 0.0) - change
    return {
        team_id: config.initial_rating + delta for team_id, delta in deltas.items()
    }


def test_final_state_equals_sum_of_feature_snapshot_deltas() -> None:
    matches, config = _five_match_frame()
    features = compute_team_elo_features(matches, config=config)
    state = compute_team_elo_state(matches, config=config).set_index(TEAM_ID_COLUMN)
    reconstructed = _final_ratings_from_feature_snapshots(
        matches, features, config=config
    )

    assert set(state.index) == set(reconstructed)
    for team_id, elo in reconstructed.items():
        assert state.loc[team_id, ELO_COLUMN] == pytest.approx(elo)
        assert state.loc[team_id, "elo_after_last_match"] == pytest.approx(elo)


def test_state_extraction_does_not_change_feature_snapshots() -> None:
    matches, config = _five_match_frame()
    before = compute_team_elo_features(matches, config=config)
    compute_team_elo_state(matches, config=config)
    after = compute_team_elo_features(matches, config=config)
    pd.testing.assert_frame_equal(before, after)
    for column in TEAM_ELO_FEATURE_COLUMNS:
        pd.testing.assert_series_equal(before[column], after[column])


def test_ranking_is_strictly_descending_by_elo() -> None:
    matches, config = _five_match_frame()
    state = compute_team_elo_state(matches, config=config)
    ranked = rank_team_elo_state(state)
    assert ranked[ELO_COLUMN].is_monotonic_decreasing
    assert list(ranked.columns) == list(state.columns)


def test_ranking_breaks_ties_by_team_id_and_does_not_mutate_input() -> None:
    state = pd.DataFrame(
        {
            TEAM_ID_COLUMN: [30, 10, 20],
            ELO_COLUMN: [1600.0, 1600.0, 1500.0],
        }
    )
    original = state.copy()
    ranked = rank_team_elo_state(state)
    pd.testing.assert_frame_equal(state, original)
    assert list(ranked[TEAM_ID_COLUMN]) == [10, 30, 20]


def test_active_filter_uses_dataset_cutoff_not_wall_clock() -> None:
    """T_MAX is 2024-06-01. Wall-clock 'today' is far later than 90 days
    after that, so a wall-clock filter would mark every team inactive.
    The dataset-relative cutoff must still keep teams that played at T_MAX.
    """
    matches = _frame(
        [
            _match(
                1,
                start_time=T_OLD,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
            _match(
                2,
                start_time=T_MAX,
                radiant_team_id=TEAM_C,
                dire_team_id=TEAM_D,
                radiant_win=True,
            ),
        ]
    )
    state = compute_team_elo_state(matches)
    dataset_max = matches["start_time"].max()
    assert dataset_max == T_MAX
    assert datetime.now(UTC) - pd.Timestamp(dataset_max) > timedelta(
        days=DEFAULT_ACTIVE_DAYS
    )

    active = filter_active_team_elo(
        state, dataset_max_timestamp=dataset_max, active_days=DEFAULT_ACTIVE_DAYS
    )
    assert set(active[TEAM_ID_COLUMN]) == {TEAM_C, TEAM_D}

    cutoff = active_team_elo_cutoff(dataset_max, active_days=DEFAULT_ACTIVE_DAYS)
    assert cutoff == pd.Timestamp(T_MAX) - pd.Timedelta(days=DEFAULT_ACTIVE_DAYS)
    assert cutoff.year == 2024


def test_active_filter_includes_a_match_exactly_on_the_cutoff() -> None:
    matches = _frame(
        [
            _match(
                1,
                start_time=T_MAX - timedelta(days=90),
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
            _match(
                2,
                start_time=T_MAX,
                radiant_team_id=TEAM_C,
                dire_team_id=TEAM_D,
                radiant_win=True,
            ),
        ]
    )
    state = compute_team_elo_state(matches)
    active = filter_active_team_elo(
        state, dataset_max_timestamp=T_MAX, active_days=90
    )
    assert set(active[TEAM_ID_COLUMN]) == {TEAM_A, TEAM_B, TEAM_C, TEAM_D}


def test_empty_matches_yield_empty_state_with_schema() -> None:
    empty = pd.DataFrame(columns=["match_id", "start_time", "radiant_team_id", "dire_team_id", "radiant_win"])
    state = compute_team_elo_state(empty)
    assert len(state) == 0
    assert list(state.columns) == list(TEAM_ELO_STATE_COLUMNS)


def test_tied_last_group_records_pre_group_before_and_batched_after() -> None:
    config = EloConfig()
    matches = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
            _match(
                2,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_C,
                radiant_win=True,
            ),
        ]
    )
    state = compute_team_elo_state(matches, config=config).set_index(TEAM_ID_COLUMN)
    gain = config.k_factor * (
        1.0 - expected_score(config.initial_rating, config.initial_rating)
    )
    assert state.loc[TEAM_A, "elo_before_last_match"] == pytest.approx(
        config.initial_rating
    )
    assert state.loc[TEAM_A, ELO_COLUMN] == pytest.approx(
        config.initial_rating + 2 * gain
    )
    assert int(state.loc[TEAM_A, "last_group_n_matches"]) == 2
    assert int(state.loc[TEAM_A, "n_matches"]) == 2
    assert int(state.loc[TEAM_A, "wins"]) == 2


def test_elo_column_equals_elo_after_last_match() -> None:
    matches, config = _five_match_frame()
    state = compute_team_elo_state(matches, config=config)
    pd.testing.assert_series_equal(
        state[ELO_COLUMN], state["elo_after_last_match"], check_names=False
    )


def test_trajectory_current_elo_is_the_leaderboard_elo_column() -> None:
    """Trajectories rename `elo` to `current_elo`; they do not replay."""
    matches, config = _five_match_frame()
    ranked = rank_team_elo_state(compute_team_elo_state(matches, config=config))
    traj = team_elo_trajectories(ranked, n=3)
    pd.testing.assert_series_equal(
        traj[CURRENT_ELO_COLUMN],
        ranked[ELO_COLUMN].head(3).reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        traj[TEAM_ID_COLUMN],
        ranked[TEAM_ID_COLUMN].head(3).reset_index(drop=True),
        check_names=False,
    )


def test_trajectory_does_not_recompute_ratings() -> None:
    """A planted `elo` must come through as `current_elo` unchanged."""
    planted = 1796.160613
    state = pd.DataFrame(
        {
            TEAM_ID_COLUMN: [TEAM_A, TEAM_B],
            ELO_COLUMN: [planted, 1500.0],
            "starting_elo": [1500.0, 1500.0],
            "peak_elo": [1930.5, 1500.0],
            "lowest_elo": [1500.0, 1484.0],
            "n_matches": [294, 1],
        }
    )
    traj = team_elo_trajectories(state)
    assert traj.loc[0, CURRENT_ELO_COLUMN] == planted
    assert traj.loc[0, TEAM_ID_COLUMN] == TEAM_A
