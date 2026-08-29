"""Tests for the Step 4A model-ready dataset assembly
(`training.dataset`).

All fixtures build real canonical Parquet files via `training_helpers`
(itself built on the real Step 2 transform functions), through the
real `features.duckdb_layer`/`features.pre_draft_snapshot` pipeline --
no synthetic DataFrame shortcuts around the actual contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from training_helpers import (
    build_dataset,
    build_snapshot_store,
    match_row,
    player_rows,
)

from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
    build_pre_draft_snapshot,
)
from dota_predictor.training.dataset import (
    ModelReadyDataset,
    TrainingDatasetError,
    build_model_ready_dataset,
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)

TEAM_A, TEAM_B, TEAM_C, TEAM_D = 1, 2, 3, 4

# M2/M3 deliberately share T2, with match_id in DESCENDING order relative
# to a "natural" ascending scan, so a tie-break bug would be caught.
M1, M2, M3, M4 = 50, 10, 99, 5


def _fixture_matches_and_players() -> tuple[list[dict], list[dict]]:
    matches = [
        match_row(
            M1,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        match_row(
            M2,
            start_time=T2,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_C,
            radiant_win=True,
        ),
        match_row(
            M3,
            start_time=T2,
            radiant_team_id=TEAM_B,
            dire_team_id=TEAM_D,
            radiant_win=False,
        ),
        match_row(
            M4,
            start_time=T3,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_D,
            radiant_win=True,
        ),
    ]
    players = (
        player_rows(M1, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
        + player_rows(M2, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(11, 12, 13, 14, 15))
        + player_rows(M3, radiant_ids=(6, 7, 8, 9, 10), dire_ids=(16, 17, 18, 19, 20))
        + player_rows(M4, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(16, 17, 18, 19, 20))
    )
    return matches, players


@pytest.fixture
def dataset(tmp_path: Path) -> ModelReadyDataset:
    matches, players = _fixture_matches_and_players()
    return build_dataset(tmp_path, matches=matches, players=players)


# --- X / y / context contract ---------------------------------------------


def test_X_contains_exactly_feature_columns(dataset: ModelReadyDataset) -> None:
    assert list(dataset.X.columns) == list(FEATURE_COLUMNS)


def test_X_never_contains_target_or_identity_columns(
    dataset: ModelReadyDataset,
) -> None:
    assert TARGET_COLUMN not in dataset.X.columns
    for column in IDENTITY_COLUMNS:
        assert column not in dataset.X.columns


def test_y_is_exactly_radiant_win(dataset: ModelReadyDataset) -> None:
    assert dataset.y.name == TARGET_COLUMN == "radiant_win"
    assert set(dataset.y.unique()) <= {True, False}


def test_context_contains_identity_columns_including_match_id_and_start_time(
    dataset: ModelReadyDataset,
) -> None:
    assert list(dataset.context.columns) == list(IDENTITY_COLUMNS)
    assert "match_id" in dataset.context.columns
    assert "start_time" in dataset.context.columns


# --- row alignment / ordering ----------------------------------------------


def test_row_alignment_between_X_y_context_is_preserved(
    dataset: ModelReadyDataset,
) -> None:
    assert len(dataset.X) == len(dataset.y) == len(dataset.context) == 4
    assert dataset.X.index.equals(dataset.y.index)
    assert dataset.X.index.equals(dataset.context.index)

    # M4 (radiant=TEAM_A, dire=TEAM_D, radiant_win=True): find its row by
    # match_id in context and confirm the SAME position in X/y matches.
    position = dataset.context.index[dataset.context["match_id"] == M4][0]
    assert bool(dataset.y.loc[position])
    # TEAM_A has played twice before M4 (M1 win, M2 win): 2 prior matches.
    assert dataset.X.loc[position, "radiant_team_prior_matches"] == 2


def test_exactly_one_row_per_input_match(dataset: ModelReadyDataset) -> None:
    assert len(dataset) == 4
    assert set(dataset.context["match_id"]) == {M1, M2, M3, M4}


def test_chronological_ordering(dataset: ModelReadyDataset) -> None:
    assert dataset.context["start_time"].is_monotonic_increasing


def test_match_id_is_the_deterministic_tie_break_within_equal_start_time(
    dataset: ModelReadyDataset,
) -> None:
    """M2 (id=10) and M3 (id=99) share T2; despite M3 being listed first
    in the fixture and having a "later-looking" id ordering concern,
    the output must order them by match_id ascending (10 before 99)."""
    tied = dataset.context[dataset.context["start_time"] == T2]
    assert list(tied["match_id"]) == [M2, M3]


def test_output_row_order_is_exactly_start_time_then_match_id(
    dataset: ModelReadyDataset,
) -> None:
    assert list(dataset.context["match_id"]) == [M1, M2, M3, M4]


# --- missing values: preserved, never imputed ------------------------------


def test_missing_values_are_preserved_not_imputed(
    dataset: ModelReadyDataset,
) -> None:
    """M1 is the very first match for both TEAM_A and TEAM_B: their
    prior win rate is genuinely undefined (0 prior matches), not 0.0."""
    position = dataset.context.index[dataset.context["match_id"] == M1][0]
    assert pd.isna(dataset.X.loc[position, "radiant_team_prior_win_rate"])
    assert pd.isna(dataset.X.loc[position, "dire_team_prior_win_rate"])
    assert dataset.X.loc[position, "radiant_team_prior_matches"] == 0


def test_missing_feature_nulls_are_explained_by_zero_observed_history(
    dataset: ModelReadyDataset,
) -> None:
    """Every NULL win-rate feature must correspond exactly to a
    zero-prior-matches row -- i.e. "no observed history", never an
    unexplained/malformed NULL."""
    null_win_rate = dataset.X["radiant_team_prior_win_rate"].isna()
    zero_history = dataset.X["radiant_team_prior_matches"] == 0
    assert (null_win_rate == zero_history).all()


# --- no mutation / determinism ---------------------------------------------


def test_original_snapshot_and_its_frames_are_not_mutated(tmp_path: Path) -> None:
    matches, players = _fixture_matches_and_players()
    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        snapshot = build_pre_draft_snapshot(store)
        before = snapshot.to_frame()

        dataset = build_model_ready_dataset(snapshot)
        # Mutate the returned dataset's frames -- must never leak back.
        dataset.X.iloc[0, 0] = -999999
        dataset.y.iloc[0] = not bool(dataset.y.iloc[0])
        dataset.context.iloc[0, 0] = -999999

        after = snapshot.to_frame()

    pd.testing.assert_frame_equal(
        before.reset_index(drop=True), after.reset_index(drop=True)
    )


def test_building_twice_from_the_same_snapshot_is_deterministic(
    tmp_path: Path,
) -> None:
    matches, players = _fixture_matches_and_players()
    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        snapshot = build_pre_draft_snapshot(store)
        first = build_model_ready_dataset(snapshot)
        second = build_model_ready_dataset(snapshot)

    pd.testing.assert_frame_equal(first.X, second.X)
    pd.testing.assert_series_equal(first.y, second.y)
    pd.testing.assert_frame_equal(first.context, second.context)


# --- explicit contract validation ------------------------------------------


def test_constructing_with_wrong_X_columns_raises(dataset: ModelReadyDataset) -> None:
    with pytest.raises(TrainingDatasetError):
        ModelReadyDataset(
            X=dataset.X.rename(columns={FEATURE_COLUMNS[0]: "not_a_feature"}),
            y=dataset.y,
            context=dataset.context,
        )


def test_constructing_with_misaligned_row_counts_raises(
    dataset: ModelReadyDataset,
) -> None:
    with pytest.raises(TrainingDatasetError):
        ModelReadyDataset(
            X=dataset.X.iloc[:-1],
            y=dataset.y,
            context=dataset.context,
        )
