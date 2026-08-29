"""Tests for the Step 4A chronological train/validation/test split
(`training.split`).

Datasets are built through the real Step 2/3/4A pipeline via
`training_helpers.build_dataset` (real Parquet + DuckDB +
`PreDraftSnapshot`), matching this repo's existing convention of never
short-circuiting the real contract with synthetic DataFrame shortcuts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from training_helpers import build_dataset, match_row, player_rows

from dota_predictor.training.dataset import ModelReadyDataset
from dota_predictor.training.split import (
    ChronologicalSplitConfig,
    ChronologicalSplitError,
    SplitBoundaries,
    chronological_split,
    resolve_split_boundaries,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _sequential_dataset(
    tmp_path: Path, timestamps: list[datetime]
) -> ModelReadyDataset:
    """One match per entry of `timestamps` (repeats are intentional --
    ties support the equal-start_time tests below), each between two
    brand-new teams/rosters so history features carry no signal that
    could interfere with pure split-boundary arithmetic."""
    matches = []
    players = []
    player_counter = 1
    for i, start_time in enumerate(timestamps):
        match_id = 1000 + i
        radiant_team_id = 2 * i + 1
        dire_team_id = 2 * i + 2
        radiant_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        dire_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        matches.append(
            match_row(
                match_id,
                start_time=start_time,
                radiant_team_id=radiant_team_id,
                dire_team_id=dire_team_id,
                radiant_win=(i % 2 == 0),
            )
        )
        players.extend(
            player_rows(match_id, radiant_ids=radiant_ids, dire_ids=dire_ids)
        )
    return build_dataset(tmp_path, matches=matches, players=players)


@pytest.fixture
def twenty_distinct_timestamps_dataset(tmp_path: Path) -> ModelReadyDataset:
    """20 matches, each at its own distinct `start_time` one day apart:
    0.70/0.15/0.15 lands on exact row counts (14/3/3) with no ties to
    complicate the arithmetic."""
    timestamps = [T0 + timedelta(days=i) for i in range(20)]
    return _sequential_dataset(tmp_path, timestamps)


@pytest.fixture
def tied_group_dataset(tmp_path: Path) -> ModelReadyDataset:
    """10 matches across 6 distinct timestamps, with deliberately-sized
    tied groups (3 matches share one timestamp, 2 share another)
    positioned so a naive row-count split would want to cut through the
    middle of a group."""
    timestamps = (
        [T0]
        + [T0 + timedelta(days=1)]
        + [T0 + timedelta(days=2)] * 3
        + [T0 + timedelta(days=3)]
        + [T0 + timedelta(days=4)] * 2
        + [T0 + timedelta(days=5)] * 2
    )
    return _sequential_dataset(tmp_path, timestamps)


# --- default proportions / basic shape -------------------------------------


def test_default_split_matches_70_15_15_when_timestamps_are_all_distinct(
    twenty_distinct_timestamps_dataset: ModelReadyDataset,
) -> None:
    split = chronological_split(twenty_distinct_timestamps_dataset)
    assert len(split.train) == 14
    assert len(split.validation) == 3
    assert len(split.test) == 3


def test_row_counts_sum_to_the_total(
    twenty_distinct_timestamps_dataset: ModelReadyDataset,
) -> None:
    split = chronological_split(twenty_distinct_timestamps_dataset)
    assert len(split.train) + len(split.validation) + len(split.test) == len(
        twenty_distinct_timestamps_dataset
    )


# --- no temporal overlap ----------------------------------------------------


def test_no_temporal_overlap_across_partitions(
    twenty_distinct_timestamps_dataset: ModelReadyDataset,
) -> None:
    split = chronological_split(twenty_distinct_timestamps_dataset)
    assert (
        split.train.context["start_time"].max()
        < split.validation.context["start_time"].min()
    )
    assert (
        split.validation.context["start_time"].max()
        < split.test.context["start_time"].min()
    )


def test_chronological_ordering_is_preserved_within_each_partition(
    twenty_distinct_timestamps_dataset: ModelReadyDataset,
) -> None:
    split = chronological_split(twenty_distinct_timestamps_dataset)
    for partition in (split.train, split.validation, split.test):
        assert partition.context["start_time"].is_monotonic_increasing


# --- equal-start_time groups never cross a split boundary ------------------


def test_tied_group_of_three_stays_together_in_one_partition(
    tied_group_dataset: ModelReadyDataset,
) -> None:
    config = ChronologicalSplitConfig(train_fraction=0.4, validation_fraction=0.3)
    split = chronological_split(tied_group_dataset, config=config)

    tied_timestamp = T0 + timedelta(days=2)
    tied_in_train = (split.train.context["start_time"] == tied_timestamp).sum()
    tied_in_validation = (
        split.validation.context["start_time"] == tied_timestamp
    ).sum()
    tied_in_test = (split.test.context["start_time"] == tied_timestamp).sum()

    assert tied_in_train == 3
    assert tied_in_validation == 0
    assert tied_in_test == 0


def test_tied_group_of_two_stays_together_and_split_sizes_are_group_aware(
    tied_group_dataset: ModelReadyDataset,
) -> None:
    """With train_fraction=0.4/validation_fraction=0.3 over this
    fixture's [1,1,3,1,2,2]-sized groups, a naive row-count cut would
    put train at 4 rows and validation at 3 -- but since ties are
    indivisible, train absorbs the whole 3-row tied group (5 rows) and
    validation absorbs the whole 2-row tied group at day 4 (3 rows),
    leaving the day-5 pair as test."""
    config = ChronologicalSplitConfig(train_fraction=0.4, validation_fraction=0.3)
    split = chronological_split(tied_group_dataset, config=config)

    assert len(split.train) == 5
    assert len(split.validation) == 3
    assert len(split.test) == 2

    day4_timestamp = T0 + timedelta(days=4)
    assert (split.validation.context["start_time"] == day4_timestamp).sum() == 2
    assert (split.train.context["start_time"] == day4_timestamp).sum() == 0
    assert (split.test.context["start_time"] == day4_timestamp).sum() == 0


def test_resolve_split_boundaries_always_lands_on_a_present_start_time(
    tied_group_dataset: ModelReadyDataset,
) -> None:
    config = ChronologicalSplitConfig(train_fraction=0.4, validation_fraction=0.3)
    boundaries = resolve_split_boundaries(tied_group_dataset.context, config)

    present = set(tied_group_dataset.context["start_time"])
    assert boundaries.train_end in present
    assert boundaries.validation_end in present


# --- determinism -------------------------------------------------------------


def test_split_is_deterministic(
    twenty_distinct_timestamps_dataset: ModelReadyDataset,
) -> None:
    first = chronological_split(twenty_distinct_timestamps_dataset)
    second = chronological_split(twenty_distinct_timestamps_dataset)

    pd.testing.assert_frame_equal(first.train.X, second.train.X)
    pd.testing.assert_series_equal(first.train.y, second.train.y)
    pd.testing.assert_frame_equal(first.validation.context, second.validation.context)
    assert first.boundaries == second.boundaries


# --- target class values preserved exactly ----------------------------------


def test_target_values_are_preserved_exactly_across_partitions(
    twenty_distinct_timestamps_dataset: ModelReadyDataset,
) -> None:
    split = chronological_split(twenty_distinct_timestamps_dataset)
    recombined = pd.concat(
        [split.train.y, split.validation.y, split.test.y], ignore_index=True
    )
    original = twenty_distinct_timestamps_dataset.y.reset_index(drop=True)
    pd.testing.assert_series_equal(recombined, original)


# --- context retained for auditing ------------------------------------------


def test_each_partition_context_retains_match_id_and_start_time(
    twenty_distinct_timestamps_dataset: ModelReadyDataset,
) -> None:
    split = chronological_split(twenty_distinct_timestamps_dataset)
    for partition in (split.train, split.validation, split.test):
        assert "match_id" in partition.context.columns
        assert "start_time" in partition.context.columns
        assert len(partition.context) == len(partition.X) == len(partition.y)


# --- missing values preserved through split ---------------------------------


def test_missing_values_survive_the_split_untouched(tmp_path: Path) -> None:
    """The earliest match in the dataset has zero team history (NULL
    win rate); after splitting it must still be NULL in whichever
    partition it lands in, not dropped or imputed."""
    t1, t2, t3 = T0, T0 + timedelta(days=30), T0 + timedelta(days=60)
    matches = [
        match_row(
            1, start_time=t1, radiant_team_id=1, dire_team_id=2, radiant_win=True
        ),
        match_row(
            2, start_time=t2, radiant_team_id=1, dire_team_id=3, radiant_win=True
        ),
        match_row(
            3, start_time=t3, radiant_team_id=1, dire_team_id=4, radiant_win=True
        ),
    ]
    players = (
        player_rows(1, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
        + player_rows(2, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(11, 12, 13, 14, 15))
        + player_rows(3, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(16, 17, 18, 19, 20))
    )
    dataset = build_dataset(tmp_path, matches=matches, players=players)

    boundaries = SplitBoundaries(train_end=t1, validation_end=t2)
    split = chronological_split(dataset, boundaries=boundaries)

    assert len(split.train) == 1
    assert len(split.validation) == 1
    assert len(split.test) == 1
    assert pd.isna(split.train.X.loc[0, "radiant_team_prior_win_rate"])


# --- explicit boundaries (date-based configurability) -----------------------


def test_explicit_boundaries_are_honored_instead_of_fractions(
    twenty_distinct_timestamps_dataset: ModelReadyDataset,
) -> None:
    context = twenty_distinct_timestamps_dataset.context
    sorted_times = sorted(context["start_time"].unique())
    boundaries = SplitBoundaries(
        train_end=sorted_times[9], validation_end=sorted_times[14]
    )

    split = chronological_split(
        twenty_distinct_timestamps_dataset, boundaries=boundaries
    )

    assert len(split.train) == 10
    assert len(split.validation) == 5
    assert len(split.test) == 5
    assert split.boundaries == boundaries


def test_giving_both_config_and_boundaries_raises(
    twenty_distinct_timestamps_dataset: ModelReadyDataset,
) -> None:
    with pytest.raises(ChronologicalSplitError):
        chronological_split(
            twenty_distinct_timestamps_dataset,
            config=ChronologicalSplitConfig(),
            boundaries=SplitBoundaries(
                train_end=T0, validation_end=T0 + timedelta(days=1)
            ),
        )


# --- config/boundary validation ---------------------------------------------


def test_split_config_rejects_fractions_that_leave_no_room_for_test() -> None:
    with pytest.raises(ChronologicalSplitError):
        ChronologicalSplitConfig(train_fraction=0.8, validation_fraction=0.3)


def test_split_config_rejects_out_of_range_fractions() -> None:
    with pytest.raises(ChronologicalSplitError):
        ChronologicalSplitConfig(train_fraction=1.5, validation_fraction=0.1)


def test_split_boundaries_reject_non_increasing_cutoffs() -> None:
    with pytest.raises(ChronologicalSplitError):
        SplitBoundaries(train_end=T0 + timedelta(days=1), validation_end=T0)
