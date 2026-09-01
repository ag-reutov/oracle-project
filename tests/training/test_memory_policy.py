"""Tests for walk-forward training-memory selection.

Covers eligibility masks and fold restriction only: expanding past,
calendar cutoffs, version windows, chronology, and trailing validation
inside the memory policy. Does not re-test logistic fitting.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from training_helpers import build_dataset, match_row, player_rows

from dota_predictor.training.memory_policy import (
    POLICY_CURRENT_PLUS_PREVIOUS_VERSION,
    POLICY_EXPANDING,
    POLICY_LAST_180D,
    POLICY_LAST_365D,
    calendar_cutoff,
    current_and_previous_version_ids,
    eligible_past_mask,
    restrict_fold_to_memory,
)
from dota_predictor.training.walk_forward import (
    WalkForwardConfig,
    resolve_walk_forward_folds,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _context(
    rows: list[tuple[int, datetime, int]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [row[0] for row in rows],
            "start_time": [row[1] for row in rows],
            "game_version_id": [row[2] for row in rows],
        }
    )


def _ids(context: pd.DataFrame, mask: pd.Series) -> list[int]:
    return list(context.loc[mask, "match_id"])


def test_expanding_includes_all_eligible_past_and_excludes_future() -> None:
    past_end = T0 + timedelta(days=10)
    context = _context(
        [
            (1, T0, 1),
            (2, T0 + timedelta(days=10), 1),
            (3, T0 + timedelta(days=10, seconds=1), 1),
            (4, T0 + timedelta(days=40), 2),
        ]
    )
    mask = eligible_past_mask(
        context, policy=POLICY_EXPANDING, past_end=past_end
    )
    assert _ids(context, mask) == [1, 2]
    assert 3 not in _ids(context, mask)
    assert 4 not in _ids(context, mask)


def test_last_365d_cutoff_is_inclusive_of_the_boundary() -> None:
    past_end = T0 + timedelta(days=365)
    cutoff = calendar_cutoff(past_end, days=365)
    assert cutoff == pd.Timestamp(T0)
    context = _context(
        [
            (1, T0 - timedelta(seconds=1), 1),
            (2, T0, 1),
            (3, T0 + timedelta(days=1), 1),
            (4, past_end, 1),
            (5, past_end + timedelta(seconds=1), 1),
        ]
    )
    mask = eligible_past_mask(
        context, policy=POLICY_LAST_365D, past_end=past_end
    )
    assert _ids(context, mask) == [2, 3, 4]
    assert 1 not in _ids(context, mask)
    assert 5 not in _ids(context, mask)


def test_last_180d_cutoff_and_future_exclusion() -> None:
    past_end = T0 + timedelta(days=200)
    context = _context(
        [
            (1, T0, 1),
            (2, past_end - timedelta(days=180), 1),
            (3, past_end - timedelta(days=179), 1),
            (4, past_end, 1),
            (5, past_end + timedelta(days=1), 1),
        ]
    )
    mask = eligible_past_mask(
        context, policy=POLICY_LAST_180D, past_end=past_end
    )
    assert _ids(context, mask) == [2, 3, 4]
    assert 1 not in _ids(context, mask)
    assert 5 not in _ids(context, mask)


def test_equal_timestamp_peers_stay_together_at_the_cutoff() -> None:
    past_end = T0 + timedelta(days=180)
    cutoff_time = past_end - timedelta(days=180)
    context = _context(
        [
            (1, cutoff_time - timedelta(days=1), 1),
            (2, cutoff_time, 1),
            (3, cutoff_time, 1),
            (4, cutoff_time, 1),
            (5, past_end + timedelta(days=1), 1),
        ]
    )
    mask = eligible_past_mask(
        context, policy=POLICY_LAST_180D, past_end=past_end
    )
    assert _ids(context, mask) == [2, 3, 4]


def test_current_plus_previous_uses_past_versions_only() -> None:
    past_end = T0 + timedelta(days=20)
    context = _context(
        [
            (1, T0, 10),
            (2, T0 + timedelta(days=5), 11),
            (3, T0 + timedelta(days=15), 11),
            (4, T0 + timedelta(days=20), 12),
            (5, T0 + timedelta(days=25), 12),
            (6, T0 + timedelta(days=30), 13),
        ]
    )
    current, previous = current_and_previous_version_ids(
        context, past_end=past_end
    )
    assert current == 12
    assert previous == 11
    mask = eligible_past_mask(
        context,
        policy=POLICY_CURRENT_PLUS_PREVIOUS_VERSION,
        past_end=past_end,
    )
    assert _ids(context, mask) == [2, 3, 4]
    assert 1 not in _ids(context, mask)
    assert 5 not in _ids(context, mask)
    assert 6 not in _ids(context, mask)


def test_current_plus_previous_excludes_future_rows_of_the_current_version() -> None:
    past_end = T0 + timedelta(days=10)
    context = _context(
        [
            (1, T0, 20),
            (2, T0 + timedelta(days=10), 21),
            (3, T0 + timedelta(days=11), 21),
        ]
    )
    mask = eligible_past_mask(
        context,
        policy=POLICY_CURRENT_PLUS_PREVIOUS_VERSION,
        past_end=past_end,
    )
    assert _ids(context, mask) == [1, 2]
    assert 3 not in _ids(context, mask)


def test_current_plus_previous_does_not_peek_at_evaluation_versions() -> None:
    past_end = T0 + timedelta(days=8)
    context = _context(
        [
            (1, T0, 30),
            (2, T0 + timedelta(days=8), 31),
            (3, T0 + timedelta(days=9), 32),
            (4, T0 + timedelta(days=12), 32),
        ]
    )
    current, previous = current_and_previous_version_ids(
        context, past_end=past_end
    )
    assert current == 31
    assert previous == 30
    mask = eligible_past_mask(
        context,
        policy=POLICY_CURRENT_PLUS_PREVIOUS_VERSION,
        past_end=past_end,
    )
    assert _ids(context, mask) == [1, 2]
    assert set(_ids(context, mask)).isdisjoint({3, 4})


def _sequential_dataset(
    tmp_path: Path,
    timestamps: list[datetime],
    *,
    versions: list[int] | None = None,
):
    matches = []
    players = []
    player_counter = 1
    for i, start_time in enumerate(timestamps):
        match_id = 1000 + i
        radiant_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        dire_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        version = versions[i] if versions is not None else 176
        matches.append(
            match_row(
                match_id,
                start_time=start_time,
                radiant_team_id=2 * i + 1,
                dire_team_id=2 * i + 2,
                radiant_win=(i % 2 == 0),
                game_version_id=version,
            )
        )
        players.extend(
            player_rows(match_id, radiant_ids=radiant_ids, dire_ids=dire_ids)
        )
    return build_dataset(tmp_path, matches=matches, players=players)


def test_expanding_restriction_matches_original_walk_forward_past(
    tmp_path: Path,
) -> None:
    timestamps = [T0 + timedelta(days=i) for i in range(20)]
    dataset = _sequential_dataset(tmp_path, timestamps)
    config = WalkForwardConfig(n_blocks=5)
    folds = resolve_walk_forward_folds(dataset, config=config)
    fold = folds[0]
    restricted = restrict_fold_to_memory(
        dataset,
        fold,
        policy=POLICY_EXPANDING,
        train_fraction_of_past=config.train_fraction_of_past,
    )
    assert restricted.skipped is False
    assert restricted.train is not None
    assert restricted.validation is not None
    assert set(restricted.train.context["match_id"]) == set(
        fold.train.context["match_id"]
    )
    assert set(restricted.validation.context["match_id"]) == set(
        fold.validation.context["match_id"]
    )
    assert set(restricted.test.context["match_id"]) == set(
        fold.test.context["match_id"]
    )


def test_validation_stays_inside_calendar_memory_window(tmp_path: Path) -> None:
    timestamps = [T0 + timedelta(days=i) for i in range(40)]
    dataset = _sequential_dataset(tmp_path, timestamps)
    config = WalkForwardConfig(n_blocks=4)
    fold = resolve_walk_forward_folds(dataset, config=config)[0]
    restricted = restrict_fold_to_memory(
        dataset,
        fold,
        policy=POLICY_LAST_180D,
        train_fraction_of_past=config.train_fraction_of_past,
    )
    assert restricted.skipped is False
    assert restricted.train is not None
    assert restricted.validation is not None
    cutoff = calendar_cutoff(fold.validation_end, days=180)
    for partition in (restricted.train, restricted.validation):
        times = pd.to_datetime(partition.context["start_time"])
        assert (times >= cutoff).all()
        assert (times <= pd.Timestamp(fold.validation_end)).all()
    assert (
        restricted.validation.context["start_time"].max()
        < fold.test.context["start_time"].min()
    )


def test_restricted_test_window_is_the_original_oos_fold(tmp_path: Path) -> None:
    timestamps = [T0 + timedelta(days=i) for i in range(24)]
    versions = [170] * 8 + [171] * 8 + [172] * 8
    dataset = _sequential_dataset(tmp_path, timestamps, versions=versions)
    config = WalkForwardConfig(n_blocks=3)
    fold = resolve_walk_forward_folds(dataset, config=config)[-1]
    restricted = restrict_fold_to_memory(
        dataset,
        fold,
        policy=POLICY_CURRENT_PLUS_PREVIOUS_VERSION,
        train_fraction_of_past=config.train_fraction_of_past,
    )
    assert restricted.skipped is False
    assert restricted.train is not None
    assert restricted.validation is not None
    assert set(restricted.test.context["match_id"]) == set(
        fold.test.context["match_id"]
    )
    allowed = {
        restricted.current_version_id,
        restricted.previous_version_id,
    }
    train_versions = set(restricted.train.context["game_version_id"])
    val_versions = set(restricted.validation.context["game_version_id"])
    assert train_versions <= allowed
    assert val_versions <= allowed
    future_current = dataset.context.loc[
        (dataset.context["game_version_id"] == restricted.current_version_id)
        & (
            pd.to_datetime(dataset.context["start_time"])
            > pd.Timestamp(fold.validation_end)
        )
    ]
    assert set(future_current["match_id"]).isdisjoint(
        set(restricted.train.context["match_id"])
        | set(restricted.validation.context["match_id"])
    )


def test_unknown_policy_is_rejected() -> None:
    context = _context([(1, T0, 1)])
    with pytest.raises(ValueError, match="unknown memory policy"):
        eligible_past_mask(context, policy="LAST_90D", past_end=T0)
