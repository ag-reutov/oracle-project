"""Tests for expanding walk-forward block ablation.

Folds reuse the post-draft matrix and the six predefined Elo + draft
blocks. Equal-``start_time`` groups are never split. Paired deltas are
spec minus Elo. Game-version tables are diagnostics on OOS rows.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from training_helpers import (
    build_dataset,
    build_snapshot_store,
    match_row,
    player_rows,
)

from dota_predictor.training.evaluation import REGULARIZATION_CANDIDATES
from dota_predictor.training.feature_sets import POST_DRAFT_BLOCK_ABLATION_SPECS
from dota_predictor.training.post_draft import build_post_draft_model_ready_dataset
from dota_predictor.training.walk_forward import (
    ELO_BLOCK_SPEC_NAME,
    WalkForwardConfig,
    resolve_walk_forward_folds,
    run_post_draft_walk_forward,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


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


def _post_draft_dataset(tmp_path: Path, n: int = 36) -> object:
    matches = []
    players = []
    player_counter = 1
    block = max(n // 3, 1)
    for i in range(n):
        match_id = 1000 + i
        start_time = T0 + timedelta(days=i)
        radiant_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        dire_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        matches.append(
            match_row(
                match_id,
                start_time=start_time,
                radiant_team_id=2 * i + 1,
                dire_team_id=2 * i + 2,
                radiant_win=(i % 2 == 0),
                game_version_id=170 + (i // block),
            )
        )
        players.extend(
            player_rows(match_id, radiant_ids=radiant_ids, dire_ids=dire_ids)
        )
    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        return build_post_draft_model_ready_dataset(store)


def test_walk_forward_folds_are_expanding_and_oos_once(tmp_path: Path) -> None:
    timestamps = [T0 + timedelta(days=i) for i in range(20)]
    dataset = _sequential_dataset(tmp_path, timestamps)
    folds = resolve_walk_forward_folds(
        dataset, config=WalkForwardConfig(n_blocks=5)
    )
    assert len(folds) == 4
    seen: set[int] = set()
    for fold in folds:
        test_ids = set(fold.test.context["match_id"])
        train_ids = set(fold.train.context["match_id"])
        val_ids = set(fold.validation.context["match_id"])
        assert test_ids.isdisjoint(seen)
        seen |= test_ids
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert (
            fold.train.context["start_time"].max()
            < fold.validation.context["start_time"].min()
        )
        assert (
            fold.validation.context["start_time"].max()
            < fold.test.context["start_time"].min()
        )
    all_ids = set(dataset.context["match_id"])
    assert seen < all_ids
    first_block_times = dataset.context.loc[
        ~dataset.context["match_id"].isin(seen), "start_time"
    ]
    assert first_block_times.max() < folds[0].test.context["start_time"].min()
    for earlier, later in itertools.pairwise(folds):
        earlier_past = set(earlier.train.context["match_id"]) | set(
            earlier.validation.context["match_id"]
        )
        later_past = set(later.train.context["match_id"]) | set(
            later.validation.context["match_id"]
        )
        assert earlier_past < later_past
        assert set(earlier.test.context["match_id"]) <= later_past


def test_walk_forward_keeps_equal_start_time_groups_together(
    tmp_path: Path,
) -> None:
    timestamps = (
        [T0] * 3
        + [T0 + timedelta(days=1)] * 2
        + [T0 + timedelta(days=d) for d in range(2, 14)]
    )
    dataset = _sequential_dataset(tmp_path, timestamps)
    folds = resolve_walk_forward_folds(
        dataset, config=WalkForwardConfig(n_blocks=3)
    )
    for fold in folds:
        for partition in (fold.train, fold.validation, fold.test):
            times = set(partition.context["start_time"])
            full = dataset.context[dataset.context["start_time"].isin(times)]
            assert set(full["match_id"]) == set(partition.context["match_id"])


def test_walk_forward_block_ablation_paired_delta_and_versions(
    tmp_path: Path,
) -> None:
    dataset = _post_draft_dataset(tmp_path, n=36)
    report = run_post_draft_walk_forward(
        dataset, config=WalkForwardConfig(n_blocks=3)
    )
    expected = [spec.name for spec in POST_DRAFT_BLOCK_ABLATION_SPECS]
    assert [spec.name for spec in report.specs] == expected
    assert ELO_BLOCK_SPEC_NAME in expected
    assert set(report.fold_metrics["model"]) == set(expected)
    assert set(report.pooled_metrics["model"]) == set(expected)
    assert set(report.selected_C["C"]) <= set(REGULARIZATION_CANDIDATES)
    elo = report.pooled_metrics.loc[
        report.pooled_metrics["model"] == ELO_BLOCK_SPEC_NAME
    ].iloc[0]
    assert elo["mean_delta_vs_elo"] == pytest.approx(0.0)
    assert elo["frac_better_than_elo"] == pytest.approx(0.0)
    oos_elo = report.oos_predictions.loc[
        report.oos_predictions["model"] == ELO_BLOCK_SPEC_NAME
    ]
    assert oos_elo["match_id"].nunique() == len(oos_elo)
    assert set(report.version_breakdown["game_version_id"]).issubset(
        set(dataset.context["game_version_id"])
    )
    assert report.version_fold_counts["n"].sum() == len(oos_elo)
    for spec in POST_DRAFT_BLOCK_ABLATION_SPECS:
        assert set(spec.feature_columns).issubset(dataset.X.columns)
