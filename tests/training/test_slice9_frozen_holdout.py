"""Tests for the Slice 9 frozen-model temporal holdout protocol.

No new features. Does not retune Slice 8 gates or production
FEATURE_COLUMNS. The real holdout is not scored here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from training_helpers import build_snapshot_store, match_row, player_rows

from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.dataset import TrainingDatasetError
from dota_predictor.training.evaluation import REGULARIZATION_CANDIDATES
from dota_predictor.training.feature_sets import (
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    SLICE8_INTERACTION_COLUMNS,
    SLICE9_CANDIDATE_SPEC,
    SLICE9_CANDIDATE_SPEC_NAME,
)
from dota_predictor.training.slice8_player_hero_gating import (
    build_slice8_model_ready_dataset,
    run_slice8_player_hero_gating_benchmark,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    FROZEN_HOLDOUT_EVALUATION_FILENAME,
    FrozenHoldoutAlreadyEvaluatedError,
    FrozenHoldoutEmptyError,
    assert_development_frame_excludes_holdout,
    chronology_bins,
    development_end_from_slice8_frame,
    evaluate_frozen_holdout,
    holdout_mask,
    inventory_holdout,
    record_frozen_holdout_protocol,
    resolve_frozen_holdout_split,
    utc_datetime,
)
from dota_predictor.training.walk_forward import WalkForwardConfig

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _sequential_store(tmp_path: Path, n: int = 24, *, start: int = 0):
    matches = []
    players = []
    player_counter = 1
    for i in range(start, start + n):
        match_id = 1000 + i
        start_time = T0 + timedelta(days=i)
        radiant_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        dire_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        if i >= 5:
            radiant_ids = (1, 2, 3, 4, 5)
            dire_ids = (6, 7, 8, 9, 10)
        matches.append(
            match_row(
                match_id,
                start_time=start_time,
                radiant_team_id=2 * i + 1,
                dire_team_id=2 * i + 2,
                radiant_win=(i % 2 == 0),
                game_version_id=170 + (i // 8),
            )
        )
        players.extend(
            player_rows(match_id, radiant_ids=radiant_ids, dire_ids=dire_ids)
        )
    return build_snapshot_store(tmp_path, matches=matches, players=players)


def _development_and_holdout_store(
    tmp_path: Path,
    *,
    n_development: int = 24,
    n_holdout: int = 6,
    holdout_league_id: int = 1,
):
    matches = []
    players = []
    player_counter = 1
    n_total = n_development + n_holdout
    for i in range(n_total):
        match_id = 1000 + i
        if i < n_development:
            start_time = T0 + timedelta(days=i)
        else:
            start_time = T0 + timedelta(days=n_development + 10 + (i - n_development))
        radiant_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        dire_ids = tuple(range(player_counter, player_counter + 5))
        player_counter += 5
        if i >= 5:
            radiant_ids = (1, 2, 3, 4, 5)
            dire_ids = (6, 7, 8, 9, 10)
        matches.append(
            match_row(
                match_id,
                start_time=start_time,
                radiant_team_id=2 * i + 1,
                dire_team_id=2 * i + 2,
                radiant_win=(i % 2 == 0),
                game_version_id=170 + (i // 8),
                league_id=holdout_league_id if i >= n_development else 1,
            )
        )
        players.extend(
            player_rows(match_id, radiant_ids=radiant_ids, dire_ids=dire_ids)
        )
    return build_snapshot_store(tmp_path, matches=matches, players=players)


def test_frozen_boundary_is_timezone_aware_utc() -> None:
    assert FROZEN_DEVELOPMENT_END.tzinfo is UTC
    assert utc_datetime(FROZEN_DEVELOPMENT_END) == FROZEN_DEVELOPMENT_END
    paris = pd.Timestamp("2026-07-19 19:49:01+02:00")
    assert utc_datetime(paris) == FROZEN_DEVELOPMENT_END


def test_holdout_mask_is_strictly_after_boundary() -> None:
    times = pd.Series(
        [
            pd.Timestamp(FROZEN_DEVELOPMENT_END) - pd.Timedelta(seconds=1),
            pd.Timestamp(FROZEN_DEVELOPMENT_END),
            pd.Timestamp(FROZEN_DEVELOPMENT_END) + pd.Timedelta(seconds=1),
        ]
    )
    mask = holdout_mask(times, FROZEN_DEVELOPMENT_END)
    assert list(mask) == [False, False, True]


def test_inventory_reports_n_and_range_without_predictions() -> None:
    context = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "start_time": [
                FROZEN_DEVELOPMENT_END - timedelta(days=2),
                FROZEN_DEVELOPMENT_END,
                FROZEN_DEVELOPMENT_END + timedelta(days=1),
                FROZEN_DEVELOPMENT_END + timedelta(days=3),
            ],
            "league_id": [10, 10, 20, 20],
            "game_version_id": [182, 182, 183, 183],
        }
    )
    inventory = inventory_holdout(context, development_end=FROZEN_DEVELOPMENT_END)
    assert inventory.n == 2
    assert inventory.start == FROZEN_DEVELOPMENT_END + timedelta(days=1)
    assert inventory.end == FROZEN_DEVELOPMENT_END + timedelta(days=3)
    assert inventory.match_ids == (3, 4)
    assert inventory.n_leagues == 1
    assert "p_spec" not in context.columns


def test_slice8_frame_end_matches_last_fold_test_end(tmp_path: Path) -> None:
    config = WalkForwardConfig(n_blocks=3)
    with _sequential_store(tmp_path, n=18) as store:
        assembly = build_slice8_model_ready_dataset(store)
        frame_end = development_end_from_slice8_frame(
            assembly.dataset, config=config
        )
    assert frame_end == utc_datetime(assembly.dataset.context["start_time"].max())


def test_record_protocol_freezes_career_spec_and_does_not_evaluate(
    tmp_path: Path,
) -> None:
    n_development = 18
    n_holdout = 6
    development_end = T0 + timedelta(days=n_development - 1)
    config = WalkForwardConfig(n_blocks=3)
    with _development_and_holdout_store(
        tmp_path, n_development=n_development, n_holdout=n_holdout
    ) as store:
        protocol = record_frozen_holdout_protocol(
            store, config=config, development_end=development_end
        )
    assert protocol.evaluated is False
    assert protocol.candidate_spec == SLICE9_CANDIDATE_SPEC
    assert protocol.candidate_spec.name == SLICE9_CANDIDATE_SPEC_NAME
    assert protocol.candidate_spec.feature_columns == ELO_PLUS_PLAYER_HERO_COLUMNS
    assert set(protocol.candidate_spec.feature_columns).isdisjoint(
        SLICE8_INTERACTION_COLUMNS
    )
    assert protocol.regularization_candidates == REGULARIZATION_CANDIDATES
    assert protocol.development_end == development_end
    assert protocol.n_development == n_development
    assert protocol.holdout.n == n_holdout
    assert protocol.canonical_later.n == n_holdout
    assert protocol.holdout.start == T0 + timedelta(days=n_development + 10)
    assert protocol.holdout.end == T0 + timedelta(
        days=n_development + 10 + n_holdout - 1
    )
    assert protocol.holdout.start > protocol.development_end


def test_frozen_split_keeps_holdout_out_of_train_and_validation(
    tmp_path: Path,
) -> None:
    n_development = 18
    n_holdout = 6
    development_end = T0 + timedelta(days=n_development - 1)
    config = WalkForwardConfig(n_blocks=3)
    with _development_and_holdout_store(
        tmp_path, n_development=n_development, n_holdout=n_holdout
    ) as store:
        assembly = build_slice8_model_ready_dataset(store)
        split = resolve_frozen_holdout_split(
            assembly.dataset,
            development_end=development_end,
            config=config,
            require_holdout=True,
        )
    assert len(split.train) + len(split.validation) == n_development
    assert len(split.holdout) == n_holdout
    assert utc_datetime(split.train.context["start_time"].max()) < utc_datetime(
        split.validation.context["start_time"].min()
    )
    assert utc_datetime(split.validation.context["start_time"].max()) <= development_end
    assert utc_datetime(split.holdout.context["start_time"].min()) > development_end
    train_ids = set(split.train.context["match_id"])
    val_ids = set(split.validation.context["match_id"])
    holdout_ids = set(split.holdout.context["match_id"])
    assert train_ids.isdisjoint(holdout_ids)
    assert val_ids.isdisjoint(holdout_ids)
    assert holdout_ids == set(range(1000 + n_development, 1000 + n_development + n_holdout))


def test_empty_holdout_inventory_and_blocked_scoring(tmp_path: Path) -> None:
    config = WalkForwardConfig(n_blocks=3)
    with _sequential_store(tmp_path, n=18) as store:
        assembly = build_slice8_model_ready_dataset(store)
        development_end = utc_datetime(assembly.dataset.context["start_time"].max())
        protocol = record_frozen_holdout_protocol(
            store, config=config, development_end=development_end
        )
        with pytest.raises(FrozenHoldoutEmptyError):
            resolve_frozen_holdout_split(
                assembly.dataset,
                development_end=development_end,
                config=config,
                require_holdout=True,
            )
    assert protocol.holdout.n == 0
    assert protocol.holdout.start is None
    assert protocol.evaluated is False


def test_slice8_runner_refuses_later_matches_after_frozen_boundary(
    tmp_path: Path,
) -> None:
    matches = []
    players = []
    player_counter = 1
    times = [FROZEN_DEVELOPMENT_END + timedelta(days=i) for i in range(18)]
    for i, start_time in enumerate(times):
        match_id = 9000 + i
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
            )
        )
        players.extend(
            player_rows(match_id, radiant_ids=radiant_ids, dire_ids=dire_ids)
        )
    config = WalkForwardConfig(n_blocks=3)
    with build_snapshot_store(
        tmp_path, matches=matches, players=players
    ) as store, pytest.raises(TrainingDatasetError, match="untouched holdout"):
        run_slice8_player_hero_gating_benchmark(store, config=config)


def test_guard_allows_slice8_fixture_dates_before_freeze() -> None:
    context = pd.DataFrame(
        {
            "match_id": [1, 2],
            "start_time": [T0, T0 + timedelta(days=10)],
        }
    )
    assert_development_frame_excludes_holdout(context)


def test_frozen_candidate_is_not_a_production_feature_column_change() -> None:
    extra = set(SLICE9_CANDIDATE_SPEC.feature_columns) - set(FEATURE_COLUMNS)
    assert extra == set(ELO_PLUS_PLAYER_HERO_COLUMNS) - set(FEATURE_COLUMNS)
    assert extra.isdisjoint(SLICE8_INTERACTION_COLUMNS)


def test_chronology_bins_are_equal_count_and_ignore_outcomes() -> None:
    times = pd.Series(
        [T0 + timedelta(hours=i) for i in range(6)],
        name="start_time",
    )
    bins = chronology_bins(times)
    assert list(bins) == ["early", "early", "middle", "middle", "late", "late"]


def test_evaluate_frozen_holdout_scores_once_without_using_holdout_for_c(
    tmp_path: Path,
) -> None:
    n_development = 18
    n_holdout = 6
    development_end = T0 + timedelta(days=n_development - 1)
    config = WalkForwardConfig(n_blocks=3)
    output_dir = tmp_path / "eval"
    with _development_and_holdout_store(
        tmp_path,
        n_development=n_development,
        n_holdout=n_holdout,
        holdout_league_id=19719,
    ) as store:
        report = evaluate_frozen_holdout(
            store,
            config=config,
            development_end=development_end,
            expected_holdout_n=n_holdout,
            expected_holdout_league_id=19719,
            output_dir=output_dir,
        )
        with pytest.raises(FrozenHoldoutAlreadyEvaluatedError, match="already evaluated"):
            evaluate_frozen_holdout(
                store,
                config=config,
                development_end=development_end,
                expected_holdout_n=n_holdout,
                expected_holdout_league_id=19719,
                output_dir=output_dir,
            )

    assert report.protocol.evaluated is True
    assert report.protocol.candidate_spec == SLICE9_CANDIDATE_SPEC
    assert set(report.protocol.candidate_spec.feature_columns).isdisjoint(
        SLICE8_INTERACTION_COLUMNS
    )
    assert report.protocol.regularization_candidates == REGULARIZATION_CANDIDATES
    assert report.candidate_metrics.n_samples == n_holdout
    assert report.reference_metrics.n_samples == n_holdout
    assert len(report.predictions) == n_holdout
    assert report.paired_delta_log_loss == pytest.approx(
        report.mean_paired_log_loss_diff
    )
    assert report.n_candidate_better_log_loss == int(
        (report.predictions["delta_log_loss"] < 0).sum()
    )
    assert report.bootstrap_delta_log_loss_ci95[0] <= report.paired_delta_log_loss
    assert report.paired_delta_log_loss <= report.bootstrap_delta_log_loss_ci95[1]
    for spec_name, c in report.selected_C.items():
        assert spec_name in {
            "logistic_elo_only",
            "logistic_elo_plus_player_hero",
        }
        assert c in REGULARIZATION_CANDIDATES
    holdout_ids = set(report.predictions["match_id"])
    train_ids = set(report.split.train.context["match_id"])
    val_ids = set(report.split.validation.context["match_id"])
    assert holdout_ids.isdisjoint(train_ids)
    assert holdout_ids.isdisjoint(val_ids)
    assert set(report.predictions["league_id"]) == {19719}
    assert (output_dir / FROZEN_HOLDOUT_EVALUATION_FILENAME).is_file()
    assert (output_dir / "predictions.parquet").is_file()
    assert report.chronology["n"].sum() == n_holdout
    assert report.winner_side["n"].sum() == n_holdout
    assert report.career_evidence["n"].sum() == n_holdout


def test_evaluate_refuses_wrong_holdout_league(tmp_path: Path) -> None:
    development_end = T0 + timedelta(days=17)
    config = WalkForwardConfig(n_blocks=3)
    with _development_and_holdout_store(
        tmp_path, n_development=18, n_holdout=6, holdout_league_id=2
    ) as store, pytest.raises(TrainingDatasetError, match="holdout leagues"):
        evaluate_frozen_holdout(
            store,
            config=config,
            development_end=development_end,
            expected_holdout_n=6,
            expected_holdout_league_id=19719,
        )


def test_already_evaluated_lock_is_checked_before_refit(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / FROZEN_HOLDOUT_EVALUATION_FILENAME).write_text(
        '{"evaluated": true}\n', encoding="utf-8"
    )
    with _development_and_holdout_store(
        tmp_path
    ) as store, pytest.raises(FrozenHoldoutAlreadyEvaluatedError):
        evaluate_frozen_holdout(
            store,
            config=WalkForwardConfig(n_blocks=3),
            development_end=T0 + timedelta(days=23),
            output_dir=output_dir,
        )
