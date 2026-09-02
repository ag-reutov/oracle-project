"""Slice 12 player-performance target diagnostics: no rating, no features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import (
    build_feature_store_config,
    build_snapshot_store,
    match_row,
    player_rows,
)

from dota_predictor.data.canonical_schema import MATCH_PLAYER_BOX_SCORE_COLUMNS
from dota_predictor.features.duckdb_layer import MATCH_PLAYERS_VIEW, connect
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.player_performance_target import (
    CANDIDATE_COLUMN_NAMES,
    attach_candidate_targets,
    build_player_performance_frame,
    elo_residualized,
    explicit_position_mask,
    ols_residual,
    parse_position_number,
    per_minute,
    position_adjusted,
    position_duration_residual,
    position_r_squared,
    position_standardized,
    prior_player_history,
    restrict_development,
    run_player_performance_target_diagnostics,
    slope_coefficient,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END

RADIANT_IDS = (11, 12, 13, 14, 15)
DIRE_IDS = (21, 22, 23, 24, 25)
POSITIONS = (
    "POSITION_1",
    "POSITION_2",
    "POSITION_3",
    "POSITION_4",
    "POSITION_5",
)


def _box_for_slot(slot: int) -> dict[str, int]:
    return {
        "kills": 10 - slot,
        "deaths": slot,
        "assists": 5 + slot,
        "gold_per_minute": 600 - slot * 80,
        "experience_per_minute": 550 - slot * 70,
        "num_last_hits": 300 - slot * 60,
        "num_denies": 20 - slot * 4,
        "networth": 20000 - slot * 3000,
        "hero_damage": 25000 - slot * 4000,
        "tower_damage": 3000 - slot * 500,
        "hero_healing": slot * 400,
        "level": 25 - slot,
    }


def _annotate_players(
    rows: list[dict[str, object]],
    *,
    extra_by_player: dict[int, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        slot = int(item["slot_in_side"])
        item["position"] = POSITIONS[slot]
        item.update(_box_for_slot(slot))
        player_id = int(item["player_id"])
        if extra_by_player and player_id in extra_by_player:
            item.update(extra_by_player[player_id])
        annotated.append(item)
    return annotated


def _frame_from_store(
    tmp_path: Path,
    *,
    matches: list[dict[str, object]],
    players: list[dict[str, object]],
) -> pd.DataFrame:
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        return build_player_performance_frame(store)


def test_parse_position_number_keeps_unknown_and_null_unimputed() -> None:
    assert parse_position_number("POSITION_1") == 1
    assert parse_position_number("POSITION_5") == 5
    assert np.isnan(parse_position_number(None))
    assert np.isnan(parse_position_number("UNKNOWN"))
    assert np.isnan(parse_position_number("FILTERED"))
    assert np.isnan(parse_position_number("ALL"))


def test_position_adjustment_matches_hand_computed_means() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [1, 1, 5, 5, np.nan],
            "gold_per_minute": [10.0, 20.0, 0.0, 4.0, 100.0],
        }
    )
    adjusted = position_adjusted(frame, "gold_per_minute")
    standardized = position_standardized(frame, "gold_per_minute")
    assert adjusted.iloc[0] == pytest.approx(-5.0)
    assert adjusted.iloc[1] == pytest.approx(5.0)
    assert adjusted.iloc[2] == pytest.approx(-2.0)
    assert adjusted.iloc[3] == pytest.approx(2.0)
    assert np.isnan(adjusted.iloc[4])
    assert standardized.iloc[0] == pytest.approx(-1.0)
    assert standardized.iloc[1] == pytest.approx(1.0)
    assert np.isnan(standardized.iloc[4])
    ss_within = 50.0 + 8.0
    eligible = np.array([10.0, 20.0, 0.0, 4.0])
    ss_tot = float(np.sum((eligible - eligible.mean()) ** 2))
    assert position_r_squared(frame, "gold_per_minute") == pytest.approx(
        1.0 - ss_within / ss_tot
    )


def test_null_position_is_excluded_from_role_residuals() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [1.0, 1.0, np.nan],
            "gold_per_minute": [100.0, 200.0, 10_000.0],
        }
    )
    adjusted = position_adjusted(frame, "gold_per_minute")
    assert adjusted.iloc[0] == pytest.approx(-50.0)
    assert adjusted.iloc[1] == pytest.approx(50.0)
    assert np.isnan(adjusted.iloc[2])
    assert not explicit_position_mask(frame).iloc[2]


def test_zero_is_preserved_distinct_from_null() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [5.0, 5.0],
            "deaths": [0, 0],
            "hero_healing": [0, None],
        }
    )
    assert frame["deaths"].tolist() == [0, 0]
    assert pd.isna(frame["hero_healing"].iloc[1])
    adjusted = position_adjusted(frame, "deaths")
    assert adjusted.tolist() == [0.0, 0.0]
    healing = position_adjusted(frame, "hero_healing")
    assert healing.iloc[0] == pytest.approx(0.0)
    assert np.isnan(healing.iloc[1])


def test_duration_residualization_fits_a_perfect_line() -> None:
    frame = pd.DataFrame(
        {
            "position_number": [1.0, 1.0],
            "duration_seconds": [1800.0, 3600.0],
            "num_last_hits": [90.0, 180.0],
        }
    )
    residual = position_duration_residual(frame, "num_last_hits")
    assert residual.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert residual.iloc[1] == pytest.approx(0.0, abs=1e-9)
    rates = per_minute(frame["num_last_hits"], frame["duration_seconds"])
    assert rates.tolist() == [3.0, 3.0]


def test_development_cutoff_includes_boundary_and_drops_later(
    tmp_path: Path,
) -> None:
    boundary = FROZEN_DEVELOPMENT_END
    later = boundary + timedelta(days=1)
    earlier = boundary - timedelta(days=1)
    matches = [
        match_row(
            1,
            start_time=earlier,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
        match_row(
            2,
            start_time=boundary,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=False,
        ),
        match_row(
            3,
            start_time=later,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
    ]
    players = _annotate_players(
        player_rows(1, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(2, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(3, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
    )
    frame = _frame_from_store(tmp_path, matches=matches, players=players)
    development = restrict_development(frame)
    assert set(development["match_id"].unique()) == {1, 2}
    assert 3 not in set(development["match_id"].unique())
    assert pd.Timestamp(development["start_time"].max()) <= pd.Timestamp(boundary)


def test_elo_expected_win_is_pre_match_not_post_match(tmp_path: Path) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 1, 8, tzinfo=UTC)
    matches = [
        match_row(
            10,
            start_time=t0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
        match_row(
            11,
            start_time=t1,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
    ]
    players = _annotate_players(
        player_rows(10, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(11, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
    )
    frame = _frame_from_store(tmp_path, matches=matches, players=players)
    first = frame.loc[frame["match_id"] == 10, "elo_expected_win"]
    second = frame.loc[
        (frame["match_id"] == 11) & (frame["side"] == "RADIANT"),
        "elo_expected_win",
    ]
    assert first.nunique() == 1
    assert first.iloc[0] == pytest.approx(0.5)
    assert second.iloc[0] > 0.5
    gpm = frame.loc[frame["match_id"] == 11, "gold_per_minute"]
    elo = frame.loc[frame["match_id"] == 11, "elo_expected_win"]
    residual = elo_residualized(gpm, elo)
    beta = slope_coefficient(gpm, elo)
    assert residual.notna().all()
    assert np.isfinite(beta)


def test_prior_history_is_strictly_earlier_and_same_timestamp_blind() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    t2 = datetime(2026, 3, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "player_id": [11, 11, 11, 11],
            "match_id": [1, 2, 3, 4],
            "start_time": [t0, t1, t1, t2],
            "gold_per_minute": [10.0, 20.0, 30.0, 40.0],
        }
    )
    prior_mean, prior_n = prior_player_history(frame, "gold_per_minute")
    assert prior_n.tolist() == [0, 1, 1, 3]
    assert np.isnan(prior_mean.iloc[0])
    assert prior_mean.iloc[1] == pytest.approx(10.0)
    assert prior_mean.iloc[2] == pytest.approx(10.0)
    assert prior_mean.iloc[3] == pytest.approx(20.0)


def test_future_rows_do_not_contribute_to_repeatability() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "player_id": [11, 11],
            "match_id": [2, 1],
            "start_time": [t1, t0],
            "gold_per_minute": [99.0, 10.0],
        }
    )
    prior_mean, prior_n = prior_player_history(frame, "gold_per_minute")
    later = frame["start_time"] == t1
    earlier = frame["start_time"] == t0
    assert int(prior_n.loc[later].iloc[0]) == 1
    assert prior_mean.loc[later].iloc[0] == pytest.approx(10.0)
    assert int(prior_n.loc[earlier].iloc[0]) == 0


def test_candidate_calculations_are_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "match_id": [1, 1, 2, 2],
            "player_id": [11, 15, 11, 15],
            "hero_id": [1, 5, 1, 5],
            "team_id": [100, 100, 100, 100],
            "side": ["RADIANT", "RADIANT", "RADIANT", "RADIANT"],
            "position": ["POSITION_1", "POSITION_5", "POSITION_1", "POSITION_5"],
            "position_number": [1.0, 5.0, 1.0, 5.0],
            "team_won": [1, 1, 0, 0],
            "elo_expected_win": [0.5, 0.5, 0.6, 0.6],
            "duration_seconds": [1800, 1800, 2400, 2400],
            "duration_minutes": [30.0, 30.0, 40.0, 40.0],
            "kills": [10, 1, 8, 2],
            "deaths": [1, 8, 2, 7],
            "assists": [4, 14, 5, 12],
            "gold_per_minute": [600, 220, 580, 240],
            "experience_per_minute": [550, 300, 540, 310],
            "num_last_hits": [300, 40, 360, 50],
            "num_denies": [20, 2, 18, 3],
            "networth": [20000, 8000, 24000, 9000],
            "hero_damage": [20000, 6000, 22000, 7000],
            "tower_damage": [4000, 200, 3500, 100],
            "hero_healing": [0, 3000, 0, 2800],
            "level": [25, 18, 24, 19],
        }
    )
    first = attach_candidate_targets(frame)
    second = attach_candidate_targets(frame)
    for column in CANDIDATE_COLUMN_NAMES:
        left = first[column].to_numpy(dtype=float)
        right = second[column].to_numpy(dtype=float)
        np.testing.assert_allclose(left, right, equal_nan=True)
        assert first[column].notna().all()


def test_candidates_do_not_enter_feature_columns_or_specs() -> None:
    for name in CANDIDATE_COLUMN_NAMES:
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 33
    assert tuple(spec.name for spec in SLICE9_FROZEN_SPECS) == (
        SLICE9_REFERENCE_SPEC_NAME,
        SLICE9_CANDIDATE_SPEC_NAME,
    )
    assert [spec.name for spec in POST_DRAFT_BLOCK_ABLATION_SPECS] == [
        "logistic_elo_only",
        "logistic_elo_plus_player_hero",
        "logistic_elo_plus_team_hero",
        "logistic_elo_plus_hero_meta",
        "logistic_elo_plus_player_and_team_hero",
        "logistic_elo_plus_all_three",
    ]


def test_box_scores_stay_off_the_feature_match_players_view(tmp_path: Path) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    matches = [
        match_row(
            1,
            start_time=t0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        )
    ]
    players = _annotate_players(
        player_rows(1, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
    )
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
        frame = build_player_performance_frame(store)
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in view_columns
        assert column in frame.columns
    assert frame["deaths"].notna().all()


def test_missing_position_row_keeps_box_score_and_skips_residuals(
    tmp_path: Path,
) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    matches = [
        match_row(
            1,
            start_time=t0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        )
    ]
    players = _annotate_players(
        player_rows(1, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS),
        extra_by_player={
            11: {"position": None, "gold_per_minute": 0, "deaths": 0},
        },
    )
    frame = _frame_from_store(tmp_path, matches=matches, players=players)
    row = frame.loc[frame["player_id"] == 11].iloc[0]
    assert pd.isna(row["position_number"])
    assert row["gold_per_minute"] == 0
    assert row["deaths"] == 0
    attached = attach_candidate_targets(frame)
    missing = attached.loc[attached["player_id"] == 11, "gpm_position_standardized"]
    assert missing.isna().all()
    others = attached.loc[attached["player_id"] != 11, "gpm_position_standardized"]
    assert others.notna().all()


def test_ols_residual_and_full_diagnostic_run(tmp_path: Path) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    matches = [
        match_row(
            1,
            start_time=t0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
            game_version_id=176,
        ),
        match_row(
            2,
            start_time=t1,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=False,
            game_version_id=177,
        ),
    ]
    players = _annotate_players(
        player_rows(1, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(2, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
    )
    y = pd.Series([1.0, 3.0, 5.0])
    x = pd.DataFrame({"intercept": [1.0, 1.0, 1.0], "x": [0.0, 1.0, 2.0]})
    residual, coef = ols_residual(y, x)
    assert residual.to_numpy() == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert coef[1] == pytest.approx(2.0)

    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        report = run_player_performance_target_diagnostics(store)
    assert report.n_development_matches == 2
    assert report.n_development_player_rows == 20
    assert report.integrity["candidate_in_feature_columns"] is False
    assert report.integrity["player_rating_persisted"] is False
    assert report.integrity["model_trained"] is False
    assert report.integrity["box_scores_in_feature_match_players_view"] is False
    assert report.n_holdout_excluded == 0
    assert "gpm_position_standardized" in set(report.candidate_quality["candidate"])
    assert not report.candidate_position_means.empty
    gpm_means = report.candidate_position_means.loc[
        report.candidate_position_means["candidate"] == "gpm_position_standardized"
    ].iloc[0]
    for number in (1, 2, 3, 4, 5):
        assert gpm_means[f"pos{number}_mean"] == pytest.approx(0.0, abs=1e-9)
