"""Slice 15 leakage-safe pre-draft farming comparison: frozen k, no win model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import (
    build_snapshot_store,
    match_row,
    player_rows,
)

from dota_predictor.data.canonical_schema import MATCH_PLAYER_BOX_SCORE_COLUMNS
from dota_predictor.features.duckdb_layer import MATCH_PLAYERS_VIEW
from dota_predictor.features.player_farming_comparison import (
    FARMING_CAUSAL_B_COLUMN,
    PLAYER_FARMING_FEATURE_COLUMNS,
    PLAYER_FARMING_REQUIRED_COLUMNS,
    player_farming_comparison_from_players,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
    SLICE15_CANDIDATE_SPEC,
    SLICE15_FROZEN_SPECS,
)
from dota_predictor.training.player_farming_comparison import (
    build_player_farming_comparison,
    build_player_farming_state,
    run_player_farming_comparison_diagnostics,
)
from dota_predictor.training.player_farming_state import (
    CAUSAL_B_COLUMN,
    FROZEN_SHRINKAGE_K,
    attach_player_farming_state,
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
T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = datetime(2026, 2, 1, tzinfo=UTC)
T2 = datetime(2026, 3, 1, tzinfo=UTC)
T3 = datetime(2026, 4, 1, tzinfo=UTC)


def _appearance(
    *,
    match_id: int,
    player_id: int,
    start_time: datetime,
    position: int,
    duration: float,
    last_hits: float,
    side: str = "RADIANT",
    team_id: int = 100,
    hero_id: int = 1,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "hero_id": hero_id,
        "team_id": team_id,
        "side": side,
        "slot_in_side": position - 1,
        "position": f"POSITION_{position}",
        "position_number": float(position),
        "start_time": start_time,
        "game_version_id": 176,
        "duration_seconds": float(duration),
        "num_last_hits": float(last_hits),
        "team_won": 1,
        "elo_expected_win": 0.5,
        "kills": 1,
        "deaths": 1,
        "assists": 1,
        "gold_per_minute": 400,
        "experience_per_minute": 400,
        "num_denies": 0,
        "networth": 10000,
        "hero_damage": 10000,
        "tower_damage": 1000,
        "hero_healing": 0,
        "level": 20,
    }


def _match_players(
    match_id: int,
    start_time: datetime,
    duration: float,
    *,
    radiant_hits: tuple[float, float, float, float, float],
    dire_hits: tuple[float, float, float, float, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, last_hits in enumerate(radiant_hits, start=1):
        rows.append(
            _appearance(
                match_id=match_id,
                player_id=10 + position,
                start_time=start_time,
                position=position,
                duration=duration,
                last_hits=last_hits,
                side="RADIANT",
                team_id=100,
                hero_id=position,
            )
        )
    for position, last_hits in enumerate(dire_hits, start=1):
        rows.append(
            _appearance(
                match_id=match_id,
                player_id=20 + position,
                start_time=start_time,
                position=position,
                duration=duration,
                last_hits=last_hits,
                side="DIRE",
                team_id=200,
                hero_id=position + 5,
            )
        )
    return rows


def _causal_ready_two_sided() -> pd.DataFrame:
    rows = (
        _match_players(
            1,
            T0,
            1000.0,
            radiant_hits=(80.0, 60.0, 40.0, 20.0, 10.0),
            dire_hits=(50.0, 40.0, 30.0, 20.0, 10.0),
        )
        + _match_players(
            2,
            T1,
            2000.0,
            radiant_hits=(200.0, 160.0, 100.0, 50.0, 20.0),
            dire_hits=(90.0, 70.0, 50.0, 30.0, 15.0),
        )
        + _match_players(
            3,
            T2,
            1500.0,
            radiant_hits=(120.0, 90.0, 70.0, 40.0, 15.0),
            dire_hits=(80.0, 60.0, 45.0, 25.0, 12.0),
        )
    )
    return pd.DataFrame(rows)


def _fill_box_scores(
    players: list[dict[str, object]], *, last_hits: int = 300
) -> list[dict[str, object]]:
    for item in players:
        slot = int(item["slot_in_side"])
        item["position"] = POSITIONS[slot]
        item["num_last_hits"] = last_hits - slot * 40
        item["kills"] = 1
        item["deaths"] = 1
        item["assists"] = 1
        item["gold_per_minute"] = 400
        item["experience_per_minute"] = 400
        item["num_denies"] = 0
        item["networth"] = 10000
        item["hero_damage"] = 10000
        item["tower_damage"] = 1000
        item["hero_healing"] = 0
        item["level"] = 20
    return players


def test_builder_uses_frozen_k_not_a_search() -> None:
    assert FROZEN_SHRINKAGE_K == 5.0
    frame = pd.DataFrame(
        {
            "match_id": [1, 2],
            "player_id": [11, 11],
            "start_time": [T0, T1],
            "game_version_id": [176, 176],
            "team_id": [100, 100],
            "side": ["RADIANT", "RADIANT"],
            "slot_in_side": [0, 0],
            CAUSAL_B_COLUMN: [2.0, 0.0],
        }
    )
    state = build_player_farming_state(frame)
    at_t1 = state["start_time"] == T1
    assert float(state.loc[at_t1, "farming_shrunk_b"].iloc[0]) == pytest.approx(
        1.0 / (1.0 + FROZEN_SHRINKAGE_K) * 2.0
    )


def test_current_match_last_hits_cannot_change_that_match_comparison() -> None:
    frame = _causal_ready_two_sided()
    original_state = attach_player_farming_state(frame, k=FROZEN_SHRINKAGE_K)
    original = player_farming_comparison_from_players(original_state)
    mutated = frame.copy()
    mutated.loc[mutated["match_id"] == 3, "num_last_hits"] = 50_000.0
    rerun_state = attach_player_farming_state(mutated, k=FROZEN_SHRINKAGE_K)
    rerun = player_farming_comparison_from_players(rerun_state)
    left = original.loc[original["match_id"] == 3].iloc[0]
    right = rerun.loc[rerun["match_id"] == 3].iloc[0]
    for column in PLAYER_FARMING_FEATURE_COLUMNS:
        assert float(left[column]) == pytest.approx(float(right[column]))
    # Current causal B *did* move, so we are not accidentally using it.
    orig_b = original_state.loc[
        original_state["match_id"] == 3, CAUSAL_B_COLUMN
    ].to_numpy(dtype=float)
    new_b = rerun_state.loc[rerun_state["match_id"] == 3, CAUSAL_B_COLUMN].to_numpy(
        dtype=float
    )
    assert not np.allclose(orig_b, new_b, atol=1e-6, equal_nan=True)


def test_current_match_result_and_hero_cannot_change_comparison() -> None:
    frame = _causal_ready_two_sided()
    original = player_farming_comparison_from_players(
        attach_player_farming_state(frame, k=FROZEN_SHRINKAGE_K)
    )
    mutated = frame.copy()
    mutated.loc[mutated["match_id"] == 3, "team_won"] = 0
    mutated.loc[mutated["match_id"] == 3, "hero_id"] = 99
    rerun = player_farming_comparison_from_players(
        attach_player_farming_state(mutated, k=FROZEN_SHRINKAGE_K)
    )
    pd.testing.assert_series_equal(
        original.loc[original["match_id"] == 3, list(PLAYER_FARMING_FEATURE_COLUMNS)]
        .iloc[0]
        .reset_index(drop=True),
        rerun.loc[rerun["match_id"] == 3, list(PLAYER_FARMING_FEATURE_COLUMNS)]
        .iloc[0]
        .reset_index(drop=True),
        check_names=False,
    )


def test_future_match_cannot_change_earlier_comparison() -> None:
    frame = _causal_ready_two_sided()
    original = player_farming_comparison_from_players(
        attach_player_farming_state(frame, k=FROZEN_SHRINKAGE_K)
    )
    future = pd.DataFrame(
        _match_players(
            99,
            T3,
            1800.0,
            radiant_hits=(500.0, 10.0, 10.0, 10.0, 10.0),
            dire_hits=(10.0, 10.0, 10.0, 10.0, 500.0),
        )
    )
    combined = player_farming_comparison_from_players(
        attach_player_farming_state(
            pd.concat([frame, future], ignore_index=True), k=FROZEN_SHRINKAGE_K
        )
    )
    earlier = original.sort_values("match_id").reset_index(drop=True)
    later = (
        combined.loc[combined["match_id"].isin(original["match_id"])]
        .sort_values("match_id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(earlier, later)


def test_same_timestamp_matches_are_mutually_blind_in_comparison() -> None:
    extra_a = _match_players(
        30,
        T2,
        1500.0,
        radiant_hits=(400.0, 10.0, 10.0, 10.0, 10.0),
        dire_hits=(10.0, 10.0, 10.0, 10.0, 10.0),
    )
    extra_b = _match_players(
        30,
        T2,
        1500.0,
        radiant_hits=(10.0, 10.0, 10.0, 10.0, 10.0),
        dire_hits=(400.0, 10.0, 10.0, 10.0, 10.0),
    )
    base = _causal_ready_two_sided()
    left = player_farming_comparison_from_players(
        attach_player_farming_state(
            pd.concat([base, pd.DataFrame(extra_a)], ignore_index=True),
            k=FROZEN_SHRINKAGE_K,
        )
    )
    right = player_farming_comparison_from_players(
        attach_player_farming_state(
            pd.concat([base, pd.DataFrame(extra_b)], ignore_index=True),
            k=FROZEN_SHRINKAGE_K,
        )
    )
    core_left = left.loc[left["match_id"] == 3, "mean_farming_shrunk_b_diff"].iloc[0]
    core_right = right.loc[right["match_id"] == 3, "mean_farming_shrunk_b_diff"].iloc[0]
    assert float(core_left) == pytest.approx(float(core_right))


def test_required_columns_exclude_post_match_and_draft_keys() -> None:
    for name in (
        "hero_id",
        "num_last_hits",
        "duration_seconds",
        "position",
        "player_id",
        FARMING_CAUSAL_B_COLUMN,
    ):
        assert name not in PLAYER_FARMING_REQUIRED_COLUMNS


def test_development_cutoff_excludes_later_rows_from_diagnostics(
    tmp_path: Path,
) -> None:
    boundary = FROZEN_DEVELOPMENT_END
    later = boundary + timedelta(days=1)
    earlier = boundary - timedelta(days=20)
    mid = boundary - timedelta(days=10)
    matches = []
    players: list[dict[str, object]] = []
    for match_id, stamp, duration in (
        (1, earlier, 1000),
        (2, mid, 2000),
        (3, boundary, 1500),
        (4, later, 1800),
    ):
        row = match_row(
            match_id,
            start_time=stamp,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        )
        row["duration_seconds"] = duration
        matches.append(row)
        batch = player_rows(match_id, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        players.extend(_fill_box_scores(batch))

    with build_snapshot_store(tmp_path, matches=matches, players=players) as store:
        report = run_player_farming_comparison_diagnostics(store)
        comparison = build_player_farming_comparison(
            store, development_end=FROZEN_DEVELOPMENT_END
        )
        view_columns = store.relation(MATCH_PLAYERS_VIEW).columns

    assert report.n_development_matches == 3
    assert report.n_development_player_rows == 30
    assert report.n_holdout_excluded == 10
    assert report.frozen_k == FROZEN_SHRINKAGE_K
    assert report.development_end == FROZEN_DEVELOPMENT_END
    assert set(comparison["match_id"]) == {1, 2, 3}
    assert "mean_farming_shrunk_b_diff" in comparison.columns
    assert report.integrity["holdout_used_for_k"] is False
    assert report.integrity["holdout_used_for_feature"] is False
    assert report.integrity["k_re_searched"] is False
    assert report.integrity["ti2026_used_for_k"] is False
    assert report.integrity["model_trained"] is False
    assert report.integrity["win_model_benchmarked"] is False
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["comparison_in_feature_columns"] is False
    assert report.integrity["candidate_in_feature_columns"] is False
    assert report.integrity["causal_b_in_comparison_columns"] is False
    assert report.integrity["hero_id_in_required_columns"] is False
    assert report.integrity["slice15_candidate_uses_frozen_feature"] is True
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in view_columns


def test_comparison_columns_do_not_enter_feature_columns_or_slice9() -> None:
    for name in PLAYER_FARMING_FEATURE_COLUMNS:
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
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
    extra = set(SLICE15_CANDIDATE_SPEC.feature_columns) - set(
        SLICE9_FROZEN_SPECS[0].feature_columns
    )
    assert extra == set(PLAYER_FARMING_FEATURE_COLUMNS)
    assert extra.isdisjoint(FEATURE_COLUMNS)
    assert extra.isdisjoint(ALL_FEATURE_COLUMNS)
    assert tuple(spec.name for spec in SLICE15_FROZEN_SPECS) == (
        SLICE9_REFERENCE_SPEC_NAME,
        "logistic_elo_plus_player_farming",
    )
