"""Slice 19 leakage-safe pre-draft combat comparison: frozen k, no win model."""

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
from dota_predictor.features.player_combat_comparison import (
    COMBAT_CAUSAL_C_COLUMN,
    PLAYER_COMBAT_FEATURE_COLUMNS,
    PLAYER_COMBAT_REQUIRED_COLUMNS,
    player_combat_comparison_from_players,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.combat_performance_target import (
    COMBAT_C_POSITION,
    FROZEN_COMBAT_CANDIDATE,
)
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
    SLICE19_CANDIDATE_SPEC,
    SLICE19_FROZEN_SPECS,
)
from dota_predictor.training.player_combat_comparison import (
    build_player_combat_comparison,
    build_player_combat_state,
    run_player_combat_comparison_diagnostics,
)
from dota_predictor.training.player_combat_state import (
    CAUSAL_C_COLUMN,
    FROZEN_COMBAT_SHRINKAGE_K,
    attach_player_combat_state,
)
from dota_predictor.training.player_farming_state import (
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
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
    hero_damage: float,
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
        "duration_seconds": 1800.0,
        "num_last_hits": 100.0,
        "team_won": 1,
        "elo_expected_win": 0.5,
        "kills": 1,
        "deaths": 1,
        "assists": 1,
        "gold_per_minute": 400,
        "experience_per_minute": 400,
        "num_denies": 0,
        "networth": 10000,
        "hero_damage": float(hero_damage),
        "tower_damage": 1000,
        "hero_healing": 0,
        "level": 20,
    }


def _match_players(
    match_id: int,
    start_time: datetime,
    *,
    radiant_damage: tuple[float, float, float, float, float],
    dire_damage: tuple[float, float, float, float, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, damage in enumerate(radiant_damage, start=1):
        rows.append(
            _appearance(
                match_id=match_id,
                player_id=10 + position,
                start_time=start_time,
                position=position,
                hero_damage=damage,
                side="RADIANT",
                team_id=100,
                hero_id=position,
            )
        )
    for position, damage in enumerate(dire_damage, start=1):
        rows.append(
            _appearance(
                match_id=match_id,
                player_id=20 + position,
                start_time=start_time,
                position=position,
                hero_damage=damage,
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
            radiant_damage=(10.0, 8.0, 6.0, 4.0, 2.0),
            dire_damage=(5.0, 5.0, 5.0, 5.0, 5.0),
        )
        + _match_players(
            2,
            T1,
            radiant_damage=(20.0, 10.0, 5.0, 3.0, 2.0),
            dire_damage=(8.0, 7.0, 6.0, 5.0, 4.0),
        )
        + _match_players(
            3,
            T2,
            radiant_damage=(12.0, 9.0, 6.0, 3.0, 0.0),
            dire_damage=(7.0, 6.0, 5.0, 4.0, 3.0),
        )
    )
    return pd.DataFrame(rows)


def _fill_box_scores(
    players: list[dict[str, object]], *, hero_damage: int = 20000
) -> list[dict[str, object]]:
    for item in players:
        slot = int(item["slot_in_side"])
        item["position"] = POSITIONS[slot]
        item["num_last_hits"] = 300 - slot * 40
        item["kills"] = 1
        item["deaths"] = 1
        item["assists"] = 1
        item["gold_per_minute"] = 400
        item["experience_per_minute"] = 400
        item["num_denies"] = 0
        item["networth"] = 10000
        item["hero_damage"] = hero_damage - slot * 3000
        item["tower_damage"] = 1000
        item["hero_healing"] = 0
        item["level"] = 20
    return players


def test_builder_uses_frozen_combat_k_not_a_search() -> None:
    assert FROZEN_COMBAT_SHRINKAGE_K == 20.0
    frame = pd.DataFrame(
        {
            "match_id": [1, 2],
            "player_id": [11, 11],
            "start_time": [T0, T1],
            "game_version_id": [176, 176],
            "team_id": [100, 100],
            "side": ["RADIANT", "RADIANT"],
            "slot_in_side": [0, 0],
            CAUSAL_C_COLUMN: [0.20, 0.0],
        }
    )
    state = build_player_combat_state(frame)
    at_t1 = state["start_time"] == T1
    assert float(state.loc[at_t1, "combat_shrunk_c"].iloc[0]) == pytest.approx(
        1.0 / (1.0 + FROZEN_COMBAT_SHRINKAGE_K) * 0.20
    )


def test_repeated_appearances_accumulate_state() -> None:
    frame = pd.concat(
        [
            _causal_ready_two_sided(),
            pd.DataFrame(
                _match_players(
                    4,
                    T3,
                    radiant_damage=(15.0, 10.0, 8.0, 4.0, 2.0),
                    dire_damage=(9.0, 8.0, 6.0, 5.0, 4.0),
                )
            ),
        ],
        ignore_index=True,
    )
    state = attach_player_combat_state(frame, k=FROZEN_COMBAT_SHRINKAGE_K)
    player = state.loc[state["player_id"] == 11].sort_values("start_time")
    values = player["combat_shrunk_c"].to_numpy(dtype=float)
    counts = player["combat_prior_n"].to_numpy(dtype=float)
    causal = player[CAUSAL_C_COLUMN].to_numpy(dtype=float)
    assert counts[0] == 0
    assert values[0] == pytest.approx(0.0)
    assert np.isnan(causal[0])
    # T0 C is NULL, so T1 still has n=0. T1 C is finite and enters T2.
    assert counts[1] == 0
    assert values[1] == pytest.approx(0.0)
    assert np.isfinite(causal[1])
    assert counts[2] == 1
    assert counts[3] == 2
    assert values[2] != pytest.approx(values[3])
    assert values[2] != pytest.approx(0.0)


def test_current_match_combat_cannot_change_that_match_comparison() -> None:
    frame = _causal_ready_two_sided()
    original_state = attach_player_combat_state(frame, k=FROZEN_COMBAT_SHRINKAGE_K)
    original = player_combat_comparison_from_players(original_state)
    mutated = frame.copy()
    current = mutated["match_id"] == 3
    mutated.loc[current, "hero_damage"] = 50_000.0
    mutated.loc[current, "kills"] = 99
    mutated.loc[current, "assists"] = 99
    mutated.loc[current, "deaths"] = 99
    rerun_state = attach_player_combat_state(mutated, k=FROZEN_COMBAT_SHRINKAGE_K)
    rerun = player_combat_comparison_from_players(rerun_state)
    left = original.loc[original["match_id"] == 3].iloc[0]
    right = rerun.loc[rerun["match_id"] == 3].iloc[0]
    for column in PLAYER_COMBAT_FEATURE_COLUMNS:
        assert float(left[column]) == pytest.approx(float(right[column]))
    orig_c = original_state.loc[
        original_state["match_id"] == 3, CAUSAL_C_COLUMN
    ].to_numpy(dtype=float)
    new_c = rerun_state.loc[rerun_state["match_id"] == 3, CAUSAL_C_COLUMN].to_numpy(
        dtype=float
    )
    assert not np.allclose(orig_c, new_c, atol=1e-6, equal_nan=True)


def test_current_result_hero_and_position_cannot_change_comparison() -> None:
    frame = _causal_ready_two_sided()
    original = player_combat_comparison_from_players(
        attach_player_combat_state(frame, k=FROZEN_COMBAT_SHRINKAGE_K)
    )
    mutated = frame.copy()
    current = mutated["match_id"] == 3
    mutated.loc[current, "team_won"] = 0
    mutated.loc[current, "hero_id"] = 99
    mutated.loc[current, "position"] = "POSITION_5"
    mutated.loc[current, "position_number"] = 5.0
    rerun = player_combat_comparison_from_players(
        attach_player_combat_state(mutated, k=FROZEN_COMBAT_SHRINKAGE_K)
    )
    pd.testing.assert_series_equal(
        original.loc[original["match_id"] == 3, list(PLAYER_COMBAT_FEATURE_COLUMNS)]
        .iloc[0]
        .reset_index(drop=True),
        rerun.loc[rerun["match_id"] == 3, list(PLAYER_COMBAT_FEATURE_COLUMNS)]
        .iloc[0]
        .reset_index(drop=True),
        check_names=False,
    )


def test_future_match_cannot_change_earlier_comparison() -> None:
    frame = _causal_ready_two_sided()
    original = player_combat_comparison_from_players(
        attach_player_combat_state(frame, k=FROZEN_COMBAT_SHRINKAGE_K)
    )
    future = pd.DataFrame(
        _match_players(
            99,
            T3,
            radiant_damage=(500.0, 1.0, 1.0, 1.0, 1.0),
            dire_damage=(1.0, 1.0, 1.0, 1.0, 500.0),
        )
    )
    combined = player_combat_comparison_from_players(
        attach_player_combat_state(
            pd.concat([frame, future], ignore_index=True), k=FROZEN_COMBAT_SHRINKAGE_K
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
        radiant_damage=(400.0, 1.0, 1.0, 1.0, 1.0),
        dire_damage=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    extra_b = _match_players(
        30,
        T2,
        radiant_damage=(1.0, 1.0, 1.0, 1.0, 1.0),
        dire_damage=(400.0, 1.0, 1.0, 1.0, 1.0),
    )
    base = _causal_ready_two_sided()
    left = player_combat_comparison_from_players(
        attach_player_combat_state(
            pd.concat([base, pd.DataFrame(extra_a)], ignore_index=True),
            k=FROZEN_COMBAT_SHRINKAGE_K,
        )
    )
    right = player_combat_comparison_from_players(
        attach_player_combat_state(
            pd.concat([base, pd.DataFrame(extra_b)], ignore_index=True),
            k=FROZEN_COMBAT_SHRINKAGE_K,
        )
    )
    core_left = left.loc[left["match_id"] == 3, "mean_combat_shrunk_c_diff"].iloc[0]
    core_right = right.loc[right["match_id"] == 3, "mean_combat_shrunk_c_diff"].iloc[0]
    assert float(core_left) == pytest.approx(float(core_right))


def test_swapping_radiant_dire_negates_feature() -> None:
    frame = _causal_ready_two_sided()
    original = player_combat_comparison_from_players(
        attach_player_combat_state(frame, k=FROZEN_COMBAT_SHRINKAGE_K)
    )
    swapped = frame.copy()
    swapped["side"] = swapped["side"].map({"RADIANT": "DIRE", "DIRE": "RADIANT"})
    rerun = player_combat_comparison_from_players(
        attach_player_combat_state(swapped, k=FROZEN_COMBAT_SHRINKAGE_K)
    )
    merged = original.merge(rerun, on="match_id", suffixes=("_old", "_new"))
    np.testing.assert_allclose(
        merged["mean_combat_shrunk_c_diff_new"].to_numpy(dtype=float),
        -merged["mean_combat_shrunk_c_diff_old"].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_required_columns_exclude_post_match_and_draft_keys() -> None:
    for name in (
        "hero_id",
        "position",
        "hero_damage",
        "kills",
        "assists",
        "deaths",
        "num_last_hits",
        "duration_seconds",
        "networth",
        COMBAT_CAUSAL_C_COLUMN,
    ):
        assert name not in PLAYER_COMBAT_REQUIRED_COLUMNS


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
        report = run_player_combat_comparison_diagnostics(store)
        comparison = build_player_combat_comparison(
            store, development_end=FROZEN_DEVELOPMENT_END
        )
        view_columns = store.relation(MATCH_PLAYERS_VIEW).columns

    assert report.n_development_matches == 3
    assert report.n_development_player_rows == 30
    assert report.n_holdout_excluded == 10
    assert report.frozen_combat_k == FROZEN_COMBAT_SHRINKAGE_K
    assert report.frozen_farming_k == FROZEN_SHRINKAGE_K
    assert report.development_end == FROZEN_DEVELOPMENT_END
    assert set(comparison["match_id"]) == {1, 2, 3}
    assert "mean_combat_shrunk_c_diff" in comparison.columns
    assert report.integrity["holdout_used_for_k"] is False
    assert report.integrity["holdout_used_for_feature"] is False
    assert report.integrity["k_re_searched"] is False
    assert report.integrity["ti2026_used_for_k"] is False
    assert report.integrity["model_trained"] is False
    assert report.integrity["win_model_benchmarked"] is False
    assert report.integrity["holdout_scored"] is False
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["comparison_in_feature_columns"] is False
    assert report.integrity["candidate_in_feature_columns"] is False
    assert report.integrity["causal_c_in_required_columns"] is False
    assert report.integrity["hero_id_in_required_columns"] is False
    assert report.integrity["position_in_required_columns"] is False
    assert report.integrity["slice19_candidate_uses_frozen_feature"] is True
    assert report.integrity["evidence_in_candidate_spec"] is False
    assert report.integrity["combat_k_is_20"] is True
    assert report.integrity["farming_k_is_5"] is True
    assert report.integrity["slice17_candidate_unchanged"] is True
    assert report.integrity["alternative_combat_aggregation_searched"] is False
    assert report.integrity["stratz_called"] is False
    assert report.roster_integrity["n_complete_10_player_roster"] == 3
    assert report.roster_integrity["n_complete_combat_state_join"] == 3
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in view_columns


def test_comparison_columns_do_not_enter_feature_columns_or_slice9() -> None:
    assert FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION
    assert FROZEN_CANDIDATE_B == CANDIDATE_B
    assert FROZEN_COMBAT_SHRINKAGE_K == 20.0
    assert FROZEN_SHRINKAGE_K == 5.0
    for name in PLAYER_COMBAT_FEATURE_COLUMNS:
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
    extra = set(SLICE19_CANDIDATE_SPEC.feature_columns) - set(
        SLICE9_FROZEN_SPECS[0].feature_columns
    )
    assert extra == set(PLAYER_COMBAT_FEATURE_COLUMNS)
    assert extra.isdisjoint(FEATURE_COLUMNS)
    assert extra.isdisjoint(ALL_FEATURE_COLUMNS)
    assert tuple(spec.name for spec in SLICE19_FROZEN_SPECS) == (
        SLICE9_REFERENCE_SPEC_NAME,
        "logistic_elo_plus_player_combat",
    )
