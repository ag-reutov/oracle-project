"""Slice 25 causal Player × Position hero-pool state.

Temporal leakage, explicit-position buckets, scoring wrapper, holdout.
Research state only; not a production feature.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import (
    build_feature_store_config,
    match_row,
    player_rows,
)

from dota_predictor.data.canonical_schema import MATCH_PLAYER_BOX_SCORE_COLUMNS
from dota_predictor.features.availability import (
    SnapshotStage,
    columns_allowed_for_stage,
)
from dota_predictor.features.duckdb_layer import MATCH_PLAYERS_VIEW, connect
from dota_predictor.features.hero_meta import RECENT_WINDOW_DAYS
from dota_predictor.features.player_position import RECENT_POSITION_WINDOWS
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_CANDIDATE_SPEC_NAME,
    SLICE9_FROZEN_SPECS,
    SLICE9_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.hero_position_meta_state import RECENT_WINDOW_DAYS_ALT
from dota_predictor.training.player_farming_state import development_tune_end
from dota_predictor.training.player_hero_pool_state import (
    HIERARCHICAL_K_GRID,
    POOL_WINDOW_SPECS,
    SCORING_MIXTURE_EPSILON,
    SLICE25_DIAGNOSTIC_ONLY,
    SLICE25_FROZEN_COMPONENTS,
    SLICE25_STATE_COLUMNS,
    _PlayerRolePool,
    _cross_position_from_pools,
    attach_player_hero_pool_state,
    classify_slice25,
    effective_pool_size,
    pool_entropy,
    run_player_hero_pool_diagnostics,
    score_distribution,
    scoring_candidates,
    select_hierarchical_k,
    slice25_report_to_jsonable,
)
from dota_predictor.training.player_performance_target import restrict_development
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
T4 = datetime(2026, 5, 1, tzinfo=UTC)


def _row(
    *,
    match_id: int,
    player_id: int,
    hero_id: int,
    position: int | None,
    start_time: datetime,
    game_version_id: int = 176,
    position_label: str | None = None,
) -> dict[str, object]:
    if position is None:
        number = float("nan")
        label = position_label if position_label is not None else "UNKNOWN"
    else:
        number = float(position)
        label = f"POSITION_{position}"
    return {
        "match_id": match_id,
        "player_id": player_id,
        "hero_id": hero_id,
        "position_number": number,
        "position": label,
        "start_time": start_time,
        "game_version_id": game_version_id,
        "team_won": 1,
        "duration_seconds": 1800,
        "num_last_hits": 200,
        "hero_damage": 10_000,
        "expected_position": 5,
        "slot_in_side": 0,
    }


def _annotate_players(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        slot = int(item["slot_in_side"])
        item["position"] = POSITIONS[slot]
        item["num_last_hits"] = 300 - slot * 40
        item["kills"] = 1
        item["deaths"] = 1
        item["assists"] = 1
        item["gold_per_minute"] = 400
        item["experience_per_minute"] = 400
        item["num_denies"] = 0
        item["networth"] = 10_000
        item["hero_damage"] = 12_000 - slot * 1_500
        item["tower_damage"] = 1_000
        item["hero_healing"] = 0
        item["level"] = 20
        annotated.append(item)
    return annotated


def test_windows_follow_existing_conventions() -> None:
    names = tuple(spec.name for spec in POOL_WINDOW_SPECS)
    assert names == (
        "expanding",
        "last_5_at_role",
        "last_10_at_role",
        "last_20_at_role",
        "recent_90d",
        "recent_180d",
        "current_version",
        "current_plus_previous",
    )
    assert RECENT_POSITION_WINDOWS == (5, 10, 20)
    assert RECENT_WINDOW_DAYS == 90
    assert RECENT_WINDOW_DAYS_ALT == 180
    by_name = {spec.name: spec for spec in POOL_WINDOW_SPECS}
    assert by_name["last_5_at_role"].appearance_window == 5
    assert by_name["recent_90d"].window_days == 90
    assert by_name["recent_180d"].window_days == 180
    assert by_name["current_version"].version_mode == "current"
    assert HIERARCHICAL_K_GRID[0] == 0.0
    assert SCORING_MIXTURE_EPSILON == 1e-3
    assert SLICE25_DIAGNOSTIC_ONLY is True


def test_strict_prior_history() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=2,
                position=1,
                start_time=T1,
            ),
            _row(
                match_id=3,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T2,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    first = state.iloc[0]
    assert float(first["pool_n_role"]) == 0.0
    assert pd.isna(first["pool_expanding_realized_share"])
    second = state.iloc[1]
    assert float(second["pool_n_role"]) == 1.0
    assert float(second["pool_expanding_realized_share"]) == pytest.approx(0.0)
    assert float(second["pool_expanding_realized_n"]) == 0.0
    third = state.iloc[2]
    assert float(third["pool_n_role"]) == 2.0
    assert float(third["pool_expanding_realized_share"]) == pytest.approx(0.5)
    extra = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    _row(
                        match_id=4,
                        player_id=11,
                        hero_id=9,
                        position=1,
                        start_time=T3,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    later = attach_player_hero_pool_state(extra)
    np.testing.assert_allclose(
        later.iloc[:3]["pool_n_role"].to_numpy(dtype=float),
        state["pool_n_role"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        later.iloc[:3]["pool_expanding_realized_share"].to_numpy(dtype=float),
        state["pool_expanding_realized_share"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_same_timestamp_rows_are_mutually_blind() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=2,
                position=1,
                start_time=T1,
            ),
            _row(
                match_id=3,
                player_id=11,
                hero_id=3,
                position=1,
                start_time=T1,
            ),
            _row(
                match_id=4,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T2,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    at_t1 = state.loc[state["start_time"] == T1]
    assert at_t1["pool_n_role"].tolist() == [1.0, 1.0]
    assert at_t1["pool_expanding_realized_share"].tolist() == [0.0, 0.0]
    t2 = state.loc[state["start_time"] == T2].iloc[0]
    assert float(t2["pool_n_role"]) == 3.0
    assert float(t2["pool_expanding_realized_share"]) == pytest.approx(1.0 / 3.0)


def test_current_hero_cannot_affect_its_own_distribution() -> None:
    base = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=2,
                position=1,
                start_time=T1,
            ),
        ]
    )
    swapped = base.copy()
    swapped.loc[1, "hero_id"] = 99
    left = attach_player_hero_pool_state(base)
    right = attach_player_hero_pool_state(swapped)
    assert left.iloc[1]["_pool_mass_expanding"] == {1: 1}
    assert right.iloc[1]["_pool_mass_expanding"] == {1: 1}
    assert float(left.iloc[1]["pool_n_role"]) == float(right.iloc[1]["pool_n_role"])
    assert float(left.iloc[1]["pool_expanding_realized_share"]) == pytest.approx(0.0)
    assert float(right.iloc[1]["pool_expanding_realized_share"]) == pytest.approx(0.0)


def test_current_observed_position_does_not_enter_construction() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=10,
                position=3,
                start_time=T0,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=20,
                position=1,
                start_time=T1,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    alt = frame.copy()
    alt.loc[1, "position_number"] = 4.0
    alt.loc[1, "position"] = "POSITION_4"
    other = attach_player_hero_pool_state(alt)
    for role in (1, 2, 3, 4, 5):
        column = f"pool_n_at_position_{role}"
        np.testing.assert_array_equal(
            state[column].to_numpy(dtype=float),
            other[column].to_numpy(dtype=float),
        )
    assert state.iloc[1]["_pool_mass_by_role"] == other.iloc[1]["_pool_mass_by_role"]
    assert float(state.iloc[1]["pool_n_role"]) == 0.0
    assert float(other.iloc[1]["pool_n_role"]) == 0.0
    assert float(state.iloc[1]["pool_n_at_position_3"]) == 1.0
    assert float(state.iloc[1]["pool_n_at_position_1"]) == 0.0


def test_historical_non_explicit_positions_do_not_enter_role_buckets() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=None,
                start_time=T0,
                position_label="UNKNOWN",
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=1,
                position=None,
                start_time=T1,
                position_label="FILTERED",
            ),
            _row(
                match_id=3,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T2,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    third = state.iloc[2]
    assert float(third["pool_n_player_explicit"]) == 0.0
    assert float(third["pool_n_role"]) == 0.0
    assert float(third["pool_n_at_position_1"]) == 0.0
    assert third["_pool_mass_expanding"] == {}


def test_hero_at_pos3_has_zero_share_at_pos1_while_unconditioned_positive() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=10,
                position=3,
                start_time=T0,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=20,
                position=1,
                start_time=T1,
            ),
            _row(
                match_id=3,
                player_id=11,
                hero_id=10,
                position=1,
                start_time=T2,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    current = state.iloc[2]
    assert float(current["pool_n_player_hero_explicit"]) == 1.0
    assert float(current["pool_uncond_realized_share"]) == pytest.approx(0.5)
    assert float(current["pool_n_role"]) == 1.0
    assert float(current["pool_expanding_realized_n"]) == 0.0
    assert float(current["pool_expanding_realized_share"]) == pytest.approx(0.0)
    assert float(current["pool_role_gap"]) == 1.0


def test_n_role_zero_produces_null_role_distribution() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=10,
                position=3,
                start_time=T0,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=10,
                position=1,
                start_time=T1,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    current = state.iloc[1]
    assert float(current["pool_n_role"]) == 0.0
    assert pd.isna(current["pool_expanding_realized_share"])
    assert pd.isna(current["pool_expanding_breadth"])
    assert pd.isna(current["pool_expanding_entropy"])
    assert pd.isna(current["pool_expanding_top1_share"])
    assert float(current["pool_n_player_explicit"]) == 1.0
    assert float(current["pool_uncond_realized_share"]) == pytest.approx(1.0)


def test_unseen_hero_at_established_role_has_empirical_share_zero() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T1,
            ),
            _row(
                match_id=3,
                player_id=11,
                hero_id=2,
                position=1,
                start_time=T2,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    current = state.iloc[2]
    assert float(current["pool_n_role"]) == 2.0
    assert float(current["pool_expanding_realized_n"]) == 0.0
    assert float(current["pool_expanding_realized_share"]) == pytest.approx(0.0)
    assert float(current["pool_expanding_top1_share"]) == pytest.approx(1.0)


def test_last_n_at_role_counts_appearances_at_r_not_global() -> None:
    rows = []
    for i in range(8):
        rows.append(
            _row(
                match_id=i + 1,
                player_id=11,
                hero_id=100 + i,
                position=2,
                start_time=T0 + timedelta(days=i),
            )
        )
    for i, hero in enumerate((1, 2, 3)):
        rows.append(
            _row(
                match_id=20 + i,
                player_id=11,
                hero_id=hero,
                position=1,
                start_time=T0 + timedelta(days=10 + i),
            )
        )
    rows.append(
        _row(
            match_id=30,
            player_id=11,
            hero_id=9,
            position=1,
            start_time=T0 + timedelta(days=20),
        )
    )
    state = attach_player_hero_pool_state(pd.DataFrame(rows))
    current = state.iloc[-1]
    assert float(current["pool_n_player_explicit"]) == 11.0
    assert float(current["pool_n_role"]) == 3.0
    assert float(current["pool_last_5_at_role_n_role"]) == 3.0
    assert float(current["pool_last_5_at_role_breadth"]) == 3.0
    mass = current["_pool_mass_last_5_at_role"]
    assert set(mass) == {1, 2, 3}
    assert 100 not in mass


def test_version_windows_do_not_use_future_versions() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
                game_version_id=176,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=2,
                position=1,
                start_time=T1,
                game_version_id=177,
            ),
            _row(
                match_id=3,
                player_id=11,
                hero_id=3,
                position=1,
                start_time=T2,
                game_version_id=177,
            ),
            _row(
                match_id=4,
                player_id=11,
                hero_id=4,
                position=1,
                start_time=T3,
                game_version_id=178,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    at_t2 = state.iloc[2]
    assert float(at_t2["pool_n_role"]) == 2.0
    assert float(at_t2["pool_current_version_n_role"]) == 1.0
    assert at_t2["_pool_mass_current_version"] == {2: 1}
    assert set(at_t2["_pool_mass_current_plus_previous"]) == {1, 2}
    at_t1 = state.iloc[1]
    assert float(at_t1["pool_current_version_n_role"]) == 0.0
    assert pd.isna(at_t1["pool_current_version_realized_share"])
    later = frame.copy()
    later.loc[3, "game_version_id"] = 179
    swapped = attach_player_hero_pool_state(later)
    assert swapped.iloc[2]["_pool_mass_current_version"] == {2: 1}
    assert swapped.iloc[2]["_pool_mass_current_plus_previous"] == {1: 1, 2: 1}


def test_recent_window_excludes_older_than_cutoff() -> None:
    old = T2 - timedelta(days=91)
    inside = T2 - timedelta(days=30)
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=old,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=2,
                position=1,
                start_time=inside,
            ),
            _row(
                match_id=3,
                player_id=11,
                hero_id=3,
                position=1,
                start_time=T2,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    current = state.iloc[2]
    assert float(current["pool_n_role"]) == 2.0
    assert float(current["pool_recent_90d_n_role"]) == 1.0
    assert current["_pool_mass_recent_90d"] == {2: 1}
    assert float(current["pool_recent_180d_n_role"]) == 2.0


def test_expected_position_and_box_score_do_not_enter_counts() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=1,
                position=1,
                start_time=T0,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=2,
                position=1,
                start_time=T1,
            ),
        ]
    )
    left = attach_player_hero_pool_state(frame)
    right_src = frame.copy()
    right_src["expected_position"] = 2
    right_src["num_last_hits"] = 999
    right_src["hero_damage"] = 1
    right_src["team_won"] = 0
    right_src["duration_seconds"] = 1
    right = attach_player_hero_pool_state(right_src)
    np.testing.assert_allclose(
        left["pool_n_role"].to_numpy(dtype=float),
        right["pool_n_role"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert left.iloc[1]["_pool_mass_expanding"] == right.iloc[1]["_pool_mass_expanding"]


def test_common_candidate_universe_scoring() -> None:
    """Every estimator is scored on the same C_T, not estimator support."""
    realized = 99
    candidates = frozenset({1, 2, 3, 4, realized})
    expanding = {1: 2, 2: 1}
    last_hero = {1: 1.0}
    uncond = {1: 2, 3: 1}
    population = {1: 10, 2: 5, 4: 1}
    uniform = {1: 1.0, 2: 1.0}
    scored = {
        name: score_distribution(
            mass, realized_hero=realized, candidates=candidates
        )
        for name, mass in {
            "expanding": expanding,
            "last_hero": last_hero,
            "uncond": uncond,
            "population": population,
            "uniform": uniform,
        }.items()
    }
    for name, metrics in scored.items():
        assert metrics["n_candidates"] == float(len(candidates)), name
        assert metrics["p_realized"] > 0.0, name
        assert np.isfinite(metrics["log_loss"]), name
    # Unseen realized hero: raw p=0, mixture gives eps/|C|.
    eps = SCORING_MIXTURE_EPSILON
    expected_q = eps / len(candidates)
    assert scored["expanding"]["p_realized"] == pytest.approx(expected_q)
    assert scored["last_hero"]["p_realized"] == pytest.approx(expected_q)
    assert scored["last_hero"]["log_loss"] == pytest.approx(
        scored["expanding"]["log_loss"]
    )
    # Raw Brier for a zero-probability realized class over |C|.
    assert scored["expanding"]["brier"] == pytest.approx(1.0 + (2 / 3) ** 2 + (1 / 3) ** 2)
    in_pool = score_distribution(
        {1: 3}, realized_hero=1, candidates=frozenset({1, 2, 3})
    )
    # q(1) = (1-eps)*1 + eps/3
    q1 = (1.0 - eps) * 1.0 + eps / 3.0
    assert in_pool["log_loss"] == pytest.approx(-np.log(q1))
    assert in_pool["rank"] == 1.0
    assert in_pool["brier"] == pytest.approx(0.0)
    assert scoring_candidates(frozenset({1, 2}), realized_hero=99) == frozenset(
        {1, 2, 99}
    )


def test_hierarchical_backs_off_when_role_history_empty() -> None:
    frame = pd.DataFrame(
        [
            _row(
                match_id=1,
                player_id=11,
                hero_id=10,
                position=3,
                start_time=T0,
            ),
            _row(
                match_id=2,
                player_id=11,
                hero_id=10,
                position=1,
                start_time=T1,
            ),
        ]
    )
    state = attach_player_hero_pool_state(frame)
    current = state.iloc[1]
    mixed = score_distribution(
        {10: 1.0},
        realized_hero=10,
        candidates=frozenset({10}),
    )
    assert float(current["pool_n_role"]) == 0.0
    assert float(current["pool_n_player_explicit"]) == 1.0
    eps = SCORING_MIXTURE_EPSILON
    assert mixed["p_realized"] == pytest.approx((1.0 - eps) * 1.0 + eps)
    assert current["_pool_candidates_prior"] == frozenset({10})


def test_vectorized_scorer_matches_reference_scorer() -> None:
    """Vectorized and Series-based scorers must agree bit-for-bit on metrics."""
    from dota_predictor.training.player_hero_pool_state import (
        _score_frame,
        _score_frame_reference,
    )

    rows: list[dict[str, object]] = []
    for i, (hero, pos, version) in enumerate(
        (
            (1, 1, 176),
            (2, 1, 176),
            (1, 1, 177),
            (3, 2, 177),
            (2, 1, 177),
            (9, 1, 178),
            (1, 2, 178),
            (4, 1, 178),
        )
    ):
        rows.append(
            _row(
                match_id=i + 1,
                player_id=11,
                hero_id=hero,
                position=pos,
                start_time=T0 + timedelta(days=i),
                game_version_id=version,
            )
        )
    # Second player with sparse / cold-start cases.
    rows.extend(
        [
            _row(
                match_id=100,
                player_id=12,
                hero_id=5,
                position=1,
                start_time=T0 + timedelta(days=1),
            ),
            _row(
                match_id=101,
                player_id=12,
                hero_id=5,
                position=3,
                start_time=T0 + timedelta(days=5),
            ),
        ]
    )
    state = attach_player_hero_pool_state(pd.DataFrame(rows))
    estimators = (
        "expanding",
        "unconditioned",
        "last_hero_at_role",
        "population",
        "uniform_at_role",
        "last_5_at_role",
        "recent_90d",
        "current_version",
        "current_plus_previous",
        "hierarchical",
    )
    for row_set in ("native", "role_history"):
        for estimator in estimators:
            vectorized = _score_frame(
                state,
                estimator=estimator,
                split="fixture",
                hierarchical_k=5.0,
                row_set=row_set,
            ).iloc[0]
            reference = _score_frame_reference(
                state,
                estimator=estimator,
                split="fixture",
                hierarchical_k=5.0,
                row_set=row_set,
            ).iloc[0]
            assert int(vectorized["n_scored"]) == int(reference["n_scored"]), estimator
            assert int(vectorized["n_undefined"]) == int(
                reference["n_undefined"]
            ), estimator
            for metric in (
                "log_loss",
                "brier",
                "mean_rank",
                "hit_1",
                "hit_3",
                "hit_5",
                "mean_p_realized",
            ):
                left = float(vectorized[metric])
                right = float(reference[metric])
                if np.isnan(left) and np.isnan(right):
                    continue
                assert left == pytest.approx(right, rel=0, abs=1e-12), (
                    f"{estimator}/{row_set}/{metric}: {left} vs {right}"
                )


def test_classify_a_b_c() -> None:
    a = classify_slice25(
        role_conditioning_confirmed=True,
        role_conditioning_partial=False,
        window_confirmed=False,
        window_name=None,
        hierarchical_confirmed=False,
    )
    assert a.iloc[0]["classification"] == "A"
    assert a.iloc[0]["frozen_components"] == ("expanding_pxrxh",)
    b = classify_slice25(
        role_conditioning_confirmed=False,
        role_conditioning_partial=True,
        window_confirmed=False,
        window_name=None,
        hierarchical_confirmed=False,
    )
    assert b.iloc[0]["classification"] == "B"
    assert b.iloc[0]["frozen_components"] == ("expanding_pxrxh",)
    c = classify_slice25(
        role_conditioning_confirmed=False,
        role_conditioning_partial=False,
        window_confirmed=False,
        window_name=None,
        hierarchical_confirmed=False,
    )
    assert c.iloc[0]["classification"] == "C"
    assert c.iloc[0]["frozen_components"] == ()
    with_window = classify_slice25(
        role_conditioning_confirmed=True,
        role_conditioning_partial=False,
        window_confirmed=True,
        window_name="last_10_at_role",
        hierarchical_confirmed=False,
    )
    assert with_window.iloc[0]["frozen_components"] == (
        "expanding_pxrxh",
        "last_10_at_role",
    )


def test_feature_columns_remain_thirty_three() -> None:
    assert len(FEATURE_COLUMNS) == 33
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)
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
    for name in (
        *SLICE25_STATE_COLUMNS,
        "pool_expanding_realized_share",
        "player_hero_pool",
        "flex_score",
        "assignment_entropy",
    ):
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
        assert name not in SNAPSHOT_COLUMNS
        assert name not in PRE_DRAFT_SNAPSHOT_SQL
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in FEATURE_COLUMNS
        pre_draft = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
        assert column not in pre_draft
    assert SLICE25_FROZEN_COMPONENTS == () or isinstance(
        SLICE25_FROZEN_COMPONENTS, tuple
    )


def test_development_cutoff_and_holdout_exclusion(tmp_path: Path) -> None:
    later = FROZEN_DEVELOPMENT_END + timedelta(days=1)
    matches = [
        match_row(
            1,
            start_time=T0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
            game_version_id=176,
        ),
        match_row(
            2,
            start_time=T1,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=False,
            game_version_id=177,
        ),
        match_row(
            3,
            start_time=later,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
            game_version_id=177,
        ),
    ]
    players = _annotate_players(
        player_rows(1, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(2, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        + player_rows(3, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
    )
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
        for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
            assert column not in view_columns
        report = run_player_hero_pool_diagnostics(store)
    assert report.n_development_matches == 2
    assert report.n_development_player_rows == 20
    assert report.n_holdout_excluded == 10
    assert report.integrity["holdout_used_for_window_selection"] is False
    assert report.integrity["holdout_used_for_validation"] is False
    assert report.integrity["holdout_used_for_k_selection"] is False
    assert report.integrity["holdout_used_for_smoothing_selection"] is False
    assert report.integrity["smoothing_tuned"] is False
    assert report.integrity["win_model_run"] is False
    assert report.integrity["slice26_assignment_built"] is False
    assert report.integrity["expected_position_used_to_build_counts"] is False
    assert report.development_end == FROZEN_DEVELOPMENT_END
    assert report.scoring_mixture_epsilon == SCORING_MIXTURE_EPSILON
    payload = slice25_report_to_jsonable(report)
    assert payload["n_holdout_excluded"] == 10
    assert payload["scoring_mixture_epsilon"] == SCORING_MIXTURE_EPSILON
    assert not bool(report.classification.iloc[0]["slice26_built"])
    holdout = pd.DataFrame({"start_time": [later], "hero_id": [1], "player_id": [11]})
    restricted = restrict_development(holdout)
    assert restricted.empty


def test_integrity_flags_on_full_run(tmp_path: Path) -> None:
    matches = [
        match_row(
            i,
            start_time=T0 + timedelta(days=i),
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=bool(i % 2),
            game_version_id=176 + (i // 2),
        )
        for i in range(1, 8)
    ]
    players = _annotate_players(
        [
            row
            for i in range(1, 8)
            for row in player_rows(i, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        ]
    )
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        report = run_player_hero_pool_diagnostics(store)
    assert report.integrity["stratz_called"] is False
    assert report.integrity["win_model_run"] is False
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["slice23_fit_used"] is False
    assert report.integrity["slice24_outcome_used"] is False
    assert report.integrity["synergy_or_counter_created"] is False
    assert report.semantics["scoring_wrapper_in_frozen_state"] is False
    assert "n_player_hero_pairs_at_2plus_positions" in report.cross_position.columns
    assert "n_heroes_represented_at_2plus_positions" not in report.cross_position.columns


def test_cross_position_metric_counts_player_hero_pairs() -> None:
    """798-style totals are (player, hero) pairs, not distinct Dota heroes."""
    pools: dict[int, _PlayerRolePool] = {}
    for player_id, hero, roles in (
        (11, 1, (1, 2)),
        (11, 2, (1,)),
        (12, 1, (3, 4)),
        (12, 3, (5,)),
    ):
        pool = pools.setdefault(player_id, _PlayerRolePool())
        for role in roles:
            pool.add(
                role=role,
                hero=hero,
                start_time=T0,
                version_id=176,
            )
    summary = _cross_position_from_pools(pools).iloc[0]
    # Hero 1 for player 11 and hero 1 for player 12 → two pairs, one distinct hero.
    assert int(summary["n_player_hero_pairs_at_2plus_positions"]) == 2
    assert int(summary["n_distinct_heroes_at_2plus_positions_any_player"]) == 1
