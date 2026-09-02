"""Slice 21 hero resource/combat profiles: leakage, freeze boundaries, no fit."""

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
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.training.combat_performance_target import (
    COMBAT_C,
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
)
from dota_predictor.training.hero_performance_profile import (
    COMBAT_C1,
    COMBAT_C2,
    FARMING_F1,
    FARMING_F2,
    HERO_COMBAT_PROFILE_KEY,
    HERO_COMBAT_PROFILE_TARGET,
    HERO_FARMING_PROFILE_KEY,
    HERO_FARMING_PROFILE_TARGET,
    PLAYER_X_HERO_FIT_NAMES,
    PROFILE_SPECS,
    all_observation_and_leave_player_out,
    assign_chronological_blocks,
    attach_causal_group_mean,
    attach_hero_profile_observations,
    group_split_half,
    run_hero_performance_profile_diagnostics,
)
from dota_predictor.training.player_combat_state import (
    CAUSAL_C_COLUMN,
    FROZEN_COMBAT_SHRINKAGE_K,
)
from dota_predictor.training.player_farming_state import (
    CAUSAL_B_COLUMN,
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
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


def _appearance(
    *,
    match_id: int,
    player_id: int,
    start_time: datetime,
    position: int,
    hero_id: int,
    last_hits: float,
    hero_damage: float,
    side: str = "RADIANT",
    team_id: int = 100,
    team_won: int = 1,
    game_version_id: int = 176,
    duration_seconds: float = 1800.0,
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
        "game_version_id": game_version_id,
        "duration_seconds": duration_seconds,
        "num_last_hits": float(last_hits),
        "team_won": team_won,
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


def _two_sided_match(
    match_id: int,
    start_time: datetime,
    *,
    radiant_heroes: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5),
    dire_heroes: tuple[int, int, int, int, int] = (6, 7, 8, 9, 10),
    radiant_lh: tuple[float, float, float, float, float] = (300, 220, 160, 40, 20),
    dire_lh: tuple[float, float, float, float, float] = (280, 200, 150, 35, 18),
    radiant_damage: tuple[float, float, float, float, float] = (
        10000,
        8000,
        6000,
        4000,
        2000,
    ),
    dire_damage: tuple[float, float, float, float, float] = (
        9000,
        7000,
        5000,
        3000,
        1000,
    ),
    player_base_radiant: int = 10,
    player_base_dire: int = 20,
    game_version_id: int = 176,
    radiant_won: bool = True,
    duration_seconds: float | None = None,
) -> list[dict[str, object]]:
    duration = (
        float(duration_seconds)
        if duration_seconds is not None
        else 1500.0 + 120.0 * match_id
    )
    rows: list[dict[str, object]] = []
    for position in range(1, 6):
        rows.append(
            _appearance(
                match_id=match_id,
                player_id=player_base_radiant + position,
                start_time=start_time,
                position=position,
                hero_id=radiant_heroes[position - 1],
                last_hits=radiant_lh[position - 1],
                hero_damage=radiant_damage[position - 1],
                side="RADIANT",
                team_id=100,
                team_won=1 if radiant_won else 0,
                game_version_id=game_version_id,
                duration_seconds=duration,
            )
        )
        rows.append(
            _appearance(
                match_id=match_id,
                player_id=player_base_dire + position,
                start_time=start_time,
                position=position,
                hero_id=dire_heroes[position - 1],
                last_hits=dire_lh[position - 1],
                hero_damage=dire_damage[position - 1],
                side="DIRE",
                team_id=200,
                team_won=0 if radiant_won else 1,
                game_version_id=game_version_id,
                duration_seconds=duration,
            )
        )
    return rows


def _warmup_frame(extra: list[dict[str, object]] | None = None) -> pd.DataFrame:
    rows = (
        _two_sided_match(1, T0)
        + _two_sided_match(
            10,
            T0,
            duration_seconds=2100.0,
            player_base_radiant=110,
            player_base_dire=120,
        )
        + _two_sided_match(2, T1)
        + _two_sided_match(3, T2)
    )
    if extra:
        rows.extend(extra)
    return pd.DataFrame(rows)


def _profiles(frame: pd.DataFrame) -> pd.DataFrame:
    observed = attach_hero_profile_observations(frame)
    observed = attach_causal_group_mean(
        observed,
        value_column=CAUSAL_B_COLUMN,
        group_columns=("hero_id",),
        out_column="f1_mean",
        n_column="f1_n",
    )
    observed = attach_causal_group_mean(
        observed,
        value_column=CAUSAL_B_COLUMN,
        group_columns=("hero_id", "position_number"),
        out_column="f2_mean",
        n_column="f2_n",
    )
    observed = attach_causal_group_mean(
        observed,
        value_column=CAUSAL_C_COLUMN,
        group_columns=("hero_id",),
        out_column="c1_mean",
        n_column="c1_n",
    )
    return attach_causal_group_mean(
        observed,
        value_column=CAUSAL_C_COLUMN,
        group_columns=("hero_id", "position_number"),
        out_column="c2_mean",
        n_column="c2_n",
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


def _annotate_players(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        slot = int(item["slot_in_side"])
        item["position"] = POSITIONS[slot]
        item.update(_box_for_slot(slot))
        annotated.append(item)
    return annotated


def test_future_hero_observations_do_not_change_earlier_profile() -> None:
    base = _profiles(_warmup_frame())
    t1 = base.loc[base["start_time"] == T1].sort_values(
        ["match_id", "player_id"], kind="mergesort"
    )
    extra = _two_sided_match(
        4,
        T3,
        radiant_lh=(900, 800, 700, 50, 10),
        radiant_damage=(40000, 30000, 20000, 1000, 500),
    )
    later = _profiles(_warmup_frame(extra))
    t1_later = later.loc[later["start_time"] == T1].sort_values(
        ["match_id", "player_id"], kind="mergesort"
    )
    for column in ("f1_mean", "f2_mean", "c1_mean", "c2_mean", "f1_n", "c2_n"):
        np.testing.assert_allclose(
            t1[column].to_numpy(dtype=float),
            t1_later[column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_same_timestamp_hero_observations_are_mutually_blind() -> None:
    extra = _two_sided_match(
        30,
        T2,
        radiant_heroes=(1, 2, 3, 4, 5),
        dire_heroes=(1, 7, 8, 9, 10),
        player_base_radiant=50,
        player_base_dire=60,
        radiant_lh=(10, 10, 10, 10, 10),
        dire_lh=(10, 10, 10, 10, 10),
    )
    attached = _profiles(_warmup_frame(extra))
    t2 = attached.loc[attached["start_time"] == T2]
    hero1 = t2.loc[t2["hero_id"] == 1]
    assert hero1["f1_n"].nunique() == 1
    assert float(hero1["f1_n"].iloc[0]) >= 1
    means = hero1["f1_mean"].to_numpy(dtype=float)
    np.testing.assert_allclose(means, means[0], equal_nan=True)


def test_hero_a_observations_do_not_alter_hero_b_profile() -> None:
    base = _profiles(_warmup_frame())
    extra = _two_sided_match(
        4,
        T1 + timedelta(days=1),
        radiant_heroes=(99, 98, 97, 96, 95),
        dire_heroes=(94, 93, 92, 91, 90),
        radiant_lh=(1, 1, 1, 1, 1),
        dire_lh=(1, 1, 1, 1, 1),
    )
    later = _profiles(_warmup_frame(extra))
    t2_base = base.loc[base["start_time"] == T2].sort_values("player_id")
    t2_later = later.loc[later["start_time"] == T2].sort_values("player_id")
    np.testing.assert_allclose(
        t2_base["f1_mean"].to_numpy(dtype=float),
        t2_later["f1_mean"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_other_position_does_not_enter_hero_position_profile() -> None:
    extra = [
        _appearance(
            match_id=40,
            player_id=400 + position,
            start_time=T1 + timedelta(hours=1),
            position=position,
            hero_id=1 if position == 5 else 50 + position,
            last_hits=5.0 if position == 5 else 20.0,
            hero_damage=100.0 if position == 5 else 1000.0,
            side="RADIANT",
        )
        for position in range(1, 6)
    ]
    extra.extend(
        _appearance(
            match_id=40,
            player_id=410 + position,
            start_time=T1 + timedelta(hours=1),
            position=position,
            hero_id=60 + position,
            last_hits=20.0,
            hero_damage=1000.0,
            side="DIRE",
            team_id=200,
            team_won=0,
        )
        for position in range(1, 6)
    )
    attached = _profiles(_warmup_frame(extra))
    t2_pos1 = attached.loc[
        (attached["start_time"] == T2)
        & (attached["hero_id"] == 1)
        & (attached["position_number"] == 1.0)
    ]
    base = _profiles(_warmup_frame())
    base_pos1 = base.loc[
        (base["start_time"] == T2)
        & (base["hero_id"] == 1)
        & (base["position_number"] == 1.0)
    ]
    np.testing.assert_allclose(
        t2_pos1["f2_mean"].to_numpy(dtype=float),
        base_pos1["f2_mean"].to_numpy(dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        t2_pos1["c2_mean"].to_numpy(dtype=float),
        base_pos1["c2_mean"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_position_conditioned_profiles_use_explicit_positions_only() -> None:
    extra = _two_sided_match(4, T1 + timedelta(days=2), radiant_heroes=(1, 2, 3, 4, 5))
    frame = pd.DataFrame(_warmup_frame(extra).to_dict(orient="records"))
    unknown = frame["match_id"] == 4
    frame.loc[unknown, "position"] = "UNKNOWN"
    frame.loc[unknown, "position_number"] = np.nan
    attached = _profiles(frame)
    unknown_rows = attached.loc[unknown]
    assert unknown_rows["f2_mean"].isna().all()
    assert (unknown_rows["f2_n"] == 0).all()
    assert unknown_rows["c2_mean"].isna().all()
    t2 = attached.loc[attached["start_time"] == T2]
    base = _profiles(_warmup_frame()).loc[lambda d: d["start_time"] == T2]
    np.testing.assert_allclose(
        t2.sort_values("player_id")["f2_n"].to_numpy(dtype=float),
        base.sort_values("player_id")["f2_n"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_current_result_does_not_enter_profile_construction() -> None:
    base = _profiles(_warmup_frame())
    flipped = _warmup_frame()
    flipped["team_won"] = 1 - flipped["team_won"]
    mutated = _profiles(flipped)
    for column in (
        "f1_mean",
        "f2_mean",
        "c1_mean",
        "c2_mean",
        CAUSAL_B_COLUMN,
        CAUSAL_C_COLUMN,
    ):
        np.testing.assert_allclose(
            base[column].to_numpy(dtype=float),
            mutated[column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_current_match_box_scores_are_not_pre_draft_state() -> None:
    base = _profiles(_warmup_frame())
    mutated = _warmup_frame()
    current = mutated["start_time"] == T2
    mutated.loc[current, "num_last_hits"] = 0.0
    mutated.loc[current, "hero_damage"] = 1.0
    attached = _profiles(mutated)
    t2_base = base.loc[base["start_time"] == T2].sort_values("player_id")
    t2_mut = attached.loc[attached["start_time"] == T2].sort_values("player_id")
    for column in ("f1_mean", "f2_mean", "c1_mean", "c2_mean", "f1_n", "c2_n"):
        np.testing.assert_allclose(
            t2_base[column].to_numpy(dtype=float),
            t2_mut[column].to_numpy(dtype=float),
            equal_nan=True,
        )
    assert not np.allclose(
        t2_base[CAUSAL_B_COLUMN].to_numpy(dtype=float),
        t2_mut[CAUSAL_B_COLUMN].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_chronological_split_half_is_deterministic() -> None:
    frame = attach_hero_profile_observations(_warmup_frame())
    first = group_split_half(
        frame,
        value_column=CAUSAL_B_COLUMN,
        group_columns=("hero_id",),
        min_each=1,
    )
    second = group_split_half(
        frame,
        value_column=CAUSAL_B_COLUMN,
        group_columns=("hero_id",),
        min_each=1,
    )
    assert first["n_profiles"] == second["n_profiles"]
    np.testing.assert_allclose(first["pearson"], second["pearson"], equal_nan=True)
    np.testing.assert_allclose(first["spearman"], second["spearman"], equal_nan=True)
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    third = group_split_half(
        shuffled,
        value_column=CAUSAL_B_COLUMN,
        group_columns=("hero_id",),
        min_each=1,
    )
    assert first["n_profiles"] == third["n_profiles"]
    if np.isfinite(first["pearson"]) and np.isfinite(third["pearson"]):
        assert first["pearson"] == pytest.approx(third["pearson"])


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
        report = run_hero_performance_profile_diagnostics(store)
    assert report.n_development_matches == 2
    assert report.n_development_player_rows == 20
    assert report.n_holdout_excluded == 10
    assert report.integrity["holdout_used_for_selection"] is False
    assert report.integrity["holdout_used_for_stability"] is False
    assert report.development_end == FROZEN_DEVELOPMENT_END
    holdout = pd.DataFrame(
        {
            "start_time": [later],
            "hero_id": [1],
            "player_id": [11],
        }
    )
    restricted = restrict_development(holdout)
    assert restricted.empty


def test_frozen_farming_and_combat_contracts_unchanged() -> None:
    assert FROZEN_CANDIDATE_B == CANDIDATE_B
    assert CANDIDATE_B == "last_hits_per_min_position_duration_residual_z"
    assert FROZEN_SHRINKAGE_K == 5.0
    assert FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION
    assert FROZEN_COMBAT_CANDIDATE == "hero_damage_share_position_adj"
    assert FROZEN_COMBAT_SHRINKAGE_K == 20.0
    assert HERO_FARMING_PROFILE_TARGET == CAUSAL_B_COLUMN
    assert HERO_COMBAT_PROFILE_TARGET == CAUSAL_C_COLUMN
    assert HERO_FARMING_PROFILE_KEY == "hero_id × position"
    assert HERO_COMBAT_PROFILE_KEY == "hero_id × position"
    names = {spec.name for spec in PROFILE_SPECS}
    assert names == {FARMING_F1, FARMING_F2, COMBAT_C1, COMBAT_C2}


def test_no_fit_feature_and_feature_columns_remain_thirty_three() -> None:
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
    for name in PLAYER_X_HERO_FIT_NAMES:
        assert name not in FEATURE_COLUMNS
        assert name not in ALL_FEATURE_COLUMNS
        assert name not in SNAPSHOT_COLUMNS
        assert name not in PRE_DRAFT_SNAPSHOT_SQL
    for name in (
        CAUSAL_B_COLUMN,
        CAUSAL_C_COLUMN,
        COMBAT_C,
        "last_hits_per_minute",
        "f1_mean",
        "hero_farming_profile",
        "hero_combat_profile",
    ):
        assert name not in FEATURE_COLUMNS
        assert name not in PRE_DRAFT_SNAPSHOT_SQL
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in FEATURE_COLUMNS
        pre_draft = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
        assert column not in pre_draft


def test_leave_player_out_excludes_the_current_player() -> None:
    frame = pd.DataFrame(
        {
            "hero_id": [1, 1, 1, 2],
            "player_id": [10, 10, 11, 12],
            "position_number": [1.0, 1.0, 1.0, 1.0],
            "match_id": [1, 2, 3, 4],
            "start_time": [T0, T1, T2, T3],
            "value": [1.0, 3.0, 5.0, 9.0],
        }
    )
    compared = all_observation_and_leave_player_out(
        frame, value_column="value", group_columns=("hero_id",)
    )
    hero1 = compared.loc[frame["hero_id"] == 1]
    # All-observation mean of hero 1 is (1+3+5)/3 = 3.
    np.testing.assert_allclose(hero1["all_mean"].to_numpy(), 3.0)
    # Player 10's LPO is 5 (only player 11). Player 11's LPO is 2.
    p10 = compared.loc[frame["player_id"] == 10]
    p11 = compared.loc[frame["player_id"] == 11]
    np.testing.assert_allclose(p10["lpo_mean"].to_numpy(), 5.0)
    np.testing.assert_allclose(p11["lpo_mean"].to_numpy(), 2.0)


def test_lpo_causal_history_excludes_same_player() -> None:
    rows = _two_sided_match(1, T0) + _two_sided_match(2, T1)
    # Same player 11 on hero 1 at T2 as well.
    extra = _two_sided_match(3, T2)
    frame = pd.DataFrame(rows + extra)
    observed = attach_hero_profile_observations(frame)
    all_hist = attach_causal_group_mean(
        observed,
        value_column=CAUSAL_B_COLUMN,
        group_columns=("hero_id",),
        out_column="all_mean",
        n_column="all_n",
    )
    lpo_hist = attach_causal_group_mean(
        observed,
        value_column=CAUSAL_B_COLUMN,
        group_columns=("hero_id",),
        out_column="lpo_mean",
        n_column="lpo_n",
        leave_player_out=True,
    )
    t2_p11 = all_hist.loc[
        (all_hist["start_time"] == T2) & (all_hist["player_id"] == 11)
    ]
    t2_lpo = lpo_hist.loc[
        (lpo_hist["start_time"] == T2) & (lpo_hist["player_id"] == 11)
    ]
    if t2_p11["all_n"].notna().any():
        assert int(t2_lpo["lpo_n"].iloc[0]) <= int(t2_p11["all_n"].iloc[0])


def test_chronological_blocks_are_match_count_contiguous() -> None:
    frame = pd.DataFrame(
        {
            "match_id": [1, 1, 2, 3, 4, 5],
            "start_time": [T0, T0, T1, T2, T3, T3 + timedelta(days=1)],
            "player_id": [1, 2, 3, 4, 5, 6],
        }
    )
    blocks = assign_chronological_blocks(frame, n_blocks=5)
    assert blocks.nunique() == 5
    assert blocks.iloc[0] == blocks.iloc[1]


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
        for i in range(1, 6)
    ]
    players = _annotate_players(
        [
            row
            for i in range(1, 6)
            for row in player_rows(i, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        ]
    )
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        report = run_hero_performance_profile_diagnostics(store)
    assert report.integrity["stratz_called"] is False
    assert report.integrity["ingestion_modified"] is False
    assert report.integrity["schema_modified"] is False
    assert report.integrity["farming_candidate_b_unchanged"] is True
    assert report.integrity["farming_k_is_5"] is True
    assert report.integrity["combat_candidate_c_unchanged"] is True
    assert report.integrity["combat_k_is_20"] is True
    assert report.integrity["player_hero_fit_created"] is False
    assert report.integrity["team_feature_created"] is False
    assert report.integrity["win_model_run"] is False
    assert report.integrity["feature_columns_unchanged_length"] is True
    assert report.integrity["current_result_used_for_profile"] is False
    assert report.integrity["current_position_treated_as_pre_draft"] is False
    assert not report.classification.empty
    assert set(report.farming_comparison["representation"]) == {FARMING_F1, FARMING_F2}
    assert set(report.combat_comparison["representation"]) == {COMBAT_C1, COMBAT_C2}
    assert report.integrity["hero_shrinkage_k_frozen"] is False
