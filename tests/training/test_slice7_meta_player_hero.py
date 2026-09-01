"""Tests for Slice 7 walk-forward plumbing.

Does not retune folds, Elo, or production FEATURE_COLUMNS. Career
Player × Hero remains the existing comparison block.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from training_helpers import build_snapshot_store, match_row, player_rows

from dota_predictor.features.draft_comparison import build_draft_comparison
from dota_predictor.features.player_hero_meta import build_player_hero_meta
from dota_predictor.features.player_hero_meta_comparison import (
    SLICE7_COMPARISON_COLUMNS,
    SLICE7_RECENT20_COUNT_DIFF_COLUMNS,
    SLICE7_RECENT20_RATE_DIFF_COLUMNS,
    SLICE7_ROLE_DIFF_COLUMNS,
    SLICE7_SAME_VERSION_COUNT_DIFF_COLUMNS,
    SLICE7_SAME_VERSION_RATE_DIFF_COLUMNS,
    player_hero_meta_comparison_from_players,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    PLAYER_HERO_COMPARISON_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE7_META_PLAYER_HERO_SPECS,
)
from dota_predictor.training.post_draft import build_post_draft_model_ready_dataset
from dota_predictor.training.slice7_meta_player_hero import (
    CONTEST_SHIFT_ABS_DELTA,
    MIN_EXPLICIT_FOR_ROLE_SHIFT,
    assign_career_sample_bucket,
    assign_compatibility_bucket,
    assign_patch_maturity_bin,
    build_slice7_model_ready_dataset,
    describe_hero_shift_groups,
)
from dota_predictor.training.walk_forward import (
    WalkForwardConfig,
    resolve_walk_forward_folds,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _sequential_store(tmp_path: Path, n: int = 24):
    matches = []
    players = []
    player_counter = 1
    for i in range(n):
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


def test_slice7_not_added_to_production_feature_matrix() -> None:
    for column in SLICE7_COMPARISON_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
        assert column not in PRE_DRAFT_SNAPSHOT_SQL
    for spec in POST_DRAFT_BLOCK_ABLATION_SPECS:
        assert set(SLICE7_COMPARISON_COLUMNS).isdisjoint(spec.feature_columns)


def test_existing_career_player_hero_spec_is_unchanged() -> None:
    by_name = {spec.name: spec for spec in POST_DRAFT_BLOCK_ABLATION_SPECS}
    assert (
        by_name["logistic_elo_plus_player_hero"].feature_columns
        == ELO_PLUS_PLAYER_HERO_COLUMNS
    )
    slice7_career = {
        spec.name: spec for spec in SLICE7_META_PLAYER_HERO_SPECS
    }["logistic_elo_plus_player_hero"]
    assert slice7_career.feature_columns == ELO_PLUS_PLAYER_HERO_COLUMNS
    assert slice7_career.feature_columns == (
        ELO_ONLY_FEATURE_COLUMNS + PLAYER_HERO_COMPARISON_COLUMNS
    )


def test_count_only_specs_contain_no_win_rate_columns() -> None:
    by_name = {spec.name: spec for spec in SLICE7_META_PLAYER_HERO_SPECS}
    for name in (
        "logistic_elo_plus_same_version_volume",
        "logistic_elo_plus_recent20_volume",
    ):
        columns = by_name[name].feature_columns
        assert not any("win_rate" in column for column in columns)
    assert by_name["logistic_elo_plus_same_version_volume"].feature_columns == (
        ELO_ONLY_FEATURE_COLUMNS + SLICE7_SAME_VERSION_COUNT_DIFF_COLUMNS
    )
    assert by_name["logistic_elo_plus_recent20_volume"].feature_columns == (
        ELO_ONLY_FEATURE_COLUMNS + SLICE7_RECENT20_COUNT_DIFF_COLUMNS
    )


def test_role_and_combined_specs_contain_exactly_intended_blocks() -> None:
    by_name = {spec.name: spec for spec in SLICE7_META_PLAYER_HERO_SPECS}
    role_extra = set(by_name["logistic_elo_plus_role_meta"].feature_columns) - set(
        ELO_ONLY_FEATURE_COLUMNS
    )
    assert role_extra == set(SLICE7_ROLE_DIFF_COLUMNS)
    same_plus = by_name["logistic_elo_plus_same_version_volume_performance"]
    assert set(SLICE7_SAME_VERSION_RATE_DIFF_COLUMNS).issubset(same_plus.feature_columns)
    combined = by_name["logistic_elo_plus_same_version_role"]
    assert combined.feature_columns == (
        same_plus.feature_columns + SLICE7_ROLE_DIFF_COLUMNS
    )
    recent_plus = by_name["logistic_elo_plus_recent20_volume_performance"]
    assert set(SLICE7_RECENT20_RATE_DIFF_COLUMNS).issubset(recent_plus.feature_columns)
    assert by_name["logistic_elo_plus_recent20_role"].feature_columns == (
        recent_plus.feature_columns + SLICE7_ROLE_DIFF_COLUMNS
    )
    assert by_name["logistic_elo_plus_career_role"].feature_columns == (
        ELO_PLUS_PLAYER_HERO_COLUMNS + SLICE7_ROLE_DIFF_COLUMNS
    )


def test_slice7_reuses_post_draft_oos_fold_boundaries(tmp_path: Path) -> None:
    with _sequential_store(tmp_path, n=24) as store:
        post = build_post_draft_model_ready_dataset(store)
        assembly = build_slice7_model_ready_dataset(store)
    config = WalkForwardConfig(n_blocks=3)
    post_folds = resolve_walk_forward_folds(post, config=config)
    slice7_folds = resolve_walk_forward_folds(assembly.dataset, config=config)
    assert len(post_folds) == len(slice7_folds)
    for left, right in zip(post_folds, slice7_folds, strict=True):
        assert left.train_end == right.train_end
        assert left.validation_end == right.validation_end
        assert left.test_end == right.test_end
        assert list(left.test.context["match_id"]) == list(
            right.test.context["match_id"]
        )
    assert list(post.context["match_id"]) == list(assembly.dataset.context["match_id"])
    assert assembly.n_missing_slice6_comparison == 0
    assert set(SLICE7_COMPARISON_COLUMNS).issubset(assembly.dataset.X.columns)
    assert list(assembly.dataset.X[list(ELO_ONLY_FEATURE_COLUMNS)].columns) == list(
        ELO_ONLY_FEATURE_COLUMNS
    )
    pd.testing.assert_frame_equal(
        post.X[list(ELO_ONLY_FEATURE_COLUMNS)].reset_index(drop=True),
        assembly.dataset.X[list(ELO_ONLY_FEATURE_COLUMNS)].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        post.X[list(PLAYER_HERO_COMPARISON_COLUMNS)].reset_index(drop=True),
        assembly.dataset.X[list(PLAYER_HERO_COMPARISON_COLUMNS)].reset_index(
            drop=True
        ),
    )


def test_career_count_aggregation_matches_existing_player_hero_block(
    tmp_path: Path,
) -> None:
    with _sequential_store(tmp_path, n=8) as store:
        career = build_draft_comparison(store).to_frame()
        meta = build_player_hero_meta(store).to_frame()
    aggregated = player_hero_meta_comparison_from_players(
        meta,
        count_columns=("prior_games_on_hero",),
        rate_columns=(),
    )
    merged = career.merge(aggregated, on="match_id", suffixes=("_career", "_slice7"))
    pd.testing.assert_series_equal(
        merged["mean_player_prior_games_on_hero_diff"],
        merged["mean_prior_games_on_hero_diff"],
        check_names=False,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        merged["min_player_prior_games_on_hero_diff"],
        merged["min_prior_games_on_hero_diff"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["players_with_zero_prior_games_on_hero_diff_career"],
        merged["players_with_zero_prior_games_on_hero_diff_slice7"],
        check_names=False,
    )
    first_meta = meta.loc[meta["match_id"] == 1000]
    assert (first_meta["prior_games_on_hero"] == 0).all()
    assert (first_meta["player_hero_same_version_matches"] == 0).all()
    later = meta.loc[meta["match_id"] == 1007]
    assert later["prior_games_on_hero"].max() > 0


def test_patch_maturity_bins_are_fixed_and_strictly_prior() -> None:
    assert assign_patch_maturity_bin(0) == "opening (0–49 prior matches)"
    assert assign_patch_maturity_bin(49) == "opening (0–49 prior matches)"
    assert assign_patch_maturity_bin(50) == "early (50–199)"
    assert assign_patch_maturity_bin(199) == "early (50–199)"
    assert assign_patch_maturity_bin(200) == "mature (200+)"
    assert assign_patch_maturity_bin(500) == "mature (200+)"
    with pytest.raises(ValueError):
        assign_patch_maturity_bin(-1)


def test_compatibility_buckets_do_not_use_outcomes() -> None:
    values = pd.Series([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    outcomes = pd.Series([1, 0, 1, 0, 1, 0])
    buckets = [
        assign_compatibility_bucket(float(value), q25=q25, q75=q75)
        for value in values
    ]
    shuffled = [
        assign_compatibility_bucket(float(value), q25=q25, q75=q75)
        for value in values
    ]
    assert buckets == shuffled
    assert assign_compatibility_bucket(None, q25=q25, q75=q75) == "NULL"
    assert "radiant_win" not in assign_compatibility_bucket.__code__.co_varnames
    del outcomes


def test_career_sample_buckets_are_fixed() -> None:
    assert assign_career_sample_bucket(0.0) == "0"
    assert assign_career_sample_bucket(3.2) == "1–4"
    assert assign_career_sample_bucket(7.0) == "5–9"
    assert assign_career_sample_bucket(10.0) == "10–19"
    assert assign_career_sample_bucket(25.0) == "20+"


def test_role_shift_labels_do_not_use_outcomes() -> None:
    t0 = T0
    t1 = T0 + timedelta(days=10)
    rows = []
    for match_id, start_time, version, hero_id, modal, contest, explicit in (
        (1, t0, 10, 1, 1, 0.40, 20),
        (2, t1, 11, 1, 3, 0.42, 20),
        (3, t0, 10, 2, 2, 0.10, 20),
        (4, t1, 11, 2, 2, 0.50, 20),
        (5, t0, 10, 3, 1, None, 3),
        (6, t1, 11, 3, 5, None, 3),
    ):
        row = {
            "match_id": match_id,
            "hero_id": hero_id,
            "start_time": start_time,
            "game_version_id": version,
            "hero_same_version_position_explicit_count": explicit,
            "hero_same_version_contest_rate": contest,
            "radiant_win": True,
        }
        for position in (1, 2, 3, 4, 5):
            row[f"hero_same_version_position_{position}_share"] = (
                1.0 if position == modal else 0.0
            )
        rows.append(row)
    groups = describe_hero_shift_groups(pd.DataFrame(rows)).set_index("hero_id")
    assert MIN_EXPLICIT_FOR_ROLE_SHIFT == 8
    assert CONTEST_SHIFT_ABS_DELTA == pytest.approx(0.20)
    assert groups.loc[1, "shift_group"] == "role_shifted"
    assert groups.loc[2, "shift_group"] == "contest_shifted"
    assert groups.loc[3, "shift_group"] == "unclassified"
    without_outcome = pd.DataFrame(rows).drop(columns=["radiant_win"])
    again = describe_hero_shift_groups(without_outcome).set_index("hero_id")
    pd.testing.assert_series_equal(groups["shift_group"], again["shift_group"])
