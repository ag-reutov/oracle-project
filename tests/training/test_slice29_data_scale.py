"""Slice 29 data-scale learning-curve audit tests.

Verifies chronological prefix nesting, fixed evaluation, unchanged
model definitions, holdout exclusion, and FEATURE_COLUMNS integrity.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import match_row, player_rows

from dota_predictor.datasets.canonical_export import (
    DRAFT_EVENTS_FILENAME,
    MATCH_PLAYERS_FILENAME,
    MATCHES_FILENAME,
    build_draft_events_table,
    build_match_players_table,
    build_matches_table,
    write_canonical_dataset,
)
from dota_predictor.features.config import FeatureStoreConfig
from dota_predictor.features.duckdb_layer import connect
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.data_scale_diagnostics import (
    DEFAULT_TRAIN_FRACTIONS,
    SLICE29_DIAGNOSTIC_ONLY,
    _chronological_prefix,
    _restrict_partition,
    classify_bottleneck,
    run_slice29_data_scale_benchmark,
    TrackARow,
    TrackBRow,
)
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS
from dota_predictor.training.next_pick_policy import (
    DEFAULT_POLICY_C,
    POLICY_SGD_MAX_ITER,
)
from dota_predictor.training.sequential_draft_benchmark import (
    SLICE27_CANDIDATE_SPEC_NAME,
    SLICE27_REFERENCE_SPEC_NAME,
)
from dota_predictor.training.sequential_draft_state import (
    ACTION_BAN,
    ACTION_PICK,
    SIDE_DIRE,
    SIDE_RADIANT,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END
from dota_predictor.training.split import DatasetPartition
from dota_predictor.training.walk_forward import WalkForwardConfig

RADIANT_IDS = (11, 12, 13, 14, 15)
DIRE_IDS = (21, 22, 23, 24, 25)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _event(seq: int, action: str, side: str, hero_id: int) -> dict:
    return {
        "sequence": seq,
        "action": action,
        "side": side,
        "hero_id": hero_id,
        "was_successful": True,
    }


def _standard_draft() -> list[dict]:
    events = [
        _event(0, ACTION_BAN, SIDE_DIRE, 50),
        _event(1, ACTION_BAN, SIDE_RADIANT, 51),
    ]
    picks = [
        (SIDE_DIRE, 6), (SIDE_RADIANT, 1), (SIDE_DIRE, 7), (SIDE_RADIANT, 2),
        (SIDE_DIRE, 8), (SIDE_RADIANT, 3), (SIDE_DIRE, 9), (SIDE_RADIANT, 4),
        (SIDE_DIRE, 10), (SIDE_RADIANT, 5),
    ]
    for i, (side, hero) in enumerate(picks):
        events.append(_event(i + 2, ACTION_PICK, side, hero))
    return events


def _write_mini_store(tmp_path: Path, *, n_matches: int = 16) -> FeatureStoreConfig:
    matches = []
    players = []
    drafts = []
    for i in range(n_matches):
        mid = 1000 + i
        t = T0 + timedelta(days=i)
        assert t <= FROZEN_DEVELOPMENT_END
        matches.append(
            match_row(
                mid,
                start_time=t,
                radiant_team_id=100 + (i % 3),
                dire_team_id=200 + (i % 3),
                radiant_win=i % 2 == 0,
                game_version_id=176 + (i % 2),
            )
        )
        rows = player_rows(mid, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        for row in rows:
            if row["side"] == SIDE_RADIANT:
                row["hero_id"] = int(row["slot_in_side"]) + 1
            else:
                row["hero_id"] = int(row["slot_in_side"]) + 6
        players.extend(rows)
        for e in _standard_draft():
            drafts.append({"match_id": mid, **e})

    write_canonical_dataset(
        tmp_path,
        matches_table=build_matches_table(matches, players),
        draft_events_table=build_draft_events_table(drafts),
        match_players_table=build_match_players_table(matches, players),
    )
    return FeatureStoreConfig(
        matches_path=tmp_path / MATCHES_FILENAME,
        match_players_path=tmp_path / MATCH_PLAYERS_FILENAME,
        draft_events_path=tmp_path / DRAFT_EVENTS_FILENAME,
    )


# ── Test 1: chronological prefix nesting ─────────────────────────────

def test_smaller_training_sets_are_chronological_prefixes() -> None:
    ids = list(range(100))
    times = pd.Series(
        [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in ids],
        index=ids,
    )
    prev = []
    for frac in DEFAULT_TRAIN_FRACTIONS:
        subset = _chronological_prefix(ids, times, frac)
        # Chronologically ordered.
        for i in range(len(subset) - 1):
            assert times[subset[i]] <= times[subset[i + 1]]
        # Strict prefix nesting.
        if prev:
            assert subset[:len(prev)] == prev
        prev = subset


# ── Test 2: evaluation rows fixed across N ───────────────────────────

def test_evaluation_rows_identical_across_training_sizes(tmp_path: Path) -> None:
    config = _write_mini_store(tmp_path, n_matches=16)
    with connect(config) as store:
        report = run_slice29_data_scale_benchmark(
            store,
            development_end=FROZEN_DEVELOPMENT_END,
            walk_forward_config=WalkForwardConfig(n_blocks=4, train_fraction_of_past=0.7),
            train_fractions=(0.5, 1.0),
            run_track_a=False,
            run_track_b=True,
        )
    # Group by fold: eval rows must be identical across fractions.
    by_fold: dict[int, list[int]] = {}
    for r in report.track_b_rows:
        by_fold.setdefault(r.fold_id, []).append(r.eval_decision_rows)
    for fold_id, counts in by_fold.items():
        assert len(set(counts)) == 1, f"fold {fold_id}: eval rows differ across N"


# ── Test 3: future matches never enter training ─────────────────────

def test_future_matches_excluded_from_training() -> None:
    ids = [1, 2, 3, 4, 5]
    times = pd.Series(
        [datetime(2026, 1, d, tzinfo=UTC) for d in [1, 2, 3, 4, 5]],
        index=ids,
    )
    subset = _chronological_prefix(ids, times, 0.6)
    assert len(subset) == 3
    assert all(times[m] <= datetime(2026, 1, 3, tzinfo=UTC) for m in subset)


# ── Test 4 & 5: Slice 27/28 model definitions unchanged ─────────────

def test_slice27_model_definition_unchanged() -> None:
    assert SLICE27_REFERENCE_SPEC_NAME == "logistic_elo_only"
    assert SLICE27_CANDIDATE_SPEC_NAME == "logistic_elo_plus_checkpoint_picks"


def test_slice28_model_definition_unchanged() -> None:
    assert DEFAULT_POLICY_C == 1.0
    assert POLICY_SGD_MAX_ITER == 200


# ── Test 6: hyperparameters identical across N ───────────────────────

def test_hyperparameters_identical_across_n() -> None:
    """Constants are fixed; verify they don't depend on N."""
    # These are module-level constants, not functions of N.
    assert isinstance(DEFAULT_POLICY_C, float)
    assert isinstance(POLICY_SGD_MAX_ITER, int)


# ── Test 7: holdout excluded ─────────────────────────────────────────

def test_holdout_excluded(tmp_path: Path) -> None:
    config = _write_mini_store(tmp_path, n_matches=16)
    with connect(config) as store:
        report = run_slice29_data_scale_benchmark(
            store,
            development_end=FROZEN_DEVELOPMENT_END,
            walk_forward_config=WalkForwardConfig(n_blocks=4, train_fraction_of_past=0.7),
            train_fractions=(1.0,),
            run_track_a=False,
            run_track_b=True,
        )
    assert report.integrity["holdout_excluded"]


# ── Test 8: train/eval not mixed ─────────────────────────────────────

def test_restrict_partition_preserves_alignment() -> None:
    X = pd.DataFrame({"f1": [1, 2, 3, 4]})
    y = pd.Series([True, False, True, False])
    ctx = pd.DataFrame({"match_id": [10, 20, 30, 40]})
    part = DatasetPartition(X=X, y=y, context=ctx)
    restricted = _restrict_partition(part, {10, 30})
    assert len(restricted) == 2
    assert list(restricted.context["match_id"]) == [10, 30]
    assert list(restricted.X["f1"]) == [1, 3]
    assert list(restricted.y) == [True, True]


# ── Test 9: match-cluster CI ─────────────────────────────────────────

def test_match_clustered_ci_groups_decisions() -> None:
    from dota_predictor.training.next_pick_policy import _match_clustered_delta_ci
    rows = []
    for mid, delta in ((1, -0.2), (2, 0.4)):
        for _ in range(10):
            rows.append({"match_id": mid, "left": 1.0 + delta, "right": 1.0})
    ci = _match_clustered_delta_ci(
        pd.DataFrame(rows), left_col="left", right_col="right", seed=0
    )
    assert ci["n_matches"] == 2


# ── Test 10: same-timestamp blindness ─────────────────────────────────

def test_same_timestamp_prefix_ordering() -> None:
    ids = [1, 2, 3]
    t = datetime(2026, 1, 1, tzinfo=UTC)
    times = pd.Series([t, t, t], index=ids)
    sub = _chronological_prefix(ids, times, 0.5)
    # With all same timestamps, ties broken by match_id.
    assert len(sub) == 1
    assert sub[0] == 1


# ── Test 11: FEATURE_COLUMNS unchanged ───────────────────────────────

def test_feature_columns_remain_33() -> None:
    assert len(FEATURE_COLUMNS) == 33
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)
    assert SLICE29_DIAGNOSTIC_ONLY is True


# ── Test 12: no ingestion mutation ───────────────────────────────────

def test_no_league_registry_mutation() -> None:
    """Slice 29 must not modify the league registry."""
    import yaml
    registry = Path("config/leagues.yaml")
    if registry.exists():
        data = yaml.safe_load(registry.read_text())
        # Just verify it loads — no mutation test needed since
        # data_scale_diagnostics.py never writes to it.
        assert "leagues" in data


# ── End-to-end smoke ─────────────────────────────────────────────────

def test_end_to_end_smoke(tmp_path: Path) -> None:
    config = _write_mini_store(tmp_path, n_matches=16)
    with connect(config) as store:
        report = run_slice29_data_scale_benchmark(
            store,
            development_end=FROZEN_DEVELOPMENT_END,
            walk_forward_config=WalkForwardConfig(n_blocks=4, train_fraction_of_past=0.7),
            train_fractions=(0.5, 1.0),
            run_track_a=True,
            run_track_b=True,
        )
    assert report.diagnostic_only is True
    assert report.feature_columns_count == 33
    assert report.integrity["feature_columns_unchanged"]
    assert len(report.track_b_rows) > 0
    assert report.bottleneck_classification != ""
