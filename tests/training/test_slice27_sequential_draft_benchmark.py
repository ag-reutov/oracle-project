"""Slice 27 incremental draft-value benchmark.

Checkpoint isolation, side-aware encoding, Slice 26 consumption, holdout.
Research only; not a production feature.
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
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS
from dota_predictor.training.sequential_draft_benchmark import (
    CHECKPOINTS,
    SLICE27_DIAGNOSTIC_ONLY,
    ban_column_name,
    boundary_after_n_successful_picks,
    build_match_draft_index,
    checkpoint_pick_ban_features,
    classify_slice27,
    encode_side_aware_indicators,
    hero_column_name,
    run_slice27_sequential_draft_benchmark,
    successful_ban_prefix,
    successful_pick_prefix,
    train_pick_vocabulary,
)
from dota_predictor.training.sequential_draft_state import (
    ACTION_BAN,
    ACTION_PICK,
    BOUNDARY_CONVENTION,
    SIDE_DIRE,
    SIDE_RADIANT,
    build_draft_prefix_state,
    event_is_actual,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END

RADIANT_IDS = (11, 12, 13, 14, 15)
DIRE_IDS = (21, 22, 23, 24, 25)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _event(
    sequence: int,
    action: str,
    side: str,
    hero_id: int,
    *,
    was_successful: bool | None = None,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "action": action,
        "side": side,
        "hero_id": hero_id,
        "was_successful": was_successful,
    }


def _draft_with_bans_and_picks(*, failed_ban: bool = False) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        _event(0, ACTION_BAN, SIDE_DIRE, 50, was_successful=True),
        _event(1, ACTION_BAN, SIDE_RADIANT, 51, was_successful=True),
    ]
    if failed_ban:
        events.append(_event(2, ACTION_BAN, SIDE_RADIANT, 99, was_successful=False))
        seq = 3
    else:
        seq = 2
    # Interleaved picks: D, R, D, R, ... then fill to 10.
    pick_plan = [
        (SIDE_DIRE, 6),
        (SIDE_RADIANT, 1),
        (SIDE_DIRE, 7),
        (SIDE_RADIANT, 2),
        (SIDE_DIRE, 8),
        (SIDE_RADIANT, 3),
        (SIDE_DIRE, 9),
        (SIDE_RADIANT, 4),
        (SIDE_DIRE, 10),
        (SIDE_RADIANT, 5),
    ]
    for side, hero in pick_plan:
        events.append(_event(seq, ACTION_PICK, side, hero))
        seq += 1
    return events


def test_checkpoints_and_slice26_convention() -> None:
    assert CHECKPOINTS == (0, 2, 4, 6, 8, 10)
    assert BOUNDARY_CONVENTION == "before_event_t"
    assert SLICE27_DIAGNOSTIC_ONLY is True


def test_checkpoint_0_has_no_picks() -> None:
    events = _draft_with_bans_and_picks()
    boundary = boundary_after_n_successful_picks(events, n=0)
    assert boundary == 2  # first pick at sequence 2
    picks = successful_pick_prefix(events, n=0)
    assert picks == []
    state = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=boundary,
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    assert state["radiant_pick_hero_ids"] == ()
    assert state["dire_pick_hero_ids"] == ()
    assert state["dire_ban_hero_ids"] == (50,)
    assert state["radiant_ban_hero_ids"] == (51,)


def test_checkpoint_2_contains_first_two_successful_picks() -> None:
    events = _draft_with_bans_and_picks()
    picks = successful_pick_prefix(events, n=2)
    assert [(p["side"], p["hero_id"]) for p in picks] == [
        (SIDE_DIRE, 6),
        (SIDE_RADIANT, 1),
    ]
    boundary = boundary_after_n_successful_picks(events, n=2)
    assert boundary == int(picks[1]["sequence"]) + 1
    state = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=boundary,
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    assert state["dire_pick_hero_ids"] == (6,)
    assert state["radiant_pick_hero_ids"] == (1,)


def test_checkpoint_4_cannot_see_later_picks() -> None:
    events = _draft_with_bans_and_picks()
    boundary = boundary_after_n_successful_picks(events, n=4)
    state = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=boundary,
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    visible = set(state["radiant_pick_hero_ids"]) | set(state["dire_pick_hero_ids"])
    assert visible == {6, 1, 7, 2}
    assert 3 not in visible and 8 not in visible and 10 not in visible


def test_mutating_future_events_does_not_change_earlier_checkpoint() -> None:
    events = _draft_with_bans_and_picks()
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    vocab = (1, 2, 3, 6, 7, 8, 99)
    base = checkpoint_pick_ban_features(
        index, n_picks=4, pick_vocabulary=vocab, verify_slice26=True
    )
    mutated = [dict(e) for e in events]
    mutated[-1]["hero_id"] = 42
    mutated[-2]["hero_id"] = 43
    index2 = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in mutated])
    )[1]
    after = checkpoint_pick_ban_features(
        index2, n_picks=4, pick_vocabulary=vocab, verify_slice26=True
    )
    assert base == after


def test_side_encoding_distinguishes_radiant_vs_dire() -> None:
    encoded = encode_side_aware_indicators(
        radiant_hero_ids=[1, 2],
        dire_hero_ids=[6],
        vocabulary=[1, 2, 6, 9],
    )
    assert encoded[hero_column_name(1)] == 1.0
    assert encoded[hero_column_name(2)] == 1.0
    assert encoded[hero_column_name(6)] == -1.0
    assert encoded[hero_column_name(9)] == 0.0


def test_failed_bans_do_not_enter_successful_ban_features() -> None:
    events = _draft_with_bans_and_picks(failed_ban=True)
    assert event_is_actual(ACTION_BAN, False) is False
    boundary = boundary_after_n_successful_picks(events, n=2)
    bans = successful_ban_prefix(events, boundary_t=boundary)
    ban_heroes = {b["hero_id"] for b in bans}
    assert 99 not in ban_heroes
    assert 50 in ban_heroes and 51 in ban_heroes
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    feats = checkpoint_pick_ban_features(
        index,
        n_picks=2,
        pick_vocabulary=(1, 6),
        ban_vocabulary=(50, 51, 99),
        include_bans=True,
        verify_slice26=True,
    )
    assert feats is not None
    assert feats[ban_column_name(99)] == 0.0
    assert feats[ban_column_name(50)] == -1.0
    assert feats[ban_column_name(51)] == 1.0


def test_bans_after_checkpoint_are_invisible() -> None:
    events = [
        _event(0, ACTION_PICK, SIDE_RADIANT, 1),
        _event(1, ACTION_PICK, SIDE_DIRE, 6),
        _event(2, ACTION_BAN, SIDE_RADIANT, 50, was_successful=True),
        _event(3, ACTION_PICK, SIDE_RADIANT, 2),
    ]
    boundary = boundary_after_n_successful_picks(events, n=2)
    assert boundary == 2
    bans = successful_ban_prefix(events, boundary_t=boundary)
    assert bans == []


def test_terminal_10_pick_state_has_final_picks() -> None:
    events = _draft_with_bans_and_picks()
    picks = successful_pick_prefix(events, n=10)
    assert len(picks) == 10
    boundary = boundary_after_n_successful_picks(events, n=10)
    state = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=boundary,
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    assert set(state["radiant_pick_hero_ids"]) == {1, 2, 3, 4, 5}
    assert set(state["dire_pick_hero_ids"]) == {6, 7, 8, 9, 10}


def test_matrix_has_no_position_assignment_outcome_leakage() -> None:
    events = _draft_with_bans_and_picks()
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    feats = checkpoint_pick_ban_features(
        index, n_picks=6, pick_vocabulary=(1, 2, 3, 6, 7, 8)
    )
    assert feats is not None
    forbidden = {
        "position",
        "slot_in_side",
        "player_id",
        "radiant_win",
        "duration_seconds",
        "kills",
        "assignment",
    }
    assert forbidden.isdisjoint(feats.keys())


def test_unseen_hero_vocabulary_is_deterministic() -> None:
    events = _draft_with_bans_and_picks()
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    # Train vocab lacks hero 5; encoding still deterministic zeros for known cols.
    vocab = (1, 6)
    feats = checkpoint_pick_ban_features(index, n_picks=10, pick_vocabulary=vocab)
    assert feats is not None
    assert set(feats) == {hero_column_name(1), hero_column_name(6)}
    assert hero_column_name(5) not in feats
    train_vocab = train_pick_vocabulary({1: index}, [1])
    assert 5 in train_vocab


def test_feature_columns_remain_33() -> None:
    assert len(FEATURE_COLUMNS) == 33
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)


def test_classify_slice27_c_when_no_improvement() -> None:
    curve = pd.DataFrame(
        {
            "n_picks": [2, 4, 6, 8, 10],
            "delta_log_loss": [0.01, 0.02, 0.01, 0.0, 0.005],
        }
    )
    folds = pd.DataFrame(
        {
            "n_picks": [10, 10, 10, 10],
            "fold_id": [1, 2, 3, 4],
            "delta_log_loss": [0.01, 0.02, 0.0, 0.01],
        }
    )
    bootstrap = {
        str(n): {"ci_low": 0.0, "ci_high": 0.02, "mean": 0.01, "n": 100}
        for n in (2, 4, 6, 8, 10)
    }
    label, _rationale, pattern, frozen = classify_slice27(
        checkpoint_curve=curve, fold_deltas=folds, bootstrap=bootstrap
    )
    assert label.startswith("C")
    assert pattern == "D"
    assert frozen == ()


def test_holdout_excluded_from_benchmark(tmp_path: Path) -> None:
    end = FROZEN_DEVELOPMENT_END
    matches = []
    players = []
    draft_events = []
    # Enough chronological spread for walk-forward with small n_blocks.
    times = [
        end - timedelta(days=d)
        for d in (400, 350, 300, 250, 200, 150, 100, 50, 20, 5)
    ]
    hold_time = end + timedelta(days=10)
    all_times = times + [hold_time]
    for i, start in enumerate(all_times, start=1):
        matches.append(
            match_row(
                i,
                start_time=start,
                radiant_team_id=100 + (i % 3),
                dire_team_id=200 + (i % 3),
                radiant_win=bool(i % 2),
            )
        )
        rows = player_rows(i, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        for row in rows:
            if row["side"] == SIDE_RADIANT:
                row["hero_id"] = int(row["slot_in_side"]) + 1
            else:
                row["hero_id"] = int(row["slot_in_side"]) + 6
        players.extend(rows)
        for event in _draft_with_bans_and_picks():
            draft_events.append({**event, "match_id": i})

    matches_table = build_matches_table(matches, players)
    players_table = build_match_players_table(matches, players)
    draft_table = build_draft_events_table(draft_events)
    write_canonical_dataset(
        tmp_path,
        matches_table=matches_table,
        draft_events_table=draft_table,
        match_players_table=players_table,
    )
    config = FeatureStoreConfig(
        matches_path=tmp_path / MATCHES_FILENAME,
        match_players_path=tmp_path / MATCH_PLAYERS_FILENAME,
        draft_events_path=tmp_path / DRAFT_EVENTS_FILENAME,
    )
    from dota_predictor.training.walk_forward import WalkForwardConfig

    with connect(config) as store:
        report = run_slice27_sequential_draft_benchmark(
            store,
            walk_forward_config=WalkForwardConfig(n_blocks=3),
            run_ban_ablation=False,
        )
    assert report.n_holdout_excluded == 1
    assert report.n_development_matches == 10
    assert report.integrity["holdout_scored"] is False
    assert report.integrity["feature_columns_unchanged"] is True
    # Reference and candidate share identical OOS match counts per checkpoint.
    assert report.checkpoint_curve["n_oos"].nunique() == 1
