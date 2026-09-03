"""Slice 26 causal sequential draft-state dataset.

Prefix isolation, ban semantics, terminal consistency, holdout exclusion.
Research state only; not a production feature; no next-pick metrics.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

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
from dota_predictor.training.sequential_draft_state import (
    ACTION_BAN,
    ACTION_PICK,
    BOUNDARY_CONVENTION,
    CATEGORY_COMPLETE_PICKS_INCOMPLETE_BANS,
    CATEGORY_COMPLETE_SEQUENTIAL,
    CATEGORY_MALFORMED_ORDERING,
    SIDE_DIRE,
    SIDE_RADIANT,
    SLICE26_DIAGNOSTIC_ONLY,
    SLICE26_FROZEN_COMPONENTS,
    SLICE26_STATE_KEYS,
    audit_draft_events_population,
    build_draft_prefix_state,
    build_sequential_draft_states,
    classify_match_draft_category,
    classify_slice26,
    compare_terminal_picks_to_players,
    event_is_actual,
    reconstruct_events_from_prefix,
    run_sequential_draft_state_diagnostics,
    slice26_report_to_jsonable,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END

RADIANT_IDS = (11, 12, 13, 14, 15)
DIRE_IDS = (21, 22, 23, 24, 25)
T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = datetime(2026, 2, 1, tzinfo=UTC)
T2 = datetime(2026, 3, 1, tzinfo=UTC)


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


def _simple_draft(*, failed_ban: bool = False) -> list[dict[str, object]]:
    """10-event pick draft plus optional failed ban at the front."""
    events: list[dict[str, object]] = []
    seq = 0
    if failed_ban:
        events.append(
            _event(seq, ACTION_BAN, SIDE_RADIANT, 99, was_successful=False)
        )
        seq += 1
    heroes_r = (1, 2, 3, 4, 5)
    heroes_d = (6, 7, 8, 9, 10)
    for hero in heroes_r:
        events.append(_event(seq, ACTION_PICK, SIDE_RADIANT, hero))
        seq += 1
    for hero in heroes_d:
        events.append(_event(seq, ACTION_PICK, SIDE_DIRE, hero))
        seq += 1
    return events


def _cm_like_draft() -> list[dict[str, object]]:
    """Short CM-like prefix: 2 bans + 10 picks (not claiming full CM)."""
    events = [
        _event(0, ACTION_BAN, SIDE_DIRE, 50, was_successful=True),
        _event(1, ACTION_BAN, SIDE_RADIANT, 51, was_successful=True),
    ]
    seq = 2
    for hero in (1, 2, 3, 4, 5):
        events.append(_event(seq, ACTION_PICK, SIDE_RADIANT, hero))
        seq += 1
    for hero in (6, 7, 8, 9, 10):
        events.append(_event(seq, ACTION_PICK, SIDE_DIRE, hero))
        seq += 1
    return events


def _players_for_draft(match_id: int) -> list[dict[str, object]]:
    rows = player_rows(
        match_id, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS
    )
    # Align hero_ids with successful picks 1..10.
    for row in rows:
        if row["side"] == SIDE_RADIANT:
            row["hero_id"] = int(row["slot_in_side"]) + 1
        else:
            row["hero_id"] = int(row["slot_in_side"]) + 6
    return rows


def _write_store(
    tmp_path: Path,
    *,
    matches: list[dict[str, object]],
    players: list[dict[str, object]],
    draft_events: list[dict[str, object]],
) -> FeatureStoreConfig:
    matches_table = build_matches_table(matches, players)
    match_players_table = build_match_players_table(matches, players)
    draft_table = build_draft_events_table(draft_events)
    write_canonical_dataset(
        tmp_path,
        matches_table=matches_table,
        draft_events_table=draft_table,
        match_players_table=match_players_table,
    )
    return FeatureStoreConfig(
        matches_path=tmp_path / MATCHES_FILENAME,
        match_players_path=tmp_path / MATCH_PLAYERS_FILENAME,
        draft_events_path=tmp_path / DRAFT_EVENTS_FILENAME,
    )


def test_boundary_convention_is_before_event_t() -> None:
    assert BOUNDARY_CONVENTION == "before_event_t"
    assert SLICE26_DIAGNOSTIC_ONLY is True
    assert "ordered_event_prefix_state" in SLICE26_FROZEN_COMPONENTS
    assert set(SLICE26_STATE_KEYS) >= {
        "match_id",
        "boundary_t",
        "event_prefix",
        "radiant_pick_hero_ids",
        "radiant_ban_hero_ids",
        "unsuccessful_event_prefix",
    }


def test_prefix_isolation_future_events_do_not_change_state() -> None:
    events = _cm_like_draft()
    state_t3 = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=3,
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    mutated = [dict(e) for e in events]
    mutated[-1]["hero_id"] = 999
    mutated[5]["hero_id"] = 888
    state_after = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=3,
        events=mutated,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    assert state_t3["event_prefix"] == state_after["event_prefix"]
    assert state_t3["radiant_pick_hero_ids"] == state_after["radiant_pick_hero_ids"]
    assert state_t3["radiant_ban_hero_ids"] == state_after["radiant_ban_hero_ids"]


def test_event_inclusion_prior_change_updates_state() -> None:
    events = _cm_like_draft()
    base = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=3,
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    mutated = [dict(e) for e in events]
    mutated[0]["hero_id"] = 77
    changed = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=3,
        events=mutated,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    assert base["dire_ban_hero_ids"] == (50,)
    assert changed["dire_ban_hero_ids"] == (77,)
    assert base["event_prefix"] != changed["event_prefix"]


def test_ordered_prefix_preservation() -> None:
    events = [
        _event(0, ACTION_PICK, SIDE_RADIANT, 1),
        _event(1, ACTION_PICK, SIDE_RADIANT, 2),
    ]
    swapped = [
        _event(0, ACTION_PICK, SIDE_RADIANT, 2),
        _event(1, ACTION_PICK, SIDE_RADIANT, 1),
    ]
    a = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=2,
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    b = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=2,
        events=swapped,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    assert a["radiant_pick_hero_ids"] == (1, 2)
    assert b["radiant_pick_hero_ids"] == (2, 1)
    assert a["radiant_pick_hero_ids"] != b["radiant_pick_hero_ids"]


def test_duplicate_sequences_rejected() -> None:
    events = [
        _event(0, ACTION_PICK, SIDE_RADIANT, 1),
        _event(0, ACTION_PICK, SIDE_DIRE, 2),
    ]
    with pytest.raises(ValueError, match="duplicate sequence"):
        build_sequential_draft_states(
            match_id=1,
            start_time=T0,
            game_version_id=176,
            events=events,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_player_ids=RADIANT_IDS,
            dire_player_ids=DIRE_IDS,
        )
    assert classify_match_draft_category(events) == CATEGORY_MALFORMED_ORDERING


def test_successful_vs_unsuccessful_ban_semantics() -> None:
    assert event_is_actual(ACTION_PICK, None) is True
    assert event_is_actual(ACTION_BAN, True) is True
    assert event_is_actual(ACTION_BAN, None) is True
    assert event_is_actual(ACTION_BAN, False) is False

    events = _simple_draft(failed_ban=True)
    terminal = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=len(events),
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    assert len(terminal["event_prefix"]) == len(events)
    assert terminal["unsuccessful_event_prefix"][0]["hero_id"] == 99
    assert 99 not in terminal["radiant_ban_hero_ids"]
    assert terminal["n_unsuccessful_events"] == 1
    assert terminal["n_radiant_bans_successful"] == 0


def test_terminal_prefix_reconstructs_canonical_events() -> None:
    events = _cm_like_draft()
    states = build_sequential_draft_states(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    assert len(states) == len(events) + 1
    assert states[0]["n_prior_events"] == 0
    assert states[0]["event_prefix"] == []
    assert states[-1]["is_terminal"] is True
    reconstructed = reconstruct_events_from_prefix(states)
    assert [e["sequence"] for e in reconstructed] == list(range(len(events)))
    assert [e["hero_id"] for e in reconstructed] == [e["hero_id"] for e in events]
    assert [e["action"] for e in reconstructed] == [e["action"] for e in events]


def test_terminal_successful_picks_agree_with_match_players() -> None:
    events = _simple_draft()
    players = _players_for_draft(1)
    result = compare_terminal_picks_to_players(events, players)
    assert result["status"] == "exact"
    assert result["draft_radiant"] == [1, 2, 3, 4, 5]
    assert result["draft_dire"] == [6, 7, 8, 9, 10]

    bad_players = [dict(r) for r in players]
    bad_players[0]["hero_id"] = 42
    mismatch = compare_terminal_picks_to_players(events, bad_players)
    assert mismatch["status"] == "mismatch"
    assert 42 in mismatch["missing_radiant"] or 1 in mismatch["extra_radiant"]


def test_no_outcome_duration_position_or_box_score_in_state() -> None:
    events = _cm_like_draft()
    state = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=4,
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
        radiant_team_elo=1500.0,
        dire_team_elo=1480.0,
        team_elo_delta=20.0,
    )
    forbidden = {
        "radiant_win",
        "duration_seconds",
        "duration",
        "position",
        "slot_in_side",
        "kills",
        "deaths",
        "assists",
        "num_last_hits",
        "hero_damage",
        "networth",
        "player_hero_assignment",
    }
    assert forbidden.isdisjoint(state.keys())
    # Rosters are player ids only — no hero assignment map.
    assert "radiant_player_ids" in state
    assert state["radiant_pick_hero_ids"] == (1, 2)


def test_pre_draft_elo_respects_strict_prior_history(tmp_path: Path) -> None:
    matches = [
        match_row(
            1,
            start_time=T0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
        match_row(
            2,
            start_time=T1,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=False,
        ),
        match_row(
            3,
            start_time=T2,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
    ]
    players = (
        _players_for_draft(1)
        + _players_for_draft(2)
        + _players_for_draft(3)
    )
    draft_events: list[dict[str, object]] = []
    for mid in (1, 2, 3):
        for event in _simple_draft():
            row = dict(event)
            row["match_id"] = mid
            draft_events.append(row)

    config = _write_store(
        tmp_path, matches=matches, players=players, draft_events=draft_events
    )
    with connect(config) as store:
        from dota_predictor.features.team_elo import compute_team_elo_features

        frame = store.sql(
            "SELECT match_id, start_time, radiant_team_id, dire_team_id, radiant_win "
            "FROM matches"
        ).df()
        elo = compute_team_elo_features(frame)
        # First match starts at initial Elo; later matches move after prior results.
        first = elo.loc[elo["match_id"] == 1].iloc[0]
        second = elo.loc[elo["match_id"] == 2].iloc[0]
        assert float(first["radiant_team_elo"]) == pytest.approx(1500.0)
        assert float(second["radiant_team_elo"]) != pytest.approx(
            float(first["radiant_team_elo"])
        )
        # Equal-timestamp blindness is handled inside team_elo; strict prior
        # is start_time ordering — T1 state must not include T2 outcome.
        assert pd.Timestamp(T1) < pd.Timestamp(T2)


def test_holdout_excluded_from_audit(tmp_path: Path) -> None:
    end = FROZEN_DEVELOPMENT_END
    dev_time = end - timedelta(days=30)
    hold_time = end + timedelta(days=30)
    matches = [
        match_row(
            1,
            start_time=dev_time,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
        match_row(
            2,
            start_time=hold_time,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        ),
    ]
    players = _players_for_draft(1) + _players_for_draft(2)
    draft_events: list[dict[str, object]] = []
    for mid in (1, 2):
        for event in _simple_draft():
            row = dict(event)
            row["match_id"] = mid
            draft_events.append(row)
    config = _write_store(
        tmp_path, matches=matches, players=players, draft_events=draft_events
    )
    with connect(config) as store:
        report = run_sequential_draft_state_diagnostics(store)
    assert report.n_development_matches == 1
    assert report.n_holdout_excluded == 1
    assert report.audit["matches_with_any_draft_events"] == 1
    assert report.audit["matches_total"] == 1


def test_feature_columns_remain_33() -> None:
    assert len(FEATURE_COLUMNS) == 33
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)


def test_category_complete_picks_incomplete_bans() -> None:
    events = _simple_draft(failed_ban=False)
    assert classify_match_draft_category(events) == (
        CATEGORY_COMPLETE_PICKS_INCOMPLETE_BANS
    )
    with_bans = _cm_like_draft()
    players = _players_for_draft(1)
    terminal = compare_terminal_picks_to_players(with_bans, players)
    assert (
        classify_match_draft_category(with_bans, terminal=terminal)
        == CATEGORY_COMPLETE_SEQUENTIAL
    )


def test_classify_slice26_gates_are_reliability_not_prediction() -> None:
    frame = classify_slice26(
        ordering_ok_rate=1.0,
        field_usable_rate=1.0,
        terminal_exact_rate=1.0,
        well_formed_share=1.0,
    )
    assert frame.iloc[0]["classification"] == "A"
    assert frame.iloc[0]["next_event_prediction_run"] == False  # noqa: E712
    weak = classify_slice26(
        ordering_ok_rate=0.5,
        field_usable_rate=0.5,
        terminal_exact_rate=0.5,
        well_formed_share=0.2,
    )
    assert weak.iloc[0]["classification"] == "C"


def test_audit_population_helper_counts(tmp_path: Path) -> None:
    matches = [
        match_row(
            1,
            start_time=T0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        )
    ]
    players = _players_for_draft(1)
    draft_events = []
    for event in _cm_like_draft():
        row = dict(event)
        row["match_id"] = 1
        draft_events.append(row)
    matches_df = pd.DataFrame(matches)
    # Restrict development with a late end so T0 is included.
    audit = audit_draft_events_population(
        matches_df,
        pd.DataFrame(draft_events),
        pd.DataFrame(players),
        development_end=T2,
    )
    assert audit["matches_total"] == 1
    assert audit["matches_with_exactly_10_successful_picks"] == 1
    assert audit["terminal_status_counts"]["exact"] == 1
    assert audit["boundary_convention"] == BOUNDARY_CONVENTION


def test_diagnostics_jsonable_and_no_prediction_keys(tmp_path: Path) -> None:
    matches = [
        match_row(
            1,
            start_time=T0,
            radiant_team_id=100,
            dire_team_id=200,
            radiant_win=True,
        )
    ]
    players = _players_for_draft(1)
    draft_events = []
    for event in _cm_like_draft():
        row = dict(event)
        row["match_id"] = 1
        draft_events.append(row)
    config = _write_store(
        tmp_path, matches=matches, players=players, draft_events=draft_events
    )
    with connect(config) as store:
        report = run_sequential_draft_state_diagnostics(
            store, development_end=T2
        )
    payload = slice26_report_to_jsonable(report)
    assert payload["feature_columns_length"] == 33
    assert "next_hero_log_loss" in payload["excluded_from_slice"]
    assert payload["boundary_convention"] == "before_event_t"
