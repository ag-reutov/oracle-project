"""Tests for leakage-safe PRE_DRAFT expected-position assignment.

Observed current-match STRATZ position is evaluation-only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from expected_position_helpers import assign_expected_positions_frame
from hero_meta_helpers import match_row
from player_position_helpers import assign_positions, draft_and_player_rows

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import REFERENCE_SCHEMA_VERSION
from dota_predictor.features.expected_position import (
    EXPECTED_POSITION_COLUMNS,
    LEAKAGE_COLUMNS,
    assign_expected_positions,
    player_position_scores,
)
from dota_predictor.features.player_position import EXPLICIT_POSITION_LABELS
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
)
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS

T0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
T_TIE = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
T2 = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)

VERSION_A = 10
M1, M2, M3 = 4001, 1002, 3003
P1, P2, P3, P4, P5 = 1, 2, 3, 4, 5
P6, P7, P8, P9, P10 = 6, 7, 8, 9, 10
P11, P12, P13, P14, P15 = 11, 12, 13, 14, 15
P16, P17, P18, P19, P20 = 16, 17, 18, 19, 20
RADIANT = (P1, P2, P3, P4, P5)
DIRE = (P6, P7, P8, P9, P10)
RADIANT_HEROES = (1, 2, 3, 4, 5)
DIRE_HEROES = (6, 7, 8, 9, 10)
TEAM_A, TEAM_B = 100, 200


def _row(frame: pd.DataFrame, match_id: int, player_id: int) -> pd.Series:
    subset = frame[(frame["match_id"] == match_id) & (frame["player_id"] == player_id)]
    assert len(subset) == 1
    return subset.iloc[0]


def _unique_side_positions(match_id: int, players: tuple[int, ...]) -> dict[tuple[int, int], str]:
    return {
        (match_id, player_id): label
        for player_id, label in zip(players, EXPLICIT_POSITION_LABELS, strict=True)
    }


def _tables(
    specs: list[dict],
    *,
    positions: dict[tuple[int, int], str | None] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    matches: list[dict] = []
    players: list[dict] = []
    drafts: list[dict] = []
    for spec in specs:
        matches.append(
            match_row(
                spec["match_id"],
                start_time=spec["start_time"],
                radiant_win=spec["radiant_win"],
                game_version_id=spec.get("game_version_id", VERSION_A),
                radiant_team_id=spec.get("radiant_team_id", TEAM_A),
                dire_team_id=spec.get("dire_team_id", TEAM_B),
            )
        )
        draft_rows, player_rows = draft_and_player_rows(
            spec["match_id"],
            radiant_player_ids=spec.get("radiant_players", RADIANT),
            dire_player_ids=spec.get("dire_players", DIRE),
            radiant_hero_ids=spec.get("radiant_heroes", RADIANT_HEROES),
            dire_hero_ids=spec.get("dire_heroes", DIRE_HEROES),
        )
        drafts.extend(draft_rows)
        players.extend(player_rows)
    if positions:
        assign_positions(players, positions)
    return matches, players, drafts


def _assemble(
    tmp_path: Path,
    specs: list[dict],
    *,
    positions: dict[tuple[int, int], str | None] | None = None,
    method: str = "previous",
    match_id: int | None = None,
) -> pd.DataFrame:
    matches, players, drafts = _tables(specs, positions=positions)
    return assign_expected_positions_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        method=method,
        match_id=match_id,
    )


def test_expected_position_not_in_win_model_features() -> None:
    for column in ("expected_position", "assigned_position_score"):
        assert column not in FEATURE_COLUMNS
        assert column not in SNAPSHOT_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
    assert "expected_position" not in PRE_DRAFT_SNAPSHOT_SQL


def test_schema_versions_unchanged() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 3
    assert REFERENCE_SCHEMA_VERSION == 1


def test_scoring_ignores_current_observed_position() -> None:
    row = pd.DataFrame(
        [
            {
                "previous_explicit_position": "POSITION_5",
                "position": "POSITION_1",
                "lane": "SAFE_LANE",
                "role": "CORE",
                "hero_id": 99,
                "won": True,
                "slot_in_side": 0,
                "prior_share_position_1": 1.0,
                "prior_share_position_2": 0.0,
                "prior_share_position_3": 0.0,
                "prior_share_position_4": 0.0,
                "prior_share_position_5": 0.0,
                "recent_5_explicit_games": 0,
                "recent_10_explicit_games": 0,
                "recent_20_explicit_games": 0,
                **{
                    f"recent_{w}_games_position_{n}": 0
                    for w in (5, 10, 20)
                    for n in range(1, 6)
                },
            }
        ]
    )
    scores = player_position_scores(row, method="previous")
    assert scores.shape == (1, 5)
    assert scores[0, 4] == pytest.approx(1.0)
    assert scores[0, 0] == pytest.approx(0.0)
    for column in LEAKAGE_COLUMNS:
        assert column not in (
            "previous_explicit_position",
            "prior_share_position_1",
        )


def test_first_match_assigns_by_player_id_with_zero_confidence(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [{"match_id": M1, "start_time": T0, "radiant_win": True}],
        positions=_unique_side_positions(M1, RADIANT)
        | _unique_side_positions(M1, DIRE),
        method="previous",
    )
    assert list(frame.columns) == list(EXPECTED_POSITION_COLUMNS)
    radiant = frame[(frame["match_id"] == M1) & (frame["side"] == "RADIANT")].sort_values(
        "player_id"
    )
    assert list(radiant["expected_position"]) == list(EXPLICIT_POSITION_LABELS)
    assert (radiant["assigned_position_score"] == 0).all()
    assert (radiant["roster_assignment_margin"] == 0).all()
    assert (radiant["evidence_tier"] == "none").all()
    # Observed current labels are not the assignment source.
    assert _row(frame, M1, P1)["observed_position"] == "POSITION_1"
    assert _row(frame, M1, P1)["expected_position"] == "POSITION_1"


def test_unique_previous_positions_are_copied_jointly(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        {
            (M2, P1): "POSITION_5",
            (M2, P2): "POSITION_4",
            (M2, P3): "POSITION_3",
            (M2, P4): "POSITION_2",
            (M2, P5): "POSITION_1",
        }
    )
    positions.update(_unique_side_positions(M2, DIRE))
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        method="previous",
        match_id=M2,
    )
    for player_id, label in zip(RADIANT, EXPLICIT_POSITION_LABELS, strict=True):
        row = _row(frame, M2, player_id)
        assert row["expected_position"] == label
        assert row["previous_explicit_position"] == label
        assert row["assigned_position_score"] == pytest.approx(1.0)
    # Current observed for P1 is POSITION_5; expected stays previous POSITION_1.
    assert _row(frame, M2, P1)["observed_position"] == "POSITION_5"
    assert _row(frame, M2, P1)["expected_position"] == "POSITION_1"


def test_collision_uses_player_id_not_observed_position(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, DIRE)
    positions.update(
        {
            (M1, P1): "POSITION_1",
            (M1, P2): "POSITION_1",
            (M1, P3): "POSITION_2",
            (M1, P4): "POSITION_3",
            (M1, P5): "POSITION_4",
            (M2, P1): "POSITION_5",
            (M2, P2): "POSITION_1",
            (M2, P3): "POSITION_2",
            (M2, P4): "POSITION_3",
            (M2, P5): "POSITION_4",
        }
    )
    positions.update(_unique_side_positions(M2, DIRE))
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        method="previous",
        match_id=M2,
    )
    # Lower player_id keeps the contested POSITION_1; the other gets leftover 5.
    assert _row(frame, M2, P1)["expected_position"] == "POSITION_1"
    assert _row(frame, M2, P2)["expected_position"] == "POSITION_5"
    assert _row(frame, M2, P2)["assigned_position_score"] == pytest.approx(0.0)
    assert _row(frame, M2, P2)["observed_position"] == "POSITION_1"


def test_timestamp_tie_does_not_use_peer_observed_position(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(
        {
            (M2, P1): "POSITION_5",
            (M3, P1): "POSITION_4",
        }
    )
    positions.update(
        {
            (M2, pid): label
            for pid, label in zip((P11, P12, P13, P14, P15), EXPLICIT_POSITION_LABELS)
        }
    )
    positions.update(
        {
            (M3, pid): label
            for pid, label in zip((P16, P17, P18, P19, P20), EXPLICIT_POSITION_LABELS)
        }
    )
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {
                "match_id": M2,
                "start_time": T_TIE,
                "radiant_win": True,
                "dire_players": (P11, P12, P13, P14, P15),
                "dire_heroes": (16, 17, 18, 19, 20),
            },
            {
                "match_id": M3,
                "start_time": T_TIE,
                "radiant_win": False,
                "radiant_heroes": (11, 12, 13, 14, 15),
                "dire_players": (P16, P17, P18, P19, P20),
                "dire_heroes": (16, 17, 18, 19, 20),
            },
        ],
        positions=positions,
        method="previous",
    )
    for match_id in (M2, M3):
        row = _row(frame, match_id, P1)
        assert row["previous_explicit_position"] == "POSITION_1"
        assert row["expected_position"] == "POSITION_1"


def test_later_series_map_may_use_earlier_map(tmp_path: Path) -> None:
    positions = _unique_side_positions(M1, RADIANT) | _unique_side_positions(M1, DIRE)
    positions.update(_unique_side_positions(M2, RADIANT) | _unique_side_positions(M2, DIRE))
    frame = _assemble(
        tmp_path,
        [
            {"match_id": M1, "start_time": T0, "radiant_win": True},
            {"match_id": M2, "start_time": T1, "radiant_win": False},
        ],
        positions=positions,
        method="previous",
        match_id=M2,
    )
    assert _row(frame, M2, P1)["expected_position"] == "POSITION_1"


def test_side_assignment_is_always_unique_1_to_5(tmp_path: Path) -> None:
    frame = _assemble(
        tmp_path,
        [{"match_id": M1, "start_time": T0, "radiant_win": True}],
        method="career",
    )
    for side in ("RADIANT", "DIRE"):
        labels = set(frame.loc[frame["side"] == side, "expected_position"])
        assert labels == set(EXPLICIT_POSITION_LABELS)


def test_assign_empty_state() -> None:
    empty = pd.DataFrame(
        columns=["match_id", "player_id", "side", "position"]
    )
    out = assign_expected_positions(empty, method="previous")
    assert list(out.columns) == list(EXPECTED_POSITION_COLUMNS)
    assert len(out) == 0


def _zero_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {
        "start_time": T1,
        "game_version_id": VERSION_A,
        "team_id": TEAM_A,
        "previous_explicit_position": None,
        "prior_explicit_position_games": 0,
        "recent_position_stability": None,
        "historical_modal_position": None,
        "recent_5_modal_position": None,
        "recent_10_modal_position": None,
        "recent_20_modal_position": None,
        **{f"prior_share_position_{n}": 0.0 for n in range(1, 6)},
        **{f"recent_{w}_explicit_games": 0 for w in (5, 10, 20)},
        **{
            f"recent_{w}_games_position_{n}": 0
            for w in (5, 10, 20)
            for n in range(1, 6)
        },
        **{f"version_prior_games_position_{n}": 0 for n in range(1, 6)},
    }
    return metrics


def _side_state(
    *,
    match_id: int,
    side: str,
    players: tuple[int, ...],
    observed: tuple[str | None, ...],
    overrides: dict[int, dict[str, object]] | None = None,
) -> pd.DataFrame:
    rows = []
    for player_id, position in zip(players, observed, strict=True):
        row: dict[str, object] = {
            "match_id": match_id,
            "player_id": player_id,
            "side": side,
            "position": position,
            **_zero_metrics(),
        }
        if overrides and player_id in overrides:
            row.update(overrides[player_id])
        rows.append(row)
    return pd.DataFrame(rows)


def test_hierarchical_uses_recent_10_not_previous_one_hot() -> None:
    """Last explicit match can disagree with the recent-10 share."""
    radiant = _side_state(
        match_id=M2,
        side="RADIANT",
        players=RADIANT,
        observed=("POSITION_5", "POSITION_2", "POSITION_3", "POSITION_4", "POSITION_1"),
        overrides={
            P1: {
                "previous_explicit_position": "POSITION_5",
                "prior_explicit_position_games": 10,
                "prior_share_position_1": 0.9,
                "prior_share_position_5": 0.1,
                "recent_10_explicit_games": 10,
                "recent_10_games_position_1": 9,
                "recent_10_games_position_5": 1,
            },
            P2: {
                "previous_explicit_position": "POSITION_2",
                "prior_explicit_position_games": 10,
                "prior_share_position_2": 1.0,
                "recent_10_explicit_games": 10,
                "recent_10_games_position_2": 10,
            },
            P3: {
                "previous_explicit_position": "POSITION_3",
                "prior_explicit_position_games": 10,
                "prior_share_position_3": 1.0,
                "recent_10_explicit_games": 10,
                "recent_10_games_position_3": 10,
            },
            P4: {
                "previous_explicit_position": "POSITION_4",
                "prior_explicit_position_games": 10,
                "prior_share_position_4": 1.0,
                "recent_10_explicit_games": 10,
                "recent_10_games_position_4": 10,
            },
            P5: {
                "previous_explicit_position": "POSITION_1",
                "prior_explicit_position_games": 10,
                "prior_share_position_1": 0.1,
                "prior_share_position_5": 0.9,
                "recent_10_explicit_games": 10,
                "recent_10_games_position_5": 10,
            },
        },
    )
    dire = _side_state(
        match_id=M2,
        side="DIRE",
        players=DIRE,
        observed=tuple(EXPLICIT_POSITION_LABELS),
    )
    state = pd.concat([radiant, dire], ignore_index=True)
    previous = assign_expected_positions(state, method="previous")
    hierarchical = assign_expected_positions(state, method="hierarchical")
    assert _row(previous, M2, P1)["expected_position"] == "POSITION_5"
    assert _row(hierarchical, M2, P1)["expected_position"] == "POSITION_1"
    assert _row(hierarchical, M2, P1)["evidence_tier"] == "recent_10"
    assert _row(hierarchical, M2, P5)["expected_position"] == "POSITION_5"


def test_naive_uniqueness_detects_duplicates_and_missing() -> None:
    from dota_predictor.features.expected_position import audit_naive_roster_uniqueness

    unique = _side_state(
        match_id=M1,
        side="RADIANT",
        players=RADIANT,
        observed=tuple(EXPLICIT_POSITION_LABELS),
        overrides={
            player_id: {
                "previous_explicit_position": label,
                "historical_modal_position": label,
                "recent_5_modal_position": label,
                "recent_10_modal_position": label,
                "recent_20_modal_position": label,
            }
            for player_id, label in zip(RADIANT, EXPLICIT_POSITION_LABELS, strict=True)
        },
    )
    collided = _side_state(
        match_id=M2,
        side="RADIANT",
        players=RADIANT,
        observed=tuple(EXPLICIT_POSITION_LABELS),
        overrides={
            P1: {"previous_explicit_position": "POSITION_1"},
            P2: {"previous_explicit_position": "POSITION_1"},
            P3: {"previous_explicit_position": "POSITION_2"},
            P4: {"previous_explicit_position": "POSITION_3"},
            P5: {"previous_explicit_position": None},
        },
    )
    audit = audit_naive_roster_uniqueness(pd.concat([unique, collided], ignore_index=True))
    unique_row = audit[(audit["match_id"] == M1)].iloc[0]
    collided_row = audit[(audit["match_id"] == M2)].iloc[0]
    assert unique_row["previous_explicit_position"] == "unique_1_to_5"
    assert collided_row["previous_explicit_position"] == "missing_and_duplicate"


def test_evaluate_excludes_null_observed_from_denominators() -> None:
    from dota_predictor.features.expected_position import evaluate_expected_position

    assigned = pd.DataFrame(
        [
            {
                "match_id": M1,
                "player_id": player_id,
                "side": "RADIANT",
                "expected_position": expected,
                "observed_position": observed,
                "previous_explicit_position": expected,
            }
            for player_id, expected, observed in zip(
                RADIANT,
                EXPLICIT_POSITION_LABELS,
                ("POSITION_1", "POSITION_2", "POSITION_3", "POSITION_4", None),
                strict=True,
            )
        ]
    )
    report = evaluate_expected_position(assigned)
    assert report["n_eligible_players"] == 4
    assert report["n_eligible_sides"] == 0
    assert report["player_accuracy"] == pytest.approx(1.0)


def test_same_version_scores_ignore_current_observed_position() -> None:
    row = pd.DataFrame(
        [
            {
                "previous_explicit_position": "POSITION_1",
                "position": "POSITION_5",
                "version_prior_games_position_1": 0,
                "version_prior_games_position_2": 0,
                "version_prior_games_position_3": 0,
                "version_prior_games_position_4": 0,
                "version_prior_games_position_5": 4,
            }
        ]
    )
    scores = player_position_scores(row, method="same_version")
    assert scores[0, 4] == pytest.approx(1.0)
    assert scores[0, 0] == pytest.approx(0.0)
