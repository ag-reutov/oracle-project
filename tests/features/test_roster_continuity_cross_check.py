"""Cross-check: Slice 5 team-roster retained count vs the pre-draft feature.

The pre-draft snapshot (`features.pre_draft_snapshot`) has a pre-existing
roster-continuity feature: `radiant_roster_players_retained` /
`dire_roster_players_retained`, computed in DuckDB SQL with the same
strict `<` boundary (`h.start_time < c.start_time`) and the same
`ORDER BY start_time DESC, match_id DESC` tie-breaker for the most recent
prior team match.

Slice 5 (`data.roster_state.derive_team_roster_state`) is the canonical
research-state equivalent of that feature. These tests verify, on the same
synthetic canonical Parquet corpus, that:

* For every complete-five current lineup with a complete-five previous
  lineup, Slice 5's `players_retained_from_previous_match` equals the
  pre-draft feature's `radiant/dire_roster_players_retained`.
* For a team with no prior match, both are NULL/absent.
* The one deliberate edge-case difference: when a lineup is incomplete,
  the pre-draft `COUNT(*)` can return a partial number, while Slice 5
  returns NULL (undefined) and keeps the malformation explicit.

The pre-draft snapshot is a production `FEATURE_COLUMNS` set; this test
does not modify it. It documents that Slice 5 can later provide a cleaner
canonical historical-state definition that validates/replaces the
duplicated feature implementation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pre_draft_helpers import build_feature_store_config, match_row, player_rows

from dota_predictor.data.roster_state import derive_team_roster_state
from dota_predictor.features.duckdb_layer import connect
from dota_predictor.features.pre_draft_snapshot import build_pre_draft_snapshot

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)
T4 = datetime(2024, 4, 1, tzinfo=UTC)

TEAM_A, TEAM_B, TEAM_C, TEAM_D, TEAM_E = 1, 2, 3, 4, 5
M1, M2, M3, M4 = 4001, 1002, 3003, 2004

# Same corpus shape as test_pre_draft_snapshot's multi-match fixture:
# match_ids deliberately unrelated to start_time, teams swapping sides,
# player 1 moving teams, brand-new teams/players, and partial/full roster
# continuity.
_MATCHES = [
    match_row(M1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
    match_row(M2, start_time=T2, radiant_team_id=TEAM_B, dire_team_id=TEAM_A, radiant_win=True),
    match_row(M3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=False),
    match_row(M4, start_time=T4, radiant_team_id=TEAM_D, dire_team_id=TEAM_E, radiant_win=True),
]
_PLAYERS = (
    player_rows(M1, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
    + player_rows(M2, radiant_ids=(6, 7, 8, 9, 999), dire_ids=(1, 2, 3, 4, 5))
    + player_rows(M3, radiant_ids=(1, 2, 3, 4, 777), dire_ids=(11, 12, 13, 14, 15))
    + player_rows(M4, radiant_ids=(1, 20, 21, 22, 23), dire_ids=(24, 25, 26, 27, 28))
)


def _slice5_retained_by_team_match() -> dict[tuple[int, int], int | None]:
    """Slice 5 retained counts keyed by (match_id, team_id)."""
    observations = []
    for match in _MATCHES:
        for player in _PLAYERS:
            if player["match_id"] == match["match_id"]:
                team = (
                    match["radiant_team_id"]
                    if player["side"] == "RADIANT"
                    else match["dire_team_id"]
                )
                observations.append(
                    (player["player_id"], team, match["match_id"], match["start_time"])
                )
    states = derive_team_roster_state(observations)
    return {(s.match_id, s.team_id): s.players_retained_from_previous_match for s in states}


def test_slice5_retained_equals_pre_draft_roster_continuity(tmp_path: Path) -> None:
    config = build_feature_store_config(tmp_path, matches=_MATCHES, players=_PLAYERS)
    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame().set_index("match_id")

    slice5 = _slice5_retained_by_team_match()

    # For each match, the pre-draft feature names retained by the current
    # side; the current side's team is radiant_team_id / dire_team_id.
    radiant_team = {m["match_id"]: m["radiant_team_id"] for m in _MATCHES}
    dire_team = {m["match_id"]: m["dire_team_id"] for m in _MATCHES}

    for match_id, row in df.iterrows():
        for side, col, team in (
            ("RADIANT", "radiant_roster_players_retained", radiant_team[match_id]),
            ("DIRE", "dire_roster_players_retained", dire_team[match_id]),
        ):
            feature_value = row[col]
            slice5_value = slice5[(match_id, team)]

            if pd.isna(feature_value):
                # No prior team match: both are undefined (feature NULL,
                # Slice 5 None).
                assert slice5_value is None
            else:
                # Complete fives with a complete prior: identical retained
                # counts.
                assert slice5_value == int(feature_value)


def test_slice5_matches_pre_draft_on_known_multi_match_fixture(
    tmp_path: Path,
) -> None:
    config = build_feature_store_config(tmp_path, matches=_MATCHES, players=_PLAYERS)
    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame().set_index("match_id")

    slice5 = _slice5_retained_by_team_match()

    # TeamA's M2 dire roster (1,2,3,4,5) is identical to its M1 radiant
    # roster: 5/5 retained in both derivations.
    assert df.loc[M2, "dire_roster_players_retained"] == 5
    assert slice5[(M2, TEAM_A)] == 5

    # TeamB's M2 radiant roster (6,7,8,9,999) drops player 10 from its M1
    # dire roster (6,7,8,9,10): 4/5 retained in both.
    assert df.loc[M2, "radiant_roster_players_retained"] == 4
    assert slice5[(M2, TEAM_B)] == 4

    # TeamC (M3 dire) and TeamD/TeamE (M4) are brand new: NULL in both.
    assert pd.isna(df.loc[M3, "dire_roster_players_retained"])
    assert slice5[(M3, TEAM_C)] is None
    assert pd.isna(df.loc[M4, "radiant_roster_players_retained"])
    assert slice5[(M4, TEAM_D)] is None


def test_incomplete_lineup_documented_edge_case_difference() -> None:
    """Documented difference: for an incomplete current lineup the pre-draft
    feature's `COUNT(*)` semantics return a partial integer (it counts the
    current rows that matched the prior roster), while Slice 5 returns NULL
    (undefined) and keeps the malformation explicit via
    `is_complete_five = False`.

    Note: the canonical Parquet exporter (`canonical_export`) enforces
    exactly five roster slots per side, so this malformed state cannot reach
    the DuckDB pre-draft feature pipeline through the canonical export path;
    the raw Postgres research layer can still contain it, which is why Slice
    5 handles it explicitly. This test reproduces the pre-draft `COUNT(*)`
    semantics directly to document the definitional difference.
    """
    matches = [
        match_row(M1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        match_row(M2, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    players = (
        player_rows(M1, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
        + player_rows(M2, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(11, 12, 13, 14, 15))
    )
    # M2 TeamA plays with only four players (drop one radiant player row).
    players = [p for p in players if not (p["match_id"] == M2 and p["player_id"] == 5)]

    observations = []
    for match in matches:
        for player in players:
            if player["match_id"] == match["match_id"]:
                team = (
                    match["radiant_team_id"]
                    if player["side"] == "RADIANT"
                    else match["dire_team_id"]
                )
                observations.append(
                    (player["player_id"], team, match["match_id"], match["start_time"])
                )

    states = derive_team_roster_state(observations)
    team_a_m2 = next(s for s in states if s.team_id == TEAM_A and s.match_id == M2)

    # Reproduce the pre-draft COUNT(*) semantics: current TeamA roster rows
    # that also appear in TeamA's most recent prior roster.
    current_ids = {p["player_id"] for p in players if p["match_id"] == M2 and p["side"] == "RADIANT"}
    prior_ids = {p["player_id"] for p in players if p["match_id"] == M1 and p["side"] == "RADIANT"}
    pre_draft_count = len(current_ids & prior_ids)

    # Pre-draft COUNT(*) semantics: partial integer (the rows that matched).
    assert pre_draft_count == 4
    # Slice 5: undefined, malformation explicit.
    assert team_a_m2.players_retained_from_previous_match is None
    assert team_a_m2.is_complete_five is False
    assert team_a_m2.n_resolved_players == 4


def test_slice5_uses_start_time_not_match_id_for_previous_match(
    tmp_path: Path,
) -> None:
    """The most recent prior roster must be selected by start_time, not
    match_id, in both the pre-draft feature and Slice 5."""
    matches = [
        # Chronologically earliest for TeamA, but the largest match_id.
        match_row(9999, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        # Chronologically the most recent PRIOR match for TeamA, small id.
        match_row(1, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
        # Current match.
        match_row(5000, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_D, radiant_win=True),
    ]
    players = (
        player_rows(9999, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
        + player_rows(1, radiant_ids=(1, 2, 3, 4, 99), dire_ids=(11, 12, 13, 14, 15))
        + player_rows(5000, radiant_ids=(1, 2, 3, 4, 99), dire_ids=(16, 17, 18, 19, 20))
    )
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame().set_index("match_id")

    observations = []
    for match in matches:
        for player in players:
            if player["match_id"] == match["match_id"]:
                team = (
                    match["radiant_team_id"]
                    if player["side"] == "RADIANT"
                    else match["dire_team_id"]
                )
                observations.append(
                    (player["player_id"], team, match["match_id"], match["start_time"])
                )
    states = derive_team_roster_state(observations)
    slice5_by_match = {s.match_id: s for s in states if s.team_id == TEAM_A}

    # Correct "most recent prior roster" is match_id=1's roster
    # (1,2,3,4,99), identical to the current roster: 5/5 in both.
    assert df.loc[5000, "radiant_roster_players_retained"] == 5
    assert slice5_by_match[5000].players_retained_from_previous_match == 5
    assert slice5_by_match[5000].previous_match_id == 1