"""Tests for the pure player-identity derivation logic (Slice 2).

These are database-free tests of `dota_predictor.data.player_identity`:
one canonical player per valid source `player_id`, deterministic
first/last-seen + match-count derivation, and the deterministic
display-name resolution rule (most recently observed valid name, with a
deterministic tie-break) that applies once name observations exist in
source data. DB-touching behavior (registry sync, universe query, audit)
is covered separately in `tests/storage/test_player_identity_storage.py`
and `tests/research/test_player_identity_views.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dota_predictor.data.player_identity import (
    PlayerName,
    derive_player_names,
    derive_player_summaries,
    resolve_display_name,
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)


def _names(*entries) -> list[PlayerName]:
    return [
        entry if isinstance(entry, PlayerName) else PlayerName(**entry)
        for entry in entries
    ]


# --- derive_player_summaries -------------------------------------------------


def test_one_summary_row_per_player():
    summaries = derive_player_summaries([(10, T1), (20, T2), (10, T3), (10, T1)])
    assert len(summaries) == 2
    by_id = {s.player_id: s for s in summaries}
    assert by_id[10].first_seen_at == T1
    assert by_id[10].last_seen_at == T3
    assert by_id[10].match_count == 3
    assert by_id[20].first_seen_at == T2
    assert by_id[20].last_seen_at == T2
    assert by_id[20].match_count == 1


def test_summaries_are_deterministic_regardless_of_input_order():
    observations = [(10, T1), (20, T2), (10, T3)]
    assert derive_player_summaries(observations) == derive_player_summaries(
        list(reversed(observations))
    )


def test_summaries_handle_empty():
    assert derive_player_summaries([]) == []


# --- derive_player_names ------------------------------------------------------


def test_names_group_by_player_id_and_name_with_periods():
    names = derive_player_names(
        [
            (10, "player_a", T1),
            (10, "player_a", T3),
            (10, None, T2),
            (20, "player_a", T2),
        ]
    )
    by_key = {(n.player_id, n.name): n for n in names}
    assert len(names) == 2
    assert by_key[(10, "player_a")].observation_count == 2
    assert by_key[(10, "player_a")].first_seen_at == T1
    assert by_key[(10, "player_a")].last_seen_at == T3
    assert by_key[(20, "player_a")].observation_count == 1


def test_names_ignore_missing_names():
    assert derive_player_names([(10, None, T1)]) == []
    assert derive_player_names([]) == []


def test_nickname_change_preserves_name_history():
    names = derive_player_names([(10, "Old Name", T1), (10, "New Name", T3)])
    assert {(n.name, n.observation_count) for n in names} == {
        ("Old Name", 1),
        ("New Name", 1),
    }


# --- resolve_display_name -----------------------------------------------------


def test_display_name_is_most_recently_observed_valid_name():
    names = _names(
        {
            "player_id": 10,
            "name": "Old Name",
            "first_seen_at": T1,
            "last_seen_at": T1,
            "observation_count": 3,
        },
        {
            "player_id": 10,
            "name": "New Name",
            "first_seen_at": T2,
            "last_seen_at": T3,
            "observation_count": 2,
        },
    )
    assert resolve_display_name(names) == "New Name"


def test_display_name_survives_nickname_changes_without_creating_players():
    """A player's identity is the player_id, not the name: a nickname change
    resolves to the latest name for the same single identity."""
    names = _names(
        {
            "player_id": 10,
            "name": "Old",
            "first_seen_at": T1,
            "last_seen_at": T1,
            "observation_count": 1,
        },
        {
            "player_id": 10,
            "name": "New",
            "first_seen_at": T2,
            "last_seen_at": T2,
            "observation_count": 1,
        },
    )
    assert resolve_display_name(names) == "New"


def test_identical_names_on_different_player_ids_remain_distinct():
    """Same display name for two player ids must never merge the identities."""
    names = _names(
        {
            "player_id": 10,
            "name": "player_a",
            "first_seen_at": T1,
            "last_seen_at": T3,
            "observation_count": 2,
        },
        {
            "player_id": 20,
            "name": "player_a",
            "first_seen_at": T2,
            "last_seen_at": T2,
            "observation_count": 1,
        },
    )
    by_id = {n.player_id: resolve_display_name([n]) for n in names}
    assert by_id == {10: "player_a", 20: "player_a"}


def test_display_name_resolution_handles_capitalization_change():
    names = _names(
        {
            "player_id": 10,
            "name": "player_a",
            "first_seen_at": T1,
            "last_seen_at": T1,
            "observation_count": 1,
        },
        {
            "player_id": 10,
            "name": "PLAYER_A",
            "first_seen_at": T2,
            "last_seen_at": T2,
            "observation_count": 1,
        },
    )
    assert resolve_display_name(names) == "PLAYER_A"


def test_display_name_returns_none_for_no_valid_names():
    assert resolve_display_name([]) is None


def test_display_name_tie_break_is_deterministic():
    """Two names first observed together at the same instant resolve to the
    lexicographically smallest name, independent of input order."""
    names_a = _names(
        {
            "player_id": 10,
            "name": "zeta",
            "first_seen_at": T1,
            "last_seen_at": T1,
            "observation_count": 1,
        },
        {
            "player_id": 10,
            "name": "alpha",
            "first_seen_at": T1,
            "last_seen_at": T1,
            "observation_count": 1,
        },
    )
    names_b = _names(*list(reversed(names_a)))
    assert resolve_display_name(names_a) == "alpha"
    assert resolve_display_name(names_b) == "alpha"
