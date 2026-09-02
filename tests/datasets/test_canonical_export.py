"""Pure, in-memory tests for the canonical Postgres -> Parquet transforms.

These do not touch Postgres: they feed synthetic relational rows (plain
dicts, mirroring the shape of SQLAlchemy `RowMapping`s) directly into the
transform/validation functions.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dota_predictor.datasets.canonical_export import (
    DRAFT_EVENTS_FILENAME,
    MATCH_PLAYERS_FILENAME,
    MATCHES_FILENAME,
    DatasetTransformError,
    DatasetValidationError,
    build_draft_events_table,
    build_match_players_table,
    build_matches_table,
    validate_draft_events_table,
    validate_match_players_table,
    validate_matches_table,
    write_canonical_dataset,
)

START_TIME = datetime(2025, 1, 1, tzinfo=UTC)


def _match_row(
    match_id: int,
    *,
    radiant_team_id: int = 100,
    dire_team_id: int = 200,
    game_number_in_series: int | None = None,
) -> dict:
    return {
        "match_id": match_id,
        "league_id": 1,
        "start_time": START_TIME,
        "league_name": "Test League",
        "series_id": 10,
        "series_type": "BEST_OF_THREE",
        "game_number_in_series": game_number_in_series,
        "game_version_id": 176,
        "radiant_team_id": radiant_team_id,
        "radiant_team_name_observed": "Radiant Team",
        "dire_team_id": dire_team_id,
        "dire_team_name_observed": "Dire Team",
        "radiant_win": True,
        "duration_seconds": 1800,
        "mapper_version": 1,
        "canonicalized_at": START_TIME,
    }


def _player_rows(
    match_id: int,
    *,
    radiant_ids: list[int],
    dire_ids: list[int],
    radiant_heroes: list[int] | None = None,
    dire_heroes: list[int] | None = None,
) -> list[dict]:
    if radiant_heroes is None:
        radiant_heroes = [10 + i for i in range(5)]
    if dire_heroes is None:
        dire_heroes = [20 + i for i in range(5)]
    rows = []
    for slot, player_id in enumerate(radiant_ids):
        rows.append(
            {
                "match_id": match_id,
                "side": "RADIANT",
                "slot_in_side": slot,
                "player_id": player_id,
                "hero_id": radiant_heroes[slot],
            }
        )
    for slot, player_id in enumerate(dire_ids):
        rows.append(
            {
                "match_id": match_id,
                "side": "DIRE",
                "slot_in_side": slot,
                "player_id": player_id,
                "hero_id": dire_heroes[slot],
            }
        )
    return rows


def _pick_heroes_for_draft(num_bans: int) -> tuple[list[int], list[int]]:
    """Hero ids assigned to the 5 radiant then 5 dire picks in `_draft_rows`."""
    start = 1000 + num_bans
    return list(range(start, start + 5)), list(range(start + 5, start + 10))


def _draft_rows(match_id: int, *, num_bans: int) -> list[dict]:
    """`num_bans` alternating-side bans, then 5 radiant picks, 5 dire picks."""
    rows = []
    sequence = 0
    for i in range(num_bans):
        rows.append(
            {
                "match_id": match_id,
                "sequence": sequence,
                "action": "BAN",
                "side": "RADIANT" if i % 2 == 0 else "DIRE",
                "hero_id": 1000 + sequence,
                "was_successful": True,
            }
        )
        sequence += 1
    for side in ("RADIANT", "DIRE"):
        for _ in range(5):
            rows.append(
                {
                    "match_id": match_id,
                    "sequence": sequence,
                    "action": "PICK",
                    "side": side,
                    "hero_id": 1000 + sequence,
                    "was_successful": None,
                }
            )
            sequence += 1
    return rows


# --- match/player pivot -----------------------------------------------


def test_pivot_preserves_slot_correspondence_not_query_order() -> None:
    radiant_ids = [111, 222, 333, 444, 555]
    dire_ids = [666, 777, 888, 999, 1010]
    match_rows = [_match_row(1)]
    player_rows = _player_rows(1, radiant_ids=radiant_ids, dire_ids=dire_ids)
    # Shuffle input row order to prove slot correspondence comes from
    # `slot_in_side`, not from the order rows happen to arrive in.
    random.Random(42).shuffle(player_rows)

    table = build_matches_table(match_rows, player_rows)
    row = table.to_pylist()[0]

    for slot, expected in enumerate(radiant_ids):
        assert row[f"radiant_player_{slot}_id"] == expected
    for slot, expected in enumerate(dire_ids):
        assert row[f"dire_player_{slot}_id"] == expected


def test_pivot_raises_on_missing_player_slot() -> None:
    match_rows = [_match_row(1)]
    player_rows = _player_rows(
        1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    # Drop the dire slot-3 row entirely.
    player_rows = [
        r for r in player_rows if not (r["side"] == "DIRE" and r["slot_in_side"] == 3)
    ]

    with pytest.raises(DatasetTransformError, match="missing DIRE player slot"):
        build_matches_table(match_rows, player_rows)


def test_pivot_raises_on_duplicate_player_slot() -> None:
    match_rows = [_match_row(1)]
    player_rows = _player_rows(
        1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    # A second, conflicting row for the same (match, side, slot).
    player_rows.append(
        {"match_id": 1, "side": "RADIANT", "slot_in_side": 0, "player_id": 999}
    )

    with pytest.raises(DatasetTransformError, match="duplicate RADIANT slot 0"):
        build_matches_table(match_rows, player_rows)


def test_build_matches_table_raises_when_match_has_no_players() -> None:
    match_rows = [_match_row(1), _match_row(2)]
    # Only match 1 has match_players rows.
    player_rows = _player_rows(
        1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )

    with pytest.raises(DatasetTransformError, match="no match_players rows found"):
        build_matches_table(match_rows, player_rows)


def test_matches_table_preserves_game_number_in_series_verbatim() -> None:
    """This export does not derive `game_number_in_series` -- it passes the
    canonical value through unchanged, including `None`."""
    match_rows = [_match_row(1, game_number_in_series=None)]
    player_rows = _player_rows(
        1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )

    table = build_matches_table(match_rows, player_rows)
    assert table.to_pylist()[0]["game_number_in_series"] is None


# --- determinism ---------------------------------------------------------


def test_build_matches_table_orders_by_match_id_regardless_of_input_order() -> None:
    match_rows = [_match_row(3), _match_row(1), _match_row(2)]
    player_rows = [
        row
        for match_id in (1, 2, 3)
        for row in _player_rows(
            match_id,
            radiant_ids=[match_id * 10 + i for i in range(5)],
            dire_ids=[match_id * 100 + i for i in range(5)],
        )
    ]

    table = build_matches_table(match_rows, player_rows)
    assert table.column("match_id").to_pylist() == [1, 2, 3]


def test_build_draft_events_table_orders_by_match_id_then_sequence() -> None:
    rows = _draft_rows(2, num_bans=4) + _draft_rows(1, num_bans=6)
    random.Random(7).shuffle(rows)

    table = build_draft_events_table(rows)

    pairs = list(
        zip(table.column("match_id").to_pylist(), table.column("sequence").to_pylist())
    )
    assert pairs == sorted(pairs)
    assert pairs[0][0] == 1  # match 1 sorts first


# --- variable draft length ------------------------------------------------


@pytest.mark.parametrize("num_bans", [0, 6, 14])
def test_build_draft_events_table_preserves_variable_length(num_bans: int) -> None:
    """`num_bans=0` is the real, observed all-pick/zero-ban 10-event shape."""
    rows = _draft_rows(1, num_bans=num_bans)
    table = build_draft_events_table(rows)
    assert table.num_rows == num_bans + 10

    actions = table.column("action").to_pylist()
    assert actions.count("BAN") == num_bans
    assert actions.count("PICK") == 10


def test_build_draft_events_table_does_not_assume_24_events() -> None:
    """Mixing a 10-event and a 24-event draft in one build must not raise
    or silently coerce either to the other's length."""
    rows = _draft_rows(1, num_bans=0) + _draft_rows(2, num_bans=14)
    table = build_draft_events_table(rows)

    counts = {1: 0, 2: 0}
    for match_id in table.column("match_id").to_pylist():
        counts[match_id] += 1
    assert counts[1] == 10
    assert counts[2] == 24


# --- validation ------------------------------------------------------------


def test_validate_matches_table_detects_duplicate_match_id() -> None:
    match_rows = [_match_row(1, radiant_team_id=100, dire_team_id=200), _match_row(1)]
    player_rows = _player_rows(
        1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )

    table = build_matches_table(match_rows, player_rows)
    with pytest.raises(DatasetValidationError, match="duplicate match_id"):
        validate_matches_table(table, expected_row_count=table.num_rows)


def test_validate_matches_table_detects_row_count_mismatch() -> None:
    match_rows = [_match_row(1)]
    player_rows = _player_rows(
        1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    table = build_matches_table(match_rows, player_rows)

    with pytest.raises(DatasetValidationError, match="row count"):
        validate_matches_table(table, expected_row_count=table.num_rows + 1)


def test_validate_matches_table_detects_null_player_id() -> None:
    # Constructed directly (bypassing build_matches_table, which structurally
    # forbids this) to exercise the validation function itself.
    table = pa.table(
        {
            "match_id": [1],
            "radiant_player_0_id": [None],
            "radiant_player_1_id": [2],
            "radiant_player_2_id": [3],
            "radiant_player_3_id": [4],
            "radiant_player_4_id": [5],
            "dire_player_0_id": [6],
            "dire_player_1_id": [7],
            "dire_player_2_id": [8],
            "dire_player_3_id": [9],
            "dire_player_4_id": [10],
            "radiant_team_id": [100],
            "dire_team_id": [200],
        }
    )
    with pytest.raises(DatasetValidationError, match="null player id"):
        validate_matches_table(table, expected_row_count=1)


def test_validate_draft_events_table_detects_duplicate_sequence() -> None:
    matches_table = build_matches_table(
        [_match_row(1)],
        _player_rows(1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]),
    )
    draft_table = pa.table(
        {
            "match_id": [1, 1],
            "sequence": [0, 0],
            "action": ["BAN", "PICK"],
            "side": ["RADIANT", "DIRE"],
            "hero_id": [1, 2],
            "was_successful": [True, None],
        }
    )
    with pytest.raises(DatasetValidationError, match="duplicate"):
        validate_draft_events_table(draft_table, matches_table)


def test_validate_draft_events_table_detects_orphan_match() -> None:
    matches_table = build_matches_table(
        [_match_row(1)],
        _player_rows(1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]),
    )
    draft_table = build_draft_events_table(_draft_rows(2, num_bans=4))

    with pytest.raises(DatasetValidationError, match="absent from matches.parquet"):
        validate_draft_events_table(draft_table, matches_table)


def test_validate_draft_events_table_allows_variable_length_without_error() -> None:
    """Sanity check: validation itself never rejects a non-24-event draft."""
    matches_table = build_matches_table(
        [_match_row(1)],
        _player_rows(1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]),
    )
    draft_table = build_draft_events_table(_draft_rows(1, num_bans=0))
    validate_draft_events_table(draft_table, matches_table)  # must not raise


# --- Parquet round-trip ----------------------------------------------------


def test_write_canonical_dataset_round_trip(tmp_path: Path) -> None:
    match_rows = [_match_row(1), _match_row(2)]
    player_rows = _player_rows(
        1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    ) + _player_rows(2, radiant_ids=[11, 12, 13, 14, 15], dire_ids=[16, 17, 18, 19, 20])
    draft_rows = _draft_rows(1, num_bans=0) + _draft_rows(2, num_bans=14)

    matches_table = build_matches_table(match_rows, player_rows)
    draft_events_table = build_draft_events_table(draft_rows)

    write_canonical_dataset(
        tmp_path, matches_table=matches_table, draft_events_table=draft_events_table
    )

    written_files = sorted(p.name for p in tmp_path.iterdir())
    assert written_files == sorted([MATCHES_FILENAME, DRAFT_EVENTS_FILENAME])

    read_matches = pq.read_table(tmp_path / MATCHES_FILENAME)
    read_drafts = pq.read_table(tmp_path / DRAFT_EVENTS_FILENAME)

    assert read_matches.num_rows == 2
    assert read_matches.column("match_id").to_pylist() == [1, 2]
    assert read_matches.schema.equals(matches_table.schema)

    assert read_drafts.num_rows == 10 + 24
    assert read_drafts.schema.equals(draft_events_table.schema)

    per_match_counts: dict[int, int] = {}
    for match_id in read_drafts.column("match_id").to_pylist():
        per_match_counts[match_id] = per_match_counts.get(match_id, 0) + 1
    assert per_match_counts == {1: 10, 2: 24}


def test_write_canonical_dataset_overwrites_previous_build(tmp_path: Path) -> None:
    first_matches = build_matches_table(
        [_match_row(1)],
        _player_rows(1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]),
    )
    first_drafts = build_draft_events_table(_draft_rows(1, num_bans=4))
    write_canonical_dataset(
        tmp_path, matches_table=first_matches, draft_events_table=first_drafts
    )

    second_matches = build_matches_table(
        [_match_row(1), _match_row(2)],
        _player_rows(1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10])
        + _player_rows(
            2, radiant_ids=[11, 12, 13, 14, 15], dire_ids=[16, 17, 18, 19, 20]
        ),
    )
    second_drafts = build_draft_events_table(
        _draft_rows(1, num_bans=4) + _draft_rows(2, num_bans=0)
    )
    write_canonical_dataset(
        tmp_path, matches_table=second_matches, draft_events_table=second_drafts
    )

    read_matches = pq.read_table(tmp_path / MATCHES_FILENAME)
    assert read_matches.num_rows == 2  # fully replaced, not appended
    # No leftover temp files/directories from either build.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [MATCHES_FILENAME, DRAFT_EVENTS_FILENAME]
    )


def _aligned_players(
    match_id: int,
    *,
    num_bans: int,
    radiant_ids: list[int],
    dire_ids: list[int],
    radiant_team_id: int = 100,
    dire_team_id: int = 200,
) -> tuple[dict, list[dict], list[dict]]:
    radiant_heroes, dire_heroes = _pick_heroes_for_draft(num_bans)
    match = _match_row(
        match_id, radiant_team_id=radiant_team_id, dire_team_id=dire_team_id
    )
    players = _player_rows(
        match_id,
        radiant_ids=radiant_ids,
        dire_ids=dire_ids,
        radiant_heroes=radiant_heroes,
        dire_heroes=dire_heroes,
    )
    drafts = _draft_rows(match_id, num_bans=num_bans)
    return match, players, drafts


def test_build_match_players_table_long_form_and_team_derivation() -> None:
    match, players, _drafts = _aligned_players(
        1,
        num_bans=4,
        radiant_ids=[1, 2, 3, 4, 5],
        dire_ids=[6, 7, 8, 9, 10],
        radiant_team_id=111,
        dire_team_id=222,
    )
    random.Random(3).shuffle(players)
    table = build_match_players_table([match], players)
    assert table.num_rows == 10
    rows = table.to_pylist()
    radiant = [row for row in rows if row["side"] == "RADIANT"]
    dire = [row for row in rows if row["side"] == "DIRE"]
    assert len(radiant) == 5
    assert len(dire) == 5
    assert [row["player_id"] for row in radiant] == [1, 2, 3, 4, 5]
    assert all(row["team_id"] == 111 for row in radiant)
    assert all(row["team_id"] == 222 for row in dire)
    assert "hero_id" in rows[0]
    assert all(row["hero_id"] is not None for row in rows)


def test_match_players_parquet_preserves_null_and_unknown_position() -> None:
    match, players, drafts = _aligned_players(
        1, num_bans=4, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    players[0]["position"] = "POSITION_1"
    players[0]["lane"] = "SAFE_LANE"
    players[0]["role"] = "CORE"
    players[1]["position"] = "UNKNOWN"
    players[1]["lane"] = None
    players[1]["role"] = "LIGHT_SUPPORT"
    table = build_match_players_table([match], players)
    by_player = {row["player_id"]: row for row in table.to_pylist()}
    assert by_player[1]["position"] == "POSITION_1"
    assert by_player[1]["lane"] == "SAFE_LANE"
    assert by_player[1]["role"] == "CORE"
    assert by_player[2]["position"] == "UNKNOWN"
    assert by_player[2]["lane"] is None
    assert by_player[3]["position"] is None
    matches_table = build_matches_table([match], players)
    drafts_table = build_draft_events_table(drafts)
    validate_match_players_table(table, matches_table, drafts_table)


def test_match_players_parquet_preserves_zero_and_null_box_scores() -> None:
    match, players, drafts = _aligned_players(
        1, num_bans=4, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    players[0]["kills"] = 0
    players[0]["deaths"] = 3
    players[0]["num_last_hits"] = 0
    players[0]["hero_damage"] = None
    players[0]["level"] = 11
    table = build_match_players_table([match], players)
    by_player = {row["player_id"]: row for row in table.to_pylist()}
    assert by_player[1]["kills"] == 0
    assert by_player[1]["deaths"] == 3
    assert by_player[1]["num_last_hits"] == 0
    assert by_player[1]["hero_damage"] is None
    assert by_player[1]["level"] == 11
    assert by_player[2]["kills"] is None
    matches_table = build_matches_table([match], players)
    drafts_table = build_draft_events_table(drafts)
    validate_match_players_table(table, matches_table, drafts_table)


def test_validate_match_players_table_accepts_short_draft() -> None:
    match, players, drafts = _aligned_players(
        1, num_bans=0, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    matches_table = build_matches_table([match], players)
    players_table = build_match_players_table([match], players)
    drafts_table = build_draft_events_table(drafts)
    validate_match_players_table(players_table, matches_table, drafts_table)


def test_validate_match_players_table_requires_ten_rows() -> None:
    match, players, drafts = _aligned_players(
        1, num_bans=4, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    players = [
        row
        for row in players
        if not (row["side"] == "DIRE" and row["slot_in_side"] == 4)
    ]
    matches_table = build_matches_table(
        [match],
        _player_rows(1, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]),
    )
    players_table = build_match_players_table([match], players)
    drafts_table = build_draft_events_table(drafts)
    with pytest.raises(DatasetValidationError, match="row count"):
        validate_match_players_table(players_table, matches_table, drafts_table)


def test_validate_match_players_table_detects_duplicate_player() -> None:
    match, players, drafts = _aligned_players(
        1, num_bans=4, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    players[5]["player_id"] = players[0]["player_id"]
    matches_table = build_matches_table([match], players)
    players_table = build_match_players_table([match], players)
    drafts_table = build_draft_events_table(drafts)
    with pytest.raises(DatasetValidationError, match="duplicate player_id"):
        validate_match_players_table(players_table, matches_table, drafts_table)


def test_validate_match_players_table_detects_duplicate_hero_per_side() -> None:
    match, players, drafts = _aligned_players(
        1, num_bans=4, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    players[1]["hero_id"] = players[0]["hero_id"]
    matches_table = build_matches_table([match], players)
    players_table = build_match_players_table([match], players)
    drafts_table = build_draft_events_table(drafts)
    with pytest.raises(DatasetValidationError, match="not 5 distinct"):
        validate_match_players_table(players_table, matches_table, drafts_table)


def test_validate_match_players_table_detects_hero_set_mismatch() -> None:
    match, players, drafts = _aligned_players(
        1, num_bans=4, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    players[0]["hero_id"] = 9999
    matches_table = build_matches_table([match], players)
    players_table = build_match_players_table([match], players)
    drafts_table = build_draft_events_table(drafts)
    with pytest.raises(DatasetValidationError, match="successful PICK set"):
        validate_match_players_table(players_table, matches_table, drafts_table)


def test_validate_match_players_table_detects_wrong_team_id() -> None:
    match, players, drafts = _aligned_players(
        1,
        num_bans=4,
        radiant_ids=[1, 2, 3, 4, 5],
        dire_ids=[6, 7, 8, 9, 10],
        radiant_team_id=111,
        dire_team_id=222,
    )
    matches_table = build_matches_table([match], players)
    players_table = build_match_players_table([match], players)
    # Tamper after build: swap a radiant row onto the dire team id.
    mutated = players_table.to_pylist()
    for row in mutated:
        if row["side"] == "RADIANT" and row["slot_in_side"] == 0:
            row["team_id"] = 222
    from dota_predictor.datasets.canonical_export import MATCH_PLAYERS_SCHEMA

    tampered = pa.Table.from_pylist(mutated, schema=MATCH_PLAYERS_SCHEMA)
    drafts_table = build_draft_events_table(drafts)
    with pytest.raises(DatasetValidationError, match="team_id"):
        validate_match_players_table(tampered, matches_table, drafts_table)


def test_write_canonical_dataset_includes_match_players(tmp_path: Path) -> None:
    match, players, drafts = _aligned_players(
        1, num_bans=0, radiant_ids=[1, 2, 3, 4, 5], dire_ids=[6, 7, 8, 9, 10]
    )
    matches_table = build_matches_table([match], players)
    players_table = build_match_players_table([match], players)
    drafts_table = build_draft_events_table(drafts)
    validate_match_players_table(players_table, matches_table, drafts_table)

    write_canonical_dataset(
        tmp_path,
        matches_table=matches_table,
        draft_events_table=drafts_table,
        match_players_table=players_table,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [MATCHES_FILENAME, MATCH_PLAYERS_FILENAME, DRAFT_EVENTS_FILENAME]
    )
    read_players = pq.read_table(tmp_path / MATCH_PLAYERS_FILENAME)
    assert read_players.num_rows == 10
    assert "hero_id" in read_players.column_names
    assert "position" in read_players.column_names
    assert "lane" in read_players.column_names
    assert "role" in read_players.column_names
    assert all(row["position"] is None for row in read_players.to_pylist())
    assert "kills" in read_players.column_names
    assert "gold_per_minute" in read_players.column_names
    assert all(row["kills"] is None for row in read_players.to_pylist())
    assert (
        "radiant_player_0_hero_id"
        not in pq.read_table(tmp_path / MATCHES_FILENAME).column_names
    )
