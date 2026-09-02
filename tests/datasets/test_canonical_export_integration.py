"""Integration tests for `build_canonical_dataset` against real Postgres.

Follows the project's test-isolation rule: DB-touching tests use
`TEST_DATABASE_URL` only (via the `engine` fixture / `get_test_engine`),
never `DATABASE_URL`, and are skipped -- not redirected -- when
`TEST_DATABASE_URL` is unset.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from helpers import build_canonical_match, requires_test_database, seed_ingestion_league

from dota_predictor.datasets.canonical_export import (
    ANALYTICAL_SCHEMA_VERSION,
    DRAFT_EVENTS_FILENAME,
    MATCH_PLAYERS_FILENAME,
    MATCHES_FILENAME,
)
from dota_predictor.datasets.canonical_export import (
    build_canonical_dataset as build_dataset,
)
from dota_predictor.storage.schema import STRATZ_RAW_MATCHES
from dota_predictor.storage.writer import write_canonical_match

pytestmark = requires_test_database


def test_build_canonical_dataset_end_to_end(engine, tmp_path: Path) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=1)

    # Three canonical matches, including one with the real observed
    # zero-ban/10-event draft shape.
    match_a = build_canonical_match(match_id=1001, league_id=1, num_bans=6)
    match_b = build_canonical_match(
        match_id=1002,
        league_id=1,
        num_bans=0,
        radiant_player_ids=(21, 22, 23, 24, 25),
        dire_player_ids=(26, 27, 28, 29, 30),
    )
    match_c = build_canonical_match(match_id=1003, league_id=1, num_bans=14)
    for match in (match_a, match_b, match_c):
        write_canonical_match(engine, match)

    # A raw payload with no canonical row -- simulates a match that failed
    # canonicalization (see `ingestion.pipeline`). Must NOT appear in the
    # export: this module only ever reads `matches`/`match_players`/
    # `draft_events`, never `stratz_raw_matches`.
    with engine.begin() as conn:
        conn.execute(
            STRATZ_RAW_MATCHES.insert().values(
                match_id=9999,
                league_id=1,
                payload={"id": 9999, "note": "failed canonicalization"},
                fetched_at=datetime.now(UTC),
            )
        )

    result = build_dataset(engine, tmp_path)

    assert result.matches_row_count == 3
    assert result.match_players_row_count == 30
    assert result.draft_events_row_count == 16 + 10 + 24  # 6+10, 0+10, 14+10
    assert result.output_dir == tmp_path
    assert result.schema_version == ANALYTICAL_SCHEMA_VERSION

    matches_table = pq.read_table(tmp_path / MATCHES_FILENAME)
    match_players_table = pq.read_table(tmp_path / MATCH_PLAYERS_FILENAME)
    draft_events_table = pq.read_table(tmp_path / DRAFT_EVENTS_FILENAME)

    exported_match_ids = matches_table.column("match_id").to_pylist()
    assert sorted(exported_match_ids) == [1001, 1002, 1003]
    assert 9999 not in exported_match_ids  # raw-only match excluded
    assert "radiant_player_0_hero_id" not in matches_table.column_names
    assert match_players_table.num_rows == 30
    assert set(match_players_table.column("match_id").to_pylist()) == {1001, 1002, 1003}
    match_b_players = [
        row
        for row in match_players_table.to_pylist()
        if row["match_id"] == 1002
    ]
    assert len(match_b_players) == 10
    assert {row["player_id"] for row in match_b_players if row["side"] == "RADIANT"} == {
        21,
        22,
        23,
        24,
        25,
    }
    assert all(row["hero_id"] is not None for row in match_b_players)
    assert all(row["position"] is None for row in match_b_players)
    assert "position" in match_players_table.column_names
    assert "lane" in match_players_table.column_names
    assert "role" in match_players_table.column_names
    assert "kills" in match_players_table.column_names
    assert "gold_per_minute" in match_players_table.column_names
    assert all(row["kills"] is None for row in match_b_players)
    assert len({row["hero_id"] for row in match_b_players if row["side"] == "RADIANT"}) == 5

    rows_by_id = {row["match_id"]: row for row in matches_table.to_pylist()}
    match_b_row = rows_by_id[1002]
    assert [match_b_row[f"radiant_player_{i}_id"] for i in range(5)] == [
        21,
        22,
        23,
        24,
        25,
    ]
    assert [match_b_row[f"dire_player_{i}_id"] for i in range(5)] == [
        26,
        27,
        28,
        29,
        30,
    ]

    draft_counts: dict[int, int] = {}
    for match_id in draft_events_table.column("match_id").to_pylist():
        draft_counts[match_id] = draft_counts.get(match_id, 0) + 1
    assert draft_counts == {1001: 16, 1002: 10, 1003: 24}

    # match 1002's draft has zero BAN rows -- confirms the all-pick shape
    # survived the export rather than being padded or rejected.
    match_1002_actions = [
        action
        for match_id, action in zip(
            draft_events_table.column("match_id").to_pylist(),
            draft_events_table.column("action").to_pylist(),
            strict=True,
        )
        if match_id == 1002
    ]
    assert match_1002_actions.count("BAN") == 0
    assert match_1002_actions.count("PICK") == 10


def test_build_canonical_dataset_row_count_matches_postgres(
    engine, tmp_path: Path
) -> None:
    with engine.begin() as conn:
        seed_ingestion_league(conn, league_id=2)

    for i in range(5):
        write_canonical_match(
            engine,
            build_canonical_match(match_id=2000 + i, league_id=2, num_bans=4 + i),
        )

    from sqlalchemy import func, select

    from dota_predictor.storage.schema import MATCHES

    with engine.connect() as conn:
        postgres_count = conn.execute(
            select(func.count()).select_from(MATCHES)
        ).scalar_one()

    result = build_dataset(engine, tmp_path)
    assert result.matches_row_count == postgres_count == 5
