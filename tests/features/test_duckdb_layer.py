"""Tests for the Step 3A DuckDB analytical layer.

All tests build their canonical Parquet fixture in-process via
`conftest.feature_store_config` and never touch PostgreSQL -- see
`test_no_database_connection_required` for an explicit structural check
of that property.
"""

from __future__ import annotations

import pytest
from conftest import (
    EARLY_START_TIME,
    LATE_START_TIME,
    MATCH_EARLY_DIRE_PLAYER_IDS,
    MATCH_EARLY_DIRE_TEAM_ID,
    MATCH_EARLY_ID,
    MATCH_EARLY_NUM_BANS,
    MATCH_EARLY_RADIANT_PLAYER_IDS,
    MATCH_EARLY_RADIANT_TEAM_ID,
    MATCH_LATE_DIRE_PLAYER_IDS,
    MATCH_LATE_DIRE_TEAM_ID,
    MATCH_LATE_ID,
    MATCH_LATE_NUM_BANS,
    MATCH_LATE_RADIANT_PLAYER_IDS,
    MATCH_LATE_RADIANT_TEAM_ID,
)

from dota_predictor.features.config import FeatureStoreConfig
from dota_predictor.features.duckdb_layer import (
    DRAFT_EVENTS_VIEW,
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    connect,
)
from dota_predictor.features.temporal import HISTORICAL_START_TIME_SQL_CONDITION

# --- real Parquet shape -----------------------------------------------


def test_matches_view_reflects_real_parquet_shape(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        relation = store.relation(MATCHES_VIEW)
        assert relation.count("*").fetchone()[0] == 2
        column_names = set(relation.columns)

    assert {
        "match_id",
        "start_time",
        "radiant_team_id",
        "dire_team_id",
        "radiant_player_0_id",
        "dire_player_4_id",
        "radiant_win",
        "duration_seconds",
        "mapper_version",
        "canonicalized_at",
    }.issubset(column_names)


def test_draft_events_view_reflects_real_parquet_shape(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        relation = store.relation(DRAFT_EVENTS_VIEW)
        assert relation.count("*").fetchone()[0] == (
            MATCH_EARLY_NUM_BANS + 10 + MATCH_LATE_NUM_BANS + 10
        )
        assert set(relation.columns) == {
            "match_id",
            "sequence",
            "action",
            "side",
            "hero_id",
            "was_successful",
        }


def test_zero_ban_draft_is_supported(feature_store_config: FeatureStoreConfig) -> None:
    """The real, observed 10-event all-pick/zero-ban draft shape must
    survive the DuckDB layer untouched -- no padding, no rejection."""
    with connect(feature_store_config) as store:
        rows = store.sql(
            f"SELECT action FROM {DRAFT_EVENTS_VIEW} WHERE match_id = {MATCH_LATE_ID}"
        ).fetchall()

    actions = [row[0] for row in rows]
    assert len(actions) == 10
    assert actions.count("BAN") == 0
    assert actions.count("PICK") == 10


# --- match_players reconstruction --------------------------------------


def test_match_players_reconstructs_all_ten_players_exactly_once_per_match(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        rows = store.relation(MATCH_PLAYERS_VIEW).fetchall()
        columns = store.relation(MATCH_PLAYERS_VIEW).columns

    assert columns == [
        "match_id",
        "start_time",
        "side",
        "slot_in_side",
        "player_id",
        "team_id",
    ]
    assert len(rows) == 20  # 2 matches * 10 players

    by_match: dict[int, list[tuple]] = {}
    for row in rows:
        by_match.setdefault(row[0], []).append(row)

    assert set(by_match) == {MATCH_EARLY_ID, MATCH_LATE_ID}
    for match_id, match_rows in by_match.items():
        assert len(match_rows) == 10
        player_ids = [row[4] for row in match_rows]
        assert len(set(player_ids)) == 10, f"match {match_id}: duplicate player_id"


def test_match_players_side_team_and_slot_are_correct(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        rows = store.sql(
            f"""
            SELECT match_id, side, slot_in_side, player_id, team_id
            FROM {MATCH_PLAYERS_VIEW}
            ORDER BY match_id, side, slot_in_side
            """
        ).fetchall()

    expected = []
    for slot, player_id in enumerate(MATCH_LATE_RADIANT_PLAYER_IDS):
        expected.append(
            (MATCH_LATE_ID, "RADIANT", slot, player_id, MATCH_LATE_RADIANT_TEAM_ID)
        )
    for slot, player_id in enumerate(MATCH_LATE_DIRE_PLAYER_IDS):
        expected.append(
            (MATCH_LATE_ID, "DIRE", slot, player_id, MATCH_LATE_DIRE_TEAM_ID)
        )
    for slot, player_id in enumerate(MATCH_EARLY_RADIANT_PLAYER_IDS):
        expected.append(
            (MATCH_EARLY_ID, "RADIANT", slot, player_id, MATCH_EARLY_RADIANT_TEAM_ID)
        )
    for slot, player_id in enumerate(MATCH_EARLY_DIRE_PLAYER_IDS):
        expected.append(
            (MATCH_EARLY_ID, "DIRE", slot, player_id, MATCH_EARLY_DIRE_TEAM_ID)
        )

    # `rows`/`expected` are both ordered by (match_id, side, slot_in_side);
    # match_id sorts MATCH_LATE_ID (1001) before MATCH_EARLY_ID (2002)
    # numerically, independent of start_time -- see conftest docstring.
    assert sorted(rows) == sorted(expected)
    assert rows == sorted(expected)


def test_match_players_preserves_start_time_per_match(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        rows = store.sql(
            f"SELECT DISTINCT match_id, start_time FROM {MATCH_PLAYERS_VIEW}"
        ).fetchall()

    start_times = dict(rows)
    assert start_times[MATCH_EARLY_ID] == EARLY_START_TIME
    assert start_times[MATCH_LATE_ID] == LATE_START_TIME


# --- chronological ordering uses start_time, not match_id --------------


def test_chronological_ordering_uses_start_time_not_match_id(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        by_match_id = store.sql(
            f"SELECT match_id FROM {MATCHES_VIEW} ORDER BY match_id"
        ).fetchall()
        by_start_time = store.sql(
            f"SELECT match_id FROM {MATCHES_VIEW} ORDER BY start_time"
        ).fetchall()

    # The fixture deliberately assigns match_id and start_time in opposite
    # orders, so these two orderings must differ -- proving a query that
    # ordered by match_id instead of start_time would be wrong.
    assert by_match_id != by_start_time
    assert [row[0] for row in by_start_time] == [MATCH_EARLY_ID, MATCH_LATE_ID]
    assert [row[0] for row in by_match_id] == [MATCH_LATE_ID, MATCH_EARLY_ID]


# --- historical eligibility (start_time, strict <) ----------------------


def test_historical_eligibility_sql_condition_excludes_future_and_ties(
    feature_store_config: FeatureStoreConfig,
) -> None:
    condition = HISTORICAL_START_TIME_SQL_CONDITION.format(historical="h", current="c")

    with connect(feature_store_config) as store:
        pairs = store.sql(
            f"""
            SELECT h.match_id AS historical_match_id, c.match_id AS current_match_id
            FROM {MATCHES_VIEW} AS h, {MATCHES_VIEW} AS c
            WHERE {condition}
            """
        ).fetchall()

    # Only the earlier match is historical relative to the later one; the
    # reverse pairing and both same-match (tied start_time) pairings are
    # excluded.
    assert pairs == [(MATCH_EARLY_ID, MATCH_LATE_ID)]


# --- no PostgreSQL required ---------------------------------------------


def test_no_database_connection_required(
    feature_store_config: FeatureStoreConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The feature layer must work with no PostgreSQL connection info in
    the environment at all."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

    with connect(feature_store_config) as store:
        assert store.relation(MATCHES_VIEW).count("*").fetchone()[0] == 2


def test_feature_layer_modules_do_not_import_postgres_stack() -> None:
    """Structural guard: the Step 3A modules must never come to depend on
    SQLAlchemy/psycopg, even transitively through a future edit.

    Parses each module's actual `import`/`from ... import` statements
    (not a raw substring search over the source text, which would also
    flag this test's own explanatory prose about *not* requiring
    PostgreSQL) and asserts none of the imported top-level module names
    belong to the Postgres driver stack.
    """
    import ast
    import inspect

    from dota_predictor.features import availability, config, duckdb_layer, temporal

    forbidden_top_level_modules = {"sqlalchemy", "psycopg", "psycopg2"}

    for module in (config, duckdb_layer, availability, temporal):
        source = inspect.getsource(module)
        tree = ast.parse(source, filename=module.__name__)
        imported_top_level_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_top_level_modules.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_top_level_modules.add(node.module.split(".")[0])

        offenders = imported_top_level_modules & forbidden_top_level_modules
        assert not offenders, f"{module.__name__} imports {offenders}"
