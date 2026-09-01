"""Tests for the Step 3A DuckDB analytical layer.

All tests build their canonical Parquet fixture in-process via
`conftest.feature_store_config` and never touch PostgreSQL -- see
`test_no_database_connection_required` for an explicit structural check
of that property.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    EARLY_START_TIME,
    LATE_START_TIME,
    MATCH_EARLY_DIRE_HERO_IDS,
    MATCH_EARLY_DIRE_PLAYER_IDS,
    MATCH_EARLY_DIRE_TEAM_ID,
    MATCH_EARLY_ID,
    MATCH_EARLY_NUM_BANS,
    MATCH_EARLY_RADIANT_HERO_IDS,
    MATCH_EARLY_RADIANT_PLAYER_IDS,
    MATCH_EARLY_RADIANT_TEAM_ID,
    MATCH_LATE_DIRE_HERO_IDS,
    MATCH_LATE_DIRE_PLAYER_IDS,
    MATCH_LATE_DIRE_TEAM_ID,
    MATCH_LATE_ID,
    MATCH_LATE_NUM_BANS,
    MATCH_LATE_RADIANT_HERO_IDS,
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


# --- match_players parquet-backed view ---------------------------------


MATCH_PLAYERS_COLUMNS = [
    "match_id",
    "start_time",
    "side",
    "slot_in_side",
    "player_id",
    "team_id",
    "hero_id",
    "position",
    "lane",
    "role",
]


def _expected_match_player_rows() -> list[tuple]:
    """Identity rows in `(match_id, side, slot_in_side)` order.

    Used so assertions compare logical keys, never physical Parquet order.
    """
    expected: list[tuple] = []
    specs = (
        (
            MATCH_LATE_ID,
            MATCH_LATE_RADIANT_PLAYER_IDS,
            MATCH_LATE_DIRE_PLAYER_IDS,
            MATCH_LATE_RADIANT_TEAM_ID,
            MATCH_LATE_DIRE_TEAM_ID,
            MATCH_LATE_RADIANT_HERO_IDS,
            MATCH_LATE_DIRE_HERO_IDS,
        ),
        (
            MATCH_EARLY_ID,
            MATCH_EARLY_RADIANT_PLAYER_IDS,
            MATCH_EARLY_DIRE_PLAYER_IDS,
            MATCH_EARLY_RADIANT_TEAM_ID,
            MATCH_EARLY_DIRE_TEAM_ID,
            MATCH_EARLY_RADIANT_HERO_IDS,
            MATCH_EARLY_DIRE_HERO_IDS,
        ),
    )
    for (
        match_id,
        radiant_players,
        dire_players,
        radiant_team,
        dire_team,
        radiant_heroes,
        dire_heroes,
    ) in specs:
        for slot, (player_id, hero_id) in enumerate(
            zip(radiant_players, radiant_heroes, strict=True)
        ):
            expected.append(
                (match_id, "RADIANT", slot, player_id, radiant_team, hero_id)
            )
        for slot, (player_id, hero_id) in enumerate(
            zip(dire_players, dire_heroes, strict=True)
        ):
            expected.append((match_id, "DIRE", slot, player_id, dire_team, hero_id))
    return sorted(expected)


def test_match_players_has_exactly_ten_columns_in_contract_order(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        columns = store.relation(MATCH_PLAYERS_VIEW).columns
    assert columns == MATCH_PLAYERS_COLUMNS


def test_match_players_has_ten_rows_per_match_five_per_side(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        rows = store.sql(
            f"""
            SELECT match_id, side, COUNT(*) AS n
            FROM {MATCH_PLAYERS_VIEW}
            GROUP BY match_id, side
            ORDER BY match_id, side
            """
        ).fetchall()

    by_match: dict[int, dict[str, int]] = {}
    for match_id, side, n in rows:
        by_match.setdefault(match_id, {})[side] = n
    assert set(by_match) == {MATCH_EARLY_ID, MATCH_LATE_ID}
    for match_id, sides in by_match.items():
        assert sides == {"RADIANT": 5, "DIRE": 5}, match_id


def test_match_players_match_side_slot_is_unique(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        duplicates = store.sql(
            f"""
            SELECT match_id, side, slot_in_side, COUNT(*) AS n
            FROM {MATCH_PLAYERS_VIEW}
            GROUP BY match_id, side, slot_in_side
            HAVING COUNT(*) > 1
            """
        ).fetchall()
    assert duplicates == []


def test_match_players_preserves_parquet_player_team_and_hero_ids(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        rows = store.sql(
            f"""
            SELECT match_id, side, slot_in_side, player_id, team_id, hero_id
            FROM {MATCH_PLAYERS_VIEW}
            ORDER BY match_id, side, slot_in_side
            """
        ).fetchall()

    expected = _expected_match_player_rows()
    assert rows == expected


def test_match_players_joins_start_time_from_matches(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        rows = store.sql(
            f"SELECT DISTINCT match_id, start_time FROM {MATCH_PLAYERS_VIEW}"
        ).fetchall()

    start_times = dict(rows)
    assert start_times[MATCH_EARLY_ID] == EARLY_START_TIME
    assert start_times[MATCH_LATE_ID] == LATE_START_TIME


def test_match_players_independent_of_parquet_row_order(
    feature_store_config: FeatureStoreConfig,
) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(feature_store_config.match_players_path)
    reversed_indices = list(range(table.num_rows - 1, -1, -1))
    reversed_table = table.take(reversed_indices)
    assert reversed_table.column("match_id").to_pylist() != table.column(
        "match_id"
    ).to_pylist()
    pq.write_table(reversed_table, feature_store_config.match_players_path)

    with connect(feature_store_config) as store:
        rows = store.sql(
            f"""
            SELECT match_id, side, slot_in_side, player_id, team_id, hero_id
            FROM {MATCH_PLAYERS_VIEW}
            ORDER BY match_id, side, slot_in_side
            """
        ).fetchall()

    assert rows == _expected_match_player_rows()


def test_missing_match_players_parquet_fails_clearly(
    feature_store_config: FeatureStoreConfig, tmp_path: Path
) -> None:
    missing = tmp_path / "absent" / "match_players.parquet"
    config = FeatureStoreConfig(
        matches_path=feature_store_config.matches_path,
        match_players_path=missing,
        draft_events_path=feature_store_config.draft_events_path,
    )
    with pytest.raises(FileNotFoundError, match="match_players"):
        connect(config)


def test_duckdb_layer_does_not_unpivot_matches_player_columns() -> None:
    """`match_players.parquet` is the only authoritative player-row source."""
    import inspect

    from dota_predictor.features import duckdb_layer

    source = inspect.getsource(duckdb_layer)
    assert "UNION ALL" not in source
    assert "radiant_player_0_id" not in source
    assert "_match_players_view_sql(MATCHES_VIEW)" not in source


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
