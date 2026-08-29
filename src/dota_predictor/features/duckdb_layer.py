"""DuckDB analytical layer over the canonical Parquet dataset (Step 3A).

Opens an in-memory DuckDB connection with three views registered directly
over the canonical Parquet files -- `matches`/`draft_events` are read via
`read_parquet` (DuckDB streams/pushes-down projections and filters against
the Parquet file; nothing is materialized into pandas), and
`match_players` is a SQL reconstruction of the ten pivoted player columns
in `matches.parquet` back into one long-form row per (match, side, slot).

This module is intentionally small (see the Step 3A scope note below): it
only opens the connection and defines the three reusable relations. It
does not build feature matrices, does not cache/materialize anything to
disk, and does not require PostgreSQL -- the two canonical Parquet files
are the only input.

Scope note (Step 3A)
---------------------
This is infrastructure, not feature engineering. Predictive features
(Elo, rolling win rates, player ratings, hero statistics, etc.) and model
training are out of scope here -- see `features.availability` for the
information-availability contract a future feature-building layer
(Step 3B) must consult before reading any column exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import duckdb

from dota_predictor.features.config import FeatureStoreConfig, load_feature_store_config

__all__ = [
    "DRAFT_EVENTS_VIEW",
    "MATCHES_VIEW",
    "MATCH_PLAYERS_VIEW",
    "FeatureDuckDBConnection",
    "connect",
]

MATCHES_VIEW = "matches"
DRAFT_EVENTS_VIEW = "draft_events"
MATCH_PLAYERS_VIEW = "match_players"

_PLAYERS_PER_SIDE = 5
_SIDES = ("radiant", "dire")


def _quote_literal_path(path: Path) -> str:
    """SQL-quote `path` for interpolation into a `CREATE VIEW ... read_parquet(...)`
    statement.

    `CREATE VIEW` bodies cannot use prepared-statement parameters (DuckDB
    rejects `?`/`$n` there -- the view definition must be a plain SQL
    string), so the path is escaped and inlined directly instead. Single
    quotes are doubled per standard SQL string-literal escaping; `path`
    always comes from `FeatureStoreConfig` (our own configuration), never
    from untrusted external input.
    """
    return "'" + str(path).replace("'", "''") + "'"


def _match_players_view_sql(matches_view: str) -> str:
    """`SELECT` reconstructing the long-form player relation from the ten
    pivoted `{side}_player_{slot}_id` / `{side}_team_id` columns on
    `matches_view`.

    One `SELECT ... FROM matches_view` per (side, slot) -- 10 total --
    unioned together, so every one of the 10 canonical roster slots for a
    match becomes exactly one output row, preserving the exact
    `slot_in_side` -> `player_id` correspondence Step 2 already pivoted
    losslessly (see `datasets.canonical_export`). No aggregation or
    dedup is needed: the source columns are already one-per-slot.
    """
    selects = [
        f"""
        SELECT
            match_id,
            start_time,
            '{side.upper()}' AS side,
            {slot} AS slot_in_side,
            {side}_player_{slot}_id AS player_id,
            {side}_team_id AS team_id
        FROM {matches_view}
        """
        for side in _SIDES
        for slot in range(_PLAYERS_PER_SIDE)
    ]
    return " UNION ALL ".join(selects)


@dataclass(frozen=True)
class FeatureDuckDBConnection:
    """An open DuckDB connection with the canonical analytical views
    registered.

    Use as a context manager to guarantee the underlying connection is
    closed:

        with connect() as store:
            store.relation(MATCHES_VIEW).filter("league_id = 1").df()
    """

    connection: duckdb.DuckDBPyConnection
    config: FeatureStoreConfig

    def relation(self, view: str) -> duckdb.DuckDBPyRelation:
        """A lazy DuckDB relation over `view` (one of `MATCHES_VIEW`,
        `DRAFT_EVENTS_VIEW`, `MATCH_PLAYERS_VIEW`). Nothing is executed or
        materialized until the caller consumes the relation (`.df()`,
        `.fetchall()`, `.arrow()`, ...)."""
        return self.connection.table(view)

    def sql(self, query: str) -> duckdb.DuckDBPyRelation:
        """Run an arbitrary read-only query against the registered views."""
        return self.connection.sql(query)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def connect(config: FeatureStoreConfig | None = None) -> FeatureDuckDBConnection:
    """Open a DuckDB connection with `matches`/`draft_events`/`match_players`
    views registered directly over the canonical Parquet files.

    `config` defaults to `load_feature_store_config()` (the same
    `data/processed` / `PROCESSED_DATA_DIR` resolution the Step 2 export
    uses). The connection is in-memory (DuckDB's catalog only, not the
    Parquet data itself, which stays on disk and is read/pushed-down
    lazily by `read_parquet`); no PostgreSQL connection is ever opened.
    """
    resolved_config = config if config is not None else load_feature_store_config()
    connection = duckdb.connect(database=":memory:")

    connection.execute(
        f"CREATE VIEW {MATCHES_VIEW} AS "
        f"SELECT * FROM read_parquet({_quote_literal_path(resolved_config.matches_path)})"
    )
    connection.execute(
        f"CREATE VIEW {DRAFT_EVENTS_VIEW} AS "
        f"SELECT * FROM read_parquet({_quote_literal_path(resolved_config.draft_events_path)})"
    )
    connection.execute(
        f"CREATE VIEW {MATCH_PLAYERS_VIEW} AS {_match_players_view_sql(MATCHES_VIEW)}"
    )

    return FeatureDuckDBConnection(connection=connection, config=resolved_config)
