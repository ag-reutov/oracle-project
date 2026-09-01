"""DuckDB analytical layer over the canonical Parquet dataset (Step 3A).

Opens an in-memory DuckDB connection with three views registered directly
over the canonical Parquet files. `matches` and `draft_events` are read
via `read_parquet` (DuckDB streams/pushes-down projections and filters
against the Parquet file; nothing is materialized into pandas).
`match_players` is `match_players.parquet` joined to `matches` so the
relation keeps a denormalized `start_time` (required by PRE_DRAFT
historical SQL) without storing `start_time` on the player file.

Optional STRATZ reference catalogs (`heroes`, `game_versions`) are not
registered by `connect()`. Call `register_reference_views` explicitly
when those Parquet files are present. They are never auto-joined into
the fact views.

This module is intentionally small (see the Step 3A scope note below): it
only opens the connection and defines the reusable relations. It
does not build feature matrices, does not cache/materialize anything to
disk, and does not require PostgreSQL -- the three canonical Parquet files
are the only input required by `connect()`.

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

from dota_predictor.features.config import (
    FeatureStoreConfig,
    ReferenceStoreConfig,
    load_feature_store_config,
    load_reference_store_config,
)

__all__ = [
    "DRAFT_EVENTS_VIEW",
    "GAME_VERSIONS_VIEW",
    "HEROES_VIEW",
    "MATCHES_VIEW",
    "MATCH_PLAYERS_VIEW",
    "FeatureDuckDBConnection",
    "connect",
    "register_reference_views",
]

MATCHES_VIEW = "matches"
DRAFT_EVENTS_VIEW = "draft_events"
MATCH_PLAYERS_VIEW = "match_players"
HEROES_VIEW = "heroes"
GAME_VERSIONS_VIEW = "game_versions"


def _quote_literal_path(path: Path) -> str:
    """SQL-quote `path` for interpolation into a `CREATE VIEW ... read_parquet(...)`
    statement.

    `CREATE VIEW` bodies cannot use prepared-statement parameters (DuckDB
    rejects `?`/`$n` there -- the view definition must be a plain SQL
    string), so the path is escaped and inlined directly instead. Single
    quotes are doubled per standard SQL string-literal escaping; `path`
    always comes from `FeatureStoreConfig` or `ReferenceStoreConfig`
    (our own configuration), never from untrusted external input.
    """
    return "'" + str(path).replace("'", "''") + "'"


def _require_parquet_file(path: Path, *, relation: str) -> None:
    """Raise `FileNotFoundError` if `path` is not an existing Parquet file.

    DuckDB's own missing-file error is an opaque IOException; failing here
    names the analytical relation and the resolved path before any view is
    registered.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Parquet for DuckDB relation {relation!r} is missing: {path}"
        )


def _match_players_view_sql(*, match_players_path: Path, matches_view: str) -> str:
    """Explicit projection of `match_players.parquet` joined to `matches`.

    `start_time` is taken from `matches_view`, not from the player file.
    Column order is the DuckDB `match_players` contract and must not
    follow physical Parquet column or row order.
    """
    quoted = _quote_literal_path(match_players_path)
    return f"""
        SELECT
            mp.match_id,
            m.start_time,
            mp.side,
            mp.slot_in_side,
            mp.player_id,
            mp.team_id,
            mp.hero_id,
            mp.position,
            mp.lane,
            mp.role
        FROM read_parquet({quoted}) AS mp
        JOIN {matches_view} AS m USING (match_id)
    """


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
        """A lazy DuckDB relation over `view`.

        Fact views (`MATCHES_VIEW`, `DRAFT_EVENTS_VIEW`,
        `MATCH_PLAYERS_VIEW`) are always registered by `connect()`.
        Reference views (`HEROES_VIEW`, `GAME_VERSIONS_VIEW`) exist only
        after `register_reference_views`. Nothing is executed or
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
    _require_parquet_file(resolved_config.matches_path, relation=MATCHES_VIEW)
    _require_parquet_file(
        resolved_config.match_players_path, relation=MATCH_PLAYERS_VIEW
    )
    _require_parquet_file(
        resolved_config.draft_events_path, relation=DRAFT_EVENTS_VIEW
    )

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
        f"CREATE VIEW {MATCH_PLAYERS_VIEW} AS "
        + _match_players_view_sql(
            match_players_path=resolved_config.match_players_path,
            matches_view=MATCHES_VIEW,
        )
    )

    return FeatureDuckDBConnection(connection=connection, config=resolved_config)


def _heroes_view_sql(heroes_path: Path) -> str:
    quoted = _quote_literal_path(heroes_path)
    return f"""
        SELECT
            hero_id,
            name
        FROM read_parquet({quoted})
    """


def _game_versions_view_sql(game_versions_path: Path) -> str:
    quoted = _quote_literal_path(game_versions_path)
    return f"""
        SELECT
            game_version_id,
            name,
            as_of_datetime
        FROM read_parquet({quoted})
    """


def register_reference_views(
    store: FeatureDuckDBConnection,
    config: ReferenceStoreConfig | None = None,
) -> None:
    """Register optional `heroes` and `game_versions` views on `store`.

    Requires both reference Parquet files to exist. Does not join them
    into `matches` / `match_players` / `draft_events`. `connect()` does
    not call this.
    """
    resolved = config if config is not None else load_reference_store_config()
    _require_parquet_file(resolved.heroes_path, relation=HEROES_VIEW)
    _require_parquet_file(resolved.game_versions_path, relation=GAME_VERSIONS_VIEW)

    store.connection.execute(
        f"CREATE VIEW {HEROES_VIEW} AS " + _heroes_view_sql(resolved.heroes_path)
    )
    store.connection.execute(
        f"CREATE VIEW {GAME_VERSIONS_VIEW} AS "
        + _game_versions_view_sql(resolved.game_versions_path)
    )
