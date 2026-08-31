"""STRATZ constants -> Parquet reference-dimension export.

This is the reference-catalog companion to `canonical_export`. It is a
separate logical contract:

    STRATZ `constants.heroes` / `constants.gameVersions`
        -> **reference Parquet** (`heroes.parquet`, `game_versions.parquet`)

`ANALYTICAL_SCHEMA_VERSION` (the match-fact file set) is not touched.
`REFERENCE_SCHEMA_VERSION` versions this module's Parquet contract -- the
column set/types/semantics of `heroes.parquet` and `game_versions.parquet`.

Source-of-truth boundary
-------------------------
This module never reads PostgreSQL, never reads the canonical match-fact
Parquet files, and never calls STRATZ itself. Callers (see
`scripts/build_reference_dataset.py`) fetch constants via `StratzClient`
and pass the raw lists in. Transform, validation, and atomic publish live
here.

These catalogs are metadata dimensions, not predictive features. They
are not joined onto `matches` / `match_players` / `draft_events` by this
module or by the default DuckDB `connect()`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "GAME_VERSIONS_FILENAME",
    "GAME_VERSIONS_SCHEMA",
    "HEROES_FILENAME",
    "HEROES_SCHEMA",
    "REFERENCE_SCHEMA_VERSION",
    "ReferenceBuildResult",
    "ReferenceExportError",
    "ReferenceTransformError",
    "ReferenceValidationError",
    "build_game_versions_table",
    "build_heroes_table",
    "build_reference_dataset",
    "validate_game_versions_table",
    "validate_heroes_table",
    "write_reference_dataset",
]

# Reference Parquet contract version -- independent of
# `canonical_export.ANALYTICAL_SCHEMA_VERSION`.
# v1: heroes.parquet + game_versions.parquet.
REFERENCE_SCHEMA_VERSION = 1

HEROES_FILENAME = "heroes.parquet"
GAME_VERSIONS_FILENAME = "game_versions.parquet"

HEROES_SCHEMA = pa.schema(
    [
        pa.field("hero_id", pa.int32(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
    ]
)

GAME_VERSIONS_SCHEMA = pa.schema(
    [
        pa.field("game_version_id", pa.int32(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("as_of_datetime", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


class ReferenceExportError(Exception):
    """Base class for reference-dataset export failures."""


class ReferenceTransformError(ReferenceExportError):
    """Raised when a STRATZ constants row cannot be mapped into the
    reference schema."""


class ReferenceValidationError(ReferenceExportError):
    """Raised when a built reference table fails a pre-publication
    invariant check."""


@dataclass(frozen=True)
class ReferenceBuildResult:
    """Outcome of one successful `build_reference_dataset` run."""

    heroes_row_count: int
    game_versions_row_count: int
    heroes_path: Path
    game_versions_path: Path
    output_dir: Path
    schema_version: int = REFERENCE_SCHEMA_VERSION


def _require_positive_id(field: str, value: Any) -> int:
    if value is None:
        raise ReferenceTransformError(f"{field} is missing")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceTransformError(f"{field} is not an integer: {value!r}") from exc
    if parsed <= 0:
        raise ReferenceTransformError(
            f"{field} must be a positive integer, got {parsed}"
        )
    return parsed


def _require_non_empty_name(field: str, value: Any) -> str:
    if value is None:
        raise ReferenceTransformError(f"{field} is missing")
    if not isinstance(value, str):
        raise ReferenceTransformError(f"{field} is not a string: {value!r}")
    name = value.strip()
    if not name:
        raise ReferenceTransformError(f"{field} is empty")
    return name


def _unix_seconds_to_utc(value: Any) -> datetime:
    if value is None:
        raise ReferenceTransformError("asOfDateTime is missing")
    try:
        unix_seconds = int(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceTransformError(
            f"asOfDateTime is not an integer Unix timestamp: {value!r}"
        ) from exc
    return datetime.fromtimestamp(unix_seconds, tz=UTC)


def build_heroes_table(hero_rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    """Build the `heroes.parquet` Arrow table from STRATZ `constants.heroes`.

    Maps `id` -> `hero_id` and `displayName` -> `name`. Extra STRATZ
    fields (`name`, `shortName`, `aliases`, `gameVersionId`, gameplay
    metadata) are ignored and never published. Rows are emitted in
    ascending `hero_id` order.
    """
    out_rows: list[dict[str, Any]] = []
    for row in hero_rows:
        out_rows.append(
            {
                "hero_id": _require_positive_id("hero_id", row.get("id")),
                "name": _require_non_empty_name("name", row.get("displayName")),
            }
        )
    out_rows.sort(key=lambda row: row["hero_id"])
    return pa.Table.from_pylist(out_rows, schema=HEROES_SCHEMA)


def build_game_versions_table(
    version_rows: Sequence[Mapping[str, Any]],
) -> pa.Table:
    """Build the `game_versions.parquet` Arrow table from STRATZ
    `constants.gameVersions`.

    Maps `id` -> `game_version_id`, `name` -> `name`, and `asOfDateTime`
    (Unix seconds) -> UTC `as_of_datetime`. Id gaps (e.g. missing 174)
    and non-monotonic historical timestamps are allowed. Rows are
    emitted in ascending `game_version_id` order.
    """
    out_rows: list[dict[str, Any]] = []
    for row in version_rows:
        out_rows.append(
            {
                "game_version_id": _require_positive_id(
                    "game_version_id", row.get("id")
                ),
                "name": _require_non_empty_name("name", row.get("name")),
                "as_of_datetime": _unix_seconds_to_utc(row.get("asOfDateTime")),
            }
        )
    out_rows.sort(key=lambda row: row["game_version_id"])
    return pa.Table.from_pylist(out_rows, schema=GAME_VERSIONS_SCHEMA)


def validate_heroes_table(table: pa.Table) -> None:
    """Validate `heroes.parquet` invariants before publication."""
    if not table.schema.equals(HEROES_SCHEMA):
        raise ReferenceValidationError(
            "heroes table schema does not match HEROES_SCHEMA"
        )
    if table.column("hero_id").null_count > 0:
        raise ReferenceValidationError("heroes.hero_id contains null(s)")
    if table.column("name").null_count > 0:
        raise ReferenceValidationError("heroes.name contains null(s)")

    hero_ids = table.column("hero_id").to_pylist()
    names = table.column("name").to_pylist()
    if len(set(hero_ids)) != len(hero_ids):
        raise ReferenceValidationError("heroes contains duplicate hero_id values")
    for hero_id in hero_ids:
        if hero_id is None or int(hero_id) <= 0:
            raise ReferenceValidationError(
                f"heroes.hero_id must be a positive integer, got {hero_id}"
            )
    for name in names:
        if name is None or not str(name).strip():
            raise ReferenceValidationError("heroes.name contains an empty value")


def validate_game_versions_table(table: pa.Table) -> None:
    """Validate `game_versions.parquet` invariants before publication.

    Does not require contiguous ids, monotonic ids, monotonic
    timestamps, or a 1:1 mapping onto Valve's full patch list. A gap
    such as missing STRATZ id 174 is valid.
    """
    if not table.schema.equals(GAME_VERSIONS_SCHEMA):
        raise ReferenceValidationError(
            "game_versions table schema does not match GAME_VERSIONS_SCHEMA"
        )
    for column_name in ("game_version_id", "name", "as_of_datetime"):
        if table.column(column_name).null_count > 0:
            raise ReferenceValidationError(
                f"game_versions.{column_name} contains null(s)"
            )

    version_ids = table.column("game_version_id").to_pylist()
    names = table.column("name").to_pylist()
    timestamps = table.column("as_of_datetime").to_pylist()
    if len(set(version_ids)) != len(version_ids):
        raise ReferenceValidationError(
            "game_versions contains duplicate game_version_id values"
        )
    if len(set(names)) != len(names):
        raise ReferenceValidationError("game_versions contains duplicate name values")
    for version_id in version_ids:
        if version_id is None or int(version_id) <= 0:
            raise ReferenceValidationError(
                f"game_versions.game_version_id must be a positive integer, "
                f"got {version_id}"
            )
    for name in names:
        if name is None or not str(name).strip():
            raise ReferenceValidationError("game_versions.name contains an empty value")
    for stamp in timestamps:
        if stamp is None:
            raise ReferenceValidationError(
                "game_versions.as_of_datetime contains null(s)"
            )
        if getattr(stamp, "tzinfo", None) is None:
            raise ReferenceValidationError(
                "game_versions.as_of_datetime must be timezone-aware UTC"
            )


def _write_and_read_back(table: pa.Table, path: Path) -> None:
    """Write `table` to `path` and immediately read it back, raising
    `ReferenceValidationError` if the round trip doesn't reproduce the
    same row count and schema."""
    pq.write_table(table, path)
    read_back = pq.read_table(path)
    if read_back.num_rows != table.num_rows:
        raise ReferenceValidationError(
            f"{path.name}: round-trip row count {read_back.num_rows} != "
            f"expected {table.num_rows}"
        )
    if not read_back.schema.equals(table.schema):
        raise ReferenceValidationError(f"{path.name}: round-trip schema mismatch")


def write_reference_dataset(
    output_dir: Path,
    *,
    heroes_table: pa.Table,
    game_versions_table: pa.Table,
) -> None:
    """Atomically publish both reference Parquet files under `output_dir`.

    Pattern matches `canonical_export.write_canonical_dataset`: write to
    a temporary directory inside `output_dir`, round-trip each file, then
    `os.replace` onto the final names. A failed build leaves existing
    final files untouched.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        HEROES_FILENAME: heroes_table,
        GAME_VERSIONS_FILENAME: game_versions_table,
    }

    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=".ref-build-"
    ) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tmp_paths: dict[str, Path] = {}
        for filename, table in tables.items():
            tmp_path = tmp_dir / filename
            _write_and_read_back(table, tmp_path)
            tmp_paths[filename] = tmp_path

        for filename, tmp_path in tmp_paths.items():
            os.replace(tmp_path, output_dir / filename)


def build_reference_dataset(
    output_dir: Path,
    *,
    heroes: Sequence[Mapping[str, Any]],
    game_versions: Sequence[Mapping[str, Any]],
) -> ReferenceBuildResult:
    """Transform, validate, and atomically publish both reference catalogs.

    Independent of `build_canonical_dataset`: does not read or write
    match-fact Parquet, and does not open PostgreSQL.
    """
    heroes_table = build_heroes_table(heroes)
    game_versions_table = build_game_versions_table(game_versions)

    validate_heroes_table(heroes_table)
    validate_game_versions_table(game_versions_table)

    write_reference_dataset(
        output_dir,
        heroes_table=heroes_table,
        game_versions_table=game_versions_table,
    )

    return ReferenceBuildResult(
        heroes_row_count=heroes_table.num_rows,
        game_versions_row_count=game_versions_table.num_rows,
        heroes_path=output_dir / HEROES_FILENAME,
        game_versions_path=output_dir / GAME_VERSIONS_FILENAME,
        output_dir=output_dir,
    )
