"""STRATZ constants -> Parquet reference-dimension export.

This is the reference-catalog companion to `canonical_export`. It is a
separate logical contract:

    STRATZ `constants.heroes` / `constants.gameVersions`
        -> **reference Parquet** (`heroes.parquet`, `game_versions.parquet`)

`ANALYTICAL_SCHEMA_VERSION` (the match-fact file set) is not touched.
`REFERENCE_SCHEMA_VERSION` versions this module's Parquet contract -- the
column set/types/semantics of `heroes.parquet` and `game_versions.parquet`.

Since v2 both catalogs carry provenance: `source` names the authoritative
STRATZ constant the rows came from and `retrieved_at` records when that
catalog was fetched. `heroes.parquet` additionally exposes the
STRATZ-supplied `short_name` and `aliases` identity fields; nothing here
is ever fabricated or inferred (in particular, `game_versions.as_of_datetime`
is STRATZ's authoritative patch timestamp, not a value derived from the
match corpus).

Source-of-truth boundary
------------------------
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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "GAME_VERSIONS_FILENAME",
    "GAME_VERSIONS_SCHEMA",
    "GAME_VERSIONS_SOURCE",
    "HEROES_FILENAME",
    "HEROES_SCHEMA",
    "HEROES_SOURCE",
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
# v1: heroes.parquet (hero_id, name) + game_versions.parquet
#     (game_version_id, name, as_of_datetime).
# v2: adds provenance (`source`, `retrieved_at`) to both files and, for
#     heroes, the STRATZ-supplied `short_name` + `aliases` identity fields.
REFERENCE_SCHEMA_VERSION = 2

HEROES_FILENAME = "heroes.parquet"
GAME_VERSIONS_FILENAME = "game_versions.parquet"

# Provenance constants. `source` is the authoritative STRATZ constant the
# catalog row was retrieved from; `retrieved_at` is when that catalog was
# fetched (both are the same for every row of a single build).
HEROES_SOURCE = "STRATZ constants.heroes"
GAME_VERSIONS_SOURCE = "STRATZ constants.gameVersions"

HEROES_SCHEMA = pa.schema(
    [
        pa.field("hero_id", pa.int32(), nullable=False),
        # Canonical human-readable hero name (STRATZ `displayName`).
        pa.field("name", pa.string(), nullable=False),
        # Canonical short/slug name (STRATZ `shortName`), e.g. "antimage".
        # Genuinely supplied by STRATZ; null only if the source omits it.
        pa.field("short_name", pa.string(), nullable=True),
        # STRATZ-supplied alias list (may be empty). Genuinely supplied;
        # null only if the source omits it. Never fabricated.
        pa.field("aliases", pa.list_(pa.string()), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("retrieved_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

GAME_VERSIONS_SCHEMA = pa.schema(
    [
        pa.field("game_version_id", pa.int32(), nullable=False),
        # Human-readable patch label (STRATZ `name`), e.g. "7.38".
        pa.field("name", pa.string(), nullable=False),
        # STRATZ asOfDateTime: the authoritative patch release timestamp.
        # Source-provided; NOT inferred from first-seen-in-corpus.
        pa.field("as_of_datetime", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("retrieved_at", pa.timestamp("us", tz="UTC"), nullable=False),
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
    # When the STRATZ constants were fetched (provenance for both files).
    # Defaults to the epoch so a hand-constructed result is unambiguous.
    retrieved_at: datetime = field(default_factory=lambda: datetime.fromtimestamp(0, tz=UTC))


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


def _optional_short_name(value: Any) -> str | None:
    """Normalize a STRATZ `shortName` to a stripped non-empty string or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ReferenceTransformError(f"shortName is not a string: {value!r}")
    name = value.strip()
    return name if name else None


def _optional_aliases(value: Any) -> list[str] | None:
    """Normalize a STRATZ `aliases` list to a list of non-empty strings or None.

    STRATZ genuinely supplies `aliases` (possibly empty) for every hero;
    `None` is kept only if the source omits the field. Values are never
    invented: each element is validated as a non-empty string.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ReferenceTransformError(f"aliases is not a list: {value!r}")
    aliases: list[str] = []
    for alias in value:
        if not isinstance(alias, str) or not alias.strip():
            raise ReferenceTransformError(f"aliases contains an invalid entry: {alias!r}")
        aliases.append(alias.strip())
    return aliases


def build_heroes_table(
    hero_rows: Sequence[Mapping[str, Any]], *, retrieved_at: datetime
) -> pa.Table:
    """Build the `heroes.parquet` Arrow table from STRATZ `constants.heroes`.

    Maps `id` -> `hero_id`, `displayName` -> `name`, `shortName` ->
    `short_name`, and `aliases` -> `aliases` (all genuinely supplied by
    STRATZ). Adds provenance (`source` = `HEROES_SOURCE`,
    `retrieved_at` = the caller-supplied fetch timestamp). Extra STRATZ
    gameplay metadata (`roles`, `stats`, etc.) is ignored and never
    published. Rows are emitted in ascending `hero_id` order.
    """
    out_rows: list[dict[str, Any]] = []
    for row in hero_rows:
        out_rows.append(
            {
                "hero_id": _require_positive_id("hero_id", row.get("id")),
                "name": _require_non_empty_name("name", row.get("displayName")),
                "short_name": _optional_short_name(row.get("shortName")),
                "aliases": _optional_aliases(row.get("aliases")),
                "source": HEROES_SOURCE,
                "retrieved_at": retrieved_at,
            }
        )
    out_rows.sort(key=lambda row: row["hero_id"])
    return pa.Table.from_pylist(out_rows, schema=HEROES_SCHEMA)


def build_game_versions_table(
    version_rows: Sequence[Mapping[str, Any]], *, retrieved_at: datetime
) -> pa.Table:
    """Build the `game_versions.parquet` Arrow table from STRATZ
    `constants.gameVersions`.

    Maps `id` -> `game_version_id`, `name` -> `name`, and `asOfDateTime`
    (Unix seconds) -> UTC `as_of_datetime` (the authoritative patch release
    timestamp). Adds provenance (`source` = `GAME_VERSIONS_SOURCE`,
    `retrieved_at` = the caller-supplied fetch timestamp). Id gaps (e.g.
    missing 174) and non-monotonic historical timestamps are allowed.
    Rows are emitted in ascending `game_version_id` order.
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
                "source": GAME_VERSIONS_SOURCE,
                "retrieved_at": retrieved_at,
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
    for column_name in ("hero_id", "name", "source", "retrieved_at"):
        if table.column(column_name).null_count > 0:
            raise ReferenceValidationError(
                f"heroes.{column_name} contains null(s)"
            )

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

    short_names = table.column("short_name").to_pylist()
    for short_name in short_names:
        if short_name is not None and not str(short_name).strip():
            raise ReferenceValidationError(
                "heroes.short_name contains an empty value"
            )

    aliases = table.column("aliases").to_pylist()
    for alias_list in aliases:
        if alias_list is None:
            continue
        for alias in alias_list:
            if not isinstance(alias, str) or not alias.strip():
                raise ReferenceValidationError(
                    f"heroes.aliases contains an invalid entry: {alias!r}"
                )

    for stamp in table.column("retrieved_at").to_pylist():
        if stamp is None:
            raise ReferenceValidationError("heroes.retrieved_at contains null(s)")
        if getattr(stamp, "tzinfo", None) is None:
            raise ReferenceValidationError(
                "heroes.retrieved_at must be timezone-aware UTC"
            )


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
    for column_name in (
        "game_version_id",
        "name",
        "as_of_datetime",
        "source",
        "retrieved_at",
    ):
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
    retrieved_at: datetime | None = None,
) -> ReferenceBuildResult:
    """Transform, validate, and atomically publish both reference catalogs.

    Independent of `build_canonical_dataset`: does not read or write
    match-fact Parquet, and does not open PostgreSQL.

    `retrieved_at` records when the STRATZ constants were fetched
    (provenance). When omitted it defaults to the current time; callers
    such as `scripts/build_reference_dataset.py` pass the actual fetch
    timestamp so the catalog's provenance reflects when it was really
    retrieved.
    """
    stamp = retrieved_at if retrieved_at is not None else datetime.now(UTC)
    heroes_table = build_heroes_table(heroes, retrieved_at=stamp)
    game_versions_table = build_game_versions_table(
        game_versions, retrieved_at=stamp
    )

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
        retrieved_at=stamp,
    )
