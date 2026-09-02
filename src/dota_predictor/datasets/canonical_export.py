"""Canonical PostgreSQL -> Parquet analytical dataset export.

This is the first "canonical analytical Parquet" layer in the project's
data pipeline:

    STRATZ -> ingestion -> canonical Python model -> PostgreSQL canonical
    tables -> **canonical analytical Parquet** -> feature engineering -> ...

It reads the canonical relational tables (`matches`, `match_players`,
`draft_events` -- see `storage.schema`) and rewrites them as two flat
analytical Parquet datasets:

* `matches.parquet` -- one row per canonical Dota game. The ten
  `match_players` rows for that match are pivoted into fixed
  `radiant_player_{0..4}_id` / `dire_player_{0..4}_id` columns using each
  row's canonical `slot_in_side` (never inferred from player id or query
  order). Hero ids are **not** pivoted onto this file; they live on the
  long-form `match_players.parquet` table.
* `match_players.parquet` -- one row per player per match (exactly 10
  rows per canonical match). Carries `player_id`, `hero_id`, `side`,
  `slot_in_side` (lobby order, not Dota position 1-5), `team_id`
  derived from the parent match's radiant/dire team ids, observed
  STRATZ parse labels `position` / `lane` / `role` (POST_MATCH relative
  to that row's match; NULL and UNKNOWN are preserved), and observed
  STRATZ post-match box-score scalars (POST_MATCH; NULL and zero are
  preserved, never coerced).
* `draft_events.parquet` -- the normalized long representation of
  `draft_events`, one row per pick/ban, preserved verbatim. Draft length
  is **not** fixed across matches: real canonical data contains 10-, 23-,
  and 24-event drafts, including matches where STRATZ recorded all ten
  picks but zero bans. This module never flattens, pads, or fabricates
  events to force a fixed shape -- ordering is preserved exactly and is
  reconstructable via `ORDER BY match_id, sequence`.

Source-of-truth boundary
-------------------------
This module reads exclusively from the three canonical domain tables. It
deliberately never reads `stratz_raw_matches`, `league_ingestion_state`,
`match_ingestion_errors`, `ingestion_leagues`, or `leagues` -- those are
ingestion-side operational bookkeeping (see `storage.schema` module
docstring), not analytical data. Raw matches that failed canonicalization
are therefore naturally absent from the export: this module never queries
or knows about them.

Temporal-integrity boundary
----------------------------
This module performs a *storage-format* transformation only: canonical
relational rows -> canonical analytical Parquet rows. It does not decide
what is safe to use as a model input at a given prediction time -- per
`.cursor/rules/ml.mdc` and `canonical_schema.FIELD_INFORMATION_AVAILABILITY`,
that is the future feature-building layer's responsibility.
`radiant_win`/`duration_seconds` (POST_MATCH information) are preserved in
`matches.parquet` as historical outcome/label data, exactly as they are
preserved in `CanonicalMatch` and `matches` -- they are not filtered out
here, and no separate `pre_draft.parquet`/`post_draft.parquet` files are
created. `draft_events.parquet`'s long representation is precisely what
lets a future consumer slice `draft_events[:t]` per
`FIELD_INFORMATION_AVAILABILITY`'s DRAFT semantics without this module
needing to anticipate every possible prediction-time view.

This module also does not derive `game_number_in_series`: STRATZ does not
expose that field directly, the canonical mapper
(`stratz_mapping.canonical_match_from_stratz`) leaves it `NULL`, and
deriving it (e.g. by sorting each series's games by `start_time`) is a
separate semantic transformation left to a later layer. `matches.parquet`
carries the canonical value through verbatim (currently `NULL` for every
row in the live dataset).

Versioning
----------
`ANALYTICAL_SCHEMA_VERSION` versions *this module's Parquet contract*
(the column set/types/semantics of `matches.parquet` /
`match_players.parquet` / `draft_events.parquet`)
-- it is independent of `stratz_mapping.CANONICAL_MAPPER_VERSION`, which
versions the STRATZ-to-`CanonicalMatch` mapping logic upstream in
Postgres. `matches.mapper_version`/`matches.canonicalized_at` are carried
through into `matches.parquet` unchanged, as per-row provenance of the
canonical source row; they are unrelated to `ANALYTICAL_SCHEMA_VERSION`.

`ANALYTICAL_SCHEMA_VERSION` is deliberately *not* duplicated as a column
on every row. Every build is a full, deterministic rebuild (see
`build_canonical_dataset`), so there is no per-row "which version wrote
this row" question to answer the way `mapper_version` answers it for
incremental canonicalization -- the whole file pair is always produced by
exactly one version of this module. The version is therefore a property
of the *code that produced the files*, checked the same way a library
version is checked (`dota_predictor.datasets.canonical_export.ANALYTICAL_SCHEMA_VERSION`
at build time, or a Parquet file's embedded schema metadata after the
fact), not a value a downstream reader needs to filter rows by.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Connection, Engine, select

from dota_predictor.data.canonical_schema import MATCH_PLAYER_BOX_SCORE_COLUMNS
from dota_predictor.storage.schema import DRAFT_EVENTS, MATCH_PLAYERS, MATCHES

__all__ = [
    "ANALYTICAL_SCHEMA_VERSION",
    "DRAFT_EVENTS_FILENAME",
    "DRAFT_EVENTS_SCHEMA",
    "MATCHES_FILENAME",
    "MATCHES_SCHEMA",
    "MATCH_PLAYERS_FILENAME",
    "MATCH_PLAYERS_SCHEMA",
    "CanonicalExportError",
    "DatasetBuildResult",
    "DatasetTransformError",
    "DatasetValidationError",
    "build_canonical_dataset",
    "build_draft_events_table",
    "build_match_players_table",
    "build_matches_table",
    "validate_draft_events_table",
    "validate_match_players_table",
    "validate_matches_table",
    "write_canonical_dataset",
]

# Parquet contract version -- see module docstring "Versioning" section.
# v1: matches.parquet + draft_events.parquet.
# v2: adds match_players.parquet (long-form player/hero rows). matches.parquet
#     column set is unchanged from v1.
# v3: adds observed STRATZ `position` / `lane` / `role` on
#     match_players.parquet. matches.parquet is unchanged.
# v4: adds observed STRATZ post-match box-score scalars on
#     match_players.parquet. matches.parquet is unchanged.
ANALYTICAL_SCHEMA_VERSION = 4

MATCHES_FILENAME = "matches.parquet"
MATCH_PLAYERS_FILENAME = "match_players.parquet"
DRAFT_EVENTS_FILENAME = "draft_events.parquet"

_PLAYERS_PER_SIDE = 5
_SIDES = ("RADIANT", "DIRE")

# Column types mirror the Postgres column widths in `storage.schema`
# (BIGINT -> int64, INTEGER -> int32, SMALLINT -> int16) rather than
# collapsing everything to int64, so the exported files stay type-faithful
# to the canonical source and compact on disk.
MATCHES_SCHEMA = pa.schema(
    [
        pa.field("match_id", pa.int64(), nullable=False),
        pa.field("league_id", pa.int64(), nullable=False),
        pa.field("start_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("league_name", pa.string(), nullable=True),
        pa.field("series_id", pa.int64(), nullable=True),
        pa.field("series_type", pa.string(), nullable=True),
        pa.field("game_number_in_series", pa.int16(), nullable=True),
        pa.field("game_version_id", pa.int32(), nullable=True),
        pa.field("radiant_team_id", pa.int64(), nullable=False),
        pa.field("radiant_team_name_observed", pa.string(), nullable=True),
        pa.field("dire_team_id", pa.int64(), nullable=False),
        pa.field("dire_team_name_observed", pa.string(), nullable=True),
        pa.field("radiant_player_0_id", pa.int64(), nullable=False),
        pa.field("radiant_player_1_id", pa.int64(), nullable=False),
        pa.field("radiant_player_2_id", pa.int64(), nullable=False),
        pa.field("radiant_player_3_id", pa.int64(), nullable=False),
        pa.field("radiant_player_4_id", pa.int64(), nullable=False),
        pa.field("dire_player_0_id", pa.int64(), nullable=False),
        pa.field("dire_player_1_id", pa.int64(), nullable=False),
        pa.field("dire_player_2_id", pa.int64(), nullable=False),
        pa.field("dire_player_3_id", pa.int64(), nullable=False),
        pa.field("dire_player_4_id", pa.int64(), nullable=False),
        pa.field("radiant_win", pa.bool_(), nullable=False),
        pa.field("duration_seconds", pa.int32(), nullable=False),
        pa.field("mapper_version", pa.int32(), nullable=False),
        pa.field("canonicalized_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

DRAFT_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("match_id", pa.int64(), nullable=False),
        pa.field("sequence", pa.int16(), nullable=False),
        # `action`/`side` are exported as plain strings ("PICK"/"BAN",
        # "RADIANT"/"DIRE"), not a Parquet/Arrow enum or dictionary type --
        # the value set is small and stable, and a plain string column
        # keeps the file trivially readable from any Parquet reader
        # without decoding a dictionary-encoding convention.
        pa.field("action", pa.string(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        pa.field("hero_id", pa.int32(), nullable=False),
        pa.field("was_successful", pa.bool_(), nullable=True),
    ]
)

MATCH_PLAYERS_SCHEMA = pa.schema(
    [
        pa.field("match_id", pa.int64(), nullable=False),
        pa.field("team_id", pa.int64(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        pa.field("slot_in_side", pa.int16(), nullable=False),
        pa.field("player_id", pa.int64(), nullable=False),
        pa.field("hero_id", pa.int32(), nullable=False),
        pa.field("position", pa.string(), nullable=True),
        pa.field("lane", pa.string(), nullable=True),
        pa.field("role", pa.string(), nullable=True),
        pa.field("kills", pa.int32(), nullable=True),
        pa.field("deaths", pa.int32(), nullable=True),
        pa.field("assists", pa.int32(), nullable=True),
        pa.field("gold_per_minute", pa.int32(), nullable=True),
        pa.field("experience_per_minute", pa.int32(), nullable=True),
        pa.field("num_last_hits", pa.int32(), nullable=True),
        pa.field("num_denies", pa.int32(), nullable=True),
        pa.field("networth", pa.int32(), nullable=True),
        pa.field("hero_damage", pa.int32(), nullable=True),
        pa.field("tower_damage", pa.int32(), nullable=True),
        pa.field("hero_healing", pa.int32(), nullable=True),
        pa.field("level", pa.int32(), nullable=True),
    ]
)


class CanonicalExportError(Exception):
    """Base class for canonical dataset export failures."""


class DatasetTransformError(CanonicalExportError):
    """Raised when canonical relational rows cannot be assembled into the
    analytical export schema -- e.g. a match with a missing or duplicated
    `match_players` slot. This is a defensive transformation-correctness
    check (see module docstring); it does not redefine canonical validity
    already enforced by `CanonicalMatch`/Postgres constraints upstream.
    """


class DatasetValidationError(CanonicalExportError):
    """Raised when a built Arrow table fails a pre-publication invariant
    check (see `validate_matches_table`/`validate_draft_events_table`)."""


@dataclass(frozen=True)
class DatasetBuildResult:
    """Outcome of one successful `build_canonical_dataset` run."""

    matches_row_count: int
    match_players_row_count: int
    draft_events_row_count: int
    output_dir: Path
    schema_version: int = ANALYTICAL_SCHEMA_VERSION


def _enum_value(value: Any) -> str:
    """Normalize a `Side`/`DraftAction` enum member (or a plain string, as
    used by pure in-memory tests) to its plain string value.

    `storage.schema` maps `Side`/`DraftAction` as non-native SQLAlchemy
    enums backed by the Python enum classes from `canonical_schema`, so
    rows fetched from Postgres deserialize these columns back into actual
    enum instances, not raw strings.
    """
    if isinstance(value, Enum):
        return value.value
    return str(value)


def _optional_enum_value(value: Any) -> str | None:
    """Like `_enum_value`, but NULL stays NULL."""
    if value is None:
        return None
    return _enum_value(value)


def _optional_int_value(value: Any) -> int | None:
    """Preserve NULL; keep zero as zero. Missing keys arrive as None."""
    if value is None:
        return None
    return int(value)


def _pivot_player_slots(
    match_player_rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, list[int | None]]]:
    """Pivot long `match_players` rows into per-match, per-side slot arrays.

    Returns `{match_id: {"RADIANT": [p0..p4], "DIRE": [p0..p4]}}`, using
    each row's own `slot_in_side` (never player id or row order) to place
    it. Raises `DatasetTransformError` -- rather than silently producing
    null player columns -- for any side/slot value outside the expected
    range, any slot filled twice with different data (a silent pivot
    collision), or any match left with fewer than 5 filled slots on a
    side once all rows are processed.
    """
    pivot: dict[int, dict[str, list[int | None]]] = {}
    for row in match_player_rows:
        match_id = int(row["match_id"])
        side = _enum_value(row["side"])
        slot = int(row["slot_in_side"])
        player_id = row["player_id"]

        if side not in _SIDES:
            raise DatasetTransformError(
                f"match {match_id}: unrecognized side {side!r} in match_players"
            )
        if not (0 <= slot < _PLAYERS_PER_SIDE):
            raise DatasetTransformError(
                f"match {match_id}: slot_in_side {slot} out of the expected "
                f"0-{_PLAYERS_PER_SIDE - 1} range"
            )

        slots = pivot.setdefault(
            match_id,
            {"RADIANT": [None] * _PLAYERS_PER_SIDE, "DIRE": [None] * _PLAYERS_PER_SIDE},
        )
        if slots[side][slot] is not None:
            raise DatasetTransformError(
                f"match {match_id}: duplicate {side} slot {slot} in match_players"
            )
        slots[side][slot] = player_id

    for match_id, slots in pivot.items():
        for side, values in slots.items():
            missing = [i for i, value in enumerate(values) if value is None]
            if missing:
                raise DatasetTransformError(
                    f"match {match_id}: missing {side} player slot(s) {missing}"
                )

    return pivot


def build_matches_table(
    match_rows: Sequence[Mapping[str, Any]],
    match_player_rows: Sequence[Mapping[str, Any]],
) -> pa.Table:
    """Build the `matches.parquet` Arrow table from canonical relational rows.

    `match_rows` must be `matches` rows (one per canonical match, any
    mapping-like object -- e.g. a SQLAlchemy `RowMapping` or a plain
    `dict`); `match_player_rows` must be the corresponding `match_players`
    rows (5 RADIANT + 5 DIRE per match, at slots 0-4). Raises
    `DatasetTransformError` if any match's player rows don't form a
    complete, non-duplicated 5-per-side set (see `_pivot_player_slots`),
    or if a `matches` row has no corresponding `match_players` rows at
    all.

    Rows are emitted in ascending `match_id` order regardless of input
    order, for deterministic output.
    """
    player_slots = _pivot_player_slots(match_player_rows)
    ordered_matches = sorted(match_rows, key=lambda row: int(row["match_id"]))

    out_rows: list[dict[str, Any]] = []
    for row in ordered_matches:
        match_id = int(row["match_id"])
        slots = player_slots.get(match_id)
        if slots is None:
            raise DatasetTransformError(
                f"match {match_id}: no match_players rows found"
            )

        out_row: dict[str, Any] = {
            "match_id": match_id,
            "league_id": int(row["league_id"]),
            "start_time": row["start_time"],
            "league_name": row["league_name"],
            "series_id": row["series_id"],
            "series_type": row["series_type"],
            "game_number_in_series": row["game_number_in_series"],
            "game_version_id": row["game_version_id"],
            "radiant_team_id": int(row["radiant_team_id"]),
            "radiant_team_name_observed": row["radiant_team_name_observed"],
            "dire_team_id": int(row["dire_team_id"]),
            "dire_team_name_observed": row["dire_team_name_observed"],
            "radiant_win": bool(row["radiant_win"]),
            "duration_seconds": int(row["duration_seconds"]),
            "mapper_version": int(row["mapper_version"]),
            "canonicalized_at": row["canonicalized_at"],
        }
        for i in range(_PLAYERS_PER_SIDE):
            out_row[f"radiant_player_{i}_id"] = int(slots["RADIANT"][i])
            out_row[f"dire_player_{i}_id"] = int(slots["DIRE"][i])
        out_rows.append(out_row)

    return pa.Table.from_pylist(out_rows, schema=MATCHES_SCHEMA)


def build_draft_events_table(draft_event_rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    """Build the `draft_events.parquet` Arrow table.

    Preserves the long, normalized draft representation exactly: no fixed
    event count or minimum ban count is assumed or enforced (real
    canonical data contains 10-, 23-, and 24-event drafts -- see module
    docstring). Rows are emitted ordered by `(match_id, sequence)`
    regardless of input order.
    """
    ordered = sorted(
        draft_event_rows, key=lambda row: (int(row["match_id"]), int(row["sequence"]))
    )
    out_rows = [
        {
            "match_id": int(row["match_id"]),
            "sequence": int(row["sequence"]),
            "action": _enum_value(row["action"]),
            "side": _enum_value(row["side"]),
            "hero_id": int(row["hero_id"]),
            "was_successful": row["was_successful"],
        }
        for row in ordered
    ]
    return pa.Table.from_pylist(out_rows, schema=DRAFT_EVENTS_SCHEMA)


def build_match_players_table(
    match_rows: Sequence[Mapping[str, Any]],
    match_player_rows: Sequence[Mapping[str, Any]],
) -> pa.Table:
    """Build the `match_players.parquet` Arrow table.

    One row per canonical player. `team_id` is taken from the parent
    `matches` row (`radiant_team_id` / `dire_team_id` according to
    `side`), never from raw STRATZ player objects. `slot_in_side` is
    lobby order and is not rewritten as a Dota position. Observed
    `position` / `lane` / `role` are copied as nullable strings;
    missing keys become NULL and are never inferred. Observed box-score
    scalars are copied as nullable ints; missing stays NULL and zero
    stays zero.
    """
    teams_by_match: dict[int, dict[str, int]] = {}
    for row in match_rows:
        match_id = int(row["match_id"])
        teams_by_match[match_id] = {
            "RADIANT": int(row["radiant_team_id"]),
            "DIRE": int(row["dire_team_id"]),
        }

    out_rows: list[dict[str, Any]] = []
    for row in match_player_rows:
        match_id = int(row["match_id"])
        side = _enum_value(row["side"])
        teams = teams_by_match.get(match_id)
        if teams is None:
            raise DatasetTransformError(
                f"match {match_id}: match_players row has no parent matches row"
            )
        if side not in teams:
            raise DatasetTransformError(
                f"match {match_id}: unrecognized side {side!r} in match_players"
            )
        hero_id = row.get("hero_id")
        if hero_id is None:
            raise DatasetTransformError(
                f"match {match_id}: match_players row missing hero_id"
            )
        out_rows.append(
            {
                "match_id": match_id,
                "team_id": teams[side],
                "side": side,
                "slot_in_side": int(row["slot_in_side"]),
                "player_id": int(row["player_id"]),
                "hero_id": int(hero_id),
                "position": _optional_enum_value(row.get("position")),
                "lane": _optional_enum_value(row.get("lane")),
                "role": _optional_enum_value(row.get("role")),
                **{
                    column: _optional_int_value(row.get(column))
                    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS
                },
            }
        )

    ordered = sorted(
        out_rows,
        key=lambda row: (row["match_id"], row["side"], row["slot_in_side"]),
    )
    return pa.Table.from_pylist(ordered, schema=MATCH_PLAYERS_SCHEMA)


_PLAYER_COLUMNS = tuple(
    f"radiant_player_{i}_id" for i in range(_PLAYERS_PER_SIDE)
) + tuple(f"dire_player_{i}_id" for i in range(_PLAYERS_PER_SIDE))


def validate_matches_table(table: pa.Table, *, expected_row_count: int) -> None:
    """Validate `matches.parquet` invariants before publication.

    This checks *transformation correctness* (did the pivot/assembly
    above produce a well-formed table), not canonical validity -- the
    canonical layer (`CanonicalMatch`, Postgres constraints) already
    guarantees things like "exactly 5 picks per side" upstream; this
    function does not re-derive or re-enforce stricter domain rules than
    that.
    """
    if table.num_rows != expected_row_count:
        raise DatasetValidationError(
            f"matches.parquet row count {table.num_rows} != canonical "
            f"matches row count {expected_row_count}"
        )

    match_ids = table.column("match_id").to_pylist()
    if len(set(match_ids)) != len(match_ids):
        raise DatasetValidationError(
            "matches.parquet contains duplicate match_id values"
        )

    for column_name in _PLAYER_COLUMNS:
        if table.column(column_name).null_count > 0:
            raise DatasetValidationError(
                f"matches.parquet column {column_name!r} contains null player id(s)"
            )

    for column_name in ("radiant_team_id", "dire_team_id"):
        if table.column(column_name).null_count > 0:
            raise DatasetValidationError(
                f"matches.parquet column {column_name!r} contains null team id(s)"
            )


def validate_draft_events_table(
    draft_events_table: pa.Table, matches_table: pa.Table
) -> None:
    """Validate `draft_events.parquet` invariants before publication.

    Deliberately does NOT enforce a fixed event count or minimum ban
    count -- real canonical drafts vary (10/23/24-event drafts observed,
    including zero-ban matches; see module docstring).
    """
    match_ids = draft_events_table.column("match_id").to_pylist()
    sequences = draft_events_table.column("sequence").to_pylist()

    seen: set[tuple[int, int]] = set()
    for match_id, sequence in zip(match_ids, sequences, strict=True):
        key = (match_id, sequence)
        if key in seen:
            raise DatasetValidationError(
                f"draft_events.parquet duplicate (match_id, sequence) {key}"
            )
        seen.add(key)

    exported_match_ids = set(matches_table.column("match_id").to_pylist())
    orphaned = set(match_ids) - exported_match_ids
    if orphaned:
        raise DatasetValidationError(
            "draft_events.parquet references match_id(s) absent from "
            f"matches.parquet: {sorted(orphaned)[:10]}"
        )


def validate_match_players_table(
    match_players_table: pa.Table,
    matches_table: pa.Table,
    draft_events_table: pa.Table,
) -> None:
    """Validate `match_players.parquet` invariants before publication.

    Checks transformation correctness plus the player/hero contract:
    10 rows per match, 5 per side, non-null ids, unique players, unique
    heroes per side, `team_id` derived from the parent match side, and
    per-side hero set equal to the successful PICK set. Does not treat
    `slot_in_side` as Dota position 1-5. Observed `position`/`lane`/
    `role` may be null or UNKNOWN; duplicate/missing 1–5 assignments
    are not repaired here.
    """
    match_ids = match_players_table.column("match_id").to_pylist()
    sides = match_players_table.column("side").to_pylist()
    slots = match_players_table.column("slot_in_side").to_pylist()
    player_ids = match_players_table.column("player_id").to_pylist()
    hero_ids = match_players_table.column("hero_id").to_pylist()
    team_ids = match_players_table.column("team_id").to_pylist()

    exported_match_ids = set(matches_table.column("match_id").to_pylist())
    if match_players_table.num_rows != len(exported_match_ids) * _PLAYERS_PER_SIDE * 2:
        raise DatasetValidationError(
            f"match_players.parquet row count {match_players_table.num_rows} != "
            f"canonical matches {len(exported_match_ids)} × 10"
        )

    orphaned = set(match_ids) - exported_match_ids
    if orphaned:
        raise DatasetValidationError(
            "match_players.parquet references match_id(s) absent from "
            f"matches.parquet: {sorted(orphaned)[:10]}"
        )
    missing_matches = exported_match_ids - set(match_ids)
    if missing_matches:
        raise DatasetValidationError(
            "match_players.parquet missing match_id(s) present in "
            f"matches.parquet: {sorted(missing_matches)[:10]}"
        )

    for column_name in ("player_id", "hero_id", "team_id"):
        if match_players_table.column(column_name).null_count > 0:
            raise DatasetValidationError(
                f"match_players.parquet column {column_name!r} contains null(s)"
            )

    radiant_team = dict(
        zip(
            matches_table.column("match_id").to_pylist(),
            matches_table.column("radiant_team_id").to_pylist(),
            strict=True,
        )
    )
    dire_team = dict(
        zip(
            matches_table.column("match_id").to_pylist(),
            matches_table.column("dire_team_id").to_pylist(),
            strict=True,
        )
    )

    by_match_side: dict[tuple[int, str], list[tuple[int, int, int]]] = {}
    seen_player: set[tuple[int, int]] = set()
    seen_slot: set[tuple[int, str, int]] = set()
    for match_id, side, slot, player_id, hero_id, team_id in zip(
        match_ids, sides, slots, player_ids, hero_ids, team_ids, strict=True
    ):
        player_key = (match_id, player_id)
        if player_key in seen_player:
            raise DatasetValidationError(
                f"match_players.parquet duplicate player_id {player_id} "
                f"in match {match_id}"
            )
        seen_player.add(player_key)
        slot_key = (match_id, side, slot)
        if slot_key in seen_slot:
            raise DatasetValidationError(
                f"match_players.parquet duplicate ({side}, slot {slot}) "
                f"in match {match_id}"
            )
        seen_slot.add(slot_key)
        expected_team = (
            radiant_team[match_id] if side == "RADIANT" else dire_team[match_id]
        )
        if team_id != expected_team:
            raise DatasetValidationError(
                f"match {match_id}: match_players team_id {team_id} does not "
                f"match {side} team {expected_team}"
            )
        by_match_side.setdefault((match_id, side), []).append(
            (slot, player_id, hero_id)
        )

    pick_heroes: dict[tuple[int, str], set[int]] = {}
    draft_match_ids = draft_events_table.column("match_id").to_pylist()
    draft_actions = draft_events_table.column("action").to_pylist()
    draft_sides = draft_events_table.column("side").to_pylist()
    draft_hero_ids = draft_events_table.column("hero_id").to_pylist()
    draft_success = draft_events_table.column("was_successful").to_pylist()
    for match_id, action, side, hero_id, was_successful in zip(
        draft_match_ids,
        draft_actions,
        draft_sides,
        draft_hero_ids,
        draft_success,
        strict=True,
    ):
        if action != "PICK":
            continue
        if was_successful is False:
            continue
        pick_heroes.setdefault((match_id, side), set()).add(hero_id)

    for match_id in exported_match_ids:
        for side in _SIDES:
            rows = by_match_side.get((match_id, side), [])
            if len(rows) != _PLAYERS_PER_SIDE:
                raise DatasetValidationError(
                    f"match {match_id}: expected {_PLAYERS_PER_SIDE} {side} "
                    f"match_players rows, got {len(rows)}"
                )
            heroes = [hero_id for _slot, _player_id, hero_id in rows]
            if len(set(heroes)) != _PLAYERS_PER_SIDE:
                raise DatasetValidationError(
                    f"match {match_id}: {side} hero_ids are not 5 distinct values"
                )
            expected_picks = pick_heroes.get((match_id, side), set())
            if set(heroes) != expected_picks:
                raise DatasetValidationError(
                    f"match {match_id}: {side} player hero_id set does not "
                    "match successful PICK set"
                )


def _write_and_read_back(table: pa.Table, path: Path) -> None:
    """Write `table` to `path` and immediately read it back, raising
    `DatasetValidationError` if the round trip doesn't reproduce the same
    row count and schema. This is the "validate them" step between
    "write temporary files" and "atomically replace final files"."""
    pq.write_table(table, path)
    read_back = pq.read_table(path)
    if read_back.num_rows != table.num_rows:
        raise DatasetValidationError(
            f"{path.name}: round-trip row count {read_back.num_rows} != "
            f"expected {table.num_rows}"
        )
    if not read_back.schema.equals(table.schema):
        raise DatasetValidationError(f"{path.name}: round-trip schema mismatch")


def write_canonical_dataset(
    output_dir: Path,
    *,
    matches_table: pa.Table,
    draft_events_table: pa.Table,
    match_players_table: pa.Table | None = None,
) -> None:
    """Atomically publish the canonical analytical Parquet files under
    `output_dir`.

    Always writes `matches.parquet` and `draft_events.parquet`. When
    `match_players_table` is provided (the production `build_canonical_dataset`
    path and feature-layer fixtures), also writes `match_players.parquet`.
    Omitting it is only for tests that intentionally exercise a missing
    `match_players.parquet` file.

    Pattern: write each table to a temporary file inside `output_dir`,
    read it back to confirm it is well-formed, and only then replace the
    corresponding final file via `os.replace` (a same-filesystem rename,
    which POSIX guarantees is atomic -- the temp file lives in the same
    directory as the target specifically so this holds). Nothing under
    the temporary directory survives a successful call; on any failure,
    no final file has been touched yet, since every write+read-back
    happens before the first `os.replace`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pa.Table] = {
        MATCHES_FILENAME: matches_table,
        DRAFT_EVENTS_FILENAME: draft_events_table,
    }
    if match_players_table is not None:
        tables[MATCH_PLAYERS_FILENAME] = match_players_table

    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".build-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tmp_paths: dict[str, Path] = {}
        for filename, table in tables.items():
            tmp_path = tmp_dir / filename
            _write_and_read_back(table, tmp_path)
            tmp_paths[filename] = tmp_path

        for filename, tmp_path in tmp_paths.items():
            os.replace(tmp_path, output_dir / filename)


def _fetch_match_rows(conn: Connection) -> Sequence[Mapping[str, Any]]:
    return conn.execute(select(MATCHES).order_by(MATCHES.c.match_id)).mappings().all()


def _fetch_match_player_rows(conn: Connection) -> Sequence[Mapping[str, Any]]:
    return (
        conn.execute(
            select(MATCH_PLAYERS).order_by(
                MATCH_PLAYERS.c.match_id,
                MATCH_PLAYERS.c.side,
                MATCH_PLAYERS.c.slot_in_side,
            )
        )
        .mappings()
        .all()
    )


def _fetch_draft_event_rows(conn: Connection) -> Sequence[Mapping[str, Any]]:
    return (
        conn.execute(
            select(DRAFT_EVENTS).order_by(
                DRAFT_EVENTS.c.match_id, DRAFT_EVENTS.c.sequence
            )
        )
        .mappings()
        .all()
    )


def build_canonical_dataset(engine: Engine, output_dir: Path) -> DatasetBuildResult:
    """Full pipeline: read canonical Postgres tables, transform, validate,
    and atomically publish `matches.parquet` / `match_players.parquet` /
    `draft_events.parquet` under `output_dir`.

    Always a full, deterministic rebuild (see module docstring) -- there
    is no incremental update path. Reads only `matches`, `match_players`,
    `draft_events` via the existing `storage.schema` table objects (never
    raw SQL strings, never the ingestion/operational tables).
    """
    with engine.connect() as conn:
        match_rows = _fetch_match_rows(conn)
        match_player_rows = _fetch_match_player_rows(conn)
        draft_event_rows = _fetch_draft_event_rows(conn)

    matches_table = build_matches_table(match_rows, match_player_rows)
    match_players_table = build_match_players_table(match_rows, match_player_rows)
    draft_events_table = build_draft_events_table(draft_event_rows)

    validate_matches_table(matches_table, expected_row_count=len(match_rows))
    validate_draft_events_table(draft_events_table, matches_table)
    validate_match_players_table(match_players_table, matches_table, draft_events_table)

    write_canonical_dataset(
        output_dir,
        matches_table=matches_table,
        draft_events_table=draft_events_table,
        match_players_table=match_players_table,
    )

    return DatasetBuildResult(
        matches_row_count=matches_table.num_rows,
        match_players_row_count=match_players_table.num_rows,
        draft_events_row_count=draft_events_table.num_rows,
        output_dir=output_dir,
    )
