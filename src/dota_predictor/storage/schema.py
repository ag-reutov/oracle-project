"""PostgreSQL ingestion-state schema (SQLAlchemy Core table definitions).

This module defines what the STRATZ ingestion pipeline persists. It does
NOT implement fetching, pagination, retries, or concurrency -- see the
project's ingestion-state architecture plan for that sequencing. Four
categories of state are modeled here, each documented on its table(s):

1. League curation (`leagues`, `ingestion_leagues`) -- the manually
   curated Tier 1/Tier 2 registry decided in the STRATZ tier
   investigation, plus a strict allowlist that raw/canonical/progress
   tables are gated on.
2. Raw STRATZ landing (`stratz_raw_matches`) -- durable copy of fetched
   payloads, so reprocessing doesn't require re-fetching.
3. Canonical domain data (`teams`, `players`, `matches`, `match_players`,
   `draft_events`) -- the relational form of
   `canonical_schema.CanonicalMatch`/`DraftEvent`, plus minimal identity
   registries for the team/player ids they reference.
4. Ingestion progress/error bookkeeping (`league_ingestion_state`,
   `match_ingestion_errors`) -- the minimum state a future
   pagination/retry/concurrency layer needs.

Design choices (see the architecture plan for full rationale):

* `ingestion_leagues` is a strict allowlist, separate from `leagues`.
  `leagues` intentionally contains excluded/rejected entries too (for
  audit purposes), so a foreign key to `leagues` alone would not actually
  gate scope -- an excluded league would still be a valid FK target.
  Every raw/canonical/progress table instead references
  `ingestion_leagues`, which only ever contains in-scope leagues.
* Enum-like columns (`Side`, `DraftAction`, and free-standing
  status/stage/tier text columns) are backed by CHECK constraints rather
  than native Postgres `CREATE TYPE ... AS ENUM` types, so the allowed
  value set can change with a lightweight constraint migration instead of
  `ALTER TYPE ... ADD VALUE`.
* `match_players` and `draft_events` are child tables intended to be
  fully replaced (delete then insert) per match on every write, not
  upserted -- see `storage.writer.write_canonical_match`. A upsert would
  never remove rows that existed in a prior write but are absent from a
  reprocessed one (draft-event counts and match rosters can legitimately
  change across reprocessing).
* `mapper_version` on `matches` is an integer, not a string, so "does
  this row need reprocessing" is a plain `mapper_version < N` comparison.
* `teams`/`players` are minimal identity registries, not analytical
  entity tables: each holds only its primary key (`team_id`/`player_id`,
  both STRATZ source ids -- see `canonical_schema.TeamId`/`PlayerId`).
  They exist solely so `matches`/`match_players` can have real foreign
  keys instead of embedding team/player identity as bare integers. They
  are allowed to accumulate rows that are no longer referenced by any
  current `matches`/`match_players` row after reprocessing changes an id
  (e.g. a corrected `team_id` on reprocessing) -- there is deliberately
  no cleanup/deletion logic for orphaned identity rows, since Postgres
  only needs to guarantee referenced ids exist, not that every
  registered id is currently referenced. Referential integrity is
  therefore one-directional: every team/player referenced by
  `matches`/`match_players` must exist in `teams`/`players`, but the
  reverse does not need to hold.
* Derived, analysis-facing team/player attributes (a "latest observed
  display name", first/last-seen timestamps, etc.) are deliberately NOT
  stored here. `matches.radiant_team_name_observed` /
  `matches.dire_team_name_observed` already preserve what STRATZ
  reported for that specific match (immutable historical observations);
  any aggregate/derived view over those observations belongs to the
  future Postgres -> Parquet dataset build, computed fresh from current
  canonical facts each time, not cached as mutable Postgres state that
  reprocessing could leave stale.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from dota_predictor.data.canonical_schema import (
    DraftAction,
    MatchLane,
    MatchPlayerPosition,
    MatchPlayerRole,
    Side,
)

__all__ = [
    "DRAFT_EVENTS",
    "INGESTION_LEAGUES",
    "LEAGUES",
    "LEAGUE_FETCH_MODES",
    "LEAGUE_FETCH_MODE_LEAGUE",
    "LEAGUE_FETCH_MODE_MATCH_IDS",
    "LEAGUE_INGESTION_STATE",
    "LEAGUE_INGESTION_STATUSES",
    "LIQUIPEDIA_TIERS",
    "MATCHES",
    "MATCH_INGESTION_ERRORS",
    "MATCH_INGESTION_ERROR_STAGES",
    "MATCH_PLAYERS",
    "METADATA",
    "PLAYERS",
    "STRATZ_RAW_MATCHES",
    "TEAMS",
]

# Naming convention so Alembic autogenerate produces stable, predictable
# constraint/index names instead of dialect-default anonymous ones.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

METADATA = sa.MetaData(naming_convention=NAMING_CONVENTION)

# Allowed values for plain-text enum-like columns. Deliberately CHECK
# constraints, not native enum types (see module docstring) -- changing
# this list is a lightweight constraint migration, not a type migration.
LIQUIPEDIA_TIERS = ("T1", "T2", "MINOR", "QUALIFIER", "EXCLUDED")
LEAGUE_INGESTION_STATUSES = ("PENDING", "IN_PROGRESS", "COMPLETE", "ERROR")
MATCH_INGESTION_ERROR_STAGES = ("FETCH", "MAP", "WRITE")
LEAGUE_FETCH_MODE_LEAGUE = "league"
LEAGUE_FETCH_MODE_MATCH_IDS = "match_ids"
LEAGUE_FETCH_MODES = (LEAGUE_FETCH_MODE_LEAGUE, LEAGUE_FETCH_MODE_MATCH_IDS)

# `Side`/`DraftAction` are reused from canonical_schema.py (single source
# of truth for the Python enum) but mapped as non-native SQLAlchemy enums:
# on Postgres this compiles to VARCHAR + CHECK, not CREATE TYPE ... AS ENUM.
_SIDE_TYPE = sa.Enum(
    Side,
    name="side",
    native_enum=False,
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)
_DRAFT_ACTION_TYPE = sa.Enum(
    DraftAction,
    name="draft_action",
    native_enum=False,
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)
_POSITION_TYPE = sa.Enum(
    MatchPlayerPosition,
    name="match_player_position",
    native_enum=False,
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)
_LANE_TYPE = sa.Enum(
    MatchLane,
    name="match_lane",
    native_enum=False,
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)
_ROLE_TYPE = sa.Enum(
    MatchPlayerRole,
    name="match_player_role",
    native_enum=False,
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)

# --- 1. League curation -----------------------------------------------

LEAGUES = sa.Table(
    "leagues",
    METADATA,
    sa.Column("league_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("name", sa.Text, nullable=False),
    # Raw STRATZ `LeagueTier` value. Cross-check signal only -- see the
    # tier investigation for why it cannot be the scoping authority.
    sa.Column("stratz_tier", sa.Text, nullable=True),
    sa.Column("liquipedia_tier", sa.Text, nullable=False),
    sa.Column("in_scope", sa.Boolean, nullable=False, server_default=sa.false()),
    # How allowlisted leagues are fetched. Independent of `in_scope`.
    # `league` pages `league(id) { matches }`; `match_ids` uses STRATZ
    # `match(id)` after ID discovery. Default preserves historical behavior.
    sa.Column(
        "fetch_mode",
        sa.Text,
        nullable=False,
        server_default=sa.text(f"'{LEAGUE_FETCH_MODE_LEAGUE}'"),
    ),
    sa.Column("notes", sa.Text, nullable=True),
    sa.Column("source", sa.Text, nullable=True),
    sa.Column("start_date", sa.Date, nullable=True),
    sa.Column("end_date", sa.Date, nullable=True),
    sa.Column(
        "curated_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.CheckConstraint(
        f"liquipedia_tier IN {LIQUIPEDIA_TIERS!r}",
        name="liquipedia_tier_valid",
    ),
    sa.CheckConstraint(
        f"fetch_mode IN {LEAGUE_FETCH_MODES!r}",
        name="fetch_mode_valid",
    ),
)

INGESTION_LEAGUES = sa.Table(
    "ingestion_leagues",
    METADATA,
    sa.Column(
        "league_id",
        sa.BigInteger,
        sa.ForeignKey("leagues.league_id", ondelete="RESTRICT"),
        primary_key=True,
        autoincrement=False,
    ),
    sa.Column(
        "added_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
)

# --- 2. Raw STRATZ landing ----------------------------------------------

STRATZ_RAW_MATCHES = sa.Table(
    "stratz_raw_matches",
    METADATA,
    sa.Column("match_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column(
        "league_id",
        sa.BigInteger,
        sa.ForeignKey("ingestion_leagues.league_id"),
        nullable=False,
    ),
    sa.Column("payload", JSONB, nullable=False),
    sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
)

# --- 3. Canonical domain data --------------------------------------------

TEAMS = sa.Table(
    "teams",
    METADATA,
    sa.Column("team_id", sa.BigInteger, primary_key=True, autoincrement=False),
)

PLAYERS = sa.Table(
    "players",
    METADATA,
    sa.Column("player_id", sa.BigInteger, primary_key=True, autoincrement=False),
)

MATCHES = sa.Table(
    "matches",
    METADATA,
    sa.Column("match_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column(
        "league_id",
        sa.BigInteger,
        sa.ForeignKey("ingestion_leagues.league_id"),
        nullable=False,
    ),
    sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("league_name", sa.Text, nullable=True),
    sa.Column("series_id", sa.BigInteger, nullable=True),
    sa.Column("series_type", sa.Text, nullable=True),
    sa.Column("game_number_in_series", sa.SmallInteger, nullable=True),
    sa.Column("game_version_id", sa.Integer, nullable=True),
    sa.Column(
        "radiant_team_id",
        sa.BigInteger,
        sa.ForeignKey("teams.team_id"),
        nullable=False,
    ),
    # What STRATZ reported as this team's name *for this specific match*
    # -- an immutable historical observation, not the team's current/best-
    # known display name (that is an entity-level, derived concern; see
    # module docstring and `teams` above). Deliberately never overwritten
    # or backfilled from other matches.
    sa.Column("radiant_team_name_observed", sa.Text, nullable=True),
    sa.Column(
        "dire_team_id",
        sa.BigInteger,
        sa.ForeignKey("teams.team_id"),
        nullable=False,
    ),
    sa.Column("dire_team_name_observed", sa.Text, nullable=True),
    sa.Column("radiant_win", sa.Boolean, nullable=False),
    sa.Column("duration_seconds", sa.Integer, nullable=False),
    # Monotonic version of `stratz_mapping.CANONICAL_MAPPER_VERSION` that
    # produced this row -- an integer so "needs reprocessing" is a plain
    # `mapper_version < N` comparison, not a string/semver parse.
    sa.Column("mapper_version", sa.Integer, nullable=False),
    sa.Column("canonicalized_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(
        "game_number_in_series IS NULL OR game_number_in_series >= 1",
        name="game_number_in_series_positive",
    ),
    sa.CheckConstraint(
        "radiant_team_id <> dire_team_id",
        name="radiant_dire_team_distinct",
    ),
    sa.CheckConstraint("duration_seconds > 0", name="duration_seconds_positive"),
)

MATCH_PLAYERS = sa.Table(
    "match_players",
    METADATA,
    sa.Column(
        "match_id",
        sa.BigInteger,
        sa.ForeignKey("matches.match_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("side", _SIDE_TYPE, primary_key=True),
    # Position within the canonical per-side player tuple (0-4), not
    # Dota's raw playerSlot encoding -- see canonical_schema.py.
    sa.Column("slot_in_side", sa.SmallInteger, primary_key=True),
    sa.Column(
        "player_id",
        sa.BigInteger,
        sa.ForeignKey("players.player_id"),
        nullable=False,
    ),
    # Played hero for this player in this match (STRATZ `players[].heroId`).
    # Aligned with `slot_in_side` (lobby order), not draft pick order and
    # not Dota position 1-5.
    sa.Column("hero_id", sa.Integer, nullable=False),
    # Observed STRATZ match-player parse labels. NULL = source omitted
    # the field; UNKNOWN is stored when STRATZ returns UNKNOWN. Never
    # inferred from slot_in_side. POST_MATCH relative to this match.
    sa.Column("position", _POSITION_TYPE, nullable=True),
    sa.Column("lane", _LANE_TYPE, nullable=True),
    sa.Column("role", _ROLE_TYPE, nullable=True),
    # Observed STRATZ post-match box-score scalars. NULL = source
    # omitted the field; 0 is stored when STRATZ returns 0. Never
    # coerced, ratioed, or treated as a feature. POST_MATCH relative
    # to this match.
    sa.Column("kills", sa.Integer, nullable=True),
    sa.Column("deaths", sa.Integer, nullable=True),
    sa.Column("assists", sa.Integer, nullable=True),
    sa.Column("gold_per_minute", sa.Integer, nullable=True),
    sa.Column("experience_per_minute", sa.Integer, nullable=True),
    sa.Column("num_last_hits", sa.Integer, nullable=True),
    sa.Column("num_denies", sa.Integer, nullable=True),
    sa.Column("networth", sa.Integer, nullable=True),
    sa.Column("hero_damage", sa.Integer, nullable=True),
    sa.Column("tower_damage", sa.Integer, nullable=True),
    sa.Column("hero_healing", sa.Integer, nullable=True),
    sa.Column("level", sa.Integer, nullable=True),
    sa.CheckConstraint("slot_in_side BETWEEN 0 AND 4", name="slot_in_side_valid_range"),
    sa.CheckConstraint("hero_id > 0", name="hero_id_positive"),
    sa.UniqueConstraint("match_id", "player_id", name="match_players_unique_player"),
)

DRAFT_EVENTS = sa.Table(
    "draft_events",
    METADATA,
    sa.Column(
        "match_id",
        sa.BigInteger,
        sa.ForeignKey("matches.match_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    # No fixed upper bound: draft length is not a historical invariant
    # (22 vs 24 events observed across patches; see canonical_schema.py).
    sa.Column("sequence", sa.SmallInteger, primary_key=True),
    sa.Column("action", _DRAFT_ACTION_TYPE, nullable=False),
    sa.Column("side", _SIDE_TYPE, nullable=False),
    sa.Column("hero_id", sa.Integer, nullable=False),
    sa.Column("was_successful", sa.Boolean, nullable=True),
    sa.CheckConstraint("sequence >= 0", name="sequence_non_negative"),
    sa.CheckConstraint("hero_id > 0", name="hero_id_positive"),
)

# --- 4. Ingestion progress/error bookkeeping -----------------------------

LEAGUE_INGESTION_STATE = sa.Table(
    "league_ingestion_state",
    METADATA,
    sa.Column(
        "league_id",
        sa.BigInteger,
        sa.ForeignKey("ingestion_leagues.league_id"),
        primary_key=True,
        autoincrement=False,
    ),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column(
        "matches_seen_count", sa.Integer, nullable=False, server_default=sa.text("0")
    ),
    # Deliberately opaque/forward-compatible: the future pagination
    # implementation decides its own shape (offset, last match id, last
    # timestamp, ...) here without another migration.
    sa.Column("cursor_state", JSONB, nullable=True),
    sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("error_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("last_error", sa.Text, nullable=True),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(f"status IN {LEAGUE_INGESTION_STATUSES!r}", name="status_valid"),
    sa.CheckConstraint(
        "matches_seen_count >= 0", name="matches_seen_count_non_negative"
    ),
    sa.CheckConstraint("error_count >= 0", name="error_count_non_negative"),
)

MATCH_INGESTION_ERRORS = sa.Table(
    "match_ingestion_errors",
    METADATA,
    sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
    sa.Column("match_id", sa.BigInteger, nullable=False),
    sa.Column(
        "league_id",
        sa.BigInteger,
        sa.ForeignKey("ingestion_leagues.league_id"),
        nullable=True,
    ),
    sa.Column("stage", sa.Text, nullable=False),
    sa.Column("error_message", sa.Text, nullable=False),
    sa.Column("raw_payload_snapshot", JSONB, nullable=True),
    sa.Column(
        "occurred_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column("resolved", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.CheckConstraint(
        f"stage IN {MATCH_INGESTION_ERROR_STAGES!r}", name="stage_valid"
    ),
)
