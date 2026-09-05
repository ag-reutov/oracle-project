"""PostgreSQL ingestion-state schema (SQLAlchemy Core table definitions).

This module defines what the STRATZ ingestion pipeline persists. It does
NOT implement fetching, pagination, retries, or concurrency -- see the
project's ingestion-state architecture plan for that sequencing. Four
categories of state are modeled here, each documented on its table(s):

1. League curation (`leagues`, `ingestion_leagues`) -- the manually
   curated Liquipedia T1/T2/T3 registry, plus a strict allowlist that
   raw/canonical/progress tables are gated on. Tier labels stay distinct
   so later experiments can train T1/T2 vs T1/T2+T3 without collapsing
   cohorts.
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
 *   display name", first/last-seen timestamps, etc.) are deliberately NOT
 *   stored here. `matches.radiant_team_name_observed` /
 *   `matches.dire_team_name_observed` already preserve what STRATZ
 *   reported for that specific match (immutable historical observations);
 *   any aggregate/derived view over those observations belongs to the
 *   future Postgres -> Parquet dataset build, computed fresh from current
 *   canonical facts each time, not cached as mutable Postgres state that
 *   reprocessing could leave stale.
 *
 * The explicit team-identity layer (Slice 1) extends the same principle:
 *
 * * `teams` stays a minimal identity registry (primary key only). The
 *   "source/display name", team `tag`, and first/last-seen timestamps
 *   are derived observations, not mutable canonical state -- they live in
 *   the derived `team_aliases` / `team_tags` tables, which are fully
 *   rebuilt by an idempotent deterministic backfill
 *   (`scripts/backfill_team_identity.py`) from canonical match
 *   observations / raw STRATZ payloads, so they can never go stale.
 * * `team_aliases` records every name observed for a source `team_id`
 *   and the period it was observed (`first_seen_at` / `last_seen_at`) --
 *   derived/indexed identity information, not a rewrite of historical
 *   `*_team_name_observed` match facts (which remain the source of
 *   truth). One (team_id, name) pair per row, so a future rename adds a
 *   row instead of rewriting history.
 * * `team_tags` does the same for the STRATZ team `tag`, backfilled from
 *   existing raw payloads (no re-fetching). Tags are treated as
 *   historical observations like names, never as one eternal value.
 * * `organizations` + `team_organization_memberships` are the explicit,
 *   curated organization-identity layer: raw STRATZ `team_id` -> optional
 *   organization. Mappings come only from `config/team_organizations.yaml`
 *   via `scripts/load_team_organizations.py`. Name equality alone never
 *   merges teams; absence of a membership row means "unmapped", which is
 *   normal and valid. `team_organization_memberships.team_id` is the PK,
 *   so one raw team id maps to at most one organization, and provenance
 *   (`reason`, `source`) records why a mapping was curated.
 *
 * The explicit player-identity layer (Slice 2) extends the same principle:
 *
 * * `players` stays a minimal identity registry (primary key only --
 *   the canonical STRATZ `steamAccountId`). The derived player universe
 *   (`display_name`, `first_seen_at`, `last_seen_at`, `match_count`) is
 *   NOT stored here: it is a pure aggregate over the immutable canonical
 *   facts (`match_players` joined to `matches.start_time`), computed
 *   fresh via the `research.players` view / `fetch_player_universe`
 *   (see `data.player_identity`). Caching it as mutable Postgres state
 *   would go stale on reprocessing, exactly the failure mode this
 *   registry avoids by design.
 * * No `current_team_id` / `position` (or any other time-varying or
 *   analytical attribute) column is ever added to `players`: a canonical
 *   player entity represents identity only, so historical research can
 *   never leak future team/position/rating state.
 * * The `players` registry is populated on ingest by the writer
 *   (`INSERT ... ON CONFLICT DO NOTHING`) and re-asserted by
 *   `scripts/backfill_player_identity.py` (idempotent; never deletes
 *   orphan registry rows). `match_players.player_id` is a foreign key to
 *   `players`, so every referenced id resolves to exactly one canonical
 *   player. The local corpus contains no player-name observations (raw
 *   STRATZ payloads carry only `steamAccountId`), so `display_name` is
 *   NULL for every player today; the deterministic name-resolution rule
 *   lives in `data.player_identity` for when names become available.
 *   Orphan registry ids are reported (not hidden) by
 *   `scripts/audit_player_identity.py`.
 *
 * The canonical reference-entity layer (Slice 3) follows the same rule
 * but deliberately adds NO new tables here:
 *
 * * Heroes and game versions are STRATZ constants catalogs that stay
 *   Parquet reference files (`heroes.parquet`, `game_versions.parquet`),
 *   built by `scripts/build_reference_dataset.py` and consumed via the
 *   DuckDB `register_reference_views` layer -- not Postgres state. Since
 *   `REFERENCE_SCHEMA_VERSION` v2 they carry provenance (`source`,
 *   `retrieved_at`) and, for heroes, the STRATZ-supplied `short_name`
 *   and `aliases`. See `datasets.reference_export` and
 *   `data.reference_identity`.
 * * Leagues are the curated `leagues` registry (already documented
 *   above); `research.leagues` (a plain view, no new storage) exposes
 *   the canonical league identity with the source-vs-curated tier
 *   distinction and provenance. `matches.league_id` / `match_players.hero_id`
 *   / `matches.game_version_id` resolve through exactly one canonical
 *   reference path each; `data.reference_identity` provides the typed
 *   Python accessors and the census audit
 *   (`scripts/audit_reference_entities.py`).
 * * Regions were investigated for Slice 3: STRATZ exposes server regions
 *   (`constants.regions`) but no canonical entity references a region id
 *   and there is no clean team/event-region source, so a region entity is
 *   intentionally deferred (reported by the reference-entity audit).
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
    "MATCH_CLASSIFICATIONS",
    "MATCH_INGESTION_ERRORS",
    "MATCH_INGESTION_ERROR_STAGES",
    "MATCH_PLAYERS",
    "METADATA",
    "ORGANIZATIONS",
    "PLAYERS",
    "RESEARCH_METADATA",
    "STRATZ_RAW_MATCHES",
    "TEAMS",
    "TEAM_ALIASES",
    "TEAM_ORGANIZATION_MEMBERSHIPS",
    "TEAM_STRENGTH_BUILD",
    "TEAM_STRENGTH_STATE",
    "TEAM_TAGS",
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

# Separate metadata for the derived `research`-schema tables. The public
# `METADATA` drives `METADATA.create_all`/`drop_all` in test fixtures that
# never touch the `research` schema; keeping the Slice 6 derived tables in
# their own metadata (with `schema="research"`) means those fixtures are not
# forced to pre-create the research schema. The tables are actually created
# by `dota_predictor.research.views.create_research_layer` / the Slice 6
# migration, which both emit the schema.
RESEARCH_METADATA = sa.MetaData(naming_convention=NAMING_CONVENTION)

# Allowed values for plain-text enum-like columns. Deliberately CHECK
# constraints, not native enum types (see module docstring) -- changing
# this list is a lightweight constraint migration, not a type migration.
LIQUIPEDIA_TIERS = ("T1", "T2", "T3", "MINOR", "QUALIFIER", "EXCLUDED")
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
    # When true (with `start_date`/`end_date` set and `fetch_mode` =
    # `match_ids`), match-ID discovery and ingest are restricted to the
    # league's date window. Used for catalog-null leagues whose STRATZ
    # league also contains qualifiers outside the Liquipedia main-event
    # window (so qualifiers sharing the league id are not ingested).
    sa.Column(
        "window_filter",
        sa.Boolean,
        nullable=False,
        server_default=sa.false(),
    ),
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

# --- Team identity layer (Slice 1) ---------------------------------------
# Derived/indexed identity information over the raw `teams` registry.
# Historical match facts (`matches.*_team_name_observed`) and the raw
# STRATZ payloads (`stratz_raw_matches.payload`) remain the source of
# truth; these tables are fully rebuilt by idempotent deterministic
# backfill scripts and never feed back into `matches`. See module
# docstring for the identity-layer design decisions.

TEAM_ALIASES = sa.Table(
    "team_aliases",
    METADATA,
    sa.Column(
        "team_id",
        sa.BigInteger,
        sa.ForeignKey("teams.team_id"),
        primary_key=True,
        autoincrement=False,
    ),
    # A name observed for this source team in one or more canonical
    # matches. One row per (team_id, name): a future rename adds a new
    # row instead of rewriting historical `*_team_name_observed` facts.
    sa.Column("name", sa.Text, primary_key=True),
    sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("observation_count", sa.Integer, nullable=False),
    sa.CheckConstraint(
        "observation_count > 0", name="observation_count_positive"
    ),
)

TEAM_TAGS = sa.Table(
    "team_tags",
    METADATA,
    sa.Column(
        "team_id",
        sa.BigInteger,
        sa.ForeignKey("teams.team_id"),
        primary_key=True,
        autoincrement=False,
    ),
    # A STRATZ `tag` observed for this source team in a raw payload.
    # Treated as a historical observation like names, never as one
    # eternal value. `tag` is nullable on raw payloads, so coverage is
    # expected to be partial.
    sa.Column("tag", sa.Text, primary_key=True),
    sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("observation_count", sa.Integer, nullable=False),
    sa.CheckConstraint(
        "observation_count > 0", name="observation_count_positive"
    ),
)

ORGANIZATIONS = sa.Table(
    "organizations",
    METADATA,
    # Curated organization-level identity. `organization_id` is assigned
    # explicitly in config (never derived from any team id).
    sa.Column(
        "organization_id", sa.BigInteger, primary_key=True, autoincrement=False
    ),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("notes", sa.Text, nullable=True),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.UniqueConstraint("name", name="uq_organizations_name"),
)

TEAM_ORGANIZATION_MEMBERSHIPS = sa.Table(
    "team_organization_memberships",
    METADATA,
    # One raw STRATZ `team_id` maps to at most one organization (the PK
    # enforces this). Absence of a row means "unmapped", which is normal
    # and valid -- there is no auto-merging by name or roster overlap.
    sa.Column(
        "team_id",
        sa.BigInteger,
        sa.ForeignKey("teams.team_id"),
        primary_key=True,
        autoincrement=False,
    ),
    sa.Column(
        "organization_id",
        sa.BigInteger,
        sa.ForeignKey("organizations.organization_id"),
        nullable=False,
    ),
    # Provenance for the curated mapping: why the team ids were grouped
    # and where the curation decision came from. Explicit, auditable,
    # never inferred from name equality alone.
    sa.Column("reason", sa.Text, nullable=True),
    sa.Column("source", sa.Text, nullable=True),
    sa.Column(
        "added_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
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
    # Whether `draft_events` holds a complete source draft. False when the
    # source provided no usable full draft (STRATZ `pickBans` null/empty or
    # malformed); the match is still canonical because identity/result
    # facts exist. Draft-dependent features/queries exclude rows with
    # `draft_complete = false`.
    sa.Column(
        "draft_complete",
        sa.Boolean,
        nullable=False,
        server_default=sa.true(),
    ),
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

MATCH_CLASSIFICATIONS = sa.Table(
    "match_classifications",
    METADATA,
    # One row per match that belongs to a different Liquipedia event/tier
    # than its league's default classification (see
    # config/event_match_assignments.yaml). A match's effective tier is
    # `coalesce(match_classifications.liquipedia_tier, leagues.liquipedia_tier)`.
    sa.Column(
        "match_id",
        sa.BigInteger,
        sa.ForeignKey("matches.match_id", ondelete="CASCADE"),
        primary_key=True,
        autoincrement=False,
    ),
    sa.Column("liquipedia_event", sa.Text, nullable=False),
    sa.Column("liquipedia_tier", sa.Text, nullable=False),
    sa.Column("source", sa.Text, nullable=True),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.CheckConstraint(
        f"liquipedia_tier IN {LIQUIPEDIA_TIERS!r}",
        name="liquipedia_tier_valid",
    ),
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


# --- 5. Team-strength derived research state (Slice 6) --------------------
# Persisted deterministic derived state over the canonical `matches` facts.
# `matches` remains the sole source of truth; these tables are idempotently
# rebuilt by `dota_predictor.data.team_strength.rebuild_team_strength_state`
# and are never written by the ingestion pipeline. They live in the
# `research` schema (see the Slice 6 migration / research.views) and are
# referenced by the `research.raw_team_elo_latest` view. There is
# deliberately no mutable "current Elo" column on `teams`. They are declared
# in `RESEARCH_METADATA`, not the public `METADATA`, because the public
# metadata is also used by fixtures that never create the `research` schema.

TEAM_STRENGTH_STATE = sa.Table(
    "team_strength_state",
    RESEARCH_METADATA,
    # One row per (team_id, match_id): the team's strength entering that
    # match (`elo_pre`) plus strictly-prior descriptive record.
    sa.Column("match_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("team_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("side", sa.Text, nullable=False),
    sa.Column("team_name_observed", sa.Text, nullable=True),
    sa.Column("elo_pre", sa.Double, nullable=False),
    sa.Column("elo_post", sa.Double, nullable=False),
    sa.Column("won", sa.Boolean, nullable=False),
    sa.Column("prior_match_count", sa.Integer, nullable=False),
    sa.Column("prior_win_count", sa.Integer, nullable=False),
    sa.Column("prior_loss_count", sa.Integer, nullable=False),
    sa.Column("prior_win_rate", sa.Double, nullable=True),
    sa.Column("previous_match_id", sa.BigInteger, nullable=True),
    sa.Column("previous_match_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("days_since_previous_match", sa.Double, nullable=True),
    sa.Column("is_first_observed_match", sa.Boolean, nullable=False),
    sa.CheckConstraint("elo_pre >= 0", name="elo_pre_non_negative"),
    sa.CheckConstraint("elo_post >= 0", name="elo_post_non_negative"),
    sa.CheckConstraint(
        "prior_match_count >= 0", name="prior_match_count_non_negative"
    ),
    sa.CheckConstraint("prior_win_count >= 0", name="prior_win_count_non_negative"),
    sa.CheckConstraint("prior_loss_count >= 0", name="prior_loss_count_non_negative"),
    schema="research",
)

TEAM_STRENGTH_BUILD = sa.Table(
    "team_strength_build",
    RESEARCH_METADATA,
    # Single-row provenance / staleness metadata for the derived table.
    sa.Column("id", sa.Integer, sa.Identity(always=True), primary_key=True),
    sa.Column("built_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("source_match_count", sa.BigInteger, nullable=False),
    sa.Column("source_skipped_matches", sa.BigInteger, nullable=False),
    sa.Column("source_min_start_time", sa.DateTime(timezone=True), nullable=True),
    sa.Column("source_max_start_time", sa.DateTime(timezone=True), nullable=True),
    sa.Column("source_fingerprint", sa.Text, nullable=False),
    sa.Column("rows_written", sa.BigInteger, nullable=False),
    sa.Column("elo_initial_rating", sa.Double, nullable=False),
    sa.Column("elo_k_factor", sa.Double, nullable=False),
    schema="research",
)
