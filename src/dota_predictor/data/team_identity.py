"""Derived team-identity layer (Slice 1).

This module implements the explicit team-identity distinction the Slice 1
spec requires, on top of the existing canonical warehouse:

* **Source team identity** = raw STRATZ `team_id` (the `teams` registry
  primary key). Never reinterpreted as an organization id.
* **Observed team name** = the name STRATZ reported for a specific match
  (`matches.*_team_name_observed`), which is an immutable historical fact.
* **Alias/history** = the derived, indexed set of names (and their
  observation periods) for each source team, in `team_aliases`.
* **Organization identity** = a separate, explicit, curated layer
  (`organizations` + `team_organization_memberships`) that may group
  multiple raw `team_id`s under one real-world organization.

Design decisions (documented here per the Slice 1 spec):

* Name equality alone NEVER merges source teams. Two `team_id`s sharing a
  name remain two raw teams and produce two `team_aliases` rows; grouping
  into an organization happens only through an explicit curated mapping
  (`config/team_organizations.yaml` -> `scripts/load_team_organizations.py`).
* Historical match facts are never rewritten. `derive_team_aliases` reads
  only the existing `*_team_name_observed` columns and never writes back
  to `matches`.
* Team `tag` is backfilled from existing raw STRATZ payloads (no
  re-fetching) and treated as a historical observation like names, so a
  tag that varies over time produces multiple `team_tags` rows rather
  than one silently-assumed eternal value. Tag coverage is expected to be
  partial (raw payloads may omit `tag`).
* Backfill is deterministic and idempotent: re-running it recomputes the
  same rows (upsert, never insert-and-accumulate), so it can be run after
  any future ingest without creating duplicates.

The pure derivation functions take plain observation tuples and are
testable without a database; the `sync_*` helpers write to Postgres via
`ON CONFLICT` upserts and are used by the CLI scripts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import Connection, Engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dota_predictor.storage.schema import (
    MATCHES,
    ORGANIZATIONS,
    STRATZ_RAW_MATCHES,
    TEAM_ALIASES,
    TEAM_ORGANIZATION_MEMBERSHIPS,
    TEAM_TAGS,
    TEAMS,
)

__all__ = [
    "TeamAlias",
    "TeamTag",
    "collect_team_alias_observations",
    "collect_team_tag_observations",
    "derive_team_aliases",
    "derive_team_tags",
    "list_team_ids_in_registry",
    "load_team_organizations_config",
    "run_backfill",
    "sync_team_aliases",
    "sync_team_organizations",
    "sync_team_tags",
]

_RAW_TEAM_KEYS = ("radiantTeam", "direTeam")


@dataclass(frozen=True, slots=True)
class TeamAlias:
    """One name observed for a source team, plus its observation period.

    `first_seen_at`/`last_seen_at` are the earliest/latest match start
    times in which this name was observed for this `team_id`;
    `observation_count` is the number of match-sides (radiant + dire)
    carrying that observation. Deterministically derived from
    `matches.*_team_name_observed`.
    """

    team_id: int
    name: str
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int


@dataclass(frozen=True, slots=True)
class TeamTag:
    """One STRATZ `tag` observed for a source team, plus its period.

    Backfilled from existing raw STRATZ payloads. Tag coverage is
    partial: a raw payload may omit `tag` for one or both teams.
    """

    team_id: int
    tag: str
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int


def _group_observations(
    observations: Iterable[tuple[int, str, datetime]],
) -> list[tuple[int, str, list[datetime]]]:
    """Group `(team_id, value, observed_at)` by (team_id, value).

    Returns rows sorted deterministically by (team_id, value). Callers
    filter out None values before calling so a missing observation never
    produces a row.
    """
    by_key: dict[tuple[int, str], list[datetime]] = {}
    for team_id, value, observed_at in observations:
        by_key.setdefault((team_id, value), []).append(observed_at)
    return [
        (team_id, value, times)
        for (team_id, value), times in sorted(by_key.items())
    ]


def derive_team_aliases(
    observations: Iterable[tuple[int, str | None, datetime]],
) -> list[TeamAlias]:
    """Derive deterministic `team_aliases` rows from match observations.

    Each observation is `(team_id, name, start_time)`. Observations with a
    `None` name are ignored (a missing observed name is not an alias). One
    row per (team_id, name); a future rename therefore produces an
    additional row rather than rewriting history. Output is sorted by
    (team_id, name) regardless of input order.
    """
    return [
        TeamAlias(
            team_id=team_id,
            name=name,
            first_seen_at=min(times),
            last_seen_at=max(times),
            observation_count=len(times),
        )
        for team_id, name, times in _group_observations(
            (int(team_id), str(name), observed_at)
            for team_id, name, observed_at in observations
            if name is not None
        )
    ]


def derive_team_tags(
    observations: Iterable[tuple[int, str | None, datetime]],
) -> list[TeamTag]:
    """Derive deterministic `team_tags` rows from raw payload observations.

    Each observation is `(team_id, tag, observed_at)` where `observed_at`
    is the match's start time (falling back to `fetched_at` when a payload
    lacks one). `None` tags are ignored -- missing tags are not recorded,
    which is what makes coverage partial. Output is sorted by (team_id,
    tag) regardless of input order.
    """
    return [
        TeamTag(
            team_id=team_id,
            tag=tag,
            first_seen_at=min(times),
            last_seen_at=max(times),
            observation_count=len(times),
        )
        for team_id, tag, times in _group_observations(
            (int(team_id), str(tag), observed_at)
            for team_id, tag, observed_at in observations
            if tag is not None
        )
    ]


def _upsert_rows(
    conn: Connection,
    table: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    conflict_columns: Sequence[Any],
    update_columns: Sequence[str],
) -> None:
    if not rows:
        return
    stmt = pg_insert(table).values([dict(row) for row in rows])
    conn.execute(
        stmt.on_conflict_do_update(
            index_elements=list(conflict_columns),
            set_={col: getattr(stmt.excluded, col) for col in update_columns},
        )
    )


def sync_team_aliases(conn: Connection, aliases: Sequence[TeamAlias]) -> None:
    """Upsert derived alias rows into `team_aliases` (idempotent).

    Re-running with the same observations produces the same rows, never
    duplicates.
    """
    _upsert_rows(
        conn,
        TEAM_ALIASES,
        [
            {
                "team_id": alias.team_id,
                "name": alias.name,
                "first_seen_at": alias.first_seen_at,
                "last_seen_at": alias.last_seen_at,
                "observation_count": alias.observation_count,
            }
            for alias in aliases
        ],
        conflict_columns=[TEAM_ALIASES.c.team_id, TEAM_ALIASES.c.name],
        update_columns=[
            "first_seen_at",
            "last_seen_at",
            "observation_count",
        ],
    )


def sync_team_tags(conn: Connection, tags: Sequence[TeamTag]) -> None:
    """Upsert derived tag rows into `team_tags` (idempotent)."""
    _upsert_rows(
        conn,
        TEAM_TAGS,
        [
            {
                "team_id": tag.team_id,
                "tag": tag.tag,
                "first_seen_at": tag.first_seen_at,
                "last_seen_at": tag.last_seen_at,
                "observation_count": tag.observation_count,
            }
            for tag in tags
        ],
        conflict_columns=[TEAM_TAGS.c.team_id, TEAM_TAGS.c.tag],
        update_columns=["first_seen_at", "last_seen_at", "observation_count"],
    )


def list_team_ids_in_registry(conn: Connection) -> set[int]:
    """Return the set of `team_id`s currently in the `teams` registry."""
    return {
        int(row.team_id)
        for row in conn.execute(select(TEAMS.c.team_id)).all()
    }


def load_team_organizations_config(config_path: Path) -> list[dict]:
    """Load and validate `config/team_organizations.yaml`.

    Each entry requires an explicit curated `organization_id`, a `name`,
    and a non-empty, de-duplicated list of raw STRATZ `team_id`s. This
    validation fails loudly on curation errors rather than silently
    accepting a malformed mapping.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    entries = list(raw.get("organizations") or [])
    seen_org_ids: set[int] = set()
    seen_team_ids: set[int] = set()
    for entry in entries:
        organization_id = entry.get("organization_id")
        if organization_id is None:
            raise ValueError(f"{config_path}: organization missing organization_id")
        organization_id = int(organization_id)
        if organization_id in seen_org_ids:
            raise ValueError(
                f"{config_path}: duplicate organization_id {organization_id}"
            )
        seen_org_ids.add(organization_id)
        if not entry.get("name"):
            raise ValueError(
                f"{config_path}: organization_id {organization_id} missing name"
            )
        team_ids = entry.get("team_ids")
        if not team_ids:
            raise ValueError(
                f"{config_path}: organization_id {organization_id} has no team_ids"
            )
        normalized_team_ids: list[int] = []
        for team_id in team_ids:
            if not isinstance(team_id, int) or isinstance(team_id, bool) or team_id <= 0:
                raise ValueError(
                    f"{config_path}: organization_id {organization_id}: "
                    f"invalid team_id {team_id!r}"
                )
            if team_id in seen_team_ids:
                raise ValueError(
                    f"{config_path}: team_id {team_id} appears in more than one "
                    "organization"
                )
            seen_team_ids.add(team_id)
            normalized_team_ids.append(team_id)
        entry["organization_id"] = organization_id
        entry["team_ids"] = normalized_team_ids
    return entries


def sync_team_organizations(
    conn: Connection, entries: Sequence[Mapping[str, Any]]
) -> None:
    """Converge `organizations` + `team_organization_memberships` to config.

    Upserts every configured organization and membership (so re-running
    is idempotent) and removes memberships whose raw `team_id` is no
    longer present in the config. Organizations are never deleted --
    like `leagues`, they stay in the registry for audit even when they
    momentarily have no members. Every configured `team_id` must already
    exist in the `teams` registry; a stale curation entry fails loudly.
    """
    registry = list_team_ids_in_registry(conn)
    configured_team_ids: set[int] = set()

    for entry in entries:
        org_stmt = pg_insert(ORGANIZATIONS).values(
            organization_id=entry["organization_id"],
            name=entry["name"],
            notes=entry.get("notes"),
        )
        conn.execute(
            org_stmt.on_conflict_do_update(
                index_elements=[ORGANIZATIONS.c.organization_id],
                set_={
                    "name": entry["name"],
                    "notes": entry.get("notes"),
                },
            )
        )
        for team_id in entry["team_ids"]:
            if team_id not in registry:
                raise ValueError(
                    f"organization_id {entry['organization_id']}: team_id "
                    f"{team_id} is not in the teams registry"
                )
            configured_team_ids.add(team_id)
            conn.execute(
                pg_insert(TEAM_ORGANIZATION_MEMBERSHIPS)
                .values(
                    team_id=team_id,
                    organization_id=entry["organization_id"],
                    reason=entry.get("reason"),
                    source=entry.get("source"),
                )
                .on_conflict_do_update(
                    index_elements=[TEAM_ORGANIZATION_MEMBERSHIPS.c.team_id],
                    set_={
                        "organization_id": entry["organization_id"],
                        "reason": entry.get("reason"),
                        "source": entry.get("source"),
                    },
                )
            )

    existing_membership_rows = conn.execute(
        select(TEAM_ORGANIZATION_MEMBERSHIPS.c.team_id)
    ).all()
    for row in existing_membership_rows:
        if int(row.team_id) not in configured_team_ids:
            conn.execute(
                TEAM_ORGANIZATION_MEMBERSHIPS.delete().where(
                    TEAM_ORGANIZATION_MEMBERSHIPS.c.team_id == row.team_id
                )
            )


def collect_team_alias_observations(
    conn: Connection,
) -> list[tuple[int, str | None, datetime]]:
    """Read `(team_id, name, start_time)` observations from canonical matches.

    One tuple per match side (radiant + dire). Names come only from the
    immutable `matches.*_team_name_observed` columns; this never writes to
    `matches`.
    """
    rows = conn.execute(
        select(
            MATCHES.c.radiant_team_id,
            MATCHES.c.radiant_team_name_observed,
            MATCHES.c.dire_team_id,
            MATCHES.c.dire_team_name_observed,
            MATCHES.c.start_time,
        )
    ).all()
    observations: list[tuple[int, str | None, datetime]] = []
    for row in rows:
        observations.append(
            (row.radiant_team_id, row.radiant_team_name_observed, row.start_time)
        )
        observations.append(
            (row.dire_team_id, row.dire_team_name_observed, row.start_time)
        )
    return observations


def _payload_observed_at(payload: dict) -> datetime | None:
    """The observation instant for a raw payload's team-tag facts.

    Prefers the match's own `startDateTime` (when the tag was actually
    current for that game); returns None so the caller can fall back to
    `fetched_at`.
    """
    start = payload.get("startDateTime")
    if start is None:
        return None
    return datetime.fromtimestamp(int(start), tz=UTC)


def collect_team_tag_observations(
    conn: Connection,
    *,
    registry: set[int],
) -> tuple[list[tuple[int, str | None, datetime]], list[int]]:
    """Read `(team_id, tag, observed_at)` observations from raw payloads.

    Returns the observations plus the raw team ids skipped because they
    are not in the `teams` registry (their tags are not recorded, so the
    identity tables never reference ids outside the registry). Only local
    `stratz_raw_matches` data is read -- no API calls.
    """
    rows = conn.execute(
        select(STRATZ_RAW_MATCHES.c.payload, STRATZ_RAW_MATCHES.c.fetched_at)
    ).all()
    observations: list[tuple[int, str | None, datetime]] = []
    skipped_raw_team_ids: set[int] = set()
    for row in rows:
        payload = row.payload or {}
        observed_at = _payload_observed_at(payload) or row.fetched_at
        for key in _RAW_TEAM_KEYS:
            team = payload.get(key) or {}
            team_id = team.get("id")
            tag = team.get("tag")
            if team_id is None or tag is None:
                continue
            team_id = int(team_id)
            if team_id not in registry:
                skipped_raw_team_ids.add(team_id)
                continue
            observations.append((team_id, tag, observed_at))
    return observations, sorted(skipped_raw_team_ids)


def run_backfill(engine: Engine) -> dict[str, object]:
    """Rebuild `team_aliases` + `team_tags` from local canonical/raw data.

    Idempotent and deterministic: re-running computes the same rows and
    upserts them, never creating duplicates. Returns a summary dict.
    """
    with engine.connect() as conn:
        registry = list_team_ids_in_registry(conn)
        alias_observations = collect_team_alias_observations(conn)
        tag_observations, skipped_raw_team_ids = collect_team_tag_observations(
            conn, registry=registry
        )

    aliases = derive_team_aliases(alias_observations)
    tags = derive_team_tags(tag_observations)

    with engine.begin() as conn:
        sync_team_aliases(conn, aliases)
        sync_team_tags(conn, tags)

    return {
        "registry_team_ids": len(registry),
        "alias_rows": len(aliases),
        "tag_rows": len(tags),
        "skipped_raw_team_ids": skipped_raw_team_ids,
    }