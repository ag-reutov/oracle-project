"""Canonical player-identity foundation (Slice 2).

This module implements the explicit player-identity distinction the Slice 2
spec requires, on top of the existing canonical warehouse:

* **Source player identity** = the stable STRATZ `steamAccountId`, stored
  as `players.player_id`. This is the canonical player identifier wherever
  it is available and valid. Identity never changes with team changes,
  role changes, or nickname changes.
* **`players` registry** = the minimal canonical identity table (primary
  key only). Per the project's documented convention (see `storage.schema`
  module docstring), no derived or time-varying attributes live here, so
  it is structurally impossible to add a `current_team_id` / `position`
  shortcut that would leak future state into historical research.
* **Player universe** = the derived, always-fresh identity summary
  (`player_id`, `display_name`, `first_seen_at`, `last_seen_at`,
  `match_count`), exposed as the `research.players` view
  (`PLAYER_UNIVERSE_VIEW_SQL`) and as the Python helper
  `fetch_player_universe`. These summary fields are pure aggregates over
  the immutable canonical facts (`match_players` + `matches.start_time`),
  computed fresh each time -- never cached as mutable Postgres state that
  reprocessing could leave stale.
* **Display-name resolution** = a deterministic rule: the most recently
  observed valid name, tie-broken lexicographically (`resolve_display_name`).
  Player ids are authoritative; names are attributes of identities, never
  identities themselves, and name equality alone NEVER merges player ids.

Source-data reality check (verified against the live warehouse):

* Every canonical `match_players.player_id` is the raw STRATZ
  `steamAccountId` and resolves to exactly one `players` row (a foreign
  key guarantees this, and the writer registers ids on ingest).
* The raw STRATZ match payloads contain **no player display names**
  (only `steamAccountId`; `proSteamAccount` is always null in the local
  corpus). `display_name` is therefore NULL for every player today. The
  deterministic name-resolution machinery is still implemented and tested
  so the identity foundation does not have to be redefined when name
  observations become available, and so the "nickname change never creates
  a new player" / "same name never merges players" guarantees hold.

Design decisions (documented here per the Slice 2 spec):

* `players` stays a PK-only registry; the summary fields are derived, not
  stored, so they cannot go stale and cannot leak mutable competitive
  state (team, position, rating, form, hero pool) into identity.
* No alias subsystem is built: the local data contains no player names, so
  a per-name history table would be empty. The pure derivation functions
  below are the tested, future-ready machinery instead.
* Backfill (`run_backfill`) is deterministic and idempotent: it re-asserts
  that every referenced `player_id` exists in the `players` registry via
  `INSERT ... ON CONFLICT DO NOTHING`, never deletes orphan registry ids
  (see `storage.schema` docstring for why orphan registry rows are
  intentionally kept), and never writes to `matches` / `match_players`.

The pure derivation functions take plain observation tuples and are
testable without a database; the `sync_*` / `fetch_*` / `audit_*` helpers
talk to Postgres and are used by the CLI scripts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Connection, Engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dota_predictor.storage.schema import MATCH_PLAYERS, MATCHES, PLAYERS

__all__ = [
    "PLAYER_UNIVERSE_VIEW_SQL",
    "PlayerIdentity",
    "PlayerName",
    "PlayerSummary",
    "audit_player_identity",
    "derive_player_names",
    "derive_player_summaries",
    "fetch_player_universe",
    "list_player_ids_in_registry",
    "resolve_display_name",
    "run_backfill",
    "sync_player_registry",
]


@dataclass(frozen=True, slots=True)
class PlayerSummary:
    """One canonical player's observation summary over the canonical corpus.

    `first_seen_at` / `last_seen_at` are the earliest/latest `matches`
    start times in which this player appears; `match_count` is the number
    of canonical matches observed for the player (one appearance per match
    by the `(match_id, player_id)` uniqueness constraint). Deterministically
    derived from `match_players` joined to `matches.start_time`.
    """

    player_id: int
    first_seen_at: datetime
    last_seen_at: datetime
    match_count: int


@dataclass(frozen=True, slots=True)
class PlayerName:
    """One name observed for a canonical player, plus its observation period.

    `first_seen_at`/`last_seen_at` are the earliest/latest observation
    times for this name; `observation_count` is the number of observations
    carrying it. Deterministically derived from `(player_id, name,
    start_time)` observations. Kept as a pure-history diagnostic: the local
    corpus currently has no player-name observations, but the identity
    guarantees (nickname changes never create players; identical names on
    different ids never merge) are defined and tested on this type.
    """

    player_id: int
    name: str
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    """One row of the canonical player universe.

    `display_name` is the deterministic best-known name (`None` when no
    name observations exist -- the current state of the local corpus).
    `first_seen_at` / `last_seen_at` / `match_count` are derived summaries.
    Contains no mutable team/position/rating/state columns by design.
    """

    player_id: int
    display_name: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    match_count: int


# Canonical player-universe SELECT, shared by the `research.players` view
# (research/views.py and the Alembic migration embed a frozen copy) and by
# `fetch_player_universe`. INNER join: only players observed in at least one
# canonical match are part of the universe; orphan registry ids are reported
# by the audit, not fabricated as zero-match rows.
PLAYER_UNIVERSE_VIEW_SQL = """
CREATE OR REPLACE VIEW research.players AS
SELECT
    p.player_id,
    NULL::text AS display_name,
    MIN(m.start_time) AS first_seen_at,
    MAX(m.start_time) AS last_seen_at,
    COUNT(*) AS match_count
FROM public.players p
JOIN public.match_players mp USING (player_id)
JOIN public.matches m USING (match_id)
GROUP BY p.player_id
"""


def _group_observations(
    observations: Iterable[tuple[int, str, datetime]],
) -> list[tuple[int, str, list[datetime]]]:
    """Group `(player_id, value, observed_at)` by (player_id, value).

    Returns rows sorted deterministically by (player_id, value). Callers
    filter out None values before calling so a missing observation never
    produces a row.
    """
    by_key: dict[tuple[int, str], list[datetime]] = {}
    for player_id, value, observed_at in observations:
        by_key.setdefault((int(player_id), value), []).append(observed_at)
    return [
        (player_id, value, times)
        for (player_id, value), times in sorted(by_key.items())
    ]


def derive_player_summaries(
    observations: Iterable[tuple[int, datetime]],
) -> list[PlayerSummary]:
    """Derive deterministic per-player summaries from match observations.

    Each observation is `(player_id, start_time)`. Returns one
    `PlayerSummary` per player id, sorted by player_id regardless of input
    order. `match_count` is the number of observations (one canonical
    appearance per match, enforced upstream by the
    `(match_id, player_id)` uniqueness constraint).
    """
    by_player: dict[int, list[datetime]] = {}
    for player_id, start_time in observations:
        by_player.setdefault(int(player_id), []).append(start_time)
    return [
        PlayerSummary(
            player_id=player_id,
            first_seen_at=min(times),
            last_seen_at=max(times),
            match_count=len(times),
        )
        for player_id, times in sorted(by_player.items())
    ]


def derive_player_names(
    observations: Iterable[tuple[int, str | None, datetime]],
) -> list[PlayerName]:
    """Derive deterministic per-name history rows from name observations.

    Each observation is `(player_id, name, start_time)`. Observations with
    a `None` name are ignored (a missing name is not an observation). One
    row per (player_id, name); a nickname change therefore produces an
    additional row rather than rewriting history. Output is sorted by
    (player_id, name) regardless of input order.
    """
    return [
        PlayerName(
            player_id=player_id,
            name=name,
            first_seen_at=min(times),
            last_seen_at=max(times),
            observation_count=len(times),
        )
        for player_id, name, times in _group_observations(
            (int(player_id), str(name), observed_at)
            for player_id, name, observed_at in observations
            if name is not None
        )
    ]


def resolve_display_name(names: Sequence[PlayerName]) -> str | None:
    """Deterministic canonical display-name rule for one player.

    Rule: the **most recently observed valid name** (maximum
    `last_seen_at`); ties are broken by the lexicographically smallest
    name so the result is identical regardless of input order. `None` when
    the player has no valid name observations. Player ids stay
    authoritative: this only picks a display attribute for an already-known
    identity and can never merge two player ids.
    """
    if not names:
        return None
    latest_seen = max(name.last_seen_at for name in names)
    latest = [name for name in names if name.last_seen_at == latest_seen]
    return min(latest, key=lambda name: name.name).name


def list_player_ids_in_registry(conn: Connection) -> set[int]:
    """Return the set of `player_id`s currently in the `players` registry."""
    return {
        int(row.player_id) for row in conn.execute(select(PLAYERS.c.player_id)).all()
    }


def sync_player_registry(conn: Connection, player_ids: Iterable[int]) -> int:
    """Insert any missing player ids into the `players` registry.

    Idempotent (`INSERT ... ON CONFLICT DO NOTHING`). Returns the number of
    ids newly added. Never deletes or updates existing rows -- registry
    cleanup is deliberately out of scope (see `storage.schema` docstring).
    """
    wanted = {int(player_id) for player_id in player_ids}
    if not wanted:
        return 0
    existing = list_player_ids_in_registry(conn)
    missing = wanted - existing
    if missing:
        stmt = pg_insert(PLAYERS).values(
            [{"player_id": player_id} for player_id in sorted(missing)]
        )
        conn.execute(stmt.on_conflict_do_nothing(index_elements=[PLAYERS.c.player_id]))
    return len(missing)


def fetch_player_universe(conn: Connection) -> list[PlayerIdentity]:
    """Return the canonical player universe from local canonical tables.

    Equivalent to querying `research.players` but does not require the
    research schema/view to be installed, so Python consumers can always
    retrieve the universe. Rows are sorted by `player_id`.
    """
    rows = conn.execute(
        select(
            PLAYERS.c.player_id,
            func.min(MATCHES.c.start_time).label("first_seen_at"),
            func.max(MATCHES.c.start_time).label("last_seen_at"),
            func.count().label("match_count"),
        )
        .select_from(
            PLAYERS.join(
                MATCH_PLAYERS, PLAYERS.c.player_id == MATCH_PLAYERS.c.player_id
            ).join(MATCHES, MATCH_PLAYERS.c.match_id == MATCHES.c.match_id)
        )
        .group_by(PLAYERS.c.player_id)
        .order_by(PLAYERS.c.player_id)
    ).all()
    return [
        PlayerIdentity(
            player_id=int(row.player_id),
            display_name=None,
            first_seen_at=row.first_seen_at,
            last_seen_at=row.last_seen_at,
            match_count=int(row.match_count),
        )
        for row in rows
    ]


def run_backfill(engine: Engine) -> dict[str, object]:
    """Re-assert the `players` registry covers every referenced player id.

    Deterministic and idempotent: re-running recomputes the same state and
    adds nothing once the registry is complete. Reads only
    `match_players`/`players` and writes only to `players` (never
    `matches` / `match_players`). Returns a summary dict.
    """
    with engine.connect() as conn:
        referenced = {
            int(player_id)
            for player_id in conn.execute(
                select(MATCH_PLAYERS.c.player_id).distinct()
            ).scalars()
        }

    with engine.begin() as conn:
        added = sync_player_registry(conn, referenced)

    with engine.connect() as conn:
        registry = list_player_ids_in_registry(conn)

    return {
        "registry_player_count": len(registry),
        "referenced_player_count": len(referenced),
        "added_player_ids": added,
    }


def _audit_counts(conn: Connection) -> dict[str, int]:
    """Deterministic core counts for the player-identity audit."""
    registry_count = int(
        conn.execute(select(func.count()).select_from(PLAYERS)).scalar_one()
    )
    distinct_valid = int(
        conn.execute(
            select(func.count(func.distinct(MATCH_PLAYERS.c.player_id)))
        ).scalar_one()
    )
    null_rows = int(
        conn.execute(
            select(func.count())
            .select_from(MATCH_PLAYERS)
            .where(MATCH_PLAYERS.c.player_id.is_(None))
        ).scalar_one()
    )
    invalid_rows = int(
        conn.execute(
            select(func.count())
            .select_from(MATCH_PLAYERS)
            .where(MATCH_PLAYERS.c.player_id <= 0)
        ).scalar_one()
    )
    referenced_registry_ids = int(
        conn.execute(
            select(func.count(func.distinct(MATCH_PLAYERS.c.player_id))).select_from(
                MATCH_PLAYERS.join(
                    PLAYERS, MATCH_PLAYERS.c.player_id == PLAYERS.c.player_id
                )
            )
        ).scalar_one()
    )
    return {
        "registry_count": registry_count,
        "distinct_valid": distinct_valid,
        "null_player_id_rows": null_rows,
        "invalid_player_id_rows": invalid_rows,
        "referenced_registry_ids": referenced_registry_ids,
    }


def audit_player_identity(engine: Engine) -> dict[str, object]:
    """Deterministic player-identity diagnostics over the canonical warehouse.

    Read-only; never writes, never re-fetches, never classifies. Returns a
    report dict (printed by `scripts/audit_player_identity.py`) covering
    registry population, identity coverage, orphan ids, null/invalid ids,
    name diagnostics (none today -- the corpus has no player names), date
    coverage, and a matches-per-player distribution summary.
    """
    with engine.connect() as conn:
        counts = _audit_counts(conn)
        orphan_registry_ids = sorted(
            list_player_ids_in_registry(conn)
            - {
                int(pid)
                for pid in conn.execute(
                    select(MATCH_PLAYERS.c.player_id).distinct()
                ).scalars()
            }
        )
        first_seen, last_seen = conn.execute(
            select(func.min(MATCHES.c.start_time), func.max(MATCHES.c.start_time))
        ).one()
        match_counts = sorted(
            int(row.match_count)
            for row in conn.execute(
                select(func.count().label("match_count"))
                .select_from(
                    MATCH_PLAYERS.join(
                        MATCHES, MATCH_PLAYERS.c.match_id == MATCHES.c.match_id
                    )
                )
                .group_by(MATCH_PLAYERS.c.player_id)
            ).all()
        )

    distinct_valid = counts["distinct_valid"]
    unresolved = distinct_valid - counts["referenced_registry_ids"]
    coverage_pct = (
        round(100.0 * (distinct_valid - unresolved) / distinct_valid, 4)
        if distinct_valid
        else 0.0
    )

    match_count_summary: dict[str, int] = {}
    if match_counts:
        n = len(match_counts)
        match_count_summary = {
            "min": match_counts[0],
            "median": match_counts[n // 2],
            "max": match_counts[-1],
            "count": n,
        }

    violations: list[str] = []
    if unresolved:
        violations.append(
            f"{unresolved} referenced player id(s) have no canonical player row"
        )
    if counts["null_player_id_rows"]:
        violations.append(
            f"{counts['null_player_id_rows']} match_players row(s) have a NULL player_id"
        )
    if counts["invalid_player_id_rows"]:
        violations.append(
            f"{counts['invalid_player_id_rows']} match_players row(s) have a non-positive player_id"
        )
    # Orphan registry ids (registered but referenced by no current match) are
    # NOT violations: `storage.schema` documents that the registry is allowed
    # to accumulate such rows (referential integrity is one-directional). They
    # are reported as a diagnostic observation.
    orphans_observation = (
        f"{len(orphan_registry_ids)} registry player id(s) are not referenced "
        "by any match (allowed orphan registry rows; see storage.schema)"
    )

    return {
        "canonical_player_count": counts["registry_count"],
        "distinct_valid_player_ids": distinct_valid,
        "identity_coverage_pct": coverage_pct,
        "unresolved_player_ids": unresolved,
        "orphan_registry_ids": orphan_registry_ids,
        "orphan_registry_count": len(orphan_registry_ids),
        "null_player_id_rows": counts["null_player_id_rows"],
        "invalid_player_id_rows": counts["invalid_player_id_rows"],
        # Name diagnostics: the local corpus contains no player-name
        # observations (verified across all raw payloads), so these are 0.
        "players_with_multiple_names": 0,
        "shared_name_collision_count": 0,
        "players_without_display_name": counts["registry_count"],
        "first_seen": first_seen,
        "last_seen": last_seen,
        "matches_per_player": match_count_summary,
        "diagnostic_observations": [orphans_observation],
        "integrity_violations": violations,
    }
