"""Canonical reference-entity layer (Slice 3).

This module is the identity/reference companion to the STRATZ constants
catalogs (`heroes.parquet`, `game_versions.parquet`) and the curated
PostgreSQL `leagues` registry. It exposes one canonical, inspectable
path per reference relationship:

* `hero_id` -> canonical hero (`HeroIdentity`, from `heroes.parquet`)
* `game_version_id` -> canonical patch (`GameVersionIdentity`, from
  `game_versions.parquet`)
* `league_id` -> canonical league/event (`LeagueIdentity`, from the
  `leagues` registry)

Design decisions (documented here per the Slice 3 spec):

* Storage follows the existing architecture: heroes and game versions
  stay Parquet reference catalogs (consumed via the DuckDB
  `register_reference_views` layer); leagues stay in the curated
  PostgreSQL `leagues` registry. No parallel tables are created.
* Provenance is explicit. Every hero/game-version row carries
  `source` (the STRATZ constant it came from) and `retrieved_at` (when
  that catalog was fetched). League rows distinguish the STRATZ source
  tier (`stratz_tier`) from our curated Liquipedia tier
  (`liquipedia_tier`); these are never conflated.
* Mutable labels are treated as attributes of stable ids, never as
  identities. Ids stay authoritative.
* First-seen-in-corpus for a game version is derived from the canonical
  `matches.start_time` facts and exposed separately, explicitly labelled
  as a corpus-derived observation -- it is NOT an authoritative release
  date (STRATZ `as_of_datetime` is the authoritative release timestamp
  and is preserved verbatim).
* Unsupported/missing ids are never silently invented. The pure
  resolvers return `None` for unknown ids, and the audit reports
  unresolved ids explicitly.

The pure functions take plain observation tuples / Parquet paths and are
testable without a database; the `audit_*` / `fetch_*` helpers talk to
Postgres and the reference Parquet files and are used by the CLI census
script (`scripts/audit_reference_entities.py`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from sqlalchemy import Connection, Engine, func, select

from dota_predictor.storage.schema import DRAFT_EVENTS, LEAGUES, MATCH_PLAYERS, MATCHES

__all__ = [
    "GameVersionIdentity",
    "HeroIdentity",
    "LeagueIdentity",
    "audit_reference_entities",
    "derive_game_version_first_seen",
    "fetch_league_identities",
    "fetch_referenced_game_version_ids",
    "fetch_referenced_hero_ids",
    "fetch_referenced_league_ids",
    "load_game_versions",
    "load_heroes",
    "resolve_game_version",
    "resolve_hero",
]


@dataclass(frozen=True, slots=True)
class HeroIdentity:
    """One canonical hero from `heroes.parquet`.

    `hero_id` is the stable Dota/STRATZ hero id. `name` is the canonical
    display name; `short_name` is the STRATZ-supplied short/slug name;
    `aliases` is the STRATZ-supplied alias list (possibly empty). All
    labels are attributes of the stable id, never identities. `source`
    and `retrieved_at` record provenance.
    """

    hero_id: int
    name: str
    short_name: str | None
    aliases: list[str] | None
    source: str
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class GameVersionIdentity:
    """One canonical game version from `game_versions.parquet`.

    `name` is the human-readable patch label (e.g. "7.38");
    `as_of_datetime` is STRATZ's authoritative patch release timestamp
    (source-provided, never inferred). `source` and `retrieved_at`
    record provenance.
    """

    game_version_id: int
    name: str
    as_of_datetime: datetime
    source: str
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class LeagueIdentity:
    """One canonical league/event from the curated `leagues` registry.

    `name` is the curated canonical league name. `stratz_tier` is the raw
    STRATZ `LeagueTier` value (source identity, cross-check signal only);
    `liquipedia_tier` is our curated Liquipedia classification -- the two
    are deliberately kept distinct and never conflated. `source` /
    `curated_at` record curation provenance.
    """

    league_id: int
    name: str
    stratz_tier: str | None
    liquipedia_tier: str
    in_scope: bool
    source: str | None
    start_date: object | None
    end_date: object | None
    curated_at: datetime


def load_heroes(path: Path) -> list[HeroIdentity]:
    """Load canonical heroes from `heroes.parquet`, sorted by `hero_id`."""
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows.sort(key=lambda row: row["hero_id"])
    return [
        HeroIdentity(
            hero_id=int(row["hero_id"]),
            name=row["name"],
            short_name=row.get("short_name"),
            aliases=row.get("aliases"),
            source=row["source"],
            retrieved_at=row["retrieved_at"],
        )
        for row in rows
    ]


def load_game_versions(path: Path) -> list[GameVersionIdentity]:
    """Load canonical game versions from `game_versions.parquet`, sorted by id."""
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows.sort(key=lambda row: row["game_version_id"])
    return [
        GameVersionIdentity(
            game_version_id=int(row["game_version_id"]),
            name=row["name"],
            as_of_datetime=row["as_of_datetime"],
            source=row["source"],
            retrieved_at=row["retrieved_at"],
        )
        for row in rows
    ]


def resolve_hero(heroes: Sequence[HeroIdentity], hero_id: int) -> HeroIdentity | None:
    """Return the canonical hero for `hero_id`, or `None` when unknown.

    Unknown ids are reported (never silently invented): callers must
    treat `None` as "no canonical identity exists".
    """
    for hero in heroes:
        if hero.hero_id == hero_id:
            return hero
    return None


def resolve_game_version(
    versions: Sequence[GameVersionIdentity], game_version_id: int
) -> GameVersionIdentity | None:
    """Return the canonical game version for `game_version_id`, or `None`.

    Unknown ids are reported (never silently invented): callers must
    treat `None` as "no canonical identity exists".
    """
    for version in versions:
        if version.game_version_id == game_version_id:
            return version
    return None


def derive_game_version_first_seen(
    observations: Iterable[tuple[int, datetime]],
) -> dict[int, datetime]:
    """Derive the first-seen-in-corpus date per game version.

    Each observation is `(game_version_id, start_time)`. The result maps
    a game version to the earliest `matches.start_time` in which it was
    observed. This is a corpus-derived observation, NOT an authoritative
    release date (STRATZ `as_of_datetime` is the authoritative release
    timestamp). Rows are deterministic regardless of input order.
    """
    first_seen: dict[int, datetime] = {}
    for game_version_id, start_time in observations:
        key = int(game_version_id)
        current = first_seen.get(key)
        if current is None or start_time < current:
            first_seen[key] = start_time
    return first_seen


def fetch_referenced_hero_ids(conn: Connection) -> set[int]:
    """The set of `hero_id`s referenced by canonical match facts.

    Union of `match_players.hero_id` and `draft_events.hero_id`.
    """
    mp = {
        int(hero_id)
        for hero_id in conn.execute(
            select(MATCH_PLAYERS.c.hero_id).distinct()
        ).scalars()
        if hero_id is not None
    }
    de = {
        int(hero_id)
        for hero_id in conn.execute(select(DRAFT_EVENTS.c.hero_id).distinct()).scalars()
        if hero_id is not None
    }
    return mp | de


def fetch_referenced_game_version_ids(conn: Connection) -> set[int]:
    """The set of `game_version_id`s referenced by canonical matches.

    Null `game_version_id` rows are excluded (they are reported by the
    audit, not silently treated as a version).
    """
    return {
        int(version_id)
        for version_id in conn.execute(
            select(MATCHES.c.game_version_id).distinct()
        ).scalars()
        if version_id is not None
    }


def fetch_referenced_league_ids(conn: Connection) -> set[int]:
    """The set of `league_id`s referenced by canonical matches."""
    return {
        int(league_id)
        for league_id in conn.execute(
            select(MATCHES.c.league_id).distinct()
        ).scalars()
        if league_id is not None
    }


def fetch_league_identities(conn: Connection) -> list[LeagueIdentity]:
    """Return the canonical league registry as `LeagueIdentity` rows.

    Reads only the curated `leagues` table (the `ingestion_leagues`
    allowlist is a separate ingestion concern). Rows are sorted by
    `league_id`. The source-vs-curated tier distinction is preserved:
    `stratz_tier` is the raw STRATZ value, `liquipedia_tier` is our
    curated classification.
    """
    rows = conn.execute(
        select(
            LEAGUES.c.league_id,
            LEAGUES.c.name,
            LEAGUES.c.stratz_tier,
            LEAGUES.c.liquipedia_tier,
            LEAGUES.c.in_scope,
            LEAGUES.c.source,
            LEAGUES.c.start_date,
            LEAGUES.c.end_date,
            LEAGUES.c.curated_at,
        )
        .order_by(LEAGUES.c.league_id)
    ).all()
    return [
        LeagueIdentity(
            league_id=int(row.league_id),
            name=row.name,
            stratz_tier=row.stratz_tier,
            liquipedia_tier=row.liquipedia_tier,
            in_scope=row.in_scope,
            source=row.source,
            start_date=row.start_date,
            end_date=row.end_date,
            curated_at=row.curated_at,
        )
        for row in rows
    ]


def audit_reference_entities(
    engine: Engine,
    *,
    heroes_path: Path,
    game_versions_path: Path,
) -> dict[str, object]:
    """Deterministic reference-entity census over the canonical warehouse.

    Read-only; never writes, never re-fetches, never classifies. Returns
    a report dict (printed by `scripts/audit_reference_entities.py`)
    covering hero / league / game-version counts, referenced/resolved/
    unresolved ids, duplicate/conflict counts, regions status, and any
    integrity violations. The output makes incompleteness obvious.
    """
    heroes = load_heroes(heroes_path)
    game_versions = load_game_versions(game_versions_path)

    with engine.connect() as conn:
        referenced_hero_ids = fetch_referenced_hero_ids(conn)
        referenced_game_version_ids = fetch_referenced_game_version_ids(conn)
        referenced_league_ids = fetch_referenced_league_ids(conn)
        league_identities = fetch_league_identities(conn)
        null_game_version_matches = int(
            conn.execute(
                select(func.count())
                .select_from(MATCHES)
                .where(MATCHES.c.game_version_id.is_(None))
            ).scalar_one()
        )

        game_version_observations = [
            (int(row.game_version_id), row.start_time)
            for row in conn.execute(
                select(MATCHES.c.game_version_id, MATCHES.c.start_time).where(
                    MATCHES.c.game_version_id.is_not(None)
                )
            ).all()
        ]

    hero_ids = {hero.hero_id for hero in heroes}
    game_version_ids = {version.game_version_id for version in game_versions}
    league_ids = {identity.league_id for identity in league_identities}

    resolved_hero_ids = referenced_hero_ids & hero_ids
    unresolved_hero_ids = referenced_hero_ids - hero_ids

    resolved_game_version_ids = referenced_game_version_ids & game_version_ids
    unresolved_game_version_ids = referenced_game_version_ids - game_version_ids

    resolved_league_ids = referenced_league_ids & league_ids
    unresolved_league_ids = referenced_league_ids - league_ids

    hero_duplicates = len(hero_ids) != len(heroes)
    game_version_duplicates = len(game_version_ids) != len(game_versions)
    null_hero_names = [h.hero_id for h in heroes if not h.name]
    league_name_duplicates = {
        name: sorted(i.league_id for i in league_identities if i.name == name)
        for name in {i.name for i in league_identities}
        if sum(1 for i in league_identities if i.name == name) > 1
    }

    first_seen = derive_game_version_first_seen(game_version_observations)

    violations: list[str] = []
    if unresolved_hero_ids:
        violations.append(
            f"{len(unresolved_hero_ids)} referenced hero id(s) have no canonical "
            f"hero row: {sorted(unresolved_hero_ids)}"
        )
    if unresolved_game_version_ids:
        violations.append(
            f"{len(unresolved_game_version_ids)} referenced game version id(s) "
            f"have no canonical row: {sorted(unresolved_game_version_ids)}"
        )
    if unresolved_league_ids:
        violations.append(
            f"{len(unresolved_league_ids)} referenced league id(s) have no "
            f"canonical league row: {sorted(unresolved_league_ids)}"
        )
    if null_game_version_matches:
        violations.append(
            f"{null_game_version_matches} canonical match(es) have a NULL "
            "game_version_id"
        )
    if hero_duplicates:
        violations.append("heroes.parquet contains duplicate hero_id values")
    if game_version_duplicates:
        violations.append("game_versions.parquet contains duplicate game_version_id values")
    if null_hero_names:
        violations.append("heroes.parquet contains heroes with a NULL/empty name")
    if league_name_duplicates:
        violations.append(
            "league registry has duplicate names for distinct league ids: "
            f"{league_name_duplicates}"
        )

    return {
        "hero_count": len(heroes),
        "hero_ids_referenced": len(referenced_hero_ids),
        "hero_ids_resolved": len(resolved_hero_ids),
        "hero_ids_unresolved": len(unresolved_hero_ids),
        "hero_ids_unreferenced": len(hero_ids - referenced_hero_ids),
        "hero_duplicate_ids": hero_duplicates,
        "hero_null_names": null_hero_names,
        "league_count": len(league_ids),
        "league_ids_referenced": len(referenced_league_ids),
        "league_ids_resolved": len(resolved_league_ids),
        "league_ids_unresolved": len(unresolved_league_ids),
        "league_name_duplicates": league_name_duplicates,
        "game_version_count": len(game_version_ids),
        "game_version_ids_referenced": len(referenced_game_version_ids),
        "game_version_ids_resolved": len(resolved_game_version_ids),
        "game_version_ids_unresolved": len(unresolved_game_version_ids),
        "game_version_ids_unreferenced": len(game_version_ids - referenced_game_version_ids),
        "game_version_duplicate_ids": game_version_duplicates,
        "null_game_version_matches": null_game_version_matches,
        "game_version_first_seen_in_corpus": {
            str(key): value.isoformat() for key, value in sorted(first_seen.items())
        },
        "regions": {
            "status": "deferred",
            "reason": (
                "STRATZ exposes constants.regions (server regions) but no "
                "canonical entity references a region id (match region field is "
                "null/UNSET) and no clean team/event-region source exists."
            ),
        },
        "integrity_violations": violations,
    }