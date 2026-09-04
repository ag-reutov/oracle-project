"""Read-only team-identity diagnostics over the canonical warehouse.

Reports (never writes, never re-fetches, never classifies):

* Raw team population -- registry rows, teams referenced by matches, and
  orphaned registry ids.
* Names -- missing observed names, distinct observed names, team ids seen
  under more than one name, and every name observed for more than one
  team id (collision) with the full per-id evidence (match count, first
  seen, last seen) so cases such as Virtus.pro are immediately visible.
* Organization mapping -- mapped / unmapped raw team ids, organizations
  grouping more than one id, and any invalid or dangling mappings.
* Tags -- tag coverage, ids with multiple observed tags, missing tags.

Collision rows are reported with evidence only; they are never classified
as the same organization, a stand-in stack, a rename, or a duplicate.

Usage:
    uv run python scripts/audit_team_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import Connection, Engine, func, select, text

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dota_predictor.storage.engine import get_engine
from dota_predictor.storage.schema import (
    MATCHES,
    ORGANIZATIONS,
    TEAM_ORGANIZATION_MEMBERSHIPS,
    TEAM_TAGS,
    TEAMS,
)
from dota_predictor.utils.env import load_project_env

# Observation source: the immutable per-match observed-name columns.
_NAME_OBSERVATIONS_SQL = """
    SELECT team_id, name, start_time FROM (
        SELECT radiant_team_id AS team_id, radiant_team_name_observed AS name,
               start_time FROM matches
        UNION ALL
        SELECT dire_team_id AS team_id, dire_team_name_observed AS name,
               start_time FROM matches
    ) obs WHERE name IS NOT NULL
"""


def _count(conn: Connection, stmt: object) -> int:
    return int(conn.execute(stmt).scalar_one())


def _registry_population(conn: Connection) -> dict[str, int]:
    registry = _count(conn, select(func.count()).select_from(TEAMS))
    referenced = _count(
        conn,
        select(func.count(func.distinct(text("team_id")))).select_from(
            text(
                "(SELECT radiant_team_id AS team_id FROM matches "
                "UNION SELECT dire_team_id AS team_id FROM matches) s"
            )
        ),
    )
    return {
        "registry_rows": registry,
        "referenced_by_matches": referenced,
        "orphan_registry_ids": registry - referenced,
    }


def _name_diagnostics(conn: Connection) -> dict[str, object]:
    missing_observed = _count(
        conn,
        select(func.count()).select_from(MATCHES).where(
            (MATCHES.c.radiant_team_name_observed.is_(None))
            | (MATCHES.c.dire_team_name_observed.is_(None))
        ),
    )
    distinct_names = _count(
        conn,
        select(func.count(func.distinct(text("name")))).select_from(
            text(f"({_NAME_OBSERVATIONS_SQL}) s")
        ),
    )
    teams_with_multiple_names = _count(
        conn,
        select(func.count()).select_from(
            text(
                "("
                "SELECT team_id FROM ("
                "SELECT DISTINCT team_id, name FROM ("
                "SELECT radiant_team_id AS team_id, radiant_team_name_observed AS name "
                "FROM matches WHERE radiant_team_name_observed IS NOT NULL "
                "UNION "
                "SELECT dire_team_id AS team_id, dire_team_name_observed AS name "
                "FROM matches WHERE dire_team_name_observed IS NOT NULL"
                ") o) per_team GROUP BY team_id HAVING count(*) > 1"
                ") s"
            )
        ),
    )

    collision_rows = conn.execute(
        text(
            "SELECT name, team_id, count(*) AS n_matches, "
            "min(start_time) AS first_seen, max(start_time) AS last_seen "
            f"FROM ({_NAME_OBSERVATIONS_SQL}) s "
            "GROUP BY name, team_id "
            "ORDER BY name, team_id"
        )
    ).all()
    # Build per-name collision groups directly from the per-(name, team)
    # aggregation above.
    groups: dict[str, list[dict[str, object]]] = {}
    for row in collision_rows:
        groups.setdefault(str(row.name), []).append(
            {
                "team_id": int(row.team_id),
                "n_matches": int(row.n_matches),
                "first_seen": row.first_seen,
                "last_seen": row.last_seen,
            }
        )
    collisions = {
        name: entries for name, entries in groups.items() if len(entries) > 1
    }
    return {
        "missing_observed_names_matches": missing_observed,
        "distinct_names": distinct_names,
        "team_ids_with_multiple_names": teams_with_multiple_names,
        "name_collisions": collisions,
        "name_collision_count": len(collisions),
    }


def _organization_diagnostics(conn: Connection) -> dict[str, object]:
    mapped_team_ids = _count(
        conn, select(func.count()).select_from(TEAM_ORGANIZATION_MEMBERSHIPS)
    )
    registry = _count(conn, select(func.count()).select_from(TEAMS))
    unmapped = registry - mapped_team_ids
    orgs_with_multiple_ids = _count(
        conn,
        select(func.count()).select_from(
            text(
                "("
                "SELECT organization_id FROM team_organization_memberships "
                "GROUP BY organization_id HAVING count(*) > 1"
                ") s"
            )
        ),
    )
    # Invalid / dangling mappings: memberships whose team_id or
    # organization_id no longer resolves. FKs make these impossible by
    # construction; the checks are kept as cheap integrity assertions.
    invalid_team_memberships = _count(
        conn,
        select(func.count()).select_from(
            TEAM_ORGANIZATION_MEMBERSHIPS.join(
                TEAMS,
                TEAM_ORGANIZATION_MEMBERSHIPS.c.team_id == TEAMS.c.team_id,
                isouter=True,
            )
        ).where(TEAMS.c.team_id.is_(None)),
    )
    invalid_org_memberships = _count(
        conn,
        select(func.count()).select_from(
            TEAM_ORGANIZATION_MEMBERSHIPS.join(
                ORGANIZATIONS,
                TEAM_ORGANIZATION_MEMBERSHIPS.c.organization_id
                == ORGANIZATIONS.c.organization_id,
                isouter=True,
            )
        ).where(ORGANIZATIONS.c.organization_id.is_(None)),
    )
    dangling_team_ids = conn.execute(
        text(
            "SELECT m.team_id FROM team_organization_memberships m "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM matches x "
            "WHERE x.radiant_team_id = m.team_id OR x.dire_team_id = m.team_id"
            ") ORDER BY m.team_id"
        )
    ).scalars().all()
    org_ids = conn.execute(select(ORGANIZATIONS.c.organization_id)).scalars().all()
    return {
        "mapped_team_ids": mapped_team_ids,
        "unmapped_team_ids": unmapped,
        "organizations_with_multiple_team_ids": orgs_with_multiple_ids,
        "invalid_team_memberships": invalid_team_memberships,
        "invalid_org_memberships": invalid_org_memberships,
        "dangling_mapped_team_ids": [int(x) for x in dangling_team_ids],
        "organization_ids": sorted(int(x) for x in org_ids),
    }


def _tag_diagnostics(conn: Connection) -> dict[str, object]:
    registry = _count(conn, select(func.count()).select_from(TEAMS))
    teams_with_tag = _count(
        conn, select(func.count(func.distinct(TEAM_TAGS.c.team_id)))
    )
    distinct_tags = _count(
        conn, select(func.count(func.distinct(TEAM_TAGS.c.tag)))
    )
    ids_with_multiple_tags = _count(
        conn,
        select(func.count()).select_from(
            text(
                "("
                "SELECT team_id FROM team_tags "
                "GROUP BY team_id HAVING count(*) > 1"
                ") s"
            )
        ),
    )
    missing_tag_team_ids = conn.execute(
        select(TEAMS.c.team_id)
        .where(
            ~TEAMS.c.team_id.in_(select(TEAM_TAGS.c.team_id).distinct())
        )
        .order_by(TEAMS.c.team_id)
    ).scalars().all()
    return {
        "registry_teams": registry,
        "teams_with_tag": teams_with_tag,
        "tag_coverage_pct": round(100.0 * teams_with_tag / registry, 2)
        if registry
        else 0.0,
        "distinct_tags": distinct_tags,
        "team_ids_with_multiple_tags": ids_with_multiple_tags,
        "team_ids_without_tag_count": len(missing_tag_team_ids),
    }


def audit(engine: Engine) -> dict[str, object]:
    with engine.connect() as conn:
        report: dict[str, object] = {
            "raw_team_population": _registry_population(conn),
            "names": _name_diagnostics(conn),
            "organizations": _organization_diagnostics(conn),
            "tags": _tag_diagnostics(conn),
        }
    return report


def _print_collisions(collisions: dict[str, list[dict[str, object]]]) -> None:
    if not collisions:
        return
    print("\n  -- name -> multiple team_id collisions (evidence only) --")
    for name, entries in sorted(
        collisions.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        print(f"  {name!r}:")
        for entry in entries:
            print(
                f"    team_id {entry['team_id']}: {entry['n_matches']} matches, "
                f"{entry['first_seen']} .. {entry['last_seen']}"
            )


def _print_report(report: dict[str, object]) -> None:
    pop = report["raw_team_population"]
    print("=== Raw team population ===")
    print(f"  teams registry rows: {pop['registry_rows']}")
    print(f"  team ids referenced by matches: {pop['referenced_by_matches']}")
    print(f"  orphan registry ids: {pop['orphan_registry_ids']}")

    names = report["names"]
    print("\n=== Names ===")
    print(f"  matches with a missing observed name: {names['missing_observed_names_matches']}")
    print(f"  distinct observed names: {names['distinct_names']}")
    print(f"  team ids observed under >1 name: {names['team_ids_with_multiple_names']}")
    print(f"  names observed for >1 team id: {names['name_collision_count']}")
    _print_collisions(names["name_collisions"])

    orgs = report["organizations"]
    print("\n=== Organization mapping ===")
    print(f"  mapped raw team ids: {orgs['mapped_team_ids']}")
    print(f"  unmapped raw team ids: {orgs['unmapped_team_ids']}")
    print(f"  organizations grouping >1 raw team id: {orgs['organizations_with_multiple_team_ids']}")
    print(f"  invalid team memberships: {orgs['invalid_team_memberships']}")
    print(f"  invalid org memberships: {orgs['invalid_org_memberships']}")
    print(f"  dangling mapped team ids: {orgs['dangling_mapped_team_ids']}")
    print(f"  organization ids: {orgs['organization_ids']}")

    tags = report["tags"]
    print("\n=== Tags ===")
    print(f"  registry teams: {tags['registry_teams']}")
    print(f"  teams with >=1 observed tag: {tags['teams_with_tag']} "
          f"({tags['tag_coverage_pct']}%)")
    print(f"  distinct tags: {tags['distinct_tags']}")
    print(f"  team ids with multiple observed tags: {tags['team_ids_with_multiple_tags']}")
    print(f"  registry teams with no observed tag: {tags['team_ids_without_tag_count']}")


def main() -> int:
    load_project_env(REPO_ROOT)
    engine = get_engine()
    report = audit(engine)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())