"""Read-only player-identity diagnostics over the canonical warehouse.

Reports (never writes, never re-fetches, never classifies) the Slice 2
identity-foundation state:

* Registry population -- canonical player count, distinct valid player ids
  referenced by `match_players`, identity coverage %, unresolved ids, and
  orphan registry ids (registered but referenced by no match).
* Null/invalid ids -- `match_players` rows with NULL or non-positive
  `player_id` (schema forbids these; reported if ever present).
* Names -- players observed under more than one name and display names
  shared by multiple player ids. The local corpus currently contains no
  player-name observations, so these are 0 and the reason is stated.
* Observation window -- first/last `matches.start_time` in the corpus.
* Matches-per-player distribution summary (min / median / max / count).
* Any remaining integrity violations, listed explicitly.

Deterministic and easy to rerun: the same report is reproduced from the
same warehouse state.

Usage:
    uv run python scripts/audit_player_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dota_predictor.data.player_identity import audit_player_identity
from dota_predictor.storage.engine import get_engine
from dota_predictor.utils.env import load_project_env


def _print_report(report: dict[str, object]) -> None:
    print("=== Player identity foundation audit ===")
    print(f"  canonical players (players registry): {report['canonical_player_count']}")
    print(
        f"  distinct valid player ids in match_players: {report['distinct_valid_player_ids']}"
    )
    print(f"  identity coverage %: {report['identity_coverage_pct']}")
    print(f"  unresolved referenced player ids: {report['unresolved_player_ids']}")
    print(
        f"  orphan registry player ids (unreferenced): {report['orphan_registry_count']}"
    )
    if report["orphan_registry_ids"]:
        orphan_ids = report["orphan_registry_ids"]
        print(f"    -> {orphan_ids[:10]}{'...' if len(orphan_ids) > 10 else ''}")

    print("\n=== Null / invalid ids ===")
    print(f"  match_players rows with NULL player_id: {report['null_player_id_rows']}")
    print(
        f"  match_players rows with non-positive player_id: {report['invalid_player_id_rows']}"
    )

    print("\n=== Names ===")
    print(f"  players observed under >1 name: {report['players_with_multiple_names']}")
    print(
        f"  display names shared by >1 player id: {report['shared_name_collision_count']}"
    )
    print(
        f"  players without a usable display name: {report['players_without_display_name']}"
    )
    print(
        "    (the local corpus has no player-name observations -- raw payloads "
        "carry only steamAccountId -- so display_name is NULL for every player)"
    )

    print("\n=== Observation window ===")
    print(f"  first match start_time: {report['first_seen']}")
    print(f"  last match start_time: {report['last_seen']}")

    dist = report["matches_per_player"]
    print("\n=== Matches per player (summary) ===")
    print(f"  players: {dist['count']}")
    print(f"  min: {dist['min']}  median: {dist['median']}  max: {dist['max']}")

    violations = report["integrity_violations"]
    print("\n=== Integrity violations ===")
    if violations:
        for violation in violations:
            print(f"  !! {violation}")
    else:
        print("  none")

    for observation in report["diagnostic_observations"]:
        print(f"  note: {observation}")


def main() -> int:
    load_project_env(REPO_ROOT)
    engine = get_engine()
    report = audit_player_identity(engine)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
