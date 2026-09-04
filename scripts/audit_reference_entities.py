"""Reference-entity census audit (Slice 3).

Read-only diagnostics over the canonical reference-entity layer:

* Heroes -- canonical hero count, hero ids referenced by match facts
  (`match_players` + `draft_events`) vs resolved/unresolved, reference
  heroes never observed in the corpus, duplicate ids, null names.
* Leagues -- canonical registry count, league ids referenced by
  `matches` vs resolved/unresolved, duplicate/conflicting names.
* Game versions -- canonical patch count, game-version ids referenced by
  `matches` vs resolved/unresolved, reference patches never observed,
  duplicate ids, null `game_version_id` rows, and (separately, labelled
  as corpus-derived) first-seen-in-corpus per patch.
* Regions -- status (investigated; intentionally deferred).

The output makes incompleteness obvious: every unresolved id, duplicate,
and null is reported explicitly. Read-only; never writes, never
re-fetches, never classifies.

Usage:
    uv run python scripts/audit_reference_entities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dota_predictor.data.reference_identity import audit_reference_entities
from dota_predictor.features.config import load_reference_store_config
from dota_predictor.storage.engine import get_engine
from dota_predictor.utils.env import load_project_env


def _print_report(report: dict[str, object]) -> None:
    regions = report["regions"]
    print("=== Heroes ===")
    print(f"  canonical hero count: {report['hero_count']}")
    print(f"  hero ids referenced by matches: {report['hero_ids_referenced']}")
    print(f"  hero ids resolved: {report['hero_ids_resolved']}")
    print(f"  hero ids unresolved: {report['hero_ids_unresolved']}")
    print(f"  reference heroes never observed in corpus: {report['hero_ids_unreferenced']}")
    print(f"  duplicate hero ids: {report['hero_duplicate_ids']}")
    print(f"  heroes with null/empty name: {report['hero_null_names']}")

    print("\n=== Leagues ===")
    print(f"  canonical league registry count: {report['league_count']}")
    print(f"  league ids referenced by matches: {report['league_ids_referenced']}")
    print(f"  league ids resolved: {report['league_ids_resolved']}")
    print(f"  league ids unresolved: {report['league_ids_unresolved']}")
    print(f"  duplicate league names (distinct ids): {report['league_name_duplicates']}")

    print("\n=== Game versions ===")
    print(f"  canonical game-version count: {report['game_version_count']}")
    print(f"  game-version ids referenced by matches: {report['game_version_ids_referenced']}")
    print(f"  game-version ids resolved: {report['game_version_ids_resolved']}")
    print(f"  game-version ids unresolved: {report['game_version_ids_unresolved']}")
    print(
        "  reference patches never observed in corpus: "
        f"{report['game_version_ids_unreferenced']}"
    )
    print(f"  duplicate game-version ids: {report['game_version_duplicate_ids']}")
    print(f"  matches with null game_version_id: {report['null_game_version_matches']}")
    print(
        "  first-seen-in-corpus (derived; NOT authoritative release date): "
        f"{report['game_version_first_seen_in_corpus']}"
    )

    print("\n=== Regions ===")
    print(f"  status: {regions['status']}")
    print(f"  reason: {regions['reason']}")

    violations = report["integrity_violations"]
    print("\n=== Integrity violations ===")
    if not violations:
        print("  none")
    for violation in violations:
        print(f"  - {violation}")


def main() -> int:
    load_project_env(REPO_ROOT)
    engine = get_engine()
    reference_config = load_reference_store_config(root=REPO_ROOT)
    report = audit_reference_entities(
        engine,
        heroes_path=reference_config.heroes_path,
        game_versions_path=reference_config.game_versions_path,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
