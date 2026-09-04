"""Read-only observed-roster-history diagnostics (Slice 4).

Reports (never writes, never re-fetches, never classifies) the observed
competitive roster history derived from canonical match appearances:

* Observations -- total usable player-team-match observations and how many
  rows were skipped for a NULL player id or NULL derived team id.
* Match lineups -- team-match lineups examined and the explicit cardinality
  audit: exactly-five / fewer-than-five / greater-than-five / duplicate
  player / null player anomalies. Malformed lineups are reported, never
  forced into a five-player shape.
* Player-team history -- unique players / teams / player-team pairs,
  players observed for exactly one vs multiple teams, total observed
  spells, one-match spells, players who return to a previous team after an
  intervening team (A -> B -> A), and the maximum observed team count for
  any player, plus distribution summaries.
* Integrity -- unresolved canonical player/team references (0 by foreign
  key) and duplicate/contradictory observations (a duplicate
  (match_id, player_id) or a player on both sides of one match).

Stand-ins / temporary appearances are NOT classified semantically: a
one-match spell is reported as `observed_match_count = 1` with its
previous/next observed team available in `research.player_team_spells`,
but no `standin` label is ever assigned. Official contractual roster
history is explicitly out of scope for this slice.

Usage:
    uv run python scripts/audit_roster_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dota_predictor.data.roster_history import audit_roster_history
from dota_predictor.storage.engine import get_engine
from dota_predictor.utils.env import load_project_env


def _print_dist(name: str, dist: dict[str, int]) -> None:
    print(f"    {name}: min={dist['min']} median={dist['median']} "
          f"max={dist['max']} (n={dist['count']})")


def _print_report(report: dict[str, object]) -> None:
    obs = report["observations"]
    print("=== Observations ===")
    print(f"  total usable player-team-match observations: {obs['total_player_team_match_observations']}")
    print(f"  observations with NULL player_id: {obs['null_player_id_observations']}")
    print(f"  observations with NULL derived team_id: {obs['null_team_id_observations']}")

    lu = report["match_lineups"]
    print("\n=== Match lineups ===")
    print(f"  team-match lineups examined: {lu['lineups_examined']}")
    print(f"  exactly-five lineups: {lu['exactly_five']}")
    print(f"  fewer-than-five lineups: {lu['fewer_than_five']}")
    print(f"  greater-than-five lineups: {lu['more_than_five']}")
    print(f"  duplicate-player anomalies: {lu['duplicate_player_anomalies']}")
    print(f"  null-player anomalies: {lu['null_player_anomalies']}")

    hist = report["player_team_history"]
    print("\n=== Player-team history (observed) ===")
    print(f"  unique players: {hist['unique_players']}")
    print(f"  unique teams: {hist['unique_teams']}")
    print(f"  unique player-team pairs: {hist['unique_player_team_pairs']}")
    print(f"  players observed for exactly one team: {hist['players_one_team']}")
    print(f"  players observed for multiple teams: {hist['players_multi_team']}")
    print(f"  total observed spells: {hist['total_observed_spells']}")
    print(f"  one-match observed spells: {hist['one_match_spells']}")
    print(f"  players returning to a previous team (A -> B -> A): {hist['players_with_return']}")
    print(f"  maximum observed team count for any player: {hist['max_observed_team_count']}")
    _print_dist("teams per player", hist["teams_per_player"])
    _print_dist("spells per player", hist["spells_per_player"])
    _print_dist("observed spell lengths", hist["spell_lengths"])

    integrity = report["integrity"]
    print("\n=== Integrity ===")
    print(f"  unresolved canonical player references: {integrity['unresolved_canonical_player_rows']}")
    print(f"  unresolved canonical team references: {integrity['unresolved_canonical_team_rows']}")
    print(f"  duplicate (match_id, player_id) observations: {integrity['duplicate_player_match_observations']}")
    print(f"  player-on-both-sides observations: {integrity['player_on_both_sides_observations']}")

    tsi = report["team_side_inconsistencies"]
    print(f"  player/team-side inconsistencies: {tsi['count']} ({tsi['note']})")

    src = report["official_roster_source"]
    print(f"\n  official roster source available: {src['available']}")
    print(f"    -> {src['note']}")


def main() -> int:
    load_project_env(REPO_ROOT)
    engine = get_engine()
    report = audit_roster_history(engine)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())