"""Read-only historical-roster-state diagnostics (Slice 5).

Reports (never writes, never re-fetches, never classifies) the strictly
causal pre-match roster state derived from canonical match appearances:

* Team-match state -- one row per (team_id, match_id): the previous
  observed team match, retained/changed/same-lineup fields, prior
  exact-lineup experience, and the team composition counts.
* Player-team state -- one row per (player_id, team_id, match_id): prior
  team-match count, previous observed match (any team), and the mutually
  exclusive first-observed / returning / continuing flags.
* Integrity -- incomplete lineups, impossible aggregate counts, and the
  future-deletion invariant check (a future roster change must never
  alter an earlier historical state).

Stand-ins / transfers / contracts are NOT classified: returning is a
purely observational label (A -> B -> A), never a signing.

Usage:
    uv run python scripts/audit_roster_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dota_predictor.data.roster_state import audit_roster_state
from dota_predictor.storage.engine import get_engine
from dota_predictor.utils.env import load_project_env


def _print_dist(name: str, dist: dict[str, int]) -> None:
    print(f"    {name}: min={dist['min']} median={dist['median']} "
          f"max={dist['max']} (n={dist['count']})")


def _print_histogram(name: str, hist: dict[str, int]) -> None:
    parts = ", ".join(f"{k}: {v}" for k, v in hist.items())
    print(f"    {name}: [{parts}]")


def _print_report(report: dict[str, object]) -> None:
    obs = report["observations"]
    print("=== Observations ===")
    print(f"  total usable player-team-match observations: {obs['total_usable_observations']}")
    print(f"  observations with NULL player_id: {obs['null_player_id_observations']}")
    print(f"  observations with NULL derived team_id: {obs['null_team_id_observations']}")

    team = report["team_match_state"]
    print("\n=== Team-match roster state ===")
    print(f"  team-match rows: {team['team_match_rows']}")
    print(f"  rows with previous observed team match: {team['rows_with_previous_team_match']}")
    print(f"  first observed team matches: {team['first_observed_team_matches']}")
    _print_histogram("retained-player distribution (0-5)", team["retained_player_distribution"])
    _print_histogram("changed-player distribution (0-5)", team["changed_player_distribution"])
    print(f"  same-lineup-as-previous count: {team['same_lineup_as_previous_count']}")
    print(f"  exact-lineup first-use count: {team['exact_lineup_first_use_count']}")
    print(f"  exact-lineup repeat-use count: {team['exact_lineup_repeat_use_count']}")
    _print_dist("prior exact-lineup count", team["prior_exact_lineup_count_distribution"])

    player = report["player_team_state"]
    print("\n=== Player-team state ===")
    print(f"  player-match rows: {player['player_match_rows']}")
    print(f"  continuing observations: {player['continuing_observations']}")
    print(f"  first-observed-for-team observations: {player['first_observed_for_team_observations']}")
    print(f"  returning-to-team observations: {player['returning_to_team_observations']}")
    _print_dist("prior team-match count", player["prior_team_match_count_distribution"])
    print(f"  players with previous observed team unavailable: "
          f"{player['players_with_previous_observed_team_unavailable']}")

    integrity = report["integrity"]
    print("\n=== Integrity ===")
    print(f"  incomplete lineups: {integrity['incomplete_lineup_count']}")
    print(f"  impossible aggregate counts: {integrity['impossible_aggregate_counts']}")
    print(f"  current-match-in-own-prior-counts: {integrity['current_match_in_own_prior_counts']}")
    invariant = integrity["future_deletion_invariant"]
    print(f"  future-deletion invariant: {invariant['matches_checked']} matches checked")
    print(f"    team-state violations: {len(invariant['team_state_violations'])}")
    for violation in invariant["team_state_violations"]:
        print(f"      {violation}")
    print(f"    player-team-state violations: {len(invariant['player_state_violations'])}")
    for violation in invariant["player_state_violations"]:
        print(f"      {violation}")


def main() -> int:
    load_project_env(REPO_ROOT)
    engine = get_engine()
    report = audit_roster_state(engine)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())