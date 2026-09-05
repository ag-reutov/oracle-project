"""Read-only Slice 6 team-strength diagnostics (rebuild audit + raw latest Elo).

Reports (never writes, never re-fetches, never classifies):

* Build freshness / staleness (`research.team_strength_build` source
  fingerprint vs the current canonical corpus).
* Historical-state census and the production-Elo cross-check
  (`audit_team_strength`). This contains NO leaderboard and NO Top-20: it
  reports latest raw team-ID Elo as a distribution (min/median/max), never
  as an ordinal ranking.
* Activity-age distribution of the latest raw team-ID Elo (days since last
  observed match at corpus end, bucketed; nobody is filtered out).
* Elo population / tier composition over the exact production universe.
* Conservative identity-fragmentation candidates (candidate `team_id`
  pairs/groups that may share a competitive lineage; nothing is merged).

The raw `team_id` latest-Elo state is diagnostic only: raw canonical
`team_id` is not equivalent to a current competitive team identity (identity
fragmentation, disbanded teams, no activity eligibility rule, Tier-3-heavy
population). Slice 7 will define competitive-team lineage, active-team
eligibility, rating population, and the actual current power ranking.

`--show-raw-elo` is an OPT-IN debugging flag that prints the raw latest
team-ID Elo values sorted by Elo for inspection only; it is not a ranking
and is never shown by default.

Usage:
    uv run python scripts/audit_team_strength.py
    uv run python scripts/audit_team_strength.py --no-fragmentation
    uv run python scripts/audit_team_strength.py --show-raw-elo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dota_predictor.data.team_strength import (
    audit_activity_distribution,
    audit_elo_population,
    audit_identity_fragmentation,
    audit_raw_elo_latest,
    audit_team_strength,
    check_freshness,
)
from dota_predictor.storage.engine import get_engine
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Slice 6 team-strength diagnostics."
    )
    parser.add_argument(
        "--no-fragmentation",
        action="store_true",
        help="Skip the identity-fragmentation candidate scan (can be slow).",
    )
    parser.add_argument(
        "--show-raw-elo",
        action="store_true",
        help=(
            "OPT-IN debugging output: print raw latest team-ID Elo values "
            "sorted by Elo (NOT a ranking; never shown by default)."
        ),
    )
    return parser


def _default(value: object) -> object:
    if isinstance(value, dict):
        return {k: _default(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_default(v) for v in value]
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = _project_root()
    load_project_env(root)

    engine = get_engine()

    print("=== Freshness / staleness (research.team_strength_state) ===")
    print(json.dumps(check_freshness(engine), indent=2, sort_keys=True, default=_default))
    print()

    print("=== Historical census + production-Elo cross-check (no leaderboard) ===")
    audit = audit_team_strength(engine)
    print(json.dumps(audit, indent=2, sort_keys=True, default=_default))
    print()

    print("=== Activity-age distribution (days since last observed match) ===")
    print(json.dumps(audit_activity_distribution(engine), indent=2, sort_keys=True, default=_default))
    print()

    print("=== Elo population / tier composition ===")
    print(json.dumps(audit_elo_population(engine), indent=2, sort_keys=True, default=_default))
    print()

    if not args.no_fragmentation:
        print("=== Identity-fragmentation candidates (conservative, no merge) ===")
        fragmentation = audit_identity_fragmentation(engine)
        print(json.dumps(fragmentation, indent=2, sort_keys=True, default=_default))
        print()

    if args.show_raw_elo:
        print("=== Raw latest team-ID Elo values (OPT-IN DEBUGGING, NOT A RANKING) ===")
        print(json.dumps(audit_raw_elo_latest(engine), indent=2, sort_keys=True, default=_default))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())