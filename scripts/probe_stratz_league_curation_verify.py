"""Verify league IDs in config/leagues.yaml against live STRATZ GraphQL.

Fetches each configured league_id and reports name/tier mismatches or missing
leagues. Writes results to data/raw/league_curation_verify.json.

Usage:
    uv run python scripts/probe_stratz_league_curation_verify.py
    uv run python scripts/probe_stratz_league_curation_verify.py --config config/leagues.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import yaml

GRAPHQL_ENDPOINT = "https://api.stratz.com/graphql"
USER_AGENT = "dota-predictor-league-curation-verify/0.1"
TIMEOUT_SECONDS = 30.0
REQUEST_DELAY_SECONDS = 0.35

SINGLE_LEAGUE_QUERY = """
query ProbeLeague($id: Int!) {
  league(id: $id) {
    id
    name
    displayName
    tier
    prizePool
    startDateTime
    endDateTime
  }
}
"""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_project_env(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def gql(client: httpx.Client, query: str, variables: dict) -> dict:
    response = client.post(
        GRAPHQL_ENDPOINT, json={"query": query, "variables": variables}
    )
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root() / "config" / "leagues.yaml")
    args = parser.parse_args()

    root = project_root()
    load_project_env(root)
    token = os.environ.get("STRATZ_API_TOKEN", "").strip()
    if not token:
        print("STRATZ_API_TOKEN is missing.", file=sys.stderr)
        return 1

    entries = yaml.safe_load(args.config.read_text(encoding="utf-8")).get("leagues") or []
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    results: list[dict] = []
    problems: list[str] = []

    with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS) as client:
        for entry in entries:
            league_id = entry["league_id"]
            payload = gql(client, SINGLE_LEAGUE_QUERY, {"id": league_id})
            league = (payload.get("data") or {}).get("league")
            errors = payload.get("errors")
            row = {
                "league_id": league_id,
                "configured_name": entry["name"],
                "in_scope": entry.get("in_scope"),
                "liquipedia_tier": entry.get("liquipedia_tier"),
                "errors": errors,
                "league": league,
            }
            results.append(row)

            if errors or league is None:
                problems.append(f"id={league_id}: missing or error")
            else:
                display = league.get("displayName") or league.get("name") or ""
                tier = league.get("tier")
                if entry.get("stratz_tier") and tier != entry.get("stratz_tier"):
                    problems.append(
                        f"id={league_id}: stratz_tier yaml={entry['stratz_tier']!r} live={tier!r}"
                    )
                scope = "IN" if entry.get("in_scope") else "OUT"
                print(f"[{scope}] {league_id:6} | {tier:20} | {display}")
            time.sleep(REQUEST_DELAY_SECONDS)

    out_path = root / "data" / "raw" / "league_curation_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")

    if problems:
        print("\nIssues:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
