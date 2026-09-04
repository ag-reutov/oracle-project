"""Match Liquipedia Tier 3 event names to STRATZ league IDs.

NOT the authoritative T3 ID source. STRATZ uses season umbrellas and
LeagueTier metadata; Liquipedia ``|leagueid=`` (see
``scripts/discover_tier3_league_ids.py``) is authoritative.

Kept as a rejected-approach helper. Do not copy its IDs into the registry.


Usage:
    uv run python scripts/match_tier3_ids_via_stratz.py
"""

from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from dota_predictor.utils.env import load_project_env

REPO = Path(__file__).resolve().parents[1]
EVENTS_PATH = REPO / "data" / "interim" / "tier3_discovery.json"
OUT_PATH = REPO / "data" / "interim" / "tier3_discovery_with_ids.json"
DUMP_PATH = REPO / "data" / "interim" / "stratz_leagues_dump.json"

QUERY = """
query RecentLeagues($request: LeagueRequestType!) {
  leagues(request: $request) {
    id
    name
    displayName
    tier
    startDateTime
    endDateTime
    lastMatchDate
    prizePool
  }
}
"""


def _token() -> str:
    load_project_env(REPO)
    for key in ("STRATZ_API_TOKEN", "STRATZ_TOKEN", "STRATZ_API_KEY"):
        value = os.environ.get(key)
        if value:
            return value.strip().strip('"').strip("'")
    env_path = REPO / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "STRATZ" in line and "TOKEN" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("No STRATZ API token found in environment/.env")


def norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def dump_stratz_leagues(token: str) -> list[dict]:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "dota-predictor-tier3-discovery/0.1",
        "Content-Type": "application/json",
    }
    all_leagues: list[dict] = []
    with httpx.Client(timeout=60.0) as client:
        for skip in range(0, 8000, 100):
            payload = {
                "query": QUERY,
                "variables": {
                    "request": {
                        "take": 100,
                        "skip": skip,
                        "orderBy": "LAST_MATCH_TIME",
                    }
                },
            }
            response = client.post(
                "https://api.stratz.com/graphql", headers=headers, json=payload
            )
            response.raise_for_status()
            body = response.json()
            if body.get("errors"):
                print(f"GraphQL errors at skip={skip}: {body['errors'][:1]}")
                break
            batch = (body.get("data") or {}).get("leagues") or []
            print(f"skip={skip} got={len(batch)}", flush=True)
            if not batch:
                break
            all_leagues.extend(batch)
            time.sleep(0.4)
    DUMP_PATH.write_text(json.dumps(all_leagues, indent=2), encoding="utf-8")
    print(f"Wrote {DUMP_PATH} ({len(all_leagues)} leagues)")
    return all_leagues


def best_match(
    name: str, index: list[tuple[str, dict]]
) -> tuple[dict | None, float, str]:
    needle = norm(name)
    exact = [league for key, league in index if key == needle]
    if exact:
        return exact[0], 1.0, "exact"
    contains = [league for key, league in index if needle in key or key in needle]
    if len(contains) == 1:
        return contains[0], 0.92, "contains"
    if contains:
        scored = sorted(
            (
                (
                    SequenceMatcher(
                        None,
                        needle,
                        norm(league.get("displayName") or league.get("name") or ""),
                    ).ratio(),
                    int(league["id"]),
                    league,
                )
                for league in contains
            ),
            reverse=True,
        )
        return scored[0][2], scored[0][0], "contains_best"
    best_league = None
    best_ratio = 0.0
    for key, league in index:
        ratio = SequenceMatcher(None, needle, key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_league = league
    if best_ratio >= 0.88:
        return best_league, best_ratio, "fuzzy"
    return None, best_ratio, "none"


def main() -> int:
    token = _token()
    if DUMP_PATH.exists():
        leagues = json.loads(DUMP_PATH.read_text(encoding="utf-8"))
        print(f"Loaded {len(leagues)} leagues from {DUMP_PATH}")
    else:
        leagues = dump_stratz_leagues(token)

    events = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    prior = {
        row["page"]: row
        for row in json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if OUT_PATH.exists()
    } if OUT_PATH.exists() else {}

    index: list[tuple[str, dict]] = []
    for league in leagues:
        for key in (league.get("displayName"), league.get("name")):
            if key:
                index.append((norm(key), league))

    results: list[dict] = []
    matched = 0
    for event in events:
        existing = prior.get(event["page"])
        if existing and existing.get("league_id"):
            results.append(existing)
            matched += 1
            continue
        league, score, how = best_match(event["liquipedia_name"], index)
        if league is not None and score >= 0.88:
            results.append(
                {
                    **event,
                    "league_id": int(league["id"]),
                    "league_id_source": f"stratz_{how}:{score:.2f}",
                    "stratz_name": league.get("displayName") or league.get("name"),
                    "stratz_tier": league.get("tier"),
                }
            )
            matched += 1
        else:
            results.append(
                {
                    **event,
                    "league_id": None,
                    "league_id_source": None,
                    "match_score": score,
                    "match_how": how,
                }
            )

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"matched {matched}/{len(results)}")
    unmatched = [row for row in results if not row.get("league_id")]
    print(f"unmatched {len(unmatched)}")
    for row in unmatched[:40]:
        print(
            f"  {row['liquipedia_name']} score={row.get('match_score')} "
            f"how={row.get('match_how')}"
        )
    if len(unmatched) > 40:
        print(f"  ... +{len(unmatched) - 40} more")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
