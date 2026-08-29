"""Disposable STRATZ GraphQL probe: league/tournament tier semantics.

Purpose (see task report): investigate whether STRATZ's live GraphQL
`LeagueTier` enum on `LeagueType.tier` is sufficient, on its own, to define
the "Tier 1 + Tier 2 professional Dota" universe for bulk tournament
discovery -- or whether a manually curated Liquipedia-tier registry (or at
least an exclusion/override list) is required.

This is NOT a production ingestion tool and does not write anything outside
`data/raw/`. It:

1. Introspects `LeagueTier` (enum values + descriptions) and the `tier`
   field on `LeagueType` (description), directly against the live schema
   -- not cached from an earlier probe run.
2. Fetches a fixed, by-name/ID sample of known leagues spanning the
   requested spectrum (TI, Riyadh Masters/EWC-level, DreamLeague, ESL One,
   a clear Liquipedia Tier 2 event, a qualifier, and a small/regional
   event), reading back `id`, `name`, `displayName`, `tier`, `region`,
   `prizePool`, `startDateTime`/`endDateTime`.
3. Also pulls a broad, unfiltered recent-leagues listing (ordered by last
   match time) to see the natural distribution of tier values and to spot
   any known-large events that got an unexpected/low tier value, or
   known-small events that got an unexpectedly high tier value.

Never print or persist the API token or Authorization headers.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

GRAPHQL_ENDPOINT = "https://api.stratz.com/graphql"
USER_AGENT = "dota-predictor-stratz-tier-probe/0.1"
TIMEOUT_SECONDS = 30.0
REQUEST_DELAY_SECONDS = 0.35

# Known STRATZ league IDs spanning the spectrum the task asked us to sample.
# IDs were sourced by cross-referencing this repo's existing verification
# probe (TI entries) plus independently-verified STRATZ league IDs (via
# web search of stratz.com/leagues/<id> pages and third-party sites that
# cite the same STRATZ league id, e.g. dotadata.org/dotabuff/spectral.gg)
# for each named event. All are re-verified live below by fetching
# name/displayName alongside tier, so a wrong ID shows up immediately as a
# name mismatch rather than being trusted blindly. `liquipedia_tier` is
# this task's own annotation of the commonly-cited Liquipedia
# tier/category for that event -- NOT read from STRATZ -- recorded here
# purely so the report can compare STRATZ's tier against it.
CANDIDATE_LEAGUES: tuple[tuple[int, str, str], ...] = (
    # --- The International (Liquipedia Tier 1) ---
    (16935, "The International 2024", "T1"),
    (18324, "The International 2025", "T1"),
    # --- Esports World Cup / Riyadh Masters-level (Liquipedia Tier 1) ---
    (16881, "Riyadh Masters 2024 at Esports World Cup (main event)", "T1"),
    (16740, "Riyadh Masters 2024 at Esports World Cup (qualifiers)", "Qualifier"),
    # --- ESL One (Liquipedia Tier 1) ---
    (16518, "ESL One Birmingham 2024", "T1"),
    (17795, "ESL One Raleigh 2025 (main event)", "T1"),
    (17629, "ESL One Raleigh 2025 (closed qualifiers)", "Qualifier"),
    # --- DreamLeague (Liquipedia Tier 1) ---
    (17272, "DreamLeague Season 24", "T1"),
    (17765, "DreamLeague Season 25", "T1"),
    # --- BLAST Slam (Liquipedia Tier 1) ---
    (17419, "BLAST Slam IV", "T1"),
    (17420, "BLAST Slam V (ID cross-check candidate)", "T1"),
    (17418, "BLAST Slam Season 3 (ID cross-check candidate)", "T1"),
    # --- Clear Liquipedia Tier 2 events ---
    (16905, "Elite League Season 2 (main event)", "T2"),
    (16776, "Elite League Season 2 EEU Closed Qualifiers", "Qualifier"),
    (17907, "FISSURE Universe: Episode 4", "T2"),
    (18433, "FISSURE Universe: Episode 6", "T2"),
    # --- DPC-era (2021-2023) leagues, for historical comparison ---
    (14892, "DPC 2023 WEU Winter Tour Division I", "T2 (DPC-era)"),
    (14050, "DPC NA Division II Spring Tour 2021/2022", "T2 (DPC-era)"),
    (15693, "Road to TI 2023 - WEU Regional Qualifiers", "Qualifier (DPC-era)"),
    # --- Small / regional / non-professional-scene leagues ---
    (10979, "StarLadder ImbaTV Dota2 Minor #2 (2019)", "T2 (Minor, DPC-era)"),
    (17807, "RD2L Season 38 (amateur inhouse/draft league)", "Amateur/regional"),
    (17662, "1win Series Dota 2 (Liquipedia Tier 2 2024 list)", "T2"),
)


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


def require_token() -> str:
    token = os.environ.get("STRATZ_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("STRATZ_API_TOKEN is missing.")
    return token


def gql(
    client: httpx.Client, query: str, variables: dict[str, Any] | None = None
) -> dict[str, Any]:
    response = client.post(
        GRAPHQL_ENDPOINT, json={"query": query, "variables": variables or {}}
    )
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = {"parse_error": True, "text_preview": response.text[:500]}
    return {"http_status": response.status_code, "payload": payload}


ENUM_INTROSPECTION_QUERY = """
{
  __type(name: "LeagueTier") {
    name
    kind
    description
    enumValues(includeDeprecated: true) {
      name
      description
      isDeprecated
      deprecationReason
    }
  }
}
"""

LEAGUE_TYPE_FIELD_QUERY = """
{
  __type(name: "LeagueType") {
    name
    fields(includeDeprecated: true) {
      name
      description
      isDeprecated
      deprecationReason
      type { kind name ofType { kind name } }
    }
  }
}
"""

LEAGUE_REQUEST_INPUT_QUERY = """
{
  __type(name: "LeagueRequestType") {
    name
    inputFields {
      name
      description
      type { kind name ofType { kind name ofType { kind name } } }
    }
  }
}
"""

SINGLE_LEAGUE_QUERY = """
query ProbeLeague($id: Int!) {
  league(id: $id) {
    id
    name
    displayName
    tier
    region
    prizePool
    basePrizePool
    startDateTime
    endDateTime
    lastMatchDate
    private
    hasLiveMatches
  }
}
"""

RECENT_LEAGUES_QUERY = """
query RecentLeagues($request: LeagueRequestType!) {
  leagues(request: $request) {
    id
    name
    displayName
    tier
    region
    prizePool
    startDateTime
    endDateTime
    lastMatchDate
  }
}
"""


def fetch_single_league(client: httpx.Client, league_id: int) -> dict[str, Any]:
    result = gql(client, SINGLE_LEAGUE_QUERY, {"id": league_id})
    payload = result["payload"]
    return {
        "requested_id": league_id,
        "http_status": result["http_status"],
        "errors": payload.get("errors"),
        "league": (payload.get("data") or {}).get("league"),
    }


def fetch_recent_leagues(
    client: httpx.Client, *, take: int, skip: int = 0
) -> dict[str, Any]:
    request = {"take": take, "skip": skip, "orderBy": "LAST_MATCH_TIME"}
    result = gql(client, RECENT_LEAGUES_QUERY, {"request": request})
    payload = result["payload"]
    return {
        "request": request,
        "http_status": result["http_status"],
        "errors": payload.get("errors"),
        "leagues": (payload.get("data") or {}).get("leagues") or [],
    }


def fetch_leagues_by_tier(
    client: httpx.Client, tier: str, *, take: int
) -> dict[str, Any]:
    request = {"take": take, "skip": 0, "tiers": [tier], "orderBy": "LAST_MATCH_TIME"}
    result = gql(client, RECENT_LEAGUES_QUERY, {"request": request})
    payload = result["payload"]
    return {
        "tier": tier,
        "request": request,
        "http_status": result["http_status"],
        "errors": payload.get("errors"),
        "leagues": (payload.get("data") or {}).get("leagues") or [],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    root = project_root()
    load_project_env(root)
    token = require_token()
    raw_dir = root / "data" / "raw"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    output: dict[str, Any] = {"endpoint": GRAPHQL_ENDPOINT}

    with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS) as client:
        print("1) Introspecting LeagueTier enum (live) ...")
        enum_result = gql(client, ENUM_INTROSPECTION_QUERY)
        output["league_tier_enum_introspection"] = enum_result["payload"]
        enum_type = ((enum_result["payload"].get("data") or {}).get("__type")) or {}
        tier_names = [v["name"] for v in (enum_type.get("enumValues") or [])]
        print(f"   LeagueTier values: {tier_names}")
        time.sleep(REQUEST_DELAY_SECONDS)

        print("2) Introspecting LeagueType.tier field description ...")
        field_result = gql(client, LEAGUE_TYPE_FIELD_QUERY)
        output["league_type_field_introspection"] = field_result["payload"]
        time.sleep(REQUEST_DELAY_SECONDS)

        print("3) Introspecting LeagueRequestType input fields (filter surface) ...")
        input_result = gql(client, LEAGUE_REQUEST_INPUT_QUERY)
        output["league_request_input_introspection"] = input_result["payload"]
        time.sleep(REQUEST_DELAY_SECONDS)

        print("4) Fetching known/candidate leagues by ID ...")
        candidate_results = []
        for league_id, expected_label, liquipedia_tier in CANDIDATE_LEAGUES:
            fetched = fetch_single_league(client, league_id)
            fetched["expected_label"] = expected_label
            fetched["liquipedia_tier"] = liquipedia_tier
            candidate_results.append(fetched)
            league = fetched.get("league") or {}
            print(
                f"   id={league_id} expected='{expected_label}' (liquipedia={liquipedia_tier}) ->"
                f" name={league.get('displayName') or league.get('name')!r}"
                f" tier={league.get('tier')}"
                f" region={league.get('region')}"
                f" prizePool={league.get('prizePool')}"
            )
            time.sleep(REQUEST_DELAY_SECONDS)
        output["candidate_leagues"] = candidate_results

        print("5) Fetching broad recent-leagues listing (no tier filter) ...")
        recent = fetch_recent_leagues(client, take=100, skip=0)
        output["recent_leagues_unfiltered"] = recent
        print(f"   returned {len(recent['leagues'])} leagues")
        time.sleep(REQUEST_DELAY_SECONDS)

        print("6) Fetching a sample per tier value (for distribution/spot-check) ...")
        per_tier = []
        for tier in tier_names:
            if tier == "UNSET":
                continue
            sample = fetch_leagues_by_tier(client, tier, take=15)
            per_tier.append(sample)
            print(f"   tier={tier}: {len(sample['leagues'])} leagues returned")
            time.sleep(REQUEST_DELAY_SECONDS)
        output["per_tier_samples"] = per_tier

    out_path = raw_dir / "stratz_league_tier_probe.json"
    write_json(out_path, output)
    print(f"\nWrote: {out_path}")

    # Redact check: refuse to leave the token in any saved file.
    token_pattern = re.compile(re.escape(token))
    if token_pattern.search(out_path.read_text(encoding="utf-8")):
        out_path.write_text("{}\n", encoding="utf-8")
        print(f"warning: token-like value found in {out_path.name}; file wiped.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
