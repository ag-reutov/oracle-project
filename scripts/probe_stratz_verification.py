"""Disposable STRATZ compatibility/verification probe (Tier 1 / Tier 2 sample).

This is NOT a production ingestion tool. It is a one-off evidence-gathering
script for the STRATZ verification task described in the project task
report. It:

1. Queries a deliberately-chosen, reproducible sample of real professional
   Dota 2 matches from STRATZ, stratified across Tier 1 / Tier 2
   tournaments, dates, and patch/game-version eras (see `TARGET_LEAGUES`
   below for the exact, hardcoded selection and the rationale for each
   entry).
2. Requests the specific fields the task asked to verify: durationSeconds,
   gameVersionId, leagueId, tournamentId, seriesId, series metadata,
   pickBans (all sub-fields), player ids, hero ids, radiant/dire team
   info, and the winner field.
3. Also probes `constants.gameVersions` (the id -> human-readable patch
   name lookup) and a small `playbackData` sample (to check for
   draft-timing fields).
4. Runs every sampled match through the existing
   `canonical_match_from_stratz` mapper and records success/failure.
5. Computes descriptive statistics needed for the verification report
   (draft `order` behavior, heroId/bannedHeroId agreement,
   wasBannedSuccessfully distribution, pick counts per side, series
   metadata reliability, etc.) without inferring semantics from single
   examples.
6. Writes:
   - a full raw dump of every sampled match to `data/raw/` (gitignored,
     disposable, for ad hoc re-inspection only), and
   - a JSON analysis summary to `data/raw/` (gitignored), and
   - compact, source-faithful, single-match fixtures for any match judged
     to be an anomaly worth preserving as a regression fixture, under
     `tests/data/fixtures/stratz_anomalies/` (tracked in git).

Sampling method (see task report for full detail): a fixed, manually
curated list of Tier 1 / Tier 2 leagues/tournaments spanning 2019-2025,
chosen for date and patch-era diversity, not "whatever STRATZ returns
first". Within each league, matches are taken positionally (first N, and
for long round-robin DPC leagues, an additional batch offset by `skip` to
reach a different point in the group stage) -- this is deterministic
top-of-list / systematic sampling, not random sampling. This is
disclosed explicitly rather than described as representative.

Never print or persist the API token or Authorization headers.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dota_predictor.data.canonical_schema import CanonicalMatchError
from dota_predictor.data.stratz_mapping import canonical_match_from_stratz

GRAPHQL_ENDPOINT = "https://api.stratz.com/graphql"
USER_AGENT = "dota-predictor-stratz-verification-probe/0.1"
TIMEOUT_SECONDS = 30.0
REQUEST_DELAY_SECONDS = 0.4


@dataclass(frozen=True)
class LeagueSpec:
    league_id: int
    label: str
    tier_bucket: str  # "T1" or "T2" -- this project's working classification
    stratz_tier: str
    era_note: str
    skips: tuple[int, ...] = (0,)
    take: int = 6


# Fixed, reproducible selection. NOT the output of "take whatever STRATZ
# returns first": specific tournaments were chosen by name/date to span
# 2019-2025 and multiple patch eras, within Tier 1 / Tier 2 scope only.
#
# Classification caveat (see task report): STRATZ's `LeagueTier` enum
# (AMATEUR / PROFESSIONAL / MINOR / MAJOR / INTERNATIONAL /
# DPC_QUALIFIER / DPC_LEAGUE_QUALIFIER / DPC_LEAGUE / DPC_LEAGUE_FINALS)
# does not map 1:1 onto the common Liquipedia-style Tier 1/Tier 2
# convention. DPC "Division I" leagues sit near the Tier 1/Tier 2
# boundary; they are bucketed here as T2 with that caveat noted. DPC
# "Division II" and regional Minors are unambiguous T2. Open/closed DPC
# qualifiers and AMATEUR/PROFESSIONAL-tier leagues were deliberately
# excluded from this sample per the task's scope instructions.
TARGET_LEAGUES: tuple[LeagueSpec, ...] = (
    LeagueSpec(
        10749, "The International 2019", "T1", "INTERNATIONAL", "patch ~7.22", take=20
    ),
    LeagueSpec(
        13256, "The International 2021", "T1", "INTERNATIONAL", "patch ~7.30", take=20
    ),
    LeagueSpec(
        14173, "ESL One Stockholm Major 2022", "T1", "MAJOR", "patch ~7.31", take=20
    ),
    LeagueSpec(
        15438, "The Bali Major 2023", "T1", "MAJOR", "patch ~7.33/7.34", take=20
    ),
    LeagueSpec(
        15728, "The International 2023", "T1", "INTERNATIONAL", "patch ~7.34", take=20
    ),
    LeagueSpec(
        16935, "The International 2024", "T1", "INTERNATIONAL", "patch ~7.36", take=20
    ),
    LeagueSpec(
        18324,
        "The International 2025",
        "T1",
        "INTERNATIONAL",
        "patch ~7.38/7.39",
        take=20,
    ),
    LeagueSpec(
        13960,
        "DPC 2021-2022 Tour 1 Regional Finals WEU (DreamLeague S16)",
        "T1",
        "DPC_LEAGUE_FINALS",
        "patch ~7.30/7.31",
        take=20,
    ),
    LeagueSpec(
        10979,
        "StarLadder ImbaTV Dota2 Minor #2 (2019)",
        "T2",
        "MINOR",
        "patch ~7.21/7.22",
        take=20,
    ),
    LeagueSpec(
        14050,
        "DPC NA Division II Spring Tour 2021/2022",
        "T2",
        "DPC_LEAGUE",
        "patch ~7.31",
        skips=(0, 20, 40),
        take=10,
    ),
    LeagueSpec(
        14892,
        "DPC 2023 WEU Winter Tour Division I",
        "T2",
        "DPC_LEAGUE",
        "patch ~7.32/7.33",
        skips=(0, 20, 40),
        take=10,
    ),
    LeagueSpec(
        15350,
        "DPC 2023 NA Summer Tour Division I",
        "T2",
        "DPC_LEAGUE",
        "patch ~7.33/7.34",
        skips=(0, 20, 40),
        take=10,
    ),
)

MATCH_SELECTION = """
id
didRadiantWin
durationSeconds
startDateTime
endDateTime
tournamentId
tournamentRound
leagueId
league { id name displayName tier region startDateTime endDateTime lastMatchDate prizePool }
seriesId
series {
  id
  type
  teamOneId
  teamTwoId
  leagueId
  teamOneWinCount
  teamTwoWinCount
  winningTeamId
  lastMatchDateTime
  matches { id startDateTime }
}
gameVersionId
radiantTeamId
direTeamId
radiantTeam { id name tag }
direTeam { id name tag }
players { steamAccountId isRadiant playerSlot heroId }
pickBans {
  isPick
  heroId
  order
  bannedHeroId
  isRadiant
  playerIndex
  wasBannedSuccessfully
  isCaptain
  letter
  baseWinRate
  adjustedWinRate
}
"""

MATCHES_QUERY = f"""
query VerificationSample($id: Int!, $request: LeagueMatchesRequestType!) {{
  league(id: $id) {{
    id
    name
    displayName
    tier
    matches(request: $request) {{
      {MATCH_SELECTION}
    }}
  }}
}}
"""

PLAYBACK_QUERY = """
query PlaybackProbe($id: Long!) {
  match(id: $id) {
    id
    playbackData {
      radiantCaptainHeroId
      direCaptainHeroId
      runeEvents { time }
      wardEvents { time }
      buildingEvents { time }
      towerDeathEvents { time }
      roshanEvents { time }
    }
  }
}
"""

GAME_VERSIONS_QUERY = "{ constants { gameVersions { id name asOfDateTime } } }"


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


def fetch_sample(client: httpx.Client) -> list[dict[str, Any]]:
    all_matches: list[dict[str, Any]] = []
    for spec in TARGET_LEAGUES:
        for skip in spec.skips:
            request = {"take": spec.take, "skip": skip}
            result = gql(
                client, MATCHES_QUERY, {"id": spec.league_id, "request": request}
            )
            payload = result["payload"]
            if payload.get("errors"):
                print(
                    f"  [{spec.label}] skip={skip}: GraphQL errors: {payload['errors']}"
                )
                time.sleep(REQUEST_DELAY_SECONDS)
                continue
            league_data = (payload.get("data") or {}).get("league")
            matches = (league_data or {}).get("matches") or []
            print(
                f"  [{spec.label}] ({spec.tier_bucket}, {spec.stratz_tier}, {spec.era_note})"
                f" skip={skip}: {len(matches)} matches"
            )
            for m in matches:
                m["_sample_league_label"] = spec.label
                m["_sample_tier_bucket"] = spec.tier_bucket
                m["_sample_stratz_tier"] = spec.stratz_tier
                m["_sample_era_note"] = spec.era_note
                all_matches.append(m)
            time.sleep(REQUEST_DELAY_SECONDS)
    return all_matches


def fetch_game_versions(client: httpx.Client) -> list[dict[str, Any]]:
    result = gql(client, GAME_VERSIONS_QUERY)
    payload = result["payload"]
    if payload.get("errors"):
        print("  gameVersions: GraphQL errors:", payload["errors"])
        return []
    return ((payload.get("data") or {}).get("constants") or {}).get(
        "gameVersions"
    ) or []


def fetch_playback_sample(
    client: httpx.Client, match_ids: list[int]
) -> list[dict[str, Any]]:
    out = []
    for match_id in match_ids:
        result = gql(client, PLAYBACK_QUERY, {"id": match_id})
        payload = result["payload"]
        if payload.get("errors"):
            print(f"  playbackData[{match_id}]: GraphQL errors:", payload["errors"])
            continue
        out.append((payload.get("data") or {}).get("match"))
        time.sleep(REQUEST_DELAY_SECONDS)
    return out


def order_analysis(pick_bans: list[dict[str, Any]]) -> dict[str, Any]:
    orders = [row.get("order") for row in pick_bans if row.get("order") is not None]
    if not orders:
        return {
            "orders": [],
            "min": None,
            "max": None,
            "gap_free": None,
            "has_duplicates": None,
        }
    sorted_orders = sorted(orders)
    expected = list(range(min(orders), max(orders) + 1))
    return {
        "count": len(orders),
        "min": min(orders),
        "max": max(orders),
        "zero_based": min(orders) == 0,
        "gap_free": sorted_orders == expected,
        "has_duplicates": len(set(orders)) != len(orders),
        "duplicate_values": sorted(v for v, c in Counter(orders).items() if c > 1),
    }


def hero_id_vs_banned_hero_id(pick_bans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in pick_bans:
        hero_id = row.get("heroId")
        banned_hero_id = row.get("bannedHeroId")
        rows.append(
            {
                "isPick": row.get("isPick"),
                "heroId": hero_id,
                "bannedHeroId": banned_hero_id,
                "agree": (hero_id == banned_hero_id)
                if row.get("isPick") is False
                else None,
                "heroId_null": hero_id is None,
                "bannedHeroId_null": banned_hero_id is None,
            }
        )
    return rows


@dataclass
class MapperOutcome:
    match_id: Any
    ok: bool
    error: str | None = None


def classify_failure(error_message: str) -> str:
    msg = error_message.lower()
    if "durationseconds" in msg:
        return "missing_queried_field(durationSeconds)"
    if "leagueid" in msg:
        return "missing_queried_field(leagueId)"
    if "exactly 5" in msg:
        return "schema_assumption_too_strict(five_picks)"
    if "sequence == position" in msg:
        return "schema_assumption_too_strict_or_source_ordering(draft_sequence)"
    if "more than one actual draft action" in msg:
        return "possible_mapper_bug_or_source_anomaly(duplicate_hero)"
    if "radiantteamid" in msg or "direteamid" in msg:
        return "missing_queried_field(team_id)"
    if "steamaccountid" in msg:
        return "missing_queried_field(steamAccountId)"
    if "didradiantwin" in msg:
        return "missing_queried_field(didRadiantWin)"
    return "other"


ANOMALY_DIR_NAME = "tests/data/fixtures/stratz_anomalies"


def detect_and_preserve_anomalies(
    root: Path, matches: list[dict[str, Any]], mapper_errors: dict[Any, str]
) -> list[dict[str, str]]:
    anomaly_dir = root / ANOMALY_DIR_NAME
    preserved: list[dict[str, str]] = []

    def save(match: dict[str, Any], reason: str, slug: str) -> None:
        anomaly_dir.mkdir(parents=True, exist_ok=True)
        path = anomaly_dir / f"{match.get('id')}_{slug}.json"
        if path.exists():
            return
        # Keep the raw match object source-faithful: strip only the
        # probe's own bookkeeping keys (prefixed with "_sample_"), which
        # are not part of the STRATZ response.
        clean = {k: v for k, v in match.items() if not k.startswith("_sample_")}
        path.write_text(
            json.dumps(clean, indent=2, default=str) + "\n", encoding="utf-8"
        )
        preserved.append(
            {"match_id": str(match.get("id")), "reason": reason, "path": str(path)}
        )

    for match in matches:
        match_id = match.get("id")
        pick_bans = match.get("pickBans") or []
        players = match.get("players") or []

        failed_bans = [
            row for row in pick_bans if row.get("wasBannedSuccessfully") is False
        ]
        if failed_bans:
            save(match, "wasBannedSuccessfully_false observed", "failed_ban")

        oa = order_analysis(pick_bans)
        if oa.get("has_duplicates"):
            save(match, "duplicate draft order values", "duplicate_order")
        if oa.get("orders") == []:
            pass
        elif oa.get("gap_free") is False:
            save(match, "gap in draft order sequence", "order_gap")
        if oa.get("min") not in (None, 0):
            save(
                match,
                f"draft order does not start at 0 (min={oa.get('min')})",
                "nonzero_order_start",
            )

        radiant_picks = sum(
            1 for row in pick_bans if row.get("isPick") and row.get("isRadiant") is True
        )
        dire_picks = sum(
            1
            for row in pick_bans
            if row.get("isPick") and row.get("isRadiant") is False
        )
        if radiant_picks != 5 or dire_picks != 5:
            save(
                match,
                f"fewer/more than 5 picks per side (radiant={radiant_picks}, dire={dire_picks})",
                "irregular_pick_count",
            )

        if match.get("durationSeconds") is None:
            save(match, "missing durationSeconds", "missing_duration")
        if match.get("didRadiantWin") is None:
            save(match, "missing didRadiantWin", "missing_winner")
        if match.get("radiantTeamId") is None or match.get("direTeamId") is None:
            save(match, "missing team id", "missing_team_id")
        if any(p.get("steamAccountId") is None for p in players):
            save(match, "missing player steamAccountId", "missing_player_id")
        if not pick_bans:
            save(
                match, "empty pickBans (incomplete/abandoned draft?)", "empty_pickbans"
            )

        for row in pick_bans:
            if (
                row.get("isPick")
                and row.get("heroId") != row.get("bannedHeroId")
                and row.get("bannedHeroId") is not None
            ):
                save(
                    match,
                    "heroId/bannedHeroId disagree on a pick row",
                    "heroid_bannedheroid_disagree",
                )

        if match_id in mapper_errors:
            classification = classify_failure(mapper_errors[match_id])
            if classification in {
                "other",
                "possible_mapper_bug_or_source_anomaly(duplicate_hero)",
            }:
                save(
                    match,
                    f"mapper failure: {mapper_errors[match_id]}",
                    "mapper_failure",
                )

    return preserved


def main() -> int:
    root = project_root()
    load_project_env(root)
    token = require_token()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS) as client:
        print("Fetching constants.gameVersions ...")
        game_versions = fetch_game_versions(client)
        print(f"  {len(game_versions)} game versions returned")

        print("Fetching stratified Tier1/Tier2 match sample ...")
        matches = fetch_sample(client)
        print(f"Total matches fetched: {len(matches)}")

        print("Fetching playbackData sample (3 matches) ...")
        playback_sample_ids = [m["id"] for m in matches[:3]]
        playback_samples = fetch_playback_sample(client, playback_sample_ids)

    # Run the existing mapper against every sampled match.
    outcomes: list[MapperOutcome] = []
    mapper_errors: dict[Any, str] = {}
    for match in matches:
        clean = {k: v for k, v in match.items() if not k.startswith("_sample_")}
        try:
            canonical_match_from_stratz(clean)
            outcomes.append(MapperOutcome(match_id=match.get("id"), ok=True))
        except CanonicalMatchError as exc:
            outcomes.append(
                MapperOutcome(match_id=match.get("id"), ok=False, error=str(exc))
            )
            mapper_errors[match.get("id")] = str(exc)
        except Exception as exc:  # noqa: BLE001 - want to see any unexpected mapper crash
            outcomes.append(
                MapperOutcome(
                    match_id=match.get("id"), ok=False, error=f"UNEXPECTED: {exc!r}"
                )
            )
            mapper_errors[match.get("id")] = f"UNEXPECTED: {exc!r}"

    ok_count = sum(1 for o in outcomes if o.ok)
    fail_count = len(outcomes) - ok_count

    failure_classes = Counter(classify_failure(o.error) for o in outcomes if not o.ok)

    # Aggregate order/heroId analyses across the whole sample.
    all_order_analyses = [order_analysis(m.get("pickBans") or []) for m in matches]
    all_hero_rows = [
        row
        for m in matches
        for row in hero_id_vs_banned_hero_id(m.get("pickBans") or [])
    ]
    ban_rows = [row for row in all_hero_rows if row["agree"] is not None]
    disagreements = [row for row in ban_rows if row["agree"] is False]

    duration_present = sum(1 for m in matches if m.get("durationSeconds") is not None)
    winner_present = sum(1 for m in matches if m.get("didRadiantWin") is not None)
    series_id_present = sum(1 for m in matches if m.get("seriesId") is not None)
    series_object_present = sum(1 for m in matches if m.get("series") is not None)
    game_version_present = sum(1 for m in matches if m.get("gameVersionId") is not None)
    tournament_id_present = sum(1 for m in matches if m.get("tournamentId") is not None)

    was_banned_successfully_values = Counter(
        row.get("wasBannedSuccessfully")
        for m in matches
        for row in (m.get("pickBans") or [])
        if row.get("isPick") is False
    )

    pick_count_distribution: Counter[tuple[int, int]] = Counter()
    for m in matches:
        pick_bans = m.get("pickBans") or []
        radiant_picks = sum(
            1 for r in pick_bans if r.get("isPick") and r.get("isRadiant") is True
        )
        dire_picks = sum(
            1 for r in pick_bans if r.get("isPick") and r.get("isRadiant") is False
        )
        pick_count_distribution[(radiant_picks, dire_picks)] += 1

    preserved_anomalies = detect_and_preserve_anomalies(root, matches, mapper_errors)

    summary = {
        "sample_size": len(matches),
        "leagues": [
            {
                "league_id": spec.league_id,
                "label": spec.label,
                "tier_bucket": spec.tier_bucket,
                "stratz_tier": spec.stratz_tier,
                "era_note": spec.era_note,
            }
            for spec in TARGET_LEAGUES
        ],
        "game_versions_lookup_sample": game_versions[:10],
        "game_versions_lookup_count": len(game_versions),
        "field_population": {
            "durationSeconds": f"{duration_present}/{len(matches)}",
            "didRadiantWin": f"{winner_present}/{len(matches)}",
            "seriesId": f"{series_id_present}/{len(matches)}",
            "series_object": f"{series_object_present}/{len(matches)}",
            "gameVersionId": f"{game_version_present}/{len(matches)}",
            "tournamentId": f"{tournament_id_present}/{len(matches)}",
        },
        "order_analysis_aggregate": {
            "min_of_mins": min(
                (a["min"] for a in all_order_analyses if a["min"] is not None),
                default=None,
            ),
            "max_of_maxes": max(
                (a["max"] for a in all_order_analyses if a["max"] is not None),
                default=None,
            ),
            "matches_zero_based": sum(
                1 for a in all_order_analyses if a.get("zero_based")
            ),
            "matches_gap_free": sum(1 for a in all_order_analyses if a.get("gap_free")),
            "matches_with_duplicates": sum(
                1 for a in all_order_analyses if a.get("has_duplicates")
            ),
            "matches_with_orders": sum(
                1 for a in all_order_analyses if a.get("min") is not None
            ),
        },
        "hero_id_vs_banned_hero_id": {
            "total_pick_ban_rows": len(all_hero_rows),
            "ban_rows_checked": len(ban_rows),
            "ban_rows_disagreeing": len(disagreements),
            "disagreement_examples": disagreements[:10],
            "heroId_null_count": sum(1 for r in all_hero_rows if r["heroId_null"]),
            "bannedHeroId_null_count": sum(
                1 for r in all_hero_rows if r["bannedHeroId_null"]
            ),
        },
        "was_banned_successfully_distribution": {
            str(k): v for k, v in was_banned_successfully_values.items()
        },
        "pick_count_distribution": {
            f"radiant={k[0]},dire={k[1]}": v for k, v in pick_count_distribution.items()
        },
        "mapper_results": {
            "attempted": len(outcomes),
            "succeeded": ok_count,
            "failed": fail_count,
            "failure_classes": dict(failure_classes),
        },
        "playback_data_sample": playback_samples,
        "preserved_anomaly_fixtures": preserved_anomalies,
    }

    raw_matches_path = raw_dir / "stratz_verification_sample.json"
    raw_matches_path.write_text(
        json.dumps(matches, indent=2, default=str) + "\n", encoding="utf-8"
    )

    summary_path = raw_dir / "stratz_verification_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nRaw sample written to: {raw_matches_path}")
    print(f"Summary written to: {summary_path}")
    print(f"Preserved anomaly fixtures: {len(preserved_anomalies)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
