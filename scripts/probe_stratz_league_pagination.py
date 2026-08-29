"""Disposable STRATZ league-match pagination probe.

Investigates live pagination/ordering semantics for
`league(id).matches(request: LeagueMatchesRequestType)` without touching
production ingestion code. Writes raw evidence to data/raw/.

Never print or persist the API token or Authorization headers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

GRAPHQL_ENDPOINT = "https://api.stratz.com/graphql"
USER_AGENT = "dota-predictor-stratz-pagination-probe/0.1"
TIMEOUT_SECONDS = 45.0
REQUEST_DELAY_SECONDS = 0.35

# Curated probe targets: one compact T1, one large DPC round-robin, one
# historical completed event. IDs come from config/leagues.yaml and prior
# verification probes.
PROBE_LEAGUES: tuple[tuple[int, str], ...] = (
    (16935, "The International 2024"),
    (14892, "DPC 2023 WEU Winter Tour Division I"),
    (10749, "The International 2019"),
)

MINIMAL_MATCH_FIELDS = """
id
startDateTime
endDateTime
didRadiantWin
seriesId
leagueId
"""

STABILITY_FIELDS = """
id
startDateTime
endDateTime
didRadiantWin
durationSeconds
gameVersionId
seriesId
leagueId
tournamentId
tournamentRound
radiantTeamId
direTeamId
pickBans { isPick heroId order bannedHeroId isRadiant wasBannedSuccessfully }
players { steamAccountId heroId isRadiant playerSlot }
"""


@dataclass(frozen=True)
class PageResult:
    league_id: int
    request: dict[str, Any]
    http_status: int
    errors: list[Any] | None
    match_ids: list[int]
    matches: list[dict[str, Any]]
    response_headers: dict[str, str]
    elapsed_ms: float


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
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def require_token() -> str:
    token = os.environ.get("STRATZ_API_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "STRATZ_API_TOKEN is missing. Set it in the environment or project .env."
        )
    return token


def graphql(
    client: httpx.Client,
    query: str,
    variables: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str], float]:
    started = time.perf_counter()
    response = client.post(
        GRAPHQL_ENDPOINT,
        json={"query": query, "variables": variables or {}},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    try:
        payload: dict[str, Any] = response.json()
    except json.JSONDecodeError:
        payload = {
            "parse_error": "Response was not JSON",
            "text_preview": response.text[:500],
        }
    headers = {
        k.lower(): v
        for k, v in response.headers.items()
        if k.lower()
        in {
            "retry-after",
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
            "ratelimit-limit",
            "ratelimit-remaining",
            "ratelimit-reset",
            "cf-ray",
            "server",
            "date",
        }
    }
    return (
        {
            "http_status": response.status_code,
            "http_ok": response.is_success,
            "payload": payload,
        },
        headers,
        elapsed_ms,
    )


def unwrap_named_type(type_node: dict[str, Any] | None) -> dict[str, Any] | None:
    current = type_node
    while current and current.get("kind") in {"NON_NULL", "LIST"}:
        current = current.get("ofType")
    return current


def type_name_of(type_node: dict[str, Any] | None) -> str | None:
    named = unwrap_named_type(type_node)
    return (named or {}).get("name")


def introspect_type(client: httpx.Client, name: str) -> dict[str, Any] | None:
    result, _, _ = graphql(
        client,
        """
        query IntrospectType($name: String!) {
          __type(name: $name) {
            name
            kind
            fields {
              name
              args {
                name
                defaultValue
                type {
                  kind
                  name
                  ofType { kind name ofType { kind name ofType { kind name } } }
                }
              }
              type {
                kind
                name
                ofType { kind name ofType { kind name ofType { kind name } } }
              }
            }
            inputFields {
              name
              defaultValue
              type {
                kind
                name
                ofType { kind name ofType { kind name ofType { kind name } } }
              }
            }
            enumValues { name }
          }
        }
        """,
        {"name": name},
    )
    payload = result["payload"]
    if not result["http_ok"] or payload.get("errors"):
        return {
            "_introspection_error": {
                "http_status": result["http_status"],
                "errors": payload.get("errors"),
            }
        }
    return (payload.get("data") or {}).get("__type")


def fetch_league_matches(
    client: httpx.Client,
    *,
    league_id: int,
    request: dict[str, Any],
    selection: str,
) -> PageResult:
    query = f"""
    query LeagueMatchesProbe($id: Int!, $request: LeagueMatchesRequestType!) {{
      league(id: $id) {{
        id
        name
        displayName
        tier
        startDateTime
        endDateTime
        lastMatchDate
        matches(request: $request) {{
          {selection}
        }}
      }}
    }}
    """
    result, headers, elapsed_ms = graphql(
        client, query, {"id": league_id, "request": request}
    )
    payload = result["payload"]
    league = (payload.get("data") or {}).get("league")
    matches = (league or {}).get("matches") or []
    match_ids = [int(m["id"]) for m in matches if m.get("id") is not None]
    return PageResult(
        league_id=league_id,
        request=request,
        http_status=result["http_status"],
        errors=payload.get("errors"),
        match_ids=match_ids,
        matches=matches,
        response_headers=headers,
        elapsed_ms=elapsed_ms,
    )


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def payload_hash(matches: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(matches).encode()).hexdigest()


def analyze_ordering(matches: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [m.get("id") for m in matches]
    starts = [m.get("startDateTime") for m in matches]
    ends = [m.get("endDateTime") for m in matches]
    return {
        "count": len(matches),
        "ids": ids,
        "startDateTimes": starts,
        "endDateTimes": ends,
        "ids_monotonic_increasing": ids == sorted(ids),
        "ids_monotonic_decreasing": ids == sorted(ids, reverse=True),
        "start_monotonic_increasing": starts == sorted(starts),
        "start_monotonic_decreasing": starts == sorted(starts, reverse=True),
        "end_monotonic_increasing": ends == sorted(ends),
        "end_monotonic_decreasing": ends == sorted(ends, reverse=True),
        "id_duplicates": [k for k, v in Counter(ids).items() if v > 1],
    }


def page_overlap(page_a: list[int], page_b: list[int]) -> dict[str, Any]:
    set_a, set_b = set(page_a), set(page_b)
    return {
        "overlap_count": len(set_a & set_b),
        "only_in_first": sorted(set_a - set_b),
        "only_in_second": sorted(set_b - set_a),
    }


def walk_pages(
    client: httpx.Client,
    *,
    league_id: int,
    take: int,
    max_pages: int,
) -> dict[str, Any]:
    all_ids: list[int] = []
    pages: list[dict[str, Any]] = []
    for page_idx in range(max_pages):
        skip = page_idx * take
        page = fetch_league_matches(
            client,
            league_id=league_id,
            request={"take": take, "skip": skip},
            selection=MINIMAL_MATCH_FIELDS,
        )
        time.sleep(REQUEST_DELAY_SECONDS)
        pages.append(
            {
                "page_index": page_idx,
                "skip": skip,
                "take": take,
                "returned": len(page.match_ids),
                "match_ids": page.match_ids,
                "http_status": page.http_status,
                "errors": page.errors,
                "response_headers": page.response_headers,
                "elapsed_ms": round(page.elapsed_ms, 1),
            }
        )
        if page.errors:
            break
        if not page.match_ids:
            break
        all_ids.extend(page.match_ids)
        if len(page.match_ids) < take:
            break

    id_counter = Counter(all_ids)
    return {
        "take": take,
        "pages_fetched": len(pages),
        "total_ids_seen": len(all_ids),
        "unique_ids": len(id_counter),
        "duplicate_ids": {str(k): v for k, v in id_counter.items() if v > 1},
        "pages": pages,
    }


def test_adjacent_pages(
    client: httpx.Client,
    *,
    league_id: int,
    take: int,
    skip_pairs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for skip_a, skip_b in skip_pairs:
        page_a = fetch_league_matches(
            client,
            league_id=league_id,
            request={"take": take, "skip": skip_a},
            selection=MINIMAL_MATCH_FIELDS,
        )
        time.sleep(REQUEST_DELAY_SECONDS)
        page_b = fetch_league_matches(
            client,
            league_id=league_id,
            request={"take": take, "skip": skip_b},
            selection=MINIMAL_MATCH_FIELDS,
        )
        time.sleep(REQUEST_DELAY_SECONDS)
        overlap = page_overlap(page_a.match_ids, page_b.match_ids)
        results.append(
            {
                "skip_a": skip_a,
                "skip_b": skip_b,
                "take": take,
                "count_a": len(page_a.match_ids),
                "count_b": len(page_b.match_ids),
                "ids_a_tail": page_a.match_ids[-3:],
                "ids_b_head": page_b.match_ids[:3],
                "overlap": overlap,
            }
        )
    return results


def test_take_sizes(
    client: httpx.Client,
    *,
    league_id: int,
    take_values: list[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for take in take_values:
        page = fetch_league_matches(
            client,
            league_id=league_id,
            request={"take": take, "skip": 0},
            selection=MINIMAL_MATCH_FIELDS,
        )
        time.sleep(REQUEST_DELAY_SECONDS)
        results.append(
            {
                "take": take,
                "returned": len(page.match_ids),
                "http_status": page.http_status,
                "errors": page.errors,
                "first_id": page.match_ids[0] if page.match_ids else None,
                "last_id": page.match_ids[-1] if page.match_ids else None,
                "elapsed_ms": round(page.elapsed_ms, 1),
                "response_headers": page.response_headers,
            }
        )
    return results


def test_beyond_end(
    client: httpx.Client,
    *,
    league_id: int,
    take: int,
    skip_values: list[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for skip in skip_values:
        page = fetch_league_matches(
            client,
            league_id=league_id,
            request={"take": take, "skip": skip},
            selection=MINIMAL_MATCH_FIELDS,
        )
        time.sleep(REQUEST_DELAY_SECONDS)
        results.append(
            {
                "skip": skip,
                "take": take,
                "returned": len(page.match_ids),
                "http_status": page.http_status,
                "errors": page.errors,
                "match_ids_preview": page.match_ids[:5],
            }
        )
    return results


def test_stability(
    client: httpx.Client,
    *,
    league_id: int,
    request: dict[str, Any],
    repeats: int = 3,
) -> dict[str, Any]:
    hashes: list[str] = []
    pages: list[dict[str, Any]] = []
    for i in range(repeats):
        page = fetch_league_matches(
            client,
            league_id=league_id,
            request=request,
            selection=STABILITY_FIELDS,
        )
        time.sleep(REQUEST_DELAY_SECONDS)
        digest = payload_hash(page.matches)
        hashes.append(digest)
        pages.append(
            {
                "repeat": i,
                "returned": len(page.matches),
                "payload_sha256": digest,
                "http_status": page.http_status,
                "errors": page.errors,
            }
        )
    return {
        "request": request,
        "repeats": repeats,
        "all_identical": len(set(hashes)) == 1,
        "unique_hashes": list(dict.fromkeys(hashes)),
        "runs": pages,
    }


def test_invalid_args(client: httpx.Client, *, league_id: int) -> list[dict[str, Any]]:
    probes: list[tuple[str, dict[str, Any]]] = [
        ("negative_skip", {"take": 5, "skip": -1}),
        ("negative_take", {"take": -1, "skip": 0}),
        ("zero_take", {"take": 0, "skip": 0}),
        ("very_large_take", {"take": 10_000, "skip": 0}),
        ("unknown_field", {"take": 5, "skip": 0, "orderBy": "ID"}),
        ("future_start_filter", {"take": 5, "skip": 0, "startDateTime": 4102444800}),
    ]
    results: list[dict[str, Any]] = []
    for label, request in probes:
        page = fetch_league_matches(
            client,
            league_id=league_id,
            request=request,
            selection=MINIMAL_MATCH_FIELDS,
        )
        time.sleep(REQUEST_DELAY_SECONDS)
        results.append(
            {
                "label": label,
                "request": request,
                "http_status": page.http_status,
                "errors": page.errors,
                "returned": len(page.match_ids),
            }
        )
    return results


def test_time_subdivision(
    client: httpx.Client,
    *,
    league_id: int,
    take: int,
) -> dict[str, Any]:
    """Check whether startDateTime/endDateTime filters partition results."""
    baseline = fetch_league_matches(
        client,
        league_id=league_id,
        request={"take": take, "skip": 0},
        selection=MINIMAL_MATCH_FIELDS,
    )
    time.sleep(REQUEST_DELAY_SECONDS)
    if not baseline.matches:
        return {"note": "no baseline matches"}

    starts = [m["startDateTime"] for m in baseline.matches if m.get("startDateTime")]
    if len(starts) < 4:
        return {"note": "insufficient matches for time split test"}

    mid = sorted(starts)[len(starts) // 2]
    early = fetch_league_matches(
        client,
        league_id=league_id,
        request={"take": take, "skip": 0, "endDateTime": mid},
        selection=MINIMAL_MATCH_FIELDS,
    )
    time.sleep(REQUEST_DELAY_SECONDS)
    late = fetch_league_matches(
        client,
        league_id=league_id,
        request={"take": take, "skip": 0, "startDateTime": mid},
        selection=MINIMAL_MATCH_FIELDS,
    )
    time.sleep(REQUEST_DELAY_SECONDS)
    return {
        "baseline_count": len(baseline.match_ids),
        "split_epoch": mid,
        "early_filter_endDateTime": {
            "count": len(early.match_ids),
            "ids": early.match_ids,
            "max_start": max(
                (m.get("startDateTime") for m in early.matches), default=None
            ),
        },
        "late_filter_startDateTime": {
            "count": len(late.match_ids),
            "ids": late.match_ids,
            "min_start": min(
                (m.get("startDateTime") for m in late.matches), default=None
            ),
        },
        "overlap_early_late": page_overlap(early.match_ids, late.match_ids),
        "overlap_baseline_early": page_overlap(baseline.match_ids, early.match_ids),
    }


def test_series_filter(
    client: httpx.Client,
    *,
    league_id: int,
) -> dict[str, Any]:
    sample = fetch_league_matches(
        client,
        league_id=league_id,
        request={"take": 20, "skip": 0},
        selection=MINIMAL_MATCH_FIELDS,
    )
    time.sleep(REQUEST_DELAY_SECONDS)
    series_ids = sorted({m["seriesId"] for m in sample.matches if m.get("seriesId")})
    if not series_ids:
        return {"note": "no seriesId on sample page"}
    target = series_ids[0]
    filtered = fetch_league_matches(
        client,
        league_id=league_id,
        request={"take": 50, "skip": 0, "seriesId": target},
        selection=MINIMAL_MATCH_FIELDS,
    )
    time.sleep(REQUEST_DELAY_SECONDS)
    wrong_series = [
        m["id"]
        for m in filtered.matches
        if m.get("seriesId") not in (None, target)
    ]
    return {
        "sample_series_ids": series_ids[:10],
        "filter_series_id": target,
        "filtered_count": len(filtered.match_ids),
        "filtered_ids": filtered.match_ids,
        "all_match_filter": len(wrong_series) == 0,
        "wrong_series_matches": wrong_series,
    }


def summarize_schema(
    league_type: dict[str, Any] | None,
    matches_request_type: dict[str, Any] | None,
) -> dict[str, Any]:
    matches_field = None
    for field in (league_type or {}).get("fields") or []:
        if field.get("name") == "matches":
            matches_field = {
                "name": field["name"],
                "return_type": type_name_of(field.get("type")),
                "args": [
                    {
                        "name": arg["name"],
                        "type": type_name_of(arg.get("type")),
                        "defaultValue": arg.get("defaultValue"),
                    }
                    for arg in field.get("args") or []
                ],
            }
            break

    input_fields = []
    for field in (matches_request_type or {}).get("inputFields") or []:
        input_fields.append(
            {
                "name": field["name"],
                "type": type_name_of(field.get("type")),
                "defaultValue": field.get("defaultValue"),
            }
        )

    return {
        "query_signature": (
            "league(id: Int!): LeagueType { matches(request: LeagueMatchesRequestType!): [MatchType] }"
        ),
        "league_matches_field": matches_field,
        "LeagueMatchesRequestType": {
            "kind": (matches_request_type or {}).get("kind"),
            "input_fields": input_fields,
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def redact_token_check(paths: list[Path], token: str) -> None:
    pattern = re.compile(re.escape(token))
    for path in paths:
        if path.is_file() and pattern.search(path.read_text(encoding="utf-8")):
            path.write_text("{}\n", encoding="utf-8")
            print(
                f"warning: token-like value found in {path.name}; file wiped.",
                file=sys.stderr,
            )


def main() -> int:
    root = project_root()
    load_project_env(root)
    token = require_token()
    raw_dir = root / "data" / "raw"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    evidence: dict[str, Any] = {
        "probe": "stratz_league_pagination",
        "timestamp_utc": timestamp,
        "endpoint": GRAPHQL_ENDPOINT,
        "probe_leagues": [{"id": lid, "label": label} for lid, label in PROBE_LEAGUES],
    }

    output_paths: list[Path] = []

    with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS) as client:
        league_type = introspect_type(client, "LeagueType")
        matches_request_type = introspect_type(client, "LeagueMatchesRequestType")
        time.sleep(REQUEST_DELAY_SECONDS)

        evidence["schema"] = summarize_schema(league_type, matches_request_type)

        # Baseline ordering on first page for each league.
        ordering_by_league: dict[str, Any] = {}
        for league_id, label in PROBE_LEAGUES:
            page = fetch_league_matches(
                client,
                league_id=league_id,
                request={"take": 50, "skip": 0},
                selection=MINIMAL_MATCH_FIELDS,
            )
            time.sleep(REQUEST_DELAY_SECONDS)
            ordering_by_league[str(league_id)] = {
                "label": label,
                "league_meta": {
                    "http_status": page.http_status,
                    "errors": page.errors,
                },
                "first_page_ordering": analyze_ordering(page.matches),
                "response_headers": page.response_headers,
            }
        evidence["ordering_first_page"] = ordering_by_league

        # Page-size behavior on TI 2024.
        evidence["take_size_behavior"] = {
            "league_id": 16935,
            "results": test_take_sizes(
                client,
                league_id=16935,
                take_values=[1, 5, 10, 25, 50, 100, 200, 500, 1000],
            ),
        }

        # Adjacent-page overlap/gap checks on large DPC league.
        evidence["adjacent_page_checks"] = {
            "league_id": 14892,
            "take": 20,
            "pairs": test_adjacent_pages(
                client,
                league_id=14892,
                take=20,
                skip_pairs=[(0, 20), (20, 40), (40, 60), (19, 20), (20, 21)],
            ),
        }

        # Full walk (bounded) to estimate total size and duplicate rate.
        evidence["full_walk"] = {
            str(league_id): walk_pages(client, league_id=league_id, take=50, max_pages=30)
            for league_id, _ in PROBE_LEAGUES
        }

        # Beyond-end behavior using walk-derived sizes.
        beyond_end: dict[str, Any] = {}
        for league_id, label in PROBE_LEAGUES:
            walk = evidence["full_walk"][str(league_id)]
            total_unique = walk["unique_ids"]
            beyond_end[str(league_id)] = {
                "label": label,
                "estimated_unique": total_unique,
                "tests": test_beyond_end(
                    client,
                    league_id=league_id,
                    take=20,
                    skip_values=[
                        max(0, total_unique - 5),
                        total_unique,
                        total_unique + 5,
                        total_unique + 100,
                        100_000,
                    ],
                ),
            }
        evidence["beyond_end_behavior"] = beyond_end

        # Payload stability (same request repeated).
        evidence["payload_stability"] = {
            str(league_id): test_stability(
                client,
                league_id=league_id,
                request={"take": 10, "skip": 0},
                repeats=3,
            )
            for league_id, _ in PROBE_LEAGUES
        }

        # Invalid / edge argument behavior.
        evidence["invalid_argument_probes"] = test_invalid_args(
            client, league_id=16935
        )

        # Subdivision filters.
        evidence["time_subdivision_probe"] = test_time_subdivision(
            client, league_id=14892, take=50
        )
        evidence["series_subdivision_probe"] = {
            str(league_id): test_series_filter(client, league_id=league_id)
            for league_id, _ in PROBE_LEAGUES
        }

        # Light rate-limit header capture across a short burst (5 quick calls).
        burst: list[dict[str, Any]] = []
        for i in range(5):
            page = fetch_league_matches(
                client,
                league_id=16935,
                request={"take": 1, "skip": i},
                selection="id",
            )
            burst.append(
                {
                    "call": i,
                    "http_status": page.http_status,
                    "errors": page.errors,
                    "response_headers": page.response_headers,
                    "elapsed_ms": round(page.elapsed_ms, 1),
                }
            )
            time.sleep(0.05)
        evidence["rate_limit_burst_probe"] = burst

    out_path = raw_dir / "stratz_league_pagination_probe.json"
    write_json(out_path, evidence)
    output_paths.append(out_path)

    # Compact human-oriented summary alongside raw evidence.
    summary = {
        "timestamp_utc": timestamp,
        "schema": evidence["schema"],
        "ordering_first_page": {
            k: {
                "label": v["label"],
                "ordering": v["first_page_ordering"],
            }
            for k, v in evidence["ordering_first_page"].items()
        },
        "take_size_behavior": evidence["take_size_behavior"],
        "adjacent_page_checks": evidence["adjacent_page_checks"],
        "full_walk_totals": {
            k: {
                "unique_ids": v["unique_ids"],
                "duplicate_ids": v["duplicate_ids"],
                "pages_fetched": v["pages_fetched"],
            }
            for k, v in evidence["full_walk"].items()
        },
        "beyond_end_behavior": evidence["beyond_end_behavior"],
        "payload_stability": evidence["payload_stability"],
        "invalid_argument_probes": evidence["invalid_argument_probes"],
        "time_subdivision_probe": evidence["time_subdivision_probe"],
        "series_subdivision_probe": evidence["series_subdivision_probe"],
        "rate_limit_burst_probe": evidence["rate_limit_burst_probe"],
        "recommended_cursor_state": {
            "note": (
                "Derived from observed offset pagination; see full probe JSON for evidence."
            ),
            "suggested_fields": {
                "skip": "next offset to request (equals matches_seen_count if no gaps/duplicates)",
                "take": "last page size used",
                "last_match_id": "id of last match in previous completed page (sanity check)",
                "last_start_date_time": "startDateTime of last match (ordering verification)",
                "completed": "true when a terminal empty page or short page was observed",
            },
        },
    }
    summary_path = raw_dir / "stratz_league_pagination_summary.json"
    write_json(summary_path, summary)
    output_paths.append(summary_path)

    redact_token_check(output_paths, token)

    print("STRATZ league pagination probe complete")
    print(f"  raw evidence: {out_path}")
    print(f"  summary:      {summary_path}")
    for league_id, label in PROBE_LEAGUES:
        walk = evidence["full_walk"][str(league_id)]
        ordering = evidence["ordering_first_page"][str(league_id)]["first_page_ordering"]
        print(
            f"  league {league_id} ({label}): unique={walk['unique_ids']} "
            f"pages={walk['pages_fetched']} "
            f"start_desc={ordering['start_monotonic_decreasing']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
