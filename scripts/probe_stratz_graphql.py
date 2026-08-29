"""Disposable STRATZ GraphQL exploration probe.

This is not a production client. It loads STRATZ_API_TOKEN from the
environment or project-root .env, introspects a few relevant schema types,
fetches a small sample of recent professional matches, and writes a raw
response under data/raw/ for inspection.

Never print or persist the API token or Authorization headers.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

GRAPHQL_ENDPOINT = "https://api.stratz.com/graphql"
USER_AGENT = "dota-predictor-stratz-probe/0.1"
TIMEOUT_SECONDS = 30.0
MATCH_SAMPLE_SIZE = 5

WANTED_MATCH_FIELDS = {
    "id",
    "startDateTime",
    "didRadiantWin",
    "leagueId",
    "league",
    "tournamentId",
    "tournamentRound",
    "seriesId",
    "series",
    "radiantTeamId",
    "direTeamId",
    "radiantTeam",
    "direTeam",
    "players",
    "pickBans",
}

WANTED_LEAGUE_FIELDS = {
    "id",
    "name",
    "displayName",
    "tier",
    "region",
    "startDateTime",
    "endDateTime",
    "lastMatchDate",
    "prizePool",
}

WANTED_TEAM_FIELDS = {"id", "name", "tag"}

WANTED_SERIES_FIELDS = {
    "id",
    "type",
    "teamOneId",
    "teamTwoId",
    "teamOneWinCount",
    "teamTwoWinCount",
    "winningTeamId",
    "matches",
}

WANTED_PLAYER_FIELDS = {
    "steamAccountId",
    "heroId",
    "isRadiant",
    "playerSlot",
    "steamAccount",
}

WANTED_STEAM_ACCOUNT_FIELDS = {"id", "name", "proSteamAccount"}

WANTED_PRO_ACCOUNT_FIELDS = {"name", "teamId", "team"}

WANTED_PICKBAN_FIELDS = {
    "heroId",
    "bannedHeroId",
    "isPick",
    "order",
    "isRadiant",
    "team",
    "playerIndex",
    "wasBannedSuccessfully",
    "isCaptain",
    "letter",
}

HIGH_TIER_PREFERENCE = (
    "INTERNATIONAL",
    "MAJOR",
    "DPC_LEAGUE_FINALS",
    "DPC_LEAGUE",
    "MINOR",
    "PROFESSIONAL",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_project_env(root: Path) -> None:
    """Load KEY=VALUE pairs from .env without overriding existing env vars."""
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
) -> dict[str, Any]:
    response = client.post(
        GRAPHQL_ENDPOINT,
        json={"query": query, "variables": variables or {}},
    )
    payload: dict[str, Any]
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = {
            "parse_error": "Response was not JSON",
            "text_preview": response.text[:500],
        }
    return {
        "http_status": response.status_code,
        "http_ok": response.is_success,
        "payload": payload,
    }


def unwrap_named_type(type_node: dict[str, Any] | None) -> dict[str, Any] | None:
    current = type_node
    while current and current.get("kind") in {"NON_NULL", "LIST"}:
        current = current.get("ofType")
    return current


def field_map(type_info: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not type_info:
        return {}
    return {field["name"]: field for field in type_info.get("fields") or []}


def input_field_map(type_info: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not type_info:
        return {}
    return {field["name"]: field for field in type_info.get("inputFields") or []}


def enum_values(type_info: dict[str, Any] | None) -> list[str]:
    if not type_info:
        return []
    return [item["name"] for item in type_info.get("enumValues") or []]


def introspect_type(client: httpx.Client, name: str) -> dict[str, Any] | None:
    result = graphql(
        client,
        """
        query IntrospectType($name: String!) {
          __type(name: $name) {
            name
            kind
            fields {
              name
              args { name }
              type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
            }
            inputFields {
              name
              type { kind name ofType { kind name ofType { kind name } } }
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


def select_fields(available: dict[str, dict[str, Any]], wanted: set[str]) -> list[str]:
    return [name for name in wanted if name in available]


def object_selection(
    types: dict[str, dict[str, Any] | None],
    type_name: str | None,
    wanted: set[str],
    nested: dict[str, Any] | None = None,
) -> str:
    available = field_map(types.get(type_name or ""))
    lines: list[str] = []
    for name in select_fields(available, wanted):
        named = unwrap_named_type(available[name]["type"])
        kind = (named or {}).get("kind")
        nested_type = (named or {}).get("name")
        if kind in {"OBJECT", "INTERFACE"} and nested and name in nested:
            inner = nested[name](types, nested_type)
            if inner.strip():
                lines.append(f"{name} {{ {inner} }}")
        elif kind in {"OBJECT", "INTERFACE"}:
            continue
        else:
            lines.append(name)
    return " ".join(lines)


def match_selection(types: dict[str, dict[str, Any] | None]) -> str:
    def league_sel(t: dict[str, dict[str, Any] | None], type_name: str | None) -> str:
        return object_selection(t, type_name, WANTED_LEAGUE_FIELDS)

    def team_sel(t: dict[str, dict[str, Any] | None], type_name: str | None) -> str:
        return object_selection(t, type_name, WANTED_TEAM_FIELDS)

    def series_sel(t: dict[str, dict[str, Any] | None], type_name: str | None) -> str:
        # Avoid recursing into series.matches; we only need series metadata.
        wanted = WANTED_SERIES_FIELDS - {"matches"}
        return object_selection(t, type_name, wanted)

    def steam_sel(t: dict[str, dict[str, Any] | None], type_name: str | None) -> str:
        def pro_sel(tt: dict[str, dict[str, Any] | None], n: str | None) -> str:
            return object_selection(
                tt, n, WANTED_PRO_ACCOUNT_FIELDS, {"team": team_sel}
            )

        return object_selection(
            t,
            type_name,
            WANTED_STEAM_ACCOUNT_FIELDS,
            {"proSteamAccount": pro_sel},
        )

    def player_sel(t: dict[str, dict[str, Any] | None], type_name: str | None) -> str:
        return object_selection(
            t,
            type_name,
            WANTED_PLAYER_FIELDS,
            {"steamAccount": steam_sel},
        )

    def pickban_sel(t: dict[str, dict[str, Any] | None], type_name: str | None) -> str:
        return object_selection(t, type_name, WANTED_PICKBAN_FIELDS)

    return object_selection(
        types,
        "MatchType",
        WANTED_MATCH_FIELDS,
        {
            "league": league_sel,
            "series": series_sel,
            "radiantTeam": team_sel,
            "direTeam": team_sel,
            "players": player_sel,
            "pickBans": pickban_sel,
        },
    )


def iso_from_unix(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(int(value), tz=UTC).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def pick_league(leagues: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not leagues:
        return None
    tier_rank = {name: index for index, name in enumerate(HIGH_TIER_PREFERENCE)}

    def sort_key(league: dict[str, Any]) -> tuple[int, int, int]:
        name = f"{league.get('displayName') or ''} {league.get('name') or ''}".lower()
        name_boost = 1 if "international" in name else 0
        prestige = len(HIGH_TIER_PREFERENCE) - tier_rank.get(
            str(league.get("tier")), len(HIGH_TIER_PREFERENCE)
        )
        started = int(league.get("lastMatchDate") or league.get("startDateTime") or 0)
        return name_boost, prestige, started

    return max(leagues, key=sort_key)


def summarize_match(match: dict[str, Any]) -> dict[str, Any]:
    players = match.get("players") or []
    pick_bans = match.get("pickBans") or []
    orders = [row.get("order") for row in pick_bans if row.get("order") is not None]
    hero_ids = [
        row.get("heroId")
        if row.get("isPick")
        else row.get("heroId") or row.get("bannedHeroId")
        for row in pick_bans
    ]
    missing_heroes = sum(1 for hero_id in hero_ids if not hero_id)
    picks = [row for row in pick_bans if row.get("isPick")]
    bans = [row for row in pick_bans if row.get("isPick") is False]
    return {
        "match_id": match.get("id"),
        "start_utc": iso_from_unix(match.get("startDateTime")),
        "league": (match.get("league") or {}).get("displayName")
        or (match.get("league") or {}).get("name"),
        "league_id": match.get("leagueId") or (match.get("league") or {}).get("id"),
        "series_id": match.get("seriesId") or (match.get("series") or {}).get("id"),
        "series": match.get("series"),
        "radiant": match.get("radiantTeam") or {"id": match.get("radiantTeamId")},
        "dire": match.get("direTeam") or {"id": match.get("direTeamId")},
        "did_radiant_win": match.get("didRadiantWin"),
        "player_count": len(players),
        "players_with_account_id": sum(
            1 for player in players if player.get("steamAccountId")
        ),
        "pickban_count": len(pick_bans),
        "pick_count": len(picks),
        "ban_count": len(bans),
        "orders": orders,
        "order_is_contiguous": bool(orders)
        and orders == list(range(min(orders), max(orders) + 1)),
        "pickbans_missing_hero_id": missing_heroes,
        "radiant_player_count": sum(1 for player in players if player.get("isRadiant")),
        "dire_player_count": sum(
            1 for player in players if player.get("isRadiant") is False
        ),
    }


def print_summary(
    *,
    league: dict[str, Any] | None,
    matches: list[dict[str, Any]],
    graphql_errors: list[Any] | None,
    http_status: int,
    raw_path: Path,
) -> None:
    print("STRATZ GraphQL probe summary")
    print(f"  endpoint: {GRAPHQL_ENDPOINT}")
    print(f"  http_status: {http_status}")
    if graphql_errors:
        print(f"  graphql_errors: {len(graphql_errors)}")
        for error in graphql_errors[:8]:
            message = error.get("message") if isinstance(error, dict) else str(error)
            print(f"    - {message}")
    if league:
        print(
            "  league:"
            f" id={league.get('id')}"
            f" name={league.get('displayName') or league.get('name')}"
            f" tier={league.get('tier')}"
            f" start={iso_from_unix(league.get('startDateTime'))}"
        )
        print("  league_filter: see raw file league_discovery.request")
    print(f"  matches_returned: {len(matches)}")
    for summary in (summarize_match(match) for match in matches):
        winner = (
            "Radiant"
            if summary["did_radiant_win"] is True
            else "Dire"
            if summary["did_radiant_win"] is False
            else "unknown"
        )
        radiant = summary["radiant"] or {}
        dire = summary["dire"] or {}
        print(
            f"  match {summary['match_id']}: {radiant.get('name') or radiant.get('id')}"
            f" vs {dire.get('name') or dire.get('id')} | winner={winner}"
            f" | start={summary['start_utc']}"
            f" | series={summary['series_id']}"
            f" | players={summary['player_count']}"
            f" | pickBans={summary['pickban_count']}"
            f" (picks={summary['pick_count']}, bans={summary['ban_count']})"
            f" | order_contiguous={summary['order_is_contiguous']}"
        )
    print(f"  raw_response: {raw_path}")


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

    with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS) as client:
        query_type = graphql(
            client,
            "{ __schema { queryType { name } } }",
        )
        query_type_name = (
            ((query_type["payload"].get("data") or {}).get("__schema") or {}).get(
                "queryType"
            )
            or {}
        ).get("name") or "Query"

        type_names = [
            query_type_name,
            "MatchType",
            "LeagueType",
            "LeagueRequestType",
            "LeagueMatchesRequestType",
            "MatchPlayerType",
            "MatchPickBanType",
            "MatchDraftType",
            "PickBanType",
            "SeriesType",
            "TeamType",
            "SteamAccountType",
            "SteamAccountProType",
            "ProSteamAccountType",
            "LeagueTier",
            "FilterOrderBy",
        ]
        types = {name: introspect_type(client, name) for name in type_names}

        # Recover pick-ban / series / team type names from MatchType if aliases differ.
        match_fields = field_map(types.get("MatchType"))
        for field_name in (
            "pickBans",
            "series",
            "radiantTeam",
            "direTeam",
            "players",
            "league",
        ):
            named = unwrap_named_type((match_fields.get(field_name) or {}).get("type"))
            nested_name = (named or {}).get("name")
            if nested_name and nested_name not in types:
                types[nested_name] = introspect_type(client, nested_name)

        player_fields = field_map(types.get("MatchPlayerType"))
        steam_named = unwrap_named_type(
            (player_fields.get("steamAccount") or {}).get("type")
        )
        steam_name = (steam_named or {}).get("name")
        if steam_name and steam_name not in types:
            types[steam_name] = introspect_type(client, steam_name)

        introspection_path = raw_dir / "stratz_probe_introspection.json"
        write_json(
            introspection_path,
            {
                "endpoint": GRAPHQL_ENDPOINT,
                "query_type": query_type_name,
                "types": {
                    name: {
                        "kind": (info or {}).get("kind"),
                        "fields": sorted(field_map(info)),
                        "input_fields": sorted(input_field_map(info)),
                        "enum_values": enum_values(info),
                        "error": (info or {}).get("_introspection_error"),
                    }
                    for name, info in types.items()
                    if info is not None
                },
            },
        )

        league_input = input_field_map(types.get("LeagueRequestType"))
        tier_values = enum_values(types.get("LeagueTier"))
        order_by_values = enum_values(types.get("FilterOrderBy"))
        preferred_tiers = [tier for tier in HIGH_TIER_PREFERENCE if tier in tier_values]
        # Query the top two available high tiers first (typically INTERNATIONAL + MAJOR).
        initial_tiers = preferred_tiers[:2] or preferred_tiers

        def build_league_request(
            *,
            tiers: list[str] | None,
            has_live: bool | None = None,
            ended: bool | None = None,
            require_prize_pool: bool = True,
        ) -> dict[str, Any]:
            request: dict[str, Any] = {}
            if "take" in league_input:
                request["take"] = 20
            if "skip" in league_input:
                request["skip"] = 0
            if tiers and "tiers" in league_input:
                request["tiers"] = tiers
            if has_live is not None and "hasLiveMatches" in league_input:
                request["hasLiveMatches"] = has_live
            if ended is not None and "leagueEnded" in league_input:
                request["leagueEnded"] = ended
            if require_prize_pool and "requirePrizePool" in league_input:
                request["requirePrizePool"] = True
            if "orderBy" in league_input:
                for candidate in ("LAST_MATCH_TIME", "END_DATE", "START_DATE", "ID"):
                    if candidate in order_by_values:
                        request["orderBy"] = candidate
                        break
            return request

        league_sel = object_selection(types, "LeagueType", WANTED_LEAGUE_FIELDS)
        leagues_query = f"""
        query ProbeLeagues($request: LeagueRequestType!) {{
          leagues(request: $request) {{
            {league_sel}
          }}
        }}
        """

        league_attempts = [
            build_league_request(tiers=initial_tiers, has_live=True),
            build_league_request(tiers=initial_tiers, ended=False),
            build_league_request(tiers=initial_tiers, require_prize_pool=True),
            build_league_request(tiers=preferred_tiers, require_prize_pool=False),
        ]
        leagues: list[dict[str, Any]] = []
        league_request: dict[str, Any] = league_attempts[0]
        leagues_result: dict[str, Any] = {"http_status": 0, "payload": {}}
        for attempt in league_attempts:
            leagues_result = graphql(client, leagues_query, {"request": attempt})
            leagues = (
                (leagues_result["payload"].get("data") or {}).get("leagues")
            ) or []
            league_request = attempt
            if leagues:
                break

        selected = pick_league(leagues)
        match_sel = match_selection(types)
        matches_input = input_field_map(types.get("LeagueMatchesRequestType"))
        matches_request: dict[str, Any] = {}
        if "take" in matches_input:
            matches_request["take"] = MATCH_SAMPLE_SIZE
        if "skip" in matches_input:
            matches_request["skip"] = 0

        probe_payload: dict[str, Any]
        if selected and match_sel:
            matches_query = f"""
            query ProbeLeagueMatches($id: Int!, $request: LeagueMatchesRequestType!) {{
              league(id: $id) {{
                {league_sel}
                matches(request: $request) {{
                  {match_sel}
                }}
              }}
            }}
            """
            matches_result = graphql(
                client,
                matches_query,
                {"id": selected["id"], "request": matches_request},
            )
            probe_payload = {
                "endpoint": GRAPHQL_ENDPOINT,
                "league_discovery": {
                    "request": league_request,
                    "http_status": leagues_result["http_status"],
                    "errors": leagues_result["payload"].get("errors"),
                    "leagues": leagues,
                    "selected_league_id": selected.get("id"),
                },
                "match_query": {
                    "request": matches_request,
                    "http_status": matches_result["http_status"],
                    "errors": matches_result["payload"].get("errors"),
                    "data": matches_result["payload"].get("data"),
                },
            }
            league_data = (
                (matches_result["payload"].get("data") or {}).get("league")
            ) or selected
            matches = league_data.get("matches") or []
            print_summary(
                league=league_data,
                matches=matches,
                graphql_errors=matches_result["payload"].get("errors"),
                http_status=matches_result["http_status"],
                raw_path=raw_dir / "stratz_probe_matches.json",
            )
            if not matches:
                print(
                    "  note: league query returned no matches; see raw file and"
                    " league_discovery for details."
                )
        else:
            probe_payload = {
                "endpoint": GRAPHQL_ENDPOINT,
                "league_discovery": {
                    "request": league_request,
                    "http_status": leagues_result["http_status"],
                    "errors": leagues_result["payload"].get("errors"),
                    "leagues": leagues,
                },
                "match_query": None,
                "note": "Could not select a league or build a MatchType selection.",
            }
            print("STRATZ GraphQL probe summary")
            print(f"  endpoint: {GRAPHQL_ENDPOINT}")
            print("  no league/match sample could be fetched; see raw files.")
            if leagues_result["payload"].get("errors"):
                for error in leagues_result["payload"]["errors"][:8]:
                    print(f"    - {error.get('message')}")

        raw_path = raw_dir / "stratz_probe_matches.json"
        write_json(raw_path, probe_payload)
        print(f"  introspection: {introspection_path}")

        # Redact check: refuse to continue if the token leaked into saved files.
        token_pattern = re.compile(re.escape(token))
        for path in (introspection_path, raw_path):
            if token_pattern.search(path.read_text(encoding="utf-8")):
                path.write_text("{}\n", encoding="utf-8")
                print(
                    f"  warning: token-like value found in {path.name}; file wiped.",
                    file=sys.stderr,
                )
                return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
