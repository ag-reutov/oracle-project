"""Match-ID discovery for leagues whose STRATZ `league(id)` catalog is null.

Discovery sources produce match IDs only. Canonical payloads always come
from STRATZ `match(id)` (see `pipeline.ingest_matches_by_id`). This module
does not write to Postgres.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx

from dota_predictor.ingestion.client import TeamLeagueMatchIdsFetcher
from dota_predictor.ingestion.config import DEFAULT_PAGE_SIZE

__all__ = [
    "MatchIdDiscoveryResult",
    "dedupe_match_ids",
    "discover_league_match_ids",
    "discover_match_ids_from_opendota",
    "discover_match_ids_from_team_matches",
]

OPENDOTA_EXPLORER_URL = "https://api.opendota.com/api/explorer"
OPENDOTA_USER_AGENT = "dota-predictor-ingestion/0.1"


@dataclass(frozen=True)
class MatchIdDiscoveryResult:
    """Comparison of STRATZ team-walk IDs vs an independent OpenDota list."""

    league_id: int
    team_match_ids: frozenset[int]
    opendota_match_ids: frozenset[int]
    match_ids: tuple[int, ...]
    teams_visited: frozenset[int]
    notes: tuple[str, ...]


def dedupe_match_ids(match_ids: Sequence[int]) -> tuple[int, ...]:
    """Stable unique match IDs (first occurrence kept)."""
    seen: set[int] = set()
    ordered: list[int] = []
    for match_id in match_ids:
        value = int(match_id)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def discover_match_ids_from_team_matches(
    fetcher: TeamLeagueMatchIdsFetcher,
    league_id: int,
    seed_team_ids: Sequence[int],
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> tuple[frozenset[int], frozenset[int]]:
    """BFS participating teams via `team.matches(leagueId)` and collect ids.

    Returns `(match_ids, teams_visited)`. A single seed team is not assumed
    to cover the tournament; opponents on each page are enqueued until the
    team set closes.
    """
    pending: set[int] = {int(tid) for tid in seed_team_ids if tid}
    visited: set[int] = set()
    match_ids: set[int] = set()

    while pending:
        team_id = pending.pop()
        if team_id in visited:
            continue
        visited.add(team_id)
        skip = 0
        while True:
            page = fetcher.fetch_team_league_match_ids_page(
                team_id,
                league_id=league_id,
                skip=skip,
                take=page_size,
            )
            if not page:
                break
            for row in page:
                returned_league = row.get("leagueId")
                if returned_league is not None and int(returned_league) != league_id:
                    continue
                match_id = row.get("id")
                if match_id is None:
                    continue
                match_ids.add(int(match_id))
                for key in ("radiantTeamId", "direTeamId"):
                    other = row.get(key)
                    if other is not None:
                        other_id = int(other)
                        if other_id not in visited:
                            pending.add(other_id)
            if len(page) < page_size:
                break
            skip += page_size

    return frozenset(match_ids), frozenset(visited)


def discover_match_ids_from_opendota(
    league_id: int,
    *,
    client: httpx.Client | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> frozenset[int]:
    """Independent Valve-league match list from the OpenDota explorer.

    Used only as an ID source / completeness check, never as a payload source.
    When `window_start`/`window_end` are given, only matches whose
    `start_time` falls within the window are returned (main-event window
    filtering for leagues whose STRATZ league also contains qualifiers).
    """
    start_unix = int(window_start.timestamp()) if window_start is not None else None
    end_unix = int(window_end.timestamp()) if window_end is not None else None
    bounds = ""
    if start_unix is not None or end_unix is not None:
        clauses = []
        if start_unix is not None:
            clauses.append(f"start_time >= {start_unix}")
        if end_unix is not None:
            clauses.append(f"start_time <= {end_unix}")
        bounds = f" AND {' AND '.join(clauses)}"
    sql = (
        "SELECT match_id FROM matches "
        f"WHERE leagueid = {int(league_id)}{bounds} ORDER BY match_id"
    )
    url = f"{OPENDOTA_EXPLORER_URL}?sql={quote(sql)}"
    own_client = client is None
    http = client or httpx.Client(
        headers={"User-Agent": OPENDOTA_USER_AGENT},
        timeout=60.0,
    )
    try:
        response = http.get(url)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    finally:
        if own_client:
            http.close()

    if payload.get("err") or payload.get("error"):
        raise RuntimeError(
            f"OpenDota explorer error for league {league_id}: "
            f"{payload.get('err') or payload.get('error')}"
        )
    rows = payload.get("rows") or []
    return frozenset(int(row["match_id"]) for row in rows)


def discover_league_match_ids(
    league_id: int,
    *,
    team_fetcher: TeamLeagueMatchIdsFetcher | None = None,
    seed_team_ids: Sequence[int] = (),
    opendota_client: httpx.Client | None = None,
    skip_opendota: bool = False,
    skip_team_walk: bool = False,
    page_size: int = DEFAULT_PAGE_SIZE,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> MatchIdDiscoveryResult:
    """Discover match IDs via team walk and/or OpenDota; ingest the union.

    `window_start`/`window_end` restrict the OpenDota enumeration to the
    given main-event window (for leagues whose STRATZ league also contains
    qualifiers). Team-walk IDs are not time-filtered here; callers that
    use team-walk IDs with a window must also enforce the window on the
    fetched payload (see `pipeline.ingest_matches_by_id`).
    """
    notes: list[str] = []
    team_ids: frozenset[int] = frozenset()
    teams_visited: frozenset[int] = frozenset()
    opendota_ids: frozenset[int] = frozenset()

    if not skip_team_walk:
        if team_fetcher is None:
            raise ValueError("team_fetcher is required unless skip_team_walk is True")
        if not seed_team_ids:
            notes.append("team walk skipped: no seed team ids")
        else:
            team_ids, teams_visited = discover_match_ids_from_team_matches(
                team_fetcher,
                league_id,
                seed_team_ids,
                page_size=page_size,
            )
            notes.append(
                f"STRATZ team.matches walk: {len(team_ids)} ids from "
                f"{len(teams_visited)} teams (seeds={list(seed_team_ids)})"
            )

    if not skip_opendota:
        opendota_ids = discover_match_ids_from_opendota(
            league_id,
            client=opendota_client,
            window_start=window_start,
            window_end=window_end,
        )
        if window_start is not None and window_end is not None:
            notes.append(
                f"OpenDota explorer: {len(opendota_ids)} ids "
                f"(window {window_start.date()}..{window_end.date()})"
            )
        else:
            notes.append(f"OpenDota explorer: {len(opendota_ids)} ids")

    only_team = team_ids - opendota_ids
    only_opendota = opendota_ids - team_ids
    if team_ids and opendota_ids:
        if only_team:
            notes.append(
                f"in STRATZ team walk only ({len(only_team)}): {sorted(only_team)[:20]}"
            )
        if only_opendota:
            notes.append(
                f"in OpenDota only ({len(only_opendota)}): {sorted(only_opendota)[:20]}"
            )
        if not only_team and not only_opendota:
            notes.append("STRATZ team walk and OpenDota ID sets match")
    elif not team_ids and not opendota_ids:
        notes.append("no match ids discovered")

    combined = dedupe_match_ids(sorted(team_ids | opendota_ids))
    return MatchIdDiscoveryResult(
        league_id=league_id,
        team_match_ids=team_ids,
        opendota_match_ids=opendota_ids,
        match_ids=combined,
        teams_visited=teams_visited,
        notes=tuple(notes),
    )
