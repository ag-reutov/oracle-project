"""STRATZ GraphQL HTTP client with retry and rate limiting."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, Self

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dota_predictor.ingestion.config import IngestionConfig
from dota_predictor.ingestion.errors import StratzPermanentError, StratzRetryableError
from dota_predictor.ingestion.queries import (
    GAME_VERSIONS_QUERY,
    HEROES_QUERY,
    LEAGUE_MATCHES_QUERY,
    MATCH_BY_ID_QUERY,
    TEAM_LEAGUE_MATCH_IDS_QUERY,
)

__all__ = [
    "LeagueMatchesFetcher",
    "MatchByIdFetcher",
    "StratzClient",
    "TeamLeagueMatchIdsFetcher",
    "parse_game_versions_query_payload",
    "parse_heroes_query_payload",
    "parse_match_query_payload",
]

logger = logging.getLogger(__name__)


class LeagueMatchesFetcher(Protocol):
    """Protocol for fetching league match pages (enables mocking in tests)."""

    def fetch_league_matches_page(
        self,
        league_id: int,
        *,
        skip: int,
        take: int,
    ) -> list[dict[str, Any]]: ...


class MatchByIdFetcher(Protocol):
    """Protocol for fetching one STRATZ match by id."""

    def fetch_match(self, match_id: int) -> dict[str, Any] | None: ...


class TeamLeagueMatchIdsFetcher(Protocol):
    """Protocol for paginating a team's matches in one league (id harvest)."""

    def fetch_team_league_match_ids_page(
        self,
        team_id: int,
        *,
        league_id: int,
        skip: int,
        take: int,
    ) -> list[dict[str, Any]]: ...


def parse_match_query_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the `match` object from a `match(id)` GraphQL response.

    Returns `None` when STRATZ has no row for that id (`data.match` is
    null). Nested `league` metadata may be null; callers must not require
    it. Raises `StratzPermanentError` if the payload is structurally
    unusable (missing `data`, or a non-object match).
    """
    data = payload.get("data")
    if data is None:
        raise StratzPermanentError("match(id) response missing data")
    match = data.get("match")
    if match is None:
        return None
    if not isinstance(match, dict) or match.get("id") is None:
        raise StratzPermanentError("match(id) returned a malformed match object")
    return match


def _constants_list(payload: dict[str, Any], field: str) -> list[dict[str, Any]]:
    """Extract `data.constants.<field>` as a list of objects.

    Raises `StratzPermanentError` if the payload is missing `data` or
    `constants`, or if the named field is present but not a list. A
    null/absent list is treated as empty -- catalog validation belongs
    to the reference-export layer, not this parser.
    """
    data = payload.get("data")
    if data is None:
        raise StratzPermanentError(f"constants.{field} response missing data")
    constants = data.get("constants")
    if constants is None:
        raise StratzPermanentError(f"constants.{field} response missing constants")
    rows = constants.get(field)
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise StratzPermanentError(f"constants.{field} returned a non-list value")
    return list(rows)


def parse_heroes_query_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the `heroes` list from a `constants.heroes` GraphQL response."""
    return _constants_list(payload, "heroes")


def parse_game_versions_query_payload(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract the `gameVersions` list from a `constants.gameVersions`
    GraphQL response."""
    return _constants_list(payload, "gameVersions")


def _graphql_errors_are_retryable(errors: list[dict[str, Any]]) -> bool:
    for error in errors:
        message = (error.get("message") or "").lower()
        code = (error.get("extensions") or {}).get("code", "")
        if code in {"INVALID_VALUE"}:
            return False
        if "maximum take value" in message:
            return False
        if "unrecognized input fields" in message:
            return False
    return True


def _retry_after_seconds(headers: dict[str, str], default: float = 1.0) -> float:
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        reset = headers.get("ratelimit-reset") or headers.get("RateLimit-Reset")
        if reset is not None:
            try:
                return max(float(reset), 0.25)
            except ValueError:
                pass
        return default
    try:
        return max(float(retry_after), 0.25)
    except ValueError:
        return default


class StratzClient:
    """Sequential STRATZ GraphQL client for league match pagination."""

    def __init__(self, config: IngestionConfig) -> None:
        self._config = config
        self._last_request_at: float | None = None
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {config.stratz_api_token}",
                "Content-Type": "application/json",
                "User-Agent": config.user_agent,
            },
            timeout=config.request_timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait = self._config.min_request_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    def _post_graphql(
        self,
        query: str,
        variables: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        self._throttle()
        try:
            response = self._client.post(
                self._config.graphql_endpoint,
                json={"query": query, "variables": variables},
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:
            raise StratzRetryableError(str(exc)) from exc

        self._last_request_at = time.monotonic()
        headers = {k.lower(): v for k, v in response.headers.items()}

        if response.status_code == 429:
            time.sleep(_retry_after_seconds(headers))
            raise StratzRetryableError("HTTP 429 rate limited")

        if response.status_code >= 500:
            raise StratzRetryableError(f"HTTP {response.status_code} server error")

        if response.status_code == 400:
            raise StratzPermanentError(f"HTTP 400: {response.text[:500]}")

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise StratzRetryableError("Response was not valid JSON") from exc

        if not response.is_success:
            raise StratzPermanentError(
                f"HTTP {response.status_code}: {payload!r}"[:500]
            )

        errors = payload.get("errors")
        if errors:
            if _graphql_errors_are_retryable(errors):
                raise StratzRetryableError(str(errors))
            raise StratzPermanentError(str(errors))

        return payload, headers

    def _fetch_with_retry(
        self,
        query: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        @retry(
            retry=retry_if_exception_type(StratzRetryableError),
            wait=wait_exponential_jitter(initial=1, max=60),
            stop=stop_after_attempt(self._config.max_retry_attempts),
            reraise=True,
        )
        def _do_fetch() -> dict[str, Any]:
            payload, _headers = self._post_graphql(query, variables)
            return payload

        return _do_fetch()

    def fetch_league_matches_page(
        self,
        league_id: int,
        *,
        skip: int,
        take: int,
    ) -> list[dict[str, Any]]:
        payload = self._fetch_with_retry(
            LEAGUE_MATCHES_QUERY,
            {
                "id": league_id,
                "request": {"skip": skip, "take": take},
            },
        )
        league = (payload.get("data") or {}).get("league")
        if league is None:
            raise StratzPermanentError(f"League {league_id} not found in STRATZ")
        return list(league.get("matches") or [])

    def fetch_match(self, match_id: int) -> dict[str, Any] | None:
        payload = self._fetch_with_retry(MATCH_BY_ID_QUERY, {"id": match_id})
        return parse_match_query_payload(payload)

    def fetch_team_league_match_ids_page(
        self,
        team_id: int,
        *,
        league_id: int,
        skip: int,
        take: int,
    ) -> list[dict[str, Any]]:
        payload = self._fetch_with_retry(
            TEAM_LEAGUE_MATCH_IDS_QUERY,
            {
                "teamId": team_id,
                "request": {
                    "leagueId": league_id,
                    "skip": skip,
                    "take": take,
                },
            },
        )
        team = (payload.get("data") or {}).get("team")
        if team is None:
            return []
        return list(team.get("matches") or [])

    def fetch_heroes(self) -> list[dict[str, Any]]:
        """Fetch the STRATZ hero identity catalog (`id`, `displayName`)."""
        payload = self._fetch_with_retry(HEROES_QUERY, {})
        return parse_heroes_query_payload(payload)

    def fetch_game_versions(self) -> list[dict[str, Any]]:
        """Fetch the STRATZ game-version catalog (`id`, `name`, `asOfDateTime`)."""
        payload = self._fetch_with_retry(GAME_VERSIONS_QUERY, {})
        return parse_game_versions_query_payload(payload)
