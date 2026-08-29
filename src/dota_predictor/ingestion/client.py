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
from dota_predictor.ingestion.queries import LEAGUE_MATCHES_QUERY

__all__ = ["LeagueMatchesFetcher", "StratzClient"]

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
