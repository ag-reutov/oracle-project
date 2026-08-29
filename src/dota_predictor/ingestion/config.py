"""Configuration for STRATZ ingestion (env-driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "IngestionConfig",
    "MissingStratzTokenError",
    "load_ingestion_config",
]

STRATZ_GRAPHQL_ENDPOINT = "https://api.stratz.com/graphql"
DEFAULT_PAGE_SIZE = 100
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.25  # ~4 req/s, below observed ~8/s limit
DEFAULT_REQUEST_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RETRY_ATTEMPTS = 5


class MissingStratzTokenError(RuntimeError):
    """Raised when STRATZ_API_TOKEN is not configured."""


@dataclass(frozen=True)
class IngestionConfig:
    stratz_api_token: str
    graphql_endpoint: str = STRATZ_GRAPHQL_ENDPOINT
    page_size: int = DEFAULT_PAGE_SIZE
    min_request_interval_seconds: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_retry_attempts: int = DEFAULT_MAX_RETRY_ATTEMPTS
    user_agent: str = "dota-predictor-ingestion/0.1"


def load_ingestion_config() -> IngestionConfig:
    token = os.environ.get("STRATZ_API_TOKEN", "").strip()
    if not token:
        raise MissingStratzTokenError(
            "STRATZ_API_TOKEN is missing. Set it in the environment or project .env."
        )
    return IngestionConfig(stratz_api_token=token)
