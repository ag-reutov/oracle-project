"""Ingestion-specific exceptions."""

from __future__ import annotations

__all__ = [
    "LeagueFetchModeError",
    "LeagueNotAllowlistedError",
    "LeagueNotRegisteredError",
    "PageValidationError",
    "PaginationDriftError",
    "StratzClientError",
    "StratzPermanentError",
    "StratzRetryableError",
]


class StratzClientError(Exception):
    """Base class for STRATZ client failures."""


class StratzRetryableError(StratzClientError):
    """Transient failure that may succeed on retry."""


class StratzPermanentError(StratzClientError):
    """Non-retryable client/query failure."""


class PageValidationError(Exception):
    """Fetched page failed defensive validation before persistence."""


class PaginationDriftError(Exception):
    """Resume anchor verification failed; offset pagination is unsafe."""


class LeagueNotAllowlistedError(Exception):
    """League is not present in ingestion_leagues."""


class LeagueNotRegisteredError(Exception):
    """League is not present in the `leagues` registry table."""


class LeagueFetchModeError(Exception):
    """Configured `fetch_mode` cannot be executed with the given fetcher."""
