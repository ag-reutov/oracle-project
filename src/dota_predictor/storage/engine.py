"""Postgres engine creation, resolved from configuration (env vars).

Per project convention, connection strings are never hard-coded. Two
separate env vars exist on purpose:

* `DATABASE_URL` -- the dev/prod database.
* `TEST_DATABASE_URL` -- a dedicated test database. Code that needs a
  test engine must call `get_test_engine`, which raises rather than
  silently falling back to `DATABASE_URL`, so a misconfigured test run
  can never touch dev/prod data.
"""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine

__all__ = ["get_engine", "get_test_engine"]


class MissingDatabaseUrlError(RuntimeError):
    """Raised when a required database connection env var is not set."""


def _create_engine_from_env(env_var: str) -> Engine:
    url = os.environ.get(env_var, "").strip()
    if not url:
        raise MissingDatabaseUrlError(
            f"{env_var} is not set. Copy .env.example to .env and fill it in "
            "(see docker-compose.yml for local defaults)."
        )
    return create_engine(url, future=True)


def get_engine() -> Engine:
    """Create an engine for the dev/prod database from `DATABASE_URL`."""
    return _create_engine_from_env("DATABASE_URL")


def get_test_engine() -> Engine:
    """Create an engine for the dedicated test database.

    Deliberately reads only `TEST_DATABASE_URL`, never `DATABASE_URL` --
    tests must be able to run against a real Postgres without any risk of
    touching dev/prod data if the test env var happens to be unset.
    """
    return _create_engine_from_env("TEST_DATABASE_URL")
