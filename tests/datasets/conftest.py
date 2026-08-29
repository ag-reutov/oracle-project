"""Shared fixtures for dataset-export tests.

All DB-touching tests are skipped unless `TEST_DATABASE_URL` is set --
never `DATABASE_URL` (see `helpers.requires_test_database` /
`storage.engine.get_test_engine`).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from dota_predictor.storage.engine import get_test_engine
from dota_predictor.storage.schema import METADATA


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = get_test_engine()
    METADATA.drop_all(eng)
    METADATA.create_all(eng)
    try:
        yield eng
    finally:
        METADATA.drop_all(eng)
        eng.dispose()
