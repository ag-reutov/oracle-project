"""Shared fixtures for research-layer tests.

All DB-touching tests are skipped unless `TEST_DATABASE_URL` is set --
never `DATABASE_URL` (see `storage.engine.get_test_engine`).

Builds the canonical base tables from `storage.schema.METADATA`, then
creates the `research` schema + views via `dota_predictor.research.views`
(the exact SQL the Alembic migration applies), so the tests exercise the
real view definitions.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from dota_predictor.research.views import create_research_layer, drop_research_layer
from dota_predictor.storage.engine import get_test_engine
from dota_predictor.storage.schema import METADATA


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = get_test_engine()
    METADATA.drop_all(eng)
    METADATA.create_all(eng)
    drop_research_layer(eng)
    create_research_layer(eng)
    try:
        yield eng
    finally:
        drop_research_layer(eng)
        METADATA.drop_all(eng)
        eng.dispose()
