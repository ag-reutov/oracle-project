"""Shared fixtures for ingestion tests."""

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
