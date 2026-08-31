"""Tests for optional DuckDB reference views (`heroes`, `game_versions`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dota_predictor.datasets.reference_export import build_reference_dataset
from dota_predictor.features.config import FeatureStoreConfig, ReferenceStoreConfig
from dota_predictor.features.duckdb_layer import (
    DRAFT_EVENTS_VIEW,
    GAME_VERSIONS_VIEW,
    HEROES_VIEW,
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    connect,
    register_reference_views,
)


def _write_reference(tmp_path: Path) -> ReferenceStoreConfig:
    build_reference_dataset(
        tmp_path,
        heroes=[
            {"id": 1, "displayName": "Anti-Mage"},
            {"id": 2, "displayName": "Axe"},
        ],
        game_versions=[
            {"id": 173, "name": "7.36", "asOfDateTime": 1716422400},
            {"id": 175, "name": "7.36c", "asOfDateTime": 1719187200},
        ],
    )
    return ReferenceStoreConfig(
        heroes_path=tmp_path / "heroes.parquet",
        game_versions_path=tmp_path / "game_versions.parquet",
    )


def test_connect_works_without_reference_files(
    feature_store_config: FeatureStoreConfig,
) -> None:
    with connect(feature_store_config) as store:
        assert store.relation(MATCHES_VIEW).count("*").fetchone()[0] == 2
        tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
        assert HEROES_VIEW not in tables
        assert GAME_VERSIONS_VIEW not in tables
        assert {MATCHES_VIEW, DRAFT_EVENTS_VIEW, MATCH_PLAYERS_VIEW}.issubset(tables)


def test_register_reference_views_exposes_explicit_projections(
    feature_store_config: FeatureStoreConfig, tmp_path: Path
) -> None:
    reference_config = _write_reference(tmp_path)
    with connect(feature_store_config) as store:
        register_reference_views(store, reference_config)
        heroes = store.relation(HEROES_VIEW)
        versions = store.relation(GAME_VERSIONS_VIEW)
        assert list(heroes.columns) == ["hero_id", "name"]
        assert list(versions.columns) == [
            "game_version_id",
            "name",
            "as_of_datetime",
        ]
        assert heroes.count("*").fetchone()[0] == 2
        assert versions.count("*").fetchone()[0] == 2
        assert heroes.order("hero_id").fetchall() == [(1, "Anti-Mage"), (2, "Axe")]


def test_register_reference_views_does_not_join_facts(
    feature_store_config: FeatureStoreConfig, tmp_path: Path
) -> None:
    reference_config = _write_reference(tmp_path)
    with connect(feature_store_config) as store:
        fact_columns_before = {
            MATCHES_VIEW: list(store.relation(MATCHES_VIEW).columns),
            MATCH_PLAYERS_VIEW: list(store.relation(MATCH_PLAYERS_VIEW).columns),
            DRAFT_EVENTS_VIEW: list(store.relation(DRAFT_EVENTS_VIEW).columns),
        }
        register_reference_views(store, reference_config)
        assert (
            list(store.relation(MATCHES_VIEW).columns)
            == fact_columns_before[MATCHES_VIEW]
        )
        assert (
            list(store.relation(MATCH_PLAYERS_VIEW).columns)
            == fact_columns_before[MATCH_PLAYERS_VIEW]
        )
        assert (
            list(store.relation(DRAFT_EVENTS_VIEW).columns)
            == fact_columns_before[DRAFT_EVENTS_VIEW]
        )
        assert "name" not in store.relation(MATCH_PLAYERS_VIEW).columns
        assert "as_of_datetime" not in store.relation(MATCHES_VIEW).columns


def test_missing_reference_parquet_fails_clearly(
    feature_store_config: FeatureStoreConfig, tmp_path: Path
) -> None:
    config = ReferenceStoreConfig(
        heroes_path=tmp_path / "absent" / "heroes.parquet",
        game_versions_path=tmp_path / "absent" / "game_versions.parquet",
    )
    with (
        connect(feature_store_config) as store,
        pytest.raises(FileNotFoundError, match="heroes"),
    ):
        register_reference_views(store, config)
