"""Pure tests for the STRATZ reference-dimension Parquet export.

No live STRATZ, no PostgreSQL: synthetic constants rows are transformed
and published to a temporary directory.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.datasets.reference_export import (
    GAME_VERSIONS_FILENAME,
    GAME_VERSIONS_SCHEMA,
    HEROES_FILENAME,
    HEROES_SCHEMA,
    REFERENCE_SCHEMA_VERSION,
    ReferenceTransformError,
    ReferenceValidationError,
    build_game_versions_table,
    build_heroes_table,
    build_reference_dataset,
    validate_game_versions_table,
    validate_heroes_table,
    write_reference_dataset,
)


def test_analytical_schema_version_is_unchanged() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 4
    assert REFERENCE_SCHEMA_VERSION == 1


def test_heroes_maps_display_name_and_int32_id() -> None:
    table = build_heroes_table(
        [
            {
                "id": 1,
                "displayName": "Anti-Mage",
                "name": "npc_dota_hero_antimage",
                "shortName": "antimage",
                "aliases": ["am"],
                "gameVersionId": 182,
            },
            {"id": 2, "displayName": "Axe", "name": "npc_dota_hero_axe"},
        ]
    )
    assert table.schema.equals(HEROES_SCHEMA)
    assert table.column_names == ["hero_id", "name"]
    assert table.to_pylist() == [
        {"hero_id": 1, "name": "Anti-Mage"},
        {"hero_id": 2, "name": "Axe"},
    ]
    assert table.schema.field("hero_id").type == HEROES_SCHEMA.field("hero_id").type


def test_heroes_ignores_unused_stratz_fields() -> None:
    table = build_heroes_table(
        [
            {
                "id": 145,
                "displayName": "Kez",
                "name": "npc_dota_hero_kez",
                "shortName": "kez",
                "aliases": ["bird samurai"],
                "gameVersionId": 182,
                "roles": [{"roleId": 1}],
            }
        ]
    )
    published = table.to_pylist()[0]
    assert set(published) == {"hero_id", "name"}
    assert "shortName" not in table.column_names
    assert "aliases" not in table.column_names
    assert "gameVersionId" not in table.column_names
    assert published == {"hero_id": 145, "name": "Kez"}


def test_heroes_rejects_null_id() -> None:
    with pytest.raises(ReferenceTransformError, match="hero_id"):
        build_heroes_table([{"id": None, "displayName": "Axe"}])


def test_heroes_rejects_non_positive_id() -> None:
    with pytest.raises(ReferenceTransformError, match="positive"):
        build_heroes_table([{"id": 0, "displayName": "Axe"}])
    with pytest.raises(ReferenceTransformError, match="positive"):
        build_heroes_table([{"id": -1, "displayName": "Axe"}])


def test_heroes_rejects_empty_name() -> None:
    with pytest.raises(ReferenceTransformError, match="empty"):
        build_heroes_table([{"id": 1, "displayName": ""}])
    with pytest.raises(ReferenceTransformError, match="empty"):
        build_heroes_table([{"id": 1, "displayName": "   "}])


def test_heroes_rejects_missing_display_name() -> None:
    with pytest.raises(ReferenceTransformError, match="name"):
        build_heroes_table([{"id": 1}])


def test_heroes_rejects_duplicate_ids() -> None:
    table = build_heroes_table(
        [
            {"id": 1, "displayName": "Anti-Mage"},
            {"id": 1, "displayName": "Axe"},
        ]
    )
    with pytest.raises(ReferenceValidationError, match="duplicate hero_id"):
        validate_heroes_table(table)


def test_game_versions_unix_seconds_to_utc_and_preserves_name() -> None:
    table = build_game_versions_table(
        [
            {"id": 170, "name": "7.35b", "asOfDateTime": 1703203200},
            {"id": 179, "name": "7.38", "asOfDateTime": 1739923200},
        ]
    )
    assert table.schema.equals(GAME_VERSIONS_SCHEMA)
    rows = table.to_pylist()
    assert rows[0]["name"] == "7.35b"
    assert rows[1]["name"] == "7.38"
    assert rows[0]["as_of_datetime"] == datetime(2023, 12, 22, tzinfo=UTC)
    assert rows[1]["as_of_datetime"] == datetime(2025, 2, 19, tzinfo=UTC)


def test_game_versions_allows_id_gap() -> None:
    """STRATZ id 174 is missing in production; that must not fail validation."""
    table = build_game_versions_table(
        [
            {"id": 173, "name": "7.36", "asOfDateTime": 1716422400},
            {"id": 175, "name": "7.36c", "asOfDateTime": 1719187200},
        ]
    )
    validate_game_versions_table(table)
    assert table.column("game_version_id").to_pylist() == [173, 175]


def test_game_versions_allows_non_monotonic_timestamps() -> None:
    table = build_game_versions_table(
        [
            {"id": 1, "name": "6.70", "asOfDateTime": 1295308800},
            {"id": 2, "name": "6.70b", "asOfDateTime": 1293494400},
        ]
    )
    validate_game_versions_table(table)
    stamps = table.column("as_of_datetime").to_pylist()
    assert stamps[0] > stamps[1]


def test_game_versions_rejects_duplicate_ids() -> None:
    table = build_game_versions_table(
        [
            {"id": 179, "name": "7.38", "asOfDateTime": 1739923200},
            {"id": 179, "name": "7.38-dup", "asOfDateTime": 1739923201},
        ]
    )
    with pytest.raises(ReferenceValidationError, match="duplicate game_version_id"):
        validate_game_versions_table(table)


def test_game_versions_rejects_duplicate_names() -> None:
    table = build_game_versions_table(
        [
            {"id": 179, "name": "7.38", "asOfDateTime": 1739923200},
            {"id": 180, "name": "7.38", "asOfDateTime": 1748563200},
        ]
    )
    with pytest.raises(ReferenceValidationError, match="duplicate name"):
        validate_game_versions_table(table)


def test_game_versions_rejects_missing_timestamp() -> None:
    with pytest.raises(ReferenceTransformError, match="asOfDateTime"):
        build_game_versions_table([{"id": 179, "name": "7.38", "asOfDateTime": None}])


def test_write_reference_dataset_round_trip(tmp_path: Path) -> None:
    heroes = build_heroes_table(
        [{"id": 1, "displayName": "Anti-Mage"}, {"id": 2, "displayName": "Axe"}]
    )
    versions = build_game_versions_table(
        [
            {"id": 173, "name": "7.36", "asOfDateTime": 1716422400},
            {"id": 175, "name": "7.36c", "asOfDateTime": 1719187200},
        ]
    )
    write_reference_dataset(tmp_path, heroes_table=heroes, game_versions_table=versions)

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [HEROES_FILENAME, GAME_VERSIONS_FILENAME]
    )
    read_heroes = pq.read_table(tmp_path / HEROES_FILENAME)
    read_versions = pq.read_table(tmp_path / GAME_VERSIONS_FILENAME)
    assert read_heroes.schema.equals(HEROES_SCHEMA)
    assert read_versions.schema.equals(GAME_VERSIONS_SCHEMA)
    assert read_heroes.to_pylist() == heroes.to_pylist()
    assert read_versions.to_pylist() == versions.to_pylist()


def test_build_reference_dataset_publishes_both_files(tmp_path: Path) -> None:
    result = build_reference_dataset(
        tmp_path,
        heroes=[{"id": 1, "displayName": "Anti-Mage"}],
        game_versions=[
            {"id": 173, "name": "7.36", "asOfDateTime": 1716422400},
            {"id": 175, "name": "7.36c", "asOfDateTime": 1719187200},
        ],
    )
    assert result.schema_version == REFERENCE_SCHEMA_VERSION
    assert result.heroes_row_count == 1
    assert result.game_versions_row_count == 2
    assert result.heroes_path == tmp_path / HEROES_FILENAME
    assert result.game_versions_path == tmp_path / GAME_VERSIONS_FILENAME
    assert (tmp_path / "matches.parquet").exists() is False
