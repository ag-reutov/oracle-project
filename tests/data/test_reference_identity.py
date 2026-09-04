"""Tests for the pure reference-identity derivation logic (Slice 3).

These are database-free tests of `dota_predictor.data.reference_identity`:
canonical hero / game-version loaders from reference Parquet, the
resolvers (`None` for unknown ids -- never silently invented), and the
first-seen-in-corpus derivation (labelled as corpus-derived, distinct from
the authoritative STRATZ release timestamp). DB-touching behavior (the
reference-entity census audit and `research.leagues` view) is covered
separately in `tests/storage/test_reference_identity_storage.py` and
`tests/research/test_league_identity_views.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dota_predictor.data.reference_identity import (
    GameVersionIdentity,
    HeroIdentity,
    derive_game_version_first_seen,
    load_game_versions,
    load_heroes,
    resolve_game_version,
    resolve_hero,
)
from dota_predictor.datasets.reference_export import build_reference_dataset

RETRIEVED = datetime(2026, 1, 1, tzinfo=UTC)


def _write_reference(tmp_path: Path) -> Path:
    build_reference_dataset(
        tmp_path,
        heroes=[
            {
                "id": 1,
                "displayName": "Anti-Mage",
                "shortName": "antimage",
                "aliases": ["am", "wei"],
            },
            {"id": 129, "displayName": "Mars", "shortName": "mars", "aliases": ["mars"]},
        ],
        game_versions=[
            {"id": 179, "name": "7.38", "asOfDateTime": 1739923200},
            {"id": 180, "name": "7.39", "asOfDateTime": 1748563200},
        ],
        retrieved_at=RETRIEVED,
    )
    return tmp_path


def test_load_heroes_reads_canonical_identity_with_provenance(tmp_path: Path) -> None:
    root = _write_reference(tmp_path)
    heroes = load_heroes(root / "heroes.parquet")
    assert [h.hero_id for h in heroes] == [1, 129]
    first = heroes[0]
    assert first.name == "Anti-Mage"
    assert first.short_name == "antimage"
    assert first.aliases == ["am", "wei"]
    assert first.source == "STRATZ constants.heroes"
    assert first.retrieved_at == RETRIEVED


def test_load_game_versions_reads_patch_label_and_release_timestamp(
    tmp_path: Path,
) -> None:
    root = _write_reference(tmp_path)
    versions = load_game_versions(root / "game_versions.parquet")
    assert [v.game_version_id for v in versions] == [179, 180]
    assert versions[0].name == "7.38"
    assert versions[0].as_of_datetime == datetime(2025, 2, 19, tzinfo=UTC)
    assert versions[0].source == "STRATZ constants.gameVersions"
    assert versions[0].retrieved_at == RETRIEVED


def test_resolve_hero_returns_identity_or_none(tmp_path: Path) -> None:
    heroes = load_heroes(_write_reference(tmp_path) / "heroes.parquet")
    resolved = resolve_hero(heroes, 1)
    assert isinstance(resolved, HeroIdentity)
    assert resolved.name == "Anti-Mage"
    # Unknown ids are reported as missing, never silently invented.
    assert resolve_hero(heroes, 999999) is None


def test_resolve_game_version_returns_identity_or_none(tmp_path: Path) -> None:
    versions = load_game_versions(_write_reference(tmp_path) / "game_versions.parquet")
    resolved = resolve_game_version(versions, 180)
    assert isinstance(resolved, GameVersionIdentity)
    assert resolved.name == "7.39"
    assert resolve_game_version(versions, 0) is None
    assert resolve_game_version(versions, 3000) is None


def test_derive_game_version_first_seen_is_deterministic() -> None:
    t1 = datetime(2024, 1, 10, tzinfo=UTC)
    t2 = datetime(2024, 2, 20, tzinfo=UTC)
    t3 = datetime(2024, 3, 15, tzinfo=UTC)
    observations = [(170, t2), (170, t1), (171, t3), (170, t2)]
    first_seen = derive_game_version_first_seen(observations)
    assert first_seen == {170: t1, 171: t3}
    # Reversed input order yields the same result.
    assert derive_game_version_first_seen(reversed(observations)) == first_seen