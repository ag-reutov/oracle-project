"""Feature-layer configuration (Step 3A).

The DuckDB analytical layer under `features/` reads exactly the three files
`datasets.canonical_export` writes (`matches.parquet`, `match_players.parquet`,
`draft_events.parquet`) from the canonical processed-data directory. This
module resolves the *paths to those three files*, but deliberately delegates
directory resolution to the existing `datasets.config.load_dataset_export_config`
(same `PROCESSED_DATA_DIR` env var, same `data/processed` default) instead
of re-implementing that env/path logic here -- there is exactly one
place in the codebase that decides where the canonical Parquet dataset
lives.

Reference catalogs (`heroes.parquet`, `game_versions.parquet`) share that
directory but are a separate contract (`REFERENCE_SCHEMA_VERSION`). They
are resolved by `ReferenceStoreConfig` and are never required by
`load_feature_store_config` / `connect()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dota_predictor.datasets.canonical_export import (
    DRAFT_EVENTS_FILENAME,
    MATCH_PLAYERS_FILENAME,
    MATCHES_FILENAME,
)
from dota_predictor.datasets.config import load_dataset_export_config
from dota_predictor.datasets.reference_export import (
    GAME_VERSIONS_FILENAME,
    HEROES_FILENAME,
)

__all__ = [
    "FeatureStoreConfig",
    "ReferenceStoreConfig",
    "load_feature_store_config",
    "load_reference_store_config",
]


@dataclass(frozen=True)
class FeatureStoreConfig:
    """Paths to the canonical Parquet files the feature layer reads."""

    matches_path: Path
    match_players_path: Path
    draft_events_path: Path


@dataclass(frozen=True)
class ReferenceStoreConfig:
    """Paths to the optional STRATZ reference-dimension Parquet files.

    Defaults to the same processed-data directory as `FeatureStoreConfig`.
    Missing files are not an error at config-load time; `connect()` does
    not require them. `register_reference_views` requires both to exist.
    """

    heroes_path: Path
    game_versions_path: Path


def load_feature_store_config(*, root: Path | None = None) -> FeatureStoreConfig:
    """Resolve the feature layer's canonical Parquet file paths.

    `root`, if given, is forwarded to `load_dataset_export_config` the
    same way CLI scripts already do (see
    `scripts/build_canonical_dataset.py`), so the feature layer and the
    Step 2 export agree on where the dataset lives regardless of the
    current working directory the caller happens to run from.
    """
    export_config = load_dataset_export_config(root=root)
    return FeatureStoreConfig(
        matches_path=export_config.output_dir / MATCHES_FILENAME,
        match_players_path=export_config.output_dir / MATCH_PLAYERS_FILENAME,
        draft_events_path=export_config.output_dir / DRAFT_EVENTS_FILENAME,
    )


def load_reference_store_config(*, root: Path | None = None) -> ReferenceStoreConfig:
    """Resolve reference-dimension Parquet paths in the processed-data directory."""
    export_config = load_dataset_export_config(root=root)
    return ReferenceStoreConfig(
        heroes_path=export_config.output_dir / HEROES_FILENAME,
        game_versions_path=export_config.output_dir / GAME_VERSIONS_FILENAME,
    )
