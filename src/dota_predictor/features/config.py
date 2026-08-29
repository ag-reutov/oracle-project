"""Feature-layer configuration (Step 3A).

The DuckDB analytical layer under `features/` reads exactly the two files
`datasets.canonical_export` writes (`matches.parquet`, `draft_events.parquet`)
from the canonical processed-data directory. This module resolves the
*paths to those two files*, but deliberately delegates directory
resolution to the existing `datasets.config.load_dataset_export_config`
(same `PROCESSED_DATA_DIR` env var, same `data/processed` default) instead
of re-implementing that env/path logic here -- there is exactly one
place in the codebase that decides where the canonical Parquet dataset
lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dota_predictor.datasets.canonical_export import (
    DRAFT_EVENTS_FILENAME,
    MATCHES_FILENAME,
)
from dota_predictor.datasets.config import load_dataset_export_config

__all__ = ["FeatureStoreConfig", "load_feature_store_config"]


@dataclass(frozen=True)
class FeatureStoreConfig:
    """Paths to the canonical Parquet files the feature layer reads."""

    matches_path: Path
    draft_events_path: Path


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
        draft_events_path=export_config.output_dir / DRAFT_EVENTS_FILENAME,
    )
