"""CLI for building the STRATZ reference-dimension Parquet catalogs.

Usage:
    uv run python scripts/build_reference_dataset.py

Fetches `constants.heroes` and `constants.gameVersions` via the existing
`StratzClient`, then delegates transform/validate/publish to
`datasets.reference_export.build_reference_dataset`. Independent of
`build_canonical_dataset`: this script never opens PostgreSQL and never
rewrites match-fact Parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dota_predictor.datasets.config import load_dataset_export_config
from dota_predictor.datasets.reference_export import (
    REFERENCE_SCHEMA_VERSION,
    ReferenceExportError,
    build_reference_dataset,
)
from dota_predictor.ingestion.client import StratzClient
from dota_predictor.ingestion.config import (
    MissingStratzTokenError,
    load_ingestion_config,
)
from dota_predictor.ingestion.errors import StratzClientError
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _project_root()
    load_project_env(root)

    try:
        ingestion_config = load_ingestion_config()
    except MissingStratzTokenError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    export_config = load_dataset_export_config(root=root)

    try:
        with StratzClient(ingestion_config) as client:
            heroes = client.fetch_heroes()
            game_versions = client.fetch_game_versions()
        result = build_reference_dataset(
            export_config.output_dir,
            heroes=heroes,
            game_versions=game_versions,
        )
    except (ReferenceExportError, StratzClientError) as exc:
        print(f"Reference dataset build failed: {exc}", file=sys.stderr)
        return 1

    print("Reference dataset build complete")
    print(f"schema version: {REFERENCE_SCHEMA_VERSION}")
    print(f"heroes: {result.heroes_row_count}")
    print(f"game versions: {result.game_versions_row_count}")
    print(f"heroes path: {result.heroes_path}")
    print(f"game versions path: {result.game_versions_path}")
    print(f"output: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
