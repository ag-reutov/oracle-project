"""CLI for building the canonical PostgreSQL -> Parquet analytical dataset.

Usage:
    uv run python scripts/build_canonical_dataset.py

Reads `DATABASE_URL` (never a hard-coded connection string) and the
optional `PROCESSED_DATA_DIR` override (see `datasets.config`), then
delegates the actual read/transform/validate/publish work to
`datasets.canonical_export.build_canonical_dataset`. This script itself
contains no transformation logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dota_predictor.datasets.canonical_export import (
    CanonicalExportError,
    build_canonical_dataset,
)
from dota_predictor.datasets.config import load_dataset_export_config
from dota_predictor.storage.engine import MissingDatabaseUrlError, get_engine
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _project_root()
    load_project_env(root)

    try:
        engine = get_engine()
    except MissingDatabaseUrlError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    config = load_dataset_export_config(root=root)

    try:
        result = build_canonical_dataset(engine, config.output_dir)
    except CanonicalExportError as exc:
        print(f"Canonical dataset build failed: {exc}", file=sys.stderr)
        return 1

    print("Canonical dataset build complete")
    print(f"matches: {result.matches_row_count}")
    print(f"draft events: {result.draft_events_row_count}")
    print(f"output: {result.output_dir}")
    print(f"schema version: {result.schema_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
