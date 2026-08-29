"""Configuration for the canonical PostgreSQL -> Parquet dataset export.

Mirrors `ingestion.config`'s env-driven pattern (see `.cursor/rules/project.mdc`:
paths/settings are resolved from configuration, never hard-coded in
application code): the output directory is resolved from an optional
environment variable, defaulting to the project's existing `data/processed`
convention (see `.gitignore` -- `data/raw` / `data/interim` / `data/processed`
are already established, currently-empty pipeline stages).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DatasetExportConfig", "load_dataset_export_config"]

# Relative to the project root; matches the existing data/raw, data/interim,
# data/processed layout.
DEFAULT_PROCESSED_DATA_DIR = Path("data/processed")


@dataclass(frozen=True)
class DatasetExportConfig:
    """Where the canonical analytical Parquet dataset is written."""

    output_dir: Path


def load_dataset_export_config(*, root: Path | None = None) -> DatasetExportConfig:
    """Resolve dataset export configuration.

    If the `PROCESSED_DATA_DIR` environment variable is set, it is used
    verbatim (absolute, or relative to the current working directory).
    Otherwise defaults to `data/processed` under `root` -- callers such as
    CLI scripts should pass their own resolved project root (see
    `utils.env.load_project_env` for the equivalent pattern), so the
    default does not depend on the current working directory the script
    happens to be invoked from. If `root` is not given, the default falls
    back to being relative to the current working directory.
    """
    raw = os.environ.get("PROCESSED_DATA_DIR", "").strip()
    if raw:
        return DatasetExportConfig(output_dir=Path(raw))
    base = root if root is not None else Path.cwd()
    return DatasetExportConfig(output_dir=base / DEFAULT_PROCESSED_DATA_DIR)
