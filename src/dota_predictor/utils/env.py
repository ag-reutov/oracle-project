"""Project `.env` loading helper.

Loads `KEY=VALUE` pairs from the project-root `.env` file into
`os.environ`, without overriding variables already set in the
environment (so real environment variables always win over the file).

Several probe scripts under `scripts/` each inline an equivalent
function; this is the canonical version for new code (e.g. Alembic's
`migrations/env.py`) to import instead of duplicating it again.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_project_env"]


def load_project_env(root: Path) -> None:
    """Load `.env` from `root` into `os.environ`, if the file exists."""
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
