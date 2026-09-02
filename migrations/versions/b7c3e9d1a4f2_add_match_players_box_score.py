"""add match_players post-match box-score scalars

Revision ID: b7c3e9d1a4f2
Revises: f4a91c2d8e37
Create Date: 2026-09-01 18:00:00.000000

Adds observed STRATZ match-player box-score scalars. Columns are
nullable: missing source values must not fail canonicalization and must
not be coerced to zero. Stored raw payloads do not currently contain
these fields; a separate refetch/backfill writes them without replacing
unrelated raw or canonical columns.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7c3e9d1a4f2"
down_revision: str | Sequence[str] | None = "f4a91c2d8e37"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = (
    "kills",
    "deaths",
    "assists",
    "gold_per_minute",
    "experience_per_minute",
    "num_last_hits",
    "num_denies",
    "networth",
    "hero_damage",
    "tower_damage",
    "hero_healing",
    "level",
)


def upgrade() -> None:
    for column in _COLUMNS:
        op.add_column("match_players", sa.Column(column, sa.Integer(), nullable=True))


def downgrade() -> None:
    for column in reversed(_COLUMNS):
        op.drop_column("match_players", column)
