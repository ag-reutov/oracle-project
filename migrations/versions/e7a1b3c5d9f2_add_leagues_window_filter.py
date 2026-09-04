"""add leagues.window_filter

Revision ID: e7a1b3c5d9f2
Revises: d4f9a2b6c7e1
Create Date: 2026-09-04 00:00:00.000000

Adds `leagues.window_filter` so `match_ids`-mode leagues can restrict
match-ID discovery/ingest to their Liquipedia main-event date window.
This is how catalog-null leagues whose STRATZ league also contains
qualifiers (e.g. ESL Challenger China Season 2, PREMIER SERIES) ingest
only main-event matches. Default false preserves existing behavior.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7a1b3c5d9f2"
down_revision: str | Sequence[str] | None = "d4f9a2b6c7e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "leagues",
        sa.Column(
            "window_filter",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("leagues", "window_filter")