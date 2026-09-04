"""add matches.draft_complete

Revision ID: d4f9a2b6c7e1
Revises: a1b2c3d4e5f6
Create Date: 2026-09-04 00:00:00.000000

Adds `matches.draft_complete` so a professional match whose source draft
is absent (STRATZ `pickBans` null/empty or malformed) can exist
canonically without a draft. Existing canonical rows all have complete
drafts (draft-less payloads previously failed canonicalization and were
never written), so the default `true` is a faithful backfill.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4f9a2b6c7e1"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "matches",
        sa.Column(
            "draft_complete",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("matches", "draft_complete")