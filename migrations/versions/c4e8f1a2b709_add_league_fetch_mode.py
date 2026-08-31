"""add league fetch_mode

Revision ID: c4e8f1a2b709
Revises: 37102448abc6
Create Date: 2026-08-31 12:30:00.000000

Adds `leagues.fetch_mode` so catalog-null STRATZ leagues can be ingested
via `match(id)` without depending on a previous run having stored
`cursor_state.mode = match_ids`. Default `league` preserves the existing
`league(id) { matches }` path. Independent of `in_scope`.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e8f1a2b709"
down_revision: str | Sequence[str] | None = "37102448abc6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FETCH_MODES = ("league", "match_ids")


def upgrade() -> None:
    op.add_column(
        "leagues",
        sa.Column(
            "fetch_mode",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'league'"),
        ),
    )
    op.create_check_constraint(
        op.f("ck_leagues_fetch_mode_valid"),
        "leagues",
        f"fetch_mode IN {_FETCH_MODES!r}",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_leagues_fetch_mode_valid"), "leagues", type_="check")
    op.drop_column("leagues", "fetch_mode")
