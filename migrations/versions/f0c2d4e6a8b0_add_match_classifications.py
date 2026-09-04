"""add match_classifications

Revision ID: f0c2d4e6a8b0
Revises: e7a1b3c5d9f2
Create Date: 2026-09-04 00:00:00.000000

Adds `match_classifications` so matches in a shared STRATZ league that
belong to a different Liquipedia event/tier than the league's default can
be classified explicitly (e.g. the 9 Asian Champions League 2025 finals
matches inside T3 league 17875). Populated from
config/event_match_assignments.yaml by scripts/apply_event_classifications.py.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f0c2d4e6a8b0"
down_revision: str | Sequence[str] | None = "e7a1b3c5d9f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LIQUIPEDIA_TIERS = ("T1", "T2", "T3", "MINOR", "QUALIFIER", "EXCLUDED")


def upgrade() -> None:
    op.create_table(
        "match_classifications",
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("liquipedia_event", sa.Text(), nullable=False),
        sa.Column("liquipedia_tier", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.match_id"],
            ondelete="CASCADE",
            name=op.f("fk_match_classifications_match_id_matches"),
        ),
        sa.PrimaryKeyConstraint("match_id", name=op.f("pk_match_classifications")),
        sa.CheckConstraint(
            f"liquipedia_tier IN {_LIQUIPEDIA_TIERS!r}",
            name=op.f("ck_match_classifications_liquipedia_tier_valid"),
        ),
    )


def downgrade() -> None:
    op.drop_table("match_classifications")