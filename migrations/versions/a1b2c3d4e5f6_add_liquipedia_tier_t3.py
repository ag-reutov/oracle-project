"""add T3 to liquipedia_tier CHECK constraint

Revision ID: a1b2c3d4e5f6
Revises: b7c3e9d1a4f2
Create Date: 2026-09-03 19:30:00.000000

Allows Liquipedia Tier 3 professional leagues in the registry so T1/T2/T3
cohorts remain distinct and joinable for later training experiments.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "b7c3e9d1a4f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = ("T1", "T2", "MINOR", "QUALIFIER", "EXCLUDED")
_NEW = ("T1", "T2", "T3", "MINOR", "QUALIFIER", "EXCLUDED")


def upgrade() -> None:
    op.drop_constraint(op.f("ck_leagues_liquipedia_tier_valid"), "leagues", type_="check")
    op.create_check_constraint(
        op.f("ck_leagues_liquipedia_tier_valid"),
        "leagues",
        f"liquipedia_tier IN {_NEW!r}",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_leagues_liquipedia_tier_valid"), "leagues", type_="check")
    op.create_check_constraint(
        op.f("ck_leagues_liquipedia_tier_valid"),
        "leagues",
        f"liquipedia_tier IN {_OLD!r}",
    )
