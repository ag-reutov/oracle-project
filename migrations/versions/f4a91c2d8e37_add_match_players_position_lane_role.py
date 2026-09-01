"""add match_players position lane role

Revision ID: f4a91c2d8e37
Revises: e8f3a1c7b204
Create Date: 2026-09-01 10:00:00.000000

Adds observed STRATZ match-player `position`, `lane`, and `role`.
Columns are nullable: missing parse metadata must not fail
canonicalization. Stored raw payloads do not currently contain these
fields; a separate refetch/backfill writes them without replacing
unrelated raw or canonical columns.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f4a91c2d8e37"
down_revision: str | Sequence[str] | None = "e8f3a1c7b204"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_POSITION_VALUES = (
    "POSITION_1",
    "POSITION_2",
    "POSITION_3",
    "POSITION_4",
    "POSITION_5",
    "UNKNOWN",
    "FILTERED",
    "ALL",
)
_LANE_VALUES = (
    "ROAMING",
    "SAFE_LANE",
    "MID_LANE",
    "OFF_LANE",
    "JUNGLE",
    "UNKNOWN",
)
_ROLE_VALUES = ("CORE", "LIGHT_SUPPORT", "HARD_SUPPORT", "UNKNOWN")


def _in_list(values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"({quoted})"


def upgrade() -> None:
    op.add_column("match_players", sa.Column("position", sa.String(), nullable=True))
    op.add_column("match_players", sa.Column("lane", sa.String(), nullable=True))
    op.add_column("match_players", sa.Column("role", sa.String(), nullable=True))
    op.create_check_constraint(
        op.f("ck_match_players_position_known"),
        "match_players",
        f"position IS NULL OR position IN {_in_list(_POSITION_VALUES)}",
    )
    op.create_check_constraint(
        op.f("ck_match_players_lane_known"),
        "match_players",
        f"lane IS NULL OR lane IN {_in_list(_LANE_VALUES)}",
    )
    op.create_check_constraint(
        op.f("ck_match_players_role_known"),
        "match_players",
        f"role IS NULL OR role IN {_in_list(_ROLE_VALUES)}",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_match_players_role_known"), "match_players", type_="check"
    )
    op.drop_constraint(
        op.f("ck_match_players_lane_known"), "match_players", type_="check"
    )
    op.drop_constraint(
        op.f("ck_match_players_position_known"), "match_players", type_="check"
    )
    op.drop_column("match_players", "role")
    op.drop_column("match_players", "lane")
    op.drop_column("match_players", "position")
