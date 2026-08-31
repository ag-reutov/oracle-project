"""add match_players.hero_id

Revision ID: e8f3a1c7b204
Revises: c4e8f1a2b709
Create Date: 2026-08-31 14:20:00.000000

Adds the played hero for each canonical match-player row, sourced from
already-persisted STRATZ `players[].heroId` (not draft pick order and not
`pickBans.playerIndex`). Existing rows are backfilled from
`stratz_raw_matches.payload` before the column is made NOT NULL.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e8f3a1c7b204"
down_revision: str | Sequence[str] | None = "c4e8f1a2b709"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("match_players", sa.Column("hero_id", sa.Integer(), nullable=True))
    op.execute(
        """
        UPDATE match_players AS mp
        SET hero_id = src.hero_id
        FROM (
            SELECT
                r.match_id,
                (player ->> 'steamAccountId')::bigint AS player_id,
                (player ->> 'heroId')::integer AS hero_id
            FROM stratz_raw_matches AS r,
                 jsonb_array_elements(r.payload -> 'players') AS player
            WHERE jsonb_typeof(r.payload -> 'players') = 'array'
        ) AS src
        WHERE mp.match_id = src.match_id
          AND mp.player_id = src.player_id
        """
    )
    op.alter_column("match_players", "hero_id", existing_type=sa.Integer(), nullable=False)
    op.create_check_constraint(
        op.f("ck_match_players_hero_id_positive"),
        "match_players",
        "hero_id > 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_match_players_hero_id_positive"),
        "match_players",
        type_="check",
    )
    op.drop_column("match_players", "hero_id")
