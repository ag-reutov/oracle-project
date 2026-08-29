"""add teams and players identity registries

Revision ID: 37102448abc6
Revises: fc8432b8f4cf
Create Date: 2026-08-29 11:54:53.525543

Introduces `teams`/`players` as minimal identity registries (primary key
only -- see `storage.schema` module docstring for why no derived/cached
columns live here), and renames `matches.radiant_team_name` /
`matches.dire_team_name` to `*_team_name_observed` to make explicit that
these are immutable per-match observations, not entity-level display
names.

Order matters:
1. Rename the `matches` name columns first, so the backfill below reads
   from their final names.
2. Create `teams`/`players`.
3. Backfill them with every distinct id already referenced by
   `matches`/`match_players`, so the foreign keys added in step 4 do not
   fail against existing data.
4. Add the foreign keys.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '37102448abc6'
down_revision: str | Sequence[str] | None = 'fc8432b8f4cf'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('matches', 'radiant_team_name', new_column_name='radiant_team_name_observed')
    op.alter_column('matches', 'dire_team_name', new_column_name='dire_team_name_observed')

    op.create_table(
        'teams',
        sa.Column('team_id', sa.BigInteger(), autoincrement=False, nullable=False),
        sa.PrimaryKeyConstraint('team_id', name=op.f('pk_teams')),
    )
    op.create_table(
        'players',
        sa.Column('player_id', sa.BigInteger(), autoincrement=False, nullable=False),
        sa.PrimaryKeyConstraint('player_id', name=op.f('pk_players')),
    )

    # Backfill: every distinct team_id/player_id already referenced by
    # existing canonical data. No aggregation, no name resolution -- these
    # are identity registries only (see module docstring).
    op.execute(
        """
        INSERT INTO teams (team_id)
        SELECT DISTINCT team_id FROM (
            SELECT radiant_team_id AS team_id FROM matches
            UNION
            SELECT dire_team_id AS team_id FROM matches
        ) t
        """
    )
    op.execute(
        """
        INSERT INTO players (player_id)
        SELECT DISTINCT player_id FROM match_players
        """
    )

    op.create_foreign_key(
        op.f('fk_matches_radiant_team_id_teams'),
        'matches', 'teams',
        ['radiant_team_id'], ['team_id'],
    )
    op.create_foreign_key(
        op.f('fk_matches_dire_team_id_teams'),
        'matches', 'teams',
        ['dire_team_id'], ['team_id'],
    )
    op.create_foreign_key(
        op.f('fk_match_players_player_id_players'),
        'match_players', 'players',
        ['player_id'], ['player_id'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f('fk_match_players_player_id_players'), 'match_players', type_='foreignkey')
    op.drop_constraint(op.f('fk_matches_dire_team_id_teams'), 'matches', type_='foreignkey')
    op.drop_constraint(op.f('fk_matches_radiant_team_id_teams'), 'matches', type_='foreignkey')

    op.drop_table('players')
    op.drop_table('teams')

    op.alter_column('matches', 'dire_team_name_observed', new_column_name='dire_team_name')
    op.alter_column('matches', 'radiant_team_name_observed', new_column_name='radiant_team_name')
