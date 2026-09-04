"""add team identity layer (aliases, tags, organizations)

Revision ID: 96ee182840e3
Revises: c1d9f3a7b5e0
Create Date: 2026-09-04 18:30:00.000000

Adds the explicit team-identity layer (Slice 1) on top of the existing
`teams` registry:

* `team_aliases` -- every name observed for a raw STRATZ `team_id` in
  canonical matches, with its observation period and count. Derived
  identity information; the source of truth remains the immutable
  `matches.*_team_name_observed` columns.
* `team_tags` -- every STRATZ `tag` observed for a `team_id` in existing
  raw payloads, with its observation period and count. Backfilled from
  local data only (no API calls); tags are treated as historical
  observations like names, so coverage is intentionally partial.
* `organizations` -- curated organization-level identity. `organization_id`
  is assigned explicitly in `config/team_organizations.yaml`, never
  derived from any team id.
* `team_organization_memberships` -- explicit raw `team_id` ->
  organization mapping with provenance (`reason`, `source`). One team id
  maps to at most one organization (PK); absence of a row means
  "unmapped", which is normal and valid. Name equality alone never
  merges teams.

This migration only creates the tables -- it performs NO backfill, so it
is safe and non-destructive on the existing database. The derived tables
are populated by the idempotent, deterministic helper scripts
`scripts/backfill_team_identity.py` and `scripts/load_team_organizations.py`
(see `storage.schema` module docstring and `data.team_identity`).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "96ee182840e3"
down_revision: str | Sequence[str] | None = "c1d9f3a7b5e0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "team_aliases",
        sa.Column("team_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "observation_count > 0",
            name=op.f("ck_team_aliases_observation_count_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["teams.team_id"],
            name=op.f("fk_team_aliases_team_id_teams"),
        ),
        sa.PrimaryKeyConstraint("team_id", "name", name=op.f("pk_team_aliases")),
    )
    op.create_table(
        "team_tags",
        sa.Column("team_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "observation_count > 0",
            name=op.f("ck_team_tags_observation_count_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["teams.team_id"],
            name=op.f("fk_team_tags_team_id_teams"),
        ),
        sa.PrimaryKeyConstraint("team_id", "tag", name=op.f("pk_team_tags")),
    )
    op.create_table(
        "organizations",
        sa.Column(
            "organization_id", sa.BigInteger(), autoincrement=False, nullable=False
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("organization_id", name=op.f("pk_organizations")),
        sa.UniqueConstraint("name", name=op.f("uq_organizations_name")),
    )
    op.create_table(
        "team_organization_memberships",
        sa.Column("team_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("organization_id", sa.BigInteger(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column(
            "added_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["teams.team_id"],
            name=op.f("fk_team_organization_memberships_team_id_teams"),
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.organization_id"],
            name=op.f("fk_team_organization_memberships_organization_id_organizations"),
        ),
        sa.PrimaryKeyConstraint(
            "team_id", name=op.f("pk_team_organization_memberships")
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("team_organization_memberships")
    op.drop_table("organizations")
    op.drop_table("team_tags")
    op.drop_table("team_aliases")