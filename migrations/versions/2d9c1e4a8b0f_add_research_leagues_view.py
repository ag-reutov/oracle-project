"""add research.leagues canonical league/event view

Revision ID: 2d9c1e4a8b0f
Revises: e533fb8010a7
Create Date: 2026-09-04 20:00:00.000000

Adds the Slice 3 canonical league/event identity view `research.leagues`
over the curated `leagues` registry:

    league_id | league_name | stratz_tier | liquipedia_tier
    | in_scope | fetch_mode | curation_source | start_date | end_date
    | window_filter | curated_at

* `league_id` is the stable STRATZ league id (the `leagues` registry
  primary key).
* `league_name` is the curated canonical league name from `config/leagues.yaml`.
* `stratz_tier` is the raw STRATZ `LeagueTier` value (source identity,
  cross-check signal only) and `liquipedia_tier` is our curated Liquipedia
  classification. The two are deliberately kept distinct and never
  conflated -- our T1/T2/T3 classification is a curated property, not an
  intrinsic STRATZ property.
* `curation_source` / `curated_at` record where the curation decision came
  from and when it was made (provenance).

The view is a plain read-only projection over the existing `public.leagues`
table; it duplicates no storage and feeds back into nothing.

FROZEN SNAPSHOT: Alembic migrations are historical snapshots. The DDL
below is the exact view definition for this revision and must not be
edited to track later changes. It intentionally does NOT import from
`dota_predictor.research.views` (the current runtime/test definition), so
a future change to that module can never alter this migration's behavior.
Later reference-entity changes get their own migration.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2d9c1e4a8b0f"
down_revision: str | Sequence[str] | None = "e533fb8010a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# --- FROZEN view DDL for this revision (historical snapshot, see docstring). ---

_LEAGUES_VIEW_SQL = """
CREATE OR REPLACE VIEW research.leagues AS
SELECT
    league_id,
    name AS league_name,
    stratz_tier AS stratz_tier,
    liquipedia_tier AS liquipedia_tier,
    in_scope,
    fetch_mode,
    source AS curation_source,
    start_date,
    end_date,
    window_filter,
    curated_at
FROM public.leagues
"""

_GRANTS_SQL = """
DO $research_grants$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'metabase_reader')
    THEN
        GRANT USAGE ON SCHEMA research TO metabase_reader;
        GRANT SELECT ON ALL TABLES IN SCHEMA research TO metabase_reader;
        ALTER DEFAULT PRIVILEGES FOR ROLE dota_predictor IN SCHEMA research
            GRANT SELECT ON TABLES TO metabase_reader;
    END IF;
END
$research_grants$;
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE SCHEMA IF NOT EXISTS research")
    op.execute(_LEAGUES_VIEW_SQL)
    op.execute(_GRANTS_SQL)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP VIEW IF EXISTS research.leagues")
