"""add research.players player-universe view

Revision ID: e533fb8010a7
Revises: 96ee182840e3
Create Date: 2026-09-04 19:30:00.000000

Adds the Slice 2 canonical player-universe view `research.players` on top
of the existing `players` identity registry:

    player_id | display_name | first_seen_at | last_seen_at | match_count

* `player_id` is the canonical STRATZ `steamAccountId` (the `players`
  registry primary key) -- one row per canonical player, never duplicated.
* `display_name` is the deterministic best-known name; the local corpus
  contains no player-name observations (raw payloads carry only
  `steamAccountId`), so it is NULL for every row. The deterministic
  resolution rule (most recently observed valid name) lives in
  `dota_predictor.data.player_identity` for when names become available.
* `first_seen_at` / `last_seen_at` / `match_count` are pure aggregates
  over the immutable canonical facts (`match_players` joined to
  `matches.start_time`), computed fresh every time the view is read.

The view intentionally exposes NO mutable competitive state (no current
team, no position, no rating, no hero pool), so future-state leakage into
historical research is structurally impossible. Orphan registry ids (in
`players` but not referenced by any match) are excluded; they are reported
by `scripts/audit_player_identity.py` instead.

FROZEN SNAPSHOT: Alembic migrations are historical snapshots. The DDL
below is the exact view definition for this revision and must not be
edited to track later changes. It intentionally does NOT import from
`dota_predictor.data.player_identity` (the current runtime/test
definition), so a future change to that module can never alter this
migration's behavior. Later player-identity changes get their own
migration.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e533fb8010a7"
down_revision: str | Sequence[str] | None = "96ee182840e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# --- FROZEN view DDL for this revision (historical snapshot, see docstring). ---

_PLAYERS_VIEW_SQL = """
CREATE OR REPLACE VIEW research.players AS
SELECT
    p.player_id,
    NULL::text AS display_name,
    MIN(m.start_time) AS first_seen_at,
    MAX(m.start_time) AS last_seen_at,
    COUNT(*) AS match_count
FROM public.players p
JOIN public.match_players mp USING (player_id)
JOIN public.matches m USING (match_id)
GROUP BY p.player_id
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
    op.execute(_PLAYERS_VIEW_SQL)
    op.execute(_GRANTS_SQL)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP VIEW IF EXISTS research.players")
