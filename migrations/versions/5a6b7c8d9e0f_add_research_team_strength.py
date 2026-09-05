"""add research team-strength derived state and ranking (Slice 6)

Revision ID: 5a6b7c8d9e0f
Revises: 3c4d5e6f7a8b
Create Date: 2026-09-05 00:00:00.000000

Adds the Slice 6 team-strength foundation on top of the canonical
`matches` facts:

* `research.team_strength_state` -- a *persisted deterministic derived
  table* at grain `(team_id, match_id)`: each team's Elo rating entering
  that match (`elo_pre`), the post-match bookkeeping rating (`elo_post`,
  only available after the result), and a strictly-prior descriptive
  record (`prior_match_count` / `prior_win_count` / `prior_loss_count` /
  `prior_win_rate`, `previous_match_id` / `previous_match_at`,
  `days_since_previous_match`). Elo is a sequential recurrence that a
  plain PostgreSQL view cannot express, so the state is persisted and
  idempotently rebuilt (one transaction) from the canonical facts by
  `dota_predictor.data.team_strength.rebuild_team_strength_state`, which
  reuses the production `features.team_elo` definition. `matches` remains
  the sole source of truth; there is no mutable "current Elo" column on
  `teams`.
* `research.team_strength_build` -- a single-row provenance / staleness
  marker recording the source corpus snapshot the derived table was built
  from (`source_match_count`, `source_min_start_time`,
  `source_max_start_time`, `source_fingerprint`, `rows_written`, Elo
  config). `source_fingerprint` is a deterministic SHA-256 over the
  canonical match fields that determine the derived state, so a correction
  to an old result/team/time is detected even when count and corpus
  extrema are unchanged.
* `research.raw_team_elo_latest` -- an ordinary SQL view over the derived
  table: the latest post-match Elo per canonical/source `team_id` (latest
  temporal group), joined to display/identity metadata and the optional
  curated organization. The Elo recurrence is never reimplemented in SQL;
  the terminal rating is derived as `elo_pre + SUM(elo_post - elo_pre)`
  over the latest group. It exposes NO ordinal `rank` and NO global
  ordering: it is a latest raw Elo STATE per source `team_id`, NOT a
  ranking. It is keyed by raw canonical `team_id` (one competitive team may
  appear under multiple `team_id`s), historical or disbanded teams remain
  rated, there is no active-team eligibility rule, and the Elo universe is
  the full canonical corpus (including large amounts of Tier 3 data).
  Canonical `team_id`s are never merged by organization.

FROZEN SNAPSHOT: Alembic migrations are historical snapshots. The DDL
below is the exact definition for this revision and must not be edited to
track later changes. It intentionally does NOT import from
`dota_predictor.research.views` or `dota_predictor.storage.schema`, so a
future change to those modules can never alter this migration's behavior.
Later team-strength changes get their own migration.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5a6b7c8d9e0f"
down_revision: str | Sequence[str] | None = "3c4d5e6f7a8b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# --- FROZEN DDL for this revision (historical snapshot, see docstring). ---

_TEAM_STRENGTH_STATE_TABLE_SQL = """
CREATE TABLE research.team_strength_state (
    match_id BIGINT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    team_id BIGINT NOT NULL,
    side TEXT NOT NULL,
    team_name_observed TEXT,
    elo_pre DOUBLE PRECISION NOT NULL,
    elo_post DOUBLE PRECISION NOT NULL,
    won BOOLEAN NOT NULL,
    prior_match_count INTEGER NOT NULL,
    prior_win_count INTEGER NOT NULL,
    prior_loss_count INTEGER NOT NULL,
    prior_win_rate DOUBLE PRECISION,
    previous_match_id BIGINT,
    previous_match_at TIMESTAMPTZ,
    days_since_previous_match DOUBLE PRECISION,
    is_first_observed_match BOOLEAN NOT NULL,
    PRIMARY KEY (team_id, match_id),
    CONSTRAINT ck_elo_pre_non_negative CHECK (elo_pre >= 0),
    CONSTRAINT ck_elo_post_non_negative CHECK (elo_post >= 0),
    CONSTRAINT ck_prior_match_count_non_negative CHECK (prior_match_count >= 0),
    CONSTRAINT ck_prior_win_count_non_negative CHECK (prior_win_count >= 0),
    CONSTRAINT ck_prior_loss_count_non_negative CHECK (prior_loss_count >= 0)
)
"""

_TEAM_STRENGTH_BUILD_TABLE_SQL = """
CREATE TABLE research.team_strength_build (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    built_at TIMESTAMPTZ NOT NULL,
    source_match_count BIGINT NOT NULL,
    source_skipped_matches BIGINT NOT NULL,
    source_min_start_time TIMESTAMPTZ,
    source_max_start_time TIMESTAMPTZ,
    source_fingerprint TEXT NOT NULL,
    rows_written BIGINT NOT NULL,
    elo_initial_rating DOUBLE PRECISION NOT NULL,
    elo_k_factor DOUBLE PRECISION NOT NULL
)
"""

_RAW_TEAM_ELO_LATEST_VIEW_SQL = """
CREATE OR REPLACE VIEW research.raw_team_elo_latest AS
WITH team_summary AS (
    SELECT
        team_id,
        start_time AS last_match_at,
        match_id AS last_match_id,
        elo_pre
            + SUM(elo_post - elo_pre) OVER (PARTITION BY team_id, start_time)
            AS rating,
        ROW_NUMBER() OVER (
            PARTITION BY team_id ORDER BY start_time DESC, match_id DESC
        ) AS rn,
        COUNT(*) OVER (PARTITION BY team_id) AS observed_match_count,
        SUM(CASE WHEN won THEN 1 ELSE 0 END) OVER (PARTITION BY team_id) AS wins,
        SUM(CASE WHEN won THEN 0 ELSE 1 END) OVER (PARTITION BY team_id) AS losses,
        FIRST_VALUE(team_name_observed) OVER (
            PARTITION BY team_id ORDER BY start_time DESC, match_id DESC
        ) AS team_name
    FROM research.team_strength_state
)
SELECT
    s.team_id,
    s.team_name,
    s.rating,
    s.last_match_id,
    s.last_match_at,
    s.observed_match_count,
    s.wins,
    s.losses,
    o.organization_id,
    o.name AS organization_name,
    rro.lineup_player_ids AS latest_lineup_player_ids,
    rro.lineup_key AS latest_lineup_key,
    (SELECT max(start_time) FROM research.team_strength_state) AS as_of_at,
    CASE WHEN (SELECT max(start_time) FROM research.team_strength_state) IS NOT NULL
         THEN EXTRACT(EPOCH FROM (
                (SELECT max(start_time) FROM research.team_strength_state)
                - s.last_match_at
              )) / 86400.0
         ELSE NULL END AS days_since_last_match_as_of_corpus_end
FROM team_summary s
LEFT JOIN public.team_organization_memberships tom
    ON tom.team_id = s.team_id
LEFT JOIN public.organizations o
    ON o.organization_id = tom.organization_id
LEFT JOIN LATERAL (
    SELECT lineage.lineup_player_ids, lineage.lineup_key
    FROM research.team_match_lineups lineage
    WHERE lineage.team_id = s.team_id
      AND lineage.start_time = s.last_match_at
    ORDER BY lineage.match_id DESC
    LIMIT 1
) rro ON TRUE
WHERE s.rn = 1
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
    op.execute(_TEAM_STRENGTH_STATE_TABLE_SQL)
    op.execute(_TEAM_STRENGTH_BUILD_TABLE_SQL)
    op.execute(_RAW_TEAM_ELO_LATEST_VIEW_SQL)
    op.execute(_GRANTS_SQL)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP VIEW IF EXISTS research.raw_team_elo_latest")
    op.execute("DROP TABLE IF EXISTS research.team_strength_state")
    op.execute("DROP TABLE IF EXISTS research.team_strength_build")