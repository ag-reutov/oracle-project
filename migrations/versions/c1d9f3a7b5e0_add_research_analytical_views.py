"""add research analytical views

Revision ID: c1d9f3a7b5e0
Revises: f0c2d4e6a8b0
Create Date: 2026-09-04 12:00:00.000000

Creates the read-only analytical `research` schema and its views
(`research.matches`, `research.player_matches`, `research.draft_events`, and
the population views `research.t12_matches`, `research.pro_matches`,
`research.t12_draft_matches`), plus the read-only grants for the documented
`metabase_reader` role.

FROZEN SNAPSHOT: Alembic migrations are historical snapshots. The SQL below
is the exact research-schema/view/grant DDL for this revision and must not
be edited to track later changes. It intentionally does NOT import from
`dota_predictor.research.views` (the current runtime/test definition), so a
future change to `views.py` can never alter this migration's behavior.
Later research-layer changes get their own migration.

Effective tier/event is `coalesce(match_classifications.*, leagues.*)`: the
match-level override wins over the league default, so shared/multi-tier
STRATZ leagues (ACL 2025 in league 17875, 1win Spring/Summer/Fall/Punch in
league 16427) resolve correctly. Population membership is centralized in the
`is_*_main_event` booleans.

The `metabase_reader` grants are applied only when the role exists (the role
is created separately by docker/postgres/init/02-create-metabase-reader.sh).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c1d9f3a7b5e0"
down_revision: str | Sequence[str] | None = "f0c2d4e6a8b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# --- FROZEN view DDL for this revision (historical snapshot, see docstring). ---

_MATCHES_VIEW_SQL = """
CREATE OR REPLACE VIEW research.matches AS
SELECT
    m.match_id,
    m.start_time,
    (m.start_time AT TIME ZONE 'UTC')::date AS match_date,
    EXTRACT(YEAR FROM (m.start_time AT TIME ZONE 'UTC'))::integer AS year,
    EXTRACT(MONTH FROM (m.start_time AT TIME ZONE 'UTC'))::integer AS month,
    m.league_id,
    l.name AS league_name,
    l.liquipedia_tier AS default_tier,
    m.league_name AS stratz_league_name,
    COALESCE(mc.liquipedia_event, l.name) AS effective_event,
    COALESCE(mc.liquipedia_tier, l.liquipedia_tier) AS effective_tier,
    CASE
        WHEN mc.match_id IS NOT NULL THEN 'match-level override'
        ELSE 'league default'
    END AS classification_source,
    m.radiant_team_id,
    m.radiant_team_name_observed AS radiant_team_name,
    m.dire_team_id,
    m.dire_team_name_observed AS dire_team_name,
    m.radiant_win,
    CASE WHEN m.radiant_win THEN 'RADIANT' ELSE 'DIRE' END AS winning_side,
    m.duration_seconds,
    m.game_version_id,
    m.draft_complete,
    m.series_id,
    m.series_type,
    m.game_number_in_series,
    COALESCE(mc.liquipedia_tier, l.liquipedia_tier) IN ('T1', 'T2', 'T3')
        AS is_main_event,
    COALESCE(mc.liquipedia_tier, l.liquipedia_tier) IN ('T1', 'T2')
        AND (m.start_time AT TIME ZONE 'UTC')::date >= DATE '2024-01-01'
        AS is_t12_main_event,
    COALESCE(mc.liquipedia_tier, l.liquipedia_tier) IN ('T1', 'T2', 'T3')
        AND (m.start_time AT TIME ZONE 'UTC')::date >= DATE '2024-01-01'
        AS is_t123_main_event
FROM public.matches m
JOIN public.leagues l USING (league_id)
LEFT JOIN public.match_classifications mc USING (match_id);
"""

_PLAYER_MATCHES_VIEW_SQL = """
CREATE OR REPLACE VIEW research.player_matches AS
SELECT
    mp.match_id,
    r.start_time,
    r.match_date,
    r.year,
    r.month,
    r.effective_event,
    r.effective_tier,
    r.league_id,
    r.game_version_id,
    r.draft_complete,
    mp.player_id,
    CASE WHEN mp.side = 'RADIANT' THEN r.radiant_team_id ELSE r.dire_team_id END
        AS team_id,
    CASE WHEN mp.side = 'RADIANT' THEN r.radiant_team_name ELSE r.dire_team_name END
        AS team_name,
    mp.side,
    mp.slot_in_side,
    mp.hero_id,
    mp.position,
    mp.lane,
    mp.role,
    (mp.side = 'RADIANT' AND r.radiant_win)
        OR (mp.side = 'DIRE' AND NOT r.radiant_win) AS player_win,
    mp.kills,
    mp.deaths,
    mp.assists,
    mp.gold_per_minute,
    mp.experience_per_minute,
    mp.num_last_hits,
    mp.num_denies,
    mp.networth,
    mp.hero_damage,
    mp.tower_damage,
    mp.hero_healing,
    mp.level
FROM public.match_players mp
JOIN research.matches r USING (match_id);
"""

_DRAFT_EVENTS_VIEW_SQL = """
CREATE OR REPLACE VIEW research.draft_events AS
SELECT
    de.match_id,
    r.start_time,
    r.match_date,
    r.year,
    r.month,
    r.effective_event,
    r.effective_tier,
    r.league_id,
    r.game_version_id,
    r.draft_complete,
    de.sequence,
    de.action,
    de.side,
    de.hero_id,
    de.was_successful
FROM public.draft_events de
JOIN research.matches r USING (match_id);
"""

_T12_MATCHES_VIEW_SQL = """
CREATE OR REPLACE VIEW research.t12_matches AS
SELECT * FROM research.matches WHERE is_t12_main_event;
"""

_PRO_MATCHES_VIEW_SQL = """
CREATE OR REPLACE VIEW research.pro_matches AS
SELECT * FROM research.matches WHERE is_t123_main_event;
"""

_T12_DRAFT_MATCHES_VIEW_SQL = """
CREATE OR REPLACE VIEW research.t12_draft_matches AS
SELECT * FROM research.t12_matches WHERE draft_complete;
"""

# Order matters: `research.matches` is a dependency of the player/draft views,
# which are dependencies of the population views.
_VIEW_SQL: tuple[str, ...] = (
    _MATCHES_VIEW_SQL,
    _PLAYER_MATCHES_VIEW_SQL,
    _DRAFT_EVENTS_VIEW_SQL,
    _T12_MATCHES_VIEW_SQL,
    _PRO_MATCHES_VIEW_SQL,
    _T12_DRAFT_MATCHES_VIEW_SQL,
)

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
    op.execute("CREATE SCHEMA IF NOT EXISTS research")
    for sql in _VIEW_SQL:
        op.execute(sql)
    op.execute(_GRANTS_SQL)


def downgrade() -> None:
    op.execute("DROP SCHEMA IF EXISTS research CASCADE")
