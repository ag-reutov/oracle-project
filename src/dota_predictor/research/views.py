"""Research-layer PostgreSQL view definitions.

A thin analytical layer over the canonical warehouse. Everything here is a
plain view over the existing `public` canonical tables (`matches`,
`match_players`, `draft_events`, `leagues`, `match_classifications`). No
duplicated storage, no materialized views, no refresh lifecycle.

The single classification derivation lives in `research.matches`:

    effective_tier  = coalesce(match_classifications.liquipedia_tier,
                               leagues.liquipedia_tier)
    effective_event = coalesce(match_classifications.liquipedia_event,
                               leagues.name)

`research.leagues` (Slice 3 reference entities) exposes the curated
league registry with the source-vs-curated tier distinction preserved:
`stratz_tier` is the raw STRATZ LeagueTier, `liquipedia_tier` is our
curated Liquipedia classification, and provenance (`source`, `curated_at`)
records where the curation came from.

Population membership is centralized in the `is_*_main_event` booleans on
`research.matches` and the population views (`research.t12_matches`,
`research.pro_matches`, `research.t12_draft_matches`).

The SQL here is the current research-layer definition, used by the test suite
(and available for ad-hoc reapplication). The Alembic migration that created
the schema is a FROZEN snapshot of these statements -- it intentionally does
not import from this module, so changing this module never rewrites history.
New view changes get their own migration.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import Engine

from dota_predictor.data.player_identity import PLAYER_UNIVERSE_VIEW_SQL

__all__ = [
    "GRANTS_SQL",
    "RESEARCH_SCHEMA",
    "RESEARCH_VIEW_NAMES",
    "RESEARCH_VIEW_SQL",
    "create_research_layer",
    "drop_research_layer",
]

RESEARCH_SCHEMA = "research"

# Order matters: `research.matches` is a dependency of the player/draft views,
# which are dependencies of the population views.
RESEARCH_VIEW_NAMES: tuple[str, ...] = (
    "leagues",
    "matches",
    "player_matches",
    "players",
    "draft_events",
    "t12_matches",
    "pro_matches",
    "t12_draft_matches",
)

# Canonical league/event identity (Slice 3 reference entities). Exposes the
# curated `leagues` registry with the source-vs-curated tier distinction
# preserved: `stratz_tier` is the raw STRATZ LeagueTier (cross-check signal
# only), `liquipedia_tier` is our curated Liquipedia classification, and the
# two are never conflated. Provenance (`source`, `curated_at`) records where
# the curation decision came from.
LEAGUES_VIEW_SQL = """
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

MATCHES_VIEW_SQL = """
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

PLAYER_MATCHES_VIEW_SQL = """
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

DRAFT_EVENTS_VIEW_SQL = """
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

T12_MATCHES_VIEW_SQL = """
CREATE OR REPLACE VIEW research.t12_matches AS
SELECT * FROM research.matches WHERE is_t12_main_event;
"""

PRO_MATCHES_VIEW_SQL = """
CREATE OR REPLACE VIEW research.pro_matches AS
SELECT * FROM research.matches WHERE is_t123_main_event;
"""

T12_DRAFT_MATCHES_VIEW_SQL = """
CREATE OR REPLACE VIEW research.t12_draft_matches AS
SELECT * FROM research.t12_matches WHERE draft_complete;
"""

RESEARCH_VIEW_SQL: dict[str, str] = {
    "leagues": LEAGUES_VIEW_SQL,
    "matches": MATCHES_VIEW_SQL,
    "player_matches": PLAYER_MATCHES_VIEW_SQL,
    # Player universe (Slice 2 player-identity foundation). Defined in
    # `dota_predictor.data.player_identity` so the Python helper
    # (`fetch_player_universe`) and the SQL view share one canonical
    # definition. The Alembic migration that applied this view is a frozen
    # copy (see its docstring).
    "players": PLAYER_UNIVERSE_VIEW_SQL,
    "draft_events": DRAFT_EVENTS_VIEW_SQL,
    "t12_matches": T12_MATCHES_VIEW_SQL,
    "pro_matches": PRO_MATCHES_VIEW_SQL,
    "t12_draft_matches": T12_DRAFT_MATCHES_VIEW_SQL,
}

# Read-only grants for the Metabase reader role, applied only when the role
# exists (it is created separately by docker/postgres/init/02-create-
# metabase-reader.sh). `metabase_reader` is the documented read-only role.
GRANTS_SQL = """
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


def create_research_layer(bind: Engine) -> None:
    """Create the `research` schema, all views, and read-only grants.

    Idempotent: safe to call against a database where the research layer
    already exists (each view uses `CREATE OR REPLACE`). Used by the Alembic
    migration and by the test suite.
    """
    with bind.begin() as conn:
        conn.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {RESEARCH_SCHEMA}"))
        for name in RESEARCH_VIEW_NAMES:
            conn.execute(sa.text(RESEARCH_VIEW_SQL[name]))
        conn.execute(sa.text(GRANTS_SQL))


def drop_research_layer(bind: Engine) -> None:
    """Drop the `research` schema and everything in it (test teardown)."""
    with bind.begin() as conn:
        conn.execute(sa.text(f"DROP SCHEMA IF EXISTS {RESEARCH_SCHEMA} CASCADE"))
