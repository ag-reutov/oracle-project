"""add research roster-history views

Revision ID: a7b8c9d0e1f2
Revises: 2d9c1e4a8b0f
Create Date: 2026-09-04 21:00:00.000000

Adds the Slice 4 observed-roster-history views on top of the canonical
match facts:

* `research.team_match_lineups` -- one row per (match_id, team_id): the
  players observed for that team in that match (sorted
  `lineup_player_ids`, deterministic `lineup_key`) with an explicit
  cardinality audit (`n_players`, `n_resolved_players`,
  `n_null_player_ids`, `n_distinct_players`, `has_duplicate_players`,
  `has_fewer_than_five`, `has_more_than_five`, `has_exactly_five`,
  `is_complete_five`, `team_is_match_team`). Malformed lineups are
  reported, never forced into a five-player shape.
* `research.player_team_spells` -- one row per (player_id, spell_index):
  a player's maximal run of matches observed for one team, in
  chronological order. A new spell begins only when the observed
  `team_id` changes; a later return to a previous team is a new spell
  (A -> B -> A is three spells); a time gap with no intervening team
  observation does NOT split a spell. `first_seen_at` / `last_seen_at`
  are observed match times only -- never invented joined/left dates.
  This view mirrors `dota_predictor.data.roster_history.derive_observed_spells`.

`research.player_matches` already is the canonical roster-appearance
relation, so no duplicate `roster_appearances` view is created. These two
views are the only new objects; there is no new storage (both are plain
views over `matches` / `match_players`).

FROZEN SNAPSHOT: Alembic migrations are historical snapshots. The DDL
below is the exact view definition for this revision and must not be
edited to track later changes. It intentionally does NOT import from
`dota_predictor.research.views` (the current runtime/test definition), so
a future change to that module can never alter this migration's behavior.
Later roster-history changes get their own migration.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: str | Sequence[str] | None = "2d9c1e4a8b0f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# --- FROZEN view DDL for this revision (historical snapshot, see docstring). ---

_TEAM_MATCH_LINEUPS_VIEW_SQL = """
CREATE OR REPLACE VIEW research.team_match_lineups AS
WITH observations AS (
    SELECT mp.match_id,
           m.start_time,
           CASE WHEN mp.side = 'RADIANT' THEN m.radiant_team_id ELSE m.dire_team_id END
               AS team_id,
           mp.player_id
    FROM public.match_players mp
    JOIN public.matches m USING (match_id)
)
SELECT match_id,
       start_time,
       team_id,
       count(*) AS n_players,
       count(player_id) AS n_resolved_players,
       count(*) FILTER (WHERE player_id IS NULL) AS n_null_player_ids,
       count(DISTINCT player_id) AS n_distinct_players,
       count(DISTINCT player_id) < count(player_id) AS has_duplicate_players,
       count(player_id) < 5 AS has_fewer_than_five,
       count(player_id) > 5 AS has_more_than_five,
       count(player_id) = 5 AS has_exactly_five,
       (count(player_id) = 5 AND count(DISTINCT player_id) = 5
            AND count(*) FILTER (WHERE player_id IS NULL) = 0)
           AS is_complete_five,
       array_agg(player_id ORDER BY player_id)
           FILTER (WHERE player_id IS NOT NULL) AS lineup_player_ids,
       string_agg(player_id::text, ',' ORDER BY player_id)
           FILTER (WHERE player_id IS NOT NULL) AS lineup_key,
       TRUE AS team_is_match_team
FROM observations
GROUP BY match_id, start_time, team_id
ORDER BY match_id, team_id
"""

_PLAYER_TEAM_SPELLS_VIEW_SQL = """
CREATE OR REPLACE VIEW research.player_team_spells AS
WITH observations AS (
    SELECT mp.player_id,
           CASE WHEN mp.side = 'RADIANT' THEN m.radiant_team_id ELSE m.dire_team_id END
               AS team_id,
           m.match_id,
           m.start_time
    FROM public.match_players mp
    JOIN public.matches m USING (match_id)
    WHERE mp.player_id IS NOT NULL
      AND m.radiant_team_id IS NOT NULL
      AND m.dire_team_id IS NOT NULL
),
ranked AS (
    SELECT player_id, team_id, match_id, start_time,
           row_number() OVER (
               PARTITION BY player_id
               ORDER BY start_time, match_id, team_id
           ) AS rn
    FROM observations
),
spell_marks AS (
    SELECT player_id, team_id, match_id, start_time, rn,
           CASE WHEN LAG(team_id) OVER (PARTITION BY player_id ORDER BY rn)
                IS DISTINCT FROM team_id
                THEN 1 ELSE 0 END AS new_spell
    FROM ranked
),
spell_ids AS (
    SELECT player_id, team_id, match_id, start_time,
           sum(new_spell) OVER (PARTITION BY player_id ORDER BY rn) AS spell_index
    FROM spell_marks
),
spell_rows AS (
    SELECT player_id, team_id, spell_index, match_id, start_time,
           count(*) OVER (PARTITION BY player_id, spell_index) AS observed_match_count,
           first_value(match_id) OVER (
               PARTITION BY player_id, spell_index
               ORDER BY start_time, match_id, team_id
           ) AS first_match_id,
           last_value(match_id) OVER (
               PARTITION BY player_id, spell_index
               ORDER BY start_time, match_id, team_id
               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
           ) AS last_match_id,
           first_value(start_time) OVER (
               PARTITION BY player_id, spell_index
               ORDER BY start_time, match_id, team_id
           ) AS first_seen_at,
           last_value(start_time) OVER (
               PARTITION BY player_id, spell_index
               ORDER BY start_time, match_id, team_id
               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
           ) AS last_seen_at
    FROM spell_ids
)
SELECT player_id,
       team_id,
       spell_index,
       min(observed_match_count) AS observed_match_count,
       min(first_seen_at) AS first_seen_at,
       min(last_seen_at) AS last_seen_at,
       min(first_match_id) AS first_match_id,
       min(last_match_id) AS last_match_id
FROM spell_rows
GROUP BY player_id, team_id, spell_index
ORDER BY player_id, spell_index
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
    op.execute(_TEAM_MATCH_LINEUPS_VIEW_SQL)
    op.execute(_PLAYER_TEAM_SPELLS_VIEW_SQL)
    op.execute(_GRANTS_SQL)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP VIEW IF EXISTS research.player_team_spells")
    op.execute("DROP VIEW IF EXISTS research.team_match_lineups")