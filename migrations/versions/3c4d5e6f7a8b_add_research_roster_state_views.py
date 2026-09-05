"""add research roster-state views (Slice 5)

Revision ID: 3c4d5e6f7a8b
Revises: a7b8c9d0e1f2
Create Date: 2026-09-05 00:00:00.000000

Adds the Slice 5 strictly-causal historical roster-state views on top of
the Slice 4 roster views / canonical match facts:

* `research.player_team_state` -- one row per (player_id, team_id,
  match_id): a player's observed relationship to a team immediately
  before that match. `prior_team_match_count` and the first/last prior
  same-team times use only observations with `start_time` strictly
  before the current match; `previous_observed_*` is the player's most
  recent strictly earlier match for any team. The flags
  (`is_first_observed_match_for_team`, `is_returning_to_team`,
  `is_continuing_with_team`) are mutually exclusive observational
  classifications. `consecutive_prior_team_appearances` is spell-so-far
  (causal run length), never the eventual spell outcome.
* `research.team_roster_state` -- one row per (match_id, team_id): the
  team's historical roster state immediately before that match. The
  current lineup is reused from `research.team_match_lineups`; the
  previous observed team match is the team's most recent strictly
  earlier match. Retained/changed/same-lineup fields are only defined
  for complete fives; `prior_exact_lineup_match_count` counts strictly
  earlier same-team matches with the identical lineup_key.

Both views mirror `dota_predictor.data.roster_state.derive_player_team_state`
/ `derive_team_roster_state`. Strict `<` on `start_time` (never
`match_id` except as a presentation tie-breaker among already-strictly-
prior rows) guarantees equal timestamps never become causal precedent and
that no future observation contributes to an earlier state.

FROZEN SNAPSHOT: Alembic migrations are historical snapshots. The DDL
below is the exact view definition for this revision and must not be
edited to track later changes. It intentionally does NOT import from
`dota_predictor.research.views`, so a future change to that module can
never alter this migration's behavior. Later roster-state changes get
their own migration.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3c4d5e6f7a8b"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# --- FROZEN view DDL for this revision (historical snapshot, see docstring). ---

_PLAYER_TEAM_STATE_VIEW_SQL = """
CREATE OR REPLACE VIEW research.player_team_state AS
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
           ) AS rn_asc
    FROM observations
),
spell_marks AS (
    SELECT player_id, team_id, match_id, start_time, rn_asc,
           CASE WHEN LAG(team_id) OVER (PARTITION BY player_id ORDER BY rn_asc)
                IS DISTINCT FROM team_id
                THEN 1 ELSE 0 END AS new_spell
    FROM ranked
),
spell_ids AS (
    SELECT player_id, team_id, match_id, start_time, rn_asc,
           sum(new_spell) OVER (PARTITION BY player_id ORDER BY rn_asc) AS spell_index
    FROM spell_marks
),
spell_pos AS (
    SELECT player_id, team_id, match_id, start_time, spell_index,
           row_number() OVER (
               PARTITION BY player_id, spell_index ORDER BY rn_asc
           ) AS position_in_spell
    FROM spell_ids
),
prior_team AS (
    SELECT o.player_id, o.team_id, o.match_id, o.start_time,
           count(ho.match_id) AS prior_team_match_count,
           min(ho.start_time) AS first_prior_team_match_at,
           max(ho.start_time) AS last_prior_team_match_at
    FROM observations o
    LEFT JOIN observations ho
        ON ho.player_id = o.player_id
       AND ho.team_id = o.team_id
       AND ho.start_time < o.start_time
    GROUP BY o.player_id, o.team_id, o.match_id, o.start_time
),
prior_obs_ranked AS (
    SELECT o.player_id, o.team_id, o.match_id,
           ho.team_id AS previous_observed_team_id,
           ho.match_id AS previous_observed_match_id,
           ho.start_time AS previous_observed_match_at,
           row_number() OVER (
               PARTITION BY o.player_id, o.team_id, o.match_id
               ORDER BY ho.start_time DESC, ho.match_id DESC
           ) AS rn
    FROM observations o
    JOIN observations ho
        ON ho.player_id = o.player_id
       AND ho.start_time < o.start_time
),
prior_obs AS (
    SELECT * FROM prior_obs_ranked WHERE rn = 1
)
SELECT pt.player_id,
       pt.team_id,
       pt.match_id,
       pt.start_time,
       pt.prior_team_match_count,
       pt.first_prior_team_match_at,
       pt.last_prior_team_match_at,
       po.previous_observed_team_id,
       po.previous_observed_match_id,
       po.previous_observed_match_at,
       (pt.prior_team_match_count = 0) AS is_first_observed_match_for_team,
       (pt.prior_team_match_count > 0
            AND po.previous_observed_team_id IS NOT NULL
            AND po.previous_observed_team_id <> pt.team_id)
           AS is_returning_to_team,
       (po.previous_observed_team_id IS NOT NULL
            AND po.previous_observed_team_id = pt.team_id)
           AS is_continuing_with_team,
       sp.position_in_spell - 1 AS consecutive_prior_team_appearances,
       CASE WHEN po.previous_observed_match_at IS NOT NULL
            THEN EXTRACT(EPOCH FROM (pt.start_time - po.previous_observed_match_at))
                 / 86400.0
            ELSE NULL END AS days_since_player_previous_match,
       CASE WHEN pt.last_prior_team_match_at IS NOT NULL
            THEN EXTRACT(EPOCH FROM (pt.start_time - pt.last_prior_team_match_at))
                 / 86400.0
            ELSE NULL END AS days_since_player_previous_team_match
FROM prior_team pt
LEFT JOIN prior_obs po
    ON po.player_id = pt.player_id
   AND po.team_id = pt.team_id
   AND po.match_id = pt.match_id
LEFT JOIN spell_pos sp
    ON sp.player_id = pt.player_id
   AND sp.team_id = pt.team_id
   AND sp.match_id = pt.match_id
ORDER BY pt.player_id, pt.team_id, pt.match_id
"""

_TEAM_ROSTER_STATE_VIEW_SQL = """
CREATE OR REPLACE VIEW research.team_roster_state AS
WITH previous_ranked AS (
    SELECT cur.match_id,
           cur.team_id,
           hist.match_id AS previous_match_id,
           hist.start_time AS previous_match_at,
           hist.lineup_player_ids AS previous_lineup_player_ids,
           hist.lineup_key AS previous_lineup_key,
           hist.is_complete_five AS previous_is_complete_five,
           row_number() OVER (
               PARTITION BY cur.match_id, cur.team_id
               ORDER BY hist.start_time DESC, hist.match_id DESC
           ) AS rn
    FROM research.team_match_lineups cur
    JOIN research.team_match_lineups hist
        ON hist.team_id = cur.team_id
       AND hist.start_time < cur.start_time
),
previous AS (
    SELECT * FROM previous_ranked WHERE rn = 1
),
exact_ranked AS (
    SELECT cur.match_id,
           cur.team_id,
           hist.match_id AS exact_match_id,
           hist.start_time AS exact_match_at,
           row_number() OVER (
               PARTITION BY cur.match_id, cur.team_id
               ORDER BY hist.start_time DESC, hist.match_id DESC
           ) AS rn2
    FROM research.team_match_lineups cur
    JOIN research.team_match_lineups hist
        ON hist.team_id = cur.team_id
       AND hist.lineup_key = cur.lineup_key
       AND hist.start_time < cur.start_time
       AND cur.is_complete_five
       AND hist.is_complete_five
       AND cur.lineup_key IS NOT NULL
),
exact AS (
    SELECT match_id,
           team_id,
           count(*) AS exact_lineup_match_count,
           max(CASE WHEN rn2 = 1 THEN exact_match_id END) AS last_exact_lineup_match_id,
           max(CASE WHEN rn2 = 1 THEN exact_match_at END) AS last_exact_lineup_at
    FROM exact_ranked
    GROUP BY match_id, team_id
),
composition AS (
    SELECT match_id,
           team_id,
           count(*) FILTER (WHERE is_continuing_with_team) AS continuing_player_count,
           count(*) FILTER (WHERE is_first_observed_match_for_team) AS first_observed_for_team_count,
           count(*) FILTER (WHERE is_returning_to_team) AS returning_player_count
    FROM research.player_team_state
    GROUP BY match_id, team_id
)
SELECT lu.match_id,
       lu.start_time,
       lu.team_id,
       lu.lineup_player_ids,
       lu.lineup_key,
       lu.n_resolved_players,
       lu.n_distinct_players,
       lu.n_null_player_ids,
       lu.has_duplicate_players,
       lu.has_fewer_than_five,
       lu.has_more_than_five,
       lu.has_exactly_five,
       lu.is_complete_five,
       p.previous_match_id,
       p.previous_match_at,
       p.previous_lineup_player_ids,
       p.previous_lineup_key,
       p.previous_is_complete_five,
       CASE WHEN lu.is_complete_five AND p.previous_is_complete_five
            THEN (SELECT count(*) FROM unnest(lu.lineup_player_ids) AS x(pid)
                  WHERE x.pid = ANY(p.previous_lineup_player_ids))
            ELSE NULL END AS players_retained_from_previous_match,
       CASE WHEN lu.is_complete_five AND p.previous_is_complete_five
            THEN 5 - (SELECT count(*) FROM unnest(lu.lineup_player_ids) AS x(pid)
                      WHERE x.pid = ANY(p.previous_lineup_player_ids))
            ELSE NULL END AS players_changed_from_previous_match,
       CASE WHEN lu.is_complete_five AND p.previous_is_complete_five
            THEN lu.lineup_key = p.previous_lineup_key
            ELSE NULL END AS same_lineup_as_previous_match,
       CASE WHEN lu.is_complete_five
            THEN COALESCE(e.exact_lineup_match_count, 0)
            ELSE NULL END AS prior_exact_lineup_match_count,
       CASE WHEN lu.is_complete_five
            THEN e.last_exact_lineup_match_id
            ELSE NULL END AS last_exact_lineup_match_id,
       CASE WHEN lu.is_complete_five
            THEN e.last_exact_lineup_at
            ELSE NULL END AS last_exact_lineup_at,
       COALESCE(c.continuing_player_count, 0) AS continuing_player_count,
       COALESCE(c.first_observed_for_team_count, 0) AS first_observed_for_team_count,
       COALESCE(c.returning_player_count, 0) AS returning_player_count,
       CASE WHEN p.previous_match_at IS NOT NULL
            THEN EXTRACT(EPOCH FROM (lu.start_time - p.previous_match_at)) / 86400.0
            ELSE NULL END AS days_since_team_previous_match
FROM research.team_match_lineups lu
LEFT JOIN previous p USING (match_id, team_id)
LEFT JOIN exact e USING (match_id, team_id)
LEFT JOIN composition c USING (match_id, team_id)
ORDER BY lu.match_id, lu.team_id
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
    op.execute(_PLAYER_TEAM_STATE_VIEW_SQL)
    op.execute(_TEAM_ROSTER_STATE_VIEW_SQL)
    op.execute(_GRANTS_SQL)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP VIEW IF EXISTS research.team_roster_state")
    op.execute("DROP VIEW IF EXISTS research.player_team_state")