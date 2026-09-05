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
    "RAW_TEAM_ELO_LATEST_VIEW_SQL",
    "RESEARCH_SCHEMA",
    "RESEARCH_VIEW_NAMES",
    "RESEARCH_VIEW_SQL",
    "TEAM_STRENGTH_BUILD_TABLE_SQL",
    "TEAM_STRENGTH_STATE_TABLE_SQL",
    "create_research_layer",
    "drop_research_layer",
]

RESEARCH_SCHEMA = "research"

# Order matters: `research.matches` is a dependency of the player/draft views,
# which are dependencies of the population views. The Slice 4 roster views
# (`team_match_lineups`, `player_team_spells`) depend only on the public
# canonical tables, so they are appended after the population views. The Slice 5
# roster-state views depend on the Slice 4 views: `team_roster_state` references
# `team_match_lineups` (the current lineup identity) and `player_team_state`
# (the player classifications), so they come last. The Slice 6
# `raw_team_elo_latest` view depends on the derived `team_strength_state`
# table (created before the views by `create_research_layer` / the Slice 6
# migration), so it comes after.
RESEARCH_VIEW_NAMES: tuple[str, ...] = (
    "leagues",
    "matches",
    "player_matches",
    "players",
    "draft_events",
    "t12_matches",
    "pro_matches",
    "t12_draft_matches",
    "team_match_lineups",
    "player_team_spells",
    "player_team_state",
    "team_roster_state",
    "raw_team_elo_latest",
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

# --- Slice 4: observed roster history -----------------------------------------
# `research.player_matches` already is the canonical roster-appearance
# relation (match_id, start_time, player_id, team_id, side), so no
# duplicate `roster_appearances` view is created. These two thin views
# answer the remaining Slice 4 questions without storing anything.

# One row per (match_id, team_id): the players observed for that team in
# that match, with an explicit cardinality audit. `team_id` is derived from
# the parent match's radiant/dire teams by side, so `team_is_match_team` is
# structurally always TRUE (the invariant is exposed so it stays checkable).
# `lineup_player_ids` (sorted canonical player ids) and `lineup_key` (the
# same ids as a deterministic comma-joined string) are the deterministic
# lineup identity derived from the sorted canonical ids. Malformed lineups
# are flagged (has_fewer_than_five / has_more_than_five /
# has_duplicate_players / null ids), never forced into a five-player shape.
TEAM_MATCH_LINEUPS_VIEW_SQL = """
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

# One row per (player_id, spell_index): a player's maximal run of matches
# observed for one team, in chronological order. Spell semantics:
#   * order observations by (start_time, match_id, team_id);
#   * a new spell begins only when the observed team_id changes;
#   * a later return to a previous team is a NEW spell (A -> B -> A is
#     three spells);
#   * a gap in time with no intervening team observation does NOT split a
#     spell;
#   * first/last seen are observed match times -- never invented
#     joined/left dates.
# This mirrors `dota_predictor.data.roster_history.derive_observed_spells`.
PLAYER_TEAM_SPELLS_VIEW_SQL = """
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

# --- Slice 5: historical roster state --------------------------------------------
# Strictly causal pre-match roster state at grain (team_id, match_id) and
# (player_id, team_id, match_id). Every prior/previous/first/last value uses
# only observations with `start_time` strictly before the current match's
# `start_time` (equal timestamps never become prior evidence). The current
# match's own lineup is reused from `research.team_match_lineups` (the Slice 4
# lineup identity) and never contributes to its own prior counts.
# These views mirror `dota_predictor.data.roster_state.derive_player_team_state`
# / `derive_team_roster_state`.

# One row per (player_id, team_id, match_id): a player's observed relationship
# to a team immediately before that match. `prior_team_match_count` counts
# strictly earlier matches of the same player for the same team; the previous
# observed match is the player's most recent strictly earlier match for ANY
# team (tie-broken by match_id DESC only among strictly prior rows). The three
# flags are mutually exclusive observational classifications (first observed /
# returning / continuing); no transfer/stand-in semantics are implied.
# `consecutive_prior_team_appearances` is spell-so-far -- the number of
# immediately preceding observations for the same team -- never the eventual
# spell length (which would leak future information).
PLAYER_TEAM_STATE_VIEW_SQL = """
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
-- Spell-so-far bookkeeping (causal run length only; never the eventual
-- spell end). Mirrors research.player_team_spells ordering.
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
    -- Prior same-team match count and first/last prior same-team times,
    -- using strict start_time < only.
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
    -- Most recent strictly earlier observation for the player (any team).
    -- match_id DESC is a tie-breaker among rows already filtered to
    -- ho.start_time < o.start_time.
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

# One row per (match_id, team_id): the team's historical roster state
# immediately before that match. The current lineup is reused verbatim from
# research.team_match_lineups (Slice 4). `previous_match_id` is the team's most
# recent strictly earlier observed match; `players_retained_from_previous_match`
# / `players_changed_from_previous_match` / `same_lineup_as_previous_match` are
# only defined when both the current and previous lineups are complete fives.
# `prior_exact_lineup_match_count` counts strictly earlier complete-five
# matches of the same team with the identical lineup_key (only defined for a
# complete-five current lineup). Team-composition counts are over the resolved
# lineup players; for a complete five they reconcile to 5.
TEAM_ROSTER_STATE_VIEW_SQL = """
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
    -- Strictly earlier same-team matches with the identical lineup_key,
    -- complete fives only.
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

# --- Slice 6: team strength & ranking ---------------------------------------
# `research.team_strength_state` is a *persisted deterministic derived table*,
# not a view: Elo is a sequential recurrence that a plain PostgreSQL view
# cannot express (see `dota_predictor.data.team_strength` for the exact
# derivation, which reuses the production `features.team_elo` definition).
# The canonical `matches` facts remain the sole source of truth; the table is
# idempotently rebuilt in one transaction by
# `scripts/rebuild_team_strength.py`, and `research.team_strength_build` is a
# single-row provenance/staleness marker (source corpus snapshot +
# deterministic SHA-256 source fingerprint). `research.raw_team_elo_latest` is
# an ordinary view over the derived table (latest post-match Elo per raw
# team id), so the latest-Elo state is never separately persisted.

TEAM_STRENGTH_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS research.team_strength_state (
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

TEAM_STRENGTH_BUILD_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS research.team_strength_build (
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

# One row per canonical/source team id: its terminal post-match Elo (the
# rating after its final observed temporal group) with display/identity
# metadata. The terminal rating is derived from the persisted per-match
# `elo_pre`/`elo_post` as `elo_pre + SUM(elo_post - elo_pre)` over the team's
# latest `start_time` group -- the Elo recurrence itself is never
# reimplemented in SQL. This is a latest raw Elo STATE per source `team_id`,
# NOT a ranking: it deliberately exposes no ordinal `rank` and no global
# ordering. The view is keyed by raw canonical/source `team_id` (a
# competitive team may appear under multiple `team_id`s), historical or
# disbanded teams remain rated, there is no active-team eligibility rule, and
# the Elo universe is the full canonical match corpus (including large
# amounts of Tier 3 data). Separate `team_id`s are never merged, even when
# mapped to the same organization. Ordering is a query concern, not part of
# the entity; a plain `SELECT * FROM research.raw_team_elo_latest` must not
# imply a rank. Activity is exposed (last_match_at,
# days_since_last_match_as_of_corpus_end) rather than hidden behind a cutoff.
# `as_of_at` is the corpus maximum `start_time` -- never wall-clock "now".
RAW_TEAM_ELO_LATEST_VIEW_SQL = """
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
    "team_match_lineups": TEAM_MATCH_LINEUPS_VIEW_SQL,
    "player_team_spells": PLAYER_TEAM_SPELLS_VIEW_SQL,
    "player_team_state": PLAYER_TEAM_STATE_VIEW_SQL,
    "team_roster_state": TEAM_ROSTER_STATE_VIEW_SQL,
    "raw_team_elo_latest": RAW_TEAM_ELO_LATEST_VIEW_SQL,
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
    already exists (each view uses `CREATE OR REPLACE`; the Slice 6
    derived tables use `CREATE TABLE IF NOT EXISTS`). Used by the Alembic
    migration and by the test suite. The Slice 6 `team_strength_state` /
    `team_strength_build` tables are created here as empty shells so the
    `raw_team_elo_latest` view can reference them; their rows are populated
    only by `dota_predictor.data.team_strength.rebuild_team_strength_state`.
    """
    with bind.begin() as conn:
        conn.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {RESEARCH_SCHEMA}"))
        conn.execute(sa.text(TEAM_STRENGTH_STATE_TABLE_SQL))
        conn.execute(sa.text(TEAM_STRENGTH_BUILD_TABLE_SQL))
        for name in RESEARCH_VIEW_NAMES:
            conn.execute(sa.text(RESEARCH_VIEW_SQL[name]))
        conn.execute(sa.text(GRANTS_SQL))


def drop_research_layer(bind: Engine) -> None:
    """Drop the `research` schema and everything in it (test teardown)."""
    with bind.begin() as conn:
        conn.execute(sa.text(f"DROP SCHEMA IF EXISTS {RESEARCH_SCHEMA} CASCADE"))
