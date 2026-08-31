"""Descriptive hero-meta historical state (not a predictive feature yet).

Grain
-----
One row per `(match_id, hero_id)`: the observed professional hero meta
immediately BEFORE that match, for that hero. This is historical state,
not a static property of a hero.

Temporal integrity
------------------
Every historical match `h` contributing to the row for current match `c`
satisfies `h.start_time < c.start_time` (see `features.temporal`). Equal
timestamps are not historical: matches that share a `start_time` are
mutually blind. The current match itself is never included.

SQL implements that strict-`<` rule with window frames
`RANGE ... CURRENT ROW EXCLUDE GROUP` (current row and same-timestamp
peers omitted) rather than a current×historical self-join.

Two contexts (v1 -- no other windows, no patch-family grouping)
---------------------------------------------------------------
* Same STRATZ game version:
  `h.game_version_id = c.game_version_id AND h.start_time < c.start_time`
* Recent 90 days (may cross game-version boundaries):
  `h.start_time < c.start_time
   AND h.start_time >= c.start_time - INTERVAL 90 DAY`

Availability
------------
Source `hero_id` observations are DRAFT (and `radiant_win` is POST_MATCH)
*relative to the historical match that produced them*. Aggregating those
facts over matches strictly earlier than the current match yields
PRE_DRAFT historical state for the current match:

    historical hero draft/result facts
            ↓
    aggregated using only past matches
            ↓
    PRE_DRAFT historical state for current match

This layer does not expose the current match's own hero selections,
draft events, or result. It is queryable independently of
`PRE_DRAFT_SNAPSHOT_SQL` and is not part of the training feature matrix.

Draft / win semantics
---------------------
Pick / ban / contest counts use canonical `draft_events` at **match
grain** (a hero contributes at most one pick, one ban, and one contest
per historical match). Only successful canonical draft actions count:
picks always; bans unless `was_successful is False` -- the same rule as
`DraftEvent.is_actual`. Unsuccessful ban attempts are neither bans nor
contests.

Win / loss uses `match_players.hero_id` + `match_players.side` +
`matches.radiant_win`. Played side is never inferred from draft order.
`prior_wins` / `prior_losses` / `win_rate` only include historical
matches where the hero was actually picked; `win_rate` denominator is
`prior_picks`. Zero prior picks → NULL win rate (never 0 or 0.5).

Rates are raw floating-point ratios. No smoothing, shrinkage, or
regularization. A zero rate with `prior_matches > 0` is observed 0%
(historical evidence). A NULL rate means the denominator is zero (no
historical evidence in that context).

Hero universe
-------------
When the `heroes` reference view is registered, every catalog hero is
represented for every current match -- not only heroes in the current
draft. Otherwise the universe is the distinct `hero_id` values observed
on `draft_events` and `match_players`. The catalog size is never
hard-coded.

This module never writes Parquet, never bumps schema versions, and never
adds columns to the fact/reference files.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    DRAFT_EVENTS_VIEW,
    HEROES_VIEW,
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)

__all__ = [
    "HERO_META_COLUMNS",
    "HERO_META_IDENTITY_COLUMNS",
    "HERO_META_METRIC_COLUMNS",
    "MATCH_ID_COLUMN",
    "RECENT_WINDOW_DAYS",
    "HeroMetaState",
    "build_hero_meta",
    "hero_meta_sql",
    "rank_hero_meta",
]


MATCH_ID_COLUMN = "match_id"
HERO_ID_COLUMN = "hero_id"
HERO_NAME_COLUMN = "hero_name"

RECENT_WINDOW_DAYS = 90

HERO_META_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    HERO_ID_COLUMN,
    HERO_NAME_COLUMN,
)

_SAME_VERSION_METRICS: tuple[str, ...] = (
    "same_version_prior_matches",
    "same_version_prior_picks",
    "same_version_prior_bans",
    "same_version_prior_contests",
    "same_version_pick_rate",
    "same_version_ban_rate",
    "same_version_contest_rate",
    "same_version_prior_wins",
    "same_version_prior_losses",
    "same_version_win_rate",
)

_RECENT_90D_METRICS: tuple[str, ...] = (
    "recent_90d_prior_matches",
    "recent_90d_prior_picks",
    "recent_90d_prior_bans",
    "recent_90d_prior_contests",
    "recent_90d_pick_rate",
    "recent_90d_ban_rate",
    "recent_90d_contest_rate",
    "recent_90d_prior_wins",
    "recent_90d_prior_losses",
    "recent_90d_win_rate",
)

HERO_META_METRIC_COLUMNS: tuple[str, ...] = _SAME_VERSION_METRICS + _RECENT_90D_METRICS

HERO_META_COLUMNS: tuple[str, ...] = HERO_META_IDENTITY_COLUMNS + HERO_META_METRIC_COLUMNS

# Window equivalent of `historical.start_time < current.start_time`:
# RANGE through CURRENT ROW, then EXCLUDE GROUP so the current row and
# every peer with the same start_time are omitted.
_STRICT_PRIOR_RANGE = (
    "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP"
)
_RECENT_90D_RANGE = (
    f"RANGE BETWEEN INTERVAL {RECENT_WINDOW_DAYS} DAY PRECEDING "
    "AND CURRENT ROW EXCLUDE GROUP"
)


def _hero_universe_sql(*, catalog_registered: bool) -> str:
    if catalog_registered:
        return f"SELECT hero_id, name FROM {HEROES_VIEW}"
    return f"""
        SELECT hero_id, CAST(NULL AS VARCHAR) AS name
        FROM (
            SELECT DISTINCT hero_id FROM {DRAFT_EVENTS_VIEW}
            UNION
            SELECT DISTINCT hero_id FROM {MATCH_PLAYERS_VIEW}
        ) observed
        WHERE hero_id IS NOT NULL
    """


def _rate_sql(numerator: str, denominator: str) -> str:
    """Raw floating-point ratio, NULL when the denominator is zero."""
    return (
        f"CASE WHEN {denominator} > 0 "
        f"THEN {numerator}::DOUBLE / {denominator} "
        f"ELSE NULL END"
    )


def hero_meta_sql(
    *, catalog_registered: bool = True, match_id: int | None = None
) -> str:
    """SQL for leakage-safe hero-meta state at `(match_id, hero_id)`.

    Aggregations are DuckDB window functions over a dense
    `(match, hero)` grid -- not a current×historical self-join -- so the
    full current dataset stays practical. `RANGE ... EXCLUDE GROUP`
    implements `historical.start_time < current.start_time`, including
    same-timestamp blindness.

    `catalog_registered=True` uses the `heroes` reference view as the
    universe. Optional `match_id` filters the output after the windows
    run over all matches (windows need the full ordered history).
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE match_id = {int(match_id)}"

    universe = _hero_universe_sql(catalog_registered=catalog_registered)

    pick_rate_sv = _rate_sql("same_version_prior_picks", "same_version_prior_matches")
    ban_rate_sv = _rate_sql("same_version_prior_bans", "same_version_prior_matches")
    contest_rate_sv = _rate_sql(
        "same_version_prior_contests", "same_version_prior_matches"
    )
    win_rate_sv = _rate_sql("same_version_prior_wins", "same_version_prior_picks")
    pick_rate_r90 = _rate_sql("recent_90d_prior_picks", "recent_90d_prior_matches")
    ban_rate_r90 = _rate_sql("recent_90d_prior_bans", "recent_90d_prior_matches")
    contest_rate_r90 = _rate_sql(
        "recent_90d_prior_contests", "recent_90d_prior_matches"
    )
    win_rate_r90 = _rate_sql("recent_90d_prior_wins", "recent_90d_prior_picks")

    return f"""
WITH hero_universe AS (
    {universe}
),

-- Successful canonical draft actions only (`DraftEvent.is_actual`):
-- picks always count; bans count unless was_successful is False.
successful_draft_actions AS (
    SELECT
        de.match_id,
        de.hero_id,
        de.action
    FROM {DRAFT_EVENTS_VIEW} de
    WHERE de.action = 'PICK'
       OR (de.action = 'BAN' AND de.was_successful IS DISTINCT FROM FALSE)
),

-- Match grain: at most one pick, one ban, one contest per (match, hero).
hero_match_actions AS (
    SELECT
        match_id,
        hero_id,
        MAX(CASE WHEN action = 'PICK' THEN 1 ELSE 0 END)::BIGINT AS was_picked,
        MAX(CASE WHEN action = 'BAN' THEN 1 ELSE 0 END)::BIGINT AS was_banned
    FROM successful_draft_actions
    GROUP BY match_id, hero_id
),

-- Played-side win/loss from match_players, never from draft order.
hero_match_results AS (
    SELECT
        mp.match_id,
        mp.hero_id,
        CASE
            WHEN mp.side = 'RADIANT' THEN m.radiant_win
            ELSE NOT m.radiant_win
        END AS hero_won
    FROM {MATCH_PLAYERS_VIEW} mp
    JOIN {MATCHES_VIEW} m ON m.match_id = mp.match_id
),

hero_match_facts AS (
    SELECT
        a.match_id,
        a.hero_id,
        a.was_picked,
        a.was_banned,
        CASE WHEN a.was_picked = 1 OR a.was_banned = 1 THEN 1 ELSE 0 END
            AS was_contested,
        CASE WHEN a.was_picked = 1 AND r.hero_won THEN 1 ELSE 0 END AS was_win,
        CASE WHEN a.was_picked = 1 AND r.hero_won = FALSE THEN 1 ELSE 0 END
            AS was_loss
    FROM hero_match_actions a
    LEFT JOIN hero_match_results r
        ON r.match_id = a.match_id AND r.hero_id = a.hero_id
),

-- Dense (match, hero) grid. Flags are 0 when the hero was not
-- successfully picked/banned in that match. Window COUNT over this
-- grid equals the number of prior matches in the context.
grid AS (
    SELECT
        m.match_id,
        m.start_time,
        m.game_version_id,
        u.hero_id,
        u.name AS hero_name,
        COALESCE(f.was_picked, 0) AS was_picked,
        COALESCE(f.was_banned, 0) AS was_banned,
        COALESCE(f.was_contested, 0) AS was_contested,
        COALESCE(f.was_win, 0) AS was_win,
        COALESCE(f.was_loss, 0) AS was_loss
    FROM {MATCHES_VIEW} m
    CROSS JOIN hero_universe u
    LEFT JOIN hero_match_facts f
        ON f.match_id = m.match_id AND f.hero_id = u.hero_id
),

windowed AS (
    SELECT
        match_id,
        hero_id,
        hero_name,
        COALESCE(SUM(was_picked) OVER w_sv, 0)::BIGINT AS same_version_prior_picks,
        COALESCE(SUM(was_banned) OVER w_sv, 0)::BIGINT AS same_version_prior_bans,
        COALESCE(SUM(was_contested) OVER w_sv, 0)::BIGINT
            AS same_version_prior_contests,
        COALESCE(SUM(was_win) OVER w_sv, 0)::BIGINT AS same_version_prior_wins,
        COALESCE(SUM(was_loss) OVER w_sv, 0)::BIGINT AS same_version_prior_losses,
        COUNT(*) OVER w_sv AS same_version_prior_matches,
        COALESCE(SUM(was_picked) OVER w_90, 0)::BIGINT AS recent_90d_prior_picks,
        COALESCE(SUM(was_banned) OVER w_90, 0)::BIGINT AS recent_90d_prior_bans,
        COALESCE(SUM(was_contested) OVER w_90, 0)::BIGINT
            AS recent_90d_prior_contests,
        COALESCE(SUM(was_win) OVER w_90, 0)::BIGINT AS recent_90d_prior_wins,
        COALESCE(SUM(was_loss) OVER w_90, 0)::BIGINT AS recent_90d_prior_losses,
        COUNT(*) OVER w_90 AS recent_90d_prior_matches
    FROM grid
    WINDOW
        w_sv AS (
            PARTITION BY game_version_id, hero_id
            ORDER BY start_time
            {_STRICT_PRIOR_RANGE}
        ),
        w_90 AS (
            PARTITION BY hero_id
            ORDER BY start_time
            {_RECENT_90D_RANGE}
        )
)

SELECT
    match_id,
    hero_id,
    hero_name,
    same_version_prior_matches,
    same_version_prior_picks,
    same_version_prior_bans,
    same_version_prior_contests,
    {pick_rate_sv} AS same_version_pick_rate,
    {ban_rate_sv} AS same_version_ban_rate,
    {contest_rate_sv} AS same_version_contest_rate,
    same_version_prior_wins,
    same_version_prior_losses,
    {win_rate_sv} AS same_version_win_rate,
    recent_90d_prior_matches,
    recent_90d_prior_picks,
    recent_90d_prior_bans,
    recent_90d_prior_contests,
    {pick_rate_r90} AS recent_90d_pick_rate,
    {ban_rate_r90} AS recent_90d_ban_rate,
    {contest_rate_r90} AS recent_90d_contest_rate,
    recent_90d_prior_wins,
    recent_90d_prior_losses,
    {win_rate_r90} AS recent_90d_win_rate
FROM windowed
{output_filter}
"""


def _heroes_view_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return HEROES_VIEW in tables


@dataclass(frozen=True)
class HeroMetaState:
    """Lazy `(match_id, hero_id)` hero-meta relation over an open store.

    Nothing is materialized until `to_frame` is called. The owning
    `FeatureDuckDBConnection` must stay open for that call.
    """

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per `(match_id, hero_id)` in `HERO_META_COLUMNS` order."""
        frame = self.relation.df()
        return frame[list(HERO_META_COLUMNS)]


def build_hero_meta(
    store: FeatureDuckDBConnection, *, match_id: int | None = None
) -> HeroMetaState:
    """Build descriptive hero-meta state from the registered analytical views.

    Independent of `build_pre_draft_snapshot`. Uses the `heroes` catalog
    as the universe when that view is registered; otherwise distinct
    observed `hero_id` values. Optional `match_id` filters output rows
    after windows run over the full ordered match history.
    """
    sql = hero_meta_sql(
        catalog_registered=_heroes_view_registered(store),
        match_id=match_id,
    )
    return HeroMetaState(relation=store.sql(sql))


def rank_hero_meta(state: pd.DataFrame) -> pd.DataFrame:
    """Copy of `state` sorted by same-version contest rate, then support.

    Primary key is `same_version_contest_rate` descending (NULL rates
    sort last). Ties break on `same_version_prior_matches` descending
    so tiny samples stay visible next to high rates, then `hero_id`
    ascending. Does not mutate `state`.
    """
    required = (
        "same_version_contest_rate",
        "same_version_prior_matches",
        HERO_ID_COLUMN,
    )
    missing = [column for column in required if column not in state.columns]
    if missing:
        raise ValueError(f"state frame is missing required columns: {missing}")
    return state.sort_values(
        [
            "same_version_contest_rate",
            "same_version_prior_matches",
            HERO_ID_COLUMN,
        ],
        ascending=[False, False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
