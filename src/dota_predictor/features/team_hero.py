"""Descriptive team × hero historical state (not a predictive feature yet).

Grain
-----
One row per current `(match_id, team_id, hero_id)`: the team on one of
its currently drafted heroes, with familiarity metrics computed from
strictly earlier matches. This is historical state keyed by the current
draft, not a static property of a team or a hero, and not a dense
team×hero grid.

A team counts as having played a hero when one of its five players in
that historical match has `hero_id = H`. Played side is never inferred
from draft order.

Temporal integrity
------------------
Every historical match `h` contributing to the row for current match `c`
satisfies `h.start_time < c.start_time` (see `features.temporal`). Equal
timestamps are not historical: matches that share a `start_time` are
mutually blind. The current match itself is never included.

SQL implements that strict-`<` rule with window frames
`RANGE ... CURRENT ROW EXCLUDE GROUP` (current row and same-timestamp
peers omitted) rather than a current×historical self-join. Team-match
counts and team×hero counts are windowed on collapsed grains so five
lobby slots never inflate a match into five historical games.

Contexts
--------
* All-time prior (any game version):
  `h.start_time < c.start_time`
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
historical state for the current match. The current row's `hero_id` is
only the lookup key (which hero this team drafted now); the current
match's own result is never mixed into that row's metrics.

This layer does not expose in-game events, duration, gold/xp, or the
current match result. It is queryable independently of
`PRE_DRAFT_SNAPSHOT_SQL` and is not part of the training feature matrix.

Win / loss / share semantics
----------------------------
Win / loss uses `match_players.hero_id` + `match_players.side` +
`matches.radiant_win`. Played side is never inferred from draft order.
`slot_in_side` is lobby order and is never treated as Dota position 1-5,
a lane, or a role; it is not used in any window partition and is not
projected.

History is keyed by `team_id` (and `hero_id` where the metric is
team×hero). Changing the five `player_id` values on a team does not
reset that history. Canonical `team_id` is used as-is; roster-aware
corrections are out of scope here.

Rates and shares are raw floating-point ratios. No smoothing, shrinkage,
or regularization. NULL when the denominator is zero (no historical
evidence in that context). A zero rate/share with a positive denominator
is observed 0%. `days_since_team_played_hero` is NULL when the team has
never previously played that hero.

This module never writes Parquet, never bumps schema versions, and never
adds columns to the fact/reference files.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    HEROES_VIEW,
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)

__all__ = [
    "MATCH_ID_COLUMN",
    "RECENT_WINDOW_DAYS",
    "TEAM_HERO_COLUMNS",
    "TEAM_HERO_IDENTITY_COLUMNS",
    "TEAM_HERO_METRIC_COLUMNS",
    "TeamHeroState",
    "build_team_hero",
    "team_hero_sql",
]


MATCH_ID_COLUMN = "match_id"
TEAM_ID_COLUMN = "team_id"
HERO_ID_COLUMN = "hero_id"
HERO_NAME_COLUMN = "hero_name"

RECENT_WINDOW_DAYS = 90

TEAM_HERO_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    TEAM_ID_COLUMN,
    HERO_ID_COLUMN,
    HERO_NAME_COLUMN,
    "side",
)

TEAM_HERO_METRIC_COLUMNS: tuple[str, ...] = (
    "team_prior_games_with_hero",
    "team_prior_wins_with_hero",
    "team_prior_losses_with_hero",
    "team_prior_win_rate_with_hero",
    "same_version_team_games_with_hero",
    "same_version_team_wins_with_hero",
    "same_version_team_win_rate_with_hero",
    "recent_90d_team_games_with_hero",
    "recent_90d_team_wins_with_hero",
    "recent_90d_team_win_rate_with_hero",
    "team_prior_games",
    "team_hero_share",
    "recent_90d_team_games",
    "recent_90d_team_hero_share",
    "days_since_team_played_hero",
)

TEAM_HERO_COLUMNS: tuple[str, ...] = (
    TEAM_HERO_IDENTITY_COLUMNS + TEAM_HERO_METRIC_COLUMNS
)

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


def _rate_sql(numerator: str, denominator: str) -> str:
    """Raw floating-point ratio, NULL when the denominator is zero."""
    return (
        f"CASE WHEN {denominator} > 0 "
        f"THEN {numerator}::DOUBLE / {denominator} "
        f"ELSE NULL END"
    )


def _hero_name_select(*, catalog_registered: bool) -> str:
    if catalog_registered:
        return f"{HEROES_VIEW}.name AS hero_name"
    return "CAST(NULL AS VARCHAR) AS hero_name"


def _hero_name_join(*, catalog_registered: bool) -> str:
    if catalog_registered:
        return f"LEFT JOIN {HEROES_VIEW} ON {HEROES_VIEW}.hero_id = h.hero_id"
    return ""


def team_hero_sql(
    *, catalog_registered: bool = True, match_id: int | None = None
) -> str:
    """SQL for leakage-safe team×hero state at `(match_id, team_id, hero_id)`.

    `match_players` is first collapsed to one row per current drafted
    `(match, team, hero)` and one row per `(match, team)`, then
    aggregations are DuckDB window functions over those grains -- not a
    current×historical self-join. `RANGE ... EXCLUDE GROUP` implements
    `historical.start_time < current.start_time`, including same-timestamp
    blindness.

    `catalog_registered=True` left-joins the `heroes` reference view for
    display names. Optional `match_id` filters the output after the
    windows run over all matches (windows need the full ordered history).
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE h.match_id = {int(match_id)}"

    win_rate = _rate_sql(
        "h.team_prior_wins_with_hero", "h.team_prior_games_with_hero"
    )
    win_rate_sv = _rate_sql(
        "h.same_version_team_wins_with_hero", "h.same_version_team_games_with_hero"
    )
    win_rate_r90 = _rate_sql(
        "h.recent_90d_team_wins_with_hero", "h.recent_90d_team_games_with_hero"
    )
    hero_share = _rate_sql("h.team_prior_games_with_hero", "g.team_prior_games")
    hero_share_r90 = _rate_sql(
        "h.recent_90d_team_games_with_hero", "g.recent_90d_team_games"
    )
    hero_name = _hero_name_select(catalog_registered=catalog_registered)
    hero_join = _hero_name_join(catalog_registered=catalog_registered)

    return f"""
WITH appearances AS (
    SELECT
        mp.match_id,
        mp.start_time,
        m.game_version_id,
        mp.team_id,
        mp.hero_id,
        mp.side,
        CASE
            WHEN mp.side = 'RADIANT' THEN m.radiant_win
            ELSE NOT m.radiant_win
        END AS team_won
    FROM {MATCH_PLAYERS_VIEW} mp
    JOIN {MATCHES_VIEW} m ON m.match_id = mp.match_id
),

team_heroes AS (
    SELECT
        match_id,
        start_time,
        game_version_id,
        team_id,
        hero_id,
        ANY_VALUE(side) AS side,
        BOOL_OR(team_won) AS team_won
    FROM appearances
    GROUP BY match_id, start_time, game_version_id, team_id, hero_id
),

team_matches AS (
    SELECT
        match_id,
        start_time,
        team_id
    FROM appearances
    GROUP BY match_id, start_time, team_id
),

flagged_heroes AS (
    SELECT
        match_id,
        start_time,
        game_version_id,
        team_id,
        hero_id,
        side,
        CASE WHEN team_won THEN 1 ELSE 0 END AS was_win,
        CASE WHEN team_won = FALSE THEN 1 ELSE 0 END AS was_loss
    FROM team_heroes
),

hero_windowed AS (
    SELECT
        match_id,
        start_time,
        game_version_id,
        team_id,
        hero_id,
        side,
        COALESCE(COUNT(*) OVER w_th, 0)::BIGINT AS team_prior_games_with_hero,
        COALESCE(SUM(was_win) OVER w_th, 0)::BIGINT AS team_prior_wins_with_hero,
        COALESCE(SUM(was_loss) OVER w_th, 0)::BIGINT AS team_prior_losses_with_hero,
        COALESCE(COUNT(*) OVER w_sv, 0)::BIGINT AS same_version_team_games_with_hero,
        COALESCE(SUM(was_win) OVER w_sv, 0)::BIGINT AS same_version_team_wins_with_hero,
        COALESCE(COUNT(*) OVER w_90_th, 0)::BIGINT AS recent_90d_team_games_with_hero,
        COALESCE(SUM(was_win) OVER w_90_th, 0)::BIGINT
            AS recent_90d_team_wins_with_hero,
        MAX(start_time) OVER w_th AS last_played_hero_at
    FROM flagged_heroes
    WINDOW
        w_th AS (
            PARTITION BY team_id, hero_id
            ORDER BY start_time
            {_STRICT_PRIOR_RANGE}
        ),
        w_sv AS (
            PARTITION BY team_id, hero_id, game_version_id
            ORDER BY start_time
            {_STRICT_PRIOR_RANGE}
        ),
        w_90_th AS (
            PARTITION BY team_id, hero_id
            ORDER BY start_time
            {_RECENT_90D_RANGE}
        )
),

match_windowed AS (
    SELECT
        match_id,
        team_id,
        COALESCE(COUNT(*) OVER w_team, 0)::BIGINT AS team_prior_games,
        COALESCE(COUNT(*) OVER w_90_team, 0)::BIGINT AS recent_90d_team_games
    FROM team_matches
    WINDOW
        w_team AS (
            PARTITION BY team_id
            ORDER BY start_time
            {_STRICT_PRIOR_RANGE}
        ),
        w_90_team AS (
            PARTITION BY team_id
            ORDER BY start_time
            {_RECENT_90D_RANGE}
        )
)

SELECT
    h.match_id,
    h.start_time,
    h.game_version_id,
    h.team_id,
    h.hero_id,
    {hero_name},
    h.side,
    h.team_prior_games_with_hero,
    h.team_prior_wins_with_hero,
    h.team_prior_losses_with_hero,
    {win_rate} AS team_prior_win_rate_with_hero,
    h.same_version_team_games_with_hero,
    h.same_version_team_wins_with_hero,
    {win_rate_sv} AS same_version_team_win_rate_with_hero,
    h.recent_90d_team_games_with_hero,
    h.recent_90d_team_wins_with_hero,
    {win_rate_r90} AS recent_90d_team_win_rate_with_hero,
    g.team_prior_games,
    {hero_share} AS team_hero_share,
    g.recent_90d_team_games,
    {hero_share_r90} AS recent_90d_team_hero_share,
    CASE
        WHEN h.last_played_hero_at IS NULL THEN NULL
        ELSE date_diff(
            'microsecond',
            h.last_played_hero_at,
            h.start_time
        )::DOUBLE / 86400000000.0
    END AS days_since_team_played_hero
FROM hero_windowed AS h
JOIN match_windowed AS g
    ON g.match_id = h.match_id AND g.team_id = h.team_id
{hero_join}
{output_filter}
"""


def _heroes_view_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return HEROES_VIEW in tables


@dataclass(frozen=True)
class TeamHeroState:
    """Lazy `(match_id, team_id, hero_id)` team×hero relation.

    Nothing is materialized until `to_frame` is called. The owning
    `FeatureDuckDBConnection` must stay open for that call.
    """

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per current team×hero in `TEAM_HERO_COLUMNS` order."""
        frame = self.relation.df()
        ordered = frame[list(TEAM_HERO_COLUMNS)]
        return ordered.sort_values(
            [MATCH_ID_COLUMN, TEAM_ID_COLUMN, HERO_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)


def build_team_hero(
    store: FeatureDuckDBConnection, *, match_id: int | None = None
) -> TeamHeroState:
    """Build descriptive team×hero state from the registered analytical views.

    Independent of `build_pre_draft_snapshot`. Uses the `heroes` catalog
    for display names when that view is registered. Optional `match_id`
    filters output rows after windows run over the full ordered match
    history.
    """
    sql = team_hero_sql(
        catalog_registered=_heroes_view_registered(store),
        match_id=match_id,
    )
    return TeamHeroState(relation=store.sql(sql))
