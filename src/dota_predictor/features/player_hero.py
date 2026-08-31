"""Descriptive player × hero historical state (not a predictive feature yet).

Grain
-----
One row per current `(match_id, player_id, hero_id)`: the player on their
currently drafted hero, with familiarity metrics computed from strictly
earlier matches. This is historical state keyed by the current draft, not
a static property of a player or a hero, and not a dense player×hero grid.

Temporal integrity
------------------
Every historical match `h` contributing to the row for current match `c`
satisfies `h.start_time < c.start_time` (see `features.temporal`). Equal
timestamps are not historical: matches that share a `start_time` are
mutually blind. The current match itself is never included.

SQL implements that strict-`<` rule with window frames
`RANGE ... CURRENT ROW EXCLUDE GROUP` (current row and same-timestamp
peers omitted) rather than a current×historical self-join.

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
only the lookup key (which hero this player drafted now); the current
match's own result is never mixed into that row's metrics.

This layer does not expose in-game events, duration, gold/xp, or the
current match result. It is queryable independently of
`PRE_DRAFT_SNAPSHOT_SQL` and is not part of the training feature matrix.

Win / loss / share semantics
----------------------------
Win / loss uses `match_players.hero_id` + `match_players.side` +
`matches.radiant_win`. Played side is never inferred from draft order.
`slot_in_side` is lobby order on the current row and is never treated as
Dota position 1-5, a lane, or a role; it is not used in any window
partition.

History is keyed by `player_id` (and `hero_id` where the metric is
player×hero). Changing `team_id` does not reset that history.

Rates and shares are raw floating-point ratios. No smoothing, shrinkage,
or regularization. NULL when the denominator is zero (no historical
evidence in that context). A zero rate/share with a positive denominator
is observed 0%. `days_since_last_played_hero` is NULL when the player has
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
    "PLAYER_HERO_COLUMNS",
    "PLAYER_HERO_IDENTITY_COLUMNS",
    "PLAYER_HERO_METRIC_COLUMNS",
    "RECENT_WINDOW_DAYS",
    "PlayerHeroState",
    "build_player_hero",
    "player_hero_sql",
]


MATCH_ID_COLUMN = "match_id"
PLAYER_ID_COLUMN = "player_id"
HERO_ID_COLUMN = "hero_id"
HERO_NAME_COLUMN = "hero_name"

RECENT_WINDOW_DAYS = 90

PLAYER_HERO_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    PLAYER_ID_COLUMN,
    HERO_ID_COLUMN,
    HERO_NAME_COLUMN,
    "side",
    "team_id",
    "slot_in_side",
)

PLAYER_HERO_METRIC_COLUMNS: tuple[str, ...] = (
    "prior_games_on_hero",
    "prior_wins_on_hero",
    "prior_losses_on_hero",
    "prior_win_rate_on_hero",
    "same_version_games_on_hero",
    "same_version_wins_on_hero",
    "same_version_win_rate_on_hero",
    "recent_90d_games_on_hero",
    "recent_90d_wins_on_hero",
    "recent_90d_win_rate_on_hero",
    "prior_player_games",
    "prior_hero_share",
    "recent_90d_player_games",
    "recent_90d_hero_share",
    "days_since_last_played_hero",
)

PLAYER_HERO_COLUMNS: tuple[str, ...] = (
    PLAYER_HERO_IDENTITY_COLUMNS + PLAYER_HERO_METRIC_COLUMNS
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
        return f"LEFT JOIN {HEROES_VIEW} ON {HEROES_VIEW}.hero_id = w.hero_id"
    return ""


def player_hero_sql(
    *, catalog_registered: bool = True, match_id: int | None = None
) -> str:
    """SQL for leakage-safe player×hero state at `(match_id, player_id, hero_id)`.

    Aggregations are DuckDB window functions over `match_players` rows --
    not a current×historical self-join -- so the full current dataset
    stays practical. `RANGE ... EXCLUDE GROUP` implements
    `historical.start_time < current.start_time`, including same-timestamp
    blindness.

    `catalog_registered=True` left-joins the `heroes` reference view for
    display names. Optional `match_id` filters the output after the
    windows run over all matches (windows need the full ordered history).
    `slot_in_side` is projected as lobby-slot identity only and is never
    a window partition or a position/lane/role proxy.
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE match_id = {int(match_id)}"

    win_rate = _rate_sql("w.prior_wins_on_hero", "w.prior_games_on_hero")
    win_rate_sv = _rate_sql(
        "w.same_version_wins_on_hero", "w.same_version_games_on_hero"
    )
    win_rate_r90 = _rate_sql("w.recent_90d_wins_on_hero", "w.recent_90d_games_on_hero")
    hero_share = _rate_sql("w.prior_games_on_hero", "w.prior_player_games")
    hero_share_r90 = _rate_sql("w.recent_90d_games_on_hero", "w.recent_90d_player_games")
    hero_name = _hero_name_select(catalog_registered=catalog_registered)
    hero_join = _hero_name_join(catalog_registered=catalog_registered)

    return f"""
WITH appearances AS (
    SELECT
        mp.match_id,
        mp.start_time,
        m.game_version_id,
        mp.player_id,
        mp.hero_id,
        mp.side,
        mp.team_id,
        mp.slot_in_side,
        CASE
            WHEN mp.side = 'RADIANT' THEN m.radiant_win
            ELSE NOT m.radiant_win
        END AS player_won
    FROM {MATCH_PLAYERS_VIEW} mp
    JOIN {MATCHES_VIEW} m ON m.match_id = mp.match_id
),

flagged AS (
    SELECT
        match_id,
        start_time,
        game_version_id,
        player_id,
        hero_id,
        side,
        team_id,
        slot_in_side,
        CASE WHEN player_won THEN 1 ELSE 0 END AS was_win,
        CASE WHEN player_won = FALSE THEN 1 ELSE 0 END AS was_loss
    FROM appearances
),

windowed AS (
    SELECT
        match_id,
        start_time,
        game_version_id,
        player_id,
        hero_id,
        side,
        team_id,
        slot_in_side,
        COALESCE(COUNT(*) OVER w_ph, 0)::BIGINT AS prior_games_on_hero,
        COALESCE(SUM(was_win) OVER w_ph, 0)::BIGINT AS prior_wins_on_hero,
        COALESCE(SUM(was_loss) OVER w_ph, 0)::BIGINT AS prior_losses_on_hero,
        COALESCE(COUNT(*) OVER w_sv, 0)::BIGINT AS same_version_games_on_hero,
        COALESCE(SUM(was_win) OVER w_sv, 0)::BIGINT AS same_version_wins_on_hero,
        COALESCE(COUNT(*) OVER w_90_ph, 0)::BIGINT AS recent_90d_games_on_hero,
        COALESCE(SUM(was_win) OVER w_90_ph, 0)::BIGINT AS recent_90d_wins_on_hero,
        COALESCE(COUNT(*) OVER w_player, 0)::BIGINT AS prior_player_games,
        COALESCE(COUNT(*) OVER w_90_player, 0)::BIGINT AS recent_90d_player_games,
        MAX(start_time) OVER w_ph AS last_played_hero_at
    FROM flagged
    WINDOW
        w_player AS (
            PARTITION BY player_id
            ORDER BY start_time
            {_STRICT_PRIOR_RANGE}
        ),
        w_ph AS (
            PARTITION BY player_id, hero_id
            ORDER BY start_time
            {_STRICT_PRIOR_RANGE}
        ),
        w_sv AS (
            PARTITION BY player_id, hero_id, game_version_id
            ORDER BY start_time
            {_STRICT_PRIOR_RANGE}
        ),
        w_90_player AS (
            PARTITION BY player_id
            ORDER BY start_time
            {_RECENT_90D_RANGE}
        ),
        w_90_ph AS (
            PARTITION BY player_id, hero_id
            ORDER BY start_time
            {_RECENT_90D_RANGE}
        )
)

SELECT
    w.match_id,
    w.start_time,
    w.game_version_id,
    w.player_id,
    w.hero_id,
    {hero_name},
    w.side,
    w.team_id,
    w.slot_in_side,
    w.prior_games_on_hero,
    w.prior_wins_on_hero,
    w.prior_losses_on_hero,
    {win_rate} AS prior_win_rate_on_hero,
    w.same_version_games_on_hero,
    w.same_version_wins_on_hero,
    {win_rate_sv} AS same_version_win_rate_on_hero,
    w.recent_90d_games_on_hero,
    w.recent_90d_wins_on_hero,
    {win_rate_r90} AS recent_90d_win_rate_on_hero,
    w.prior_player_games,
    {hero_share} AS prior_hero_share,
    w.recent_90d_player_games,
    {hero_share_r90} AS recent_90d_hero_share,
    CASE
        WHEN w.last_played_hero_at IS NULL THEN NULL
        ELSE date_diff(
            'microsecond',
            w.last_played_hero_at,
            w.start_time
        )::DOUBLE / 86400000000.0
    END AS days_since_last_played_hero
FROM windowed AS w
{hero_join}
{output_filter}
"""


def _heroes_view_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return HEROES_VIEW in tables


@dataclass(frozen=True)
class PlayerHeroState:
    """Lazy `(match_id, player_id, hero_id)` player×hero relation.

    Nothing is materialized until `to_frame` is called. The owning
    `FeatureDuckDBConnection` must stay open for that call.
    """

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per current player×hero in `PLAYER_HERO_COLUMNS` order."""
        frame = self.relation.df()
        ordered = frame[list(PLAYER_HERO_COLUMNS)]
        return ordered.sort_values(
            [MATCH_ID_COLUMN, PLAYER_ID_COLUMN, HERO_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)


def build_player_hero(
    store: FeatureDuckDBConnection, *, match_id: int | None = None
) -> PlayerHeroState:
    """Build descriptive player×hero state from the registered analytical views.

    Independent of `build_pre_draft_snapshot`. Uses the `heroes` catalog
    for display names when that view is registered. Optional `match_id`
    filters output rows after windows run over the full ordered match
    history.
    """
    sql = player_hero_sql(
        catalog_registered=_heroes_view_registered(store),
        match_id=match_id,
    )
    return PlayerHeroState(relation=store.sql(sql))
