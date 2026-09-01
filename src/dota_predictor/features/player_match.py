"""Player-match fact and strictly-prior player historical state.

Grain
-----
One row per `(match_id, player_id)`: the canonical player-match assignment
plus leakage-safe player-level history computed from strictly earlier
matches. This is infrastructure / descriptive state, not a training
feature and not a side-aggregated draft profile.

The fact relation is a SQL projection over the existing DuckDB
`match_players` and `matches` views. It is not a new Parquet dataset.

Temporal integrity
------------------
Every historical match `h` contributing to the row for current match `c`
satisfies `h.start_time < c.start_time` (see `features.temporal`). Equal
timestamps are mutually blind. The current match itself is never included.

SQL implements that strict-`<` rule with window frames
`RANGE ... CURRENT ROW EXCLUDE GROUP` via `strict_prior_window_sql`,
the same convention as `player_hero` / `team_hero` / `hero_meta`.

Windows
-------
* Career (`player_id`):
  `h.start_time < c.start_time`
* Same STRATZ game version (`player_id`, `game_version_id`):
  `h.game_version_id = c.game_version_id AND h.start_time < c.start_time`

History is keyed by `player_id`. Changing `team_id` does not reset it.
`slot_in_side` is lobby order and is never a window partition, Dota
position 1-5, a lane, or a role. Observed `position`/`lane`/`role` are
fact columns only; they are not window partitions and are not used to
build `prior_*` metrics.

Availability
------------
Fact columns inherit the match/match_players classification:
identity and roster PRE_DRAFT, `hero_id` DRAFT, `won` POST_MATCH
(relative to the match that produced the row). Observed `position` /
`lane` / `role` are POST_MATCH parse labels of that row's match and
must not be selected as PRE_DRAFT or POST_DRAFT features of the same
match. Aggregating historical `won` over strictly earlier matches is
valid historical state for the current match. The current row's `won`
is a fact about that match and is never mixed into `prior_*` /
`version_prior_*` metrics. This layer does not yet compute
position-dependent historical metrics.

NULL semantics
--------------
Counts are observed zeros when there is no history (a first appearance
has `prior_games = 0`). Rates are NULL when the denominator is zero.
`previous_match_start_time` and `days_since_previous_match` are NULL
when the player has no strictly earlier match. No smoothing, decay,
priors, or ratings.

This module never writes Parquet, never bumps schema versions, never
adds columns to the fact/reference files, and is not part of the
training feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.temporal import strict_prior_window_sql

__all__ = [
    "MATCH_ID_COLUMN",
    "PLAYER_ID_COLUMN",
    "PLAYER_MATCH_COLUMNS",
    "PLAYER_STATE_COLUMNS",
    "PLAYER_STATE_METRIC_COLUMNS",
    "PlayerMatch",
    "PlayerState",
    "build_player_match",
    "build_player_state",
    "player_match_sql",
    "player_state_sql",
]


MATCH_ID_COLUMN = "match_id"
PLAYER_ID_COLUMN = "player_id"

PLAYER_MATCH_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    PLAYER_ID_COLUMN,
    "start_time",
    "game_version_id",
    "series_id",
    "team_id",
    "side",
    "hero_id",
    "won",
    "slot_in_side",
    "position",
    "lane",
    "role",
)

PLAYER_STATE_METRIC_COLUMNS: tuple[str, ...] = (
    "prior_games",
    "prior_wins",
    "prior_win_rate",
    "previous_match_start_time",
    "days_since_previous_match",
    "prior_unique_heroes",
    "version_prior_games",
    "version_prior_wins",
    "version_prior_win_rate",
    "version_prior_unique_heroes",
)

PLAYER_STATE_COLUMNS: tuple[str, ...] = PLAYER_MATCH_COLUMNS + PLAYER_STATE_METRIC_COLUMNS

_MICROSECONDS_PER_DAY = 86_400_000_000.0


def _rate_sql(numerator: str, denominator: str) -> str:
    """Raw floating-point ratio, NULL when the denominator is zero."""
    return (
        f"CASE WHEN {denominator} > 0 "
        f"THEN {numerator}::DOUBLE / {denominator} "
        f"ELSE NULL END"
    )


def _days_since_sql(earlier: str, later: str) -> str:
    return (
        f"CASE WHEN {earlier} IS NULL THEN NULL ELSE "
        f"date_diff('microsecond', {earlier}, {later})::DOUBLE "
        f"/ {_MICROSECONDS_PER_DAY} END"
    )


def player_match_sql() -> str:
    """SQL for the canonical player-match fact grain.

    One row per `(match_id, player_id)` from `match_players` joined to
    `matches`. `won` is that player's side winning, never inferred from
    draft order. `slot_in_side` is lobby order, not Dota position 1-5.
    Observed `position`/`lane`/`role` are POST_MATCH parse labels of
    this row's match.
    """
    return f"""
SELECT
    mp.match_id,
    mp.player_id,
    mp.start_time,
    m.game_version_id,
    m.series_id,
    mp.team_id,
    mp.side,
    mp.hero_id,
    CASE
        WHEN mp.side = 'RADIANT' THEN m.radiant_win
        ELSE NOT m.radiant_win
    END AS won,
    mp.slot_in_side,
    mp.position,
    mp.lane,
    mp.role
FROM {MATCH_PLAYERS_VIEW} mp
JOIN {MATCHES_VIEW} m ON m.match_id = mp.match_id
"""


def player_state_sql(*, match_id: int | None = None) -> str:
    """SQL for leakage-safe player state at `(match_id, player_id)`.

    Window functions over the player-match fact implement
    `historical.start_time < current.start_time`, including same-timestamp
    blindness. Optional `match_id` filters output after windows run over
    the full ordered history.
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE match_id = {int(match_id)}"

    career_window = strict_prior_window_sql("player_id")
    version_window = strict_prior_window_sql("player_id", "game_version_id")
    win_rate = _rate_sql("w.prior_wins", "w.prior_games")
    version_win_rate = _rate_sql("w.version_prior_wins", "w.version_prior_games")
    days_since = _days_since_sql("w.previous_match_start_time", "w.start_time")

    return f"""
WITH player_match AS (
{player_match_sql()}
),

windowed AS (
    SELECT
        match_id,
        player_id,
        start_time,
        game_version_id,
        series_id,
        team_id,
        side,
        hero_id,
        won,
        slot_in_side,
        position,
        lane,
        role,
        COALESCE(COUNT(*) OVER w_player, 0)::BIGINT AS prior_games,
        COALESCE(SUM(CASE WHEN won THEN 1 ELSE 0 END) OVER w_player, 0)::BIGINT
            AS prior_wins,
        MAX(start_time) OVER w_player AS previous_match_start_time,
        (COUNT(DISTINCT hero_id) OVER w_player)::BIGINT AS prior_unique_heroes,
        COALESCE(COUNT(*) OVER w_version, 0)::BIGINT AS version_prior_games,
        COALESCE(SUM(CASE WHEN won THEN 1 ELSE 0 END) OVER w_version, 0)::BIGINT
            AS version_prior_wins,
        (COUNT(DISTINCT hero_id) OVER w_version)::BIGINT
            AS version_prior_unique_heroes
    FROM player_match
    WINDOW
        w_player AS ({career_window}),
        w_version AS ({version_window})
)

SELECT
    w.match_id,
    w.player_id,
    w.start_time,
    w.game_version_id,
    w.series_id,
    w.team_id,
    w.side,
    w.hero_id,
    w.won,
    w.slot_in_side,
    w.position,
    w.lane,
    w.role,
    w.prior_games,
    w.prior_wins,
    {win_rate} AS prior_win_rate,
    w.previous_match_start_time,
    {days_since} AS days_since_previous_match,
    w.prior_unique_heroes,
    w.version_prior_games,
    w.version_prior_wins,
    {version_win_rate} AS version_prior_win_rate,
    w.version_prior_unique_heroes
FROM windowed AS w
{output_filter}
"""


@dataclass(frozen=True)
class PlayerMatch:
    """Lazy `(match_id, player_id)` fact relation.

    Nothing is materialized until `to_frame` is called. The owning
    `FeatureDuckDBConnection` must stay open for that call.
    """

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per player-match in `PLAYER_MATCH_COLUMNS` order."""
        frame = self.relation.df()
        ordered = frame[list(PLAYER_MATCH_COLUMNS)]
        return ordered.sort_values(
            [MATCH_ID_COLUMN, PLAYER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)


@dataclass(frozen=True)
class PlayerState:
    """Lazy `(match_id, player_id)` fact plus strictly-prior player state.

    Nothing is materialized until `to_frame` is called. The owning
    `FeatureDuckDBConnection` must stay open for that call.
    """

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per player-match in `PLAYER_STATE_COLUMNS` order."""
        frame = self.relation.df()
        ordered = frame[list(PLAYER_STATE_COLUMNS)]
        return ordered.sort_values(
            [MATCH_ID_COLUMN, PLAYER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)


def build_player_match(store: FeatureDuckDBConnection) -> PlayerMatch:
    """Build the player-match fact relation from registered analytical views."""
    return PlayerMatch(relation=store.sql(player_match_sql()))


def build_player_state(
    store: FeatureDuckDBConnection, *, match_id: int | None = None
) -> PlayerState:
    """Build leakage-safe player state from registered analytical views.

    Independent of `build_pre_draft_snapshot` and `build_player_hero`.
    Optional `match_id` filters output rows after windows run over the
    full ordered match history.
    """
    return PlayerState(relation=store.sql(player_state_sql(match_id=match_id)))
