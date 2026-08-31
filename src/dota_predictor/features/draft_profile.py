"""Side-level descriptive draft profile (not a predictive feature yet).

Grain
-----
One row per `(match_id, side)`: a summary of the five heroes actually
drafted by that side, built from the existing leakage-safe Hero Meta,
Player × Hero, and Team × Hero layers. This is historical state keyed by
the current draft, not a composite strength score and not a training
feature.

Each side row is produced from exactly five current `match_players`
observations. Player familiarity uses the player actually assigned that
hero. Team familiarity uses the current canonical `team_id`. Hero-meta
values are the pre-match state for those same five hero ids.

Temporal integrity
------------------
This module does not recompute history. It only aggregates rows from
`player_hero_sql`, `team_hero_sql`, and `hero_meta_sql`, which already
enforce `historical.start_time < current.start_time` (including
same-timestamp blindness) via `RANGE ... EXCLUDE GROUP`. The current
match's own result is never mixed into the profile.

NULL semantics
--------------
Means and minima of rates/shares use SQL `AVG`/`MIN`, which skip NULL
inputs rather than replacing them with zero. If every drafted hero has a
NULL rate in that context (no historical evidence), the side-level mean
and min are NULL. Game counts are never NULL; a zero count is observed
zero history and is included in means, minima, and zero-history counts.

This layer does not expose in-game events, duration, gold/xp, or the
current match result. It is queryable independently of
`PRE_DRAFT_SNAPSHOT_SQL` and is not part of the training feature matrix.

This module never writes Parquet, never bumps schema versions, and never
adds columns to the fact/reference files.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    HEROES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.hero_meta import hero_meta_sql
from dota_predictor.features.player_hero import player_hero_sql
from dota_predictor.features.team_hero import team_hero_sql

__all__ = [
    "DRAFT_PROFILE_COLUMNS",
    "DRAFT_PROFILE_IDENTITY_COLUMNS",
    "DRAFT_PROFILE_METRIC_COLUMNS",
    "MATCH_ID_COLUMN",
    "DraftProfile",
    "build_draft_profile",
    "draft_profile_sql",
]


MATCH_ID_COLUMN = "match_id"
SIDE_COLUMN = "side"

DRAFT_PROFILE_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    SIDE_COLUMN,
    "team_id",
)

_PLAYER_METRICS: tuple[str, ...] = (
    "mean_player_prior_games_on_hero",
    "min_player_prior_games_on_hero",
    "mean_player_recent_90d_games_on_hero",
    "mean_player_prior_hero_share",
    "mean_player_recent_90d_hero_share",
    "players_with_zero_prior_games_on_hero",
    "players_with_zero_recent_90d_games_on_hero",
)

_TEAM_METRICS: tuple[str, ...] = (
    "mean_team_prior_games_with_hero",
    "min_team_prior_games_with_hero",
    "mean_team_recent_90d_games_with_hero",
    "mean_team_hero_share",
    "mean_team_recent_90d_hero_share",
    "heroes_never_played_by_team",
    "heroes_not_played_by_team_recent_90d",
)

_HERO_META_METRICS: tuple[str, ...] = (
    "mean_same_version_contest_rate",
    "min_same_version_contest_rate",
    "mean_recent_90d_contest_rate",
    "min_recent_90d_contest_rate",
    "mean_same_version_pick_rate",
    "mean_same_version_ban_rate",
    "mean_recent_90d_pick_rate",
    "mean_recent_90d_ban_rate",
)

DRAFT_PROFILE_METRIC_COLUMNS: tuple[str, ...] = (
    _PLAYER_METRICS + _TEAM_METRICS + _HERO_META_METRICS
)

DRAFT_PROFILE_COLUMNS: tuple[str, ...] = (
    DRAFT_PROFILE_IDENTITY_COLUMNS + DRAFT_PROFILE_METRIC_COLUMNS
)

_SIDE_SORT_ORDER = {"RADIANT": 0, "DIRE": 1}


def _zero_count_sql(column: str) -> str:
    return (
        f"SUM(CASE WHEN {column} = 0 THEN 1 ELSE 0 END)::BIGINT"
    )


def draft_profile_sql(
    *, catalog_registered: bool = True, match_id: int | None = None
) -> str:
    """SQL for a leakage-safe `(match_id, side)` draft profile.

    Composes `player_hero_sql`, `team_hero_sql`, and `hero_meta_sql` as
    subqueries (windows still run over the full ordered history) and
    aggregates the five currently drafted heroes per side. Inner layers
    already exclude the current match, future matches, and same-timestamp
    peers. Optional `match_id` filters the aggregated output only.
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE p.match_id = {int(match_id)}"

    player_sql = player_hero_sql(
        catalog_registered=catalog_registered, match_id=None
    )
    team_sql = team_hero_sql(
        catalog_registered=catalog_registered, match_id=None
    )
    meta_sql = hero_meta_sql(
        catalog_registered=catalog_registered, match_id=None
    )

    return f"""
WITH player_hero AS (
    SELECT * FROM ({player_sql}) AS player_hero_inner
),
team_hero AS (
    SELECT * FROM ({team_sql}) AS team_hero_inner
),
hero_meta AS (
    SELECT * FROM ({meta_sql}) AS hero_meta_inner
),
joined AS (
    SELECT
        ph.match_id,
        ph.start_time,
        ph.game_version_id,
        ph.side,
        ph.team_id,
        ph.prior_games_on_hero,
        ph.recent_90d_games_on_hero,
        ph.prior_hero_share,
        ph.recent_90d_hero_share,
        th.team_prior_games_with_hero,
        th.recent_90d_team_games_with_hero,
        th.team_hero_share,
        th.recent_90d_team_hero_share,
        hm.same_version_contest_rate,
        hm.recent_90d_contest_rate,
        hm.same_version_pick_rate,
        hm.same_version_ban_rate,
        hm.recent_90d_pick_rate,
        hm.recent_90d_ban_rate
    FROM player_hero AS ph
    JOIN team_hero AS th
        ON th.match_id = ph.match_id
        AND th.side = ph.side
        AND th.hero_id = ph.hero_id
    LEFT JOIN hero_meta AS hm
        ON hm.match_id = ph.match_id
        AND hm.hero_id = ph.hero_id
)

SELECT
    p.match_id,
    p.start_time,
    p.game_version_id,
    p.side,
    p.team_id,
    AVG(p.prior_games_on_hero)::DOUBLE AS mean_player_prior_games_on_hero,
    MIN(p.prior_games_on_hero)::BIGINT AS min_player_prior_games_on_hero,
    AVG(p.recent_90d_games_on_hero)::DOUBLE
        AS mean_player_recent_90d_games_on_hero,
    AVG(p.prior_hero_share)::DOUBLE AS mean_player_prior_hero_share,
    AVG(p.recent_90d_hero_share)::DOUBLE AS mean_player_recent_90d_hero_share,
    {_zero_count_sql("p.prior_games_on_hero")}
        AS players_with_zero_prior_games_on_hero,
    {_zero_count_sql("p.recent_90d_games_on_hero")}
        AS players_with_zero_recent_90d_games_on_hero,
    AVG(p.team_prior_games_with_hero)::DOUBLE
        AS mean_team_prior_games_with_hero,
    MIN(p.team_prior_games_with_hero)::BIGINT AS min_team_prior_games_with_hero,
    AVG(p.recent_90d_team_games_with_hero)::DOUBLE
        AS mean_team_recent_90d_games_with_hero,
    AVG(p.team_hero_share)::DOUBLE AS mean_team_hero_share,
    AVG(p.recent_90d_team_hero_share)::DOUBLE AS mean_team_recent_90d_hero_share,
    {_zero_count_sql("p.team_prior_games_with_hero")}
        AS heroes_never_played_by_team,
    {_zero_count_sql("p.recent_90d_team_games_with_hero")}
        AS heroes_not_played_by_team_recent_90d,
    AVG(p.same_version_contest_rate)::DOUBLE AS mean_same_version_contest_rate,
    MIN(p.same_version_contest_rate)::DOUBLE AS min_same_version_contest_rate,
    AVG(p.recent_90d_contest_rate)::DOUBLE AS mean_recent_90d_contest_rate,
    MIN(p.recent_90d_contest_rate)::DOUBLE AS min_recent_90d_contest_rate,
    AVG(p.same_version_pick_rate)::DOUBLE AS mean_same_version_pick_rate,
    AVG(p.same_version_ban_rate)::DOUBLE AS mean_same_version_ban_rate,
    AVG(p.recent_90d_pick_rate)::DOUBLE AS mean_recent_90d_pick_rate,
    AVG(p.recent_90d_ban_rate)::DOUBLE AS mean_recent_90d_ban_rate
FROM joined AS p
{output_filter}
GROUP BY
    p.match_id,
    p.start_time,
    p.game_version_id,
    p.side,
    p.team_id
"""


def _heroes_view_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return HEROES_VIEW in tables


@dataclass(frozen=True)
class DraftProfile:
    """Lazy `(match_id, side)` draft-profile relation.

    Nothing is materialized until `to_frame` is called. The owning
    `FeatureDuckDBConnection` must stay open for that call.
    """

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per side in `DRAFT_PROFILE_COLUMNS` order."""
        frame = self.relation.df()
        ordered = frame[list(DRAFT_PROFILE_COLUMNS)].copy()
        side_order = ordered[SIDE_COLUMN].map(_SIDE_SORT_ORDER)
        return (
            ordered.assign(_side_order=side_order)
            .sort_values([MATCH_ID_COLUMN, "_side_order"], kind="mergesort")
            .drop(columns="_side_order")
            .reset_index(drop=True)
        )


def build_draft_profile(
    store: FeatureDuckDBConnection, *, match_id: int | None = None
) -> DraftProfile:
    """Build a descriptive side-level draft profile from registered views.

    Independent of `build_pre_draft_snapshot`. Reuses the Player × Hero,
    Team × Hero, and Hero Meta SQL as-is. Optional `match_id` filters
    aggregated output after the inner windows run over full history.
    """
    sql = draft_profile_sql(
        catalog_registered=_heroes_view_registered(store),
        match_id=match_id,
    )
    return DraftProfile(relation=store.sql(sql))
