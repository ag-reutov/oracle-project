"""PRE_DRAFT historical training snapshot (Step 3B).

Builds one row per canonical match containing:

* identity/context columns (never predictive features -- see
  `IDENTITY_COLUMNS`),
* the label (`TARGET_COLUMN`, kept explicitly separate from
  `FEATURE_COLUMNS`),
* team-history and player-history features and roster-continuity
  features, all computed from strictly earlier matches only.

Core invariant (`.cursor/rules/ml.mdc`, `features.temporal`)
--------------------------------------------------------------
For a current match `m`, every historical row contributing to a feature
for `m` satisfies `historical.start_time < m.start_time`. `match_id` is
never used as a time proxy anywhere in this module -- see
`PRE_DRAFT_SNAPSHOT_SQL` for the exact SQL, and
`test_pre_draft_snapshot.py::test_sql_never_orders_history_by_match_id`
for a static regression guard on that property. The one place
`match_id` legitimately appears in a historical ordering context is as
a *tie-breaker* for "most recent prior match" in roster continuity, only
among rows already filtered to `start_time < m.start_time` -- this is
allowed and required by the Step 3B spec (ties in `start_time` need a
deterministic secondary key).

Why using an earlier match's own `radiant_win` is not a temporal-
integrity violation
---------------------------------------------------------------------
`features.availability` classifies `radiant_win` as `POST_MATCH`
*relative to the match that produced it*. That classification is about
whether a match's own outcome may be used as a feature *for that same
match* -- it says nothing about later matches. Once a historical match
`h` satisfies `h.start_time < m.start_time`, `h` has already completed
and its outcome is public historical record by the time `m` is played;
using `h.radiant_win` to compute *win-rate history* for `m` is exactly
what "strictly earlier matches" means and is required by the Step 3B
spec. What must never happen -- and does not happen here -- is using
`m.radiant_win` (the *current* match's own outcome) as an input
feature; it is only ever selected once, as `TARGET_COLUMN`, and
`IDENTITY_COLUMNS`/`FEATURE_COLUMNS` are asserted PRE_DRAFT-safe against
`features.availability` at import time (see `_assert_contract`).

Algorithm summary
------------------
Three long-form analytical relations are built as CTEs on top of the
Step 3A `matches`/`match_players` views:

* `team_appearances`: 2 rows per match (one per side), carrying the
  real team identity (`team_id`) and whether *that team* won,
  independent of Radiant/Dire position.
* `player_appearances`: 10 rows per match (one per roster slot),
  carrying the real player identity and whether *that player's side*
  won.
* For roster continuity, `team_appearances` is self-joined on
  `team_id` with `h.start_time < c.start_time`, and `ROW_NUMBER() OVER
  (PARTITION BY match_id, side ORDER BY start_time DESC, match_id DESC)`
  picks the single most recent prior match per (current match, side).

Team-history and player-history features are then produced by
aggregating `team_appearances`/`player_appearances` per current match
via `GROUP BY`, filtered by the strict historical-eligibility join
condition -- bulk, relation-level SQL, not one query per match.

Team Elo (Step 3C)
------------------
`TEAM_ELO_FEATURE_COLUMNS` (`radiant_team_elo`/`dire_team_elo`/
`team_elo_delta`) are computed separately by `features.team_elo` --
that module's docstring has the exact algorithm and temporal-integrity
argument. They cannot be expressed by `PRE_DRAFT_SNAPSHOT_SQL` itself
(Elo is a sequential recurrence, not a `GROUP BY`/`JOIN`-able
aggregate), so `PreDraftSnapshot.to_frame` computes them from the
materialized SQL result and merges them in as a plain left join on
`match_id`, one-to-one -- see `to_frame`. `PRE_DRAFT_SNAPSHOT_SQL`
itself is unchanged by Step 3C.

Persistence
------------
Nothing is persisted or materialized to disk by this module -- see
`build_pre_draft_snapshot`, which returns a lazy `PreDraftSnapshot`
wrapping an unmaterialized `duckdb.DuckDBPyRelation`. `to_frame` (and
therefore `feature_frame`/`target_series`) still has to materialize
that relation to a `pandas.DataFrame` to compute Elo, exactly as it
already did to return a `DataFrame` at all -- `build_pre_draft_snapshot`
itself performs no materialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import pandas as pd

from dota_predictor.features.availability import (
    SnapshotStage,
    assert_columns_allowed_for_stage,
)
from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.team_elo import (
    DEFAULT_ELO_CONFIG,
    TEAM_ELO_FEATURE_COLUMNS,
    EloConfig,
    compute_team_elo_features,
)

__all__ = [
    "FEATURE_COLUMNS",
    "IDENTITY_COLUMNS",
    "PLAYER_HISTORY_FEATURE_COLUMNS",
    "PRE_DRAFT_SNAPSHOT_SQL",
    "ROSTER_CONTINUITY_FEATURE_COLUMNS",
    "SNAPSHOT_COLUMNS",
    "TARGET_COLUMN",
    "TEAM_ELO_FEATURE_COLUMNS",
    "TEAM_HISTORY_FEATURE_COLUMNS",
    "PreDraftSnapshot",
    "build_pre_draft_snapshot",
]

# --- Output schema -------------------------------------------------------

# Identity/context: never predictive features (see module docstring and
# `.cursor/rules/ml.mdc` -- observed team *names* are deliberately
# excluded; team *ids* are identity/context only at this step).
IDENTITY_COLUMNS: tuple[str, ...] = (
    "match_id",
    "start_time",
    "league_id",
    "series_id",
    "series_type",
    "game_version_id",
    "radiant_team_id",
    "dire_team_id",
)

# The label. Kept out of FEATURE_COLUMNS everywhere in this module --
# see `PreDraftSnapshot.feature_frame`/`target_series`.
TARGET_COLUMN = "radiant_win"

TEAM_HISTORY_FEATURE_COLUMNS: tuple[str, ...] = (
    "radiant_team_prior_matches",
    "dire_team_prior_matches",
    "team_prior_matches_delta",
    "radiant_team_prior_wins",
    "dire_team_prior_wins",
    "team_prior_wins_delta",
    "radiant_team_prior_losses",
    "dire_team_prior_losses",
    "team_prior_losses_delta",
    "radiant_team_prior_win_rate",
    "dire_team_prior_win_rate",
    "team_prior_win_rate_delta",
)

PLAYER_HISTORY_FEATURE_COLUMNS: tuple[str, ...] = (
    "radiant_players_prior_matches_mean",
    "dire_players_prior_matches_mean",
    "players_prior_matches_mean_delta",
    "radiant_players_prior_matches_min",
    "dire_players_prior_matches_min",
    "players_prior_matches_min_delta",
    "radiant_players_prior_matches_max",
    "dire_players_prior_matches_max",
    "players_prior_matches_max_delta",
    "radiant_players_prior_win_rate_mean",
    "dire_players_prior_win_rate_mean",
    "players_prior_win_rate_mean_delta",
    "radiant_players_zero_prior_matches_count",
    "dire_players_zero_prior_matches_count",
    "players_zero_prior_matches_count_delta",
)

ROSTER_CONTINUITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "radiant_roster_players_retained",
    "dire_roster_players_retained",
    "roster_players_retained_delta",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    TEAM_HISTORY_FEATURE_COLUMNS
    + PLAYER_HISTORY_FEATURE_COLUMNS
    + ROSTER_CONTINUITY_FEATURE_COLUMNS
    + TEAM_ELO_FEATURE_COLUMNS
)

# Full column order of one snapshot row, as returned by
# `build_pre_draft_snapshot(...).relation`.
SNAPSHOT_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + FEATURE_COLUMNS + (TARGET_COLUMN,)
)


def _assert_contract() -> None:
    """Fail fast (at import time) if `IDENTITY_COLUMNS` ever drifts from
    what `features.availability` considers PRE_DRAFT-safe on `matches`.

    This does not (and cannot) check `FEATURE_COLUMNS` the same way --
    those are *derived* aggregate names with no matching raw Parquet
    column -- but it does pin down that the raw identity columns this
    module reads directly off `matches` are legitimately PRE_DRAFT.
    """
    assert_columns_allowed_for_stage(
        MATCHES_VIEW, SnapshotStage.PRE_DRAFT, IDENTITY_COLUMNS
    )


_assert_contract()


# --- SQL ------------------------------------------------------------------

# Every historical join condition below uses `start_time <`
# (`HISTORICAL_START_TIME_SQL_CONDITION`'s strict-`<` semantics inlined
# for readability) and never `match_id`, except the single deliberate
# `ORDER BY ... , match_id DESC` tie-breaker in
# `prior_team_match_candidates`, which only orders *already strictly
# historical* rows (see module docstring).
PRE_DRAFT_SNAPSHOT_SQL = f"""
WITH team_appearances AS (
    -- One row per (match, side) with the REAL team identity and
    -- whether that team won, independent of Radiant/Dire position.
    SELECT
        match_id,
        start_time,
        radiant_team_id AS team_id,
        'RADIANT' AS side,
        radiant_win AS team_won
    FROM {MATCHES_VIEW}
    UNION ALL
    SELECT
        match_id,
        start_time,
        dire_team_id AS team_id,
        'DIRE' AS side,
        NOT radiant_win AS team_won
    FROM {MATCHES_VIEW}
),

radiant_team_history AS (
    SELECT
        c.match_id,
        COUNT(h.match_id) AS radiant_team_prior_matches,
        SUM(CASE WHEN h.team_won THEN 1 ELSE 0 END)::BIGINT AS radiant_team_prior_wins,
        SUM(CASE WHEN h.team_won = FALSE THEN 1 ELSE 0 END)::BIGINT AS radiant_team_prior_losses,
        CASE WHEN COUNT(h.match_id) > 0
             THEN SUM(CASE WHEN h.team_won THEN 1 ELSE 0 END)::DOUBLE / COUNT(h.match_id)
             ELSE NULL END AS radiant_team_prior_win_rate
    FROM {MATCHES_VIEW} c
    LEFT JOIN team_appearances h
        ON h.team_id = c.radiant_team_id AND h.start_time < c.start_time
    GROUP BY c.match_id
),

dire_team_history AS (
    SELECT
        c.match_id,
        COUNT(h.match_id) AS dire_team_prior_matches,
        SUM(CASE WHEN h.team_won THEN 1 ELSE 0 END)::BIGINT AS dire_team_prior_wins,
        SUM(CASE WHEN h.team_won = FALSE THEN 1 ELSE 0 END)::BIGINT AS dire_team_prior_losses,
        CASE WHEN COUNT(h.match_id) > 0
             THEN SUM(CASE WHEN h.team_won THEN 1 ELSE 0 END)::DOUBLE / COUNT(h.match_id)
             ELSE NULL END AS dire_team_prior_win_rate
    FROM {MATCHES_VIEW} c
    LEFT JOIN team_appearances h
        ON h.team_id = c.dire_team_id AND h.start_time < c.start_time
    GROUP BY c.match_id
),

player_appearances AS (
    -- One row per (match, roster slot) with the REAL player identity
    -- and whether that player's side won in that historical match.
    SELECT
        mp.match_id,
        mp.start_time,
        mp.player_id,
        mp.side,
        CASE WHEN mp.side = 'RADIANT' THEN m.radiant_win ELSE NOT m.radiant_win END
            AS player_won
    FROM {MATCH_PLAYERS_VIEW} mp
    JOIN {MATCHES_VIEW} m ON m.match_id = mp.match_id
),

player_prior AS (
    SELECT
        c.match_id,
        c.side,
        c.player_id,
        COUNT(h.match_id) AS prior_matches,
        SUM(CASE WHEN h.player_won THEN 1 ELSE 0 END)::BIGINT AS prior_wins,
        SUM(CASE WHEN h.player_won = FALSE THEN 1 ELSE 0 END)::BIGINT AS prior_losses,
        CASE WHEN COUNT(h.match_id) > 0
             THEN SUM(CASE WHEN h.player_won THEN 1 ELSE 0 END)::DOUBLE / COUNT(h.match_id)
             ELSE NULL END AS prior_win_rate
    FROM {MATCH_PLAYERS_VIEW} c
    LEFT JOIN player_appearances h
        ON h.player_id = c.player_id AND h.start_time < c.start_time
    GROUP BY c.match_id, c.side, c.player_id
),

player_side_agg AS (
    -- Aggregate the 5 current players on each side. AVG() ignores NULL
    -- inputs by SQL semantics, which is exactly "mean raw prior win
    -- rate, using only players for whom a prior win rate exists".
    SELECT
        match_id,
        side,
        AVG(prior_matches) AS players_prior_matches_mean,
        MIN(prior_matches) AS players_prior_matches_min,
        MAX(prior_matches) AS players_prior_matches_max,
        AVG(prior_win_rate) AS players_prior_win_rate_mean,
        SUM(CASE WHEN prior_matches = 0 THEN 1 ELSE 0 END)::BIGINT AS players_zero_prior_matches_count
    FROM player_prior
    GROUP BY match_id, side
),

radiant_player_agg AS (
    SELECT
        match_id,
        players_prior_matches_mean AS radiant_players_prior_matches_mean,
        players_prior_matches_min AS radiant_players_prior_matches_min,
        players_prior_matches_max AS radiant_players_prior_matches_max,
        players_prior_win_rate_mean AS radiant_players_prior_win_rate_mean,
        players_zero_prior_matches_count AS radiant_players_zero_prior_matches_count
    FROM player_side_agg
    WHERE side = 'RADIANT'
),

dire_player_agg AS (
    SELECT
        match_id,
        players_prior_matches_mean AS dire_players_prior_matches_mean,
        players_prior_matches_min AS dire_players_prior_matches_min,
        players_prior_matches_max AS dire_players_prior_matches_max,
        players_prior_win_rate_mean AS dire_players_prior_win_rate_mean,
        players_zero_prior_matches_count AS dire_players_zero_prior_matches_count
    FROM player_side_agg
    WHERE side = 'DIRE'
),

prior_team_match_candidates AS (
    -- For each (current match, side)'s team, rank every strictly
    -- earlier appearance of that same team by recency. start_time DESC
    -- is the temporal key; match_id DESC is ONLY a tie-breaker among
    -- rows already filtered to h.start_time < c.start_time.
    SELECT
        c.match_id,
        c.side AS current_side,
        h.match_id AS prior_match_id,
        h.side AS prior_side,
        ROW_NUMBER() OVER (
            PARTITION BY c.match_id, c.side
            ORDER BY h.start_time DESC, h.match_id DESC
        ) AS rn
    FROM team_appearances c
    JOIN team_appearances h
        ON h.team_id = c.team_id AND h.start_time < c.start_time
),

most_recent_prior_team_match AS (
    SELECT match_id, current_side, prior_match_id, prior_side
    FROM prior_team_match_candidates
    WHERE rn = 1
),

roster_continuity AS (
    -- COUNT(*) of players present in BOTH the current roster (for this
    -- team's current side) and that same team's roster in its most
    -- recent prior match (on whichever side it played then). A team
    -- with no prior match has no row here at all -- see the LEFT JOINs
    -- below, which is exactly the required NULL-not-zero semantics.
    SELECT
        mrp.match_id,
        mrp.current_side,
        COUNT(*) AS players_retained
    FROM most_recent_prior_team_match mrp
    JOIN {MATCH_PLAYERS_VIEW} cur
        ON cur.match_id = mrp.match_id AND cur.side = mrp.current_side
    JOIN {MATCH_PLAYERS_VIEW} hist
        ON hist.match_id = mrp.prior_match_id
       AND hist.side = mrp.prior_side
       AND hist.player_id = cur.player_id
    GROUP BY mrp.match_id, mrp.current_side
),

radiant_roster AS (
    SELECT match_id, players_retained AS radiant_roster_players_retained
    FROM roster_continuity
    WHERE current_side = 'RADIANT'
),

dire_roster AS (
    SELECT match_id, players_retained AS dire_roster_players_retained
    FROM roster_continuity
    WHERE current_side = 'DIRE'
)

SELECT
    m.match_id,
    m.start_time,
    m.league_id,
    m.series_id,
    m.series_type,
    m.game_version_id,
    m.radiant_team_id,
    m.dire_team_id,

    rt.radiant_team_prior_matches,
    dt.dire_team_prior_matches,
    (rt.radiant_team_prior_matches - dt.dire_team_prior_matches) AS team_prior_matches_delta,
    rt.radiant_team_prior_wins,
    dt.dire_team_prior_wins,
    (rt.radiant_team_prior_wins - dt.dire_team_prior_wins) AS team_prior_wins_delta,
    rt.radiant_team_prior_losses,
    dt.dire_team_prior_losses,
    (rt.radiant_team_prior_losses - dt.dire_team_prior_losses) AS team_prior_losses_delta,
    rt.radiant_team_prior_win_rate,
    dt.dire_team_prior_win_rate,
    (rt.radiant_team_prior_win_rate - dt.dire_team_prior_win_rate) AS team_prior_win_rate_delta,

    rpa.radiant_players_prior_matches_mean,
    dpa.dire_players_prior_matches_mean,
    (rpa.radiant_players_prior_matches_mean - dpa.dire_players_prior_matches_mean)
        AS players_prior_matches_mean_delta,
    rpa.radiant_players_prior_matches_min,
    dpa.dire_players_prior_matches_min,
    (rpa.radiant_players_prior_matches_min - dpa.dire_players_prior_matches_min)
        AS players_prior_matches_min_delta,
    rpa.radiant_players_prior_matches_max,
    dpa.dire_players_prior_matches_max,
    (rpa.radiant_players_prior_matches_max - dpa.dire_players_prior_matches_max)
        AS players_prior_matches_max_delta,
    rpa.radiant_players_prior_win_rate_mean,
    dpa.dire_players_prior_win_rate_mean,
    (rpa.radiant_players_prior_win_rate_mean - dpa.dire_players_prior_win_rate_mean)
        AS players_prior_win_rate_mean_delta,
    rpa.radiant_players_zero_prior_matches_count,
    dpa.dire_players_zero_prior_matches_count,
    (rpa.radiant_players_zero_prior_matches_count - dpa.dire_players_zero_prior_matches_count)
        AS players_zero_prior_matches_count_delta,

    rr.radiant_roster_players_retained,
    dr.dire_roster_players_retained,
    (rr.radiant_roster_players_retained - dr.dire_roster_players_retained)
        AS roster_players_retained_delta,

    m.radiant_win

FROM {MATCHES_VIEW} m
LEFT JOIN radiant_team_history rt ON rt.match_id = m.match_id
LEFT JOIN dire_team_history dt ON dt.match_id = m.match_id
LEFT JOIN radiant_player_agg rpa ON rpa.match_id = m.match_id
LEFT JOIN dire_player_agg dpa ON dpa.match_id = m.match_id
LEFT JOIN radiant_roster rr ON rr.match_id = m.match_id
LEFT JOIN dire_roster dr ON dr.match_id = m.match_id
ORDER BY m.start_time, m.match_id
"""


@dataclass(frozen=True)
class PreDraftSnapshot:
    """One row per canonical match, PRE_DRAFT-stage historical features.

    Wraps a lazy `duckdb.DuckDBPyRelation` -- nothing is materialized
    until `to_frame`/`feature_frame`/`target_series` is called, and the
    owning `FeatureDuckDBConnection` must stay open for the lifetime of
    any of those calls.
    """

    relation: duckdb.DuckDBPyRelation
    identity_columns: tuple[str, ...] = field(default=IDENTITY_COLUMNS)
    feature_columns: tuple[str, ...] = field(default=FEATURE_COLUMNS)
    target_column: str = field(default=TARGET_COLUMN)
    elo_config: EloConfig = field(default=DEFAULT_ELO_CONFIG)

    def to_frame(self) -> pd.DataFrame:
        """The full snapshot: identity + feature + target columns, one
        row per canonical match.

        Team Elo (`TEAM_ELO_FEATURE_COLUMNS`) is computed from this
        same materialized frame (see `features.team_elo`) and merged
        in one-to-one on `match_id` -- `validate="one_to_one"` is a
        direct, load-bearing check on the "exactly one row per
        canonical match" invariant, not just documentation of intent.
        """
        base = self.relation.df()
        elo_features = compute_team_elo_features(base, config=self.elo_config)
        merged = base.merge(
            elo_features, on="match_id", how="left", validate="one_to_one"
        )
        return merged[list(SNAPSHOT_COLUMNS)]

    def feature_frame(self) -> pd.DataFrame:
        """Only `feature_columns` -- never the target, never identity
        columns. This is the availability/feature contract boundary:
        whatever is fit to a model must come from here, not `to_frame`.
        """
        return self.to_frame()[list(self.feature_columns)]

    def target_series(self) -> pd.Series:
        """Only the label column."""
        return self.to_frame()[self.target_column]


def build_pre_draft_snapshot(
    store: FeatureDuckDBConnection, *, elo_config: EloConfig | None = None
) -> PreDraftSnapshot:
    """Build the PRE_DRAFT historical snapshot: one row per canonical
    match in `store`, with strictly-earlier-match team/player history,
    roster-continuity, and team-Elo features plus the separated
    `radiant_win` target.

    `store` must be an open `FeatureDuckDBConnection` (see
    `features.duckdb_layer.connect`). This is the single public entry
    point for Step 3B/3C; see `PRE_DRAFT_SNAPSHOT_SQL` for the
    team/player-history query, `features.team_elo` for the Elo
    algorithm, and this module's docstring for the temporal-integrity
    argument. `elo_config` defaults to `team_elo.DEFAULT_ELO_CONFIG`
    (initial rating 1500.0, K-factor 32.0) when not given.
    """
    relation = store.sql(PRE_DRAFT_SNAPSHOT_SQL)
    resolved_elo_config = elo_config if elo_config is not None else DEFAULT_ELO_CONFIG
    return PreDraftSnapshot(relation=relation, elo_config=resolved_elo_config)
