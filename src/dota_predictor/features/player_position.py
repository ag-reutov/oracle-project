"""Leakage-safe historical player × position state.

Grain
-----
One row per `(match_id, player_id)`: the canonical player-match fact plus
strictly-prior descriptive state at each explicit Dota position 1–5.
This is observed-history infrastructure, not expected current position,
not a training feature, and not a new Parquet dataset.

Observed vs expected
--------------------
`position` / `lane` / `role` on the current row are STRATZ parse labels
of *this* match (POST_MATCH). They are never used as PRE_DRAFT features
of the same match and are never the lookup key for this layer.

Historical metrics describe what the player actually played in completed
prior matches. They do not infer what the player will play now.

Temporal integrity
------------------
Every historical match `H` contributing to current match `M` satisfies
`H.start_time < M.start_time` via `strict_prior_window_sql` (RANGE ...
EXCLUDE GROUP). Equal timestamps are mutually blind. History is never
ordered by `match_id`. The current row's observed position does not
enter that row's historical counts.

Windows
-------
* Career: `PARTITION BY player_id`
* Same STRATZ game version: `PARTITION BY player_id, game_version_id`
* Recent trailing matches: last 5 / 10 / 20 *strictly prior* player-match
  rows, including NULL-position matches as occupying a slot.

NULL / UNKNOWN / FILTERED / ALL
-------------------------------
Those rows remain in generic career `prior_games` and occupy a slot in
recent-match windows. They do not increment any `*_position_N` count,
do not receive an inferred position, and are excluded from the explicit
denominator used for role-distribution shares.

Share denominators
------------------
`prior_share_position_N` and modal/recent shares use **prior matches
with an explicit POSITION_1–5**, not all prior matches. Generic
`prior_games` still counts every strictly earlier match.

Omitted metric
--------------
`days_since_previous_position_change` is not emitted. Consecutive-streak
vs last-switch-event vs NULL-interrupted-streak are different quantities;
per-position recency and `previous_explicit_position` already cover the
unambiguous cases.

`recent_position_stability` is the last-10 window's modal share among
explicit positions in that window.

This module never writes Parquet, never bumps schema versions, never
alters Elo / walk-forward / Player × Hero, and is not part of the
training feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from dota_predictor.data.canonical_schema import MatchPlayerPosition
from dota_predictor.features.duckdb_layer import FeatureDuckDBConnection
from dota_predictor.features.player_match import (
    MATCH_ID_COLUMN,
    PLAYER_ID_COLUMN,
    PLAYER_MATCH_COLUMNS,
    player_match_sql,
)
from dota_predictor.features.temporal import strict_prior_window_sql

__all__ = [
    "EXPLICIT_POSITION_LABELS",
    "PLAYER_POSITION_STATE_COLUMNS",
    "PLAYER_POSITION_STATE_METRIC_COLUMNS",
    "RECENT_POSITION_WINDOWS",
    "RECENT_STABILITY_WINDOW",
    "PlayerPositionState",
    "build_player_position_state",
    "player_position_state_sql",
]

EXPLICIT_POSITION_LABELS: tuple[str, ...] = (
    MatchPlayerPosition.POSITION_1.value,
    MatchPlayerPosition.POSITION_2.value,
    MatchPlayerPosition.POSITION_3.value,
    MatchPlayerPosition.POSITION_4.value,
    MatchPlayerPosition.POSITION_5.value,
)

RECENT_POSITION_WINDOWS: tuple[int, ...] = (5, 10, 20)
RECENT_STABILITY_WINDOW = 10

_MICROSECONDS_PER_DAY = 86_400_000_000.0
_EXPLICIT_IN_SQL = ", ".join(f"'{label}'" for label in EXPLICIT_POSITION_LABELS)


def _position_suffix(label: str) -> str:
    return label.lower()


def _rate_sql(numerator: str, denominator: str) -> str:
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


def _unique_mode_sql(count_sql: dict[str, str]) -> str:
    """POSITION_N of the unique maximum count, else NULL (no history or tie)."""
    branches: list[str] = []
    for label in EXPLICIT_POSITION_LABELS:
        others = [
            f"{count_sql[label]} > {count_sql[other]}"
            for other in EXPLICIT_POSITION_LABELS
            if other != label
        ]
        cond = " AND ".join([f"{count_sql[label]} > 0", *others])
        branches.append(f"WHEN {cond} THEN '{label}'")
    return "CASE " + " ".join(branches) + " ELSE NULL END"


def _distinct_positions_sql(count_sql: dict[str, str]) -> str:
    return " + ".join(
        f"CASE WHEN {count_sql[label]} > 0 THEN 1 ELSE 0 END"
        for label in EXPLICIT_POSITION_LABELS
    )


def _recent_slice_sql(list_expr: str, window: int) -> str:
    return (
        f"CASE WHEN {list_expr} IS NULL OR len({list_expr}) = 0 THEN "
        f"CAST([] AS VARCHAR[]) ELSE list_slice({list_expr}, "
        f"GREATEST(len({list_expr}) - {window - 1}, 1), len({list_expr})) END"
    )


def _list_count_eq_sql(list_expr: str, label: str) -> str:
    return (
        f"COALESCE(len(list_filter({list_expr}, x -> x = '{label}')), 0)::BIGINT"
    )


def _per_position_metric_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for label in EXPLICIT_POSITION_LABELS:
        suffix = _position_suffix(label)
        columns.extend(
            [
                f"prior_games_{suffix}",
                f"prior_wins_{suffix}",
                f"prior_win_rate_{suffix}",
                f"prior_share_{suffix}",
                f"previous_start_time_{suffix}",
                f"days_since_{suffix}",
                f"version_prior_games_{suffix}",
                f"version_prior_wins_{suffix}",
                f"version_prior_win_rate_{suffix}",
            ]
        )
    return tuple(columns)


def _recent_metric_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for window in RECENT_POSITION_WINDOWS:
        columns.append(f"recent_{window}_explicit_games")
        columns.extend(
            f"recent_{window}_games_{_position_suffix(label)}"
            for label in EXPLICIT_POSITION_LABELS
        )
        columns.extend(
            [
                f"recent_{window}_modal_position",
                f"recent_{window}_modal_position_share",
                f"recent_{window}_distinct_positions",
            ]
        )
    return tuple(columns)


PLAYER_POSITION_STATE_METRIC_COLUMNS: tuple[str, ...] = (
    (
        "prior_games",
        "prior_explicit_position_games",
    )
    + _per_position_metric_columns()
    + (
        "historical_modal_position",
        "historical_modal_position_games",
        "historical_modal_position_share",
        "historical_distinct_positions",
        "previous_explicit_position",
        "days_since_previous_explicit_position",
        "prior_games_same_as_previous_position",
        "recent_position_stability",
    )
    + _recent_metric_columns()
)

PLAYER_POSITION_STATE_COLUMNS: tuple[str, ...] = (
    PLAYER_MATCH_COLUMNS + PLAYER_POSITION_STATE_METRIC_COLUMNS
)


def _windowed_select_sql() -> str:
    career_counts: list[str] = [
        "COALESCE(COUNT(*) OVER w_player, 0)::BIGINT AS prior_games",
        (
            "COALESCE(COUNT(*) FILTER (WHERE position IN "
            f"({_EXPLICIT_IN_SQL})) OVER w_player, 0)::BIGINT "
            "AS prior_explicit_position_games"
        ),
        (
            "arg_max(CASE WHEN position IN "
            f"({_EXPLICIT_IN_SQL}) THEN position END, start_time) "
            "OVER w_player AS previous_explicit_position"
        ),
        (
            "MAX(CASE WHEN position IN "
            f"({_EXPLICIT_IN_SQL}) THEN start_time END) "
            "OVER w_player AS previous_explicit_start_time"
        ),
        "list(position) OVER w_player AS prior_positions",
    ]
    for label in EXPLICIT_POSITION_LABELS:
        suffix = _position_suffix(label)
        career_counts.extend(
            [
                (
                    f"COALESCE(COUNT(*) FILTER (WHERE position = '{label}') "
                    f"OVER w_player, 0)::BIGINT AS prior_games_{suffix}"
                ),
                (
                    "COALESCE(SUM(CASE WHEN position = "
                    f"'{label}' AND won THEN 1 ELSE 0 END) OVER w_player, 0)"
                    f"::BIGINT AS prior_wins_{suffix}"
                ),
                (
                    f"MAX(CASE WHEN position = '{label}' THEN start_time END) "
                    f"OVER w_player AS previous_start_time_{suffix}"
                ),
                (
                    f"COALESCE(COUNT(*) FILTER (WHERE position = '{label}') "
                    f"OVER w_version, 0)::BIGINT AS version_prior_games_{suffix}"
                ),
                (
                    "COALESCE(SUM(CASE WHEN position = "
                    f"'{label}' AND won THEN 1 ELSE 0 END) OVER w_version, 0)"
                    f"::BIGINT AS version_prior_wins_{suffix}"
                ),
            ]
        )
    return ",\n        ".join(career_counts)


def _modal_games_sql(count_sql: dict[str, str]) -> str:
    branches = " ".join(
        f"WHEN '{label}' THEN {count_sql[label]}"
        for label in EXPLICIT_POSITION_LABELS
    )
    return f"CASE {_unique_mode_sql(count_sql)} {branches} ELSE NULL END"


def player_position_state_sql(*, match_id: int | None = None) -> str:
    """SQL for leakage-safe player × position state at `(match_id, player_id)`.

    Window functions over the player-match fact implement
    `historical.start_time < current.start_time`. Optional `match_id`
    filters output after windows run over the full ordered history.
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE match_id = {int(match_id)}"

    career_window = strict_prior_window_sql("player_id")
    version_window = strict_prior_window_sql("player_id", "game_version_id")
    stability_sql = ""

    # Build the sliced CTE then the outer SELECT that can reference aliases.
    slice_cols = ",\n        ".join(
        [
            "w.*",
            *[
                f"{_recent_slice_sql('w.prior_positions', window)} "
                f"AS recent_{window}_positions"
                for window in RECENT_POSITION_WINDOWS
            ],
        ]
    )

    outer_metrics: list[str] = []
    career_counts = {
        label: f"s.prior_games_{_position_suffix(label)}"
        for label in EXPLICIT_POSITION_LABELS
    }
    for label in EXPLICIT_POSITION_LABELS:
        suffix = _position_suffix(label)
        outer_metrics.extend(
            [
                f"s.prior_games_{suffix}",
                f"s.prior_wins_{suffix}",
                (
                    f"{_rate_sql(f's.prior_wins_{suffix}', f's.prior_games_{suffix}')} "
                    f"AS prior_win_rate_{suffix}"
                ),
                (
                    f"{_rate_sql(f's.prior_games_{suffix}', 's.prior_explicit_position_games')} "
                    f"AS prior_share_{suffix}"
                ),
                f"s.previous_start_time_{suffix}",
                (
                    f"{_days_since_sql(f's.previous_start_time_{suffix}', 's.start_time')} "
                    f"AS days_since_{suffix}"
                ),
                f"s.version_prior_games_{suffix}",
                f"s.version_prior_wins_{suffix}",
                (
                    f"{_rate_sql(f's.version_prior_wins_{suffix}', f's.version_prior_games_{suffix}')} "
                    f"AS version_prior_win_rate_{suffix}"
                ),
            ]
        )

    outer_metrics.extend(
        [
            f"{_unique_mode_sql(career_counts)} AS historical_modal_position",
            f"{_modal_games_sql(career_counts)} AS historical_modal_position_games",
            (
                f"{_rate_sql(_modal_games_sql(career_counts), 's.prior_explicit_position_games')} "
                "AS historical_modal_position_share"
            ),
            f"({_distinct_positions_sql(career_counts)})::BIGINT AS historical_distinct_positions",
            "s.previous_explicit_position",
            (
                f"{_days_since_sql('s.previous_explicit_start_time', 's.start_time')} "
                "AS days_since_previous_explicit_position"
            ),
            (
                "CASE s.previous_explicit_position "
                + " ".join(
                    f"WHEN '{label}' THEN s.prior_games_{_position_suffix(label)}"
                    for label in EXPLICIT_POSITION_LABELS
                )
                + " ELSE NULL END AS prior_games_same_as_previous_position"
            ),
        ]
    )

    for window in RECENT_POSITION_WINDOWS:
        count_aliases = {
            label: f"recent_{window}_games_{_position_suffix(label)}"
            for label in EXPLICIT_POSITION_LABELS
        }
        count_exprs = {
            label: _list_count_eq_sql(f"s.recent_{window}_positions", label)
            for label in EXPLICIT_POSITION_LABELS
        }
        for label in EXPLICIT_POSITION_LABELS:
            outer_metrics.append(
                f"{count_exprs[label]} AS {count_aliases[label]}"
            )
        # Repeat count expressions; DuckDB cannot reuse SELECT aliases here.
        explicit_repeat = " + ".join(
            count_exprs[label] for label in EXPLICIT_POSITION_LABELS
        )
        modal_from_exprs = _unique_mode_sql(count_exprs)
        modal_games = _modal_games_sql(count_exprs)
        outer_metrics.append(
            f"({explicit_repeat})::BIGINT AS recent_{window}_explicit_games"
        )
        outer_metrics.append(
            f"{modal_from_exprs} AS recent_{window}_modal_position"
        )
        outer_metrics.append(
            f"{_rate_sql(modal_games, f'({explicit_repeat})')} "
            f"AS recent_{window}_modal_position_share"
        )
        outer_metrics.append(
            f"({_distinct_positions_sql(count_exprs)})::BIGINT "
            f"AS recent_{window}_distinct_positions"
        )
        if window == RECENT_STABILITY_WINDOW:
            stability_sql = (
                f"{_rate_sql(modal_games, f'({explicit_repeat})')} "
                "AS recent_position_stability"
            )

    identity = (
        "s.match_id,\n"
        "    s.player_id,\n"
        "    s.start_time,\n"
        "    s.game_version_id,\n"
        "    s.series_id,\n"
        "    s.team_id,\n"
        "    s.side,\n"
        "    s.hero_id,\n"
        "    s.won,\n"
        "    s.slot_in_side,\n"
        "    s.position,\n"
        "    s.lane,\n"
        "    s.role,\n"
        "    s.prior_games,\n"
        "    s.prior_explicit_position_games"
    )
    metrics_sql = ",\n    ".join(outer_metrics)
    if stability_sql:
        metrics_sql = metrics_sql + ",\n    " + stability_sql

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
        {_windowed_select_sql()}
    FROM player_match
    WINDOW
        w_player AS ({career_window}),
        w_version AS ({version_window})
),

sliced AS (
    SELECT
        {slice_cols}
    FROM windowed AS w
)

SELECT
    {identity},
    {metrics_sql}
FROM sliced AS s
{output_filter}
"""


@dataclass(frozen=True)
class PlayerPositionState:
    """Lazy `(match_id, player_id)` observed-position historical state.

    Nothing is materialized until `to_frame` is called. The owning
    `FeatureDuckDBConnection` must stay open for that call.
    """

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per player-match in contract column order."""
        frame = self.relation.df()
        ordered = frame[list(PLAYER_POSITION_STATE_COLUMNS)]
        return ordered.sort_values(
            [MATCH_ID_COLUMN, PLAYER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)


def build_player_position_state(
    store: FeatureDuckDBConnection, *, match_id: int | None = None
) -> PlayerPositionState:
    """Build leakage-safe player × position state from analytical views.

    Independent of PRE_DRAFT snapshot SQL, Elo, and Player × Hero.
    Optional `match_id` filters output after windows run over full history.
    """
    return PlayerPositionState(
        relation=store.sql(player_position_state_sql(match_id=match_id))
    )
