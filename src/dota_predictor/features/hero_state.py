"""Leakage-safe expanding hero meta state (Slice 5).

Grain
-----
One row per `(match_id, hero_id)`: what was known about that hero
**strictly before** the current match. The hero universe and dense
`(match, hero)` expansion match `features.hero_meta`: catalog heroes
when `heroes` is registered, otherwise distinct observed `hero_id`
values. This is historical state, not a training feature and not a
composite meta-strength score.

This module does not replace `features.hero_meta`. That earlier
same-version / 90-day experiment stays untouched. Slice 5's **main**
state is expanding career history plus trailing match-count windows.
Same-version counters are diagnostic columns only and do not reset
the expanding/recent state at patch boundaries.

Temporal integrity
------------------
Every historical match `H` contributing to current match `M` satisfies
`H.start_time < M.start_time` via `RANGE ... CURRENT ROW EXCLUDE GROUP`.
Equal timestamps are mutually blind. History is never ordered by
`match_id`. The current match's own draft, result, and observed
positions never enter that row's metrics.

Windows
-------
* Career (expanding, any game version): all strictly earlier matches.
* Recent trailing matches: last 20 / 50 / 100 *strictly prior* matches
  on the dense grid (`ROWS ... EXCLUDE GROUP`). Every prior match
  occupies a slot, whether or not the hero was picked.
* Same STRATZ game version: diagnostic only.

Denominators
------------
Pick / ban / contest **rates** use eligible historical matches
(`hero_prior_matches` / the matching window match count), not only
matches where the hero appeared.

Win **rate** uses historical **picks** as the denominator. Zero picks
→ NULL, never 0 or 0.5.

Position **shares** use historical matches where the hero was observed
in an explicit STRATZ POSITION_1–5. NULL / UNKNOWN / FILTERED / ALL
do not increment any position count. Slice 3 `expected_position` is
not used. Historical rows may use their own observed position because
those matches are already in the past.

Draft semantics
---------------
Pick / ban / contest use canonical `draft_events` at match grain (at
most one pick, one ban, one contest per historical match). Successful
canonical actions only: picks always; bans unless `was_successful is
False` (`DraftEvent.is_actual`). Unsuccessful ban attempts are neither
bans nor contests. Win/loss uses `match_players` played side +
`matches.radiant_win`, never draft order.

Rates are raw floating-point ratios. No smoothing, shrinkage, or
regularization.

Availability
------------
Catalog `hero_id` and every aggregate derived only from strictly
earlier matches/drafts are PRE_DRAFT historical state for the current
match. This relation does not expose the current match's drafted
heroes, result, or observed positions. Current-match hero identity in
`match_players` / `draft_events` remains DRAFT.

This module never writes Parquet, never bumps schema versions, and is
not part of the training feature matrix.
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
from dota_predictor.features.player_position import EXPLICIT_POSITION_LABELS
from dota_predictor.features.temporal import STRICT_PRIOR_RANGE_SQL

__all__ = [
    "HERO_STATE_COLUMNS",
    "HERO_STATE_IDENTITY_COLUMNS",
    "HERO_STATE_METRIC_COLUMNS",
    "RECENT_HERO_MATCH_WINDOWS",
    "HeroState",
    "build_hero_state",
    "hero_state_sql",
    "summarize_hero_state",
]

MATCH_ID_COLUMN = "match_id"
HERO_ID_COLUMN = "hero_id"
HERO_NAME_COLUMN = "hero_name"
_MICROSECONDS_PER_DAY = 86_400_000_000.0

RECENT_HERO_MATCH_WINDOWS: tuple[int, ...] = (20, 50, 100)

POSITION_NUMBERS: tuple[int, ...] = (1, 2, 3, 4, 5)

HERO_STATE_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    HERO_ID_COLUMN,
    HERO_NAME_COLUMN,
)


def _position_count_column(position: int, *, prefix: str) -> str:
    return f"{prefix}position_{position}_count"


def _position_share_column(position: int, *, prefix: str) -> str:
    return f"{prefix}position_{position}_share"


def _career_metric_columns() -> tuple[str, ...]:
    columns = [
        "hero_prior_matches",
        "hero_pick_count",
        "hero_ban_count",
        "hero_contest_count",
        "hero_pick_rate",
        "hero_ban_rate",
        "hero_contest_rate",
        "hero_prior_wins",
        "hero_prior_losses",
        "hero_prior_win_rate",
        "hero_days_since_last_pick",
        "hero_position_explicit_count",
    ]
    for position in POSITION_NUMBERS:
        columns.append(_position_count_column(position, prefix="hero_"))
    for position in POSITION_NUMBERS:
        columns.append(_position_share_column(position, prefix="hero_"))
    return tuple(columns)


def _recent_metric_columns(window: int) -> tuple[str, ...]:
    prefix = f"hero_recent_{window}_"
    columns = [
        f"{prefix}matches",
        f"{prefix}pick_count",
        f"{prefix}ban_count",
        f"{prefix}contest_count",
        f"{prefix}pick_rate",
        f"{prefix}ban_rate",
        f"{prefix}contest_rate",
        f"{prefix}wins",
        f"{prefix}win_rate",
        f"{prefix}position_explicit_count",
    ]
    for position in POSITION_NUMBERS:
        columns.append(_position_count_column(position, prefix=prefix))
    for position in POSITION_NUMBERS:
        columns.append(_position_share_column(position, prefix=prefix))
    return tuple(columns)


def _same_version_metric_columns() -> tuple[str, ...]:
    prefix = "hero_same_version_"
    columns = [
        f"{prefix}prior_matches",
        f"{prefix}pick_count",
        f"{prefix}ban_count",
        f"{prefix}contest_count",
        f"{prefix}pick_rate",
        f"{prefix}ban_rate",
        f"{prefix}contest_rate",
        f"{prefix}prior_wins",
        f"{prefix}win_rate",
        f"{prefix}position_explicit_count",
    ]
    for position in POSITION_NUMBERS:
        columns.append(_position_count_column(position, prefix=prefix))
    for position in POSITION_NUMBERS:
        columns.append(_position_share_column(position, prefix=prefix))
    return tuple(columns)


HERO_STATE_METRIC_COLUMNS: tuple[str, ...] = (
    _career_metric_columns()
    + sum((_recent_metric_columns(window) for window in RECENT_HERO_MATCH_WINDOWS), ())
    + _same_version_metric_columns()
)

HERO_STATE_COLUMNS: tuple[str, ...] = (
    HERO_STATE_IDENTITY_COLUMNS + HERO_STATE_METRIC_COLUMNS
)


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


def _recent_rows_range_sql(window: int) -> str:
    return (
        f"ROWS BETWEEN {int(window)} PRECEDING AND CURRENT ROW EXCLUDE GROUP"
    )


def _sum_over(column: str, window: str, alias: str) -> str:
    return f"COALESCE(SUM({column}) OVER {window}, 0)::BIGINT AS {alias}"


def _count_over(window: str, alias: str) -> str:
    return f"COALESCE(COUNT(*) OVER {window}, 0)::BIGINT AS {alias}"


def _position_flag_sql() -> str:
    parts = [
        (
            f"CASE WHEN f.position = '{label}' THEN 1 ELSE 0 END "
            f"AS at_position_{index}"
        )
        for index, label in enumerate(EXPLICIT_POSITION_LABELS, start=1)
    ]
    return ",\n        ".join(parts)


def _windowed_aggregates_sql() -> str:
    parts = [
        "match_id",
        "start_time",
        "game_version_id",
        "hero_id",
        "hero_name",
        _count_over("w_career", "hero_prior_matches"),
        _sum_over("was_picked", "w_career", "hero_pick_count"),
        _sum_over("was_banned", "w_career", "hero_ban_count"),
        _sum_over("was_contested", "w_career", "hero_contest_count"),
        _sum_over("was_win", "w_career", "hero_prior_wins"),
        _sum_over("was_loss", "w_career", "hero_prior_losses"),
        (
            "MAX(CASE WHEN was_picked = 1 THEN start_time END) "
            "OVER w_career AS last_pick_at"
        ),
        _count_over("w_sv", "hero_same_version_prior_matches"),
        _sum_over("was_picked", "w_sv", "hero_same_version_pick_count"),
        _sum_over("was_banned", "w_sv", "hero_same_version_ban_count"),
        _sum_over("was_contested", "w_sv", "hero_same_version_contest_count"),
        _sum_over("was_win", "w_sv", "hero_same_version_prior_wins"),
    ]
    for position in POSITION_NUMBERS:
        parts.append(
            _sum_over(
                f"at_position_{position}",
                "w_career",
                _position_count_column(position, prefix="hero_"),
            )
        )
        parts.append(
            _sum_over(
                f"at_position_{position}",
                "w_sv",
                _position_count_column(position, prefix="hero_same_version_"),
            )
        )
    for window in RECENT_HERO_MATCH_WINDOWS:
        alias = f"w_{window}"
        prefix = f"hero_recent_{window}_"
        parts.extend(
            [
                _count_over(alias, f"{prefix}matches"),
                _sum_over("was_picked", alias, f"{prefix}pick_count"),
                _sum_over("was_banned", alias, f"{prefix}ban_count"),
                _sum_over("was_contested", alias, f"{prefix}contest_count"),
                _sum_over("was_win", alias, f"{prefix}wins"),
            ]
        )
        for position in POSITION_NUMBERS:
            parts.append(
                _sum_over(
                    f"at_position_{position}",
                    alias,
                    _position_count_column(position, prefix=prefix),
                )
            )
    return ",\n        ".join(parts)


def _explicit_count_sql(prefix: str) -> str:
    addends = " + ".join(
        _position_count_column(position, prefix=prefix) for position in POSITION_NUMBERS
    )
    return f"({addends})"


def _output_metrics_sql() -> str:
    parts = [
        "hero_prior_matches",
        "hero_pick_count",
        "hero_ban_count",
        "hero_contest_count",
        f"{_rate_sql('hero_pick_count', 'hero_prior_matches')} AS hero_pick_rate",
        f"{_rate_sql('hero_ban_count', 'hero_prior_matches')} AS hero_ban_rate",
        f"{_rate_sql('hero_contest_count', 'hero_prior_matches')} AS hero_contest_rate",
        "hero_prior_wins",
        "hero_prior_losses",
        f"{_rate_sql('hero_prior_wins', 'hero_pick_count')} AS hero_prior_win_rate",
        f"{_days_since_sql('last_pick_at', 'start_time')} AS hero_days_since_last_pick",
        (
            f"{_explicit_count_sql('hero_')}::BIGINT "
            "AS hero_position_explicit_count"
        ),
    ]
    explicit = _explicit_count_sql("hero_")
    for position in POSITION_NUMBERS:
        count = _position_count_column(position, prefix="hero_")
        share = _position_share_column(position, prefix="hero_")
        parts.append(count)
        parts.append(f"{_rate_sql(count, explicit)} AS {share}")

    for window in RECENT_HERO_MATCH_WINDOWS:
        prefix = f"hero_recent_{window}_"
        parts.extend(
            [
                f"{prefix}matches",
                f"{prefix}pick_count",
                f"{prefix}ban_count",
                f"{prefix}contest_count",
                (
                    f"{_rate_sql(f'{prefix}pick_count', f'{prefix}matches')} "
                    f"AS {prefix}pick_rate"
                ),
                (
                    f"{_rate_sql(f'{prefix}ban_count', f'{prefix}matches')} "
                    f"AS {prefix}ban_rate"
                ),
                (
                    f"{_rate_sql(f'{prefix}contest_count', f'{prefix}matches')} "
                    f"AS {prefix}contest_rate"
                ),
                f"{prefix}wins",
                (
                    f"{_rate_sql(f'{prefix}wins', f'{prefix}pick_count')} "
                    f"AS {prefix}win_rate"
                ),
            ]
        )
        recent_explicit = _explicit_count_sql(prefix)
        parts.append(
            f"{recent_explicit}::BIGINT AS {prefix}position_explicit_count"
        )
        for position in POSITION_NUMBERS:
            count = _position_count_column(position, prefix=prefix)
            share = _position_share_column(position, prefix=prefix)
            parts.append(count)
            parts.append(f"{_rate_sql(count, recent_explicit)} AS {share}")

    sv = "hero_same_version_"
    parts.extend(
        [
            f"{sv}prior_matches",
            f"{sv}pick_count",
            f"{sv}ban_count",
            f"{sv}contest_count",
            f"{_rate_sql(f'{sv}pick_count', f'{sv}prior_matches')} AS {sv}pick_rate",
            f"{_rate_sql(f'{sv}ban_count', f'{sv}prior_matches')} AS {sv}ban_rate",
            (
                f"{_rate_sql(f'{sv}contest_count', f'{sv}prior_matches')} "
                f"AS {sv}contest_rate"
            ),
            f"{sv}prior_wins",
            f"{_rate_sql(f'{sv}prior_wins', f'{sv}pick_count')} AS {sv}win_rate",
            f"{_explicit_count_sql(sv)}::BIGINT AS {sv}position_explicit_count",
        ]
    )
    sv_explicit = _explicit_count_sql(sv)
    for position in POSITION_NUMBERS:
        count = _position_count_column(position, prefix=sv)
        share = _position_share_column(position, prefix=sv)
        parts.append(count)
        parts.append(f"{_rate_sql(count, sv_explicit)} AS {share}")
    return ",\n    ".join(parts)


def hero_state_sql(
    *, catalog_registered: bool = True, match_id: int | None = None
) -> str:
    """SQL for leakage-safe expanding hero state at `(match_id, hero_id)`.

    Window functions over a dense `(match, hero)` grid implement
    `historical.start_time < current.start_time`. Optional `match_id`
    filters output after windows run over the full ordered history.
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE match_id = {int(match_id)}"

    universe = _hero_universe_sql(catalog_registered=catalog_registered)
    recent_windows = ",\n        ".join(
        f"w_{window} AS (\n"
        f"            PARTITION BY hero_id\n"
        f"            ORDER BY start_time\n"
        f"            {_recent_rows_range_sql(window)}\n"
        f"        )"
        for window in RECENT_HERO_MATCH_WINDOWS
    )

    return f"""
WITH hero_universe AS (
    {universe}
),

successful_draft_actions AS (
    SELECT
        de.match_id,
        de.hero_id,
        de.action
    FROM {DRAFT_EVENTS_VIEW} de
    WHERE de.action = 'PICK'
       OR (de.action = 'BAN' AND de.was_successful IS DISTINCT FROM FALSE)
),

hero_match_actions AS (
    SELECT
        match_id,
        hero_id,
        MAX(CASE WHEN action = 'PICK' THEN 1 ELSE 0 END)::BIGINT AS was_picked,
        MAX(CASE WHEN action = 'BAN' THEN 1 ELSE 0 END)::BIGINT AS was_banned
    FROM successful_draft_actions
    GROUP BY match_id, hero_id
),

hero_match_results AS (
    SELECT
        mp.match_id,
        mp.hero_id,
        mp.position,
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
            AS was_loss,
        r.position
    FROM hero_match_actions a
    LEFT JOIN hero_match_results r
        ON r.match_id = a.match_id AND r.hero_id = a.hero_id
),

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
        COALESCE(f.was_loss, 0) AS was_loss,
        {_position_flag_sql()}
    FROM {MATCHES_VIEW} m
    CROSS JOIN hero_universe u
    LEFT JOIN hero_match_facts f
        ON f.match_id = m.match_id AND f.hero_id = u.hero_id
),

windowed AS (
    SELECT
        {_windowed_aggregates_sql()}
    FROM grid
    WINDOW
        w_career AS (
            PARTITION BY hero_id
            ORDER BY start_time
            {STRICT_PRIOR_RANGE_SQL}
        ),
        w_sv AS (
            PARTITION BY game_version_id, hero_id
            ORDER BY start_time
            {STRICT_PRIOR_RANGE_SQL}
        ),
        {recent_windows}
)

SELECT
    match_id,
    start_time,
    game_version_id,
    hero_id,
    hero_name,
    {_output_metrics_sql()}
FROM windowed
{output_filter}
"""


def _heroes_view_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return HEROES_VIEW in tables


@dataclass(frozen=True)
class HeroState:
    """Lazy `(match_id, hero_id)` expanding hero-meta relation.

    Nothing is materialized until `to_frame` is called. The owning
    `FeatureDuckDBConnection` must stay open for that call.
    """

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per `(match_id, hero_id)` in contract order."""
        frame = self.relation.df()
        ordered = frame[list(HERO_STATE_COLUMNS)]
        return ordered.sort_values(
            [MATCH_ID_COLUMN, HERO_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)


def build_hero_state(
    store: FeatureDuckDBConnection, *, match_id: int | None = None
) -> HeroState:
    """Build leakage-safe expanding hero state from analytical views.

    Independent of PRE_DRAFT snapshot SQL, Elo, and `build_hero_meta`.
    Uses the `heroes` catalog as the universe when that view is
    registered. Optional `match_id` filters output after windows run
    over the full ordered match history.
    """
    sql = hero_state_sql(
        catalog_registered=_heroes_view_registered(store),
        match_id=match_id,
    )
    return HeroState(relation=store.sql(sql))


def _quantile_row(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
    return {
        "n": float(len(clean)),
        "mean": float(clean.mean()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
    }


def summarize_hero_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Coverage and rate/share distributions overall and by game version."""
    rows: list[dict[str, object]] = []

    def _scope_rows(subset: pd.DataFrame, *, scope: str, key: object) -> None:
        n = len(subset)
        picks = subset["hero_pick_count"].fillna(0)
        matches = subset["hero_prior_matches"].fillna(0)
        explicit = subset["hero_position_explicit_count"].fillna(0)
        rows.append(
            {
                "scope": scope,
                "key": key,
                "stat": "coverage",
                "n_rows": n,
                "prior_match_coverage": (
                    float((matches > 0).mean()) if n else float("nan")
                ),
                "prior_pick_coverage": (
                    float((picks > 0).mean()) if n else float("nan")
                ),
                "cold_start_no_picks": (
                    float((picks == 0).mean()) if n else float("nan")
                ),
                "position_evidence_coverage": (
                    float((explicit > 0).mean()) if n else float("nan")
                ),
            }
        )
        for column, label in (
            ("hero_pick_count", "hero_pick_count"),
            ("hero_prior_matches", "hero_prior_matches"),
            ("hero_pick_rate", "hero_pick_rate"),
            ("hero_ban_rate", "hero_ban_rate"),
            ("hero_contest_rate", "hero_contest_rate"),
            ("hero_prior_win_rate", "hero_prior_win_rate"),
            ("hero_position_1_share", "hero_position_1_share"),
            ("hero_position_2_share", "hero_position_2_share"),
            ("hero_position_3_share", "hero_position_3_share"),
            ("hero_position_4_share", "hero_position_4_share"),
            ("hero_position_5_share", "hero_position_5_share"),
        ):
            summary = _quantile_row(subset[column])
            rows.append(
                {
                    "scope": scope,
                    "key": key,
                    "stat": label,
                    "n_rows": int(summary["n"]),
                    **summary,
                }
            )

    _scope_rows(frame, scope="overall", key="all")
    for version, subset in frame.groupby("game_version_id", sort=True):
        _scope_rows(subset, scope="game_version_id", key=version)
    return pd.DataFrame.from_records(rows)
