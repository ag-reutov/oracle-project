"""Chronological eligibility for Step 3 snapshots.

Enforces `.cursor/rules/ml.mdc`'s strict temporal-integrity rule at the
feature layer: historical information used to predict a match at
`current_start_time` must satisfy `historical_start_time < current_start_time`,
using `start_time` -- never `match_id` -- as the temporal boundary.
`match_id` is a stable identity/ordering key (see the Step 2 canonical
export), not a proxy for time; two matches ordered by ascending
`match_id` are not guaranteed to be ordered by `start_time`.

The comparison is a strict `<`, not `<=`: a match with a `start_time`
equal to the current match's `start_time` is not treated as historical,
so a tie can never be (mis)used as "already known" information.
"""

from __future__ import annotations

from datetime import datetime

__all__ = [
    "HISTORICAL_START_TIME_SQL_CONDITION",
    "STRICT_PRIOR_RANGE_SQL",
    "is_historical",
    "strict_prior_window_sql",
]

# Reusable SQL fragment for building historical-eligibility filters over
# the `matches`/`match_players` views, e.g.:
#     HISTORICAL_START_TIME_SQL_CONDITION.format(
#         historical="h", current="c"
#     )
# expands to "h.start_time < c.start_time".
HISTORICAL_START_TIME_SQL_CONDITION = "{historical}.start_time < {current}.start_time"

# Window-frame equivalent of `historical.start_time < current.start_time`.
# RANGE through CURRENT ROW, then EXCLUDE GROUP so the current row and
# every peer with the same ORDER BY value (start_time) are omitted.
# This is the convention used by player_hero / team_hero / hero_meta.
STRICT_PRIOR_RANGE_SQL = (
    "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP"
)


def _require_sql_identifier(name: str) -> str:
    """Accept a single unquoted SQL identifier; reject anything else."""
    if not name.isidentifier():
        raise ValueError(f"not a safe SQL identifier: {name!r}")
    return name


def strict_prior_window_sql(
    *partition_columns: str, order_column: str = "start_time"
) -> str:
    """DuckDB WINDOW body implementing strict-prior eligibility.

    ``PARTITION BY <columns> ORDER BY start_time RANGE ... EXCLUDE GROUP``.
    Callers choose the partition (player_id, player_id × hero_id,
    player_id × game_version_id, later player_id × position, ...).
    History is never ordered by match_id.
    """
    if not partition_columns:
        raise ValueError("strict_prior_window_sql requires at least one partition column")
    partitions = ", ".join(_require_sql_identifier(column) for column in partition_columns)
    order = _require_sql_identifier(order_column)
    return f"PARTITION BY {partitions} ORDER BY {order} {STRICT_PRIOR_RANGE_SQL}"


def is_historical(
    *, historical_start_time: datetime, current_start_time: datetime
) -> bool:
    """Whether `historical_start_time` is strictly before `current_start_time`.

    Returns `False` for equal timestamps -- a tie is not historical
    information (see module docstring).
    """
    return historical_start_time < current_start_time
