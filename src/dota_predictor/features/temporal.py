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

__all__ = ["HISTORICAL_START_TIME_SQL_CONDITION", "is_historical"]

# Reusable SQL fragment for building historical-eligibility filters over
# the `matches`/`match_players` views, e.g.:
#     HISTORICAL_START_TIME_SQL_CONDITION.format(
#         historical="h", current="c"
#     )
# expands to "h.start_time < c.start_time".
HISTORICAL_START_TIME_SQL_CONDITION = "{historical}.start_time < {current}.start_time"


def is_historical(
    *, historical_start_time: datetime, current_start_time: datetime
) -> bool:
    """Whether `historical_start_time` is strictly before `current_start_time`.

    Returns `False` for equal timestamps -- a tie is not historical
    information (see module docstring).
    """
    return historical_start_time < current_start_time
