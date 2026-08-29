"""Historical team Elo PRE_DRAFT feature layer (Step 3C).

Adds exactly three predictive features on top of the Step 3B
`PreDraftSnapshot` contract:

* `radiant_team_elo` / `dire_team_elo`: each team's Elo rating as it
  existed strictly BEFORE the current match's outcome.
* `team_elo_delta`: `radiant_team_elo - dire_team_elo`.

Algorithm
---------
Standard Elo, keyed by canonical `team_id` (side-independent):

* every team starts at `EloConfig.initial_rating` (default 1500.0) the
  first time it is seen;
* expected score for a team rated `r_a` against an opponent rated
  `r_b` is `1 / (1 + 10 ** ((r_b - r_a) / 400))` (`expected_score`);
* after a match, a team's rating moves by
  `k_factor * (actual_score - expected_score)`, where `actual_score`
  is 1.0 for a win and 0.0 for a loss;
* updates happen once per canonical map result (one row of the
  `matches` view == one game), never per series.

Temporal integrity
-------------------
Matches are processed chronologically by `start_time`
(`.cursor/rules/ml.mdc`, `features.temporal`), grouped into "temporal
groups" of equal `start_time`. Within `compute_team_elo_features`:

1. For every match in a temporal group, the Elo feature snapshot
   (`radiant_team_elo`/`dire_team_elo`/`team_elo_delta`) is read from
   the rating state that existed BEFORE that group -- never from a
   rating already updated by another match in the same group.
2. Only once every row's snapshot for that group has been computed do
   the group's outcomes get applied to the rating state, as one batch
   of independent per-team deltas (see `_apply_group_updates`). This
   makes same-timestamp matches mutually blind to each other by
   construction: two matches sharing a `start_time` read the exact
   same pre-group rating for a team that appears in both, regardless
   of `match_id` or row order.
3. `match_id` is never used for ordering here, only `start_time`
   (ties are broken by nothing at all -- see point 2, ties simply
   never influence each other) -- this is required for
   `match_id`-permutation invariance within a temporal group.
4. A match's own `radiant_win` is read only to compute the *delta*
   applied to historical state for future matches; it is never mixed
   into that same match's own snapshot row (the snapshot for a row is
   always computed before that row's delta is even calculated).

This is intentionally a plain, sequential, single-pass Python loop
over a materialized `pandas.DataFrame`, not a SQL/window-function
formulation: unlike the Step 3B aggregate features (independent counts
that can be expressed as `GROUP BY`/`JOIN`), each Elo update depends
non-linearly on the immediately preceding rating state, i.e. it is a
genuinely sequential recurrence. Forcing that into SQL window
functions would trade transparency for premature optimization on a
dataset size where a Python loop is already fast enough -- see the
Step 3C task scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "DEFAULT_ELO_CONFIG",
    "DIRE_TEAM_ELO_COLUMN",
    "MATCH_ID_COLUMN",
    "RADIANT_TEAM_ELO_COLUMN",
    "TEAM_ELO_DELTA_COLUMN",
    "TEAM_ELO_FEATURE_COLUMNS",
    "EloConfig",
    "InvalidTeamIdError",
    "compute_team_elo_features",
    "expected_score",
]


class InvalidTeamIdError(ValueError):
    """Raised when a team id required to compute Elo is missing or invalid.

    Fails loudly instead of silently substituting a shared placeholder
    identity for missing/invalid team ids -- an invented shared
    pseudo-team would corrupt the rating history of every real team
    that happened to face it.
    """


@dataclass(frozen=True)
class EloConfig:
    """Elo hyperparameters. Step 3C is deliberately not tuned -- these
    are the standard textbook defaults, kept small and explicit rather
    than baked into the algorithm."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0


DEFAULT_ELO_CONFIG = EloConfig()

RADIANT_TEAM_ELO_COLUMN = "radiant_team_elo"
DIRE_TEAM_ELO_COLUMN = "dire_team_elo"
TEAM_ELO_DELTA_COLUMN = "team_elo_delta"

TEAM_ELO_FEATURE_COLUMNS: tuple[str, ...] = (
    RADIANT_TEAM_ELO_COLUMN,
    DIRE_TEAM_ELO_COLUMN,
    TEAM_ELO_DELTA_COLUMN,
)

MATCH_ID_COLUMN = "match_id"
_START_TIME_COLUMN = "start_time"
_RADIANT_TEAM_ID_COLUMN = "radiant_team_id"
_DIRE_TEAM_ID_COLUMN = "dire_team_id"
_RADIANT_WIN_COLUMN = "radiant_win"

_REQUIRED_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    _START_TIME_COLUMN,
    _RADIANT_TEAM_ID_COLUMN,
    _DIRE_TEAM_ID_COLUMN,
    _RADIANT_WIN_COLUMN,
)


def expected_score(rating: float, opponent_rating: float) -> float:
    """Standard Elo expected-score formula: the probability a team
    rated `rating` is expected to beat an opponent rated
    `opponent_rating`."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _validate_team_id(match_id: int, column: str, value: object) -> int:
    if _is_missing(value):
        raise InvalidTeamIdError(
            f"match_id={match_id}: {column} is missing (null); team Elo "
            "requires a real team identity for every match and never "
            "substitutes a shared placeholder team."
        )
    team_id = int(value)  # type: ignore[arg-type]
    if team_id <= 0:
        raise InvalidTeamIdError(
            f"match_id={match_id}: {column}={team_id!r} is not a valid "
            "positive team id."
        )
    return team_id


def _validate_radiant_win(match_id: int, value: object) -> bool:
    if _is_missing(value):
        raise InvalidTeamIdError(
            f"match_id={match_id}: radiant_win is missing (null); Elo "
            "cannot update historical state for a match with an unknown "
            "outcome."
        )
    return bool(value)


def _apply_group_updates(
    ratings: dict[int, float],
    pending_delta: dict[int, float],
    *,
    initial_rating: float,
) -> None:
    """Apply one temporal group's accumulated per-team rating deltas at
    once, using each team's pre-group rating as the base.

    Deltas are summed by `team_id` (see `compute_team_elo_features`)
    before this is called, so this function is the single point where
    same-`start_time` matches actually change `ratings` -- and it only
    reads `ratings` (never `pending_delta` keys as if they were
    already-updated ratings), which is what keeps every match within
    the group blind to every other.
    """
    for team_id, delta in pending_delta.items():
        ratings[team_id] = ratings.get(team_id, initial_rating) + delta


def compute_team_elo_features(
    matches: pd.DataFrame, *, config: EloConfig = DEFAULT_ELO_CONFIG
) -> pd.DataFrame:
    """One row per input match: `match_id` plus `TEAM_ELO_FEATURE_COLUMNS`.

    `matches` must contain `MATCH_ID_COLUMN`, `start_time`,
    `radiant_team_id`, `dire_team_id`, and `radiant_win` (extra columns
    are ignored). Rows are processed chronologically by `start_time`;
    see the module docstring for the exact temporal-grouping algorithm.
    `radiant_win` is read only to update state for later matches -- it
    never affects the snapshot computed for its own row.

    Raises `InvalidTeamIdError` if any `radiant_team_id`/`dire_team_id`
    is missing or not a positive integer (see that class's docstring
    for why this fails loudly instead of substituting a placeholder).
    """
    missing_columns = [c for c in _REQUIRED_COLUMNS if c not in matches.columns]
    if missing_columns:
        raise ValueError(
            f"matches frame is missing required columns: {missing_columns}"
        )

    ordered = matches[list(_REQUIRED_COLUMNS)].sort_values(
        _START_TIME_COLUMN, kind="stable"
    )

    ratings: dict[int, float] = {}
    snapshot_rows: list[dict[str, object]] = []

    for _, group in ordered.groupby(_START_TIME_COLUMN, sort=False):
        pending_delta: dict[int, float] = {}

        for record in group.itertuples(index=False):
            match_id = int(record.match_id)
            radiant_team_id = _validate_team_id(
                match_id, _RADIANT_TEAM_ID_COLUMN, record.radiant_team_id
            )
            dire_team_id = _validate_team_id(
                match_id, _DIRE_TEAM_ID_COLUMN, record.dire_team_id
            )
            radiant_win = _validate_radiant_win(match_id, record.radiant_win)

            radiant_rating = ratings.get(radiant_team_id, config.initial_rating)
            dire_rating = ratings.get(dire_team_id, config.initial_rating)

            snapshot_rows.append(
                {
                    MATCH_ID_COLUMN: match_id,
                    RADIANT_TEAM_ELO_COLUMN: radiant_rating,
                    DIRE_TEAM_ELO_COLUMN: dire_rating,
                    TEAM_ELO_DELTA_COLUMN: radiant_rating - dire_rating,
                }
            )

            actual_radiant = 1.0 if radiant_win else 0.0
            radiant_change = config.k_factor * (
                actual_radiant - expected_score(radiant_rating, dire_rating)
            )
            pending_delta[radiant_team_id] = (
                pending_delta.get(radiant_team_id, 0.0) + radiant_change
            )
            pending_delta[dire_team_id] = (
                pending_delta.get(dire_team_id, 0.0) - radiant_change
            )

        _apply_group_updates(
            ratings, pending_delta, initial_rating=config.initial_rating
        )

    return pd.DataFrame(
        snapshot_rows, columns=[MATCH_ID_COLUMN, *TEAM_ELO_FEATURE_COLUMNS]
    )
