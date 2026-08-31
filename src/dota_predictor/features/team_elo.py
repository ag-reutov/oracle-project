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

Elo *state* (the per-team rating after the last processed match, plus
career counts) is not stored separately: it is the same sequential
replay's terminal `ratings` dict. `compute_team_elo_state` exposes that
terminal state without changing the update rule. Feature snapshots
remain pre-match; the latest rating for a team is the post-update
value after its most recent temporal group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = [
    "DEFAULT_ACTIVE_DAYS",
    "DEFAULT_ELO_CONFIG",
    "DIRE_TEAM_ELO_COLUMN",
    "MATCH_ID_COLUMN",
    "RADIANT_TEAM_ELO_COLUMN",
    "TEAM_ELO_DELTA_COLUMN",
    "TEAM_ELO_FEATURE_COLUMNS",
    "TEAM_ELO_STATE_COLUMNS",
    "EloConfig",
    "InvalidTeamIdError",
    "active_team_elo_cutoff",
    "compute_team_elo_features",
    "compute_team_elo_state",
    "expected_score",
    "filter_active_team_elo",
    "rank_team_elo_state",
    "team_elo_trajectories",
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

TEAM_ID_COLUMN = "team_id"
ELO_COLUMN = "elo"
N_MATCHES_COLUMN = "n_matches"
WINS_COLUMN = "wins"
LOSSES_COLUMN = "losses"
LAST_MATCH_TIMESTAMP_COLUMN = "last_match_timestamp"
LAST_MATCH_ID_COLUMN = "last_match_id"
ELO_BEFORE_LAST_MATCH_COLUMN = "elo_before_last_match"
ELO_AFTER_LAST_MATCH_COLUMN = "elo_after_last_match"
STARTING_ELO_COLUMN = "starting_elo"
PEAK_ELO_COLUMN = "peak_elo"
LOWEST_ELO_COLUMN = "lowest_elo"
PEAK_AFTER_MATCH_ID_COLUMN = "peak_after_match_id"
PEAK_AFTER_MATCH_TIMESTAMP_COLUMN = "peak_after_match_timestamp"
LAST_GROUP_N_MATCHES_COLUMN = "last_group_n_matches"

TEAM_ELO_STATE_COLUMNS: tuple[str, ...] = (
    TEAM_ID_COLUMN,
    ELO_COLUMN,
    N_MATCHES_COLUMN,
    WINS_COLUMN,
    LOSSES_COLUMN,
    LAST_MATCH_TIMESTAMP_COLUMN,
    LAST_MATCH_ID_COLUMN,
    ELO_BEFORE_LAST_MATCH_COLUMN,
    ELO_AFTER_LAST_MATCH_COLUMN,
    STARTING_ELO_COLUMN,
    PEAK_ELO_COLUMN,
    LOWEST_ELO_COLUMN,
    PEAK_AFTER_MATCH_ID_COLUMN,
    PEAK_AFTER_MATCH_TIMESTAMP_COLUMN,
    LAST_GROUP_N_MATCHES_COLUMN,
)

DEFAULT_ACTIVE_DAYS = 90
CURRENT_ELO_COLUMN = "current_elo"

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


@dataclass
class _TeamTracker:
    """Mutable per-team bookkeeping accumulated during a sequential replay.

    Not part of the public API: `compute_team_elo_state` projects this
    into a DataFrame. Peak/lowest start at `starting_elo` (the rating
    before any match) so a team that only loses still has a defined peak
    at the initial rating, with `peak_after_match_id` left as None.
    """

    starting_elo: float
    n_matches: int = 0
    wins: int = 0
    losses: int = 0
    last_match_id: int | None = None
    last_match_timestamp: object = None
    elo_before_last_match: float = 0.0
    peak_elo: float = 0.0
    lowest_elo: float = 0.0
    peak_after_match_id: int | None = None
    peak_after_match_timestamp: object = None
    last_group_n_matches: int = 0

    def __post_init__(self) -> None:
        self.elo_before_last_match = self.starting_elo
        self.peak_elo = self.starting_elo
        self.lowest_elo = self.starting_elo


@dataclass
class _EloReplay:
    snapshots: list[dict[str, object]]
    ratings: dict[int, float]
    trackers: dict[int, _TeamTracker]
    config: EloConfig = field(repr=False)


def _require_match_columns(matches: pd.DataFrame) -> None:
    missing_columns = [c for c in _REQUIRED_COLUMNS if c not in matches.columns]
    if missing_columns:
        raise ValueError(
            f"matches frame is missing required columns: {missing_columns}"
        )


def _ensure_tracker(
    trackers: dict[int, _TeamTracker], team_id: int, *, starting_elo: float
) -> _TeamTracker:
    tracker = trackers.get(team_id)
    if tracker is None:
        tracker = _TeamTracker(starting_elo=starting_elo)
        trackers[team_id] = tracker
    return tracker


def _record_appearance(
    tracker: _TeamTracker, *, won: bool
) -> None:
    tracker.n_matches += 1
    if won:
        tracker.wins += 1
    else:
        tracker.losses += 1


def _replay_elo(
    matches: pd.DataFrame, *, config: EloConfig
) -> _EloReplay:
    """Single sequential pass used by both feature snapshots and state.

    Snapshot rows and rating updates are computed identically to the
    original `compute_team_elo_features` loop; trackers only observe
    those updates and never feed back into `ratings`.
    """
    _require_match_columns(matches)

    ordered = matches[list(_REQUIRED_COLUMNS)].sort_values(
        _START_TIME_COLUMN, kind="stable"
    )

    ratings: dict[int, float] = {}
    trackers: dict[int, _TeamTracker] = {}
    snapshot_rows: list[dict[str, object]] = []

    for start_time, group in ordered.groupby(_START_TIME_COLUMN, sort=False):
        pending_delta: dict[int, float] = {}
        group_match_ids: dict[int, list[int]] = {}
        group_pre_rating: dict[int, float] = {}

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

            group_pre_rating.setdefault(radiant_team_id, radiant_rating)
            group_pre_rating.setdefault(dire_team_id, dire_rating)
            group_match_ids.setdefault(radiant_team_id, []).append(match_id)
            group_match_ids.setdefault(dire_team_id, []).append(match_id)

            _record_appearance(
                _ensure_tracker(
                    trackers, radiant_team_id, starting_elo=config.initial_rating
                ),
                won=radiant_win,
            )
            _record_appearance(
                _ensure_tracker(
                    trackers, dire_team_id, starting_elo=config.initial_rating
                ),
                won=not radiant_win,
            )

        _apply_group_updates(
            ratings, pending_delta, initial_rating=config.initial_rating
        )

        for team_id in pending_delta:
            tracker = trackers[team_id]
            match_ids = group_match_ids[team_id]
            tracker.last_match_id = match_ids[-1]
            tracker.last_match_timestamp = start_time
            tracker.elo_before_last_match = group_pre_rating[team_id]
            tracker.last_group_n_matches = len(match_ids)
            new_elo = ratings[team_id]
            if new_elo > tracker.peak_elo:
                tracker.peak_elo = new_elo
                tracker.peak_after_match_id = match_ids[-1]
                tracker.peak_after_match_timestamp = start_time
            tracker.lowest_elo = min(tracker.lowest_elo, new_elo)

    return _EloReplay(
        snapshots=snapshot_rows,
        ratings=ratings,
        trackers=trackers,
        config=config,
    )


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
    replay = _replay_elo(matches, config=config)
    return pd.DataFrame(
        replay.snapshots, columns=[MATCH_ID_COLUMN, *TEAM_ELO_FEATURE_COLUMNS]
    )


def compute_team_elo_state(
    matches: pd.DataFrame, *, config: EloConfig = DEFAULT_ELO_CONFIG
) -> pd.DataFrame:
    """One row per team after replaying `matches` with the production Elo rule.

    Ratings, snapshots, and group-batch updates are produced by the same
    `_replay_elo` pass as `compute_team_elo_features`; this function only
    projects the terminal per-team bookkeeping. Rows are ordered by
    `team_id` (use `rank_team_elo_state` for a leaderboard). Extra input
    columns are ignored.

    `elo_before_last_match` is the pre-group rating the team carried into
    its most recent temporal group. If that group contained more than one
    match involving the team, `last_group_n_matches` is > 1 and
    `elo_after_last_match` reflects the batched update from every match
    in the group, not a sequential compound of those matches.
    """
    replay = _replay_elo(matches, config=config)
    rows: list[dict[str, object]] = []
    for team_id in sorted(replay.trackers):
        tracker = replay.trackers[team_id]
        elo = replay.ratings[team_id]
        rows.append(
            {
                TEAM_ID_COLUMN: team_id,
                ELO_COLUMN: elo,
                N_MATCHES_COLUMN: tracker.n_matches,
                WINS_COLUMN: tracker.wins,
                LOSSES_COLUMN: tracker.losses,
                LAST_MATCH_TIMESTAMP_COLUMN: tracker.last_match_timestamp,
                LAST_MATCH_ID_COLUMN: tracker.last_match_id,
                ELO_BEFORE_LAST_MATCH_COLUMN: tracker.elo_before_last_match,
                ELO_AFTER_LAST_MATCH_COLUMN: elo,
                STARTING_ELO_COLUMN: tracker.starting_elo,
                PEAK_ELO_COLUMN: tracker.peak_elo,
                LOWEST_ELO_COLUMN: tracker.lowest_elo,
                PEAK_AFTER_MATCH_ID_COLUMN: tracker.peak_after_match_id,
                PEAK_AFTER_MATCH_TIMESTAMP_COLUMN: tracker.peak_after_match_timestamp,
                LAST_GROUP_N_MATCHES_COLUMN: tracker.last_group_n_matches,
            }
        )
    return pd.DataFrame(rows, columns=list(TEAM_ELO_STATE_COLUMNS))


def rank_team_elo_state(state: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `state` sorted by Elo descending, `team_id` ascending.

    Does not mutate `state`. Equal ratings keep a deterministic order by
    `team_id` rather than an arbitrary row position.
    """
    if ELO_COLUMN not in state.columns or TEAM_ID_COLUMN not in state.columns:
        raise ValueError(
            f"state frame is missing required columns: "
            f"{[c for c in (ELO_COLUMN, TEAM_ID_COLUMN) if c not in state.columns]}"
        )
    ranked = state.sort_values(
        [ELO_COLUMN, TEAM_ID_COLUMN],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked


def active_team_elo_cutoff(
    dataset_max_timestamp: object, *, active_days: int = DEFAULT_ACTIVE_DAYS
) -> pd.Timestamp:
    """Inclusive lower bound for "played within the last `active_days` days
    of `dataset_max_timestamp`".

    Uses the dataset clock, never wall-clock `now`. `active_days=0`
    keeps only teams whose last match timestamp equals the dataset max.
    """
    if active_days < 0:
        raise ValueError(f"active_days must be >= 0, got {active_days}")
    return pd.Timestamp(dataset_max_timestamp) - pd.Timedelta(days=active_days)


def filter_active_team_elo(
    state: pd.DataFrame,
    *,
    dataset_max_timestamp: object,
    active_days: int = DEFAULT_ACTIVE_DAYS,
) -> pd.DataFrame:
    """Teams whose last rated match is within `active_days` of the dataset max.

    A team is active when
    `last_match_timestamp >= dataset_max_timestamp - active_days`.
    The cutoff is computed from `dataset_max_timestamp`, never from
    wall-clock time. Returns a copy; does not re-rank.
    """
    if LAST_MATCH_TIMESTAMP_COLUMN not in state.columns:
        raise ValueError(
            f"state frame is missing required column: {LAST_MATCH_TIMESTAMP_COLUMN!r}"
        )
    cutoff = active_team_elo_cutoff(
        dataset_max_timestamp, active_days=active_days
    )
    timestamps = pd.to_datetime(state[LAST_MATCH_TIMESTAMP_COLUMN], utc=True)
    cutoff_utc = pd.Timestamp(cutoff)
    if cutoff_utc.tzinfo is None:
        cutoff_utc = cutoff_utc.tz_localize("UTC")
    else:
        cutoff_utc = cutoff_utc.tz_convert("UTC")
    return state.loc[timestamps >= cutoff_utc].copy()


_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    TEAM_ID_COLUMN,
    STARTING_ELO_COLUMN,
    ELO_COLUMN,
    PEAK_ELO_COLUMN,
    LOWEST_ELO_COLUMN,
    N_MATCHES_COLUMN,
    PEAK_AFTER_MATCH_ID_COLUMN,
    PEAK_AFTER_MATCH_TIMESTAMP_COLUMN,
)


def team_elo_trajectories(
    state: pd.DataFrame, *, n: int | None = None
) -> pd.DataFrame:
    """Trajectory fields projected from already-computed `state` rows.

    `current_elo` is a rename of `elo` -- the same values the leaderboard
    ranks on, not a second replay or an independent reconstruction.
    Optional display columns such as `team_name` are copied through when
    present. Does not mutate `state`.
    """
    missing = [c for c in (TEAM_ID_COLUMN, ELO_COLUMN) if c not in state.columns]
    if missing:
        raise ValueError(f"state frame is missing required columns: {missing}")

    ordered: list[str] = [TEAM_ID_COLUMN]
    if "team_name" in state.columns:
        ordered.append("team_name")
    ordered.extend(c for c in _TRAJECTORY_COLUMNS if c != TEAM_ID_COLUMN and c in state.columns)

    frame = state.loc[:, ordered]
    if n is not None:
        frame = frame.head(n)
    return frame.rename(columns={ELO_COLUMN: CURRENT_ELO_COLUMN}).reset_index(drop=True)
