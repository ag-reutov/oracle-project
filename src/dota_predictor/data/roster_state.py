"""Historical roster state layer (Slice 5).

This module turns the Slice 4 observed-roster-history facts into a
**strictly causal pre-match roster state** at the grain:

    one team in one match, evaluated immediately before that match
    (team_id, match_id, start_time) -> historical roster state

The purpose is descriptive historical-state construction -- "what was
known about each team's current five-player lineup from competitive
observations strictly before that match?" -- never team-strength
modeling, chemistry, synergy, transfer values, or rankings.

The core temporal boundary (`.cursor/rules/ml.mdc`, `features.temporal`)
------------------------------------
For a current match at `current_start_time`, every piece of historical
knowledge contributing to its state must satisfy

    historical_start_time < current_start_time

using `start_time` -- never `match_id` -- as the temporal boundary. The
comparison is strict `<`, not `<=`: a match sharing the current
`start_time` is never treated as historical evidence, so an equal
timestamp can never be (mis)used as "already known" information, even
when its `match_id` sorts first. This is intentionally stricter than
Slice 4's descriptive spell ordering, where `match_id` is a legitimate
presentation tie-breaker.

Future-deletion invariance
--------------------------
Because every derivation below reads only observations strictly before
the current match's `start_time`, deleting all observations after any
time `T` leaves every state at `T` bit-identical. `check_future_deletion_invariant`
verifies this property on any corpus.

What is NOT exposed (see Slice 5 spec section 16)
-------------------------------------------------
No strength/synergy/chemistry scores, no Elo, no win-rate features, no
transfer values, no hero pools, no draft features. A full-corpus spell is
never joined into a pre-match state in a way that leaks its eventual
`last_seen_at`, eventual `observed_match_count`, whether the player later
leaves, or which team they play for next. The only spell-like state here
is causal spell-so-far: `consecutive_prior_team_appearances`, derived
from prior observations only.

Relationship to the pre-draft roster-continuity feature
-------------------------------------------------------
`features.pre_draft_snapshot.ROSTER_CONTINUITY_FEATURE_COLUMNS`
(`radiant/dire_roster_players_retained`) is a separate DuckDB feature
computed from the same canonical facts with the same strict `<`
boundary. Slice 5's `players_retained_from_previous_match` is the
canonical research-state equivalent (see module `__all__` and the
cross-check test `tests/features/test_roster_continuity_cross_check.py`).
This module does not modify the production feature -- see the
documentation in that test for the exact equivalence and the one edge
case (incomplete lineups) where they differ by design.

The pure derivation functions take plain observation tuples and are
testable without a database; `fetch_*` / `audit_*` / `check_*` helpers
talk to Postgres and are used by the CLI audit script
(`scripts/audit_roster_state.py`).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Connection, Engine

from dota_predictor.data.roster_history import (
    EXPECTED_LINEUP_SIZE,
    LineupSummary,
    classify_lineup,
    collect_player_team_observations,
)

__all__ = [
    "PlayerTeamState",
    "TeamRosterState",
    "audit_roster_state",
    "check_future_deletion_invariant",
    "derive_player_team_state",
    "derive_team_roster_state",
    "fetch_player_team_state",
    "fetch_team_roster_state",
]

_SECONDS_PER_DAY = 86400.0


def _days_between(later: datetime, earlier: datetime) -> float:
    """Whole-day-agnostic float day count between two datetimes."""
    return (later - earlier).total_seconds() / _SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class TeamRosterState:
    """The historical roster state of one team in one match, evaluated
    immediately before that match.

    Every "prior"/"previous"/"first"/"last" value is computed from
    observations with `start_time` strictly before this match's
    `start_time` (equal timestamps never become prior evidence). The
    current match's own lineup may be used as the lineup being evaluated,
    but never contributes to any of its own prior counts.

    Malformed/incomplete lineups stay explicit: `is_complete_five` and the
    resolved/distinct/null counts describe the current lineup, and the
    retained/changed/same-lineup fields are `None` (undefined) unless both
    the current and the previous lineup are complete fives. Prior
    exact-lineup experience is likewise only defined for a complete five
    current lineup.

    Team-composition counts are over the resolved players in the current
    lineup (each resolved player always has a classification, since their
    previous observed team is known). For a complete five lineup the three
    counts reconcile to 5; for a malformed lineup they reconcile to the
    number of resolved players and the malformation is visible via the
    cardinality flags. Unresolved players are never fabricated a
    classification.
    """

    match_id: int
    start_time: datetime
    team_id: int
    # --- current lineup (reused from Slice 4's team-match-lineup identity)
    lineup_player_ids: tuple[int, ...]
    lineup_key: str | None
    n_resolved_players: int
    n_distinct_players: int
    is_complete_five: bool
    # --- most recent strictly-prior observed team match
    previous_match_id: int | None
    previous_match_at: datetime | None
    previous_lineup_player_ids: tuple[int, ...] | None
    previous_lineup_key: str | None
    previous_is_complete_five: bool | None
    players_retained_from_previous_match: int | None
    players_changed_from_previous_match: int | None
    same_lineup_as_previous_match: bool | None
    # --- prior exact-lineup experience (same team, complete fives only)
    prior_exact_lineup_match_count: int | None
    last_exact_lineup_match_id: int | None
    last_exact_lineup_at: datetime | None
    # --- team-level newcomer/returner composition (over resolved players)
    continuing_player_count: int
    first_observed_for_team_count: int
    returning_player_count: int
    # --- descriptive timing
    days_since_team_previous_match: float | None


@dataclass(frozen=True, slots=True)
class PlayerTeamState:
    """One player's historical relationship to one team, evaluated
    immediately before one match.

    Grain: `(player_id, team_id, match_id)` -- one row per player in a
    team's observed lineup for a match. All counts and classifications use
    only observations with `start_time` strictly before the match's
    `start_time`.

    The flags are observational classifications, never contractual
    semantics:

    * `is_first_observed_match_for_team` -- the player has never
      previously been observed representing this team.
    * `is_returning_to_team` -- the player previously represented this
      team AND their immediately previous observed team (for any match)
      was a different team (e.g. ``A -> B -> A``).
    * `is_continuing_with_team` -- the player's immediately previous
      observed team is the same as this team.

    These are mutually exclusive for any row whose previous observed team
    is known. A player who has appeared for many teams is still a
    "continuing" player when their immediately previous observed team is
    the current team. No transfer / signing / departure / stand-in label
    is ever assigned.
    """

    player_id: int
    team_id: int
    match_id: int
    start_time: datetime
    prior_team_match_count: int
    first_prior_team_match_at: datetime | None
    last_prior_team_match_at: datetime | None
    previous_observed_team_id: int | None
    previous_observed_match_id: int | None
    previous_observed_match_at: datetime | None
    is_first_observed_match_for_team: bool
    is_returning_to_team: bool
    is_continuing_with_team: bool
    consecutive_prior_team_appearances: int
    days_since_player_previous_match: float | None
    days_since_player_previous_team_match: float | None


def _require_resolved(observations: Iterable[tuple[int | None, int | None, int, datetime]]) -> None:
    """Unresolved identities are never fabricated into state.

    An observation with a `None` player or team id raises `ValueError`
    rather than silently forming a state row, mirroring
    `roster_history.derive_observed_spells`. Callers that may encounter
    unresolved rows must filter and report them via
    `collect_player_team_observations` first.
    """
    for player_id, team_id, _match_id, _start_time in observations:
        if player_id is None or team_id is None:
            raise ValueError(
                "roster state: unresolved identity cannot form a state row "
                f"(player_id={player_id!r}, team_id={team_id!r})"
            )


def derive_player_team_state(
    observations: Iterable[tuple[int | None, int | None, int, datetime]],
) -> list[PlayerTeamState]:
    """Derive one `PlayerTeamState` per `(player_id, team_id, match_id)`.

    Each observation is `(player_id, team_id, match_id, start_time)`. Per
    player, observations are ordered by `(start_time, match_id, team_id)`
    for deterministic spell-so-far bookkeeping. All causal fields use
    strict `<` on `start_time` only.

    Raises `ValueError` on unresolved identities. Output is sorted by
    `(player_id, team_id, match_id)`.
    """
    materialized = list(observations)
    _require_resolved(materialized)
    by_player: dict[int, list[tuple[datetime, int, int]]] = {}
    for player_id, team_id, match_id, start_time in materialized:
        by_player.setdefault(int(player_id), []).append(
            (start_time, int(match_id), int(team_id))
        )

    states: list[PlayerTeamState] = []
    for player_id in sorted(by_player):
        ordered = sorted(by_player[player_id])
        prior_team_counts: dict[int, int] = {}
        first_seen_by_team: dict[int, datetime] = {}
        last_seen_by_team: dict[int, datetime] = {}
        spell_index = 0
        previous_run_team: int | None = None
        position_in_spell = 0
        # The most recent observation with `start_time` strictly less than
        # the current one. Equal-timestamp matches are never each other's
        # previous observed match, so `ordered[i-1]` is only eligible when
        # its `start_time` is strictly smaller.
        last_strict_prior: tuple[datetime, int, int] | None = None
        previous_start_time: datetime | None = None

        for i, (start_time, match_id, team_id) in enumerate(ordered):
            if previous_start_time is not None and start_time > previous_start_time:
                last_strict_prior = ordered[i - 1]
            previous_start_time = start_time

            if previous_run_team is not None and team_id != previous_run_team:
                spell_index += 1
                position_in_spell = 1
            else:
                position_in_spell += 1
            previous_run_team = team_id

            prior_count = prior_team_counts.get(team_id, 0)
            prior_team_counts[team_id] = prior_count + 1
            # Read prior-team first/last seen BEFORE registering this
            # observation, so these describe prior matches only -- the
            # current match never enters its own prior counts.
            first_prior_team_at = first_seen_by_team.get(team_id) if prior_count > 0 else None
            last_prior_team_at = last_seen_by_team.get(team_id) if prior_count > 0 else None
            if prior_count == 0:
                first_seen_by_team[team_id] = start_time
            last_seen_by_team[team_id] = start_time

            prev_at, prev_match_id, prev_team_id = (None, None, None)
            if last_strict_prior is not None:
                prev_at, prev_match_id, prev_team_id = last_strict_prior

            days_since_previous_match = (
                _days_between(start_time, prev_at) if prev_at is not None else None
            )
            days_since_previous_team_match = (
                _days_between(start_time, last_prior_team_at)
                if last_prior_team_at is not None
                else None
            )

            is_first = prior_count == 0
            is_returning = (
                prior_count > 0
                and prev_team_id is not None
                and prev_team_id != team_id
            )
            is_continuing = prev_team_id is not None and prev_team_id == team_id

            states.append(
                PlayerTeamState(
                    player_id=player_id,
                    team_id=team_id,
                    match_id=match_id,
                    start_time=start_time,
                    prior_team_match_count=prior_count,
                    first_prior_team_match_at=first_prior_team_at,
                    last_prior_team_match_at=last_prior_team_at,
                    previous_observed_team_id=prev_team_id,
                    previous_observed_match_id=prev_match_id,
                    previous_observed_match_at=prev_at,
                    is_first_observed_match_for_team=is_first,
                    is_returning_to_team=is_returning,
                    is_continuing_with_team=is_continuing,
                    consecutive_prior_team_appearances=position_in_spell - 1,
                    days_since_player_previous_match=days_since_previous_match,
                    days_since_player_previous_team_match=days_since_previous_team_match,
                )
            )
    return states


def _build_team_lineups(
    observations: Iterable[tuple[int | None, int | None, int, datetime]],
) -> dict[int, list[tuple[datetime, int, LineupSummary]]]:
    """Group observations into per-team chronological match lineups.

    Preserves duplicate observations (a player listed twice for one
    team-match) so `classify_lineup` reports them explicitly instead of a
    set silently collapsing them. Output per team is sorted by
    `(start_time, match_id)`.
    """
    per_match: dict[tuple[int, int], tuple[datetime, list[int]]] = {}
    for player_id, team_id, match_id, start_time in observations:
        key = (int(team_id), int(match_id))
        entry = per_match.get(key)
        if entry is None:
            entry = (start_time, [])
            per_match[key] = entry
        entry[1].append(int(player_id))

    by_team: dict[int, list[tuple[datetime, int, LineupSummary]]] = {}
    for (team_id, match_id), (start_time, player_ids) in per_match.items():
        by_team.setdefault(team_id, []).append(
            (start_time, match_id, classify_lineup(player_ids))
        )
    for team_id, team_rows in by_team.items():
        team_rows.sort(key=lambda row: (row[0], row[1]))
    return by_team


def derive_team_roster_state(
    observations: Iterable[tuple[int | None, int | None, int, datetime]],
) -> list[TeamRosterState]:
    """Derive one `TeamRosterState` per `(team_id, match_id)`.

    Each observation is `(player_id, team_id, match_id, start_time)`. The
    current match's lineup is the team's observed lineup in that match
    (reusing `roster_history.classify_lineup` -- the Slice 4 lineup
    identity). The previous observed team match is the most recent match
    of the same team with strictly smaller `start_time`; among equal
    prior `start_time`s the largest `match_id` is chosen as the
    deterministic presentation tie-breaker (it never manufactures causal
    precedence -- all candidates are strictly prior).

    Equal `start_time` matches for the same team are never each other's
    previous match. Prior exact-lineup counts are only defined for
    complete-five current lineups and count strictly prior complete-five
    matches of the same team with the identical `lineup_key`.

    Raises `ValueError` on unresolved identities. Output is sorted by
    `(team_id, match_id)`.
    """
    materialized = list(observations)
    _require_resolved(materialized)
    by_team = _build_team_lineups(materialized)
    player_states = derive_player_team_state(materialized)

    comp_by_match: dict[tuple[int, int], dict[int, PlayerTeamState]] = {}
    for state in player_states:
        comp_by_match.setdefault((state.team_id, state.match_id), {})[
            state.player_id
        ] = state

    states: list[TeamRosterState] = []
    for team_id in sorted(by_team):
        rows = by_team[team_id]
        # exact_counts: lineup_key -> (prior_count, last_match_id, last_at)
        # updated only after an entire equal-start_time run is processed, so
        # equal timestamps never become each other's prior exact-lineup evidence.
        exact_counts: dict[str, tuple[int, int | None, datetime | None]] = {}
        prev_strict: tuple[datetime, int, LineupSummary] | None = None

        i = 0
        n = len(rows)
        while i < n:
            run_time = rows[i][0]
            j = i
            while j < n and rows[j][0] == run_time:
                j += 1
            run_matches = rows[i:j]

            for start_time, match_id, lineup in run_matches:
                previous: tuple[datetime, int, LineupSummary] | None = prev_strict
                previous_lineup = previous[2] if previous is not None else None

                retained = changed = same = None
                if lineup.is_complete_five and previous_lineup is not None and previous_lineup.is_complete_five:
                    cur_ids = set(lineup.lineup_player_ids)
                    prev_ids = set(previous_lineup.lineup_player_ids)
                    retained = len(cur_ids & prev_ids)
                    changed = EXPECTED_LINEUP_SIZE - retained
                    same = cur_ids == prev_ids

                prior_exact_count: int | None = None
                last_exact_id: int | None = None
                last_exact_at: datetime | None = None
                if lineup.is_complete_five and lineup.lineup_key is not None:
                    prior_exact_count, last_exact_id, last_exact_at = exact_counts.get(
                        lineup.lineup_key, (0, None, None)
                    )

                comp = comp_by_match.get((team_id, match_id), {})
                continuing = sum(
                    1
                    for pid in lineup.lineup_player_ids
                    if (entry := comp.get(pid)) is not None
                    and entry.is_continuing_with_team
                )
                first_observed = sum(
                    1
                    for pid in lineup.lineup_player_ids
                    if (entry := comp.get(pid)) is not None
                    and entry.is_first_observed_match_for_team
                )
                returning = sum(
                    1
                    for pid in lineup.lineup_player_ids
                    if (entry := comp.get(pid)) is not None
                    and entry.is_returning_to_team
                )

                days_since_prev = (
                    _days_between(start_time, previous[0])
                    if previous is not None
                    else None
                )

                states.append(
                    TeamRosterState(
                        match_id=match_id,
                        start_time=start_time,
                        team_id=team_id,
                        lineup_player_ids=lineup.lineup_player_ids,
                        lineup_key=lineup.lineup_key,
                        n_resolved_players=lineup.n_resolved_players,
                        n_distinct_players=lineup.n_distinct_players,
                        is_complete_five=lineup.is_complete_five,
                        previous_match_id=previous[1] if previous is not None else None,
                        previous_match_at=previous[0] if previous is not None else None,
                        previous_lineup_player_ids=(
                            previous_lineup.lineup_player_ids
                            if previous_lineup is not None
                            else None
                        ),
                        previous_lineup_key=(
                            previous_lineup.lineup_key
                            if previous_lineup is not None
                            else None
                        ),
                        previous_is_complete_five=(
                            previous_lineup.is_complete_five
                            if previous_lineup is not None
                            else None
                        ),
                        players_retained_from_previous_match=retained,
                        players_changed_from_previous_match=changed,
                        same_lineup_as_previous_match=same,
                        prior_exact_lineup_match_count=prior_exact_count,
                        last_exact_lineup_match_id=last_exact_id,
                        last_exact_lineup_at=last_exact_at,
                        continuing_player_count=continuing,
                        first_observed_for_team_count=first_observed,
                        returning_player_count=returning,
                        days_since_team_previous_match=days_since_prev,
                    )
                )

            # After the whole run: apply exact-lineup updates so equal
            # start_times never contribute to each other's prior counts.
            for _start_time, match_id, lineup in run_matches:
                if lineup.is_complete_five and lineup.lineup_key is not None:
                    count, _lid, _lat = exact_counts.get(lineup.lineup_key, (0, None, None))
                    exact_counts[lineup.lineup_key] = (
                        count + 1,
                        match_id,
                        _start_time,
                    )

            prev_strict = rows[j - 1]
            i = j

    return states


def fetch_team_roster_state(conn: Connection) -> list[TeamRosterState]:
    """Derive team roster state from the canonical facts in `conn`.

    Equivalent to querying `research.team_roster_state` but does not
    require the research schema/view to be installed.
    """
    observations, _null_players, _null_teams = collect_player_team_observations(conn)
    return derive_team_roster_state(observations)


def fetch_player_team_state(conn: Connection) -> list[PlayerTeamState]:
    """Derive player-team state from the canonical facts in `conn`.

    Equivalent to querying `research.player_team_state` but does not
    require the research schema/view to be installed.
    """
    observations, _null_players, _null_teams = collect_player_team_observations(conn)
    return derive_player_team_state(observations)


def check_future_deletion_invariant(
    observations: Sequence[tuple[int | None, int | None, int, datetime]],
    *,
    check_match_ids: Iterable[int] | None = None,
    max_checks: int | None = None,
) -> dict[str, object]:
    """Verify future-deletion invariance for historical roster state.

    For each checked match at time `T`, the state computed from the full
    corpus must be identical to the state recomputed after deleting every
    observation with `start_time > T` (i.e. keeping only `start_time <=
    T`). Because all state only reads observations strictly before `T`,
    this must hold; the check exists to catch implementation bugs that
    accidentally read future or equal-time data.

    `check_match_ids` restricts which matches are verified; by default
    every match is checked. `max_checks` deterministically sub-samples
    (evenly spaced across chronological match order) to bound runtime on
    large corpora.

    Returns a dict with `matches_checked`, `team_state_violations`, and
    `player_state_violations` (each violation is a human-readable string;
    empty lists mean the invariant held).
    """
    normalized = [
        (int(p), int(t), int(m), s)
        for (p, t, m, s) in observations
        if p is not None and t is not None
    ]
    full_team = derive_team_roster_state(normalized)
    full_player = derive_player_team_state(normalized)

    team_by_match: dict[int, list[TeamRosterState]] = {}
    for state in full_team:
        team_by_match.setdefault(state.match_id, []).append(state)
    player_by_match: dict[int, list[PlayerTeamState]] = {}
    for state in full_player:
        player_by_match.setdefault(state.match_id, []).append(state)

    if check_match_ids is not None:
        check_ids = sorted({int(m) for m in check_match_ids})
    else:
        check_ids = sorted({m for _p, _t, m, _s in normalized})

    if max_checks is not None and len(check_ids) > max_checks:
        step = len(check_ids) / max_checks
        check_ids = [check_ids[int(i * step)] for i in range(max_checks)]

    by_start = sorted(normalized, key=lambda obs: (obs[3], obs[2]))
    times = [obs[3] for obs in by_start]

    team_violations: list[str] = []
    player_violations: list[str] = []
    for match_id in check_ids:
        current_start = next(s for s in by_start if s[2] == match_id)[3]
        hi = bisect_right(times, current_start)
        truncated = by_start[:hi]

        trunc_team = derive_team_roster_state(truncated)
        trunc_player = derive_player_team_state(truncated)

        trunc_team_this = sorted(
            (s for s in trunc_team if s.match_id == match_id),
            key=lambda s: s.team_id,
        )
        full_team_this = sorted(
            team_by_match.get(match_id, []), key=lambda s: s.team_id
        )
        if trunc_team_this != full_team_this:
            team_violations.append(
                f"match {match_id}: team roster state differs after "
                "deleting future observations"
            )

        trunc_player_this = sorted(
            (s for s in trunc_player if s.match_id == match_id),
            key=lambda s: (s.team_id, s.player_id),
        )
        full_player_this = sorted(
            player_by_match.get(match_id, []),
            key=lambda s: (s.team_id, s.player_id),
        )
        if trunc_player_this != full_player_this:
            player_violations.append(
                f"match {match_id}: player-team state differs after "
                "deleting future observations"
            )

    return {
        "matches_checked": len(check_ids),
        "team_state_violations": team_violations,
        "player_state_violations": player_violations,
    }


def _distribution(values: Sequence[int]) -> dict[str, int]:
    """Min/median/max/count summary over a (possibly empty) sequence."""
    if not values:
        return {"min": 0, "median": 0, "max": 0, "count": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "min": ordered[0],
        "median": ordered[n // 2],
        "max": ordered[-1],
        "count": n,
    }


def audit_roster_state(engine: Engine, *, max_invariant_checks: int = 25) -> dict[str, object]:
    """Deterministic historical-roster-state census over the warehouse.

    Read-only; never writes, never re-fetches. Computes the team-match
    and player-team state from canonical facts, reports the Slice 5 audit
    distributions, and runs the future-deletion invariant check on a
    deterministic sample (bounded by `max_invariant_checks`) so anomalies
    are reported rather than hidden.
    """
    with engine.connect() as conn:
        observations, null_player_count, null_team_count = (
            collect_player_team_observations(conn)
        )

    team_states = derive_team_roster_state(observations)
    player_states = derive_player_team_state(observations)

    retained_values = [
        s.players_retained_from_previous_match
        for s in team_states
        if s.players_retained_from_previous_match is not None
    ]
    changed_values = [
        s.players_changed_from_previous_match
        for s in team_states
        if s.players_changed_from_previous_match is not None
    ]
    prior_exact = [
        s.prior_exact_lineup_match_count
        for s in team_states
        if s.prior_exact_lineup_match_count is not None
    ]
    incomplete_lineups = [
        s for s in team_states
        if not s.is_complete_five
    ]
    impossible_aggregates = [
        s
        for s in team_states
        if s.is_complete_five
        and (
            s.continuing_player_count
            + s.first_observed_for_team_count
            + s.returning_player_count
        )
        != EXPECTED_LINEUP_SIZE
    ]
    previous_unavailable = [
        s for s in player_states if s.previous_observed_team_id is None
    ]

    invariant = check_future_deletion_invariant(
        observations, max_checks=max_invariant_checks
    )

    return {
        "observations": {
            "total_usable_observations": len(observations),
            "null_player_id_observations": null_player_count,
            "null_team_id_observations": null_team_count,
        },
        "team_match_state": {
            "team_match_rows": len(team_states),
            "rows_with_previous_team_match": sum(
                1 for s in team_states if s.previous_match_id is not None
            ),
            "first_observed_team_matches": sum(
                1 for s in team_states if s.previous_match_id is None
            ),
            "retained_player_distribution": _counts_by_value(retained_values, 0, EXPECTED_LINEUP_SIZE),
            "changed_player_distribution": _counts_by_value(changed_values, 0, EXPECTED_LINEUP_SIZE),
            "same_lineup_as_previous_count": sum(
                1 for s in team_states if s.same_lineup_as_previous_match
            ),
            "exact_lineup_first_use_count": sum(
                1 for s in team_states if s.prior_exact_lineup_match_count == 0
            ),
            "exact_lineup_repeat_use_count": sum(
                1
                for s in team_states
                if s.prior_exact_lineup_match_count is not None
                and s.prior_exact_lineup_match_count > 0
            ),
            "prior_exact_lineup_count_distribution": _distribution(prior_exact),
        },
        "player_team_state": {
            "player_match_rows": len(player_states),
            "continuing_observations": sum(
                1 for s in player_states if s.is_continuing_with_team
            ),
            "first_observed_for_team_observations": sum(
                1 for s in player_states if s.is_first_observed_match_for_team
            ),
            "returning_to_team_observations": sum(
                1 for s in player_states if s.is_returning_to_team
            ),
            "prior_team_match_count_distribution": _distribution(
                [s.prior_team_match_count for s in player_states]
            ),
            "players_with_previous_observed_team_unavailable": len(previous_unavailable),
        },
        "integrity": {
            "incomplete_lineup_count": len(incomplete_lineups),
            "impossible_aggregate_counts": len(impossible_aggregates),
            "future_deletion_invariant": invariant,
            "current_match_in_own_prior_counts": "by construction False"
            " (all prior counts use strict start_time <; see "
            "check_future_deletion_invariant)",
        },
    }


def _counts_by_value(
    values: Sequence[int], low: int, high: int
) -> dict[str, int]:
    """Count histogram over an inclusive integer range, zero-filled."""
    return {str(value): sum(1 for v in values if v == value) for value in range(low, high + 1)}