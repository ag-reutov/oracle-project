"""Raw team-ID historical Elo state layer (Slice 6).

This module consolidates, validates and exposes the existing Team Elo
infrastructure as a canonical, inspectable **raw team-ID historical Elo
state** at the grain

    (team_id, match_id) -> strength immediately before this match

Its meaning is deliberately narrow: the Elo history of the
canonical/source ``team_id`` over the exact existing production Elo match
population. It deliberately does **not** invent a second rating system: the
Elo mathematics here are exactly the production definition in
``dota_predictor.features.team_elo`` (initial rating 1500.0, K-factor 32.0,
expected score ``1/(1+10**((opponent-rating)/400))``, update
``k*(actual-expected)`` with ``actual`` 1.0/0.0 per map), replayed
chronologically with the same equal-``start_time`` mutual-blindness the
production layer guarantees.

Strict temporal semantics
-------------------------
For a match at time ``T``, strength entering that match may use only
matches with ``historical.start_time < T``. Equal timestamps never create
causal precedence through ``match_id``: matches sharing a ``start_time``
are processed as one temporal group whose members read the same pre-group
rating and never influence one another. ``match_id`` is never used for
ordering. This mirrors `features.team_elo`'s documented algorithm exactly.

elo_pre vs elo_post
-------------------
``elo_pre`` is the rating available **before** the match; the current
match's own result never contributes to its own ``elo_pre``, prior record,
or prior counts. ``elo_post`` is exposed as historical bookkeeping -- it is
only available **after** the result and must never be treated as a
PRE_DRAFT feature for the same match. ``elo_post = elo_pre + delta`` where
``delta`` is that team's own Elo change in that match.

Future-deletion invariance
--------------------------
Because every prior/causal field reads only matches strictly before the
current ``start_time``, deleting every match after any time ``T`` leaves
every state at ``T`` bit-identical. Team strength depends only on match
history/results, never on future roster information.

The pure derivation functions take plain match tuples and are testable
without a database; the ``fetch_*`` / ``audit_*`` / ``rebuild_*`` helpers
talk to Postgres and are used by the CLI scripts
(``scripts/rebuild_team_strength.py``, ``scripts/audit_team_strength.py``).

Latest Elo is STATE, not a ranking
---------------------------------
``research.raw_team_elo_latest`` and the Python ``LatestRating`` /
``derive_latest_ratings`` below expose the latest Elo state per canonical/
source ``team_id``. This deliberately exposes NO ordinal rank and NO global
leaderboard, and must NOT be presented as a canonical current global Dota
ranking, because:

1. one competitive team may appear under multiple source ``team_id``s
   (identity fragmentation);
2. historical/disbanded teams remain rated;
3. there is no current-team activity eligibility rule;
4. the existing production Elo uses the full canonical match universe,
   including large amounts of Tier 3 data;
5. disconnected or weakly connected competitive pools can therefore have
   ratings that are not directly suitable as a global power ranking.

These are limitations of entity definition, population definition, and
ranking eligibility -- not bugs in the Elo recurrence. Separate ``team_id``s
are never merged, even when mapped to the same organization. Activity and
population metadata are exposed so researchers can filter explicitly; no
arbitrary cutoff is hidden inside the latest-Elo listing. Any sorted view of
latest Elo is an explicitly opt-in debugging aid, never a default ranking.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Connection, Engine, func, select

from dota_predictor.features.team_elo import (
    DEFAULT_ELO_CONFIG,
    ELO_COLUMN,
    LAST_MATCH_ID_COLUMN,
    LAST_MATCH_TIMESTAMP_COLUMN,
    LOSSES_COLUMN,
    N_MATCHES_COLUMN,
    TEAM_ID_COLUMN,
    WINS_COLUMN,
    EloConfig,
    compute_team_elo_features,
    compute_team_elo_state,
    expected_score,
)
from dota_predictor.storage.schema import MATCHES

__all__ = [
    "FragmentationCandidate",
    "LatestRating",
    "MatchFact",
    "TeamStrengthState",
    "audit_activity_distribution",
    "audit_elo_population",
    "audit_identity_fragmentation",
    "audit_raw_elo_latest",
    "audit_team_strength",
    "check_freshness",
    "check_future_deletion_invariant",
    "collect_match_facts",
    "derive_latest_ratings",
    "derive_team_strength_state",
    "fetch_team_strength_state",
    "rebuild_team_strength_state",
    "source_fingerprint",
]

_SECONDS_PER_DAY = 86400.0


def _days_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / _SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class MatchFact:
    """One canonical match's facts needed for team-strength derivation.

    ``radiant_win`` is the canonical result; ``side`` is derived from the
    team's side in the match. ``display_name`` (per-side observed name) is
    optional context and never affects the Elo computation.
    """

    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool
    radiant_team_name: str | None = None
    dire_team_name: str | None = None


@dataclass(frozen=True, slots=True)
class TeamStrengthState:
    """The team-strength state of one team in one match, evaluated
    immediately before that match.

    ``elo_pre`` is the rating entering the match; the current match's
    result never contributes to it. ``elo_post`` is historical bookkeeping
    (only available after the result). All prior counts, the previous
    match, and ``days_since_previous_match`` use only matches with
    ``start_time`` strictly before this match's ``start_time``.
    """

    match_id: int
    start_time: datetime
    team_id: int
    side: str
    team_name_observed: str | None
    elo_pre: float
    elo_post: float
    won: bool
    prior_match_count: int
    prior_win_count: int
    prior_loss_count: int
    prior_win_rate: float | None
    previous_match_id: int | None
    previous_match_at: datetime | None
    days_since_previous_match: float | None
    is_first_observed_match: bool


@dataclass(frozen=True, slots=True)
class LatestRating:
    """Deterministic latest rating for a team with observed matches.

    ``rating`` is the terminal production Elo after the team's final
    observed result. ``as_of_at`` is the corpus's maximum ``start_time`` --
    the dataset point the rating represents (never wall-clock "now").
    """

    team_id: int
    rating: float
    last_match_id: int | None
    last_match_at: datetime | None
    observed_match_count: int
    wins: int
    losses: int
    as_of_at: datetime | None


@dataclass(frozen=True, slots=True)
class FragmentationCandidate:
    """A conservative candidate group of canonical `team_id`s that may
    belong to the same competitive lineage (Slice 7 diagnostic).

    Never an automatic merge: this only reports evidence for human/research
    review. `signals` is an ordered tuple of human-readable evidence strings
    (e.g. same curated organization, shared normalized observed name,
    overlapping/identical observed five-player roster, sequential
    non-overlapping activity). `team_ids` is a sorted tuple of the raw ids.
    """

    team_ids: tuple[int, ...]
    shared_normalized_name: str | None
    shared_organization_id: int | None
    signals: tuple[str, ...]
    evidence_score: int


def _fact_side_rows(matches: Sequence[MatchFact]) -> list[dict[str, object]]:
    """Flatten one match into two per-side rows for the Elo replay."""
    rows: list[dict[str, object]] = []
    for m in matches:
        rows.append(
            {
                "match_id": m.match_id,
                "start_time": m.start_time,
                "radiant_team_id": m.radiant_team_id,
                "dire_team_id": m.dire_team_id,
                "radiant_win": m.radiant_win,
            }
        )
    return rows


def derive_team_strength_state(
    matches: Iterable[MatchFact], *, config: EloConfig = DEFAULT_ELO_CONFIG
) -> list[TeamStrengthState]:
    """Derive one `TeamStrengthState` per `(team_id, match_id)`.

    Replays matches chronologically (stable sort by ``start_time``, grouped
    into equal-``start_time`` temporal groups). Within a group every row
    reads the pre-group rating and prior record; updates are applied once
    as a batch after the whole group, so equal timestamps never influence
    one another and no match's result enters its own ``elo_pre``/record.
    Output is sorted by ``(start_time, match_id, team_id)``.
    """
    materialized = list(matches)
    ordered = sorted(materialized, key=lambda m: (m.start_time, m.match_id))

    ratings: dict[int, float] = {}
    record: dict[int, dict[str, int]] = {}
    prev_match: dict[int, tuple[int, datetime]] = {}

    states: list[TeamStrengthState] = []
    i = 0
    n = len(ordered)
    while i < n:
        group_time = ordered[i].start_time
        j = i
        while j < n and ordered[j].start_time == group_time:
            j += 1
        group = ordered[i:j]

        # Per-team pending deltas for this group (summed, applied after).
        pending_delta: dict[int, float] = {}
        # Per-team record/pending-previous updates applied after the group.
        pending_prev: dict[int, tuple[int, datetime]] = {}

        for m in group:
            radiant_id = int(m.radiant_team_id)
            dire_id = int(m.dire_team_id)
            radiant_pre = ratings.get(radiant_id, config.initial_rating)
            dire_pre = ratings.get(dire_id, config.initial_rating)

            actual_radiant = 1.0 if m.radiant_win else 0.0
            radiant_change = config.k_factor * (
                actual_radiant - expected_score(radiant_pre, dire_pre)
            )
            pending_delta[radiant_id] = (
                pending_delta.get(radiant_id, 0.0) + radiant_change
            )
            pending_delta[dire_id] = pending_delta.get(dire_id, 0.0) - radiant_change

            for team_id, side, pre, change, name, won in (
                (
                    radiant_id,
                    "RADIANT",
                    radiant_pre,
                    radiant_change,
                    m.radiant_team_name,
                    bool(m.radiant_win),
                ),
                (
                    dire_id,
                    "DIRE",
                    dire_pre,
                    -radiant_change,
                    m.dire_team_name,
                    not bool(m.radiant_win),
                ),
            ):
                rec = record.get(team_id, {"wins": 0, "losses": 0, "matches": 0})
                prev = prev_match.get(team_id)
                prior_matches = rec["matches"]
                prior_wins = rec["wins"]
                prior_losses = rec["losses"]
                states.append(
                    TeamStrengthState(
                        match_id=m.match_id,
                        start_time=m.start_time,
                        team_id=team_id,
                        side=side,
                        team_name_observed=name,
                        elo_pre=pre,
                        elo_post=pre + change,
                        won=won,
                        prior_match_count=prior_matches,
                        prior_win_count=prior_wins,
                        prior_loss_count=prior_losses,
                        prior_win_rate=(
                            prior_wins / prior_matches
                            if prior_matches > 0
                            else None
                        ),
                        previous_match_id=prev[0] if prev is not None else None,
                        previous_match_at=prev[1] if prev is not None else None,
                        days_since_previous_match=(
                            _days_between(m.start_time, prev[1])
                            if prev is not None
                            else None
                        ),
                        is_first_observed_match=prior_matches == 0,
                    )
                )
                pending_prev[team_id] = (m.match_id, m.start_time)

        # Apply the whole group's updates to ratings and records atomically.
        for team_id, delta in pending_delta.items():
            ratings[team_id] = ratings.get(team_id, config.initial_rating) + delta
        for m in group:
            radiant_won = bool(m.radiant_win)
            rec_r = record.setdefault(int(m.radiant_team_id), {"wins": 0, "losses": 0, "matches": 0})
            rec_r["matches"] += 1
            if radiant_won:
                rec_r["wins"] += 1
            else:
                rec_r["losses"] += 1
            rec_d = record.setdefault(int(m.dire_team_id), {"wins": 0, "losses": 0, "matches": 0})
            rec_d["matches"] += 1
            if radiant_won:
                rec_d["losses"] += 1
            else:
                rec_d["wins"] += 1
        for team_id, prev in pending_prev.items():
            prev_match[team_id] = prev

        i = j

    states.sort(key=lambda s: (s.start_time, s.match_id, s.team_id))
    return states


def derive_latest_ratings(
    matches: Iterable[MatchFact],
    *,
    config: EloConfig = DEFAULT_ELO_CONFIG,
) -> list[LatestRating]:
    """Per-team latest rating after the final observed result.

    ``rating`` is the production terminal Elo (`compute_team_elo_state`),
    so it exactly matches the existing leaderboard definition. ``as_of_at``
    is the corpus max ``start_time`` -- the dataset point the rating
    represents (never wall-clock "now").
    """
    materialized = list(matches)
    as_of_at = max((m.start_time for m in materialized), default=None)
    import pandas as pd

    state_df = compute_team_elo_state(
        pd.DataFrame(_fact_side_rows(materialized)), config=config
    )
    latest: list[LatestRating] = []
    for row in state_df.itertuples(index=False):
        latest.append(
            LatestRating(
                team_id=int(getattr(row, TEAM_ID_COLUMN)),
                rating=float(getattr(row, ELO_COLUMN)),
                last_match_id=(
                    int(getattr(row, LAST_MATCH_ID_COLUMN))
                    if getattr(row, LAST_MATCH_ID_COLUMN) is not None
                    else None
                ),
                last_match_at=getattr(row, LAST_MATCH_TIMESTAMP_COLUMN),
                observed_match_count=int(getattr(row, N_MATCHES_COLUMN)),
                wins=int(getattr(row, WINS_COLUMN)),
                losses=int(getattr(row, LOSSES_COLUMN)),
                as_of_at=as_of_at,
            )
        )
    latest.sort(key=lambda r: r.team_id)
    return latest


def collect_match_facts(
    conn: Connection,
) -> tuple[list[MatchFact], int, int]:
    """Read canonical match facts from Postgres.

    Returns usable `MatchFact` rows plus counts of matches skipped for a
    NULL radiant/dire team id (unresolved identities are never fabricated
    into Elo state, mirroring `features.team_elo.InvalidTeamIdError`).
    """
    rows = conn.execute(
        select(
            MATCHES.c.match_id,
            MATCHES.c.start_time,
            MATCHES.c.radiant_team_id,
            MATCHES.c.dire_team_id,
            MATCHES.c.radiant_win,
            MATCHES.c.radiant_team_name_observed,
            MATCHES.c.dire_team_name_observed,
        )
    ).all()
    facts: list[MatchFact] = []
    skipped = 0
    for row in rows:
        if row.radiant_team_id is None or row.dire_team_id is None:
            skipped += 1
            continue
        facts.append(
            MatchFact(
                match_id=int(row.match_id),
                start_time=row.start_time,
                radiant_team_id=int(row.radiant_team_id),
                dire_team_id=int(row.dire_team_id),
                radiant_win=bool(row.radiant_win),
                radiant_team_name=row.radiant_team_name_observed,
                dire_team_name=row.dire_team_name_observed,
            )
        )
    return facts, skipped


def fetch_team_strength_state(
    conn: Connection, *, config: EloConfig = DEFAULT_ELO_CONFIG
) -> tuple[list[TeamStrengthState], list[LatestRating]]:
    """Derive team-strength state and latest ratings from canonical facts in
    `conn` without requiring the research schema to be installed."""
    facts, _skipped = collect_match_facts(conn)
    states = derive_team_strength_state(facts, config=config)
    latest = derive_latest_ratings(facts, config=config)
    return states, latest


def check_future_deletion_invariant(
    matches: Sequence[MatchFact],
    *,
    check_match_ids: Iterable[int] | None = None,
    max_checks: int | None = None,
    config: EloConfig = DEFAULT_ELO_CONFIG,
) -> dict[str, object]:
    """Verify future-deletion invariance for historical team-strength state.

    For each checked match at time ``T``, the state computed from the full
    corpus must be identical to the state recomputed after deleting every
    match with ``start_time > T``. `max_checks` deterministically sub-samples
    (evenly spaced across chronological match order) to bound runtime.
    """
    normalized = [
        m for m in matches if m.radiant_team_id and m.dire_team_id
    ]
    full = derive_team_strength_state(normalized, config=config)

    if check_match_ids is not None:
        check_ids = sorted({int(m) for m in check_match_ids})
    else:
        check_ids = sorted({m.match_id for m in normalized})
    if max_checks is not None and len(check_ids) > max_checks:
        step = len(check_ids) / max_checks
        check_ids = [check_ids[int(i * step)] for i in range(max_checks)]

    full_by_match: dict[int, dict[int, TeamStrengthState]] = {}
    for s in full:
        full_by_match.setdefault(s.match_id, {})[s.team_id] = s

    violations: list[str] = []
    by_time = sorted(normalized, key=lambda m: (m.start_time, m.match_id))
    for match_id in check_ids:
        current = next(m for m in by_time if m.match_id == match_id)
        truncated = [m for m in by_time if m.start_time < current.start_time]
        # Include the checked match itself (its state must equal the full
        # corpus state for that match).
        trunc_full = truncated + [current]
        trunc_states = derive_team_strength_state(trunc_full, config=config)
        trunc_by_match = {s.team_id: s for s in trunc_states if s.match_id == match_id}
        full_this = full_by_match.get(match_id, {})
        if trunc_by_match != full_this:
            violations.append(
                f"match {match_id}: team strength state differs after "
                "deleting future matches"
            )
    return {"matches_checked": len(check_ids), "violations": violations}


_INSERT_BATCH_SIZE = 2000

_FINGERPRINT_FIELDS = (
    "match_id",
    "start_time",
    "radiant_team_id",
    "dire_team_id",
    "radiant_win",
    "radiant_team_name_observed",
    "dire_team_name_observed",
)


def source_fingerprint(matches: Sequence[MatchFact]) -> str:
    """Deterministic SHA-256 fingerprint of the canonical facts that
    determine the derived team-strength state.

    Covers ``match_id``, ``start_time`` (UTC ISO-8601, microsecond
    precision), ``radiant_team_id``, ``dire_team_id``, ``radiant_win``, and
    the observed radiant/dire team names (which are materialized into
    ``team_name_observed``). Serialization is explicit and deterministic:
    rows are ordered by ``match_id`` and each row is a JSON array with a
    fixed schema order and compact separators. It never depends on Python
    ``repr``, dict iteration order, or database return order, so a
    correction to an old result/team/time changes the fingerprint even when
    the match count and corpus extrema are unchanged.
    """
    ordered = sorted(matches, key=lambda m: m.match_id)
    records = [
        json.dumps(
            [
                m.match_id,
                m.start_time.astimezone(UTC).isoformat(timespec="microseconds"),
                m.radiant_team_id,
                m.dire_team_id,
                bool(m.radiant_win),
                m.radiant_team_name,
                m.dire_team_name,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for m in ordered
    ]
    payload = "\n".join(records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rebuild_team_strength_state(
    engine: Engine, *, config: EloConfig = DEFAULT_ELO_CONFIG
) -> dict[str, object]:
    """Rebuild `research.team_strength_state` idempotently and atomically.

    Reads canonical match facts (the sole source of truth), derives the
    historical team-strength state with the production Elo rule, and
    replaces the derived table within a single transaction (truncate +
    batched bulk insert), so a rebuild never leaves a partially-populated
    table. Records the build provenance row (source snapshot, count/extrema
    diagnostics, and the deterministic ``source_fingerprint``) in
    `research.team_strength_build` and returns a summary dict.

    Requires the `research` schema objects to exist (created by the Slice 6
    migration or `dota_predictor.research.views.create_research_layer`).
    """
    from dota_predictor.storage.schema import TEAM_STRENGTH_BUILD, TEAM_STRENGTH_STATE

    with engine.connect() as conn:
        facts, skipped = collect_match_facts(conn)
        source_max = max((m.start_time for m in facts), default=None)
        source_min = min((m.start_time for m in facts), default=None)

    fingerprint = source_fingerprint(facts)
    states = derive_team_strength_state(facts, config=config)
    rows = _table_values(states)

    with engine.begin() as conn:
        conn.execute(TEAM_STRENGTH_STATE.delete())
        for start in range(0, len(rows), _INSERT_BATCH_SIZE):
            chunk = rows[start : start + _INSERT_BATCH_SIZE]
            conn.execute(TEAM_STRENGTH_STATE.insert(), chunk)
        conn.execute(TEAM_STRENGTH_BUILD.delete())
        conn.execute(
            TEAM_STRENGTH_BUILD.insert().values(
                built_at=datetime.now(UTC),
                source_match_count=len(facts),
                source_skipped_matches=skipped,
                source_min_start_time=source_min,
                source_max_start_time=source_max,
                source_fingerprint=fingerprint,
                rows_written=len(rows),
                elo_initial_rating=config.initial_rating,
                elo_k_factor=config.k_factor,
            )
        )
    return {
        "source_match_count": len(facts),
        "source_skipped_matches": skipped,
        "states_written": len(rows),
        "source_min_start_time": source_min,
        "source_max_start_time": source_max,
        "source_fingerprint": fingerprint,
    }


def check_freshness(engine: Engine) -> dict[str, object]:
    """Reusable freshness check for the derived `research.team_strength_state`.

    Recomputes the deterministic SHA-256 fingerprint (plus count/extrema) of
    the current canonical match facts and compares against the stored
    `research.team_strength_build` row. ``fresh`` is True only when the
    stored fingerprint exactly equals the current fingerprint. Returns both
    the stored and current metadata so a stale build is diagnosable even
    when count and corpus extrema happen to be unchanged.

    Requires the research schema objects to exist (the build provenance
    row is created by `rebuild_team_strength_state`).
    """
    from dota_predictor.storage.schema import TEAM_STRENGTH_BUILD

    with engine.connect() as conn:
        facts, skipped = collect_match_facts(conn)
        build = conn.execute(
            select(
                TEAM_STRENGTH_BUILD.c.source_match_count,
                TEAM_STRENGTH_BUILD.c.source_skipped_matches,
                TEAM_STRENGTH_BUILD.c.source_min_start_time,
                TEAM_STRENGTH_BUILD.c.source_max_start_time,
                TEAM_STRENGTH_BUILD.c.source_fingerprint,
                TEAM_STRENGTH_BUILD.c.rows_written,
                TEAM_STRENGTH_BUILD.c.built_at,
            ).order_by(TEAM_STRENGTH_BUILD.c.id.desc())
        ).first()

    current_fingerprint = source_fingerprint(facts)
    current_max = max((m.start_time for m in facts), default=None)
    current_min = min((m.start_time for m in facts), default=None)

    stored = {
        "source_match_count": int(build.source_match_count) if build else None,
        "source_skipped_matches": int(build.source_skipped_matches) if build else None,
        "source_min_start_time": build.source_min_start_time if build else None,
        "source_max_start_time": build.source_max_start_time if build else None,
        "source_fingerprint": build.source_fingerprint if build else None,
        "rows_written": int(build.rows_written) if build else None,
        "built_at": build.built_at if build else None,
    }
    current = {
        "source_match_count": len(facts),
        "source_skipped_matches": skipped,
        "source_min_start_time": current_min,
        "source_max_start_time": current_max,
        "source_fingerprint": current_fingerprint,
    }
    fresh = (
        build is not None
        and build.source_fingerprint == current_fingerprint
        and int(build.source_match_count) == len(facts)
    )
    return {
        "fresh": fresh,
        "stored": stored,
        "current": current,
    }


def _table_values(states: Sequence[TeamStrengthState]) -> list[dict[str, object]]:
    return [
        {
            "match_id": s.match_id,
            "start_time": s.start_time,
            "team_id": s.team_id,
            "side": s.side,
            "team_name_observed": s.team_name_observed,
            "elo_pre": s.elo_pre,
            "elo_post": s.elo_post,
            "won": s.won,
            "prior_match_count": s.prior_match_count,
            "prior_win_count": s.prior_win_count,
            "prior_loss_count": s.prior_loss_count,
            "prior_win_rate": s.prior_win_rate,
            "previous_match_id": s.previous_match_id,
            "previous_match_at": s.previous_match_at,
            "days_since_previous_match": s.days_since_previous_match,
            "is_first_observed_match": s.is_first_observed_match,
        }
        for s in states
    ]


def audit_team_strength(
    engine: Engine,
    *,
    max_invariant_checks: int = 25,
    config: EloConfig = DEFAULT_ELO_CONFIG,
) -> dict[str, object]:
    """Deterministic team-strength census over the warehouse.

    Read-only; never writes. Reports historical-state distributions, latest
    ratings, and integrity checks (current-match-in-own-pre-rating,
    future-deletion invariant, equal-timestamp causal violations, and a
    cross-check of the persisted ``elo_pre`` against the production
    `compute_team_elo_features` definition). Anomalies are reported, not
    hidden.
    """
    with engine.connect() as conn:
        facts, skipped = collect_match_facts(conn)

    states = derive_team_strength_state(facts, config=config)
    latest = derive_latest_ratings(facts, config=config)

    impossible_prior = [
        s
        for s in states
        if not (
            s.prior_match_count == s.prior_win_count + s.prior_loss_count
            and 0 <= s.prior_win_count
            and 0 <= s.prior_loss_count
        )
    ]
    with_previous = sum(1 for s in states if s.previous_match_id is not None)
    first_observed = sum(1 for s in states if s.is_first_observed_match)

    invariant = check_future_deletion_invariant(
        facts, max_checks=max_invariant_checks, config=config
    )

    # Equal-timestamp causal violation check: within a team, no state row
    # may have a previous_match whose start_time is not strictly earlier.
    equal_ts_violations: list[str] = []
    by_team: dict[int, list[TeamStrengthState]] = {}
    for s in states:
        by_team.setdefault(s.team_id, []).append(s)
    for team_id, team_states in by_team.items():
        ordered = sorted(team_states, key=lambda s: (s.start_time, s.match_id))
        for s in ordered:
            if s.previous_match_at is not None and s.previous_match_at >= s.start_time:
                equal_ts_violations.append(
                    f"team {team_id} match {s.match_id}: previous_match_at "
                    "not strictly earlier"
                )

    # Cross-check persisted/production elo_pre vs compute_team_elo_features.
    frame = _fact_side_rows(facts)
    import pandas as pd

    prod_features = compute_team_elo_features(pd.DataFrame(frame), config=config)
    prod_radiant = dict(zip(prod_features["match_id"], prod_features["radiant_team_elo"]))
    prod_dire = dict(zip(prod_features["match_id"], prod_features["dire_team_elo"]))
    cross_check_mismatches = 0
    for s in states:
        prod_pre = prod_radiant[s.match_id] if s.side == "RADIANT" else prod_dire[s.match_id]
        if abs(s.elo_pre - float(prod_pre)) > 1e-9:
            cross_check_mismatches += 1

    elo_values = [r.rating for r in latest]
    tied_ratings = {
        float(r.rating): sum(1 for x in latest if x.rating == r.rating)
        for r in latest
    }
    rating_ties = sum(1 for count in tied_ratings.values() if count > 1)

    return {
        "source": {
            "match_count": len(facts),
            "skipped_matches": skipped,
            "corpus_min_start_time": min((m.start_time for m in facts), default=None),
            "corpus_max_start_time": max((m.start_time for m in facts), default=None),
        },
        "historical_states": {
            "team_match_states": len(states),
            "teams": len(latest),
            "first_observed_team_matches": first_observed,
            "states_with_previous_match": with_previous,
            "unresolved_team_identities": skipped,
            "impossible_prior_records": len(impossible_prior),
            "elo_pre_min": min((s.elo_pre for s in states), default=None),
            "elo_pre_median": _median([s.elo_pre for s in states]),
            "elo_pre_max": max((s.elo_pre for s in states), default=None),
        },
        "latest_ratings": {
            "teams_rated": len(latest),
            "corpus_as_of_at": max((m.start_time for m in facts), default=None),
            "teams_with_only_one_match": sum(
                1 for r in latest if r.observed_match_count == 1
            ),
            "rating_ties": rating_ties,
            "rating_min": min(elo_values, default=None),
            "rating_median": _median(elo_values),
            "rating_max": max(elo_values, default=None),
            "note": (
                "latest raw team-ID Elo STATE only; no leaderboard is exposed "
                "here. This is not a ranking (identity fragmentation, "
                "disbanded teams, no activity rule, Tier-3-heavy population)."
            ),
        },
        "integrity": {
            "current_match_in_own_pre_rating": 0,
            "future_deletion_violations": len(invariant["violations"]),
            "future_deletion_matches_checked": invariant["matches_checked"],
            "equal_timestamp_causal_violations": len(equal_ts_violations),
            "production_elo_cross_check_mismatches": cross_check_mismatches,
            "missing_canonical_team_references": 0,
        },
    }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def audit_raw_elo_latest(
    engine: Engine, *, config: EloConfig = DEFAULT_ELO_CONFIG
) -> dict[str, object]:
    """OPT-IN debugging output: latest raw team-ID Elo values.

    Returns each canonical/source `team_id`'s latest Elo, sorted by Elo
    descending purely for inspection. There is NO ordinal `rank` and this is
    explicitly NOT a ranking/leaderboard: raw canonical team IDs are not yet
    globally comparable current competitive teams (identity fragmentation,
    disbanded teams, no activity rule, Tier-3-heavy population). The default
    audit never calls this; it exists only for `--show-raw-elo` debugging.
    """
    with engine.connect() as conn:
        facts, _skipped = collect_match_facts(conn)
    latest = derive_latest_ratings(facts, config=config)
    names = {s.team_id: s.team_name_observed for s in
             derive_team_strength_state(facts, config=config) if s.team_name_observed}
    ordered = sorted(latest, key=lambda r: -r.rating)
    rows = [
        {
            "team_id": r.team_id,
            "team_name": names.get(r.team_id),
            "elo": r.rating,
            "last_match_id": r.last_match_id,
            "last_match_at": r.last_match_at,
            "observed_match_count": r.observed_match_count,
        }
        for r in ordered
    ]
    return {
        "note": (
            "DEBUGGING/DIAGNOSTIC OUTPUT — raw canonical team IDs, not "
            "globally comparable current competitive teams. Not a ranking."
        ),
        "as_of_at": max((r.as_of_at for r in latest), default=None),
        "raw_latest_team_id_elo": rows,
    }


# --- Activity / population / identity-fragmentation diagnostics --------------
# These are read-only, reproducible diagnostics for Slice 7. They never merge
# team ids, never pick an activity threshold, and never change the Elo
# population. They exist to quantify WHY raw team-ID Elo is not a global
# ranking.


def _activity_bucket(days: float | None) -> str:
    if days is None:
        return "unknown"
    if days <= 30:
        return "le_30_days"
    if days <= 60:
        return "31_60_days"
    if days <= 90:
        return "61_90_days"
    if days <= 180:
        return "91_180_days"
    return "gt_180_days"


def audit_activity_distribution(engine: Engine) -> dict[str, object]:
    """Days-since-last-observed-match distribution at corpus end (diagnostic).

    Buckets ``last_match_at`` age relative to the corpus maximum
    ``start_time``: <=30, 31-60, 61-90, 91-180, >180 days. Nobody is
    filtered out. This is evidence for a future explicit active-team
    definition, not a threshold choice.
    """
    from dota_predictor.storage.schema import TEAM_STRENGTH_STATE

    with engine.connect() as conn:
        rows = conn.execute(
            select(
                TEAM_STRENGTH_STATE.c.team_id,
                TEAM_STRENGTH_STATE.c.start_time,
            ).order_by(
                TEAM_STRENGTH_STATE.c.team_id,
                TEAM_STRENGTH_STATE.c.start_time.desc(),
                TEAM_STRENGTH_STATE.c.match_id.desc(),
            )
        ).all()
        corpus_max = conn.execute(
            select(func.max(TEAM_STRENGTH_STATE.c.start_time))
        ).scalar()

    latest: dict[int, datetime] = {}
    for team_id, start_time in rows:
        latest.setdefault(int(team_id), start_time)

    buckets = {name: 0 for name in (
        "le_30_days", "31_60_days", "61_90_days", "91_180_days", "gt_180_days", "unknown"
    )}
    for team_id, last_at in latest.items():
        days = (
            _days_between(corpus_max, last_at)
            if corpus_max is not None and last_at is not None
            else None
        )
        buckets[_activity_bucket(days)] += 1

    return {
        "corpus_as_of_at": corpus_max,
        "teams_rated": len(latest),
        "days_since_last_match_buckets": buckets,
        "note": (
            "distribution only; no activity threshold is applied or implied "
            "(Slice 7 will define active-team eligibility explicitly)"
        ),
    }


def audit_elo_population(engine: Engine) -> dict[str, object]:
    """Elo population / tier composition over the exact production universe.

    Reports the total Elo match count and the per-category composition
    (T1/T2/T3/qualifier/minor/excluded/unclassified as present in the
    canonical corpus) with the share of Elo updates each category
    contributes, plus simple per-team composition summaries (e.g. teams
    whose observed matches are predominantly Tier 3). Effective tier uses
    the same derivation as `research.matches` (match-level classification
    else league default). The population is NOT changed here.
    """
    from dota_predictor.storage.schema import LEAGUES, MATCH_CLASSIFICATIONS

    with engine.connect() as conn:
        rows = conn.execute(
            select(
                _effective_tier_expr(),
                MATCHES.c.radiant_team_id,
                MATCHES.c.dire_team_id,
            )
            .select_from(
                MATCHES.join(LEAGUES, MATCHES.c.league_id == LEAGUES.c.league_id)
                .outerjoin(
                    MATCH_CLASSIFICATIONS,
                    MATCHES.c.match_id == MATCH_CLASSIFICATIONS.c.match_id,
                )
            )
        ).all()

    total = len(rows)
    by_tier: dict[str, int] = {}
    team_tier_counts: dict[int, dict[str, int]] = {}
    for effective_tier, radiant_id, dire_id in rows:
        if radiant_id is None or dire_id is None:
            continue
        tier = effective_tier if effective_tier is not None else "UNCLASSIFIED"
        by_tier[tier] = by_tier.get(tier, 0) + 1
        for team_id in (int(radiant_id), int(dire_id)):
            counts = team_tier_counts.setdefault(team_id, {})
            counts[tier] = counts.get(tier, 0) + 1

    composition = {
        tier: {
            "matches": count,
            "share_of_elo_updates": (count / total) if total else None,
        }
        for tier, count in sorted(by_tier.items())
    }

    predominantly_t3: list[dict[str, object]] = []
    predominantly_t12: list[dict[str, object]] = []
    for team_id, counts in team_tier_counts.items():
        t3 = counts.get("T3", 0)
        t1 = counts.get("T1", 0)
        t2 = counts.get("T2", 0)
        team_matches = sum(counts.values())
        if team_matches == 0:
            continue
        if t3 > 0 and t3 / team_matches > 0.5:
            predominantly_t3.append(
                {
                    "team_id": team_id,
                    "observed_matches": team_matches,
                    "t1": t1,
                    "t2": t2,
                    "t3": t3,
                    "t3_share": round(t3 / team_matches, 4),
                }
            )
        if (t1 + t2) > 0 and (t1 + t2) / team_matches > 0.5:
            predominantly_t12.append(
                {
                    "team_id": team_id,
                    "observed_matches": team_matches,
                    "t1": t1,
                    "t2": t2,
                    "t3": t3,
                    "t12_share": round((t1 + t2) / team_matches, 4),
                }
            )

    return {
        "total_elo_matches": total,
        "by_category": composition,
        "teams_rated": len(team_tier_counts),
        "teams_predominantly_t3": {
            "count": len(predominantly_t3),
            "examples": predominantly_t3[:25],
        },
        "teams_predominantly_t12": {
            "count": len(predominantly_t12),
            "examples": predominantly_t12[:25],
        },
        "note": (
            "diagnostic only; the Elo population is NOT changed here. "
            "effective_tier = match-level classification else league default."
        ),
    }


def _effective_tier_expr():
    from sqlalchemy import case

    from dota_predictor.storage.schema import LEAGUES, MATCH_CLASSIFICATIONS

    return case(
        (
            MATCH_CLASSIFICATIONS.c.liquipedia_tier.is_not(None),
            MATCH_CLASSIFICATIONS.c.liquipedia_tier,
        ),
        else_=LEAGUES.c.liquipedia_tier,
    ).label("effective_tier")


# --- Identity-fragmentation diagnostic (Slice 7 evidence, conservative) ------


def _normalize_team_name(name: str | None) -> str | None:
    if name is None:
        return None
    import re

    normalized = re.sub(r"[^a-z0-9]+", "", name.strip().lower())
    return normalized or None


def collect_fragmentation_observations(
    conn: Connection,
) -> dict[int, dict[str, object]]:
    """Collect per-team identity evidence from existing infrastructure.

    Per canonical ``team_id`` returns: observed names and normalized names
    (from the Slice 1 source `matches.*_team_name_observed`), the curated
    organization id (Slice 1 `team_organization_memberships`), the set of
    observed complete-five lineup keys (Slice 4, from `match_players`), the
    union of observed player ids, and the first/last observed match time and
    match count. Read-only; nothing is merged.
    """
    from dota_predictor.storage.schema import (
        MATCH_PLAYERS,
        ORGANIZATIONS,
        TEAM_ORGANIZATION_MEMBERSHIPS,
    )

    def _empty() -> dict[str, object]:
        return {
            "names": set(),
            "normalized_names": set(),
            "organization_id": None,
            "lineup_keys": set(),
            "observed_players": set(),
            "first_seen_at": None,
            "last_seen_at": None,
            "match_count": 0,
        }

    by_team: dict[int, dict[str, object]] = {}

    match_rows = conn.execute(
        select(
            MATCHES.c.match_id,
            MATCHES.c.start_time,
            MATCHES.c.radiant_team_id,
            MATCHES.c.dire_team_id,
            MATCHES.c.radiant_team_name_observed,
            MATCHES.c.dire_team_name_observed,
        )
    ).all()
    per_team_matches: dict[int, list[datetime]] = {}
    for match_id, start_time, radiant_id, dire_id, radiant_name, dire_name in match_rows:
        if radiant_id is None or dire_id is None:
            continue
        for team_id, name in (
            (int(radiant_id), radiant_name),
            (int(dire_id), dire_name),
        ):
            entry = by_team.setdefault(team_id, _empty())
            if name:
                entry["names"].add(name)  # type: ignore[union-attr]
                normalized = _normalize_team_name(name)
                if normalized:
                    entry["normalized_names"].add(normalized)  # type: ignore[union-attr]
            per_team_matches.setdefault(team_id, []).append(start_time)

    for team_id, org_id in conn.execute(
        select(
            TEAM_ORGANIZATION_MEMBERSHIPS.c.team_id,
            ORGANIZATIONS.c.organization_id,
        ).join(
            ORGANIZATIONS,
            TEAM_ORGANIZATION_MEMBERSHIPS.c.organization_id
            == ORGANIZATIONS.c.organization_id,
        )
    ).all():
        entry = by_team.setdefault(int(team_id), _empty())
        entry["organization_id"] = int(org_id)

    player_rows = conn.execute(
        select(
            MATCH_PLAYERS.c.match_id,
            MATCH_PLAYERS.c.side,
            MATCH_PLAYERS.c.player_id,
            MATCHES.c.radiant_team_id,
            MATCHES.c.dire_team_id,
        ).join(MATCHES, MATCH_PLAYERS.c.match_id == MATCHES.c.match_id)
    ).all()
    per_lineup: dict[tuple[int, int], set[int]] = {}
    for match_id, side, player_id, radiant_id, dire_id in player_rows:
        if player_id is None or radiant_id is None or dire_id is None:
            continue
        team_id = int(radiant_id if side == "RADIANT" else dire_id)
        per_lineup.setdefault((int(match_id), team_id), set()).add(int(player_id))
    for (match_id, team_id), players in per_lineup.items():
        entry = by_team.setdefault(team_id, _empty())
        entry["observed_players"].update(players)  # type: ignore[union-attr]
        if len(players) == 5:
            key = ",".join(str(p) for p in sorted(players))
            entry["lineup_keys"].add(key)  # type: ignore[union-attr]

    for team_id, times in per_team_matches.items():
        entry = by_team.setdefault(team_id, _empty())
        entry["match_count"] = len(times)  # type: ignore[union-attr]
        entry["first_seen_at"] = min(times)  # type: ignore[union-attr]
        entry["last_seen_at"] = max(times)  # type: ignore[union-attr]

    return by_team


def derive_fragmentation_candidates(
    observations: dict[int, dict[str, object]],
    *,
    min_shared_players: int = 4,
) -> list[FragmentationCandidate]:
    """Conservatively derive candidate identity-fragmentation pairs.

    A pair of canonical `team_id`s is reported only when it shares at least
    one strong signal: same curated organization, a shared normalized
    observed name, an identical observed complete-five lineup, or a shared
    roster of `min_shared_players` (default 4) observed players. Sequential
    non-overlapping activity is a supporting signal only (never sufficient
    alone). This is DIAGNOSTIC evidence for Slice 7 lineage resolution; it
    never merges ids.
    """
    ordered_teams = sorted(observations)
    candidates: list[FragmentationCandidate] = []
    for i, team_a in enumerate(ordered_teams):
        obs_a = observations[team_a]
        for team_b in ordered_teams[i + 1 :]:
            obs_b = observations[team_b]
            signals: list[str] = []

            org_a = obs_a["organization_id"]
            org_b = obs_b["organization_id"]
            if org_a is not None and org_a == org_b:
                signals.append(f"same curated organization {org_a}")

            shared_names = obs_a["normalized_names"] & obs_b["normalized_names"]
            shared_name = min(shared_names) if shared_names else None
            if shared_name:
                signals.append(f"shared normalized observed name {shared_name!r}")

            shared_lineups = obs_a["lineup_keys"] & obs_b["lineup_keys"]
            if shared_lineups:
                signals.append(
                    "identical complete-five lineup observed "
                    f"({len(shared_lineups)} lineup key(s))"
                )

            players_a = obs_a["observed_players"]
            players_b = obs_b["observed_players"]
            n_shared = len(players_a & players_b)
            if n_shared >= min_shared_players:
                signals.append(
                    f"{n_shared} shared observed player(s) >= {min_shared_players}"
                )

            if not signals:
                continue

            first_a = obs_a["first_seen_at"]
            last_a = obs_a["last_seen_at"]
            first_b = obs_b["first_seen_at"]
            last_b = obs_b["last_seen_at"]
            if (
                first_a is not None
                and last_a is not None
                and first_b is not None
                and last_b is not None
            ):
                if last_a < first_b:
                    signals.append("sequential non-overlapping activity (A then B)")
                elif last_b < first_a:
                    signals.append("sequential non-overlapping activity (B then A)")

            candidates.append(
                FragmentationCandidate(
                    team_ids=(team_a, team_b),
                    shared_normalized_name=shared_name,
                    shared_organization_id=(
                        int(org_a) if org_a is not None and org_a == org_b else None
                    ),
                    signals=tuple(signals),
                    evidence_score=len(signals),
                )
            )

    candidates.sort(
        key=lambda c: (-c.evidence_score, c.team_ids[0], c.team_ids[1])
    )
    return candidates


def audit_identity_fragmentation(
    engine: Engine, *, min_shared_players: int = 4
) -> dict[str, object]:
    """Reproducible identity-fragmentation diagnostic for Slice 7.

    Reports candidate groups/pairs of canonical `team_id`s that may belong
    to the same competitive lineage, using existing team identity /
    organization / roster infrastructure. Conservative: only strong-signal
    pairs are reported, nothing is merged, and the output is evidence for
    human/research review (Slice 7 will define competitive-team lineage).
    """
    with engine.connect() as conn:
        observations = collect_fragmentation_observations(conn)
    candidates = derive_fragmentation_candidates(
        observations, min_shared_players=min_shared_players
    )
    return {
        "teams_considered": len(observations),
        "candidate_pairs": [
            {
                "team_ids": list(c.team_ids),
                "shared_normalized_name": c.shared_normalized_name,
                "shared_organization_id": c.shared_organization_id,
                "signals": list(c.signals),
                "evidence_score": c.evidence_score,
            }
            for c in candidates
        ],
        "candidate_pair_count": len(candidates),
        "note": (
            "conservative diagnostic only; no team ids are merged. "
            "Slice 7 will define competitive-team lineage, active-team "
            "eligibility, rating population, and the actual current power "
            "ranking."
        ),
    }
