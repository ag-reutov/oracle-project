"""Observed roster history layer (Slice 4).

This module implements the historical **player -> team -> observed match
appearances -> chronological observed roster spells** relationship on top
of the existing canonical warehouse, without ever claiming contractual
membership:

* **Canonical roster observation** = one `match_players` row joined to its
  parent `matches` row: "a player was observed representing team X in
  match Y at time T". The team is derived from the match's own
  `radiant_team_id` / `dire_team_id` by the player's `side`, exactly as
  `research.player_matches` does. This is a source fact; no box-score
  statistics are copied into roster history and no new observation table
  is created (`research.player_matches` already is the canonical
  appearance relation).
* **Exact match lineups** = the set of players observed for one team in
  one match, exposed with an explicit cardinality audit (exactly five /
  fewer / more / duplicates / nulls). Malformed lineups are never forced
  into a five-player shape. A deterministic lineup identity is derived
  from the **sorted canonical player ids** (`lineup_key`); it is only a
  convenience for grouping "did this exact five play again", never a hash
  with hidden semantics.
* **Observed player-team spells** = the deterministic, chronological
  decomposition of a player's appearances. Order by `(start_time,
  match_id, team_id)`; begin a new spell whenever the observed `team_id`
  changes. A later return to a previous team is a **new spell**
  (`A -> B -> A` is three spells), and a long period of inactivity with no
  intervening team observation does **not** split a spell.
* **No fabricated interval boundaries**: spells expose `first_seen_at` /
  `last_seen_at` (both observed match times) and never invent exact
  `joined_at` / `left_at` dates. Unknown membership between matches is
  represented simply by the absence of a later observation -- there is no
  continuous-membership assumption.

Identity rules follow Slices 1/2 verbatim: roster history uses canonical
`team_id` (Slice 1, stable across display-name changes) and canonical
`player_id` (Slice 2). Unresolved identities are never merged or invented;
rows that cannot resolve are counted and reported by the audit, and the
pure spell derivation refuses (raises) rather than silently skipping them.

Temporal-integrity boundary: this slice is purely descriptive. It builds
no roster-strength / stability / continuity / chemistry / transfer
features. A later historical-state slice may construct "roster at time T"
using only observations strictly before `T` -- this module provides the
clean observations that make that possible, but does not itself evaluate
or store any pre-match state.

The pure derivation functions take plain observation tuples and are
testable without a database; the `collect_*` / `fetch_*` / `audit_*`
helpers talk to Postgres and are used by the CLI audit script
(`scripts/audit_roster_history.py`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Connection, Engine, func, select, text

from dota_predictor.data.canonical_schema import Side
from dota_predictor.storage.schema import MATCH_PLAYERS, MATCHES, PLAYERS, TEAMS

__all__ = [
    "LineupSummary",
    "ObservedTeamSpell",
    "audit_roster_history",
    "classify_lineup",
    "collect_player_team_observations",
    "derive_observed_spells",
    "fetch_player_team_spells",
]

# Expected professional lineup size (5v5). Used only for the explicit
# cardinality audit -- malformed lineups are reported, never repaired.
EXPECTED_LINEUP_SIZE = 5


@dataclass(frozen=True, slots=True)
class ObservedTeamSpell:
    """One maximal run of matches in which a player was observed for one team.

    "Maximal" means the run breaks only when another `team_id` is observed
    for the same player; a later return to the same team is a new spell.
    No continuous membership between observations is assumed.

    Attributes:
        player_id: Canonical player id (Slice 2).
        team_id: Canonical team id (Slice 1) the player was observed for.
        spell_index: 1-based chronological spell number for this player
            (1 = earliest observed team, 2 = next, ...).
        first_seen_at / last_seen_at: Earliest/latest observed match
            `start_time` in this spell. These are OBSERVED times -- they
            are not joined/left dates and must never be treated as
            contractual transfer dates.
        first_match_id / last_match_id: Match ids of the earliest/latest
            observations (deterministic tie-break by match id for equal
            `start_time`).
        observed_match_count: Number of matches observed in this spell.
    """

    player_id: int
    team_id: int
    spell_index: int
    first_seen_at: datetime
    first_match_id: int
    last_seen_at: datetime
    last_match_id: int
    observed_match_count: int


@dataclass(frozen=True, slots=True)
class LineupSummary:
    """Cardinality audit of one team-match lineup.

    A lineup is the set of players observed for one `team_id` in one
    `match_id`. `n_players` counts every observed row; `n_resolved_players`
    counts rows with a usable player id; `n_null_player_ids` counts the
    rest. `n_distinct_players` counts distinct resolved ids. The flags make
    malformed lineups explicit instead of silently forcing them into a
    five-player shape.
    """

    n_players: int
    n_resolved_players: int
    n_null_player_ids: int
    n_distinct_players: int
    lineup_player_ids: tuple[int, ...]
    lineup_key: str | None

    @property
    def has_duplicate_players(self) -> bool:
        return self.n_distinct_players < self.n_resolved_players

    @property
    def has_fewer_than_five(self) -> bool:
        return self.n_resolved_players < EXPECTED_LINEUP_SIZE

    @property
    def has_more_than_five(self) -> bool:
        return self.n_resolved_players > EXPECTED_LINEUP_SIZE

    @property
    def has_exactly_five(self) -> bool:
        return self.n_resolved_players == EXPECTED_LINEUP_SIZE

    @property
    def is_complete_five(self) -> bool:
        return (
            self.n_resolved_players == EXPECTED_LINEUP_SIZE
            and self.n_distinct_players == EXPECTED_LINEUP_SIZE
            and self.n_null_player_ids == 0
        )


def _spell_rows_to_spell(
    player_id: int, team_id: int, spell_index: int, rows: Sequence[tuple[datetime, int]]
) -> ObservedTeamSpell:
    """Assemble one `ObservedTeamSpell` from its already-ordered rows."""
    return ObservedTeamSpell(
        player_id=player_id,
        team_id=team_id,
        spell_index=spell_index,
        first_seen_at=rows[0][0],
        first_match_id=rows[0][1],
        last_seen_at=rows[-1][0],
        last_match_id=rows[-1][1],
        observed_match_count=len(rows),
    )


def derive_observed_spells(
    observations: Iterable[tuple[int | None, int | None, int, datetime]],
) -> list[ObservedTeamSpell]:
    """Derive deterministic observed player-team spells from observations.

    Each observation is `(player_id, team_id, match_id, start_time)`.
    Per player, observations are ordered by `(start_time, match_id,
    team_id)` -- the team id breaks exact `(start_time, match_id)` ties
    deterministically. A new spell begins whenever the observed `team_id`
    changes; a later return to a previously observed team is a new spell.
    A gap in time with no intervening team observation does not split a
    spell.

    Unresolved identities are never fabricated: an observation with a
    `None` player or team id raises `ValueError` rather than silently
    forming a spell. Callers that may encounter unresolved rows must
    filter and report them (see `collect_player_team_observations`) before
    calling this.

    The result is sorted by `(player_id, spell_index)` regardless of input
    order, and `spell_index` is 1-based (1 = earliest observed team).
    """
    by_player: dict[int, list[tuple[datetime, int, int]]] = {}
    for player_id, team_id, match_id, start_time in observations:
        if player_id is None or team_id is None:
            raise ValueError(
                "derive_observed_spells: unresolved identity cannot form an "
                f"observed spell (player_id={player_id!r}, team_id={team_id!r})"
            )
        by_player.setdefault(int(player_id), []).append(
            (start_time, int(match_id), int(team_id))
        )

    spells: list[ObservedTeamSpell] = []
    for player_id in sorted(by_player):
        ordered = sorted(by_player[player_id])
        current_team: int | None = None
        next_spell_index = 1
        spell_rows: list[tuple[datetime, int]] = []
        for start_time, match_id, team_id in ordered:
            if current_team is not None and team_id != current_team:
                spells.append(
                    _spell_rows_to_spell(
                        player_id, current_team, next_spell_index, spell_rows
                    )
                )
                next_spell_index += 1
                spell_rows = []
            current_team = team_id
            spell_rows.append((start_time, match_id))
        if current_team is not None:
            spells.append(
                _spell_rows_to_spell(
                    player_id, current_team, next_spell_index, spell_rows
                )
            )
    return spells


def classify_lineup(player_ids: Iterable[int | None]) -> LineupSummary:
    """Classify one team-match lineup's cardinality from observed player ids.

    `player_ids` is the unordered list of player ids observed for one team
    in one match (null entries are unresolved rows). Resolved ids are
    sorted for the deterministic `lineup_player_ids` / `lineup_key`.
    Malformed lineups are classified, never repaired.
    """
    ids = [int(player_id) for player_id in player_ids if player_id is not None]
    null_count = sum(1 for player_id in player_ids if player_id is None)
    distinct = sorted(set(ids))
    return LineupSummary(
        n_players=len(ids) + null_count,
        n_resolved_players=len(ids),
        n_null_player_ids=null_count,
        n_distinct_players=len(distinct),
        lineup_player_ids=tuple(distinct),
        lineup_key=",".join(str(player_id) for player_id in distinct) if distinct else None,
    )


def collect_player_team_observations(
    conn: Connection,
) -> tuple[list[tuple[int, int, int, datetime]], int, int]:
    """Read canonical `(player_id, team_id, match_id, start_time)` facts.

    One tuple per `match_players` row. `team_id` is derived from the parent
    match's `radiant_team_id` / `dire_team_id` by the player's `side`
    (never from any raw player-team field -- STRATZ payloads carry none).
    Returns the usable observations plus the counts of rows skipped for a
    NULL player id and a NULL derived team id, so unresolved rows stay
    explicit instead of disappearing.
    """
    rows = conn.execute(
        select(
            MATCH_PLAYERS.c.match_id,
            MATCH_PLAYERS.c.side,
            MATCH_PLAYERS.c.player_id,
            MATCHES.c.start_time,
            MATCHES.c.radiant_team_id,
            MATCHES.c.dire_team_id,
        ).join(MATCHES, MATCH_PLAYERS.c.match_id == MATCHES.c.match_id)
    ).all()

    observations: list[tuple[int, int, int, datetime]] = []
    null_player_count = 0
    null_team_count = 0
    for row in rows:
        team_id = (
            row.radiant_team_id if row.side is Side.RADIANT else row.dire_team_id
        )
        if row.player_id is None or team_id is None:
            if row.player_id is None:
                null_player_count += 1
            if team_id is None:
                null_team_count += 1
            continue
        observations.append(
            (int(row.player_id), int(team_id), int(row.match_id), row.start_time)
        )
    return observations, null_player_count, null_team_count


def fetch_player_team_spells(conn: Connection) -> list[ObservedTeamSpell]:
    """Derive observed spells from the canonical facts in `conn`.

    Equivalent to querying `research.player_team_spells` but does not
    require the research schema/view to be installed, so Python consumers
    can always retrieve the spells from canonical tables.
    """
    observations, _null_players, _null_teams = collect_player_team_observations(conn)
    return derive_observed_spells(observations)


def _lineup_audit(conn: Connection) -> dict[str, int]:
    """Lineup cardinality counts computed directly from canonical facts."""
    row = conn.execute(
        text(
            """
            WITH observations AS (
                SELECT mp.match_id,
                       CASE WHEN mp.side = 'RADIANT'
                            THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                       mp.player_id
                FROM match_players mp
                JOIN matches m USING (match_id)
            ),
            lineups AS (
                SELECT match_id, team_id,
                       count(*) AS n_players,
                       count(player_id) AS n_resolved,
                       count(DISTINCT player_id) AS n_distinct,
                       count(*) FILTER (WHERE player_id IS NULL) AS n_null
                FROM observations
                GROUP BY match_id, team_id
            )
            SELECT
                count(*) AS lineups_examined,
                count(*) FILTER (WHERE n_resolved = 5) AS exactly_five,
                count(*) FILTER (WHERE n_resolved < 5) AS fewer_than_five,
                count(*) FILTER (WHERE n_resolved > 5) AS more_than_five,
                count(*) FILTER (WHERE n_distinct < n_resolved) AS duplicate_anomalies,
                count(*) FILTER (WHERE n_null > 0) AS null_player_anomalies
            FROM lineups
            """
        )
    ).one()
    return {
        "lineups_examined": int(row.lineups_examined),
        "exactly_five": int(row.exactly_five),
        "fewer_than_five": int(row.fewer_than_five),
        "more_than_five": int(row.more_than_five),
        "duplicate_player_anomalies": int(row.duplicate_anomalies),
        "null_player_anomalies": int(row.null_player_anomalies),
    }


def _integrity_audit(conn: Connection) -> dict[str, int]:
    """Integrity counts over canonical references and observation constraints."""
    unresolved_player_rows = int(
        conn.execute(
            select(func.count())
            .select_from(
                MATCH_PLAYERS.join(
                    PLAYERS, MATCH_PLAYERS.c.player_id == PLAYERS.c.player_id, isouter=True
                )
            )
            .where(PLAYERS.c.player_id.is_(None))
        ).scalar_one()
    )
    unresolved_team_rows = int(
        conn.execute(
            select(func.count())
            .select_from(
                MATCHES.join(
                    TEAMS, MATCHES.c.radiant_team_id == TEAMS.c.team_id, isouter=True
                )
            )
            .where(TEAMS.c.team_id.is_(None))
        ).scalar_one()
    ) + int(
        conn.execute(
            select(func.count())
            .select_from(
                MATCHES.join(
                    TEAMS, MATCHES.c.dire_team_id == TEAMS.c.team_id, isouter=True
                )
            )
            .where(TEAMS.c.team_id.is_(None))
        ).scalar_one()
    )
    duplicate_observations = int(
        conn.execute(
            text(
                "SELECT count(*) FROM ("
                "SELECT match_id, player_id FROM match_players "
                "GROUP BY match_id, player_id HAVING count(*) > 1) d"
            )
        ).scalar_one()
    )
    # A player appearing on both sides of the same match would imply two
    # teams for one player-match observation; the (match_id, player_id)
    # unique constraint makes it structurally impossible, but it is
    # reported explicitly so it can never silently reappear.
    both_sides_rows = int(
        conn.execute(
            text(
                "SELECT count(*) FROM ("
                "SELECT match_id, player_id FROM match_players "
                "WHERE side = 'RADIANT'"
                "INTERSECT "
                "SELECT match_id, player_id FROM match_players "
                "WHERE side = 'DIRE') c"
            )
        ).scalar_one()
    )
    return {
        "unresolved_canonical_player_rows": unresolved_player_rows,
        "unresolved_canonical_team_rows": unresolved_team_rows,
        "duplicate_player_match_observations": duplicate_observations,
        "player_on_both_sides_observations": both_sides_rows,
    }


def _spell_history_summary(
    conn: Connection, observations: Sequence[tuple[int, int, int, datetime]]
) -> dict[str, object]:
    """Player-team history counts derived from the usable observations."""
    spells = derive_observed_spells(observations)
    players = sorted({player_id for player_id, _team, _match, _time in observations})
    teams = sorted({team_id for _player, team_id, _match, _time in observations})

    teams_per_player: dict[int, set[int]] = {}
    for player_id, team_id, _match_id, _start_time in observations:
        teams_per_player.setdefault(player_id, set()).add(team_id)

    return_players: set[int] = set()
    seen_teams: dict[int, set[int]] = {}
    for spell in spells:
        prior = seen_teams.get(spell.player_id, set())
        if spell.team_id in prior:
            return_players.add(spell.player_id)
        seen_teams.setdefault(spell.player_id, set()).add(spell.team_id)

    team_counts = sorted(len(teams) for teams in teams_per_player.values())
    spell_lengths = sorted(spell.observed_match_count for spell in spells)
    spells_per_player = sorted(
        sum(1 for s in spells if s.player_id == player_id) for player_id in players
    )

    return {
        "unique_players": len(players),
        "unique_teams": len(teams),
        "unique_player_team_pairs": len(
            {(player_id, team_id) for player_id, team_id, _match, _time in observations}
        ),
        "players_one_team": sum(1 for teams in teams_per_player.values() if len(teams) == 1),
        "players_multi_team": sum(1 for teams in teams_per_player.values() if len(teams) > 1),
        "total_observed_spells": len(spells),
        "one_match_spells": sum(1 for spell in spells if spell.observed_match_count == 1),
        "players_with_return": len(return_players),
        "max_observed_team_count": team_counts[-1] if team_counts else 0,
        "teams_per_player": _min_median_max(team_counts),
        "spells_per_player": _min_median_max(spells_per_player),
        "spell_lengths": _min_median_max(spell_lengths),
    }


def _min_median_max(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "median": 0, "max": 0, "count": 0}
    n = len(values)
    return {
        "min": values[0],
        "median": values[n // 2],
        "max": values[-1],
        "count": n,
    }


def audit_roster_history(engine: Engine) -> dict[str, object]:
    """Deterministic observed-roster-history census over the warehouse.

    Read-only; never writes, never re-fetches, never classifies a player as
    a stand-in, and never assigns contractual membership. Reports the
    observation, lineup, spell, and integrity facts required by Slice 4,
    computing everything directly from canonical tables so the report is
    reproducible regardless of whether the research views are installed.
    Anomalies are reported even when they are structurally impossible under
    the current schema (e.g. a player on both sides of one match), so they
    can never silently reappear.
    """
    with engine.connect() as conn:
        observations, null_player_count, null_team_count = (
            collect_player_team_observations(conn)
        )
        lineup = _lineup_audit(conn)
        integrity = _integrity_audit(conn)
        history = _spell_history_summary(conn, observations)

    return {
        "observations": {
            "total_player_team_match_observations": len(observations),
            "null_player_id_observations": null_player_count,
            "null_team_id_observations": null_team_count,
        },
        "match_lineups": lineup,
        "player_team_history": history,
        "integrity": integrity,
        "team_side_inconsistencies": {
            "count": 0,
            "note": (
                "team_id is derived from the parent match's radiant/dire "
                "teams by the player's side, so a player-team assignment can "
                "never be inconsistent with the match's two teams."
            ),
        },
        "official_roster_source": {
            "available": False,
            "note": (
                "Slice 4 represents observed competitive roster history "
                "derived from match appearances; official contractual roster "
                "history remains a separate future data-source problem."
            ),
        },
    }