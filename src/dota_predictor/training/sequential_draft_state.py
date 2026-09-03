"""Slice 26: causal sequential draft-state dataset.

Research only. Reconstructs ``(match_id, t) -> draft prefix state S_(M,t)``
from canonical ``draft_events``. Does **not** build a next-pick model,
compute next-event log-loss/accuracy, win probability, draft-value
movement, synergy/counter effects, assignment entropy, or flex scores.

Question
--------
What exactly was known after each draft action boundary?

Boundary convention (frozen)
----------------------------
``S_(M,t)`` is the state **before** the event with ``sequence == t``.
It contains only PRE_DRAFT context plus canonical draft events with
``sequence < t``. For a match with ``N`` events (sequences ``0 .. N-1``),
boundaries are ``t = 0, 1, ..., N``. The terminal boundary ``t = N``
contains the full ordered prefix and supports future prediction
semantics ``S_(M,t) -> event_t`` for ``t < N``.

Do not mix “after event i” wording with this “before event t”
implementation.

What is stored
--------------
Deterministic causal prefix state only: match identifiers, PRE_DRAFT
team/roster context, optional pre-match team Elo from existing code,
the ordered event prefix, and derived successful pick/ban lists and
counts. Failed bans remain in the ordered prefix; derived
``*_ban_hero_ids`` include only successful bans
(``was_successful is not False``, matching ``DraftEvent.is_actual``).

What is excluded
----------------
Player↔picked-hero assignment, observed player position, match outcome,
duration, box scores, future events, Slice 23/24/25 research states,
and any claim that the complement of picked/banned heroes is the legal
hero catalog.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    DRAFT_EVENTS_VIEW,
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.features.team_elo import (
    DEFAULT_ELO_CONFIG,
    TEAM_ELO_FEATURE_COLUMNS,
    EloConfig,
    compute_team_elo_features,
)
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS
from dota_predictor.training.player_performance_target import (
    _jsonable_value,
    restrict_development,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    utc_datetime,
)

__all__ = [
    "ACTION_BAN",
    "ACTION_PICK",
    "BOUNDARY_CONVENTION",
    "CATEGORY_COMPLETE_PICKS_INCOMPLETE_BANS",
    "CATEGORY_COMPLETE_SEQUENTIAL",
    "CATEGORY_INCOMPLETE_PICKS",
    "CATEGORY_MALFORMED_ORDERING",
    "CATEGORY_TERMINAL_MISMATCH",
    "CATEGORY_UNKNOWN_FIELDS",
    "CLASSIFICATION_A",
    "CLASSIFICATION_B",
    "CLASSIFICATION_C",
    "DRAFT_EVENT_COLUMNS",
    "SIDE_DIRE",
    "SIDE_RADIANT",
    "SLICE26_DIAGNOSTIC_ONLY",
    "SLICE26_FROZEN_COMPONENTS",
    "SLICE26_RESEARCH_CLASSIFICATION",
    "SLICE26_STATE_KEYS",
    "Slice26DiagnosticReport",
    "audit_draft_events_population",
    "build_draft_prefix_state",
    "build_sequential_draft_states",
    "classify_match_draft_category",
    "classify_slice26",
    "compare_terminal_picks_to_players",
    "event_is_actual",
    "reconstruct_events_from_prefix",
    "run_sequential_draft_state_diagnostics",
    "slice26_report_to_jsonable",
]


# ---------------------------------------------------------------------------
# Frozen convention / classification text
# ---------------------------------------------------------------------------

BOUNDARY_CONVENTION = "before_event_t"
"""``S_(M,t)`` contains events with ``sequence < t`` (state before event t)."""

ACTION_PICK = "PICK"
ACTION_BAN = "BAN"
SIDE_RADIANT = "RADIANT"
SIDE_DIRE = "DIRE"

DRAFT_EVENT_COLUMNS: tuple[str, ...] = (
    "match_id",
    "sequence",
    "action",
    "side",
    "hero_id",
    "was_successful",
)

SLICE26_STATE_KEYS: tuple[str, ...] = (
    "match_id",
    "start_time",
    "game_version_id",
    "boundary_t",
    "n_prior_events",
    "is_terminal",
    "radiant_team_id",
    "dire_team_id",
    "radiant_player_ids",
    "dire_player_ids",
    "radiant_team_elo",
    "dire_team_elo",
    "team_elo_delta",
    "event_prefix",
    "radiant_pick_hero_ids",
    "dire_pick_hero_ids",
    "radiant_ban_hero_ids",
    "dire_ban_hero_ids",
    "unsuccessful_event_prefix",
    "n_radiant_picks",
    "n_dire_picks",
    "n_radiant_bans_successful",
    "n_dire_bans_successful",
    "n_unsuccessful_events",
)

CATEGORY_COMPLETE_SEQUENTIAL = "complete_sequential"
CATEGORY_COMPLETE_PICKS_INCOMPLETE_BANS = "complete_picks_incomplete_bans"
CATEGORY_INCOMPLETE_PICKS = "incomplete_picks"
CATEGORY_MALFORMED_ORDERING = "malformed_ordering"
CATEGORY_TERMINAL_MISMATCH = "terminal_mismatch"
CATEGORY_UNKNOWN_FIELDS = "unknown_side_action_hero"

CLASSIFICATION_A = (
    "A — freeze sequential draft-state construction: ordering is "
    "reliably reconstructable, action/side/hero coverage is usable, "
    "terminal consistency is strong, malformed cases are "
    "deterministically identifiable, and causal prefix semantics are "
    "unambiguous. Freeze the dataset/state definition only."
)
CLASSIFICATION_B = (
    "B — partial freeze: useful sequential reconstruction exists only "
    "for a clearly defined subset/regime (years/versions/sources or "
    "picks-only). Freeze only that supported scope."
)
CLASSIFICATION_C = (
    "C — do not freeze: event ordering/identity is too incomplete or "
    "inconsistent for trustworthy sequential draft reconstruction. "
    "Poor next-pick predictability is NOT a reason for C."
)

# Recorded after development audit (see run_sequential_draft_state_diagnostics).
SLICE26_RESEARCH_CLASSIFICATION = "A"
SLICE26_DIAGNOSTIC_ONLY = True
SLICE26_FROZEN_COMPONENTS: tuple[str, ...] = (
    "boundary_convention_before_event_t",
    "ordered_event_prefix_state",
    "successful_vs_unsuccessful_ban_semantics",
    "terminal_boundary_after_last_event",
)

# Modal professional Captain's Mode length in the development corpus.
MODAL_EVENT_COUNT = 24
MODAL_BAN_COUNT = 14
EXPECTED_SUCCESSFUL_PICKS = 10
PICKS_PER_SIDE = 5

# Reliability gates for A (data/state reliability, not prediction).
MIN_ORDERING_OK_RATE = 0.99
MIN_FIELD_USABLE_RATE = 0.99
MIN_TERMINAL_EXACT_RATE = 0.99
MIN_WELL_FORMED_SHARE = 0.95


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def event_is_actual(action: object, was_successful: object) -> bool:
    """Whether the event changed draft availability.

    Picks always count. Bans count unless explicitly unsuccessful
    (``was_successful is False``). Unknown/null ban success is treated
    as actual — matching ``DraftEvent.is_actual``.
    """
    if action == ACTION_PICK:
        return True
    if action == ACTION_BAN:
        return was_successful is not False
    return False


def _normalize_optional_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return bool(value)


def _normalize_event_row(row: Mapping[str, Any]) -> dict[str, Any]:
    sequence = row.get("sequence")
    action = row.get("action")
    side = row.get("side")
    hero_id = row.get("hero_id")
    was_successful = _normalize_optional_bool(row.get("was_successful"))

    def _str_or_none(value: object) -> str | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return str(value)

    return {
        "sequence": None if sequence is None or pd.isna(sequence) else int(sequence),
        "action": _str_or_none(action),
        "side": _str_or_none(side),
        "hero_id": None if hero_id is None or pd.isna(hero_id) else int(hero_id),
        "was_successful": was_successful,
    }


def reconstruct_events_from_prefix(
    states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recover the ordered canonical event list from a terminal prefix state.

    Expects the terminal ``S_(M, N)`` row (``is_terminal`` true) whose
    ``event_prefix`` equals the full match sequence.
    """
    terminals = [s for s in states if s.get("is_terminal")]
    if len(terminals) != 1:
        raise ValueError(
            f"expected exactly one terminal state, got {len(terminals)}"
        )
    prefix = terminals[0]["event_prefix"]
    if not isinstance(prefix, (list, tuple)):
        raise TypeError("terminal event_prefix must be a sequence")
    return [dict(event) for event in prefix]


# ---------------------------------------------------------------------------
# Sequence integrity / match categories
# ---------------------------------------------------------------------------


def _sequence_integrity(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Characterize ordering for one match's draft events."""
    normalized = [_normalize_event_row(e) for e in events]
    sequences = [e["sequence"] for e in normalized]
    n = len(normalized)
    null_sequence = sum(s is None for s in sequences)
    present = [s for s in sequences if s is not None]
    duplicate_sequences = sorted(
        seq for seq, count in Counter(present).items() if count > 1
    )
    sorted_present = sorted(present)
    expected = list(range(n)) if null_sequence == 0 else None
    gap_free_zero_indexed = (
        null_sequence == 0
        and not duplicate_sequences
        and sorted_present == expected
    )
    # Monotonic in input order only when already sorted by sequence.
    input_sequences = sequences
    non_monotonic = False
    if null_sequence == 0 and not duplicate_sequences:
        non_monotonic = input_sequences != sorted_present

    unknown_action = sum(
        e["action"] not in {ACTION_PICK, ACTION_BAN} for e in normalized
    )
    unknown_side = sum(
        e["side"] not in {SIDE_RADIANT, SIDE_DIRE} for e in normalized
    )
    unknown_hero = sum(
        e["hero_id"] is None or (e["hero_id"] is not None and e["hero_id"] <= 0)
        for e in normalized
    )

    successful_picks = [
        e
        for e in normalized
        if e["action"] == ACTION_PICK and event_is_actual(e["action"], e["was_successful"])
    ]
    successful_bans = [
        e
        for e in normalized
        if e["action"] == ACTION_BAN and event_is_actual(e["action"], e["was_successful"])
    ]
    unsuccessful = [
        e
        for e in normalized
        if e["action"] == ACTION_BAN and e["was_successful"] is False
    ]

    pick_heroes = [e["hero_id"] for e in successful_picks if e["hero_id"] is not None]
    repeated_successful_pick = len(pick_heroes) != len(set(pick_heroes))
    radiant_picks = {
        e["hero_id"]
        for e in successful_picks
        if e["side"] == SIDE_RADIANT and e["hero_id"] is not None
    }
    dire_picks = {
        e["hero_id"]
        for e in successful_picks
        if e["side"] == SIDE_DIRE and e["hero_id"] is not None
    }
    same_hero_both_sides = bool(radiant_picks & dire_picks)

    banned: set[int] = set()
    pick_after_ban = False
    ordered = sorted(
        normalized,
        key=lambda e: (e["sequence"] is None, e["sequence"] if e["sequence"] is not None else -1),
    )
    for event in ordered:
        hero = event["hero_id"]
        if hero is None:
            continue
        if event["action"] == ACTION_BAN and event_is_actual(
            event["action"], event["was_successful"]
        ):
            banned.add(hero)
        elif event["action"] == ACTION_PICK and event_is_actual(
            event["action"], event["was_successful"]
        ):
            if hero in banned:
                pick_after_ban = True

    return {
        "n_events": n,
        "null_sequence_count": null_sequence,
        "duplicate_sequences": tuple(duplicate_sequences),
        "gap_free_zero_indexed": gap_free_zero_indexed,
        "non_monotonic_input_order": non_monotonic,
        "unknown_action_count": unknown_action,
        "unknown_side_count": unknown_side,
        "unknown_hero_count": unknown_hero,
        "n_successful_picks": len(successful_picks),
        "n_successful_bans": len(successful_bans),
        "n_unsuccessful_events": len(unsuccessful),
        "n_radiant_picks": len(radiant_picks),
        "n_dire_picks": len(dire_picks),
        "repeated_successful_pick": repeated_successful_pick,
        "same_hero_both_sides": same_hero_both_sides,
        "pick_after_successful_ban": pick_after_ban,
        "normalized_events": ordered if gap_free_zero_indexed else normalized,
    }


def compare_terminal_picks_to_players(
    events: Sequence[Mapping[str, Any]],
    player_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare successful draft picks to ``match_players.hero_id`` sets."""
    integrity = _sequence_integrity(events)
    picks = [
        e
        for e in integrity["normalized_events"]
        if e["action"] == ACTION_PICK
        and event_is_actual(e["action"], e["was_successful"])
        and e["hero_id"] is not None
        and e["side"] in {SIDE_RADIANT, SIDE_DIRE}
    ]
    draft_by_side: dict[str, set[int]] = {
        SIDE_RADIANT: set(),
        SIDE_DIRE: set(),
    }
    for event in picks:
        draft_by_side[str(event["side"])].add(int(event["hero_id"]))

    players_by_side: dict[str, list[int]] = {
        SIDE_RADIANT: [],
        SIDE_DIRE: [],
    }
    incomplete_players = False
    unknown_player_hero = False
    for row in player_rows:
        side = row.get("side")
        hero = row.get("hero_id")
        if side not in {SIDE_RADIANT, SIDE_DIRE}:
            incomplete_players = True
            continue
        if hero is None or pd.isna(hero) or int(hero) <= 0:
            unknown_player_hero = True
            continue
        players_by_side[str(side)].append(int(hero))

    for side in (SIDE_RADIANT, SIDE_DIRE):
        if len(players_by_side[side]) != PICKS_PER_SIDE:
            incomplete_players = True

    player_sets = {side: set(heroes) for side, heroes in players_by_side.items()}
    player_duplicates = any(
        len(heroes) != len(set(heroes)) for heroes in players_by_side.values()
    )

    if incomplete_players or unknown_player_hero:
        status = "incomparable_incomplete_players"
    elif (
        draft_by_side[SIDE_RADIANT] == player_sets[SIDE_RADIANT]
        and draft_by_side[SIDE_DIRE] == player_sets[SIDE_DIRE]
    ):
        status = "exact"
    else:
        status = "mismatch"

    return {
        "status": status,
        "draft_radiant": sorted(draft_by_side[SIDE_RADIANT]),
        "draft_dire": sorted(draft_by_side[SIDE_DIRE]),
        "player_radiant": sorted(player_sets[SIDE_RADIANT]),
        "player_dire": sorted(player_sets[SIDE_DIRE]),
        "missing_radiant": sorted(
            player_sets[SIDE_RADIANT] - draft_by_side[SIDE_RADIANT]
        ),
        "extra_radiant": sorted(
            draft_by_side[SIDE_RADIANT] - player_sets[SIDE_RADIANT]
        ),
        "missing_dire": sorted(player_sets[SIDE_DIRE] - draft_by_side[SIDE_DIRE]),
        "extra_dire": sorted(draft_by_side[SIDE_DIRE] - player_sets[SIDE_DIRE]),
        "side_swap": (
            draft_by_side[SIDE_RADIANT] == player_sets[SIDE_DIRE]
            and draft_by_side[SIDE_DIRE] == player_sets[SIDE_RADIANT]
        ),
        "player_duplicates": player_duplicates,
        "incomplete_players": incomplete_players,
        "unknown_player_hero": unknown_player_hero,
    }


def classify_match_draft_category(
    events: Sequence[Mapping[str, Any]],
    *,
    terminal: Mapping[str, Any] | None = None,
) -> str:
    """Assign one explicit problematic/usable category per match."""
    integrity = _sequence_integrity(events)
    if (
        integrity["null_sequence_count"]
        or integrity["duplicate_sequences"]
        or not integrity["gap_free_zero_indexed"]
        or integrity["non_monotonic_input_order"]
    ):
        return CATEGORY_MALFORMED_ORDERING
    if (
        integrity["unknown_action_count"]
        or integrity["unknown_side_count"]
        or integrity["unknown_hero_count"]
    ):
        return CATEGORY_UNKNOWN_FIELDS
    if terminal is None:
        terminal = compare_terminal_picks_to_players(events, [])
    if terminal["status"] == "mismatch":
        return CATEGORY_TERMINAL_MISMATCH
    if integrity["n_successful_picks"] != EXPECTED_SUCCESSFUL_PICKS:
        return CATEGORY_INCOMPLETE_PICKS
    if (
        integrity["n_radiant_picks"] != PICKS_PER_SIDE
        or integrity["n_dire_picks"] != PICKS_PER_SIDE
        or integrity["repeated_successful_pick"]
        or integrity["same_hero_both_sides"]
        or integrity["pick_after_successful_ban"]
    ):
        return CATEGORY_INCOMPLETE_PICKS
    if integrity["n_successful_bans"] == 0:
        return CATEGORY_COMPLETE_PICKS_INCOMPLETE_BANS
    return CATEGORY_COMPLETE_SEQUENTIAL


# ---------------------------------------------------------------------------
# Prefix state construction
# ---------------------------------------------------------------------------


def build_draft_prefix_state(
    *,
    match_id: int,
    start_time: datetime | pd.Timestamp,
    game_version_id: int | None,
    boundary_t: int,
    events: Sequence[Mapping[str, Any]],
    radiant_team_id: int,
    dire_team_id: int,
    radiant_player_ids: Sequence[int],
    dire_player_ids: Sequence[int],
    radiant_team_elo: float | None = None,
    dire_team_elo: float | None = None,
    team_elo_delta: float | None = None,
) -> dict[str, Any]:
    """Build ``S_(M,t)``: state before event ``sequence == boundary_t``.

    ``events`` must be the match's full canonical event list (any order);
    only rows with ``sequence < boundary_t`` enter the prefix. Raises if
    duplicate ``sequence`` values appear among included events.
    """
    if boundary_t < 0:
        raise ValueError(f"boundary_t must be >= 0, got {boundary_t}")

    normalized = [_normalize_event_row(e) for e in events]
    included = [e for e in normalized if e["sequence"] is not None and e["sequence"] < boundary_t]
    sequences = [int(e["sequence"]) for e in included]
    if len(sequences) != len(set(sequences)):
        raise ValueError(
            f"match_id={match_id}: duplicate sequence values in prefix "
            f"for boundary_t={boundary_t}"
        )
    included.sort(key=lambda e: int(e["sequence"]))

    radiant_picks: list[int] = []
    dire_picks: list[int] = []
    radiant_bans: list[int] = []
    dire_bans: list[int] = []
    unsuccessful: list[dict[str, Any]] = []

    for event in included:
        action = event["action"]
        side = event["side"]
        hero = event["hero_id"]
        successful_flag = event["was_successful"]
        if action == ACTION_BAN and successful_flag is False:
            unsuccessful.append(dict(event))
            continue
        if hero is None:
            continue
        if action == ACTION_PICK:
            if side == SIDE_RADIANT:
                radiant_picks.append(hero)
            elif side == SIDE_DIRE:
                dire_picks.append(hero)
        elif action == ACTION_BAN and event_is_actual(action, successful_flag):
            if side == SIDE_RADIANT:
                radiant_bans.append(hero)
            elif side == SIDE_DIRE:
                dire_bans.append(hero)

    n_events = len(normalized)
    return {
        "match_id": int(match_id),
        "start_time": start_time,
        "game_version_id": (
            None if game_version_id is None or pd.isna(game_version_id) else int(game_version_id)
        ),
        "boundary_t": int(boundary_t),
        "n_prior_events": len(included),
        "is_terminal": int(boundary_t) == n_events,
        "radiant_team_id": int(radiant_team_id),
        "dire_team_id": int(dire_team_id),
        "radiant_player_ids": tuple(int(x) for x in radiant_player_ids),
        "dire_player_ids": tuple(int(x) for x in dire_player_ids),
        "radiant_team_elo": radiant_team_elo,
        "dire_team_elo": dire_team_elo,
        "team_elo_delta": team_elo_delta,
        "event_prefix": [dict(e) for e in included],
        "radiant_pick_hero_ids": tuple(radiant_picks),
        "dire_pick_hero_ids": tuple(dire_picks),
        "radiant_ban_hero_ids": tuple(radiant_bans),
        "dire_ban_hero_ids": tuple(dire_bans),
        "unsuccessful_event_prefix": unsuccessful,
        "n_radiant_picks": len(radiant_picks),
        "n_dire_picks": len(dire_picks),
        "n_radiant_bans_successful": len(radiant_bans),
        "n_dire_bans_successful": len(dire_bans),
        "n_unsuccessful_events": len(unsuccessful),
    }


def build_sequential_draft_states(
    *,
    match_id: int,
    start_time: datetime | pd.Timestamp,
    game_version_id: int | None,
    events: Sequence[Mapping[str, Any]],
    radiant_team_id: int,
    dire_team_id: int,
    radiant_player_ids: Sequence[int],
    dire_player_ids: Sequence[int],
    radiant_team_elo: float | None = None,
    dire_team_elo: float | None = None,
    team_elo_delta: float | None = None,
    reject_duplicate_sequences: bool = True,
) -> list[dict[str, Any]]:
    """Build ``S_(M,t)`` for ``t = 0 .. N`` including the terminal boundary.

    When ``reject_duplicate_sequences`` is True (default), duplicate
    ``(match_id, sequence)`` values raise ``ValueError``. Callers that
    want to classify rather than build may set it False and inspect
    integrity first.
    """
    normalized = [_normalize_event_row(e) for e in events]
    sequences = [e["sequence"] for e in normalized if e["sequence"] is not None]
    if reject_duplicate_sequences and len(sequences) != len(set(sequences)):
        raise ValueError(
            f"match_id={match_id}: duplicate sequence values "
            f"{sorted(seq for seq, c in Counter(sequences).items() if c > 1)}"
        )
    n = len(normalized)
    return [
        build_draft_prefix_state(
            match_id=match_id,
            start_time=start_time,
            game_version_id=game_version_id,
            boundary_t=t,
            events=normalized,
            radiant_team_id=radiant_team_id,
            dire_team_id=dire_team_id,
            radiant_player_ids=radiant_player_ids,
            dire_player_ids=dire_player_ids,
            radiant_team_elo=radiant_team_elo,
            dire_team_elo=dire_team_elo,
            team_elo_delta=team_elo_delta,
        )
        for t in range(n + 1)
    ]


# ---------------------------------------------------------------------------
# Population audit
# ---------------------------------------------------------------------------


def _players_by_match(players: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in players.to_dict(orient="records"):
        out[int(row["match_id"])].append(row)
    return out


def _events_by_match(draft_events: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in draft_events.to_dict(orient="records"):
        out[int(row["match_id"])].append(row)
    return out


def audit_draft_events_population(
    matches: pd.DataFrame,
    draft_events: pd.DataFrame,
    players: pd.DataFrame,
    *,
    development_end: datetime | None = None,
) -> dict[str, Any]:
    """Characterize ``draft_events`` on the development population only."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    development = restrict_development(matches, development_end=end)
    holdout_n = int(len(matches) - len(development))
    dev_ids = set(int(x) for x in development["match_id"].tolist())
    events = draft_events.loc[draft_events["match_id"].isin(dev_ids)].copy()
    player_dev = players.loc[players["match_id"].isin(dev_ids)].copy()

    events_by_match = _events_by_match(events)
    players_by_match = _players_by_match(player_dev)

    matches_total = int(len(development))
    matches_with_any = sum(1 for mid in dev_ids if mid in events_by_match)
    pick_count = Counter()
    ban_count = Counter()
    event_count = Counter()
    unsuccessful_total = 0
    duplicate_pair_rows = 0
    missing_sequence_rows = 0
    malformed_matches = 0
    unknown_action_rows = 0
    unknown_side_rows = 0
    unknown_hero_rows = 0
    repeated_pick_matches = 0
    both_sides_matches = 0
    pick_after_ban_matches = 0
    terminal_status = Counter()
    categories = Counter()
    year_total: Counter[int] = Counter()
    year_with_draft: Counter[int] = Counter()
    version_total: Counter[Any] = Counter()
    version_with_draft: Counter[Any] = Counter()
    mapper_total: Counter[Any] = Counter()
    signatures_24: Counter[str] = Counter()
    signatures_other: Counter[str] = Counter()

    # Duplicate (match_id, sequence) at table grain.
    if not events.empty:
        dup_mask = events.duplicated(subset=["match_id", "sequence"], keep=False)
        duplicate_pair_rows = int(dup_mask.sum())
        missing_sequence_rows = int(events["sequence"].isna().sum())
        unknown_action_rows = int(
            (~events["action"].isin([ACTION_PICK, ACTION_BAN])).sum()
        )
        unknown_side_rows = int(
            (~events["side"].isin([SIDE_RADIANT, SIDE_DIRE])).sum()
        )
        hero = pd.to_numeric(events["hero_id"], errors="coerce")
        unknown_hero_rows = int((hero.isna() | (hero <= 0)).sum())
        unsuccessful_total = int(
            (
                (events["action"] == ACTION_BAN)
                & (events["was_successful"] == False)  # noqa: E712
            ).sum()
        )

    for row in development.to_dict(orient="records"):
        mid = int(row["match_id"])
        year = int(pd.Timestamp(row["start_time"]).year)
        year_total[year] += 1
        gv = row.get("game_version_id")
        version_total[gv] += 1
        if "mapper_version" in row:
            mapper_total[row["mapper_version"]] += 1

        match_events = events_by_match.get(mid, [])
        if match_events:
            year_with_draft[year] += 1
            version_with_draft[gv] += 1

        integrity = _sequence_integrity(match_events)
        pick_count[integrity["n_successful_picks"]] += 1
        ban_count[integrity["n_successful_bans"]] += 1
        event_count[integrity["n_events"]] += 1
        if not integrity["gap_free_zero_indexed"] or integrity["duplicate_sequences"]:
            malformed_matches += 1
        if integrity["repeated_successful_pick"]:
            repeated_pick_matches += 1
        if integrity["same_hero_both_sides"]:
            both_sides_matches += 1
        if integrity["pick_after_successful_ban"]:
            pick_after_ban_matches += 1

        terminal = compare_terminal_picks_to_players(
            match_events, players_by_match.get(mid, [])
        )
        terminal_status[terminal["status"]] += 1
        category = classify_match_draft_category(match_events, terminal=terminal)
        categories[category] += 1

        if match_events and integrity["gap_free_zero_indexed"]:
            ordered = sorted(match_events, key=lambda e: int(e["sequence"]))
            sig = "".join(
                f"{str(e['action'])[0]}{str(e['side'])[0]}" for e in ordered
            )
            if len(ordered) == MODAL_EVENT_COUNT:
                signatures_24[sig] += 1
            else:
                signatures_other[sig] += 1

    n_pick_ge_10 = sum(
        n for k, n in pick_count.items() if k >= EXPECTED_SUCCESSFUL_PICKS
    )
    n_pick_eq_10 = pick_count[EXPECTED_SUCCESSFUL_PICKS]
    n_pick_lt_10 = sum(
        n for k, n in pick_count.items() if k < EXPECTED_SUCCESSFUL_PICKS
    )
    n_pick_gt_10 = sum(
        n for k, n in pick_count.items() if k > EXPECTED_SUCCESSFUL_PICKS
    )

    ordering_ok_rate = (
        1.0 - (malformed_matches / matches_total) if matches_total else float("nan")
    )
    field_usable_rate = (
        1.0
        - (
            (unknown_action_rows + unknown_side_rows + unknown_hero_rows)
            / max(int(len(events)), 1)
        )
        if matches_total
        else float("nan")
    )
    terminal_exact = terminal_status.get("exact", 0)
    terminal_exact_rate = (
        terminal_exact / matches_total if matches_total else float("nan")
    )
    well_formed = (
        categories[CATEGORY_COMPLETE_SEQUENTIAL]
        + categories[CATEGORY_COMPLETE_PICKS_INCOMPLETE_BANS]
    )
    well_formed_share = well_formed / matches_total if matches_total else float("nan")

    return {
        "development_end": end,
        "matches_total": matches_total,
        "holdout_matches_excluded": holdout_n,
        "matches_with_any_draft_events": matches_with_any,
        "matches_with_at_least_10_successful_picks": n_pick_ge_10,
        "matches_with_exactly_10_successful_picks": n_pick_eq_10,
        "matches_with_fewer_than_10_successful_picks": n_pick_lt_10,
        "matches_with_more_than_10_successful_picks": n_pick_gt_10,
        "successful_pick_count_distribution": dict(sorted(pick_count.items())),
        "successful_ban_count_distribution": dict(sorted(ban_count.items())),
        "event_count_distribution": dict(sorted(event_count.items())),
        "unsuccessful_draft_events": unsuccessful_total,
        "duplicate_match_id_sequence_rows": duplicate_pair_rows,
        "missing_sequence_rows": missing_sequence_rows,
        "malformed_sequence_matches": malformed_matches,
        "unknown_action_rows": unknown_action_rows,
        "unknown_side_rows": unknown_side_rows,
        "unknown_hero_rows": unknown_hero_rows,
        "repeated_successful_pick_matches": repeated_pick_matches,
        "same_hero_both_sides_matches": both_sides_matches,
        "pick_after_successful_ban_matches": pick_after_ban_matches,
        "terminal_status_counts": dict(terminal_status),
        "terminal_exact_rate": terminal_exact_rate,
        "category_counts": dict(categories),
        "coverage_by_year": {
            "total": dict(sorted(year_total.items())),
            "with_draft": dict(sorted(year_with_draft.items())),
        },
        "coverage_by_game_version_id": {
            "total": {str(k): v for k, v in sorted(version_total.items(), key=lambda x: (x[0] is None, x[0]))},
            "with_draft": {
                str(k): v
                for k, v in sorted(
                    version_with_draft.items(), key=lambda x: (x[0] is None, x[0])
                )
            },
        },
        "coverage_by_mapper_version": {
            str(k): v for k, v in sorted(mapper_total.items(), key=lambda x: (x[0] is None, x[0]))
        },
        "action_side_signatures_24": dict(signatures_24.most_common()),
        "action_side_signatures_other_top": dict(signatures_other.most_common(20)),
        "ordering_ok_rate": ordering_ok_rate,
        "field_usable_rate": field_usable_rate,
        "well_formed_share": well_formed_share,
        "modal_event_count": MODAL_EVENT_COUNT,
        "modal_ban_count": MODAL_BAN_COUNT,
        "boundary_convention": BOUNDARY_CONVENTION,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_slice26(
    *,
    ordering_ok_rate: float,
    field_usable_rate: float,
    terminal_exact_rate: float,
    well_formed_share: float,
    partial_regime_label: str | None = None,
) -> pd.DataFrame:
    """Classify Slice 26 on data/state reliability gates only."""
    gates_a = (
        ordering_ok_rate >= MIN_ORDERING_OK_RATE
        and field_usable_rate >= MIN_FIELD_USABLE_RATE
        and terminal_exact_rate >= MIN_TERMINAL_EXACT_RATE
        and well_formed_share >= MIN_WELL_FORMED_SHARE
    )
    gates_b = (
        ordering_ok_rate >= 0.9
        and field_usable_rate >= 0.9
        and terminal_exact_rate >= 0.9
        and well_formed_share >= 0.5
    )
    if gates_a:
        classification = "A"
        gate = CLASSIFICATION_A
        frozen = list(SLICE26_FROZEN_COMPONENTS)
        next_slice = (
            "Use the frozen sequential draft-state substrate for a later "
            "next-pick / draft-policy research slice. Do not revive "
            "Slice 23–25 rejected states as frozen inputs."
        )
    elif gates_b:
        classification = "B"
        gate = CLASSIFICATION_B
        frozen = [
            "boundary_convention_before_event_t",
            "ordered_event_prefix_state",
        ]
        if partial_regime_label:
            frozen.append(partial_regime_label)
        next_slice = (
            "Freeze only the supported sequential regime. Expand coverage "
            "diagnostics before a full next-pick model."
        )
    else:
        classification = "C"
        gate = CLASSIFICATION_C
        frozen = []
        next_slice = (
            "Do not freeze sequential draft state. Investigate source "
            "ordering/identity before draft modeling."
        )
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "gate": gate,
                "frozen_components": tuple(frozen),
                "ordering_ok_rate": ordering_ok_rate,
                "field_usable_rate": field_usable_rate,
                "terminal_exact_rate": terminal_exact_rate,
                "well_formed_share": well_formed_share,
                "next_slice": next_slice,
                "next_event_prediction_run": False,
                "win_model_run": False,
                "feature_columns_unchanged": len(FEATURE_COLUMNS) == 33,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Diagnostics driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slice26DiagnosticReport:
    """Development-only Slice 26 sequential draft-state audit."""

    development_end: datetime
    n_development_matches: int
    n_holdout_excluded: int
    audit: dict[str, Any]
    classification: pd.DataFrame
    recommended_modeling_categories: tuple[str, ...]
    phase_format_notes: str
    hero_catalog_notes: str
    state_schema: tuple[str, ...]
    boundary_convention: str
    feature_columns_length: int
    recorded_classification: str
    frozen_components: tuple[str, ...]


def _phase_format_notes(audit: Mapping[str, Any]) -> str:
    n_sig = len(audit.get("action_side_signatures_24") or {})
    return (
        "No explicit draft-format or phase column exists on canonical "
        f"draft_events. Among {MODAL_EVENT_COUNT}-event matches, "
        f"{n_sig} distinct action/side order signatures appear; these "
        "resemble Captain's Mode first/second-pick variants but are not "
        "assigned semantic phase labels in Slice 26 v1. Raw ordered "
        "prefixes are preserved without hard-coding one modern CM pattern "
        "across history. Pick-only (0-ban) and atypical ban-count matches "
        "are classified explicitly rather than forced into CM phases."
    )


def _hero_catalog_notes() -> str:
    return (
        "Slice 26 stores heroes already picked/banned in the ordered "
        "prefix. It does not define remaining legal heroes as the "
        "complement of heroes observed in professional matches before T. "
        "A later slice may introduce a patch-aware legal hero registry."
    )


def run_sequential_draft_state_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice26DiagnosticReport:
    """Audit draft sequential integrity and classify Slice 26.

    Holdout matches (``start_time > FROZEN_DEVELOPMENT_END``) are
    excluded from every summary. No next-event predictive metrics are
    computed.
    """
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    matches = store.sql(
        f"""
        SELECT
            match_id,
            start_time,
            game_version_id,
            radiant_team_id,
            dire_team_id,
            radiant_player_0_id,
            radiant_player_1_id,
            radiant_player_2_id,
            radiant_player_3_id,
            radiant_player_4_id,
            dire_player_0_id,
            dire_player_1_id,
            dire_player_2_id,
            dire_player_3_id,
            dire_player_4_id,
            radiant_win,
            duration_seconds,
            mapper_version
        FROM {MATCHES_VIEW}
        """
    ).df()
    draft_events = store.sql(
        f"""
        SELECT match_id, sequence, action, side, hero_id, was_successful
        FROM {DRAFT_EVENTS_VIEW}
        """
    ).df()
    players = store.sql(
        f"""
        SELECT match_id, side, slot_in_side, player_id, hero_id, team_id
        FROM {MATCH_PLAYERS_VIEW}
        """
    ).df()

    # Elo uses full chronological history for causal pre-match ratings.
    # Ratings are available to attach onto S_(M,t) via build_* helpers;
    # the integrity audit itself does not require materializing every
    # boundary row.
    elo = compute_team_elo_features(matches, config=elo_config)
    development = restrict_development(matches, development_end=end)
    elo_dev = development[["match_id"]].merge(elo, on="match_id", how="left")
    elo_coverage = float(elo_dev[TEAM_ELO_FEATURE_COLUMNS[0]].notna().mean())

    audit = audit_draft_events_population(
        matches, draft_events, players, development_end=end
    )
    audit["pre_match_team_elo_coverage"] = elo_coverage
    audit["team_elo_feature_columns"] = list(TEAM_ELO_FEATURE_COLUMNS)

    classification = classify_slice26(
        ordering_ok_rate=float(audit["ordering_ok_rate"]),
        field_usable_rate=float(audit["field_usable_rate"]),
        terminal_exact_rate=float(audit["terminal_exact_rate"]),
        well_formed_share=float(audit["well_formed_share"]),
    )
    recorded = str(classification.iloc[0]["classification"])
    frozen = tuple(classification.iloc[0]["frozen_components"])

    recommended = (
        CATEGORY_COMPLETE_SEQUENTIAL,
        CATEGORY_COMPLETE_PICKS_INCOMPLETE_BANS,
    )

    return Slice26DiagnosticReport(
        development_end=end,
        n_development_matches=int(audit["matches_total"]),
        n_holdout_excluded=int(audit["holdout_matches_excluded"]),
        audit=audit,
        classification=classification,
        recommended_modeling_categories=recommended,
        phase_format_notes=_phase_format_notes(audit),
        hero_catalog_notes=_hero_catalog_notes(),
        state_schema=SLICE26_STATE_KEYS,
        boundary_convention=BOUNDARY_CONVENTION,
        feature_columns_length=len(FEATURE_COLUMNS),
        recorded_classification=recorded,
        frozen_components=frozen,
    )


def slice26_report_to_jsonable(report: Slice26DiagnosticReport) -> dict[str, Any]:
    """JSON-serializable Slice 26 report."""
    return {
        "slice": 26,
        "title": "causal sequential draft-state dataset",
        "diagnostic_only": SLICE26_DIAGNOSTIC_ONLY,
        "development_end": report.development_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_holdout_excluded": report.n_holdout_excluded,
        "boundary_convention": report.boundary_convention,
        "state_schema": list(report.state_schema),
        "audit": {key: _jsonable_value(value) for key, value in report.audit.items()},
        "classification": report.classification.to_dict(orient="records"),
        "recommended_modeling_categories": list(report.recommended_modeling_categories),
        "phase_format_notes": report.phase_format_notes,
        "hero_catalog_notes": report.hero_catalog_notes,
        "feature_columns_length": report.feature_columns_length,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "recorded_classification": report.recorded_classification,
        "frozen_components": list(report.frozen_components),
        "excluded_from_slice": [
            "next_hero_log_loss",
            "next_pick_accuracy",
            "win_probability",
            "draft_value_movement",
            "synergy_counter_effects",
            "assignment_entropy",
            "flex_scores",
            "slice23_compatibility",
            "slice24_current_hxP",
            "slice25_pxrxh_pool",
            "player_hero_assignment",
            "observed_position",
            "legal_hero_complement",
        ],
        "team_elo_feature_columns": list(TEAM_ELO_FEATURE_COLUMNS),
    }
