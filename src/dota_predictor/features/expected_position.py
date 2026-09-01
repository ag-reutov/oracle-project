"""Leakage-safe PRE_DRAFT expected-position assignment.

Grain
-----
One row per `(match_id, player_id)`. `expected_position` is a joint
1–5 assignment for the five current players on a side. It is inferred
only from Slice 2 strictly-prior historical evidence.

Observed vs expected
--------------------
`observed_position` is the current match's STRATZ parse label
(POST_MATCH). It is copied for evaluation only and is never an input to
scoring or assignment.

`expected_position` is PRE_DRAFT inference: what the roster is expected
to play before the draft, using `H.start_time < M.start_time` history.

Scoring
-------
`score(player, candidate_position)` is a deterministic non-negative
evidence score. No weights are fit on current-match observed positions.
Methods:

* ``previous``: 1 if the candidate equals ``previous_explicit_position``,
  else 0. All zeros when the player has no prior explicit position.
* ``recent_5`` / ``recent_10`` / ``recent_20``: share of explicit
  positions in that trailing window (0 if the window has none).
* ``career``: career ``prior_share_position_N`` (0 if no explicit history).
* ``same_version``: share of explicit positions in the same STRATZ
  ``game_version_id`` (0 if none).
* ``hierarchical``: recent-10 shares if that window has any explicit
  history, else recent-20, else career. No mixing weights. This is
  distinct from ``previous`` (last explicit match as a one-hot).

Joint assignment
----------------
For each `(match_id, side)` the five players are assigned to
{POSITION_1..POSITION_5} by maximizing the sum of scores over the 120
permutations. Ties break by lexicographically smallest assignment tuple
after sorting players by ``player_id`` (stable PRE_DRAFT identity).
Current observed STRATZ position is not used.

Cold start
----------
A player with no explicit prior history contributes an all-zero score
row. They still receive a leftover position so the side is a permutation.
Confidence descriptors (assigned score, player margin, roster margin,
prior explicit games) are 0 / NULL as appropriate — no fabricated
confidence.

This module does not write Parquet, does not bump schema versions, does
not alter Elo or the win-model feature matrix, and does not fill
canonical NULL observed positions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import FeatureDuckDBConnection
from dota_predictor.features.player_position import (
    EXPLICIT_POSITION_LABELS,
    build_player_position_state,
)

__all__ = [
    "DEFAULT_EXPECTED_POSITION_METHOD",
    "EXPECTED_POSITION_COLUMNS",
    "EXPECTED_POSITION_EVIDENCE_COLUMNS",
    "EXPECTED_POSITION_METHODS",
    "LEAKAGE_COLUMNS",
    "NAIVE_PREFERENCE_SOURCES",
    "ExpectedPosition",
    "assign_expected_positions",
    "audit_naive_roster_uniqueness",
    "build_expected_position",
    "evaluate_expected_position",
    "player_position_scores",
]

N_POSITIONS = 5
POSITION_INDEX: dict[str, int] = {
    label: i for i, label in enumerate(EXPLICIT_POSITION_LABELS)
}
_ALL_PERMS: tuple[tuple[int, ...], ...] = tuple(permutations(range(N_POSITIONS)))
# itertools.permutations is lexicographic, so the first max-total index
# among this array is the (-total, perm) tie-break.
_PERM_ARRAY = np.array(_ALL_PERMS, dtype=np.intp)

EXPECTED_POSITION_METHODS: tuple[str, ...] = (
    "previous",
    "recent_5",
    "recent_10",
    "recent_20",
    "career",
    "same_version",
    "hierarchical",
)
# Career position shares plus joint assignment: simplest rule whose
# accuracy is effectively tied with recent-window methods on the
# processed dataset. See scripts/evaluate_expected_position.py.
DEFAULT_EXPECTED_POSITION_METHOD = "career"

NAIVE_PREFERENCE_SOURCES: tuple[str, ...] = (
    "previous_explicit_position",
    "recent_5_modal_position",
    "recent_10_modal_position",
    "recent_20_modal_position",
    "historical_modal_position",
)

# Current-match POST_MATCH / DRAFT columns that must not enter scoring.
LEAKAGE_COLUMNS: tuple[str, ...] = (
    "position",
    "lane",
    "role",
    "hero_id",
    "won",
    "slot_in_side",
)

EXPECTED_POSITION_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "previous_explicit_position",
    "assigned_position_score",
    "player_score_margin",
    "roster_assignment_margin",
    "prior_explicit_position_games",
    "recent_position_stability",
    "method",
    "evidence_tier",
)

EXPECTED_POSITION_COLUMNS: tuple[str, ...] = (
    "match_id",
    "player_id",
    "start_time",
    "game_version_id",
    "team_id",
    "side",
    "expected_position",
    *EXPECTED_POSITION_EVIDENCE_COLUMNS,
    "observed_position",
)


def _suffix(label: str) -> str:
    return label.lower()


def _as_float(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if pd.isna(value):
        return 0.0
    return float(value)


def _explicit_label(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    text = str(value)
    return text if text in POSITION_INDEX else None


def _numeric_column(rows: pd.DataFrame, column: str) -> np.ndarray:
    if column not in rows.columns:
        return np.zeros(len(rows), dtype=np.float64)
    return pd.to_numeric(rows[column], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float64
    )


def _column_matrix(rows: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.column_stack([_numeric_column(rows, column) for column in columns])


def _share_matrix(
    rows: pd.DataFrame, *, games_prefix: str, explicit_column: str
) -> np.ndarray:
    explicit = _numeric_column(rows, explicit_column)
    counts = _column_matrix(
        rows, [f"{games_prefix}_{_suffix(label)}" for label in EXPLICIT_POSITION_LABELS]
    )
    scores = np.zeros((len(rows), N_POSITIONS), dtype=np.float64)
    mask = explicit > 0
    if mask.any():
        scores[mask] = counts[mask] / explicit[mask, None]
    return scores


def _career_matrix(rows: pd.DataFrame) -> np.ndarray:
    return _column_matrix(
        rows, [f"prior_share_{_suffix(label)}" for label in EXPLICIT_POSITION_LABELS]
    )


def _version_matrix(rows: pd.DataFrame) -> np.ndarray:
    counts = _column_matrix(
        rows,
        [f"version_prior_games_{_suffix(label)}" for label in EXPLICIT_POSITION_LABELS],
    )
    totals = counts.sum(axis=1)
    scores = np.zeros_like(counts)
    mask = totals > 0
    if mask.any():
        scores[mask] = counts[mask] / totals[mask, None]
    return scores


def _evidence_tiers(rows: pd.DataFrame, *, method: str) -> list[str]:
    n = len(rows)
    if method == "previous":
        return [
            "previous" if _explicit_label(value) is not None else "none"
            for value in rows["previous_explicit_position"].tolist()
        ]
    if method.startswith("recent_"):
        column = f"{method}_explicit_games"
        if column not in rows.columns:
            return ["none"] * n
        return [
            method if _as_float(value) > 0 else "none" for value in rows[column].tolist()
        ]
    if method == "career":
        if "prior_explicit_position_games" not in rows.columns:
            return ["none"] * n
        return [
            "career" if _as_float(value) > 0 else "none"
            for value in rows["prior_explicit_position_games"].tolist()
        ]
    if method == "same_version":
        totals = _column_matrix(
            rows,
            [
                f"version_prior_games_{_suffix(label)}"
                for label in EXPLICIT_POSITION_LABELS
            ],
        ).sum(axis=1)
        return ["same_version" if total > 0 else "none" for total in totals]
    if method == "hierarchical":
        r10 = _numeric_column(rows, "recent_10_explicit_games")
        r20 = _numeric_column(rows, "recent_20_explicit_games")
        career = _numeric_column(rows, "prior_explicit_position_games")
        tiers: list[str] = []
        for i in range(n):
            if r10[i] > 0:
                tiers.append("recent_10")
            elif r20[i] > 0:
                tiers.append("recent_20")
            elif career[i] > 0:
                tiers.append("career")
            else:
                tiers.append("none")
        return tiers
    raise ValueError(
        f"unknown expected-position method {method!r}; "
        f"expected one of {EXPECTED_POSITION_METHODS}"
    )


def player_position_scores(rows: pd.DataFrame, *, method: str) -> np.ndarray:
    """Return `(n_players, 5)` scores. Drops current-match leakage columns."""
    if method not in EXPECTED_POSITION_METHODS:
        raise ValueError(
            f"unknown expected-position method {method!r}; "
            f"expected one of {EXPECTED_POSITION_METHODS}"
        )
    if rows.empty:
        return np.zeros((0, N_POSITIONS), dtype=np.float64)
    safe = rows.drop(columns=[c for c in LEAKAGE_COLUMNS if c in rows.columns])
    if method == "previous":
        scores = np.zeros((len(safe), N_POSITIONS), dtype=np.float64)
        for i, value in enumerate(safe["previous_explicit_position"].tolist()):
            label = _explicit_label(value)
            if label is not None:
                scores[i, POSITION_INDEX[label]] = 1.0
        return scores
    if method == "recent_5":
        return _share_matrix(
            safe, games_prefix="recent_5_games", explicit_column="recent_5_explicit_games"
        )
    if method == "recent_10":
        return _share_matrix(
            safe,
            games_prefix="recent_10_games",
            explicit_column="recent_10_explicit_games",
        )
    if method == "recent_20":
        return _share_matrix(
            safe,
            games_prefix="recent_20_games",
            explicit_column="recent_20_explicit_games",
        )
    if method == "career":
        return _career_matrix(safe)
    if method == "same_version":
        return _version_matrix(safe)
    if method == "hierarchical":
        recent_10 = _share_matrix(
            safe,
            games_prefix="recent_10_games",
            explicit_column="recent_10_explicit_games",
        )
        recent_20 = _share_matrix(
            safe,
            games_prefix="recent_20_games",
            explicit_column="recent_20_explicit_games",
        )
        career = _career_matrix(safe)
        out = career.copy()
        use_20 = (recent_10.sum(axis=1) <= 0) & (recent_20.sum(axis=1) > 0)
        use_10 = recent_10.sum(axis=1) > 0
        out[use_20] = recent_20[use_20]
        out[use_10] = recent_10[use_10]
        return out
    raise ValueError(
        f"unknown expected-position method {method!r}; "
        f"expected one of {EXPECTED_POSITION_METHODS}"
    )


def _assign_sides(
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized joint assignment for ``(n_sides, 5, 5)`` score tensors.

    Returns assigned position indices, assigned scores, player margins, and
    per-side roster margins. Ties break to the lexicographically smallest
    permutation because ``_PERM_ARRAY`` is already in lex order.
    """
    if scores.ndim != 3 or scores.shape[1:] != (N_POSITIONS, N_POSITIONS):
        raise ValueError(f"expected (n, 5, 5) score tensor, got {scores.shape}")
    totals = np.zeros((scores.shape[0], _PERM_ARRAY.shape[0]), dtype=np.float64)
    for i in range(N_POSITIONS):
        totals += scores[:, i, _PERM_ARRAY[:, i]]
    best_total = totals.max(axis=1, keepdims=True)
    is_best = totals == best_total
    best_idx = is_best.argmax(axis=1)
    best_perm = _PERM_ARRAY[best_idx]
    excluded = np.where(is_best, -np.inf, totals)
    second = excluded.max(axis=1)
    roster_margin = best_total.ravel() - np.where(
        np.isfinite(second), second, best_total.ravel()
    )
    side_idx = np.arange(scores.shape[0])[:, None]
    player_idx = np.arange(N_POSITIONS)[None, :]
    assigned_scores = scores[side_idx, player_idx, best_perm]
    other = scores.copy()
    other[side_idx, player_idx, best_perm] = -np.inf
    second_best = other.max(axis=2)
    player_margins = assigned_scores - second_best
    return best_perm, assigned_scores, player_margins, roster_margin


def _assign_side(
    player_ids: np.ndarray, scores: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Maximize total score; ties → lexicographically smallest perm.

    ``player_ids`` must already be sorted ascending. ``scores`` is
    aligned `(5, 5)`.
    """
    if scores.shape != (N_POSITIONS, N_POSITIONS):
        raise ValueError(f"expected 5x5 score matrix, got {scores.shape}")
    if len(player_ids) != N_POSITIONS:
        raise ValueError(f"expected 5 player_ids, got {len(player_ids)}")
    perm, assigned_scores, player_margins, roster_margin = _assign_sides(
        scores[np.newaxis, :, :]
    )
    assigned = [EXPLICIT_POSITION_LABELS[j] for j in perm[0]]
    return assigned, assigned_scores[0], player_margins[0], float(roster_margin[0])


def assign_expected_positions(
    state: pd.DataFrame, *, method: str = DEFAULT_EXPECTED_POSITION_METHOD
) -> pd.DataFrame:
    """Jointly assign expected 1–5 positions from Slice 2 historical state."""
    if method not in EXPECTED_POSITION_METHODS:
        raise ValueError(
            f"unknown expected-position method {method!r}; "
            f"expected one of {EXPECTED_POSITION_METHODS}"
        )
    if state.empty:
        return pd.DataFrame(columns=list(EXPECTED_POSITION_COLUMNS))

    ordered = state.sort_values(
        ["match_id", "side", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    sizes = ordered.groupby(["match_id", "side"], sort=False).size()
    if not bool((sizes == N_POSITIONS).all()):
        match_id, side = sizes[sizes != N_POSITIONS].index[0]
        raise ValueError(
            f"side ({match_id}, {side}) has {int(sizes.loc[(match_id, side)])} "
            "players; expected 5"
        )
    n_sides = len(sizes)
    scores = player_position_scores(ordered, method=method).reshape(
        n_sides, N_POSITIONS, N_POSITIONS
    )
    perm, assigned_scores, margins, roster_margin = _assign_sides(scores)
    labels = np.array(EXPLICIT_POSITION_LABELS, dtype=object)
    frame = pd.DataFrame(
        {
            "match_id": ordered["match_id"].to_numpy(),
            "player_id": ordered["player_id"].to_numpy(),
            "start_time": ordered["start_time"].to_numpy(),
            "game_version_id": ordered["game_version_id"].to_numpy(),
            "team_id": ordered["team_id"].to_numpy(),
            "side": ordered["side"].to_numpy(),
            "expected_position": labels[perm].ravel(),
            "previous_explicit_position": ordered["previous_explicit_position"].to_numpy(),
            "assigned_position_score": assigned_scores.ravel(),
            "player_score_margin": margins.ravel(),
            "roster_assignment_margin": np.repeat(roster_margin, N_POSITIONS),
            "prior_explicit_position_games": ordered[
                "prior_explicit_position_games"
            ].to_numpy(),
            "recent_position_stability": ordered["recent_position_stability"].to_numpy(),
            "method": method,
            "evidence_tier": _evidence_tiers(ordered, method=method),
            "observed_position": (
                ordered["position"].to_numpy()
                if "position" in ordered.columns
                else None
            ),
        }
    )
    return frame[list(EXPECTED_POSITION_COLUMNS)]


def _side_preference_status(values: list[str | None]) -> str:
    if len(values) != N_POSITIONS:
        raise ValueError("expected five preference values")
    present = [value for value in values if value is not None]
    missing = len(values) - len(present)
    unique = set(present)
    has_duplicate = len(present) != len(unique)
    if missing == 0 and unique == set(EXPLICIT_POSITION_LABELS):
        return "unique_1_to_5"
    if missing > 0 and has_duplicate:
        return "missing_and_duplicate"
    if missing > 0:
        return "missing"
    if has_duplicate:
        return "duplicate"
    return "other"


def audit_naive_roster_uniqueness(state: pd.DataFrame) -> pd.DataFrame:
    """Classify each side's five independent preferences. No joint solver."""
    rows: list[dict[str, object]] = []
    ordered = state.sort_values(["match_id", "side", "player_id"], kind="mergesort")
    for (match_id, side), group in ordered.groupby(["match_id", "side"], sort=False):
        record: dict[str, object] = {
            "match_id": match_id,
            "side": side,
            "game_version_id": group["game_version_id"].iloc[0],
            "start_time": group["start_time"].iloc[0],
        }
        for source in NAIVE_PREFERENCE_SOURCES:
            values = [_explicit_label(value) for value in group[source].tolist()]
            record[source] = _side_preference_status(values)
            record[f"{source}_missing_players"] = sum(v is None for v in values)
            counts = Counter(v for v in values if v is not None)
            record[f"{source}_duplicate_positions"] = tuple(
                sorted(pos for pos, n in counts.items() if n > 1)
            )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def evaluate_expected_position(assigned: pd.DataFrame) -> dict[str, object]:
    """Score assignments against observed STRATZ position.

    NULL / non-explicit current observed rows are excluded from
    player-level denominators. A side is eligible only when all five
    observed positions are explicit 1–5.
    """
    players = assigned[assigned["observed_position"].isin(EXPLICIT_POSITION_LABELS)].copy()
    players["correct"] = players["expected_position"] == players["observed_position"]
    confusion = (
        pd.crosstab(players["observed_position"], players["expected_position"])
        .reindex(index=list(EXPLICIT_POSITION_LABELS), columns=list(EXPLICIT_POSITION_LABELS))
        .fillna(0)
        .astype(int)
    )

    side_obs = assigned.groupby(["match_id", "side"], sort=False)
    eligible_sides = []
    side_correct_counts = []
    for key, group in side_obs:
        explicit = group["observed_position"].isin(EXPLICIT_POSITION_LABELS)
        if int(explicit.sum()) != N_POSITIONS:
            continue
        n_correct = int(
            (group["expected_position"] == group["observed_position"]).sum()
        )
        eligible_sides.append(key)
        side_correct_counts.append(n_correct)
    side_counts = Counter(side_correct_counts)

    switched = players[
        players["previous_explicit_position"].isin(EXPLICIT_POSITION_LABELS)
        & (players["observed_position"] != players["previous_explicit_position"])
    ]
    return {
        "n_player_rows": len(assigned),
        "n_eligible_players": len(players),
        "player_accuracy": float(players["correct"].mean()) if len(players) else float("nan"),
        "confusion": confusion,
        "n_eligible_sides": len(eligible_sides),
        "side_correct_counts": {k: side_counts.get(k, 0) for k in range(N_POSITIONS + 1)},
        "side_exact_accuracy": (
            side_counts.get(N_POSITIONS, 0) / len(eligible_sides)
            if eligible_sides
            else float("nan")
        ),
        "n_switch_eligible_players": len(switched),
        "switch_player_accuracy": (
            float(switched["correct"].mean()) if len(switched) else float("nan")
        ),
    }


@dataclass(frozen=True)
class ExpectedPosition:
    """Materialized expected-position assignment."""

    frame: pd.DataFrame
    method: str

    def to_frame(self) -> pd.DataFrame:
        ordered = self.frame[list(EXPECTED_POSITION_COLUMNS)]
        return ordered.sort_values(
            ["match_id", "player_id"], kind="mergesort"
        ).reset_index(drop=True)


def build_expected_position(
    store: FeatureDuckDBConnection,
    *,
    method: str = DEFAULT_EXPECTED_POSITION_METHOD,
    match_id: int | None = None,
) -> ExpectedPosition:
    """Build expected positions from registered analytical views."""
    state = build_player_position_state(store, match_id=match_id).to_frame()
    assigned = assign_expected_positions(state, method=method)
    return ExpectedPosition(frame=assigned, method=method)
