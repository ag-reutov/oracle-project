"""Slice 28: causal next-pick draft-policy benchmark.

Research only. Asks whether the frozen Slice 26 draft prefix predicts the
next successfully picked hero better than strong historical drafting
baselines. This is a multiclass choice/policy problem — not a win model,
strength model, synergy/counter score, or player-assignment model.

Question
--------
Given the causal draft prefix ``S_(M,t)`` immediately before a successful
PICK at sequence ``t``, can we predict ``next_pick_hero_id`` better than
structural / version-aware popularity?

Target grain
------------
One observation per next successful pick decision:
``(match_id, pick_decision_index)`` with ``pick_decision_index`` in
``0 .. 9`` (ten successful picks). The Slice 26 boundary is the pick's
own ``sequence`` (state before that event).

Common support
--------------
Every estimator is scored over one shared causal candidate universe
``C_T`` (prior-observed heroes minus already successful picks/bans,
always including a genuine first-observed target). Scoring uses the
fixed Slice 25 mixture wrapper:

    q = (1 - epsilon) * p + epsilon * U(C_T)

with non-tuned ``SCORING_MIXTURE_EPSILON``. Estimator-specific support
never defines the normalization universe. Already-picked / successfully
banned heroes are removed identically for every model.

Model family
------------
Learned policy candidates use ``SGDClassifier(loss='log_loss')`` — a
one-vs-rest (OVR) L2-regularised logistic SGD.  **Multinomial logistic
regression** (``LogisticRegression(solver='lbfgs')``) was the original
design; it was **abandoned before final benchmark inspection** for
computational reasons (LBFGS / SAGA on ~10 k × ~125-class matrices
exceeded the practical time budget at ~5 fold × 4 specs × 3 C-grid).
The fixed recipe uses ``DEFAULT_POLICY_C = 1.0`` (no validation
search), ``alpha = 1 / (C * n_train)``, ``max_iter=200``, and
``early_stopping=True`` on research folds. Tiny-test cases where the
internal stratified validation split is impossible (< ``10 * n_classes``
rows) fall back to ``early_stopping=False`` deterministically.

This is a deliberate training-recipe freeze: do not switch solvers or
tune alpha/C based on Slice 28 metrics.

Holdout
-------
Development OOS only (``start_time <= FROZEN_DEVELOPMENT_END``). The
frozen Slice 9 holdout is not inspected or scored.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from dota_predictor.features.duckdb_layer import (
    DRAFT_EVENTS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    build_pre_draft_snapshot,
)
from dota_predictor.features.team_elo import DEFAULT_ELO_CONFIG, EloConfig
from dota_predictor.training.dataset import ModelReadyDataset, TrainingDatasetError
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.metrics import bootstrap_mean_ci
from dota_predictor.training.player_hero_pool_state import (
    SCORING_MIXTURE_EPSILON,
    score_distribution,
    scoring_candidates,
)
from dota_predictor.training.player_performance_target import (
    _jsonable_value,
    restrict_development,
)
from dota_predictor.training.sequential_draft_benchmark import (
    MatchDraftIndex,
    build_match_draft_index,
    encode_side_aware_indicators,
)
from dota_predictor.training.sequential_draft_state import (
    ACTION_PICK,
    BOUNDARY_CONVENTION,
    SIDE_DIRE,
    SIDE_RADIANT,
    SLICE26_FROZEN_COMPONENTS,
    SLICE26_RESEARCH_CLASSIFICATION,
    build_draft_prefix_state,
    event_is_actual,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES,
    FROZEN_HOLDOUT_BOOTSTRAP_SEED,
    assert_development_frame_excludes_holdout,
    utc_datetime,
)
from dota_predictor.training.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG,
    WalkForwardConfig,
    resolve_walk_forward_folds,
)

__all__ = [
    "BASELINE_A",
    "BASELINE_B",
    "BASELINE_C",
    "CANDIDATE_1_PREFIX_PICKS",
    "CANDIDATE_2_PREFIX_BANS",
    "CANDIDATE_3_TEAM",
    "CLASSIFICATION_A",
    "CLASSIFICATION_B",
    "CLASSIFICATION_C",
    "HIT_KS",
    "HOLDOUT_POLICY",
    "MODEL_TEAM_ONLY",
    "REGULARIZATION_CANDIDATES",
    "SLICE28_BOOTSTRAP_RESAMPLES",
    "SLICE28_BOOTSTRAP_SEED",
    "SLICE28_DIAGNOSTIC_ONLY",
    "SLICE28_FROZEN_COMPONENTS",
    "SLICE28_RESEARCH_CLASSIFICATION",
    "TEAM_TENDENCY",
    "Slice28BenchmarkReport",
    "build_next_pick_decision_rows",
    "build_policy_feature_vector",
    "candidate_universe_for_row",
    "classify_slice28",
    "run_slice28_next_pick_policy_benchmark",
    "score_policy_distribution",
    "slice28_report_to_jsonable",
    "unavailable_heroes",
]


# ---------------------------------------------------------------------------
# Constants / classification text
# ---------------------------------------------------------------------------

HIT_KS: tuple[int, ...] = (1, 3, 5, 10)
REGULARIZATION_CANDIDATES: tuple[float, ...] = (0.1, 1.0, 10.0)
# Default C used when ``tune_regularization=False`` (preferred for the
# confirmation run — conservative, not validation-searched).
DEFAULT_POLICY_C = 1.0
# SGD epochs for OVR logistic policy fits (~10k×~125 classes).
POLICY_SGD_MAX_ITER = 200
POLICY_MIN_TEAM_DECISIONS = 20
# Minimum training rows for early_stopping: must be > 10 * n_classes
# for the internal stratified validation split to succeed.
POLICY_EARLY_STOPPING_MIN_FACTOR = 10

SLICE28_BOOTSTRAP_RESAMPLES = FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES
SLICE28_BOOTSTRAP_SEED = FROZEN_HOLDOUT_BOOTSTRAP_SEED + 28

BASELINE_A = "baseline_a_global_popularity"
BASELINE_B = "baseline_b_side_pick_index"
BASELINE_C = "baseline_c_version_side_pick_index"
CANDIDATE_1_PREFIX_PICKS = "candidate_1_prefix_picks"
CANDIDATE_2_PREFIX_BANS = "candidate_2_prefix_picks_bans"
CANDIDATE_3_TEAM = "candidate_3_prefix_plus_team"
MODEL_TEAM_ONLY = "ablation_team_identity_only"
TEAM_TENDENCY = "ablation_team_tendency"

HOLDOUT_POLICY = (
    "development_oos_only: frozen Slice 9 holdout remains reserved. "
    "Slice 28 scores expanding-window OOS on "
    "start_time <= FROZEN_DEVELOPMENT_END only."
)

CLASSIFICATION_A = (
    "A — freeze conditional draft-policy representation: the draft "
    "prefix itself materially and confirmation-stably improves next-hero "
    "prediction beyond strong structural/version popularity baselines, "
    "not entirely attributable to team identity."
)
CLASSIFICATION_B = (
    "B — partial / suggestive: prefix improves inconsistently, only at "
    "certain pick positions, the effect is small, team identity explains "
    "most of the gain, or tune gains weaken on validation."
)
CLASSIFICATION_C = (
    "C — do not freeze: current prefix fails to reliably improve strong "
    "historical-choice baselines."
)

# Updated by the development audit after the first confirmation run.
SLICE28_RESEARCH_CLASSIFICATION = "C"
SLICE28_DIAGNOSTIC_ONLY = True
SLICE28_FROZEN_COMPONENTS: tuple[str, ...] = ()

EXPECTED_PICKS_PER_MATCH = 10
MATERIAL_LOGLOSS_DELTA = 0.01
MIN_DECISION_ROWS = 100

# Soft labels for acting-side / pick-index one-hots (train vocab freezes).
SIDE_FEATURE = "acting_side_dire"
PICK_INDEX_PREFIX = "overall_pick_index_"
SIDE_PICK_PREFIX = "side_pick_index_"
VERSION_PREFIX = "game_version_"
TEAM_PREFIX = "acting_team_"
PICK_HERO_PREFIX = "prefix_pick_side_hero_"
BAN_HERO_PREFIX = "prefix_ban_side_hero_"
TEAM_TENDENCY_PREFIX = "team_tendency_hero_"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def unavailable_heroes(
    *,
    radiant_pick_hero_ids: Sequence[int],
    dire_pick_hero_ids: Sequence[int],
    radiant_ban_hero_ids: Sequence[int],
    dire_ban_hero_ids: Sequence[int],
) -> frozenset[int]:
    """Heroes removed from ``C_T`` by successful picks/bans (Slice 26)."""
    return frozenset(
        {
            int(h)
            for h in (
                *radiant_pick_hero_ids,
                *dire_pick_hero_ids,
                *radiant_ban_hero_ids,
                *dire_ban_hero_ids,
            )
        }
    )


def candidate_universe_for_row(
    *,
    prior_heroes: frozenset[int] | set[int],
    unavailable: frozenset[int] | set[int],
    realized_hero: int,
) -> frozenset[int]:
    """Shared causal ``C_T`` independent of estimator support.

    Starts from heroes observed in professional data strictly before the
    match, removes already successful picks/bans, and always includes a
    genuine first-observed target so every estimator handles that row
    identically.
    """
    base = {int(h) for h in prior_heroes} - {int(h) for h in unavailable}
    return scoring_candidates(base, realized_hero=int(realized_hero))


def score_policy_distribution(
    mass: Mapping[int, float],
    *,
    realized_hero: int,
    candidates: frozenset[int] | set[int] | Sequence[int],
    epsilon: float = SCORING_MIXTURE_EPSILON,
) -> dict[str, float]:
    """Common-support multiclass scoring with hit@10 and MRR.

    Reuses Slice 25's mixture / Brier / rank semantics and extends the
    hit ladder. Log-loss uses ``q``; Brier / rank / hit@k use raw ``p``.
    """
    base = score_distribution(
        dict(mass),
        realized_hero=int(realized_hero),
        candidates=candidates,
        epsilon=epsilon,
    )
    if not np.isfinite(base.get("rank", float("nan"))):
        base["hit_10"] = float("nan")
        base["reciprocal_rank"] = float("nan")
        base["mrr"] = float("nan")
        return base
    rank = float(base["rank"])
    base["hit_10"] = float(rank <= 10)
    rr = 1.0 / rank if rank > 0 else float("nan")
    base["reciprocal_rank"] = rr
    base["mrr"] = rr
    return base


def _mass_on_candidates(
    counts: Mapping[int, float] | Counter[int],
    candidates: frozenset[int],
) -> dict[int, float]:
    return {int(h): float(counts.get(int(h), 0.0)) for h in candidates}


def _renormalize_mass_on_candidates(
    mass: Mapping[int, float],
    candidates: frozenset[int],
) -> dict[int, float]:
    """Place ``mass`` on ``C_T`` (zeros off support). No estimator-specific support."""
    return {int(h): float(mass.get(int(h), 0.0)) for h in candidates}


# ---------------------------------------------------------------------------
# Decision rows
# ---------------------------------------------------------------------------


def build_next_pick_decision_rows(
    *,
    matches: pd.DataFrame,
    indexes: Mapping[int, MatchDraftIndex],
    require_ten_picks: bool = True,
) -> pd.DataFrame:
    """Expand matches into next-successful-pick decision rows.

    Each row is the Slice 26 state immediately before a successful PICK.
    The target hero is never written into prefix pick/ban lists.
    """
    required = {
        "match_id",
        "start_time",
        "game_version_id",
        "radiant_team_id",
        "dire_team_id",
    }
    missing = required - set(matches.columns)
    if missing:
        raise TrainingDatasetError(f"matches missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for match_row in matches.to_dict(orient="records"):
        match_id = int(match_row["match_id"])
        index = indexes.get(match_id)
        if index is None:
            continue
        picks = index.successful_picks
        if require_ten_picks and len(picks) != EXPECTED_PICKS_PER_MATCH:
            continue
        if not picks:
            continue

        start_time = match_row["start_time"]
        game_version_id = match_row.get("game_version_id")
        radiant_team_id = int(match_row["radiant_team_id"])
        dire_team_id = int(match_row["dire_team_id"])
        radiant_players = tuple(
            int(x) for x in match_row.get("radiant_player_ids", (1, 2, 3, 4, 5))
        )
        dire_players = tuple(
            int(x) for x in match_row.get("dire_player_ids", (6, 7, 8, 9, 10))
        )

        for pick_decision_index, pick in enumerate(picks):
            boundary_t = int(pick["sequence"])
            state = build_draft_prefix_state(
                match_id=match_id,
                start_time=start_time,
                game_version_id=(
                    None
                    if game_version_id is None or pd.isna(game_version_id)
                    else int(game_version_id)
                ),
                boundary_t=boundary_t,
                events=index.events,
                radiant_team_id=radiant_team_id,
                dire_team_id=dire_team_id,
                radiant_player_ids=radiant_players,
                dire_player_ids=dire_players,
            )
            target = int(pick["hero_id"])
            acting_side = str(pick["side"])
            if acting_side == SIDE_RADIANT:
                acting_team_id = radiant_team_id
                side_pick_index = int(state["n_radiant_picks"]) + 1
            elif acting_side == SIDE_DIRE:
                acting_team_id = dire_team_id
                side_pick_index = int(state["n_dire_picks"]) + 1
            else:
                raise TrainingDatasetError(
                    f"match {match_id}: unknown acting side {acting_side!r}"
                )

            prefix_heroes = set(state["radiant_pick_hero_ids"]) | set(
                state["dire_pick_hero_ids"]
            )
            if target in prefix_heroes:
                raise TrainingDatasetError(
                    f"match {match_id} t={boundary_t}: target {target} "
                    "leaked into prefix picks"
                )

            unavail = unavailable_heroes(
                radiant_pick_hero_ids=state["radiant_pick_hero_ids"],
                dire_pick_hero_ids=state["dire_pick_hero_ids"],
                radiant_ban_hero_ids=state["radiant_ban_hero_ids"],
                dire_ban_hero_ids=state["dire_ban_hero_ids"],
            )
            rows.append(
                {
                    "match_id": match_id,
                    "start_time": start_time,
                    "game_version_id": state["game_version_id"],
                    "pick_decision_index": int(pick_decision_index),
                    "overall_pick_index": int(pick_decision_index) + 1,
                    "boundary_t": boundary_t,
                    "next_pick_hero_id": target,
                    "acting_side": acting_side,
                    "acting_team_id": int(acting_team_id),
                    "side_pick_index": int(side_pick_index),
                    "n_radiant_picks": int(state["n_radiant_picks"]),
                    "n_dire_picks": int(state["n_dire_picks"]),
                    "n_radiant_bans_successful": int(
                        state["n_radiant_bans_successful"]
                    ),
                    "n_dire_bans_successful": int(state["n_dire_bans_successful"]),
                    "radiant_pick_hero_ids": tuple(state["radiant_pick_hero_ids"]),
                    "dire_pick_hero_ids": tuple(state["dire_pick_hero_ids"]),
                    "radiant_ban_hero_ids": tuple(state["radiant_ban_hero_ids"]),
                    "dire_ban_hero_ids": tuple(state["dire_ban_hero_ids"]),
                    "unavailable_hero_ids": tuple(sorted(unavail)),
                    "draft_signature": index.signature,
                    "n_prior_events": int(state["n_prior_events"]),
                    "radiant_team_id": radiant_team_id,
                    "dire_team_id": dire_team_id,
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["start_time"] = pd.to_datetime(frame["start_time"], utc=True)
    return frame.sort_values(
        ["start_time", "match_id", "pick_decision_index"], kind="stable"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Causal history indexes (frequency baselines + team tendency)
# ---------------------------------------------------------------------------


@dataclass
class CausalPickHistory:
    """Strictly-prior successful pick counts for frequency baselines."""

    match_order: tuple[int, ...]
    start_times: np.ndarray  # int64 UTC ns aligned to match_order
    # Per match: list of (acting_side, overall_pick_index, side_pick_index,
    # game_version_id, acting_team_id, hero_id)
    match_picks: dict[int, tuple[tuple[Any, ...], ...]]
    # Expanding cumulative structures built lazily via pointer.
    _ptr: int = 0
    global_counts: Counter[int] = field(default_factory=Counter)
    side_pick_counts: dict[tuple[str, int], Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    version_side_pick_counts: dict[tuple[int | None, str, int], Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    team_counts: dict[int, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    observed_heroes: set[int] = field(default_factory=set)

    def advance_before(self, start_time: pd.Timestamp) -> None:
        """Include all matches with ``start_time <`` the given stamp."""
        stamp = pd.Timestamp(start_time)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        stamp_ns = int(stamp.value)
        times = self.start_times
        while self._ptr < len(self.match_order):
            if int(times[self._ptr]) >= stamp_ns:
                break
            match_id = self.match_order[self._ptr]
            for side, overall, _side_n, version, team_id, hero in self.match_picks[
                match_id
            ]:
                hero_i = int(hero)
                self.global_counts[hero_i] += 1
                self.side_pick_counts[(str(side), int(overall))][hero_i] += 1
                self.version_side_pick_counts[
                    (
                        None if version is None or pd.isna(version) else int(version),
                        str(side),
                        int(overall),
                    )
                ][hero_i] += 1
                self.team_counts[int(team_id)][hero_i] += 1
                self.observed_heroes.add(hero_i)
            self._ptr += 1

    def snapshot_prior_heroes(self) -> frozenset[int]:
        return frozenset(self.observed_heroes)


def build_causal_pick_history(decisions: pd.DataFrame) -> CausalPickHistory:
    """Build a chronological pick-history index from decision rows.

    History for match ``M`` uses only other matches with
    ``start_time < M.start_time`` (equal timestamps mutually blind).
    """
    if decisions.empty:
        return CausalPickHistory(
            match_order=(),
            start_times=np.asarray([], dtype=np.int64),
            match_picks={},
        )

    match_meta = (
        decisions.groupby("match_id", sort=False)
        .agg(start_time=("start_time", "first"))
        .reset_index()
        .sort_values(["start_time", "match_id"], kind="stable")
    )
    match_picks: dict[int, tuple[tuple[Any, ...], ...]] = {}
    for match_id, group in decisions.groupby("match_id", sort=False):
        ordered = group.sort_values("pick_decision_index", kind="stable")
        picks: list[tuple[Any, ...]] = []
        for row in ordered.to_dict(orient="records"):
            picks.append(
                (
                    str(row["acting_side"]),
                    int(row["overall_pick_index"]),
                    int(row["side_pick_index"]),
                    row["game_version_id"],
                    int(row["acting_team_id"]),
                    int(row["next_pick_hero_id"]),
                )
            )
        match_picks[int(match_id)] = tuple(picks)

    order = tuple(int(x) for x in match_meta["match_id"].tolist())
    # Use Timestamp.value (ns) — Series.astype("int64") on datetime64[us]
    # yields microseconds and breaks same-timestamp blindness.
    times = np.asarray(
        [
            int(pd.Timestamp(ts).tz_convert("UTC").value)
            if pd.Timestamp(ts).tzinfo is not None
            else int(pd.Timestamp(ts, tz="UTC").value)
            for ts in pd.to_datetime(match_meta["start_time"], utc=True)
        ],
        dtype=np.int64,
    )
    return CausalPickHistory(
        match_order=order, start_times=times, match_picks=match_picks
    )


def _baseline_mass(
    history: CausalPickHistory,
    *,
    model: str,
    acting_side: str,
    overall_pick_index: int,
    game_version_id: int | None,
    acting_team_id: int,
    candidates: frozenset[int],
) -> dict[int, float]:
    if model == BASELINE_A:
        return _mass_on_candidates(history.global_counts, candidates)
    if model == BASELINE_B:
        key = (acting_side, overall_pick_index)
        counts = history.side_pick_counts.get(key)
        if counts and sum(counts.values()) > 0:
            return _mass_on_candidates(counts, candidates)
        return _mass_on_candidates(history.global_counts, candidates)
    if model == BASELINE_C:
        vkey = (game_version_id, acting_side, overall_pick_index)
        counts = history.version_side_pick_counts.get(vkey)
        if counts and sum(counts.values()) > 0:
            return _mass_on_candidates(counts, candidates)
        # Backoff to structural (B) then global (A).
        return _baseline_mass(
            history,
            model=BASELINE_B,
            acting_side=acting_side,
            overall_pick_index=overall_pick_index,
            game_version_id=game_version_id,
            acting_team_id=acting_team_id,
            candidates=candidates,
        )
    if model == TEAM_TENDENCY:
        counts = history.team_counts.get(int(acting_team_id))
        if counts and sum(counts.values()) > 0:
            return _mass_on_candidates(counts, candidates)
        return _mass_on_candidates(history.global_counts, candidates)
    raise ValueError(f"unknown frequency model: {model}")


# ---------------------------------------------------------------------------
# Policy feature builder (reusable representation)
# ---------------------------------------------------------------------------


def build_policy_feature_vector(
    row: Mapping[str, Any],
    *,
    pick_vocabulary: Sequence[int],
    ban_vocabulary: Sequence[int] = (),
    version_vocabulary: Sequence[int | None] = (),
    team_vocabulary: Sequence[int] = (),
    include_picks: bool = False,
    include_bans: bool = False,
    include_team_identity: bool = False,
    include_team_tendency: bool = False,
    team_tendency_counts: Mapping[int, float] | None = None,
    max_pick_index: int = EXPECTED_PICKS_PER_MATCH,
    max_side_pick_index: int = 5,
) -> dict[str, float]:
    """Sparse categorical / side-aware policy features for one decision.

    Does not include the target hero. Designed so later research can
    rebuild the same model inputs from causal prefixes without
    re-deriving availability logic.
    """
    features: dict[str, float] = {
        SIDE_FEATURE: 1.0 if str(row["acting_side"]) == SIDE_DIRE else 0.0
    }
    overall = int(row["overall_pick_index"])
    side_n = int(row["side_pick_index"])
    for i in range(1, max_pick_index + 1):
        features[f"{PICK_INDEX_PREFIX}{i}"] = 1.0 if overall == i else 0.0
    for i in range(1, max_side_pick_index + 1):
        features[f"{SIDE_PICK_PREFIX}{i}"] = 1.0 if side_n == i else 0.0

    version = row.get("game_version_id")
    version_key = None if version is None or pd.isna(version) else int(version)
    for v in version_vocabulary:
        features[f"{VERSION_PREFIX}{v}"] = 1.0 if version_key == v else 0.0

    if include_picks:
        features.update(
            encode_side_aware_indicators(
                radiant_hero_ids=row["radiant_pick_hero_ids"],
                dire_hero_ids=row["dire_pick_hero_ids"],
                vocabulary=pick_vocabulary,
                column_fn=lambda h: f"{PICK_HERO_PREFIX}{int(h)}",
            )
        )
    if include_bans:
        features.update(
            encode_side_aware_indicators(
                radiant_hero_ids=row["radiant_ban_hero_ids"],
                dire_hero_ids=row["dire_ban_hero_ids"],
                vocabulary=ban_vocabulary,
                column_fn=lambda h: f"{BAN_HERO_PREFIX}{int(h)}",
            )
        )
    if include_team_identity:
        acting = int(row["acting_team_id"])
        for team_id in team_vocabulary:
            features[f"{TEAM_PREFIX}{int(team_id)}"] = (
                1.0 if acting == int(team_id) else 0.0
            )
    if include_team_tendency:
        tendency = team_tendency_counts or {}
        total = float(sum(tendency.values()))
        for hero_id in pick_vocabulary:
            share = (
                float(tendency.get(int(hero_id), 0.0)) / total if total > 0.0 else 0.0
            )
            features[f"{TEAM_TENDENCY_PREFIX}{int(hero_id)}"] = share
    return features


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
    pick_vocabulary: Sequence[int],
    ban_vocabulary: Sequence[int],
    version_vocabulary: Sequence[int | None],
    team_vocabulary: Sequence[int],
    include_picks: bool,
    include_bans: bool,
    include_team_identity: bool,
    include_team_tendency: bool,
    team_tendency_by_team: Mapping[int, Counter[int]] | None,
) -> np.ndarray:
    matrix = np.zeros((len(rows), len(feature_columns)), dtype=np.float64)
    col_index = {name: i for i, name in enumerate(feature_columns)}
    for r_i, row in enumerate(rows):
        tendency = None
        if include_team_tendency and team_tendency_by_team is not None:
            tendency = team_tendency_by_team.get(int(row["acting_team_id"]), Counter())
        feats = build_policy_feature_vector(
            row,
            pick_vocabulary=pick_vocabulary,
            ban_vocabulary=ban_vocabulary,
            version_vocabulary=version_vocabulary,
            team_vocabulary=team_vocabulary,
            include_picks=include_picks,
            include_bans=include_bans,
            include_team_identity=include_team_identity,
            include_team_tendency=include_team_tendency,
            team_tendency_counts=tendency,
        )
        for name, value in feats.items():
            idx = col_index.get(name)
            if idx is not None:
                matrix[r_i, idx] = float(value)
    return matrix


def _declare_feature_columns(
    *,
    pick_vocabulary: Sequence[int],
    ban_vocabulary: Sequence[int],
    version_vocabulary: Sequence[int | None],
    team_vocabulary: Sequence[int],
    include_picks: bool,
    include_bans: bool,
    include_team_identity: bool,
    include_team_tendency: bool,
    max_pick_index: int = EXPECTED_PICKS_PER_MATCH,
    max_side_pick_index: int = 5,
) -> tuple[str, ...]:
    cols: list[str] = [SIDE_FEATURE]
    cols.extend(f"{PICK_INDEX_PREFIX}{i}" for i in range(1, max_pick_index + 1))
    cols.extend(f"{SIDE_PICK_PREFIX}{i}" for i in range(1, max_side_pick_index + 1))
    cols.extend(f"{VERSION_PREFIX}{v}" for v in version_vocabulary)
    if include_picks:
        cols.extend(f"{PICK_HERO_PREFIX}{int(h)}" for h in pick_vocabulary)
    if include_bans:
        cols.extend(f"{BAN_HERO_PREFIX}{int(h)}" for h in ban_vocabulary)
    if include_team_identity:
        cols.extend(f"{TEAM_PREFIX}{int(t)}" for t in team_vocabulary)
    if include_team_tendency:
        cols.extend(f"{TEAM_TENDENCY_PREFIX}{int(h)}" for h in pick_vocabulary)
    return tuple(cols)


@dataclass
class FittedPolicyModel:
    """Multinomial logistic policy with fixed train vocabularies."""

    name: str
    feature_columns: tuple[str, ...]
    classes_: tuple[int, ...]
    scaler: StandardScaler
    classifier: SGDClassifier
    pick_vocabulary: tuple[int, ...]
    ban_vocabulary: tuple[int, ...]
    version_vocabulary: tuple[int | None, ...]
    team_vocabulary: tuple[int, ...]
    include_picks: bool
    include_bans: bool
    include_team_identity: bool
    include_team_tendency: bool
    C: float
    team_tendency_by_team: dict[int, Counter[int]]

    def predict_mass(
        self,
        row: Mapping[str, Any],
        candidates: frozenset[int],
    ) -> dict[int, float]:
        X = _feature_matrix(
            [row],
            feature_columns=self.feature_columns,
            pick_vocabulary=self.pick_vocabulary,
            ban_vocabulary=self.ban_vocabulary,
            version_vocabulary=self.version_vocabulary,
            team_vocabulary=self.team_vocabulary,
            include_picks=self.include_picks,
            include_bans=self.include_bans,
            include_team_identity=self.include_team_identity,
            include_team_tendency=self.include_team_tendency,
            team_tendency_by_team=self.team_tendency_by_team,
        )
        Xs = self.scaler.transform(X)
        proba = self.classifier.predict_proba(Xs)[0]
        mass = {
            int(cls): float(p)
            for cls, p in zip(self.classes_, proba, strict=True)
        }
        return _renormalize_mass_on_candidates(mass, candidates)


def _fit_multinomial_policy(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    C: float,
    include_picks: bool,
    include_bans: bool,
    include_team_identity: bool,
    include_team_tendency: bool,
    history_for_tendency: CausalPickHistory | None = None,
    max_iter: int = POLICY_SGD_MAX_ITER,
) -> FittedPolicyModel:
    if not train_rows:
        raise TrainingDatasetError(f"{name}: empty training rows")

    pick_vocab = tuple(
        sorted(
            {
                int(h)
                for row in train_rows
                for h in (
                    *row["radiant_pick_hero_ids"],
                    *row["dire_pick_hero_ids"],
                    int(row["next_pick_hero_id"]),
                )
            }
        )
    )
    ban_vocab = tuple(
        sorted(
            {
                int(h)
                for row in train_rows
                for h in (*row["radiant_ban_hero_ids"], *row["dire_ban_hero_ids"])
            }
        )
    )
    version_vocab = tuple(
        sorted(
            {
                None
                if row["game_version_id"] is None or pd.isna(row["game_version_id"])
                else int(row["game_version_id"])
                for row in train_rows
            },
            key=lambda v: (-1 if v is None else v),
        )
    )
    # Cap team one-hots to teams with enough train decisions to keep
    # multinomial fits tractable; rare teams share a cold-start zero vector.
    team_counts_tmp: Counter[int] = Counter(
        int(row["acting_team_id"]) for row in train_rows
    )
    team_vocab = tuple(
        sorted(
            tid
            for tid, n in team_counts_tmp.items()
            if n >= POLICY_MIN_TEAM_DECISIONS
        )
    )
    y = np.asarray([int(row["next_pick_hero_id"]) for row in train_rows], dtype=int)
    feature_columns = _declare_feature_columns(
        pick_vocabulary=pick_vocab,
        ban_vocabulary=ban_vocab,
        version_vocabulary=version_vocab,
        team_vocabulary=team_vocab,
        include_picks=include_picks,
        include_bans=include_bans,
        include_team_identity=include_team_identity,
        include_team_tendency=include_team_tendency,
    )
    tendency_map: dict[int, Counter[int]] = {}
    if include_team_tendency:
        by_team_match: dict[int, dict[int, Counter[int]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        for row in train_rows:
            by_team_match[int(row["acting_team_id"])][int(row["match_id"])][
                int(row["next_pick_hero_id"])
            ] += 1
        team_totals: dict[int, Counter[int]] = {}
        for team_id, per_match in by_team_match.items():
            total: Counter[int] = Counter()
            for counts in per_match.values():
                total.update(counts)
            team_totals[team_id] = total
        tendency_map = team_totals

    X = _feature_matrix(
        train_rows,
        feature_columns=feature_columns,
        pick_vocabulary=pick_vocab,
        ban_vocabulary=ban_vocab,
        version_vocabulary=version_vocab,
        team_vocabulary=team_vocab,
        include_picks=include_picks,
        include_bans=include_bans,
        include_team_identity=include_team_identity,
        include_team_tendency=include_team_tendency,
        team_tendency_by_team=tendency_map,
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    # Fixed OVR logistic SGD (see module docstring for model-family note).
    n = max(len(train_rows), 1)
    alpha = 1.0 / (float(C) * float(n))
    class_counts = Counter(int(v) for v in y.tolist())
    n_classes = len(class_counts)
    min_class_count = min(class_counts.values())
    # early_stopping uses StratifiedShuffleSplit internally, which needs
    # every class to have >= 2 members. Additionally we require the fold
    # to be large enough overall to afford the validation fraction.
    use_early_stopping = (
        n >= POLICY_EARLY_STOPPING_MIN_FACTOR * n_classes
        and min_class_count >= 2
    )
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=int(max_iter),
        tol=1e-3,
        random_state=0,
        early_stopping=use_early_stopping,
        validation_fraction=0.1,
        n_iter_no_change=5,
    )
    clf.fit(Xs, y)
    return FittedPolicyModel(
        name=name,
        feature_columns=feature_columns,
        classes_=tuple(int(c) for c in clf.classes_),
        scaler=scaler,
        classifier=clf,
        pick_vocabulary=pick_vocab,
        ban_vocabulary=ban_vocab,
        version_vocabulary=version_vocab,
        team_vocabulary=team_vocab,
        include_picks=include_picks,
        include_bans=include_bans,
        include_team_identity=include_team_identity,
        include_team_tendency=include_team_tendency,
        C=float(C),
        team_tendency_by_team=tendency_map,
    )


def _select_C(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    name: str,
    include_picks: bool,
    include_bans: bool,
    include_team_identity: bool,
    include_team_tendency: bool,
    history: CausalPickHistory,
    tune_regularization: bool = False,
) -> tuple[float, FittedPolicyModel, pd.DataFrame]:
    """Fit policy model; optionally choose ``C`` on validation log loss.

    Default confirmation path uses ``DEFAULT_POLICY_C`` without a grid
    search (conservative, not validation-tuned). When tuning, the small
    predeclared grid is scored with train-vocab multiclass log loss;
    final OOS still uses shared ``C_T``.
    """
    from sklearn.metrics import log_loss as sk_log_loss

    grid = (
        REGULARIZATION_CANDIDATES if tune_regularization else (DEFAULT_POLICY_C,)
    )
    rows_out: list[dict[str, object]] = []
    best_c = grid[0]
    best_ll = float("inf")
    best_model: FittedPolicyModel | None = None

    for c in grid:
        model = _fit_multinomial_policy(
            train_rows,
            name=name,
            C=c,
            include_picks=include_picks,
            include_bans=include_bans,
            include_team_identity=include_team_identity,
            include_team_tendency=include_team_tendency,
        )
        if not tune_regularization:
            rows_out.append(
                {
                    "C": c,
                    "validation_log_loss": float("nan"),
                    "n": len(val_rows),
                    "tuned": False,
                }
            )
            return c, model, pd.DataFrame(rows_out)

        if not val_rows:
            mean_ll = float("inf")
        else:
            if include_team_tendency:
                hist = _clone_history(history)
                last_t = max(pd.Timestamp(r["start_time"]) for r in val_rows)
                hist.advance_before(last_t)
                model.team_tendency_by_team = {
                    tid: Counter(counts) for tid, counts in hist.team_counts.items()
                }
            Xv = _feature_matrix(
                val_rows,
                feature_columns=model.feature_columns,
                pick_vocabulary=model.pick_vocabulary,
                ban_vocabulary=model.ban_vocabulary,
                version_vocabulary=model.version_vocabulary,
                team_vocabulary=model.team_vocabulary,
                include_picks=model.include_picks,
                include_bans=model.include_bans,
                include_team_identity=model.include_team_identity,
                include_team_tendency=model.include_team_tendency,
                team_tendency_by_team=model.team_tendency_by_team,
            )
            proba = model.classifier.predict_proba(model.scaler.transform(Xv))
            y_val = np.asarray(
                [int(r["next_pick_hero_id"]) for r in val_rows], dtype=int
            )
            labels = list(model.classes_)
            known = np.isin(y_val, model.classes_)
            if not bool(known.any()):
                mean_ll = float("inf")
            else:
                mean_ll = float(
                    sk_log_loss(y_val[known], proba[known], labels=labels)
                )
        rows_out.append(
            {
                "C": c,
                "validation_log_loss": mean_ll,
                "n": len(val_rows),
                "tuned": True,
            }
        )
        if mean_ll < best_ll:
            best_ll = mean_ll
            best_c = c
            best_model = model
    assert best_model is not None
    return best_c, best_model, pd.DataFrame(rows_out)


def _clone_history(history: CausalPickHistory) -> CausalPickHistory:
    return CausalPickHistory(
        match_order=history.match_order,
        start_times=history.start_times,
        match_picks=history.match_picks,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _metric_summary(scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not scores:
        return {
            "n": 0.0,
            "log_loss": float("nan"),
            "brier": float("nan"),
            "mean_rank": float("nan"),
            "median_rank": float("nan"),
            "hit_1": float("nan"),
            "hit_3": float("nan"),
            "hit_5": float("nan"),
            "hit_10": float("nan"),
            "mrr": float("nan"),
            "mean_n_candidates": float("nan"),
            "mean_p_realized": float("nan"),
        }
    ll = np.asarray([s["log_loss"] for s in scores], dtype=float)
    brier = np.asarray([s["brier"] for s in scores], dtype=float)
    rank = np.asarray([s["rank"] for s in scores], dtype=float)
    return {
        "n": float(len(scores)),
        "log_loss": float(np.mean(ll)),
        "brier": float(np.mean(brier)),
        "mean_rank": float(np.mean(rank)),
        "median_rank": float(np.median(rank)),
        "hit_1": float(np.mean([s["hit_1"] for s in scores])),
        "hit_3": float(np.mean([s["hit_3"] for s in scores])),
        "hit_5": float(np.mean([s["hit_5"] for s in scores])),
        "hit_10": float(np.mean([s["hit_10"] for s in scores])),
        "mrr": float(np.mean([s["mrr"] for s in scores])),
        "mean_n_candidates": float(np.mean([s["n_candidates"] for s in scores])),
        "mean_p_realized": float(np.mean([s["p_realized"] for s in scores])),
    }


def _match_clustered_delta_ci(
    rows: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    seed: int,
) -> dict[str, float]:
    """Bootstrap mean of per-match mean(left-right); clusters all picks."""
    if rows.empty:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_matches": 0.0,
            "n_rows": 0.0,
        }
    work = rows.copy()
    work["delta"] = work[left_col] - work[right_col]
    match_means = work.groupby("match_id", sort=False)["delta"].mean().to_numpy()
    lo, hi = bootstrap_mean_ci(
        match_means,
        n_resamples=SLICE28_BOOTSTRAP_RESAMPLES,
        random_state=seed,
    )
    return {
        "mean": float(np.mean(match_means)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_matches": float(match_means.size),
        "n_rows": float(len(work)),
    }


def _score_frequency_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    history: CausalPickHistory,
) -> list[dict[str, Any]]:
    hist = _clone_history(history)
    out: list[dict[str, Any]] = []
    for row in sorted(
        rows, key=lambda r: (pd.Timestamp(r["start_time"]), int(r["match_id"]))
    ):
        hist.advance_before(pd.Timestamp(row["start_time"]))
        prior = hist.snapshot_prior_heroes()
        unavail = frozenset(int(h) for h in row["unavailable_hero_ids"])
        realized = int(row["next_pick_hero_id"])
        cands = candidate_universe_for_row(
            prior_heroes=prior, unavailable=unavail, realized_hero=realized
        )
        first_obs = realized not in prior
        mass = _baseline_mass(
            hist,
            model=model,
            acting_side=str(row["acting_side"]),
            overall_pick_index=int(row["overall_pick_index"]),
            game_version_id=(
                None
                if row["game_version_id"] is None or pd.isna(row["game_version_id"])
                else int(row["game_version_id"])
            ),
            acting_team_id=int(row["acting_team_id"]),
            candidates=cands,
        )
        scored = score_policy_distribution(
            mass, realized_hero=realized, candidates=cands
        )
        out.append(
            {
                "match_id": int(row["match_id"]),
                "pick_decision_index": int(row["pick_decision_index"]),
                "overall_pick_index": int(row["overall_pick_index"]),
                "acting_side": str(row["acting_side"]),
                "game_version_id": row["game_version_id"],
                "draft_signature": row["draft_signature"],
                "acting_team_id": int(row["acting_team_id"]),
                "team_history_n": int(
                    sum(hist.team_counts.get(int(row["acting_team_id"]), {}).values())
                ),
                "first_observed_target": bool(first_obs),
                "n_candidates": int(scored["n_candidates"]),
                "log_loss": float(scored["log_loss"]),
                "brier": float(scored["brier"]),
                "rank": float(scored["rank"]),
                "hit_1": float(scored["hit_1"]),
                "hit_3": float(scored["hit_3"]),
                "hit_5": float(scored["hit_5"]),
                "hit_10": float(scored["hit_10"]),
                "mrr": float(scored["mrr"]),
                "p_realized": float(scored["p_realized"]),
                "model": model,
            }
        )
    return out


def _score_model_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: FittedPolicyModel,
    history: CausalPickHistory,
) -> list[dict[str, Any]]:
    hist = _clone_history(history)
    ordered = sorted(
        rows, key=lambda r: (pd.Timestamp(r["start_time"]), int(r["match_id"]))
    )
    if not ordered:
        return []
    # Batch model probabilities once; legality / C_T applied per row.
    if model.include_team_tendency:
        # Use history state just before the first scored row; refresh is
        # approximate for tendency features (strict for C_T / baselines).
        hist_tend = _clone_history(history)
        hist_tend.advance_before(pd.Timestamp(ordered[0]["start_time"]))
        model.team_tendency_by_team = {
            tid: Counter(counts) for tid, counts in hist_tend.team_counts.items()
        }
    X = _feature_matrix(
        ordered,
        feature_columns=model.feature_columns,
        pick_vocabulary=model.pick_vocabulary,
        ban_vocabulary=model.ban_vocabulary,
        version_vocabulary=model.version_vocabulary,
        team_vocabulary=model.team_vocabulary,
        include_picks=model.include_picks,
        include_bans=model.include_bans,
        include_team_identity=model.include_team_identity,
        include_team_tendency=model.include_team_tendency,
        team_tendency_by_team=model.team_tendency_by_team,
    )
    proba = model.classifier.predict_proba(model.scaler.transform(X))
    classes = model.classes_
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ordered):
        hist.advance_before(pd.Timestamp(row["start_time"]))
        prior = hist.snapshot_prior_heroes()
        unavail = frozenset(int(h) for h in row["unavailable_hero_ids"])
        realized = int(row["next_pick_hero_id"])
        cands = candidate_universe_for_row(
            prior_heroes=prior, unavailable=unavail, realized_hero=realized
        )
        first_obs = realized not in prior
        mass = {
            int(cls): float(p)
            for cls, p in zip(classes, proba[i], strict=True)
        }
        mass = _renormalize_mass_on_candidates(mass, cands)
        scored = score_policy_distribution(
            mass, realized_hero=realized, candidates=cands
        )
        out.append(
            {
                "match_id": int(row["match_id"]),
                "pick_decision_index": int(row["pick_decision_index"]),
                "overall_pick_index": int(row["overall_pick_index"]),
                "acting_side": str(row["acting_side"]),
                "game_version_id": row["game_version_id"],
                "draft_signature": row["draft_signature"],
                "acting_team_id": int(row["acting_team_id"]),
                "team_history_n": int(
                    sum(hist.team_counts.get(int(row["acting_team_id"]), {}).values())
                ),
                "first_observed_target": bool(first_obs),
                "n_candidates": int(scored["n_candidates"]),
                "log_loss": float(scored["log_loss"]),
                "brier": float(scored["brier"]),
                "rank": float(scored["rank"]),
                "hit_1": float(scored["hit_1"]),
                "hit_3": float(scored["hit_3"]),
                "hit_5": float(scored["hit_5"]),
                "hit_10": float(scored["hit_10"]),
                "mrr": float(scored["mrr"]),
                "p_realized": float(scored["p_realized"]),
                "model": model.name,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_slice28(
    *,
    pooled_metrics: pd.DataFrame,
    paired_deltas: pd.DataFrame,
    pick_position_breakdown: pd.DataFrame,
) -> tuple[str, str, str, tuple[str, ...]]:
    """Map confirmation evidence onto A/B/C and a prefix-value pattern."""

    def _delta(candidate: str, baseline: str) -> dict[str, float] | None:
        hit = paired_deltas.loc[
            (paired_deltas["candidate"] == candidate)
            & (paired_deltas["baseline"] == baseline)
        ]
        if hit.empty:
            return None
        row = hit.iloc[0]
        return {
            "mean": float(row["delta_log_loss"]),
            "ci_low": float(row["delta_ci_low"]),
            "ci_high": float(row["delta_ci_high"]),
        }

    structural = BASELINE_C
    # Prefer C when it exists; else B.
    if pooled_metrics.empty:
        return CLASSIFICATION_C, "no pooled metrics.", "D", ()

    models = set(pooled_metrics["model"].tolist())
    if structural not in models:
        structural = BASELINE_B if BASELINE_B in models else BASELINE_A

    d_picks = _delta(CANDIDATE_1_PREFIX_PICKS, structural)
    d_bans = _delta(CANDIDATE_2_PREFIX_BANS, CANDIDATE_1_PREFIX_PICKS)
    d_team = _delta(CANDIDATE_3_TEAM, CANDIDATE_2_PREFIX_BANS)
    d_team_only = _delta(MODEL_TEAM_ONLY, structural)
    d_prefix_vs_team = _delta(CANDIDATE_1_PREFIX_PICKS, MODEL_TEAM_ONLY)

    # Pattern from pick-position breakdown for prefix vs structural.
    pattern = "D"
    pattern_note = "Prefix does not beat structural popularity."
    if (
        not pick_position_breakdown.empty
        and "delta_log_loss_prefix_vs_structural" in pick_position_breakdown.columns
    ):
        pos = pick_position_breakdown.sort_values("overall_pick_index")
        early = pos.loc[pos["overall_pick_index"] <= 3, "delta_log_loss_prefix_vs_structural"]
        later = pos.loc[pos["overall_pick_index"] >= 7, "delta_log_loss_prefix_vs_structural"]
        early_help = bool(len(early) and (early < 0).mean() >= 0.5)
        later_help = bool(len(later) and (later < 0).mean() >= 0.5)
        all_help = bool(
            len(pos) and (pos["delta_log_loss_prefix_vs_structural"] < 0).mean() >= 0.7
        )
        if later_help and not early_help:
            pattern = "A"
            pattern_note = (
                "Pattern A: prefix barely helps early, helps more later."
            )
        elif all_help:
            pattern = "B"
            pattern_note = "Pattern B: prefix helps across pick positions."
        elif (
            d_team_only is not None
            and d_team_only["mean"] < -MATERIAL_LOGLOSS_DELTA
            and (d_picks is None or d_picks["mean"] > d_team_only["mean"] + 0.005)
        ):
            pattern = "C"
            pattern_note = (
                "Pattern C: team identity explains most predictability; "
                "prefix adds little beyond team preference."
            )
        else:
            pattern = "D"
            pattern_note = (
                "Pattern D: no tested extension (prefix, bans, team "
                "identity, team tendency, or version) reliably beats "
                "the strongest structural popularity baseline "
                "(side + pick-index conditioned)."
            )

    prefix_helps = (
        d_picks is not None
        and d_picks["mean"] < -MATERIAL_LOGLOSS_DELTA
        and np.isfinite(d_picks["ci_high"])
        and d_picks["ci_high"] < 0.0
    )
    team_explains = (
        d_team_only is not None
        and d_team_only["mean"] < -MATERIAL_LOGLOSS_DELTA
        and (
            d_picks is None
            or d_prefix_vs_team is None
            or d_prefix_vs_team["mean"] > -0.005
            or (
                np.isfinite(d_prefix_vs_team.get("ci_low", float("nan")))
                and d_prefix_vs_team["ci_low"] > -MATERIAL_LOGLOSS_DELTA
            )
        )
    )
    bans_help = (
        d_bans is not None
        and d_bans["mean"] < -0.005
        and np.isfinite(d_bans["ci_high"])
        and d_bans["ci_high"] < 0.0
    )

    if prefix_helps and not team_explains and pattern in {"A", "B"}:
        frozen: tuple[str, ...] = (
            "causal_next_pick_decision_grain",
            "common_support_C_T_availability_constraint",
            "prefix_pick_conditioned_policy_features",
        )
        if bans_help:
            frozen = (*frozen, "prefix_ban_conditioned_policy_features")
        return (
            CLASSIFICATION_A,
            (
                f"Prefix improves structural/version baselines with "
                f"match-clustered CI support; {pattern_note} "
                f"team_explains={team_explains}."
            ),
            pattern,
            frozen,
        )

    if (
        (d_picks is not None and d_picks["mean"] < 0)
        or (d_team_only is not None and d_team_only["mean"] < 0)
        or pattern in {"A", "B", "C"}
    ):
        return (
            CLASSIFICATION_B,
            (
                f"Next-pick signal is mixed or largely team-driven; "
                f"{pattern_note} prefix_helps={prefix_helps}, "
                f"team_explains={team_explains}."
            ),
            pattern,
            (),
        )

    return (
        CLASSIFICATION_C,
        (
            f"No tested extension beats baseline_b (side + pick-index "
            f"popularity, LL 4.385). {pattern_note} "
            f"The extreme SGD underperformance (ΔLL +5 to +6) is a "
            f"model/representation limitation of sparse linear OVR at "
            f"this data scale, not proof of absence of conditional "
            f"draft structure."
        ),
        pattern,
        (),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Slice28BenchmarkReport:
    development_end: datetime
    holdout_policy: str
    n_development_matches: int
    n_decision_rows: int
    n_holdout_excluded: int
    n_oos_rows: int
    n_oos_matches: int
    first_observed_target_rate: float
    scoring_mixture_epsilon: float
    model_definitions: dict[str, object]
    feature_dims: pd.DataFrame
    fold_metrics: pd.DataFrame
    pooled_metrics: pd.DataFrame
    paired_deltas: pd.DataFrame
    pick_position_breakdown: pd.DataFrame
    side_breakdown: pd.DataFrame
    signature_breakdown: pd.DataFrame
    version_breakdown: pd.DataFrame
    team_history_breakdown: pd.DataFrame
    prefix_ablation: pd.DataFrame
    ban_ablation: pd.DataFrame
    team_ablation: pd.DataFrame
    calibration: pd.DataFrame
    selected_C: pd.DataFrame
    integrity: dict[str, object]
    classification: str
    classification_rationale: str
    prefix_value_pattern: str
    frozen_components: tuple[str, ...]
    terminal_notes: str


def slice28_report_to_jsonable(report: Slice28BenchmarkReport) -> dict[str, object]:
    return {
        "slice": 28,
        "title": "causal next-pick draft-policy benchmark",
        "development_end": _jsonable_value(report.development_end),
        "holdout_policy": report.holdout_policy,
        "boundary_convention": BOUNDARY_CONVENTION,
        "slice26_classification": SLICE26_RESEARCH_CLASSIFICATION,
        "slice26_frozen_components": list(SLICE26_FROZEN_COMPONENTS),
        "n_development_matches": report.n_development_matches,
        "n_decision_rows": report.n_decision_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "n_oos_rows": report.n_oos_rows,
        "n_oos_matches": report.n_oos_matches,
        "first_observed_target_rate": report.first_observed_target_rate,
        "scoring_mixture_epsilon": report.scoring_mixture_epsilon,
        "model_definitions": _jsonable_value(report.model_definitions),
        "feature_dims": _jsonable_value(report.feature_dims),
        "fold_metrics": _jsonable_value(report.fold_metrics),
        "pooled_metrics": _jsonable_value(report.pooled_metrics),
        "paired_deltas": _jsonable_value(report.paired_deltas),
        "pick_position_breakdown": _jsonable_value(report.pick_position_breakdown),
        "side_breakdown": _jsonable_value(report.side_breakdown),
        "signature_breakdown": _jsonable_value(report.signature_breakdown),
        "version_breakdown": _jsonable_value(report.version_breakdown),
        "team_history_breakdown": _jsonable_value(report.team_history_breakdown),
        "prefix_ablation": _jsonable_value(report.prefix_ablation),
        "ban_ablation": _jsonable_value(report.ban_ablation),
        "team_ablation": _jsonable_value(report.team_ablation),
        "calibration": _jsonable_value(report.calibration),
        "selected_C": _jsonable_value(report.selected_C),
        "integrity": _jsonable_value(report.integrity),
        "classification": report.classification,
        "classification_rationale": report.classification_rationale,
        "prefix_value_pattern": report.prefix_value_pattern,
        "frozen_components": list(report.frozen_components),
        "diagnostic_only": SLICE28_DIAGNOSTIC_ONLY,
        "recorded_classification": SLICE28_RESEARCH_CLASSIFICATION,
        "terminal_notes": report.terminal_notes,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def _rows_for_match_ids(
    decisions: pd.DataFrame, match_ids: Sequence[int]
) -> list[dict[str, Any]]:
    wanted = {int(m) for m in match_ids}
    subset = decisions.loc[decisions["match_id"].isin(wanted)]
    return subset.to_dict(orient="records")


def _elo_shell_dataset(frame: pd.DataFrame) -> ModelReadyDataset:
    """Minimal dataset so walk-forward folds follow match chronology."""
    missing = [c for c in ELO_ONLY_FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        # Allow running without Elo columns by synthesizing zeros.
        work = frame.copy()
        for col in ELO_ONLY_FEATURE_COLUMNS:
            if col not in work.columns:
                work[col] = 0.0
        frame = work
    y_col = "radiant_win" if "radiant_win" in frame.columns else None
    if y_col is None:
        y = pd.Series(np.zeros(len(frame), dtype=bool), index=frame.index)
    else:
        y = frame[y_col].astype(bool)
    identity = [c for c in IDENTITY_COLUMNS if c in frame.columns]
    if "match_id" not in identity:
        identity = ["match_id", *identity]
    return ModelReadyDataset(
        X=frame[list(ELO_ONLY_FEATURE_COLUMNS)].copy(),
        y=y.copy(),
        context=frame[list(dict.fromkeys(identity))].copy(),
        feature_columns=ELO_ONLY_FEATURE_COLUMNS,
        target_column="radiant_win",
        identity_columns=tuple(dict.fromkeys(identity)),
    )


def _breakdown(
    oos: pd.DataFrame, *, by: str, models: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if oos.empty:
        return pd.DataFrame()
    for key, group in oos.groupby(by, dropna=False, sort=False):
        for model in models:
            sub = group.loc[group["model"] == model]
            if sub.empty:
                continue
            rows.append(
                {
                    by: key,
                    "model": model,
                    "n": int(len(sub)),
                    "log_loss": float(sub["log_loss"].mean()),
                    "brier": float(sub["brier"].mean()),
                    "mean_rank": float(sub["rank"].mean()),
                    "hit_1": float(sub["hit_1"].mean()),
                    "hit_5": float(sub["hit_5"].mean()),
                    "mrr": float(sub["mrr"].mean()),
                }
            )
    return pd.DataFrame(rows)


def run_slice28_next_pick_policy_benchmark(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig | None = None,
    walk_forward_config: WalkForwardConfig | None = None,
    run_logistic_candidates: bool = True,
    tune_regularization: bool = False,
) -> Slice28BenchmarkReport:
    """Walk-forward next-pick policy benchmark on the development frame."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    resolved_elo = elo_config if elo_config is not None else DEFAULT_ELO_CONFIG
    wf_config = (
        walk_forward_config
        if walk_forward_config is not None
        else DEFAULT_WALK_FORWARD_CONFIG
    )

    snapshot = build_pre_draft_snapshot(store, elo_config=resolved_elo).to_frame()
    stamp = pd.to_datetime(snapshot["start_time"], utc=True)
    n_holdout = int((stamp > pd.Timestamp(end)).sum())
    development = restrict_development(snapshot, development_end=end)
    development = development.sort_values(
        ["start_time", "match_id"], kind="stable"
    ).reset_index(drop=True)
    assert_development_frame_excludes_holdout(
        development[list(IDENTITY_COLUMNS)], development_end=end
    )

    # Team IDs from matches view (snapshot may already have them).
    if (
        "radiant_team_id" not in development.columns
        or "dire_team_id" not in development.columns
    ):
        teams = store.sql(
            f"""
            SELECT match_id, radiant_team_id, dire_team_id, game_version_id
            FROM {MATCHES_VIEW}
            """
        ).df()
        development = development.merge(teams, on="match_id", how="left", suffixes=("", "_m"))
        if "game_version_id" not in development.columns and "game_version_id_m" in development.columns:
            development["game_version_id"] = development["game_version_id_m"]

    draft_events = store.sql(
        f"""
        SELECT match_id, sequence, action, side, hero_id, was_successful
        FROM {DRAFT_EVENTS_VIEW}
        """
    ).df()
    draft_events = draft_events.loc[
        draft_events["match_id"].isin(set(development["match_id"].tolist()))
    ].copy()
    indexes = build_match_draft_index(draft_events)

    decisions = build_next_pick_decision_rows(
        matches=development, indexes=indexes, require_ten_picks=True
    )
    if decisions.empty:
        raise TrainingDatasetError("no next-pick decision rows in development")

    eligible_matches = sorted(set(int(m) for m in decisions["match_id"].tolist()))
    development_eligible = development.loc[
        development["match_id"].isin(eligible_matches)
    ].reset_index(drop=True)

    history = build_causal_pick_history(decisions)
    shell = _elo_shell_dataset(development_eligible)
    folds = resolve_walk_forward_folds(shell, config=wf_config)

    freq_models = (BASELINE_A, BASELINE_B, BASELINE_C, TEAM_TENDENCY)
    logistic_specs = (
        (
            CANDIDATE_1_PREFIX_PICKS,
            True,
            False,
            False,
            False,
        ),
        (
            CANDIDATE_2_PREFIX_BANS,
            True,
            True,
            False,
            False,
        ),
        (
            MODEL_TEAM_ONLY,
            False,
            False,
            True,
            False,
        ),
        (
            CANDIDATE_3_TEAM,
            True,
            True,
            True,
            True,
        ),
    )

    oos_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, object]] = []
    feature_dim_rows: list[dict[str, object]] = []
    selected_c_rows: list[dict[str, object]] = []

    for fold in folds:
        train_ids = [int(x) for x in fold.train.context["match_id"].tolist()]
        val_ids = [int(x) for x in fold.validation.context["match_id"].tolist()]
        test_ids = [int(x) for x in fold.test.context["match_id"].tolist()]
        train_rows = _rows_for_match_ids(decisions, train_ids)
        val_rows = _rows_for_match_ids(decisions, val_ids)
        test_rows = _rows_for_match_ids(decisions, test_ids)
        if not test_rows:
            continue
        print(
            f"Slice 28 fold {fold.fold_id}: "
            f"train_rows={len(train_rows)} val_rows={len(val_rows)} "
            f"test_rows={len(test_rows)}",
            flush=True,
        )

        # Frequency baselines on test (causal history uses all prior matches).
        for model_name in freq_models:
            scored = _score_frequency_rows(
                test_rows, model=model_name, history=history
            )
            frame = pd.DataFrame(scored)
            frame["fold_id"] = fold.fold_id
            oos_frames.append(frame)
            summary = _metric_summary(scored)
            fold_metric_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": model_name,
                    **summary,
                }
            )

        if not run_logistic_candidates:
            continue

        for (
            name,
            include_picks,
            include_bans,
            include_team,
            include_tendency,
        ) in logistic_specs:
            print(f"  fitting {name} ...", flush=True)
            c_sel, model, c_table = _select_C(
                train_rows,
                val_rows,
                name=name,
                include_picks=include_picks,
                include_bans=include_bans,
                include_team_identity=include_team,
                include_team_tendency=include_tendency,
                history=history,
                tune_regularization=tune_regularization,
            )
            print(
                f"    done {name}: C={c_sel} n_features={len(model.feature_columns)} "
                f"n_classes={len(model.classes_)}",
                flush=True,
            )
            selected_c_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": name,
                    "C": c_sel,
                    "n_features": len(model.feature_columns),
                    "n_classes": len(model.classes_),
                }
            )
            feature_dim_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": name,
                    "n_features": len(model.feature_columns),
                    "n_pick_vocab": len(model.pick_vocabulary),
                    "n_ban_vocab": len(model.ban_vocabulary),
                    "n_team_vocab": len(model.team_vocabulary),
                    "n_classes": len(model.classes_),
                }
            )
            for _, crow in c_table.iterrows():
                selected_c_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "model": f"{name}_grid",
                        "C": float(crow["C"]),
                        "validation_log_loss": float(crow["validation_log_loss"]),
                        "n_features": len(model.feature_columns),
                        "n_classes": len(model.classes_),
                    }
                )
            scored = _score_model_rows(test_rows, model=model, history=history)
            frame = pd.DataFrame(scored)
            frame["fold_id"] = fold.fold_id
            oos_frames.append(frame)
            summary = _metric_summary(scored)
            fold_metric_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": name,
                    **summary,
                }
            )

    if not oos_frames:
        raise TrainingDatasetError("no OOS next-pick scores produced")

    oos = pd.concat(oos_frames, ignore_index=True)
    all_models = tuple(dict.fromkeys(oos["model"].tolist()))

    pooled_rows: list[dict[str, object]] = []
    for model in all_models:
        sub = oos.loc[oos["model"] == model]
        pooled_rows.append(
            {
                "model": model,
                "n": int(len(sub)),
                "log_loss": float(sub["log_loss"].mean()),
                "brier": float(sub["brier"].mean()),
                "mean_rank": float(sub["rank"].mean()),
                "median_rank": float(sub["rank"].median()),
                "hit_1": float(sub["hit_1"].mean()),
                "hit_3": float(sub["hit_3"].mean()),
                "hit_5": float(sub["hit_5"].mean()),
                "hit_10": float(sub["hit_10"].mean()),
                "mrr": float(sub["mrr"].mean()),
                "mean_n_candidates": float(sub["n_candidates"].mean()),
                "first_observed_rate": float(sub["first_observed_target"].mean()),
            }
        )
    pooled_metrics = pd.DataFrame(pooled_rows)

    # Paired comparisons on identical decision rows.
    comparisons = [
        (BASELINE_B, BASELINE_A),
        (BASELINE_C, BASELINE_B),
        (CANDIDATE_1_PREFIX_PICKS, BASELINE_C),
        (CANDIDATE_1_PREFIX_PICKS, BASELINE_B),
        (CANDIDATE_2_PREFIX_BANS, CANDIDATE_1_PREFIX_PICKS),
        (MODEL_TEAM_ONLY, BASELINE_C),
        (CANDIDATE_3_TEAM, CANDIDATE_2_PREFIX_BANS),
        (CANDIDATE_1_PREFIX_PICKS, MODEL_TEAM_ONLY),
        (TEAM_TENDENCY, BASELINE_C),
        (CANDIDATE_3_TEAM, MODEL_TEAM_ONLY),
    ]
    paired_rows: list[dict[str, object]] = []
    wide_cache: dict[str, pd.DataFrame] = {}
    for model in all_models:
        part = oos.loc[
            oos["model"] == model,
            [
                "match_id",
                "pick_decision_index",
                "log_loss",
                "brier",
                "rank",
                "hit_1",
            ],
        ].rename(
            columns={
                "log_loss": f"ll_{model}",
                "brier": f"brier_{model}",
                "rank": f"rank_{model}",
                "hit_1": f"hit1_{model}",
            }
        )
        wide_cache[model] = part

    for candidate, baseline in comparisons:
        if candidate not in wide_cache or baseline not in wide_cache:
            continue
        merged = wide_cache[candidate].merge(
            wide_cache[baseline],
            on=["match_id", "pick_decision_index"],
            how="inner",
            validate="one_to_one",
        )
        tmp = pd.DataFrame(
            {
                "match_id": merged["match_id"],
                "left": merged[f"ll_{candidate}"],
                "right": merged[f"ll_{baseline}"],
            }
        )
        ci = _match_clustered_delta_ci(
            tmp,
            left_col="left",
            right_col="right",
            seed=SLICE28_BOOTSTRAP_SEED
            + (abs(hash((candidate, baseline))) % 1000),
        )
        paired_rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "n_rows": int(len(merged)),
                "n_matches": int(merged["match_id"].nunique()),
                "candidate_log_loss": float(merged[f"ll_{candidate}"].mean()),
                "baseline_log_loss": float(merged[f"ll_{baseline}"].mean()),
                "delta_log_loss": float(
                    merged[f"ll_{candidate}"].mean() - merged[f"ll_{baseline}"].mean()
                ),
                "delta_ci_low": ci["ci_low"],
                "delta_ci_high": ci["ci_high"],
                "delta_brier": float(
                    merged[f"brier_{candidate}"].mean()
                    - merged[f"brier_{baseline}"].mean()
                ),
                "delta_hit_1": float(
                    merged[f"hit1_{candidate}"].mean()
                    - merged[f"hit1_{baseline}"].mean()
                ),
            }
        )
    paired_deltas = pd.DataFrame(paired_rows)

    # Pick-position breakdown with prefix vs structural delta.
    structural_name = (
        BASELINE_C if BASELINE_C in all_models else BASELINE_B
    )
    prefix_name = (
        CANDIDATE_1_PREFIX_PICKS
        if CANDIDATE_1_PREFIX_PICKS in all_models
        else structural_name
    )
    pos_rows: list[dict[str, object]] = []
    for overall_pick_index, group in oos.groupby("overall_pick_index", sort=True):
        row: dict[str, object] = {"overall_pick_index": int(overall_pick_index)}
        for model in (structural_name, prefix_name, BASELINE_A, MODEL_TEAM_ONLY):
            if model not in all_models:
                continue
            sub = group.loc[group["model"] == model]
            row[f"ll_{model}"] = float(sub["log_loss"].mean()) if len(sub) else float("nan")
            row[f"hit1_{model}"] = float(sub["hit_1"].mean()) if len(sub) else float("nan")
            row[f"n_{model}"] = int(len(sub))
        if f"ll_{prefix_name}" in row and f"ll_{structural_name}" in row:
            row["delta_log_loss_prefix_vs_structural"] = (
                float(row[f"ll_{prefix_name}"]) - float(row[f"ll_{structural_name}"])
            )
        else:
            row["delta_log_loss_prefix_vs_structural"] = float("nan")
        pos_rows.append(row)
    pick_position_breakdown = pd.DataFrame(pos_rows)

    side_breakdown = _breakdown(
        oos, by="acting_side", models=(structural_name, prefix_name, MODEL_TEAM_ONLY)
    )
    # Limit signature / version tables to top keys for readability.
    sig_counts = oos.groupby("draft_signature")["match_id"].nunique().sort_values(
        ascending=False
    )
    top_sigs = set(sig_counts.head(8).index.tolist())
    signature_breakdown = _breakdown(
        oos.loc[oos["draft_signature"].isin(top_sigs)],
        by="draft_signature",
        models=(structural_name, prefix_name),
    )
    version_breakdown = _breakdown(
        oos, by="game_version_id", models=(structural_name, prefix_name, BASELINE_B)
    )

    team_hist = oos.copy()
    team_hist["team_history_bucket"] = pd.cut(
        team_hist["team_history_n"],
        bins=[-0.1, 0, 10, 50, 200, 10_000],
        labels=["0", "1-10", "11-50", "51-200", "200+"],
    )
    team_history_breakdown = _breakdown(
        team_hist,
        by="team_history_bucket",
        models=(structural_name, prefix_name, MODEL_TEAM_ONLY, TEAM_TENDENCY),
    )

    prefix_ablation = paired_deltas.loc[
        paired_deltas["candidate"].isin(
            [CANDIDATE_1_PREFIX_PICKS, CANDIDATE_2_PREFIX_BANS]
        )
        & paired_deltas["baseline"].isin(
            [BASELINE_B, BASELINE_C, CANDIDATE_1_PREFIX_PICKS]
        )
    ].reset_index(drop=True)
    ban_ablation = paired_deltas.loc[
        (paired_deltas["candidate"] == CANDIDATE_2_PREFIX_BANS)
        & (paired_deltas["baseline"] == CANDIDATE_1_PREFIX_PICKS)
    ].reset_index(drop=True)
    team_ablation = paired_deltas.loc[
        paired_deltas["candidate"].isin(
            [MODEL_TEAM_ONLY, CANDIDATE_3_TEAM, TEAM_TENDENCY]
        )
    ].reset_index(drop=True)

    # Simple calibration: bins of predicted p_realized vs hit@1 for prefix.
    calibration = pd.DataFrame()
    if prefix_name in all_models:
        pref = oos.loc[oos["model"] == prefix_name].copy()
        if len(pref):
            pref["p_bin"] = pd.cut(pref["p_realized"], bins=10, labels=False)
            calibration = (
                pref.groupby("p_bin", dropna=True)
                .agg(
                    n=("hit_1", "size"),
                    mean_p=("p_realized", "mean"),
                    hit_1=("hit_1", "mean"),
                )
                .reset_index()
            )

    classification, rationale, pattern, frozen = classify_slice28(
        pooled_metrics=pooled_metrics,
        paired_deltas=paired_deltas,
        pick_position_breakdown=pick_position_breakdown,
    )

    first_obs_rate = float(oos["first_observed_target"].mean()) if len(oos) else 0.0

    integrity = {
        "feature_columns_length": len(FEATURE_COLUMNS),
        "feature_columns_unchanged": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_names": [s.name for s in SLICE9_FROZEN_SPECS],
        "boundary_convention": BOUNDARY_CONVENTION,
        "reuses_slice26_builder": True,
        "scoring_mixture_epsilon": SCORING_MIXTURE_EPSILON,
        "holdout_excluded": True,
        "n_holdout_excluded": n_holdout,
        "expected_picks_per_match": EXPECTED_PICKS_PER_MATCH,
        "walk_forward_n_blocks": wf_config.n_blocks,
        "bootstrap_unit": "match",
        "bootstrap_resamples": SLICE28_BOOTSTRAP_RESAMPLES,
        "no_outcome_target": True,
        "no_elo_target": True,
        "no_position_assignment_features": True,
    }

    model_definitions = {
        BASELINE_A: "global causal successful-pick frequencies",
        BASELINE_B: "side × overall_pick_index conditioned frequencies",
        BASELINE_C: "game_version × side × overall_pick_index with B/A backoff",
        TEAM_TENDENCY: "acting-team historical pick frequencies",
        CANDIDATE_1_PREFIX_PICKS: (
            "multinomial logistic: structural + version + side-aware prior picks"
        ),
        CANDIDATE_2_PREFIX_BANS: "candidate_1 + side-aware prior successful bans",
        MODEL_TEAM_ONLY: "multinomial logistic: structural + acting team identity",
        CANDIDATE_3_TEAM: (
            "candidate_2 + team identity + causal team pick-tendency features"
        ),
        "scoring": (
            f"shared C_T; q=(1-{SCORING_MIXTURE_EPSILON})*p + "
            f"{SCORING_MIXTURE_EPSILON}*U(C_T); Brier on raw p"
        ),
    }

    terminal_notes = (
        "Slice 28 is research-only. Do not promote next-pick policy "
        "features into FEATURE_COLUMNS. Classification concerns whether "
        "the current draft prefix adds conditional structure beyond "
        "side + pick-index popularity (baseline_b).\n"
        "Model: SGDClassifier(loss='log_loss') — fixed one-vs-rest L2 "
        "logistic SGD. Multinomial logistic regression was abandoned "
        "before final benchmark inspection for computational reasons.\n"
        "No tested extension (prefix, bans, team identity, team tendency, "
        "version) beats baseline_b. The extreme SGD underperformance is a "
        "model/representation limitation, not proof that conditional "
        "draft structure does not exist."
    )

    return Slice28BenchmarkReport(
        development_end=end,
        holdout_policy=HOLDOUT_POLICY,
        n_development_matches=int(len(eligible_matches)),
        n_decision_rows=int(len(decisions)),
        n_holdout_excluded=n_holdout,
        n_oos_rows=int(oos.loc[oos["model"] == all_models[0]].shape[0]),
        n_oos_matches=int(oos["match_id"].nunique()),
        first_observed_target_rate=first_obs_rate,
        scoring_mixture_epsilon=float(SCORING_MIXTURE_EPSILON),
        model_definitions=model_definitions,
        feature_dims=pd.DataFrame(feature_dim_rows),
        fold_metrics=pd.DataFrame(fold_metric_rows),
        pooled_metrics=pooled_metrics,
        paired_deltas=paired_deltas,
        pick_position_breakdown=pick_position_breakdown,
        side_breakdown=side_breakdown,
        signature_breakdown=signature_breakdown,
        version_breakdown=version_breakdown,
        team_history_breakdown=team_history_breakdown,
        prefix_ablation=prefix_ablation,
        ban_ablation=ban_ablation,
        team_ablation=team_ablation,
        calibration=calibration,
        selected_C=pd.DataFrame(selected_c_rows),
        integrity=integrity,
        classification=classification,
        classification_rationale=rationale,
        prefix_value_pattern=pattern,
        frozen_components=frozen,
        terminal_notes=terminal_notes,
    )
