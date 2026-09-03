"""Slice 27: incremental draft-value walk-forward benchmark.

Tests whether side-aware heroes successfully picked by checkpoint ``N``
improve Radiant-win prediction beyond pre-match Elo, and how that value
accumulates as picks are revealed.

Uses the frozen Slice 26 causal sequential draft-state substrate. Does
not redesign draft reconstruction, does not add production features,
does not score the frozen holdout, and does not introduce synergies,
counters, embeddings, or player↔hero assignment.

Checkpoints
-----------
Fixed by **number of successful picks revealed**, not raw
``draft_events.sequence``. Checkpoint ``N`` is the Slice 26 state
**immediately after** the Nth successful PICK (``boundary_t =
sequence_of_Nth_pick + 1``). Leading bans before that boundary are
causally visible; the primary model uses picks only. A secondary
ablation adds successful bans observed before the same boundary.

Hero encoding
-------------
Per fold, vocabulary = heroes successfully picked in the fold's TRAIN
matches only. Each vocab hero is one column:

    +1 Radiant picked, -1 Dire picked, 0 not yet picked / unseen

Unseen heroes in validation/test have no column (deterministic zeros
relative to the train vocabulary).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    DRAFT_EVENTS_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
    build_pre_draft_snapshot,
)
from dota_predictor.features.team_elo import (
    DEFAULT_ELO_CONFIG,
    TEAM_ELO_FEATURE_COLUMNS,
    EloConfig,
)
from dota_predictor.training.dataset import ModelReadyDataset, TrainingDatasetError
from dota_predictor.training.evaluation import _fit_logistic, _select_regularization
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
)
from dota_predictor.training.logistic_model import LogisticRegressionConfig
from dota_predictor.training.metrics import (
    bootstrap_mean_ci,
    evaluate_probabilities,
    per_sample_log_loss,
)
from dota_predictor.training.player_performance_target import (
    _jsonable_value,
    restrict_development,
)
from dota_predictor.training.preprocessing import PreprocessingSpec
from dota_predictor.training.sequential_draft_state import (
    ACTION_BAN,
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
    FROZEN_DEVELOPMENT_MATCH_COUNT,
    FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES,
    FROZEN_HOLDOUT_BOOTSTRAP_SEED,
    assert_development_frame_excludes_holdout,
    utc_datetime,
)
from dota_predictor.training.split import DatasetPartition
from dota_predictor.training.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG,
    ELO_BLOCK_SPEC_NAME,
    WalkForwardConfig,
    resolve_walk_forward_folds,
)

__all__ = [
    "CHECKPOINTS",
    "CLASSIFICATION_A",
    "CLASSIFICATION_B",
    "CLASSIFICATION_C",
    "HOLDOUT_POLICY",
    "SLICE27_BOOTSTRAP_RESAMPLES",
    "SLICE27_BOOTSTRAP_SEED",
    "SLICE27_CANDIDATE_SPEC_NAME",
    "SLICE27_DIAGNOSTIC_ONLY",
    "SLICE27_FROZEN_COMPONENTS",
    "SLICE27_PICKS_PLUS_BANS_SPEC_NAME",
    "SLICE27_REFERENCE_SPEC_NAME",
    "SLICE27_RESEARCH_CLASSIFICATION",
    "MatchDraftIndex",
    "Slice27BenchmarkReport",
    "ban_column_name",
    "boundary_after_n_successful_picks",
    "build_match_draft_index",
    "classify_slice27",
    "encode_side_aware_indicators",
    "hero_column_name",
    "run_slice27_sequential_draft_benchmark",
    "slice27_report_to_jsonable",
    "successful_ban_prefix",
    "successful_pick_prefix",
    "train_ban_vocabulary",
    "train_pick_vocabulary",
]


CHECKPOINTS: tuple[int, ...] = (0, 2, 4, 6, 8, 10)

SLICE27_REFERENCE_SPEC_NAME = ELO_BLOCK_SPEC_NAME
SLICE27_CANDIDATE_SPEC_NAME = "logistic_elo_plus_checkpoint_picks"
SLICE27_PICKS_PLUS_BANS_SPEC_NAME = "logistic_elo_plus_checkpoint_picks_bans"

SLICE27_BOOTSTRAP_RESAMPLES = FROZEN_HOLDOUT_BOOTSTRAP_RESAMPLES
SLICE27_BOOTSTRAP_SEED = FROZEN_HOLDOUT_BOOTSTRAP_SEED

HOLDOUT_POLICY = (
    "development_oos_only: frozen Slice 9 holdout remains reserved. "
    "Slice 27 scores expanding-window OOS on "
    "start_time <= FROZEN_DEVELOPMENT_END only."
)

CLASSIFICATION_A = (
    "A — freeze picked-hero draft block for downstream draft research: "
    "side-aware revealed hero identities provide clear, confirmation-"
    "stable incremental outcome information beyond Elo at one or more "
    "meaningful checkpoints."
)
CLASSIFICATION_B = (
    "B — suggestive / unstable: average improvements exist but folds/"
    "signs are unstable, only isolated checkpoints help, gains are "
    "very small or signature-sensitive, or full draft helps while live "
    "intermediate states are inconclusive."
)
CLASSIFICATION_C = (
    "C — do not freeze: side-aware revealed hero identities do not "
    "reliably improve Elo."
)

SLICE27_RESEARCH_CLASSIFICATION = "C"
SLICE27_DIAGNOSTIC_ONLY = True
SLICE27_FROZEN_COMPONENTS: tuple[str, ...] = ()


def hero_column_name(hero_id: int) -> str:
    return f"pick_side_hero_{int(hero_id)}"


def ban_column_name(hero_id: int) -> str:
    return f"ban_side_hero_{int(hero_id)}"


def successful_pick_prefix(
    events: Sequence[Mapping[str, Any]], *, n: int
) -> list[dict[str, Any]]:
    """First ``n`` successful picks in canonical sequence order."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    picks = [
        {
            "sequence": int(e["sequence"]),
            "side": str(e["side"]),
            "hero_id": int(e["hero_id"]),
            "action": ACTION_PICK,
            "was_successful": e.get("was_successful"),
        }
        for e in sorted(
            (
                ev
                for ev in events
                if ev.get("action") == ACTION_PICK
                and event_is_actual(ev.get("action"), ev.get("was_successful"))
                and ev.get("sequence") is not None
                and ev.get("hero_id") is not None
            ),
            key=lambda ev: int(ev["sequence"]),
        )
    ]
    return picks[:n]


def successful_ban_prefix(
    events: Sequence[Mapping[str, Any]], *, boundary_t: int
) -> list[dict[str, Any]]:
    """Successful bans with ``sequence < boundary_t`` (Slice 26 prefix)."""
    bans: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("action") != ACTION_BAN:
            continue
        if not event_is_actual(ev.get("action"), ev.get("was_successful")):
            continue
        seq = ev.get("sequence")
        if seq is None or int(seq) >= boundary_t:
            continue
        if ev.get("hero_id") is None:
            continue
        bans.append(
            {
                "sequence": int(seq),
                "side": str(ev["side"]),
                "hero_id": int(ev["hero_id"]),
                "action": ACTION_BAN,
                "was_successful": ev.get("was_successful"),
            }
        )
    bans.sort(key=lambda row: row["sequence"])
    return bans


def boundary_after_n_successful_picks(
    events: Sequence[Mapping[str, Any]], *, n: int
) -> int | None:
    """Slice 26 ``boundary_t`` immediately after the Nth successful pick.

    For ``n == 0``, returns the sequence of the first successful pick
    (state before any pick, after any leading bans), or ``0`` when the
    match has no picks. Returns ``None`` when ``n > 0`` and fewer than
    ``n`` successful picks exist.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    all_picks = successful_pick_prefix(events, n=10**9)
    if n == 0:
        if not all_picks:
            return 0
        return int(all_picks[0]["sequence"])
    if len(all_picks) < n:
        return None
    return int(all_picks[n - 1]["sequence"]) + 1


def encode_side_aware_indicators(
    *,
    radiant_hero_ids: Sequence[int],
    dire_hero_ids: Sequence[int],
    vocabulary: Sequence[int],
    column_fn: Callable[[int], str] = hero_column_name,
) -> dict[str, float]:
    """Encode side-aware presence: +1 Radiant, -1 Dire, 0 absent."""
    radiant = {int(h) for h in radiant_hero_ids}
    dire = {int(h) for h in dire_hero_ids}
    overlap = radiant & dire
    if overlap:
        raise ValueError(f"hero id(s) appear on both sides: {sorted(overlap)}")
    out: dict[str, float] = {}
    for hero_id in vocabulary:
        name = column_fn(int(hero_id))
        if hero_id in radiant:
            out[name] = 1.0
        elif hero_id in dire:
            out[name] = -1.0
        else:
            out[name] = 0.0
    return out


def _side_lists(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    radiant: list[int] = []
    dire: list[int] = []
    for ev in events:
        side = ev["side"]
        hero = int(ev["hero_id"])
        if side == SIDE_RADIANT:
            radiant.append(hero)
        elif side == SIDE_DIRE:
            dire.append(hero)
    return radiant, dire


def _action_side_signature(events: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(
        (e for e in events if e.get("sequence") is not None),
        key=lambda e: int(e["sequence"]),
    )
    return "".join(f"{str(e['action'])[0]}{str(e['side'])[0]}" for e in ordered)


@dataclass(frozen=True)
class MatchDraftIndex:
    """Precomputed causal draft facts for one match (Slice 26 consumers)."""

    match_id: int
    events: tuple[dict[str, Any], ...]
    successful_picks: tuple[dict[str, Any], ...]
    signature: str

    @property
    def n_successful_picks(self) -> int:
        return len(self.successful_picks)

    def eligible(self, n: int) -> bool:
        return self.n_successful_picks >= n

    def boundary(self, n: int) -> int | None:
        return boundary_after_n_successful_picks(self.events, n=n)

    def picks_at(self, n: int) -> tuple[list[int], list[int]]:
        return _side_lists(successful_pick_prefix(self.events, n=n))

    def bans_before(self, boundary_t: int) -> tuple[list[int], list[int]]:
        return _side_lists(successful_ban_prefix(self.events, boundary_t=boundary_t))

    def side_pick_counts(self, n: int) -> tuple[int, int]:
        radiant, dire = self.picks_at(n)
        return len(radiant), len(dire)


def _normalize_event(row: Mapping[str, Any]) -> dict[str, Any]:
    was = row.get("was_successful")
    if was is not None and not (isinstance(was, float) and np.isnan(was)):
        was_bool: bool | None = bool(was)
    else:
        was_bool = None
    return {
        "sequence": int(row["sequence"]),
        "action": str(row["action"]),
        "side": str(row["side"]),
        "hero_id": int(row["hero_id"]),
        "was_successful": was_bool,
    }


def build_match_draft_index(draft_events: pd.DataFrame) -> dict[int, MatchDraftIndex]:
    """Index draft events by match using Slice 26 success semantics."""
    by_match: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in draft_events.to_dict(orient="records"):
        by_match[int(row["match_id"])].append(_normalize_event(row))
    out: dict[int, MatchDraftIndex] = {}
    for match_id, events in by_match.items():
        events.sort(key=lambda e: e["sequence"])
        picks = tuple(successful_pick_prefix(events, n=10**9))
        out[match_id] = MatchDraftIndex(
            match_id=match_id,
            events=tuple(events),
            successful_picks=picks,
            signature=_action_side_signature(events),
        )
    return out


def train_pick_vocabulary(
    indexes: Mapping[int, MatchDraftIndex], train_match_ids: Sequence[int]
) -> tuple[int, ...]:
    heroes: set[int] = set()
    for match_id in train_match_ids:
        index = indexes.get(int(match_id))
        if index is None:
            continue
        for pick in index.successful_picks:
            heroes.add(int(pick["hero_id"]))
    return tuple(sorted(heroes))


def train_ban_vocabulary(
    indexes: Mapping[int, MatchDraftIndex], train_match_ids: Sequence[int]
) -> tuple[int, ...]:
    heroes: set[int] = set()
    for match_id in train_match_ids:
        index = indexes.get(int(match_id))
        if index is None:
            continue
        for ev in index.events:
            if ev["action"] != ACTION_BAN:
                continue
            if not event_is_actual(ev["action"], ev["was_successful"]):
                continue
            heroes.add(int(ev["hero_id"]))
    return tuple(sorted(heroes))


def checkpoint_pick_ban_features(
    index: MatchDraftIndex,
    *,
    n_picks: int,
    pick_vocabulary: Sequence[int],
    ban_vocabulary: Sequence[int] | None = None,
    include_bans: bool = False,
    verify_slice26: bool = False,
) -> dict[str, float] | None:
    """Side-aware pick (+ optional ban) indicators at checkpoint ``n_picks``."""
    if not index.eligible(n_picks):
        return None
    boundary = index.boundary(n_picks)
    if boundary is None:
        return None
    radiant_picks, dire_picks = index.picks_at(n_picks)
    if verify_slice26:
        state = build_draft_prefix_state(
            match_id=index.match_id,
            start_time=pd.Timestamp(0, tz="UTC"),
            game_version_id=None,
            boundary_t=boundary,
            events=index.events,
            radiant_team_id=1,
            dire_team_id=2,
            radiant_player_ids=(1, 2, 3, 4, 5),
            dire_player_ids=(6, 7, 8, 9, 10),
        )
        if list(state["radiant_pick_hero_ids"]) != radiant_picks:
            raise TrainingDatasetError("Slice 26 radiant picks mismatch")
        if list(state["dire_pick_hero_ids"]) != dire_picks:
            raise TrainingDatasetError("Slice 26 dire picks mismatch")
    row = encode_side_aware_indicators(
        radiant_hero_ids=radiant_picks,
        dire_hero_ids=dire_picks,
        vocabulary=pick_vocabulary,
        column_fn=hero_column_name,
    )
    if include_bans:
        if ban_vocabulary is None:
            raise ValueError("ban_vocabulary required when include_bans=True")
        radiant_bans, dire_bans = index.bans_before(boundary)
        if verify_slice26:
            state = build_draft_prefix_state(
                match_id=index.match_id,
                start_time=pd.Timestamp(0, tz="UTC"),
                game_version_id=None,
                boundary_t=boundary,
                events=index.events,
                radiant_team_id=1,
                dire_team_id=2,
                radiant_player_ids=(1, 2, 3, 4, 5),
                dire_player_ids=(6, 7, 8, 9, 10),
            )
            if list(state["radiant_ban_hero_ids"]) != radiant_bans:
                raise TrainingDatasetError("Slice 26 radiant bans mismatch")
            if list(state["dire_ban_hero_ids"]) != dire_bans:
                raise TrainingDatasetError("Slice 26 dire bans mismatch")
        row.update(
            encode_side_aware_indicators(
                radiant_hero_ids=radiant_bans,
                dire_hero_ids=dire_bans,
                vocabulary=ban_vocabulary,
                column_fn=ban_column_name,
            )
        )
    return row


def _elo_dataset(frame: pd.DataFrame) -> ModelReadyDataset:
    missing = [c for c in ELO_ONLY_FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise TrainingDatasetError(f"missing Elo columns: {missing}")
    return ModelReadyDataset(
        X=frame[list(ELO_ONLY_FEATURE_COLUMNS)].copy(),
        y=frame[TARGET_COLUMN].astype(bool).copy(),
        context=frame[list(IDENTITY_COLUMNS)].copy(),
        feature_columns=ELO_ONLY_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        identity_columns=IDENTITY_COLUMNS,
    )


def _partition_with_features(
    base: DatasetPartition,
    feature_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> DatasetPartition:
    merged = base.context[["match_id"]].merge(
        feature_frame, on="match_id", how="left", validate="one_to_one"
    )
    if merged[list(feature_columns)].isna().any().any():
        raise TrainingDatasetError("checkpoint feature frame has NA values")
    X = merged[list(feature_columns)].copy()
    X.index = base.X.index
    return DatasetPartition(X=X, y=base.y.copy(), context=base.context.copy())


def _build_checkpoint_matrix(
    match_ids: Sequence[int],
    indexes: Mapping[int, MatchDraftIndex],
    elo_by_match: Mapping[int, Mapping[str, float]],
    *,
    n_picks: int,
    pick_vocabulary: Sequence[int],
    ban_vocabulary: Sequence[int] | None = None,
    include_bans: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pick_cols = [hero_column_name(h) for h in pick_vocabulary]
    ban_cols = (
        [ban_column_name(h) for h in ban_vocabulary]
        if include_bans and ban_vocabulary is not None
        else []
    )
    for match_id in match_ids:
        mid = int(match_id)
        encoded = checkpoint_pick_ban_features(
            indexes[mid],
            n_picks=n_picks,
            pick_vocabulary=pick_vocabulary,
            ban_vocabulary=ban_vocabulary,
            include_bans=include_bans,
            verify_slice26=False,
        )
        if encoded is None:
            raise TrainingDatasetError(
                f"match {mid} ineligible for checkpoint n_picks={n_picks}"
            )
        row: dict[str, Any] = {"match_id": mid}
        row.update(elo_by_match[mid])
        row.update(encoded)
        for col in pick_cols:
            row.setdefault(col, 0.0)
        for col in ban_cols:
            row.setdefault(col, 0.0)
        rows.append(row)
    columns = ["match_id", *ELO_ONLY_FEATURE_COLUMNS, *pick_cols, *ban_cols]
    return pd.DataFrame(rows, columns=columns)


@dataclass
class Slice27BenchmarkReport:
    """Development walk-forward incremental draft-value results."""

    development_end: datetime
    holdout_policy: str
    n_development_matches: int
    n_holdout_excluded: int
    n_oos: int
    checkpoints: tuple[int, ...]
    checkpoint_coverage: pd.DataFrame
    checkpoint_curve: pd.DataFrame
    fold_deltas: pd.DataFrame
    accumulation: pd.DataFrame
    bootstrap: dict[str, object]
    ban_ablation: pd.DataFrame
    signature_diagnostics: pd.DataFrame
    side_pick_balance: pd.DataFrame
    feature_dims: pd.DataFrame
    selected_C: pd.DataFrame
    integrity: dict[str, object]
    classification: str
    classification_rationale: str
    accumulation_pattern: str
    frozen_components: tuple[str, ...]
    terminal_notes: str


def _bootstrap_delta(deltas: np.ndarray, *, seed: int) -> dict[str, float]:
    ci_low, ci_high = bootstrap_mean_ci(
        deltas,
        n_resamples=SLICE27_BOOTSTRAP_RESAMPLES,
        random_state=seed,
    )
    return {
        "mean": float(np.mean(deltas)) if deltas.size else float("nan"),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": int(deltas.size),
    }


def classify_slice27(
    *,
    checkpoint_curve: pd.DataFrame,
    fold_deltas: pd.DataFrame,
    bootstrap: Mapping[str, Mapping[str, float]],
) -> tuple[str, str, str, tuple[str, ...]]:
    """Map walk-forward evidence onto A/B/C and an accumulation pattern."""
    meaningful = checkpoint_curve.loc[checkpoint_curve["n_picks"] > 0].copy()
    if meaningful.empty:
        return CLASSIFICATION_C, "no positive-pick checkpoints evaluated.", "D", ()

    improving = meaningful.loc[meaningful["delta_log_loss"] < 0]
    later = meaningful.loc[meaningful["n_picks"] >= 6]
    terminal = meaningful.loc[meaningful["n_picks"] == 10]
    early = meaningful.loc[meaningful["n_picks"] <= 4]

    terminal_boot = bootstrap.get("10", {})
    terminal_ci_neg = (
        np.isfinite(terminal_boot.get("ci_high", float("nan")))
        and float(terminal_boot.get("ci_high", 0.0)) < 0.0
    )
    any_ci_neg = any(
        np.isfinite(bootstrap.get(str(int(n)), {}).get("ci_high", float("nan")))
        and float(bootstrap[str(int(n))]["ci_high"]) < 0.0
        for n in meaningful["n_picks"].tolist()
    )
    later_help = bool(len(later) and (later["delta_log_loss"] < 0).all())
    early_help = bool(len(early) and (early["delta_log_loss"] < 0).mean() >= 0.5)
    only_terminal = bool(
        len(terminal)
        and float(terminal["delta_log_loss"].iloc[0]) < 0
        and len(meaningful) > 1
        and (meaningful.loc[meaningful["n_picks"] < 10, "delta_log_loss"] >= 0).all()
    )
    no_help = bool((meaningful["delta_log_loss"] >= 0).all())

    if only_terminal:
        pattern = "B"
        pattern_note = (
            "Pattern B: only the full 10-pick checkpoint improves; early "
            "live-draft updates look weak."
        )
    elif no_help:
        pattern = "D"
        pattern_note = (
            "Pattern D: no checkpoint improves on Elo with side-aware "
            "hero main effects."
        )
    elif early_help and not later_help:
        pattern = "C"
        pattern_note = (
            "Pattern C: early checkpoints improve but later ones do not; "
            "check regularization / signature confounding."
        )
    else:
        pattern = "A"
        pattern_note = (
            "Pattern A: predictive value tends to accumulate toward later "
            "checkpoints (not necessarily strictly monotonic)."
        )

    best_row = meaningful.loc[meaningful["delta_log_loss"].idxmin()]
    best_n = int(best_row["n_picks"])
    best_folds = fold_deltas.loc[fold_deltas["n_picks"] == best_n]
    mixed_folds = bool(
        len(best_folds)
        and (best_folds["delta_log_loss"] < 0).any()
        and (best_folds["delta_log_loss"] > 0).any()
    )
    n_folds_neg = int((best_folds["delta_log_loss"] < 0).sum()) if len(best_folds) else 0
    n_folds = int(len(best_folds))
    majority_neg = n_folds >= 2 and n_folds_neg >= max(2, n_folds - 1)

    if (
        any_ci_neg
        and later_help
        and (majority_neg or terminal_ci_neg)
        and not mixed_folds
        and float(best_row["delta_log_loss"]) < -1e-4
    ):
        frozen = (
            "side_aware_checkpoint_pick_indicators",
            "elo_plus_checkpoint_picks_benchmark",
            f"supporting_checkpoints={tuple(int(x) for x in improving['n_picks'])}",
        )
        return (
            CLASSIFICATION_A,
            (
                f"Paired Δ log loss improves beyond Elo at checkpoints "
                f"{list(improving['n_picks'])} with confirmation-stable "
                f"evidence; {pattern_note}"
            ),
            pattern,
            frozen,
        )

    if no_help and not any_ci_neg:
        return (
            CLASSIFICATION_C,
            f"No checkpoint yields a reliable Elo improvement. {pattern_note}",
            pattern,
            (),
        )

    return (
        CLASSIFICATION_B,
        (
            f"Incremental draft signal is mixed or unstable "
            f"(best checkpoint={best_n}, mixed_folds={mixed_folds}, "
            f"any_ci_neg={any_ci_neg}). {pattern_note}"
        ),
        pattern,
        (),
    )


def run_slice27_sequential_draft_benchmark(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig | None = None,
    walk_forward_config: WalkForwardConfig | None = None,
    checkpoints: Sequence[int] = CHECKPOINTS,
    run_ban_ablation: bool = True,
) -> Slice27BenchmarkReport:
    """Walk-forward Elo vs Elo+checkpoint picks on the development frame."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    resolved_elo = elo_config if elo_config is not None else DEFAULT_ELO_CONFIG
    wf_config = (
        walk_forward_config
        if walk_forward_config is not None
        else DEFAULT_WALK_FORWARD_CONFIG
    )
    checkpoint_tuple = tuple(int(x) for x in checkpoints)

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

    draft_events = store.sql(
        f"""
        SELECT match_id, sequence, action, side, hero_id, was_successful
        FROM {DRAFT_EVENTS_VIEW}
        """
    ).df()
    draft_events = draft_events.loc[
        draft_events["match_id"].isin(set(development["match_id"]))
    ].copy()
    indexes = build_match_draft_index(draft_events)

    missing_draft = [
        int(m) for m in development["match_id"].tolist() if int(m) not in indexes
    ]
    if missing_draft:
        raise TrainingDatasetError(
            f"{len(missing_draft)} development matches lack draft_events"
        )

    elo_dataset = _elo_dataset(development)
    folds = resolve_walk_forward_folds(elo_dataset, config=wf_config)

    elo_by_match = {
        int(row["match_id"]): {col: float(row[col]) for col in ELO_ONLY_FEATURE_COLUMNS}
        for row in development[["match_id", *ELO_ONLY_FEATURE_COLUMNS]].to_dict(
            orient="records"
        )
    }

    coverage_rows: list[dict[str, object]] = []
    for n in checkpoint_tuple:
        eligible = sum(
            1 for mid in development["match_id"] if indexes[int(mid)].eligible(n)
        )
        coverage_rows.append(
            {
                "n_picks": n,
                "n_eligible": eligible,
                "n_development": len(development),
                "coverage": eligible / len(development) if len(development) else float("nan"),
            }
        )
    checkpoint_coverage = pd.DataFrame(coverage_rows)

    balance_rows: list[dict[str, object]] = []
    sig_counter: Counter[str] = Counter()
    for mid in development["match_id"].tolist():
        sig_counter[indexes[int(mid)].signature] += 1
    for n in checkpoint_tuple:
        radiant_counts: list[int] = []
        dire_counts: list[int] = []
        for mid in development["match_id"].tolist():
            index = indexes[int(mid)]
            if not index.eligible(n):
                continue
            r_count, d_count = index.side_pick_counts(n)
            radiant_counts.append(r_count)
            dire_counts.append(d_count)
        balance_rows.append(
            {
                "n_picks": n,
                "mean_radiant_picks": (
                    float(np.mean(radiant_counts)) if radiant_counts else float("nan")
                ),
                "mean_dire_picks": (
                    float(np.mean(dire_counts)) if dire_counts else float("nan")
                ),
                "share_equal_sides": (
                    float(
                        np.mean(
                            [r == d for r, d in zip(radiant_counts, dire_counts, strict=True)]
                        )
                    )
                    if radiant_counts
                    else float("nan")
                ),
            }
        )
    side_pick_balance = pd.DataFrame(balance_rows)
    signature_diagnostics = pd.DataFrame(
        [
            {"signature": sig, "n_matches": count, "share": count / len(development)}
            for sig, count in sig_counter.most_common()
        ]
    )

    preprocessing = PreprocessingSpec()
    curve_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    feature_dim_rows: list[dict[str, object]] = []
    selected_c_rows: list[dict[str, object]] = []
    bootstrap: dict[str, dict[str, float]] = {}
    ban_rows: list[dict[str, object]] = []
    oos_by_checkpoint: dict[int, pd.DataFrame] = {}

    elo_oos_frames: list[pd.DataFrame] = []
    for fold in folds:
        c_elo, _ = _select_regularization(
            fold.train, fold.validation, ELO_ONLY_FEATURE_COLUMNS
        )
        selected_c_rows.append(
            {
                "fold_id": fold.fold_id,
                "model": SLICE27_REFERENCE_SPEC_NAME,
                "n_picks": None,
                "C": c_elo,
                "n_features": len(ELO_ONLY_FEATURE_COLUMNS),
            }
        )
        elo_model = _fit_logistic(
            fold.train,
            ELO_ONLY_FEATURE_COLUMNS,
            config=LogisticRegressionConfig(C=c_elo, preprocessing=preprocessing),
        )
        elo_pred = elo_model.predict_radiant_win_proba(fold.test.X)
        elo_oos_frames.append(
            pd.DataFrame(
                {
                    "fold_id": fold.fold_id,
                    "match_id": fold.test.context["match_id"].to_numpy(),
                    "y_true": fold.test.y.astype(int).to_numpy(),
                    "p_elo": np.asarray(elo_pred, dtype=float),
                }
            )
        )
    elo_oos = pd.concat(elo_oos_frames, ignore_index=True)

    for n_picks in checkpoint_tuple:
        pick_oos_frames: list[pd.DataFrame] = []
        ban_oos_frames: list[pd.DataFrame] = []
        for fold in folds:
            train_ids = [int(x) for x in fold.train.context["match_id"].tolist()]
            val_ids = [int(x) for x in fold.validation.context["match_id"].tolist()]
            test_ids = [int(x) for x in fold.test.context["match_id"].tolist()]
            for mid in train_ids + val_ids + test_ids:
                if not indexes[mid].eligible(n_picks):
                    raise TrainingDatasetError(
                        f"fold {fold.fold_id} match {mid} ineligible at n={n_picks}"
                    )

            pick_vocab = train_pick_vocabulary(indexes, train_ids)
            ban_vocab = (
                train_ban_vocabulary(indexes, train_ids) if run_ban_ablation else ()
            )
            pick_cols = tuple(hero_column_name(h) for h in pick_vocab)
            feature_dim_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "n_picks": n_picks,
                    "n_pick_vocab": len(pick_vocab),
                    "n_ban_vocab": len(ban_vocab),
                    "n_candidate_features": len(ELO_ONLY_FEATURE_COLUMNS) + len(pick_cols),
                }
            )

            all_ids = train_ids + val_ids + test_ids
            y = fold.test.y.astype(int).to_numpy()
            fold_elo = elo_oos.loc[elo_oos["fold_id"] == fold.fold_id]
            if not np.array_equal(
                fold_elo["match_id"].to_numpy(),
                fold.test.context["match_id"].to_numpy(),
            ):
                raise TrainingDatasetError("Elo/candidate OOS match_id mismatch")
            p_elo = fold_elo["p_elo"].to_numpy()

            if n_picks == 0:
                candidate_columns = ELO_ONLY_FEATURE_COLUMNS
                train_p = fold.train
                val_p = fold.validation
                test_p = fold.test
            else:
                candidate_columns = ELO_ONLY_FEATURE_COLUMNS + pick_cols
                pick_matrix = _build_checkpoint_matrix(
                    all_ids,
                    indexes,
                    elo_by_match,
                    n_picks=n_picks,
                    pick_vocabulary=pick_vocab,
                    include_bans=False,
                )
                train_p = _partition_with_features(
                    fold.train, pick_matrix, candidate_columns
                )
                val_p = _partition_with_features(
                    fold.validation, pick_matrix, candidate_columns
                )
                test_p = _partition_with_features(
                    fold.test, pick_matrix, candidate_columns
                )

            c_cand, _ = _select_regularization(train_p, val_p, candidate_columns)
            selected_c_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": SLICE27_CANDIDATE_SPEC_NAME,
                    "n_picks": n_picks,
                    "C": c_cand,
                    "n_features": len(candidate_columns),
                }
            )
            cand_model = _fit_logistic(
                train_p,
                candidate_columns,
                config=LogisticRegressionConfig(
                    C=c_cand, preprocessing=preprocessing
                ),
            )
            p_cand = np.asarray(
                cand_model.predict_radiant_win_proba(test_p.X), dtype=float
            )
            cand_ll = per_sample_log_loss(y, p_cand)
            elo_ll = per_sample_log_loss(y, p_elo)
            delta = cand_ll - elo_ll
            fold_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "n_picks": n_picks,
                    "n_test": len(y),
                    "elo_log_loss": float(np.mean(elo_ll)),
                    "candidate_log_loss": float(np.mean(cand_ll)),
                    "delta_log_loss": float(np.mean(delta)),
                    "frac_better_than_elo": float(np.mean(delta < 0)),
                    "C": c_cand,
                    "n_features": len(candidate_columns),
                }
            )
            pick_oos_frames.append(
                pd.DataFrame(
                    {
                        "fold_id": fold.fold_id,
                        "match_id": fold.test.context["match_id"].to_numpy(),
                        "y_true": y,
                        "p_candidate": p_cand,
                        "p_elo": p_elo,
                        "candidate_log_loss": cand_ll,
                        "elo_log_loss": elo_ll,
                        "delta_log_loss": delta,
                        "signature": [
                            indexes[int(m)].signature
                            for m in fold.test.context["match_id"].tolist()
                        ],
                    }
                )
            )

            if run_ban_ablation:
                ban_cols = tuple(ban_column_name(h) for h in ban_vocab)
                ban_matrix = _build_checkpoint_matrix(
                    all_ids,
                    indexes,
                    elo_by_match,
                    n_picks=n_picks,
                    pick_vocabulary=pick_vocab if n_picks > 0 else (),
                    ban_vocabulary=ban_vocab,
                    include_bans=True,
                )
                if n_picks == 0:
                    ban_columns = ELO_ONLY_FEATURE_COLUMNS + ban_cols
                else:
                    ban_columns = ELO_ONLY_FEATURE_COLUMNS + pick_cols + ban_cols
                train_b = _partition_with_features(fold.train, ban_matrix, ban_columns)
                val_b = _partition_with_features(
                    fold.validation, ban_matrix, ban_columns
                )
                test_b = _partition_with_features(fold.test, ban_matrix, ban_columns)
                c_ban, _ = _select_regularization(train_b, val_b, ban_columns)
                selected_c_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "model": SLICE27_PICKS_PLUS_BANS_SPEC_NAME,
                        "n_picks": n_picks,
                        "C": c_ban,
                        "n_features": len(ban_columns),
                    }
                )
                ban_model = _fit_logistic(
                    train_b,
                    ban_columns,
                    config=LogisticRegressionConfig(
                        C=c_ban, preprocessing=preprocessing
                    ),
                )
                p_ban = np.asarray(
                    ban_model.predict_radiant_win_proba(test_b.X), dtype=float
                )
                ban_ll = per_sample_log_loss(y, p_ban)
                ban_oos_frames.append(
                    pd.DataFrame(
                        {
                            "fold_id": fold.fold_id,
                            "match_id": fold.test.context["match_id"].to_numpy(),
                            "picks_log_loss": cand_ll,
                            "picks_bans_log_loss": ban_ll,
                            "delta_bans_vs_picks": ban_ll - cand_ll,
                            "y_true": y,
                        }
                    )
                )

        paired = pd.concat(pick_oos_frames, ignore_index=True)
        oos_by_checkpoint[n_picks] = paired
        y_all = paired["y_true"]
        elo_metrics = evaluate_probabilities(y_all, paired["p_elo"])
        cand_metrics = evaluate_probabilities(y_all, paired["p_candidate"])
        deltas = paired["delta_log_loss"].to_numpy(dtype=float)
        boot = _bootstrap_delta(deltas, seed=SLICE27_BOOTSTRAP_SEED + n_picks)
        bootstrap[str(n_picks)] = boot
        curve_rows.append(
            {
                "n_picks": n_picks,
                "n_oos": len(paired),
                "elo_log_loss": elo_metrics.log_loss,
                "candidate_log_loss": cand_metrics.log_loss,
                "delta_log_loss": float(np.mean(deltas)),
                "delta_ci_low": boot["ci_low"],
                "delta_ci_high": boot["ci_high"],
                "elo_brier": elo_metrics.brier_score,
                "candidate_brier": cand_metrics.brier_score,
                "elo_auc": elo_metrics.roc_auc,
                "candidate_auc": cand_metrics.roc_auc,
                "elo_accuracy": elo_metrics.accuracy_at_0_5,
                "candidate_accuracy": cand_metrics.accuracy_at_0_5,
                "elo_ece": elo_metrics.expected_calibration_error,
                "candidate_ece": cand_metrics.expected_calibration_error,
                "frac_better_than_elo": float((deltas < 0).mean()),
            }
        )

        if run_ban_ablation and ban_oos_frames:
            ban_paired = pd.concat(ban_oos_frames, ignore_index=True)
            ban_delta = ban_paired["delta_bans_vs_picks"].to_numpy(dtype=float)
            ban_boot = _bootstrap_delta(
                ban_delta, seed=SLICE27_BOOTSTRAP_SEED + 100 + n_picks
            )
            ban_rows.append(
                {
                    "n_picks": n_picks,
                    "n_oos": len(ban_paired),
                    "picks_log_loss": float(ban_paired["picks_log_loss"].mean()),
                    "picks_bans_log_loss": float(
                        ban_paired["picks_bans_log_loss"].mean()
                    ),
                    "delta_bans_vs_picks": float(np.mean(ban_delta)),
                    "delta_ci_low": ban_boot["ci_low"],
                    "delta_ci_high": ban_boot["ci_high"],
                    "frac_bans_better": float((ban_delta < 0).mean()),
                }
            )

    checkpoint_curve = pd.DataFrame(curve_rows)
    fold_deltas = pd.DataFrame(fold_rows)
    feature_dims = pd.DataFrame(feature_dim_rows)
    selected_C = pd.DataFrame(selected_c_rows)
    ban_ablation = pd.DataFrame(ban_rows)

    acc_rows: list[dict[str, object]] = []
    ordered = checkpoint_curve.sort_values("n_picks")
    prev_ll: float | None = None
    prev_n: int | None = None
    for row in ordered.to_dict(orient="records"):
        delta_prev = (
            float(row["candidate_log_loss"] - prev_ll)
            if prev_ll is not None
            else float("nan")
        )
        acc_rows.append(
            {
                "n_picks": int(row["n_picks"]),
                "delta_vs_elo": float(row["delta_log_loss"]),
                "delta_vs_previous_checkpoint": delta_prev,
                "previous_n_picks": prev_n,
            }
        )
        prev_ll = float(row["candidate_log_loss"])
        prev_n = int(row["n_picks"])
    accumulation = pd.DataFrame(acc_rows)

    classification, rationale, pattern, frozen = classify_slice27(
        checkpoint_curve=checkpoint_curve,
        fold_deltas=fold_deltas,
        bootstrap=bootstrap,
    )

    if 10 in oos_by_checkpoint and not signature_diagnostics.empty:
        term = oos_by_checkpoint[10]
        top_sigs = set(signature_diagnostics.head(4)["signature"].tolist())
        sig_rows = []
        for sig in top_sigs:
            subset = term.loc[term["signature"] == sig]
            if len(subset) < 20:
                continue
            sig_rows.append(
                {
                    "signature": sig,
                    "n_oos": len(subset),
                    "delta_log_loss": float(subset["delta_log_loss"].mean()),
                }
            )
        if sig_rows:
            signature_diagnostics = signature_diagnostics.merge(
                pd.DataFrame(sig_rows),
                on="signature",
                how="left",
            )

    integrity = {
        "holdout_policy": HOLDOUT_POLICY,
        "development_end": end.isoformat(),
        "boundary_convention_consumed": BOUNDARY_CONVENTION,
        "slice26_classification": SLICE26_RESEARCH_CLASSIFICATION,
        "slice26_frozen_components": list(SLICE26_FROZEN_COMPONENTS),
        "holdout_scored": False,
        "feature_columns_length": len(FEATURE_COLUMNS),
        "feature_columns_unchanged": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "reference_is_elo_only": list(ELO_ONLY_FEATURE_COLUMNS)
        == list(TEAM_ELO_FEATURE_COLUMNS),
        "n_development_matches": len(development),
        "population_matches_expected": len(development)
        == FROZEN_DEVELOPMENT_MATCH_COUNT,
        "n_holdout_excluded": n_holdout,
        "n_oos": int(len(elo_oos)),
        "walk_forward_n_blocks": wf_config.n_blocks,
        "n_folds": len(folds),
        "checkpoints": list(checkpoint_tuple),
        "slice9_frozen_spec_names": [s.name for s in SLICE9_FROZEN_SPECS],
        "post_draft_block_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "hero_vocab_from_train_only": True,
        "interactions_included": False,
        "player_assignment_used": False,
        "slice23_24_25_revived": False,
    }

    terminal_notes = (
        "Existing post-draft draft_comparison / hero-meta blocks measure "
        "historical profile diffs at the completed draft, not raw "
        "side-aware hero identity indicators at causal checkpoints. "
        "Slice 27's 10-pick checkpoint is therefore a different "
        "representation and is not expected to numerically match those "
        "benchmarks."
    )

    return Slice27BenchmarkReport(
        development_end=end,
        holdout_policy=HOLDOUT_POLICY,
        n_development_matches=len(development),
        n_holdout_excluded=n_holdout,
        n_oos=int(len(elo_oos)),
        checkpoints=checkpoint_tuple,
        checkpoint_coverage=checkpoint_coverage,
        checkpoint_curve=checkpoint_curve,
        fold_deltas=fold_deltas,
        accumulation=accumulation,
        bootstrap=bootstrap,
        ban_ablation=ban_ablation,
        signature_diagnostics=signature_diagnostics,
        side_pick_balance=side_pick_balance,
        feature_dims=feature_dims,
        selected_C=selected_C,
        integrity=integrity,
        classification=classification,
        classification_rationale=rationale,
        accumulation_pattern=pattern,
        frozen_components=frozen,
        terminal_notes=terminal_notes,
    )


def slice27_report_to_jsonable(report: Slice27BenchmarkReport) -> dict[str, Any]:
    """JSON-serializable Slice 27 report."""
    return {
        "slice": 27,
        "title": "incremental draft-value benchmark",
        "diagnostic_only": SLICE27_DIAGNOSTIC_ONLY,
        "development_end": report.development_end.isoformat(),
        "holdout_policy": report.holdout_policy,
        "n_development_matches": report.n_development_matches,
        "n_holdout_excluded": report.n_holdout_excluded,
        "n_oos": report.n_oos,
        "checkpoints": list(report.checkpoints),
        "checkpoint_coverage": report.checkpoint_coverage.to_dict(orient="records"),
        "checkpoint_curve": report.checkpoint_curve.to_dict(orient="records"),
        "fold_deltas": report.fold_deltas.to_dict(orient="records"),
        "accumulation": report.accumulation.to_dict(orient="records"),
        "bootstrap": {
            key: {k: _jsonable_value(v) for k, v in value.items()}
            for key, value in report.bootstrap.items()
        },
        "ban_ablation": report.ban_ablation.to_dict(orient="records"),
        "signature_diagnostics": report.signature_diagnostics.to_dict(
            orient="records"
        ),
        "side_pick_balance": report.side_pick_balance.to_dict(orient="records"),
        "feature_dims": report.feature_dims.to_dict(orient="records"),
        "selected_C": report.selected_C.to_dict(orient="records"),
        "integrity": {
            key: _jsonable_value(value) for key, value in report.integrity.items()
        },
        "classification": report.classification,
        "classification_rationale": report.classification_rationale,
        "accumulation_pattern": report.accumulation_pattern,
        "frozen_components": list(report.frozen_components),
        "terminal_notes": report.terminal_notes,
        "reference_spec": SLICE27_REFERENCE_SPEC_NAME,
        "candidate_spec": SLICE27_CANDIDATE_SPEC_NAME,
        "picks_plus_bans_spec": SLICE27_PICKS_PLUS_BANS_SPEC_NAME,
        "feature_columns_length": len(FEATURE_COLUMNS),
    }
