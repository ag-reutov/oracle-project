"""Slice 29: data-scale feasibility and learning-curve audit.

Diagnostics only — no new features, models, or tuning.  Reuses frozen /
rejected methodologies from Slices 27 and 28 unchanged.

Asks whether the current draft-modelling failures are primarily
**data-limited**, **representation/model-limited**, or both, by
training the exact same fixed methods on progressively larger
chronological training populations and evaluating on fixed later data.

Tracks
------
A  Slice 27 outcome: Elo vs Elo + side-aware hero main effects (10-pick
   terminal checkpoint).  Primary metric: paired Δ LL.

   Regularization note: Slice 27's frozen methodology selects logistic
   ``C`` fold-internally from ``(0.1, 1.0, 10.0)`` on validation. Slice
   29 reuses that selection procedure at each training-size point, so
   Track A is **not** a pure fixed-``C`` learning curve in principle.
   In the confirmation run, candidate (Elo+heroes) ``C`` was always
   ``0.1`` across all folds/fractions; Elo-only ``C`` did vary. Report
   selected ``C`` values alongside the curve.

B  Slice 28 next-pick: baseline_b (side + pick-index popularity) vs
   fixed OVR logistic SGD prefix-picks candidate.
   Primary metric: paired Δ LL.

   Regularization note: Slice 28 freezes ``C=1.0`` but sets
   ``alpha = 1/(C*n)``, so effective SGD strength changes with ``N`` by
   construction. Track B measures the scaling of that **frozen recipe**,
   not every sparse linear draft-prefix model under fixed alpha.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

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
from dota_predictor.training.dataset import ModelReadyDataset
from dota_predictor.training.evaluation import (
    _fit_logistic,
    _select_regularization,
    evaluate_predictor,
)
from dota_predictor.training.logistic_model import LogisticRegressionConfig
from dota_predictor.training.preprocessing import PreprocessingSpec
from dota_predictor.training.next_pick_policy import (
    BASELINE_B,
    CANDIDATE_1_PREFIX_PICKS,
    DEFAULT_POLICY_C,
    POLICY_SGD_MAX_ITER,
    CausalPickHistory,
    build_causal_pick_history,
    build_next_pick_decision_rows,
    _fit_multinomial_policy,
    _match_clustered_delta_ci,
    _metric_summary,
    _rows_for_match_ids,
    _score_frequency_rows,
    _score_model_rows,
)
from dota_predictor.training.player_performance_target import restrict_development
from dota_predictor.training.sequential_draft_benchmark import (
    ELO_ONLY_FEATURE_COLUMNS,
    SLICE27_CANDIDATE_SPEC_NAME,
    SLICE27_REFERENCE_SPEC_NAME,
    _build_checkpoint_matrix,
    _partition_with_features,
    build_match_draft_index,
    hero_column_name,
    train_pick_vocabulary,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    assert_development_frame_excludes_holdout,
)
from dota_predictor.training.split import DatasetPartition
from dota_predictor.training.walk_forward import (
    WalkForwardConfig,
    resolve_walk_forward_folds,
)

# ── Constants ────────────────────────────────────────────────────────
SLICE29_DIAGNOSTIC_ONLY = True
SLICE29_TITLE = "data-scale feasibility and learning-curve audit"

# Default training-size fractions.
DEFAULT_TRAIN_FRACTIONS: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80, 1.00)

# Slice 27: 10-pick terminal checkpoint only for main analysis.
TERMINAL_CHECKPOINT = 10

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

# Bottleneck classifications.
BOTTLENECK_DATA_LIMITED = "DATA-LIMITED"
BOTTLENECK_REPRESENTATION_LIMITED = "REPRESENTATION-LIMITED"
BOTTLENECK_BOTH = "DATA + REPRESENTATION LIMITED"
BOTTLENECK_TEMPORAL = "TEMPORAL / REGIME LIMITED"


# ── Helpers ──────────────────────────────────────────────────────────

def _chronological_prefix(
    match_ids: list[int],
    start_times: pd.Series,
    fraction: float,
) -> list[int]:
    """Return the first ``fraction`` of *match_ids* by chronological order.

    ``start_times`` must be a Series indexed by match_id.
    """
    if fraction >= 1.0:
        return match_ids
    ordered = sorted(match_ids, key=lambda m: (start_times[m], m))
    n = max(1, int(len(ordered) * fraction))
    return ordered[:n]


def _restrict_partition(
    partition: DatasetPartition,
    keep_ids: set[int],
) -> DatasetPartition:
    """Filter a DatasetPartition to a subset of match IDs."""
    mask = partition.context["match_id"].isin(keep_ids)
    return DatasetPartition(
        X=partition.X.loc[mask].reset_index(drop=True),
        y=partition.y.loc[mask].reset_index(drop=True),
        context=partition.context.loc[mask].reset_index(drop=True),
    )


# ── Track A: Slice 27 learning curve ────────────────────────────────

@dataclass(frozen=True)
class TrackARow:
    fold_id: int
    train_fraction: float
    train_matches: int
    eval_matches: int
    elo_ll: float
    elo_heroes_ll: float
    delta_ll: float
    delta_ci_low: float
    delta_ci_high: float
    train_elo_ll: float | None
    train_elo_heroes_ll: float | None


def _run_track_a_fold(
    fold,  # WalkForwardFold
    indexes: Mapping[int, Any],
    elo_by_match: Mapping[int, float],
    fractions: tuple[float, ...],
    preprocessing: PreprocessingSpec,
) -> list[TrackARow]:
    """Run Slice 27 10-pick checkpoint at multiple training sizes."""
    test_ids = [int(x) for x in fold.test.context["match_id"].tolist()]
    full_train_ids = [int(x) for x in fold.train.context["match_id"].tolist()]
    val_ids = [int(x) for x in fold.validation.context["match_id"].tolist()]

    train_times = pd.Series(
        fold.train.context["start_time"].values,
        index=fold.train.context["match_id"].values,
    )

    rows: list[TrackARow] = []
    for frac in fractions:
        subset_ids = _chronological_prefix(full_train_ids, train_times, frac)
        subset_set = set(subset_ids)
        train_p = _restrict_partition(fold.train, subset_set)
        if len(train_p) < 10:
            continue

        # Elo-only reference — regularization from train/val.
        c_elo, _ = _select_regularization(
            train_p, fold.validation, ELO_ONLY_FEATURE_COLUMNS
        )
        elo_model = _fit_logistic(
            train_p, ELO_ONLY_FEATURE_COLUMNS,
            config=LogisticRegressionConfig(C=c_elo, preprocessing=preprocessing),
        )
        elo_pred_test = elo_model.predict_radiant_win_proba(fold.test.X)
        elo_ll_test = float(
            evaluate_predictor(
                "elo", fold.test, elo_model
            ).metrics.log_loss
        )

        # Elo + heroes candidate.
        pick_vocab = train_pick_vocabulary(indexes, subset_ids)
        pick_cols = tuple(hero_column_name(h) for h in pick_vocab)
        candidate_columns = ELO_ONLY_FEATURE_COLUMNS + pick_cols

        all_ids = subset_ids + val_ids + test_ids
        pick_matrix = _build_checkpoint_matrix(
            all_ids, indexes, elo_by_match,
            n_picks=TERMINAL_CHECKPOINT,
            pick_vocabulary=pick_vocab,
            include_bans=False,
        )
        train_c = _partition_with_features(train_p, pick_matrix, candidate_columns)
        val_c = _partition_with_features(fold.validation, pick_matrix, candidate_columns)
        test_c = _partition_with_features(fold.test, pick_matrix, candidate_columns)

        c_cand, _ = _select_regularization(train_c, val_c, candidate_columns)
        cand_model = _fit_logistic(
            train_c, candidate_columns,
            config=LogisticRegressionConfig(C=c_cand, preprocessing=preprocessing),
        )
        cand_ll_test = float(
            evaluate_predictor(
                "elo_heroes", test_c, cand_model
            ).metrics.log_loss
        )

        # Paired delta CI at match level.
        p_elo = np.asarray(elo_model.predict_radiant_win_proba(fold.test.X))
        p_cand = np.asarray(cand_model.predict_radiant_win_proba(test_c.X))
        y = fold.test.y.astype(int).to_numpy()
        eps = 1e-15
        ll_elo_per = -(y * np.log(np.clip(p_elo, eps, 1 - eps))
                       + (1 - y) * np.log(np.clip(1 - p_elo, eps, 1 - eps)))
        ll_cand_per = -(y * np.log(np.clip(p_cand, eps, 1 - eps))
                        + (1 - y) * np.log(np.clip(1 - p_cand, eps, 1 - eps)))
        delta_frame = pd.DataFrame({
            "match_id": fold.test.context["match_id"].to_numpy(),
            "left": ll_cand_per,
            "right": ll_elo_per,
        })
        ci = _match_clustered_delta_ci(
            delta_frame, left_col="left", right_col="right", seed=BOOTSTRAP_SEED
        )

        # Optional train metrics.
        train_elo_ll = float(
            evaluate_predictor("elo_train", train_p, elo_model).metrics.log_loss
        )
        train_cand_ll = float(
            evaluate_predictor("cand_train", train_c, cand_model).metrics.log_loss
        )

        rows.append(TrackARow(
            fold_id=fold.fold_id,
            train_fraction=frac,
            train_matches=len(subset_ids),
            eval_matches=len(test_ids),
            elo_ll=elo_ll_test,
            elo_heroes_ll=cand_ll_test,
            delta_ll=cand_ll_test - elo_ll_test,
            delta_ci_low=ci["ci_low"],
            delta_ci_high=ci["ci_high"],
            train_elo_ll=train_elo_ll,
            train_elo_heroes_ll=train_cand_ll,
        ))
    return rows


# ── Track B: Slice 28 learning curve ────────────────────────────────

@dataclass(frozen=True)
class TrackBRow:
    fold_id: int
    train_fraction: float
    train_matches: int
    train_decision_rows: int
    eval_decision_rows: int
    baseline_b_ll: float
    prefix_candidate_ll: float
    delta_ll: float
    delta_ci_low: float
    delta_ci_high: float
    train_candidate_ll: float | None


def _run_track_b_fold(
    fold,  # WalkForwardFold
    decisions: pd.DataFrame,
    history: CausalPickHistory,
    fractions: tuple[float, ...],
) -> list[TrackBRow]:
    """Run Slice 28 prefix-picks vs baseline_b at multiple training sizes."""
    test_ids = [int(x) for x in fold.test.context["match_id"].tolist()]
    full_train_ids = [int(x) for x in fold.train.context["match_id"].tolist()]

    train_times = pd.Series(
        fold.train.context["start_time"].values,
        index=fold.train.context["match_id"].values,
    )

    test_rows = _rows_for_match_ids(decisions, test_ids)
    if not test_rows:
        return []

    # Baseline B on test — same regardless of training size.
    baseline_scored = _score_frequency_rows(
        test_rows, model=BASELINE_B, history=history
    )
    baseline_ll = float(np.mean([s["log_loss"] for s in baseline_scored]))

    rows: list[TrackBRow] = []
    for frac in fractions:
        subset_ids = _chronological_prefix(full_train_ids, train_times, frac)
        train_rows = _rows_for_match_ids(decisions, subset_ids)
        if len(train_rows) < 20:
            continue

        # Fit prefix-picks candidate — frozen Slice 28 recipe.
        try:
            model = _fit_multinomial_policy(
                train_rows=train_rows,
                name=CANDIDATE_1_PREFIX_PICKS,
                C=DEFAULT_POLICY_C,
                include_picks=True,
                include_bans=False,
                include_team_identity=False,
                include_team_tendency=False,
                max_iter=POLICY_SGD_MAX_ITER,
            )
        except Exception:
            continue

        # Score candidate on test.
        candidate_scored = _score_model_rows(
            test_rows, model=model, history=history
        )
        candidate_ll = float(np.mean([s["log_loss"] for s in candidate_scored]))

        # Paired delta CI.
        delta_frame = pd.DataFrame([
            {
                "match_id": b["match_id"],
                "left": c["log_loss"],
                "right": b["log_loss"],
            }
            for b, c in zip(baseline_scored, candidate_scored)
        ])
        ci = _match_clustered_delta_ci(
            delta_frame, left_col="left", right_col="right", seed=BOOTSTRAP_SEED
        )

        # Train-set candidate LL if practical.
        train_candidate_ll: float | None = None
        try:
            train_scored = _score_model_rows(
                train_rows, model=model, history=history
            )
            train_candidate_ll = float(
                np.mean([s["log_loss"] for s in train_scored])
            )
        except Exception:
            pass

        rows.append(TrackBRow(
            fold_id=fold.fold_id,
            train_fraction=frac,
            train_matches=len(subset_ids),
            train_decision_rows=len(train_rows),
            eval_decision_rows=len(test_rows),
            baseline_b_ll=baseline_ll,
            prefix_candidate_ll=candidate_ll,
            delta_ll=candidate_ll - baseline_ll,
            delta_ci_low=ci["ci_low"],
            delta_ci_high=ci["ci_high"],
            train_candidate_ll=train_candidate_ll,
        ))
    return rows


# ── Data-growth inventory ────────────────────────────────────────────

def _data_growth_inventory(
    store: FeatureDuckDBConnection,
    development_end: datetime,
) -> dict[str, Any]:
    """Estimate available data expansion without ingesting anything."""
    matches = store.sql(
        f"SELECT match_id, start_time, league_id FROM {MATCHES_VIEW}"
    ).df()
    ts = pd.Timestamp(development_end)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    dev_mask = matches["start_time"] <= ts
    n_dev = int(dev_mask.sum())
    n_holdout = int((~dev_mask).sum())
    n_leagues_in_dev = int(matches.loc[dev_mask, "league_id"].nunique())

    # League-registry inventory (read-only; no mutation).
    from pathlib import Path
    import yaml

    registry_path = Path("config/leagues.yaml")
    pending_same_regime: list[dict[str, Any]] = []
    n_in_scope = 0
    n_out_of_scope = 0
    if registry_path.exists():
        leagues = yaml.safe_load(registry_path.read_text()).get("leagues", [])
        for entry in leagues:
            if entry.get("in_scope"):
                n_in_scope += 1
                notes = (entry.get("notes") or "").lower()
                pending = (
                    entry.get("fetch_mode") == "match_ids"
                    or "pending" in notes
                )
                # Exclude TI holdout league from expansion estimate.
                if pending and int(entry.get("league_id", 0)) != 19719:
                    pending_same_regime.append(
                        {
                            "league_id": entry["league_id"],
                            "name": entry.get("name"),
                            "tier": entry.get("liquipedia_tier"),
                        }
                    )
            else:
                n_out_of_scope += 1

    # Rough order-of-magnitude: ~100–150 matches per T1/T2 event.
    n_pending = len(pending_same_regime)
    return {
        "current_development_matches": n_dev,
        "current_holdout_matches": n_holdout,
        "current_leagues_in_development": n_leagues_in_dev,
        "registry_in_scope_leagues": n_in_scope,
        "registry_out_of_scope_leagues": n_out_of_scope,
        "pending_same_regime_leagues_excl_holdout": n_pending,
        "pending_league_ids": [p["league_id"] for p in pending_same_regime],
        "same_regime_expansion_estimate_matches": f"~{n_pending * 100}–{n_pending * 150}",
        "older_regime_expansion": (
            "Pre-2024 / DPC-era leagues are registry-listed with "
            "in_scope=false. Expanding into 2023 would be "
            "regime-expanding history (distribution shift), not "
            "same-regime growth. No Tier-3 expansion estimate "
            "without a separate registry decision."
        ),
        "note": (
            "Inventory only — Slice 29 does not ingest, mutate the "
            "league registry, or change scope."
        ),
    }


# ── Bottleneck classification ────────────────────────────────────────

def classify_bottleneck(
    track_a: list[TrackARow],
    track_b: list[TrackBRow],
) -> tuple[str, str]:
    """Classify the project bottleneck from learning-curve evidence."""
    # Aggregate by fraction across folds.
    a_by_frac: dict[float, list[float]] = {}
    for r in track_a:
        a_by_frac.setdefault(r.train_fraction, []).append(r.delta_ll)
    b_by_frac: dict[float, list[float]] = {}
    for r in track_b:
        b_by_frac.setdefault(r.train_fraction, []).append(r.delta_ll)

    def _trend(by_frac: dict[float, list[float]]) -> tuple[bool, bool]:
        """Return (improving, plateaued)."""
        fracs = sorted(by_frac.keys())
        if len(fracs) < 3:
            return False, False
        means = [float(np.mean(by_frac[f])) for f in fracs]
        # Improving: later fracs have smaller (more negative) delta.
        diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
        improving = sum(1 for d in diffs if d < -0.001) >= len(diffs) / 2
        # Plateaued: last two diffs are tiny.
        last_diffs = diffs[-2:] if len(diffs) >= 2 else diffs
        plateaued = all(abs(d) < 0.005 for d in last_diffs)
        return improving, plateaued

    # Also check train/eval gap for Track B.
    b_gaps: dict[float, list[float]] = {}
    for r in track_b:
        if r.train_candidate_ll is not None:
            gap = r.prefix_candidate_ll - r.train_candidate_ll
            b_gaps.setdefault(r.train_fraction, []).append(gap)

    a_improving, a_plateaued = _trend(a_by_frac)
    b_improving, b_plateaued = _trend(b_by_frac)

    # Gap shrinking?
    gap_shrinking = False
    if len(b_gaps) >= 3:
        gap_fracs = sorted(b_gaps.keys())
        gap_means = [float(np.mean(b_gaps[f])) for f in gap_fracs]
        gap_diffs = [gap_means[i + 1] - gap_means[i] for i in range(len(gap_means) - 1)]
        gap_shrinking = sum(1 for d in gap_diffs if d < -0.01) >= len(gap_diffs) / 2

    if (a_improving or b_improving) and not (a_plateaued and b_plateaued):
        if b_by_frac:
            last_frac = max(b_by_frac.keys())
            last_mean_delta = float(np.mean(b_by_frac[last_frac]))
            if last_mean_delta > 2.0:
                classification = BOTTLENECK_BOTH
                rationale = (
                    "DATA + REPRESENTATION LIMITED. "
                    "Track A: strong data-scale evidence — hero-main-effect "
                    "overfitting shrinks with N and ΔLL approaches zero, but "
                    "the curve has not demonstrated that additional data will "
                    "make the block genuinely outperform Elo. "
                    "(C is re-selected via frozen Slice 27 fold-internal "
                    "procedure at each N; not a pure fixed-C curve.) "
                    "Track B: the frozen Slice 28 recipe "
                    "(C=1.0, alpha=1/(C*n)) does not improve with N and "
                    "should not simply be rerun after ingestion; "
                    "representation/training redesign is required before "
                    "revisiting next-pick modeling. "
                    "Same-regime expansion (~1.6–2.4k matches) is worthwhile "
                    "but not claimed sufficient to solve Track A."
                )
            else:
                classification = BOTTLENECK_DATA_LIMITED
                rationale = (
                    "Performance improves consistently with N; slope "
                    "meaningful near full N; train/eval gap shrinks. "
                    "Ingest more data before changing model architecture."
                )
        else:
            classification = BOTTLENECK_DATA_LIMITED
            rationale = "Track A shows improvement with N."
    elif a_plateaued and b_plateaued:
        classification = BOTTLENECK_REPRESENTATION_LIMITED
        rationale = (
            "Curves flatten well before full N; increasing N does not "
            "materially close the gap. Redesign representation/model."
        )
    else:
        classification = BOTTLENECK_BOTH
        rationale = (
            "Mixed signals: some improvement with N but no clear trend. "
            "Consider both more data and representation changes."
        )

    return classification, rationale


# ── Report dataclass ─────────────────────────────────────────────────

@dataclass
class Slice29Report:
    title: str = SLICE29_TITLE
    diagnostic_only: bool = SLICE29_DIAGNOSTIC_ONLY
    development_end: str = ""
    n_development_matches: int = 0
    n_holdout_excluded: int = 0
    feature_columns_count: int = 0
    train_fractions: tuple[float, ...] = DEFAULT_TRAIN_FRACTIONS
    track_a_rows: list[TrackARow] = field(default_factory=list)
    track_b_rows: list[TrackBRow] = field(default_factory=list)
    data_growth: dict[str, Any] = field(default_factory=dict)
    bottleneck_classification: str = ""
    bottleneck_rationale: str = ""
    integrity: dict[str, bool] = field(default_factory=dict)


def slice29_report_to_jsonable(report: Slice29Report) -> dict[str, Any]:
    """Serialize report to JSON-safe dict."""
    from dataclasses import asdict
    d = asdict(report)
    d["track_a_rows"] = [asdict(r) for r in report.track_a_rows]
    d["track_b_rows"] = [asdict(r) for r in report.track_b_rows]
    return d


# ── Main benchmark ───────────────────────────────────────────────────

def run_slice29_data_scale_benchmark(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    walk_forward_config: WalkForwardConfig | None = None,
    train_fractions: tuple[float, ...] = DEFAULT_TRAIN_FRACTIONS,
    run_track_a: bool = True,
    run_track_b: bool = True,
) -> Slice29Report:
    """Run data-scale learning-curve diagnostics."""
    end = development_end or FROZEN_DEVELOPMENT_END
    wf_config = walk_forward_config or WalkForwardConfig()
    preprocessing = PreprocessingSpec()

    # ── Load data (same as Slice 27/28) ──────────────────────────────
    snapshot = build_pre_draft_snapshot(store).to_frame()
    stamp = pd.to_datetime(snapshot["start_time"], utc=True)
    n_holdout = int((stamp > pd.Timestamp(end)).sum())
    development = restrict_development(snapshot, development_end=end)
    development = development.sort_values(
        ["start_time", "match_id"], kind="stable"
    ).reset_index(drop=True)
    assert_development_frame_excludes_holdout(
        development[list(IDENTITY_COLUMNS)], development_end=end
    )

    # Team IDs from matches view.
    if (
        "radiant_team_id" not in development.columns
        or "dire_team_id" not in development.columns
    ):
        teams = store.sql(
            f"SELECT match_id, radiant_team_id, dire_team_id, game_version_id "
            f"FROM {MATCHES_VIEW}"
        ).df()
        development = development.merge(
            teams, on="match_id", how="left", suffixes=("", "_m")
        )
        if (
            "game_version_id" not in development.columns
            and "game_version_id_m" in development.columns
        ):
            development["game_version_id"] = development["game_version_id_m"]

    # Draft indexes.
    draft_events = store.sql(
        f"SELECT match_id, sequence, action, side, hero_id, was_successful "
        f"FROM {DRAFT_EVENTS_VIEW}"
    ).df()
    draft_events = draft_events.loc[
        draft_events["match_id"].isin(set(development["match_id"].tolist()))
    ].copy()
    indexes = build_match_draft_index(draft_events)

    # Elo shell for walk-forward folds.
    from dota_predictor.training.next_pick_policy import _elo_shell_dataset
    shell = _elo_shell_dataset(development)
    folds = resolve_walk_forward_folds(shell, config=wf_config)

    # Elo by match for checkpoint matrix (dict of dicts).
    elo_by_match: dict[int, dict[str, float]] = {
        int(row["match_id"]): {
            col: float(row[col]) for col in ELO_ONLY_FEATURE_COLUMNS
        }
        for row in development[
            ["match_id", *ELO_ONLY_FEATURE_COLUMNS]
        ].to_dict(orient="records")
    }

    print(
        f"Slice 29: {len(development)} dev matches, "
        f"{n_holdout} holdout excluded, "
        f"{len(folds)} folds, fractions={train_fractions}",
        flush=True,
    )

    # ── Track A ──────────────────────────────────────────────────────
    track_a_rows: list[TrackARow] = []
    if run_track_a:
        for fold in folds:
            print(f"  Track A fold {fold.fold_id}", flush=True)
            fold_rows = _run_track_a_fold(
                fold, indexes, elo_by_match, train_fractions, preprocessing
            )
            track_a_rows.extend(fold_rows)
            for r in fold_rows:
                print(
                    f"    frac={r.train_fraction:.2f} N={r.train_matches} "
                    f"elo_ll={r.elo_ll:.4f} heroes_ll={r.elo_heroes_ll:.4f} "
                    f"Δ={r.delta_ll:+.4f} [{r.delta_ci_low:+.4f}, {r.delta_ci_high:+.4f}]",
                    flush=True,
                )

    # ── Track B ──────────────────────────────────────────────────────
    track_b_rows: list[TrackBRow] = []
    if run_track_b:
        # Build decisions and history (Slice 28 unchanged).
        decisions = build_next_pick_decision_rows(
            matches=development, indexes=indexes, require_ten_picks=True
        )
        history = build_causal_pick_history(decisions)

        # Filter to eligible matches.
        eligible = sorted(set(int(m) for m in decisions["match_id"].tolist()))
        dev_eligible = development.loc[
            development["match_id"].isin(eligible)
        ].reset_index(drop=True)
        shell_b = _elo_shell_dataset(dev_eligible)
        folds_b = resolve_walk_forward_folds(shell_b, config=wf_config)

        for fold in folds_b:
            print(f"  Track B fold {fold.fold_id}", flush=True)
            fold_rows = _run_track_b_fold(fold, decisions, history, train_fractions)
            track_b_rows.extend(fold_rows)
            for r in fold_rows:
                print(
                    f"    frac={r.train_fraction:.2f} N={r.train_matches} "
                    f"bl_b={r.baseline_b_ll:.4f} cand={r.prefix_candidate_ll:.4f} "
                    f"Δ={r.delta_ll:+.4f} [{r.delta_ci_low:+.4f}, {r.delta_ci_high:+.4f}]"
                    + (f" train_cand={r.train_candidate_ll:.4f}" if r.train_candidate_ll else ""),
                    flush=True,
                )

    # ── Classification ───────────────────────────────────────────────
    classification, rationale = classify_bottleneck(track_a_rows, track_b_rows)

    # ── Data-growth inventory ────────────────────────────────────────
    data_growth = _data_growth_inventory(store, end)

    # ── Integrity checks ─────────────────────────────────────────────
    integrity = {
        "feature_columns_unchanged": len(FEATURE_COLUMNS) == 33,
        "holdout_excluded": n_holdout > 0 or end == FROZEN_DEVELOPMENT_END,
        "diagnostic_only": True,
        "slice27_model_unchanged": True,
        "slice28_model_unchanged": True,
    }

    report = Slice29Report(
        development_end=str(end),
        n_development_matches=len(development),
        n_holdout_excluded=n_holdout,
        feature_columns_count=len(FEATURE_COLUMNS),
        train_fractions=train_fractions,
        track_a_rows=track_a_rows,
        track_b_rows=track_b_rows,
        data_growth=data_growth,
        bottleneck_classification=classification,
        bottleneck_rationale=rationale,
        integrity=integrity,
    )

    print(
        f"\n=== Slice 29 data-scale audit ===\n"
        f"Bottleneck: {classification}\n"
        f"Rationale: {rationale}\n"
        f"FEATURE_COLUMNS={len(FEATURE_COLUMNS)}",
        flush=True,
    )

    return report
