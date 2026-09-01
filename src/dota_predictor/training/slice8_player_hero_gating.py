"""Slice 8: leakage-safe Career Player × Hero gating (exploratory).

Evaluation only. Does not change Elo, walk-forward fold boundaries,
canonical data, Slices 0–7 specs, or production ``FEATURE_COLUMNS``.

The hypothesis is that Career Player × Hero should be trusted
conditionally: keep its contribution when evidence looks usable, and
attenuate it when career familiarity may be stale relative to
patch-local or role context.

This slice is **exploratory**. The 2024–2026 OOS window already informed
the Slice 7 hypothesis, so a later untouched temporal holdout is
required before treating any gating rule as confirmed.

Gating parameters are either threshold-free interactions or estimated
from the current fold's TRAIN (quantiles / scaling) and VAL (gate
choice). TEST never influences thresholds, imputation, ``C``, or which
gate is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import FeatureDuckDBConnection
from dota_predictor.features.player_hero_meta_comparison import MATCH_ID_COLUMN
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.dataset import ModelReadyDataset, TrainingDatasetError
from dota_predictor.training.evaluation import (
    _fit_logistic,
    _select_regularization,
    evaluate_predictor,
)
from dota_predictor.training.feature_sets import (
    ELO_PLUS_PLAYER_HERO_COLUMNS,
    PLAYER_HERO_COMPARISON_COLUMNS,
    SLICE8_CAREER_SPEC_NAME,
    SLICE8_CONTEXT_COLUMNS,
    SLICE8_DIRE_MEAN_CAREER_GAMES,
    SLICE8_FULL_CONTEXT_COLUMNS,
    SLICE8_GATE_SPEC_NAME,
    SLICE8_INTERACTION_COLUMNS,
    SLICE8_LOG1P_DIRE_MEAN_CAREER_GAMES,
    SLICE8_LOG1P_MATCH_MEAN_CAREER_GAMES,
    SLICE8_LOG1P_MATCH_MEAN_SAME_VERSION_GAMES,
    SLICE8_LOG1P_MIN_CAREER_GAMES,
    SLICE8_LOG1P_RADIANT_MEAN_CAREER_GAMES,
    SLICE8_MATCH_MEAN_CAREER_GAMES,
    SLICE8_MATCH_MEAN_HERO_META_SHARE,
    SLICE8_MATCH_MEAN_PLAYER_SHARE,
    SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY,
    SLICE8_MATCH_MEAN_SAME_VERSION_GAMES,
    SLICE8_MATCH_ZERO_SAME_VERSION_PLAYERS,
    SLICE8_META_PLAYER_HERO_SPECS,
    SLICE8_MIN_CAREER_GAMES,
    SLICE8_RADIANT_MEAN_CAREER_GAMES,
    SLICE8_STATIC_SPECS,
    BlockAblationSpec,
    slice8_interaction_column,
)
from dota_predictor.training.logistic_model import (
    LogisticRegressionConfig,
    LogisticRegressionPredictor,
    standardized_coefficients,
)
from dota_predictor.training.metrics import evaluate_probabilities, per_sample_log_loss
from dota_predictor.training.preprocessing import PreprocessingSpec
from dota_predictor.training.slice7_meta_player_hero import (
    PATCH_MATURITY_BIN_ORDER,
    Slice7Assembly,
    build_slice7_model_ready_dataset,
)
from dota_predictor.training.split import DatasetPartition
from dota_predictor.training.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG,
    ELO_BLOCK_SPEC_NAME,
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardReport,
    _pooled_metrics,
    _version_breakdown,
    resolve_walk_forward_folds,
)

__all__ = [
    "COMPATIBILITY_BIN_ORDER",
    "CROSS_CELL_ORDER",
    "EVIDENCE_BIN_ORDER",
    "GATE_CANDIDATE_ORDER",
    "SLICE8_EVIDENCE_SPEC_NAME",
    "SLICE8_FULL_SPEC_NAME",
    "SLICE8_PATCH_SPEC_NAME",
    "SLICE8_ROLE_SPEC_NAME",
    "CareerGateCandidate",
    "GateKind",
    "Slice8Assembly",
    "Slice8BenchmarkReport",
    "add_slice8_interaction_columns",
    "apply_career_gate",
    "assert_slice7_slice8_identity",
    "assign_train_tertile_bin",
    "build_slice8_context_frame",
    "build_slice8_model_ready_dataset",
    "career_gate_weights",
    "gates_from_train",
    "run_slice8_player_hero_gating_benchmark",
    "select_career_gate",
    "train_tertile_edges",
]

SLICE8_EVIDENCE_SPEC_NAME = "logistic_elo_plus_career_evidence_interaction"
SLICE8_ROLE_SPEC_NAME = "logistic_elo_plus_career_role_interaction"
SLICE8_PATCH_SPEC_NAME = "logistic_elo_plus_career_patch_interaction"
SLICE8_FULL_SPEC_NAME = "logistic_elo_plus_career_full_gating"

EVIDENCE_BIN_ORDER: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
COMPATIBILITY_BIN_ORDER: tuple[str, ...] = ("NULL", "LOW", "MEDIUM", "HIGH")
CROSS_CELL_ORDER: tuple[str, ...] = tuple(
    f"{career} × {compat}"
    for career in EVIDENCE_BIN_ORDER
    for compat in ("LOW", "MEDIUM", "HIGH")
)
GATE_CANDIDATE_ORDER: tuple[str, ...] = (
    "identity",
    "compat_scale",
    "compat_below_q25",
    "compat_below_q50",
    "career_above_q50",
    "career_above_q75",
    "high_career_q50_and_low_compat_q50",
    "high_career_q75_and_low_compat_q25",
)

_LOG1P_PAIRS: tuple[tuple[str, str], ...] = (
    (SLICE8_RADIANT_MEAN_CAREER_GAMES, SLICE8_LOG1P_RADIANT_MEAN_CAREER_GAMES),
    (SLICE8_DIRE_MEAN_CAREER_GAMES, SLICE8_LOG1P_DIRE_MEAN_CAREER_GAMES),
    (SLICE8_MATCH_MEAN_CAREER_GAMES, SLICE8_LOG1P_MATCH_MEAN_CAREER_GAMES),
    (SLICE8_MIN_CAREER_GAMES, SLICE8_LOG1P_MIN_CAREER_GAMES),
    (
        SLICE8_MATCH_MEAN_SAME_VERSION_GAMES,
        SLICE8_LOG1P_MATCH_MEAN_SAME_VERSION_GAMES,
    ),
)
_INTERACTION_CONTEXT_COLUMNS: tuple[str, ...] = (
    SLICE8_LOG1P_MATCH_MEAN_CAREER_GAMES,
    SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY,
    SLICE8_LOG1P_MATCH_MEAN_SAME_VERSION_GAMES,
)
_COEFFICIENT_FEATURE_NAMES: frozenset[str] = frozenset(
    PLAYER_HERO_COMPARISON_COLUMNS
    + SLICE8_FULL_CONTEXT_COLUMNS
    + SLICE8_INTERACTION_COLUMNS
)


class GateKind(str, Enum):
    IDENTITY = "identity"
    COMPAT_SCALE = "compat_scale"
    COMPAT_BELOW = "compat_below"
    CAREER_ABOVE = "career_above"
    HIGH_CAREER_LOW_COMPAT = "high_career_low_compat"


@dataclass(frozen=True)
class CareerGateCandidate:
    """One fold-internal Career attenuation rule.

    Thresholds are feature-space values copied from TRAIN quantiles, not
    probabilities and not OOS-derived cutoffs.
    """

    name: str
    kind: GateKind
    career_threshold: float | None = None
    compatibility_threshold: float | None = None
    compat_scale_lo: float | None = None
    compat_scale_hi: float | None = None


def train_tertile_edges(values: pd.Series) -> tuple[float, float]:
    """``1/3`` and ``2/3`` quantiles from the provided series only."""
    observed = values.astype("float64")
    if observed.dropna().empty:
        return float("nan"), float("nan")
    return float(observed.quantile(1.0 / 3.0)), float(observed.quantile(2.0 / 3.0))


def assign_train_tertile_bin(
    value: float | None, *, q_low: float, q_high: float
) -> str:
    """Map a score onto TRAIN tertiles. Does not use outcomes."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NULL"
    if np.isnan(q_low) or np.isnan(q_high):
        return "NULL"
    score = float(value)
    if score <= q_low:
        return "LOW"
    if score >= q_high:
        return "HIGH"
    return "MEDIUM"


def build_slice8_context_frame(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Match-level gating context from Slice 7 diagnostics.

    Counts use the existing career / same-version player-grain means.
    Rates keep NULL when no player on the match has evidence. No
    outcomes enter these columns.
    """
    required = {
        MATCH_ID_COLUMN,
        "radiant_mean_prior_games_on_hero",
        "dire_mean_prior_games_on_hero",
        "mean_prior_games_on_hero",
        "min_prior_games_on_hero",
        "mean_same_version_matches",
        "n_zero_same_version_players",
        "mean_role_compatibility",
        "mean_player_share_at_expected_position",
        "mean_hero_meta_share_at_expected_position",
    }
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise TrainingDatasetError(
            "Slice 7 diagnostics are missing Slice 8 context columns: "
            f"{missing}"
        )
    frame = pd.DataFrame(
        {
            MATCH_ID_COLUMN: diagnostics[MATCH_ID_COLUMN],
            SLICE8_RADIANT_MEAN_CAREER_GAMES: diagnostics[
                "radiant_mean_prior_games_on_hero"
            ],
            SLICE8_DIRE_MEAN_CAREER_GAMES: diagnostics[
                "dire_mean_prior_games_on_hero"
            ],
            SLICE8_MATCH_MEAN_CAREER_GAMES: diagnostics["mean_prior_games_on_hero"],
            SLICE8_MIN_CAREER_GAMES: diagnostics["min_prior_games_on_hero"],
            SLICE8_MATCH_MEAN_SAME_VERSION_GAMES: diagnostics[
                "mean_same_version_matches"
            ],
            SLICE8_MATCH_ZERO_SAME_VERSION_PLAYERS: diagnostics[
                "n_zero_same_version_players"
            ],
            SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY: diagnostics[
                "mean_role_compatibility"
            ],
            SLICE8_MATCH_MEAN_PLAYER_SHARE: diagnostics[
                "mean_player_share_at_expected_position"
            ],
            SLICE8_MATCH_MEAN_HERO_META_SHARE: diagnostics[
                "mean_hero_meta_share_at_expected_position"
            ],
        }
    )
    for source, dest in _LOG1P_PAIRS:
        frame[dest] = np.log1p(pd.to_numeric(frame[source], errors="coerce"))
    return frame


def add_slice8_interaction_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise Career-signal × context products. NULL × value stays NULL."""
    out = X.copy()
    for context in _INTERACTION_CONTEXT_COLUMNS:
        ctx = pd.to_numeric(out[context], errors="coerce")
        for signal in PLAYER_HERO_COMPARISON_COLUMNS:
            out[slice8_interaction_column(signal, context)] = (
                pd.to_numeric(out[signal], errors="coerce") * ctx
            )
    return out


def career_gate_weights(
    frame: pd.DataFrame, candidate: CareerGateCandidate
) -> pd.Series:
    """Attenuation weights in ``[0, 1]``. Missing compatibility does not suppress."""
    career = pd.to_numeric(frame[SLICE8_MATCH_MEAN_CAREER_GAMES], errors="coerce")
    compat = pd.to_numeric(
        frame[SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY], errors="coerce"
    )
    ones = pd.Series(1.0, index=frame.index, dtype="float64")
    if candidate.kind is GateKind.IDENTITY:
        return ones
    if candidate.kind is GateKind.COMPAT_SCALE:
        lo = candidate.compat_scale_lo
        hi = candidate.compat_scale_hi
        if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
            return ones
        if hi <= lo:
            return ones
        scaled = (compat - lo) / (hi - lo)
        return scaled.clip(lower=0.0, upper=1.0).fillna(1.0)
    if candidate.kind is GateKind.COMPAT_BELOW:
        if candidate.compatibility_threshold is None:
            return ones
        below = compat < candidate.compatibility_threshold
        return ones.mask(below.fillna(False), 0.0)
    if candidate.kind is GateKind.CAREER_ABOVE:
        if candidate.career_threshold is None:
            return ones
        above = career > candidate.career_threshold
        return ones.mask(above.fillna(False), 0.0)
    if candidate.kind is GateKind.HIGH_CAREER_LOW_COMPAT:
        if (
            candidate.career_threshold is None
            or candidate.compatibility_threshold is None
        ):
            return ones
        high = career >= candidate.career_threshold
        low = compat <= candidate.compatibility_threshold
        suppress = high.fillna(False) & low.fillna(False)
        return ones.mask(suppress, 0.0)
    raise ValueError(f"unknown gate kind: {candidate.kind!r}")


def gates_from_train(train_X: pd.DataFrame) -> tuple[CareerGateCandidate, ...]:
    """Build the predeclared gate grid from TRAIN feature quantiles only."""
    career = pd.to_numeric(train_X[SLICE8_MATCH_MEAN_CAREER_GAMES], errors="coerce")
    compat = pd.to_numeric(
        train_X[SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY], errors="coerce"
    )
    career_q50 = float(career.quantile(0.50))
    career_q75 = float(career.quantile(0.75))
    compat_q25 = float(compat.quantile(0.25))
    compat_q50 = float(compat.quantile(0.50))
    observed_compat = compat.dropna()
    compat_lo = float(observed_compat.min()) if not observed_compat.empty else float("nan")
    compat_hi = float(observed_compat.max()) if not observed_compat.empty else float("nan")
    return (
        CareerGateCandidate(name="identity", kind=GateKind.IDENTITY),
        CareerGateCandidate(
            name="compat_scale",
            kind=GateKind.COMPAT_SCALE,
            compat_scale_lo=compat_lo,
            compat_scale_hi=compat_hi,
        ),
        CareerGateCandidate(
            name="compat_below_q25",
            kind=GateKind.COMPAT_BELOW,
            compatibility_threshold=compat_q25,
        ),
        CareerGateCandidate(
            name="compat_below_q50",
            kind=GateKind.COMPAT_BELOW,
            compatibility_threshold=compat_q50,
        ),
        CareerGateCandidate(
            name="career_above_q50",
            kind=GateKind.CAREER_ABOVE,
            career_threshold=career_q50,
        ),
        CareerGateCandidate(
            name="career_above_q75",
            kind=GateKind.CAREER_ABOVE,
            career_threshold=career_q75,
        ),
        CareerGateCandidate(
            name="high_career_q50_and_low_compat_q50",
            kind=GateKind.HIGH_CAREER_LOW_COMPAT,
            career_threshold=career_q50,
            compatibility_threshold=compat_q50,
        ),
        CareerGateCandidate(
            name="high_career_q75_and_low_compat_q25",
            kind=GateKind.HIGH_CAREER_LOW_COMPAT,
            career_threshold=career_q75,
            compatibility_threshold=compat_q25,
        ),
    )


def apply_career_gate(
    partition: DatasetPartition, candidate: CareerGateCandidate
) -> DatasetPartition:
    """Multiply Career comparison columns by the candidate's weights."""
    weights = career_gate_weights(partition.X, candidate)
    gated = partition.X.copy()
    for column in PLAYER_HERO_COMPARISON_COLUMNS:
        gated[column] = pd.to_numeric(gated[column], errors="coerce") * weights
    return DatasetPartition(
        X=gated, y=partition.y.copy(), context=partition.context.copy()
    )


def select_career_gate(
    train: DatasetPartition,
    validation: DatasetPartition,
    *,
    candidates: tuple[CareerGateCandidate, ...] | None = None,
    feature_columns: tuple[str, ...] = ELO_PLUS_PLAYER_HERO_COLUMNS,
) -> CareerGateCandidate:
    """Pick one gate by VAL log loss. Does not read TEST."""
    resolved = candidates if candidates is not None else gates_from_train(train.X)
    best: CareerGateCandidate | None = None
    best_log_loss = float("inf")
    for candidate in resolved:
        model = _fit_logistic(
            apply_career_gate(train, candidate),
            feature_columns,
            config=LogisticRegressionConfig(C=1.0),
        )
        metrics = evaluate_predictor(
            candidate.name,
            apply_career_gate(validation, candidate),
            model,
        ).metrics
        if metrics.log_loss < best_log_loss:
            best_log_loss = metrics.log_loss
            best = candidate
    if best is None:
        raise TrainingDatasetError("Career gate grid was empty")
    return best


def _align_context(
    context: pd.DataFrame, match_ids: pd.Series
) -> pd.DataFrame:
    return (
        context.set_index(MATCH_ID_COLUMN)
        .reindex(match_ids.to_numpy())
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class Slice8Assembly:
    """Slice 7 rows plus row-wise gating context and interactions."""

    dataset: ModelReadyDataset
    slice7: Slice7Assembly
    match_diagnostics: pd.DataFrame
    n_post_draft_matches: int
    n_oos_identity_matches: int


def build_slice8_model_ready_dataset(
    store: FeatureDuckDBConnection,
) -> Slice8Assembly:
    """Same match rows as Slice 7, with added context and interactions."""
    slice7 = build_slice7_model_ready_dataset(store)
    context = build_slice8_context_frame(slice7.match_diagnostics)
    aligned = _align_context(context, slice7.dataset.context[MATCH_ID_COLUMN])
    X = pd.concat([slice7.dataset.X.reset_index(drop=True), aligned], axis=1)
    X = add_slice8_interaction_columns(X)
    extra = set(SLICE8_CONTEXT_COLUMNS) | set(SLICE8_INTERACTION_COLUMNS)
    overlap = set(FEATURE_COLUMNS) & extra
    if overlap:
        raise TrainingDatasetError(
            "Slice 8 columns must not appear in FEATURE_COLUMNS: "
            f"{sorted(overlap)}"
        )
    missing_extra = extra - set(X.columns)
    if missing_extra:
        raise TrainingDatasetError(
            f"Slice 8 assembly is missing columns: {sorted(missing_extra)}"
        )
    dataset = ModelReadyDataset(
        X=X,
        y=slice7.dataset.y.reset_index(drop=True).copy(),
        context=slice7.dataset.context.reset_index(drop=True).copy(),
        feature_columns=tuple(X.columns),
        target_column=slice7.dataset.target_column,
        identity_columns=slice7.dataset.identity_columns,
    )
    return Slice8Assembly(
        dataset=dataset,
        slice7=slice7,
        match_diagnostics=slice7.match_diagnostics,
        n_post_draft_matches=slice7.n_post_draft_matches,
        n_oos_identity_matches=len(dataset),
    )


def assert_slice7_slice8_identity(
    slice7: ModelReadyDataset | Slice7Assembly,
    slice8: ModelReadyDataset | Slice8Assembly,
    *,
    config: WalkForwardConfig | None = None,
) -> None:
    """Fold boundaries and test match_ids must match Slice 7."""
    left = slice7.dataset if isinstance(slice7, Slice7Assembly) else slice7
    right = slice8.dataset if isinstance(slice8, Slice8Assembly) else slice8
    if list(left.context[MATCH_ID_COLUMN]) != list(right.context[MATCH_ID_COLUMN]):
        raise TrainingDatasetError(
            "Slice 8 match_id order differs from Slice 7"
        )
    pd.testing.assert_series_equal(
        left.context["start_time"].reset_index(drop=True),
        right.context["start_time"].reset_index(drop=True),
        check_names=False,
    )
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    left_folds = resolve_walk_forward_folds(left, config=resolved)
    right_folds = resolve_walk_forward_folds(right, config=resolved)
    if len(left_folds) != len(right_folds):
        raise TrainingDatasetError("Slice 8 fold count differs from Slice 7")
    for fold7, fold8 in zip(left_folds, right_folds, strict=True):
        if fold7.train_end != fold8.train_end:
            raise TrainingDatasetError("Slice 8 train_end differs from Slice 7")
        if fold7.validation_end != fold8.validation_end:
            raise TrainingDatasetError(
                "Slice 8 validation_end differs from Slice 7"
            )
        if fold7.test_end != fold8.test_end:
            raise TrainingDatasetError("Slice 8 test_end differs from Slice 7")
        if list(fold7.test.context[MATCH_ID_COLUMN]) != list(
            fold8.test.context[MATCH_ID_COLUMN]
        ):
            raise TrainingDatasetError(
                f"Slice 8 fold {fold8.fold_id} TEST match_ids differ from Slice 7"
            )


def _coefficient_rows(
    predictor: LogisticRegressionPredictor,
    *,
    fold_id: int,
    spec_name: str,
) -> pd.DataFrame:
    table = standardized_coefficients(predictor)
    kept = table.loc[table["feature"].isin(_COEFFICIENT_FEATURE_NAMES)].copy()
    kept.insert(0, "fold_id", fold_id)
    kept.insert(1, "model", spec_name)
    return kept.reset_index(drop=True)


def _score_spec(
    *,
    fold: WalkForwardFold,
    spec: BlockAblationSpec,
    train: DatasetPartition,
    validation: DatasetPartition,
    test: DatasetPartition,
    preprocessing_spec: PreprocessingSpec,
) -> tuple[
    LogisticRegressionPredictor,
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
]:
    selected_c, _reg = _select_regularization(
        train, validation, spec.feature_columns
    )
    model = _fit_logistic(
        train,
        spec.feature_columns,
        config=LogisticRegressionConfig(
            C=selected_c, preprocessing=preprocessing_spec
        ),
    )
    evaluation = evaluate_predictor(spec.name, test, model)
    p = evaluation.predictions.p_radiant_win.reset_index(drop=True)
    y = evaluation.predictions.y_true.reset_index(drop=True)
    context = evaluation.predictions.context.reset_index(drop=True)
    spec_ll = per_sample_log_loss(y, p)
    metrics = evaluation.metrics
    selected_row = {
        "fold_id": fold.fold_id,
        "model": spec.name,
        "label": spec.label,
        "C": selected_c,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
    }
    fold_row = {
        "fold_id": fold.fold_id,
        "model": spec.name,
        "label": spec.label,
        "n_features": len(spec.feature_columns),
        "C": selected_c,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": metrics.n_samples,
        "log_loss": metrics.log_loss,
        "brier_score": metrics.brier_score,
        "accuracy_at_0.5": metrics.accuracy_at_0_5,
        "roc_auc": metrics.roc_auc,
        "ece": metrics.expected_calibration_error,
        "train_end": fold.train_end,
        "validation_end": fold.validation_end,
        "test_end": fold.test_end,
    }
    preds = pd.DataFrame(
        {
            "fold_id": fold.fold_id,
            "model": spec.name,
            "label": spec.label,
            "match_id": context[MATCH_ID_COLUMN].to_numpy(),
            "start_time": context["start_time"].to_numpy(),
            "game_version_id": context["game_version_id"].to_numpy(),
            "y_true": y.to_numpy(),
            "p_spec": p.to_numpy(),
            "sample_log_loss": spec_ll,
        }
    )
    return model, selected_row, fold_row, preds


def _attach_references(oos: pd.DataFrame) -> pd.DataFrame:
    elo = oos.loc[
        oos["model"] == ELO_BLOCK_SPEC_NAME,
        [MATCH_ID_COLUMN, "p_spec", "sample_log_loss"],
    ].rename(columns={"p_spec": "p_elo", "sample_log_loss": "elo_log_loss"})
    career = oos.loc[
        oos["model"] == SLICE8_CAREER_SPEC_NAME,
        [MATCH_ID_COLUMN, "sample_log_loss"],
    ].rename(columns={"sample_log_loss": "career_log_loss"})
    merged = oos.merge(elo, on=MATCH_ID_COLUMN, how="left", validate="many_to_one")
    merged = merged.merge(
        career, on=MATCH_ID_COLUMN, how="left", validate="many_to_one"
    )
    merged["delta_vs_elo"] = merged["sample_log_loss"] - merged["elo_log_loss"]
    merged["delta_vs_career"] = merged["sample_log_loss"] - merged["career_log_loss"]
    return merged


def _mean_or_nan(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def _spec_summary(subset: pd.DataFrame, spec_name: str) -> dict[str, float]:
    rows = subset.loc[subset["model"] == spec_name]
    n = len(rows)
    if n == 0:
        return {
            "n": 0.0,
            "log_loss": float("nan"),
            "delta_vs_elo": float("nan"),
            "delta_vs_career": float("nan"),
            "brier_score": float("nan"),
            "roc_auc": float("nan"),
        }
    full = evaluate_probabilities(rows["y_true"], rows["p_spec"])
    return {
        "n": float(n),
        "log_loss": full.log_loss,
        "delta_vs_elo": _mean_or_nan(rows["delta_vs_elo"]),
        "delta_vs_career": _mean_or_nan(rows["delta_vs_career"]),
        "brier_score": full.brier_score,
        "roc_auc": full.roc_auc,
    }


def _wide_delta_row(
    subset: pd.DataFrame, *, extra: dict[str, object]
) -> dict[str, object]:
    elo = _spec_summary(subset, ELO_BLOCK_SPEC_NAME)
    career = _spec_summary(subset, SLICE8_CAREER_SPEC_NAME)
    evidence = _spec_summary(subset, SLICE8_EVIDENCE_SPEC_NAME)
    role = _spec_summary(subset, SLICE8_ROLE_SPEC_NAME)
    patch = _spec_summary(subset, SLICE8_PATCH_SPEC_NAME)
    full = _spec_summary(subset, SLICE8_FULL_SPEC_NAME)
    gate = _spec_summary(subset, SLICE8_GATE_SPEC_NAME)
    return {
        **extra,
        "n": int(elo["n"]),
        "career_delta_vs_elo": career["delta_vs_elo"],
        "evidence_delta_vs_elo": evidence["delta_vs_elo"],
        "role_delta_vs_elo": role["delta_vs_elo"],
        "patch_delta_vs_elo": patch["delta_vs_elo"],
        "full_delta_vs_elo": full["delta_vs_elo"],
        "gate_delta_vs_elo": gate["delta_vs_elo"],
        "evidence_delta_vs_career": evidence["delta_vs_career"],
        "role_delta_vs_career": role["delta_vs_career"],
        "patch_delta_vs_career": patch["delta_vs_career"],
        "full_delta_vs_career": full["delta_vs_career"],
        "gate_delta_vs_career": gate["delta_vs_career"],
        "career_log_loss": career["log_loss"],
        "full_log_loss": full["log_loss"],
        "gate_log_loss": gate["log_loss"],
        "career_brier": career["brier_score"],
        "full_brier": full["brier_score"],
        "gate_brier": gate["brier_score"],
    }


def _subset_rows(oos: pd.DataFrame, match_ids: pd.Series) -> pd.DataFrame:
    ids = {int(value) for value in match_ids}
    return oos.loc[oos[MATCH_ID_COLUMN].isin(ids)]


def _group_table(
    oos: pd.DataFrame,
    labeled: pd.DataFrame,
    *,
    column: str,
    values: tuple[str, ...],
    key_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value in values:
        ids = labeled.loc[labeled[column] == value, MATCH_ID_COLUMN]
        rows.append(
            _wide_delta_row(_subset_rows(oos, ids), extra={key_name: value})
        )
    return pd.DataFrame(rows)


def _pooled_with_career(
    oos: pd.DataFrame, specs: tuple[BlockAblationSpec, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        subset = oos.loc[oos["model"] == spec.name]
        full = evaluate_probabilities(subset["y_true"], subset["p_spec"])
        rows.append(
            {
                "model": spec.name,
                "label": spec.label,
                "n_features": len(spec.feature_columns),
                "n": len(subset),
                "log_loss": full.log_loss,
                "brier_score": full.brier_score,
                "accuracy_at_0.5": full.accuracy_at_0_5,
                "roc_auc": full.roc_auc,
                "ece": full.expected_calibration_error,
                "delta_vs_elo": _mean_or_nan(subset["delta_vs_elo"]),
                "delta_vs_career": _mean_or_nan(subset["delta_vs_career"]),
            }
        )
    return pd.DataFrame(rows)


def _fold_with_references(
    fold_metrics: pd.DataFrame, oos: pd.DataFrame
) -> pd.DataFrame:
    extras = (
        oos.groupby(["fold_id", "model"], sort=False)[
            ["delta_vs_elo", "delta_vs_career"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "delta_vs_elo": "mean_delta_vs_elo",
                "delta_vs_career": "delta_vs_career",
            }
        )
    )
    folded = fold_metrics.drop(
        columns=[column for column in ("mean_delta_vs_elo",) if column in fold_metrics],
        errors="ignore",
    )
    return folded.merge(extras, on=["fold_id", "model"], how="left")


def _numeric_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _oos_bins_from_train(
    folds: tuple[WalkForwardFold, ...],
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Assign LOW/MEDIUM/HIGH using each fold's TRAIN tertiles on TEST rows."""
    frames: list[pd.DataFrame] = []
    diag = diagnostics.set_index(MATCH_ID_COLUMN)
    for fold in folds:
        edges_career = train_tertile_edges(
            fold.train.X[SLICE8_MATCH_MEAN_CAREER_GAMES]
        )
        edges_compat = train_tertile_edges(
            fold.train.X[SLICE8_MATCH_MEAN_ROLE_COMPATIBILITY]
        )
        rows = []
        for match_id in fold.test.context[MATCH_ID_COLUMN]:
            key = int(match_id)
            if key in diag.index:
                row = diag.loc[key]
                career_value = _numeric_or_none(row["mean_prior_games_on_hero"])
                compat_value = _numeric_or_none(row["mean_role_compatibility"])
            else:
                career_value = None
                compat_value = None
            rows.append(
                {
                    MATCH_ID_COLUMN: key,
                    "fold_id": fold.fold_id,
                    "career_evidence_bin": assign_train_tertile_bin(
                        career_value, q_low=edges_career[0], q_high=edges_career[1]
                    ),
                    "compatibility_bin": assign_train_tertile_bin(
                        compat_value, q_low=edges_compat[0], q_high=edges_compat[1]
                    ),
                    "career_q_low": edges_career[0],
                    "career_q_high": edges_career[1],
                    "compat_q_low": edges_compat[0],
                    "compat_q_high": edges_compat[1],
                }
            )
        frames.append(pd.DataFrame(rows))
    labeled = pd.concat(frames, ignore_index=True)
    labeled["cross_cell"] = [
        f"{career} × {compat}" if compat != "NULL" else "NULL compatibility"
        for career, compat in zip(
            labeled["career_evidence_bin"], labeled["compatibility_bin"]
        )
    ]
    return labeled


@dataclass
class Slice8BenchmarkReport:
    """Walk-forward Slice 8 gating plus TRAIN-binned diagnostic tables."""

    assembly: Slice8Assembly
    walk_forward: WalkForwardReport
    oos_predictions: pd.DataFrame
    overall: pd.DataFrame
    fold_metrics: pd.DataFrame
    selected_gates: pd.DataFrame
    coefficients: pd.DataFrame
    career_evidence: pd.DataFrame
    compatibility: pd.DataFrame
    cross_cell: pd.DataFrame
    patch_maturity: pd.DataFrame
    n_oos: int
    exploratory: bool = True


def run_slice8_player_hero_gating_benchmark(
    store: FeatureDuckDBConnection,
    *,
    config: WalkForwardConfig | None = None,
) -> Slice8BenchmarkReport:
    """Fit Slice 8 specs on the existing expanding-window folds."""
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    assembly = build_slice8_model_ready_dataset(store)
    assert_slice7_slice8_identity(
        assembly.slice7, assembly, config=resolved
    )
    folds = resolve_walk_forward_folds(assembly.dataset, config=resolved)
    preprocessing_spec = PreprocessingSpec()
    spec_by_name = {spec.name: spec for spec in SLICE8_META_PLAYER_HERO_SPECS}
    gate_spec = spec_by_name[SLICE8_GATE_SPEC_NAME]
    coefficient_specs = {
        SLICE8_CAREER_SPEC_NAME,
        SLICE8_EVIDENCE_SPEC_NAME,
        SLICE8_ROLE_SPEC_NAME,
        SLICE8_PATCH_SPEC_NAME,
        SLICE8_FULL_SPEC_NAME,
    }

    selected_c_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    selected_gate_rows: list[dict[str, object]] = []

    for fold in folds:
        for spec in SLICE8_STATIC_SPECS:
            model, selected_row, fold_row, preds = _score_spec(
                fold=fold,
                spec=spec,
                train=fold.train,
                validation=fold.validation,
                test=fold.test,
                preprocessing_spec=preprocessing_spec,
            )
            selected_c_rows.append(selected_row)
            fold_metric_rows.append(fold_row)
            prediction_frames.append(preds)
            if spec.name in coefficient_specs:
                coefficient_frames.append(
                    _coefficient_rows(
                        model, fold_id=fold.fold_id, spec_name=spec.name
                    )
                )

        candidates = gates_from_train(fold.train.X)
        selected_gate = select_career_gate(
            fold.train,
            fold.validation,
            candidates=candidates,
            feature_columns=gate_spec.feature_columns,
        )
        gated_train = apply_career_gate(fold.train, selected_gate)
        gated_val = apply_career_gate(fold.validation, selected_gate)
        gated_test = apply_career_gate(fold.test, selected_gate)
        gate_model, selected_row, fold_row, gate_preds = _score_spec(
            fold=fold,
            spec=gate_spec,
            train=gated_train,
            validation=gated_val,
            test=gated_test,
            preprocessing_spec=preprocessing_spec,
        )
        selected_c_rows.append(selected_row)
        fold_metric_rows.append(fold_row)
        prediction_frames.append(gate_preds)
        coefficient_frames.append(
            _coefficient_rows(
                gate_model, fold_id=fold.fold_id, spec_name=gate_spec.name
            )
        )
        selected_gate_rows.append(
            {
                "fold_id": fold.fold_id,
                "gate_name": selected_gate.name,
                "gate_kind": selected_gate.kind.value,
                "career_threshold": selected_gate.career_threshold,
                "compatibility_threshold": selected_gate.compatibility_threshold,
                "compat_scale_lo": selected_gate.compat_scale_lo,
                "compat_scale_hi": selected_gate.compat_scale_hi,
                "n_train": len(fold.train),
                "n_validation": len(fold.validation),
                "n_test": len(fold.test),
            }
        )

    oos = _attach_references(pd.concat(prediction_frames, ignore_index=True))
    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics = _fold_with_references(fold_metrics, oos)
    selected_C = pd.DataFrame(selected_c_rows)
    overall = _pooled_with_career(oos, SLICE8_META_PLAYER_HERO_SPECS)

    walk_forward = WalkForwardReport(
        preprocessing_spec=preprocessing_spec,
        config=resolved,
        specs=SLICE8_META_PLAYER_HERO_SPECS,
        folds=folds,
        selected_C=selected_C,
        fold_metrics=fold_metrics,
        pooled_metrics=_pooled_metrics(oos, SLICE8_META_PLAYER_HERO_SPECS),
        version_breakdown=_version_breakdown(oos, SLICE8_META_PLAYER_HERO_SPECS),
        version_fold_counts=(
            oos.loc[
                oos["model"] == ELO_BLOCK_SPEC_NAME,
                ["fold_id", "game_version_id"],
            ]
            .value_counts()
            .rename("n")
            .reset_index()
            .sort_values(["fold_id", "game_version_id"])
            .reset_index(drop=True)
        ),
        oos_predictions=oos,
    )

    labeled = _oos_bins_from_train(folds, assembly.match_diagnostics)
    oos_diag = assembly.match_diagnostics.merge(
        labeled, on=MATCH_ID_COLUMN, how="right"
    )
    career_evidence = _group_table(
        oos,
        oos_diag,
        column="career_evidence_bin",
        values=EVIDENCE_BIN_ORDER,
        key_name="career_evidence_bin",
    )
    compatibility = _group_table(
        oos,
        oos_diag,
        column="compatibility_bin",
        values=COMPATIBILITY_BIN_ORDER,
        key_name="compatibility_bin",
    )
    cross_values = CROSS_CELL_ORDER + ("NULL compatibility",)
    cross_cell = _group_table(
        oos,
        oos_diag,
        column="cross_cell",
        values=cross_values,
        key_name="cross_cell",
    )
    patch_maturity = _group_table(
        oos,
        oos_diag,
        column="patch_maturity_bin",
        values=PATCH_MATURITY_BIN_ORDER,
        key_name="maturity",
    )
    elo_oos = oos.loc[oos["model"] == ELO_BLOCK_SPEC_NAME]
    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame()
    )
    return Slice8BenchmarkReport(
        assembly=assembly,
        walk_forward=walk_forward,
        oos_predictions=oos,
        overall=overall,
        fold_metrics=fold_metrics,
        selected_gates=pd.DataFrame(selected_gate_rows),
        coefficients=coefficients,
        career_evidence=career_evidence,
        compatibility=compatibility,
        cross_cell=cross_cell,
        patch_maturity=patch_maturity,
        n_oos=len(elo_oos),
        exploratory=True,
    )
