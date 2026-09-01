"""Post-draft model-ready assembly: Elo + full draft-comparison diffs.

The prediction boundary here is immediately after the draft is complete.
Team Elo remains the PRE_DRAFT rating snapshot; draft-comparison diffs
are the already leakage-safe Radiant − Dire profile. This module does
not recompute history, does not change `FEATURE_COLUMNS` / PRE_DRAFT
snapshot SQL, and does not drop comparison metrics from descriptive
correlations.

`radiant_win` is the label only. Missing draft-comparison values stay
NULL until TRAIN-fitted preprocessing (median + missingness indicators).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dota_predictor.features.draft_comparison import (
    DRAFT_COMPARISON_METRIC_COLUMNS,
    MATCH_ID_COLUMN,
    build_draft_comparison,
)
from dota_predictor.features.duckdb_layer import FeatureDuckDBConnection
from dota_predictor.features.pre_draft_snapshot import (
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
    build_pre_draft_snapshot,
)
from dota_predictor.features.team_elo import EloConfig
from dota_predictor.training.baselines import (
    ConstantProbabilityBaseline,
    EloOnlyProbabilityBaseline,
    EmpiricalRateBaseline,
)
from dota_predictor.training.dataset import ModelReadyDataset, TrainingDatasetError
from dota_predictor.training.evaluation import (
    ModelEvaluation,
    _fit_logistic,
    _select_regularization,
    evaluate_predictor,
)
from dota_predictor.training.feature_sets import (
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_DRAFT_COMPARISON_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    BlockAblationSpec,
)
from dota_predictor.training.logistic_model import (
    LogisticRegressionConfig,
    standardized_coefficients,
)
from dota_predictor.training.split import ChronologicalSplit

__all__ = [
    "PostDraftBenchmarkReport",
    "PostDraftBlockAblationReport",
    "build_post_draft_model_ready_dataset",
    "run_post_draft_benchmark",
    "run_post_draft_block_ablation",
]


def build_post_draft_model_ready_dataset(
    store: FeatureDuckDBConnection,
    *,
    elo_config: EloConfig | None = None,
) -> ModelReadyDataset:
    """One row per match: Elo features plus every draft-comparison diff.

    Inner layers already exclude the current match. Joins are one-to-one
    on `match_id`. Rows are sorted by (`start_time`, `match_id`) the same
    way as `build_model_ready_dataset`.
    """
    snapshot = build_pre_draft_snapshot(store, elo_config=elo_config)
    full = snapshot.to_frame()
    comparison = build_draft_comparison(store).to_frame()

    snapshot_ids = set(full[MATCH_ID_COLUMN])
    comparison_ids = set(comparison[MATCH_ID_COLUMN])
    missing = snapshot_ids - comparison_ids
    extra = comparison_ids - snapshot_ids
    if missing or extra:
        raise TrainingDatasetError(
            "draft comparison and PRE_DRAFT snapshot must cover the same "
            f"matches; missing={len(missing)} extra={len(extra)}"
        )

    merged = full.merge(
        comparison[[MATCH_ID_COLUMN, *DRAFT_COMPARISON_METRIC_COLUMNS]],
        on=MATCH_ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    ordered = merged.sort_values(
        ["start_time", MATCH_ID_COLUMN], kind="stable"
    ).reset_index(drop=True)

    feature_columns = ELO_PLUS_DRAFT_COMPARISON_COLUMNS
    X = ordered[list(feature_columns)].copy()
    y = ordered[TARGET_COLUMN].copy()
    context = ordered[list(IDENTITY_COLUMNS)].copy()
    return ModelReadyDataset(
        X=X,
        y=y,
        context=context,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        identity_columns=IDENTITY_COLUMNS,
    )


@dataclass
class PostDraftBenchmarkReport:
    """Elo vs Elo + all draft-comparison diffs on one chronological split."""

    preprocessing_spec: object
    elo_logistic_C: float
    elo_plus_draft_C: float
    regularization_comparison: pd.DataFrame
    validation_evaluations: dict[str, ModelEvaluation]
    coefficients: pd.DataFrame
    n_draft_comparison_features: int
    test_evaluations: dict[str, ModelEvaluation] = field(default_factory=dict)


def run_post_draft_benchmark(
    split: ChronologicalSplit,
    *,
    include_test_evaluation: bool = True,
) -> PostDraftBenchmarkReport:
    """Compare formula Elo, logistic Elo, and logistic Elo + draft diffs.

    Fitting/preprocessing uses TRAIN only. Regularization ``C`` is chosen
    independently for each logistic spec on VALIDATION. TEST is evaluated
    once with those frozen specs. Every draft-comparison metric is used;
    none are dropped from descriptive correlations.
    """
    from dota_predictor.training.preprocessing import PreprocessingSpec

    preprocessing_spec = PreprocessingSpec()

    constant = ConstantProbabilityBaseline(probability=0.5)
    constant.fit(split.train.X, split.train.y)
    empirical = EmpiricalRateBaseline()
    empirical.fit(split.train.X, split.train.y)
    elo_formula = EloOnlyProbabilityBaseline()
    elo_formula.fit(split.train.X, split.train.y)

    elo_c, elo_reg = _select_regularization(
        split.train, split.validation, ELO_ONLY_FEATURE_COLUMNS
    )
    draft_c, draft_reg = _select_regularization(
        split.train, split.validation, ELO_PLUS_DRAFT_COMPARISON_COLUMNS
    )
    elo_reg = elo_reg.assign(model="logistic_elo_only")
    draft_reg = draft_reg.assign(model="logistic_elo_plus_draft_comparison")
    regularization_comparison = pd.concat(
        [elo_reg, draft_reg], ignore_index=True
    )

    elo_config = LogisticRegressionConfig(
        C=elo_c, preprocessing=preprocessing_spec
    )
    draft_config = LogisticRegressionConfig(
        C=draft_c, preprocessing=preprocessing_spec
    )
    logistic_elo = _fit_logistic(
        split.train, ELO_ONLY_FEATURE_COLUMNS, config=elo_config
    )
    logistic_draft = _fit_logistic(
        split.train, ELO_PLUS_DRAFT_COMPARISON_COLUMNS, config=draft_config
    )

    validation_evaluations = {
        "constant_0.5": evaluate_predictor(
            "constant_0.5", split.validation, constant
        ),
        "empirical_train_rate": evaluate_predictor(
            "empirical_train_rate", split.validation, empirical
        ),
        "elo_only": evaluate_predictor("elo_only", split.validation, elo_formula),
        "logistic_elo_only": evaluate_predictor(
            "logistic_elo_only", split.validation, logistic_elo
        ),
        "logistic_elo_plus_draft_comparison": evaluate_predictor(
            "logistic_elo_plus_draft_comparison",
            split.validation,
            logistic_draft,
        ),
    }

    test_evaluations: dict[str, ModelEvaluation] = {}
    if include_test_evaluation:
        test_evaluations = {
            "constant_0.5": evaluate_predictor(
                "constant_0.5", split.test, constant
            ),
            "empirical_train_rate": evaluate_predictor(
                "empirical_train_rate", split.test, empirical
            ),
            "elo_only": evaluate_predictor("elo_only", split.test, elo_formula),
            "logistic_elo_only": evaluate_predictor(
                "logistic_elo_only", split.test, logistic_elo
            ),
            "logistic_elo_plus_draft_comparison": evaluate_predictor(
                "logistic_elo_plus_draft_comparison",
                split.test,
                logistic_draft,
            ),
        }

    return PostDraftBenchmarkReport(
        preprocessing_spec=preprocessing_spec,
        elo_logistic_C=elo_c,
        elo_plus_draft_C=draft_c,
        regularization_comparison=regularization_comparison,
        validation_evaluations=validation_evaluations,
        coefficients=standardized_coefficients(logistic_draft),
        n_draft_comparison_features=len(DRAFT_COMPARISON_METRIC_COLUMNS),
        test_evaluations=test_evaluations,
    )


@dataclass
class PostDraftBlockAblationReport:
    """Predefined draft-block combinations on one chronological split."""

    preprocessing_spec: object
    selected_C: dict[str, float]
    regularization_comparison: pd.DataFrame
    validation_evaluations: dict[str, ModelEvaluation]
    n_features: dict[str, int]
    specs: tuple[BlockAblationSpec, ...]
    test_evaluations: dict[str, ModelEvaluation] = field(default_factory=dict)


def run_post_draft_block_ablation(
    split: ChronologicalSplit,
    *,
    include_test_evaluation: bool = True,
) -> PostDraftBlockAblationReport:
    """Compare Elo plus each predefined draft-comparison block.

    Uses the same chronological split, TRAIN-only ``PreprocessingSpec``,
    and per-spec validation ``C`` selection as ``run_post_draft_benchmark``.
    Feature columns are subsets of the already-assembled post-draft
    matrix; history is not recomputed.
    """
    from dota_predictor.training.preprocessing import PreprocessingSpec

    preprocessing_spec = PreprocessingSpec()
    specs = POST_DRAFT_BLOCK_ABLATION_SPECS

    selected_c: dict[str, float] = {}
    reg_frames: list[pd.DataFrame] = []
    validation_evaluations: dict[str, ModelEvaluation] = {}
    fitted: dict[str, object] = {}
    n_features = {spec.name: len(spec.feature_columns) for spec in specs}

    for spec in specs:
        c, reg = _select_regularization(
            split.train, split.validation, spec.feature_columns
        )
        selected_c[spec.name] = c
        reg_frames.append(reg.assign(model=spec.name, label=spec.label))
        config = LogisticRegressionConfig(C=c, preprocessing=preprocessing_spec)
        model = _fit_logistic(
            split.train, spec.feature_columns, config=config
        )
        fitted[spec.name] = model
        validation_evaluations[spec.name] = evaluate_predictor(
            spec.name, split.validation, model
        )

    regularization_comparison = pd.concat(reg_frames, ignore_index=True)

    test_evaluations: dict[str, ModelEvaluation] = {}
    if include_test_evaluation:
        test_evaluations = {
            spec.name: evaluate_predictor(spec.name, split.test, fitted[spec.name])
            for spec in specs
        }

    return PostDraftBlockAblationReport(
        preprocessing_spec=preprocessing_spec,
        selected_C=selected_c,
        regularization_comparison=regularization_comparison,
        validation_evaluations=validation_evaluations,
        n_features=n_features,
        specs=specs,
        test_evaluations=test_evaluations,
    )
