"""Step 4A: model-ready training dataset assembly and chronological
train/validation/test splitting, built on top of
`features.pre_draft_snapshot`.

Deliberately separate from `features/`: that package describes what
was knowable at prediction time; this package describes which of its
already-computed rows/columns are usable for ML and how they are
split. No feature/Elo/history computation lives here.
"""

from dota_predictor.training.baselines import (
    ConstantProbabilityBaseline,
    EloOnlyProbabilityBaseline,
    EmpiricalRateBaseline,
)
from dota_predictor.training.dataset import (
    ModelReadyDataset,
    TrainingDatasetError,
    build_model_ready_dataset,
)
from dota_predictor.training.diagnostics import (
    missing_value_report,
    rows_with_any_missing,
)
from dota_predictor.training.evaluation import BenchmarkReport, run_step4b_benchmark
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    DRAFT_COMPARISON_FEATURE_COLUMNS,
    ELO_ONLY_FEATURE_COLUMNS,
    ELO_PLUS_DRAFT_COMPARISON_COLUMNS,
    HISTORICAL_WITHOUT_ELO_COLUMNS,
)
from dota_predictor.training.logistic_model import (
    LogisticRegressionConfig,
    LogisticRegressionPredictor,
    build_logistic_pipeline,
    standardized_coefficients,
)
from dota_predictor.training.metrics import EvaluationMetrics, evaluate_probabilities
from dota_predictor.training.post_draft import (
    PostDraftBenchmarkReport,
    build_post_draft_model_ready_dataset,
    run_post_draft_benchmark,
)
from dota_predictor.training.preprocessing import (
    PreprocessingSpec,
    build_preprocessing_pipeline,
)
from dota_predictor.training.split import (
    DEFAULT_SPLIT_CONFIG,
    ChronologicalSplit,
    ChronologicalSplitConfig,
    ChronologicalSplitError,
    DatasetPartition,
    SplitBoundaries,
    chronological_split,
    resolve_split_boundaries,
)

__all__ = [
    "ALL_FEATURE_COLUMNS",
    "DEFAULT_SPLIT_CONFIG",
    "DRAFT_COMPARISON_FEATURE_COLUMNS",
    "ELO_ONLY_FEATURE_COLUMNS",
    "ELO_PLUS_DRAFT_COMPARISON_COLUMNS",
    "HISTORICAL_WITHOUT_ELO_COLUMNS",
    "BenchmarkReport",
    "ChronologicalSplit",
    "ChronologicalSplitConfig",
    "ChronologicalSplitError",
    "ConstantProbabilityBaseline",
    "DatasetPartition",
    "EloOnlyProbabilityBaseline",
    "EmpiricalRateBaseline",
    "EvaluationMetrics",
    "LogisticRegressionConfig",
    "LogisticRegressionPredictor",
    "ModelReadyDataset",
    "PostDraftBenchmarkReport",
    "PreprocessingSpec",
    "SplitBoundaries",
    "TrainingDatasetError",
    "build_logistic_pipeline",
    "build_model_ready_dataset",
    "build_post_draft_model_ready_dataset",
    "build_preprocessing_pipeline",
    "chronological_split",
    "evaluate_probabilities",
    "missing_value_report",
    "resolve_split_boundaries",
    "rows_with_any_missing",
    "run_post_draft_benchmark",
    "run_step4b_benchmark",
    "standardized_coefficients",
]
