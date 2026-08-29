"""Step 4A: model-ready training dataset assembly and chronological
train/validation/test splitting, built on top of
`features.pre_draft_snapshot`.

Deliberately separate from `features/`: that package describes what
was knowable at prediction time; this package describes which of its
already-computed rows/columns are usable for ML and how they are
split. No feature/Elo/history computation lives here.
"""

from dota_predictor.training.dataset import (
    ModelReadyDataset,
    TrainingDatasetError,
    build_model_ready_dataset,
)
from dota_predictor.training.diagnostics import (
    missing_value_report,
    rows_with_any_missing,
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
    "DEFAULT_SPLIT_CONFIG",
    "ChronologicalSplit",
    "ChronologicalSplitConfig",
    "ChronologicalSplitError",
    "DatasetPartition",
    "ModelReadyDataset",
    "SplitBoundaries",
    "TrainingDatasetError",
    "build_model_ready_dataset",
    "chronological_split",
    "missing_value_report",
    "resolve_split_boundaries",
    "rows_with_any_missing",
]
