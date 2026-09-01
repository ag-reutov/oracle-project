"""Slice 8 evaluation columns are not production features."""

from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
)
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    SLICE8_CONTEXT_COLUMNS,
    SLICE8_INTERACTION_COLUMNS,
)


def test_slice8_columns_are_not_production_features() -> None:
    for column in (*SLICE8_CONTEXT_COLUMNS, *SLICE8_INTERACTION_COLUMNS):
        assert column not in FEATURE_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS
        assert column not in PRE_DRAFT_SNAPSHOT_SQL
