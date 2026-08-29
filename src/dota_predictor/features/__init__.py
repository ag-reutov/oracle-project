"""DuckDB analytical layer over the canonical Parquet dataset (Step 3A)."""

from dota_predictor.features.availability import (
    FeatureAvailabilityError,
    SnapshotStage,
    assert_columns_allowed_for_stage,
    columns_allowed_for_stage,
)
from dota_predictor.features.config import FeatureStoreConfig, load_feature_store_config
from dota_predictor.features.duckdb_layer import (
    DRAFT_EVENTS_VIEW,
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
    connect,
)
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    PLAYER_HISTORY_FEATURE_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    ROSTER_CONTINUITY_FEATURE_COLUMNS,
    SNAPSHOT_COLUMNS,
    TARGET_COLUMN,
    TEAM_HISTORY_FEATURE_COLUMNS,
    PreDraftSnapshot,
    build_pre_draft_snapshot,
)
from dota_predictor.features.temporal import (
    HISTORICAL_START_TIME_SQL_CONDITION,
    is_historical,
)

__all__ = [
    "DRAFT_EVENTS_VIEW",
    "FEATURE_COLUMNS",
    "HISTORICAL_START_TIME_SQL_CONDITION",
    "IDENTITY_COLUMNS",
    "MATCHES_VIEW",
    "MATCH_PLAYERS_VIEW",
    "PLAYER_HISTORY_FEATURE_COLUMNS",
    "PRE_DRAFT_SNAPSHOT_SQL",
    "ROSTER_CONTINUITY_FEATURE_COLUMNS",
    "SNAPSHOT_COLUMNS",
    "TARGET_COLUMN",
    "TEAM_HISTORY_FEATURE_COLUMNS",
    "FeatureAvailabilityError",
    "FeatureDuckDBConnection",
    "FeatureStoreConfig",
    "PreDraftSnapshot",
    "SnapshotStage",
    "assert_columns_allowed_for_stage",
    "build_pre_draft_snapshot",
    "columns_allowed_for_stage",
    "connect",
    "is_historical",
    "load_feature_store_config",
]
