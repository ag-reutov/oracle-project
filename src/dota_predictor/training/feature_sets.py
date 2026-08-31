"""Feature-column subsets for ablation comparisons.

PRE_DRAFT views (`ALL_FEATURE_COLUMNS`, `ELO_ONLY_FEATURE_COLUMNS`,
`HISTORICAL_WITHOUT_ELO_COLUMNS`) are slices of the existing
`FEATURE_COLUMNS` contract from Step 3B/3C.

Post-draft views append the full Radiant − Dire draft-comparison
metric set. No subset is chosen from descriptive correlations.
"""

from __future__ import annotations

from dota_predictor.features.draft_comparison import DRAFT_COMPARISON_METRIC_COLUMNS
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PLAYER_HISTORY_FEATURE_COLUMNS,
    ROSTER_CONTINUITY_FEATURE_COLUMNS,
    TEAM_HISTORY_FEATURE_COLUMNS,
)
from dota_predictor.features.team_elo import TEAM_ELO_FEATURE_COLUMNS

__all__ = [
    "ALL_FEATURE_COLUMNS",
    "DRAFT_COMPARISON_FEATURE_COLUMNS",
    "ELO_ONLY_FEATURE_COLUMNS",
    "ELO_PLUS_DRAFT_COMPARISON_COLUMNS",
    "HISTORICAL_WITHOUT_ELO_COLUMNS",
]

ELO_ONLY_FEATURE_COLUMNS: tuple[str, ...] = TEAM_ELO_FEATURE_COLUMNS

HISTORICAL_WITHOUT_ELO_COLUMNS: tuple[str, ...] = (
    TEAM_HISTORY_FEATURE_COLUMNS
    + PLAYER_HISTORY_FEATURE_COLUMNS
    + ROSTER_CONTINUITY_FEATURE_COLUMNS
)

ALL_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS

# Full comparison-layer metric set. Not a correlation-selected subset.
DRAFT_COMPARISON_FEATURE_COLUMNS: tuple[str, ...] = DRAFT_COMPARISON_METRIC_COLUMNS

ELO_PLUS_DRAFT_COMPARISON_COLUMNS: tuple[str, ...] = (
    ELO_ONLY_FEATURE_COLUMNS + DRAFT_COMPARISON_FEATURE_COLUMNS
)
