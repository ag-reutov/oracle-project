"""Feature-column subsets for Step 4B ablation comparisons.

These are views over the existing `FEATURE_COLUMNS` contract from
Step 3B/3C -- no new features are defined here.
"""

from __future__ import annotations

from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    PLAYER_HISTORY_FEATURE_COLUMNS,
    ROSTER_CONTINUITY_FEATURE_COLUMNS,
    TEAM_HISTORY_FEATURE_COLUMNS,
)
from dota_predictor.features.team_elo import TEAM_ELO_FEATURE_COLUMNS

__all__ = [
    "ALL_FEATURE_COLUMNS",
    "ELO_ONLY_FEATURE_COLUMNS",
    "HISTORICAL_WITHOUT_ELO_COLUMNS",
]

ELO_ONLY_FEATURE_COLUMNS: tuple[str, ...] = TEAM_ELO_FEATURE_COLUMNS

HISTORICAL_WITHOUT_ELO_COLUMNS: tuple[str, ...] = (
    TEAM_HISTORY_FEATURE_COLUMNS
    + PLAYER_HISTORY_FEATURE_COLUMNS
    + ROSTER_CONTINUITY_FEATURE_COLUMNS
)

ALL_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS
