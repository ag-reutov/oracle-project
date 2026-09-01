"""Feature-column subsets for ablation comparisons.

PRE_DRAFT views (`ALL_FEATURE_COLUMNS`, `ELO_ONLY_FEATURE_COLUMNS`,
`HISTORICAL_WITHOUT_ELO_COLUMNS`) are slices of the existing
`FEATURE_COLUMNS` contract from Step 3B/3C.

Post-draft views append Radiant − Dire draft-comparison diffs. The
block ablation uses the three source layers (Player × Hero, Team × Hero,
Hero Meta) as predefined groups; no subset is chosen from descriptive
correlations.
"""

from __future__ import annotations

from dataclasses import dataclass

from dota_predictor.features.draft_comparison import DRAFT_COMPARISON_METRIC_COLUMNS
from dota_predictor.features.draft_profile import (
    DRAFT_PROFILE_HERO_META_METRIC_COLUMNS,
    DRAFT_PROFILE_PLAYER_METRIC_COLUMNS,
    DRAFT_PROFILE_TEAM_METRIC_COLUMNS,
)
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
    "ELO_PLUS_ALL_THREE_COLUMNS",
    "ELO_PLUS_DRAFT_COMPARISON_COLUMNS",
    "ELO_PLUS_HERO_META_COLUMNS",
    "ELO_PLUS_PLAYER_AND_TEAM_HERO_COLUMNS",
    "ELO_PLUS_PLAYER_HERO_COLUMNS",
    "ELO_PLUS_TEAM_HERO_COLUMNS",
    "HERO_META_COMPARISON_COLUMNS",
    "HISTORICAL_WITHOUT_ELO_COLUMNS",
    "PLAYER_HERO_COMPARISON_COLUMNS",
    "POST_DRAFT_BLOCK_ABLATION_SPECS",
    "TEAM_HERO_COMPARISON_COLUMNS",
    "BlockAblationSpec",
]


def _diff_columns(metrics: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{metric}_diff" for metric in metrics)


ELO_ONLY_FEATURE_COLUMNS: tuple[str, ...] = TEAM_ELO_FEATURE_COLUMNS

HISTORICAL_WITHOUT_ELO_COLUMNS: tuple[str, ...] = (
    TEAM_HISTORY_FEATURE_COLUMNS
    + PLAYER_HISTORY_FEATURE_COLUMNS
    + ROSTER_CONTINUITY_FEATURE_COLUMNS
)

ALL_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS

# Full comparison-layer metric set. Not a correlation-selected subset.
DRAFT_COMPARISON_FEATURE_COLUMNS: tuple[str, ...] = DRAFT_COMPARISON_METRIC_COLUMNS

PLAYER_HERO_COMPARISON_COLUMNS: tuple[str, ...] = _diff_columns(
    DRAFT_PROFILE_PLAYER_METRIC_COLUMNS
)
TEAM_HERO_COMPARISON_COLUMNS: tuple[str, ...] = _diff_columns(
    DRAFT_PROFILE_TEAM_METRIC_COLUMNS
)
HERO_META_COMPARISON_COLUMNS: tuple[str, ...] = _diff_columns(
    DRAFT_PROFILE_HERO_META_METRIC_COLUMNS
)

ELO_PLUS_DRAFT_COMPARISON_COLUMNS: tuple[str, ...] = (
    ELO_ONLY_FEATURE_COLUMNS + DRAFT_COMPARISON_FEATURE_COLUMNS
)

ELO_PLUS_PLAYER_HERO_COLUMNS: tuple[str, ...] = (
    ELO_ONLY_FEATURE_COLUMNS + PLAYER_HERO_COMPARISON_COLUMNS
)
ELO_PLUS_TEAM_HERO_COLUMNS: tuple[str, ...] = (
    ELO_ONLY_FEATURE_COLUMNS + TEAM_HERO_COMPARISON_COLUMNS
)
ELO_PLUS_HERO_META_COLUMNS: tuple[str, ...] = (
    ELO_ONLY_FEATURE_COLUMNS + HERO_META_COMPARISON_COLUMNS
)
ELO_PLUS_PLAYER_AND_TEAM_HERO_COLUMNS: tuple[str, ...] = (
    ELO_ONLY_FEATURE_COLUMNS
    + PLAYER_HERO_COMPARISON_COLUMNS
    + TEAM_HERO_COMPARISON_COLUMNS
)
ELO_PLUS_ALL_THREE_COLUMNS: tuple[str, ...] = ELO_PLUS_DRAFT_COMPARISON_COLUMNS


@dataclass(frozen=True)
class BlockAblationSpec:
    """One predefined post-draft feature-block combination."""

    name: str
    label: str
    feature_columns: tuple[str, ...]


POST_DRAFT_BLOCK_ABLATION_SPECS: tuple[BlockAblationSpec, ...] = (
    BlockAblationSpec(
        name="logistic_elo_only",
        label="Elo only",
        feature_columns=ELO_ONLY_FEATURE_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_player_hero",
        label="Elo + Player × Hero",
        feature_columns=ELO_PLUS_PLAYER_HERO_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_team_hero",
        label="Elo + Team × Hero",
        feature_columns=ELO_PLUS_TEAM_HERO_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_hero_meta",
        label="Elo + Hero Meta",
        feature_columns=ELO_PLUS_HERO_META_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_player_and_team_hero",
        label="Elo + Player × Hero + Team × Hero",
        feature_columns=ELO_PLUS_PLAYER_AND_TEAM_HERO_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_all_three",
        label="Elo + all three",
        feature_columns=ELO_PLUS_ALL_THREE_COLUMNS,
    ),
)
