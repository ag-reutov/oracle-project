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
from dota_predictor.features.player_hero_meta_comparison import (
    SLICE7_RECENT20_COUNT_DIFF_COLUMNS,
    SLICE7_RECENT20_RATE_DIFF_COLUMNS,
    SLICE7_ROLE_DIFF_COLUMNS,
    SLICE7_SAME_VERSION_COUNT_DIFF_COLUMNS,
    SLICE7_SAME_VERSION_RATE_DIFF_COLUMNS,
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
    "SLICE7_CAREER_SPEC_NAME",
    "SLICE7_META_PLAYER_HERO_SPECS",
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


# Slice 7 evaluation specs. Named blocks on top of Elo. Not production
# FEATURE_COLUMNS and not a replacement for POST_DRAFT_BLOCK_ABLATION_SPECS.
SLICE7_CAREER_SPEC_NAME = "logistic_elo_plus_player_hero"

ELO_PLUS_SAME_VERSION_VOLUME_COLUMNS: tuple[str, ...] = (
    ELO_ONLY_FEATURE_COLUMNS + SLICE7_SAME_VERSION_COUNT_DIFF_COLUMNS
)
ELO_PLUS_SAME_VERSION_VOLUME_PERFORMANCE_COLUMNS: tuple[str, ...] = (
    ELO_PLUS_SAME_VERSION_VOLUME_COLUMNS + SLICE7_SAME_VERSION_RATE_DIFF_COLUMNS
)
ELO_PLUS_RECENT20_VOLUME_COLUMNS: tuple[str, ...] = (
    ELO_ONLY_FEATURE_COLUMNS + SLICE7_RECENT20_COUNT_DIFF_COLUMNS
)
ELO_PLUS_RECENT20_VOLUME_PERFORMANCE_COLUMNS: tuple[str, ...] = (
    ELO_PLUS_RECENT20_VOLUME_COLUMNS + SLICE7_RECENT20_RATE_DIFF_COLUMNS
)
ELO_PLUS_ROLE_META_COLUMNS: tuple[str, ...] = (
    ELO_ONLY_FEATURE_COLUMNS + SLICE7_ROLE_DIFF_COLUMNS
)
ELO_PLUS_SAME_VERSION_ROLE_COLUMNS: tuple[str, ...] = (
    ELO_PLUS_SAME_VERSION_VOLUME_PERFORMANCE_COLUMNS + SLICE7_ROLE_DIFF_COLUMNS
)
ELO_PLUS_RECENT20_ROLE_COLUMNS: tuple[str, ...] = (
    ELO_PLUS_RECENT20_VOLUME_PERFORMANCE_COLUMNS + SLICE7_ROLE_DIFF_COLUMNS
)
ELO_PLUS_CAREER_ROLE_COLUMNS: tuple[str, ...] = (
    ELO_PLUS_PLAYER_HERO_COLUMNS + SLICE7_ROLE_DIFF_COLUMNS
)

SLICE7_META_PLAYER_HERO_SPECS: tuple[BlockAblationSpec, ...] = (
    BlockAblationSpec(
        name="logistic_elo_only",
        label="Elo only",
        feature_columns=ELO_ONLY_FEATURE_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_player_hero",
        label="Elo + career Player × Hero",
        feature_columns=ELO_PLUS_PLAYER_HERO_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_same_version_volume",
        label="Elo + same-version volume",
        feature_columns=ELO_PLUS_SAME_VERSION_VOLUME_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_same_version_volume_performance",
        label="Elo + same-version volume + WR",
        feature_columns=ELO_PLUS_SAME_VERSION_VOLUME_PERFORMANCE_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_recent20_volume",
        label="Elo + recent-20 volume",
        feature_columns=ELO_PLUS_RECENT20_VOLUME_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_recent20_volume_performance",
        label="Elo + recent-20 volume + WR",
        feature_columns=ELO_PLUS_RECENT20_VOLUME_PERFORMANCE_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_role_meta",
        label="Elo + role/meta block",
        feature_columns=ELO_PLUS_ROLE_META_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_same_version_role",
        label="Elo + same-version + role",
        feature_columns=ELO_PLUS_SAME_VERSION_ROLE_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_recent20_role",
        label="Elo + recent-20 + role",
        feature_columns=ELO_PLUS_RECENT20_ROLE_COLUMNS,
    ),
    BlockAblationSpec(
        name="logistic_elo_plus_career_role",
        label="Elo + career Player × Hero + role",
        feature_columns=ELO_PLUS_CAREER_ROLE_COLUMNS,
    ),
)
