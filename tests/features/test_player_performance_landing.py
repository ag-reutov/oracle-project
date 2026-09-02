"""Slice 11: landed player box-score scalars are POST_MATCH observations.

Not a rating, target, feature, or benchmark. Confirms the new columns are
classified POST_MATCH and are absent from production feature contracts.
"""

from __future__ import annotations

import pytest

from dota_predictor.data.canonical_schema import (
    MATCH_PLAYER_BOX_SCORE_COLUMNS,
    STRATZ_PLAYER_BOX_SCORE_FIELDS,
    InformationAvailability,
)
from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.features.availability import (
    MATCH_PLAYERS_COLUMN_AVAILABILITY,
    FeatureAvailabilityError,
    SnapshotStage,
    assert_columns_allowed_for_stage,
    columns_allowed_for_stage,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS, SNAPSHOT_COLUMNS
from dota_predictor.ingestion.queries import (
    MATCH_PLAYER_PERFORMANCE_QUERY,
    MATCH_PLAYER_POSITION_SELECTION,
    MATCH_SELECTION,
)
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS


def _players_selection(query: str) -> str:
    start = query.index("players {")
    depth = 0
    for index, char in enumerate(query[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return query[start : index + 1]
    raise AssertionError("unclosed players selection")


def test_match_selection_includes_existing_and_box_score_fields() -> None:
    players = _players_selection(MATCH_SELECTION)
    for field in (
        "steamAccountId",
        "isRadiant",
        "playerSlot",
        "heroId",
        "position",
        "lane",
        "role",
    ):
        assert field in players
    for field in STRATZ_PLAYER_BOX_SCORE_FIELDS:
        assert field in players
    for forbidden in ("imp", "award", "heroAverage", "isVictory", "partyId", "stats"):
        assert forbidden not in players
    position_players = _players_selection(MATCH_PLAYER_POSITION_SELECTION)
    for field in STRATZ_PLAYER_BOX_SCORE_FIELDS:
        assert field not in position_players


def test_performance_query_does_not_request_time_series_or_derived_scores() -> None:
    for forbidden in ("imp", "award", "heroAverage", "isVictory", "partyId", "stats"):
        assert forbidden not in MATCH_PLAYER_PERFORMANCE_QUERY
    for field in STRATZ_PLAYER_BOX_SCORE_FIELDS:
        assert field in MATCH_PLAYER_PERFORMANCE_QUERY


def test_box_score_columns_are_post_match_on_match_players() -> None:
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert (
            MATCH_PLAYERS_COLUMN_AVAILABILITY[column]
            == InformationAvailability.POST_MATCH
        )
        pre_draft = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
        post_draft = columns_allowed_for_stage(
            "match_players", SnapshotStage.POST_DRAFT
        )
        assert column not in pre_draft
        assert column not in post_draft
        with pytest.raises(FeatureAvailabilityError, match=column):
            assert_columns_allowed_for_stage(
                "match_players", SnapshotStage.PRE_DRAFT, ["match_id", column]
            )
        with pytest.raises(FeatureAvailabilityError, match=column):
            assert_columns_allowed_for_stage(
                "match_players", SnapshotStage.POST_DRAFT, [column]
            )


def test_box_score_columns_are_not_production_features() -> None:
    for column in MATCH_PLAYER_BOX_SCORE_COLUMNS:
        assert column not in FEATURE_COLUMNS
        assert column not in SNAPSHOT_COLUMNS
        assert column not in ALL_FEATURE_COLUMNS


def test_analytical_schema_version_is_v4_for_box_score_columns() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 4
