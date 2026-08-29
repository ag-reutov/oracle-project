"""Tests for the Step 3 information-availability contract
(`features.availability`).

Pure, in-memory tests: no Parquet fixtures, no DuckDB connection, no
PostgreSQL.
"""

from __future__ import annotations

import pytest

from dota_predictor.features.availability import (
    DRAFT_EVENTS_COLUMN_AVAILABILITY,
    MATCH_PLAYERS_COLUMN_AVAILABILITY,
    MATCHES_COLUMN_AVAILABILITY,
    PROVENANCE_COLUMNS,
    FeatureAvailabilityError,
    SnapshotStage,
    assert_columns_allowed_for_stage,
    columns_allowed_for_stage,
)

POST_MATCH_MATCHES_COLUMNS = {"radiant_win", "duration_seconds"}
DRAFT_ONLY_COLUMNS = {"sequence", "action", "side", "hero_id", "was_successful"}


# --- PRE_DRAFT excludes draft and post-match information ----------------


def test_pre_draft_excludes_post_match_matches_columns() -> None:
    allowed = columns_allowed_for_stage("matches", SnapshotStage.PRE_DRAFT)
    assert allowed.isdisjoint(POST_MATCH_MATCHES_COLUMNS)


def test_pre_draft_excludes_all_draft_event_columns() -> None:
    allowed = columns_allowed_for_stage("draft_events", SnapshotStage.PRE_DRAFT)
    assert allowed.isdisjoint(DRAFT_ONLY_COLUMNS)
    # match_id (identity/join key) remains available.
    assert allowed == {"match_id"}


def test_pre_draft_allows_roster_and_identity_matches_columns() -> None:
    allowed = columns_allowed_for_stage("matches", SnapshotStage.PRE_DRAFT)
    assert {
        "match_id",
        "start_time",
        "league_id",
        "radiant_team_id",
        "dire_team_id",
        "radiant_player_0_id",
        "dire_player_4_id",
    }.issubset(allowed)


# --- POST_DRAFT may expose draft, but never outcome ----------------------


def test_post_draft_allows_draft_event_columns() -> None:
    allowed = columns_allowed_for_stage("draft_events", SnapshotStage.POST_DRAFT)
    assert DRAFT_ONLY_COLUMNS.issubset(allowed)


def test_post_draft_still_excludes_post_match_matches_columns() -> None:
    allowed = columns_allowed_for_stage("matches", SnapshotStage.POST_DRAFT)
    assert allowed.isdisjoint(POST_MATCH_MATCHES_COLUMNS)


# --- provenance columns are never predictive features --------------------


@pytest.mark.parametrize("stage", [SnapshotStage.PRE_DRAFT, SnapshotStage.POST_DRAFT])
def test_provenance_columns_never_allowed(stage: SnapshotStage) -> None:
    allowed = columns_allowed_for_stage("matches", stage)
    assert allowed.isdisjoint(PROVENANCE_COLUMNS)


def test_provenance_columns_absent_from_matches_availability_map() -> None:
    # mapper_version/canonicalized_at are deliberately not classified via
    # InformationAvailability at all -- they are pipeline metadata, not
    # game-timeline information (see module docstring).
    assert PROVENANCE_COLUMNS.isdisjoint(MATCHES_COLUMN_AVAILABILITY)


# --- match_players columns are all PRE_DRAFT -----------------------------


def test_match_players_columns_all_available_pre_draft() -> None:
    allowed = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
    assert allowed == set(MATCH_PLAYERS_COLUMN_AVAILABILITY)


# --- assert_columns_allowed_for_stage raises for violations --------------


def test_assert_raises_for_post_match_column_at_pre_draft() -> None:
    with pytest.raises(FeatureAvailabilityError, match="radiant_win"):
        assert_columns_allowed_for_stage(
            "matches", SnapshotStage.PRE_DRAFT, ["match_id", "radiant_win"]
        )


def test_assert_raises_for_draft_column_at_pre_draft() -> None:
    with pytest.raises(FeatureAvailabilityError, match="hero_id"):
        assert_columns_allowed_for_stage(
            "draft_events", SnapshotStage.PRE_DRAFT, ["match_id", "hero_id"]
        )


def test_assert_raises_for_post_match_column_at_post_draft() -> None:
    with pytest.raises(FeatureAvailabilityError, match="duration_seconds"):
        assert_columns_allowed_for_stage(
            "matches", SnapshotStage.POST_DRAFT, ["duration_seconds"]
        )


def test_assert_raises_for_provenance_column_at_either_stage() -> None:
    with pytest.raises(FeatureAvailabilityError, match="mapper_version"):
        assert_columns_allowed_for_stage(
            "matches", SnapshotStage.POST_DRAFT, ["mapper_version"]
        )


def test_assert_passes_for_allowed_columns() -> None:
    assert_columns_allowed_for_stage(
        "matches", SnapshotStage.PRE_DRAFT, ["match_id", "radiant_team_id"]
    )
    assert_columns_allowed_for_stage(
        "draft_events", SnapshotStage.POST_DRAFT, ["match_id", "hero_id", "action"]
    )


def test_unknown_view_raises() -> None:
    with pytest.raises(FeatureAvailabilityError, match="unknown view"):
        columns_allowed_for_stage("not_a_real_view", SnapshotStage.PRE_DRAFT)


def test_draft_events_column_availability_has_no_post_match_entries() -> None:
    # The whole draft_events view is DRAFT-classified (plus match_id as an
    # identity key); it must never contain a POST_MATCH-classified column.
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert (
        InformationAvailability.POST_MATCH
        not in DRAFT_EVENTS_COLUMN_AVAILABILITY.values()
    )
