"""Tests for the Step 3 information-availability contract
(`features.availability`).

Pure, in-memory tests: no Parquet fixtures, no DuckDB connection, no
PostgreSQL.
"""

from __future__ import annotations

import pytest

from dota_predictor.features.availability import (
    DRAFT_EVENTS_COLUMN_AVAILABILITY,
    EXPECTED_POSITION_COLUMN_AVAILABILITY,
    GAME_VERSIONS_COLUMN_AVAILABILITY,
    HERO_STATE_COLUMN_AVAILABILITY,
    HEROES_COLUMN_AVAILABILITY,
    MATCH_PLAYERS_COLUMN_AVAILABILITY,
    MATCHES_COLUMN_AVAILABILITY,
    PLAYER_HERO_ELO_COLUMN_AVAILABILITY,
    PLAYER_HERO_META_COLUMN_AVAILABILITY,
    PLAYER_HERO_POSITION_COLUMN_AVAILABILITY,
    PLAYER_MATCH_COLUMN_AVAILABILITY,
    PLAYER_POSITION_STATE_COLUMN_AVAILABILITY,
    PROVENANCE_COLUMNS,
    FeatureAvailabilityError,
    SnapshotStage,
    assert_columns_allowed_for_stage,
    columns_allowed_for_stage,
)
from dota_predictor.features.hero_state import HERO_STATE_METRIC_COLUMNS
from dota_predictor.features.player_hero_elo import PLAYER_HERO_ELO_METRIC_COLUMNS
from dota_predictor.features.player_hero_meta import PLAYER_HERO_META_METRIC_COLUMNS

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


# --- match_players columns -----------------------------------------------

MATCH_PLAYERS_PRE_DRAFT_COLUMNS = {
    "match_id",
    "start_time",
    "side",
    "slot_in_side",
    "player_id",
    "team_id",
}


def test_match_players_roster_columns_available_pre_draft() -> None:
    allowed = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
    assert allowed == MATCH_PLAYERS_PRE_DRAFT_COLUMNS


def test_match_players_hero_id_is_draft_not_pre_draft() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert MATCH_PLAYERS_COLUMN_AVAILABILITY["hero_id"] == InformationAvailability.DRAFT
    for column in MATCH_PLAYERS_PRE_DRAFT_COLUMNS:
        assert (
            MATCH_PLAYERS_COLUMN_AVAILABILITY[column]
            == InformationAvailability.PRE_DRAFT
        )

    pre_draft = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
    assert "hero_id" not in pre_draft

    post_draft = columns_allowed_for_stage("match_players", SnapshotStage.POST_DRAFT)
    assert "hero_id" in post_draft
    assert MATCH_PLAYERS_PRE_DRAFT_COLUMNS.issubset(post_draft)


def test_unclassified_match_players_column_is_rejected() -> None:
    with pytest.raises(FeatureAvailabilityError, match="not_a_real_column"):
        assert_columns_allowed_for_stage(
            "match_players",
            SnapshotStage.POST_DRAFT,
            ["match_id", "not_a_real_column"],
        )


def test_match_players_hero_id_rejected_at_pre_draft() -> None:
    with pytest.raises(FeatureAvailabilityError, match="hero_id"):
        assert_columns_allowed_for_stage(
            "match_players", SnapshotStage.PRE_DRAFT, ["match_id", "hero_id"]
        )


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


# --- reference views do not reclassify fact hero_id ----------------------


def test_heroes_dimension_is_pre_draft_static_metadata() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert HEROES_COLUMN_AVAILABILITY == {
        "hero_id": InformationAvailability.PRE_DRAFT,
        "name": InformationAvailability.PRE_DRAFT,
    }
    allowed = columns_allowed_for_stage("heroes", SnapshotStage.PRE_DRAFT)
    assert allowed == {"hero_id", "name"}


def test_game_versions_dimension_is_pre_draft() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert set(GAME_VERSIONS_COLUMN_AVAILABILITY) == {
        "game_version_id",
        "name",
        "as_of_datetime",
    }
    assert set(GAME_VERSIONS_COLUMN_AVAILABILITY.values()) == {
        InformationAvailability.PRE_DRAFT
    }
    allowed = columns_allowed_for_stage("game_versions", SnapshotStage.PRE_DRAFT)
    assert allowed == {"game_version_id", "name", "as_of_datetime"}


def test_hero_catalog_does_not_make_match_players_hero_id_pre_draft() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert MATCH_PLAYERS_COLUMN_AVAILABILITY["hero_id"] == InformationAvailability.DRAFT
    pre_draft = columns_allowed_for_stage("match_players", SnapshotStage.PRE_DRAFT)
    assert "hero_id" not in pre_draft
    with pytest.raises(FeatureAvailabilityError, match="hero_id"):
        assert_columns_allowed_for_stage(
            "match_players", SnapshotStage.PRE_DRAFT, ["hero_id"]
        )


def test_hero_catalog_does_not_make_draft_events_hero_id_pre_draft() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert DRAFT_EVENTS_COLUMN_AVAILABILITY["hero_id"] == InformationAvailability.DRAFT
    pre_draft = columns_allowed_for_stage("draft_events", SnapshotStage.PRE_DRAFT)
    assert "hero_id" not in pre_draft


def test_matches_game_version_id_remains_pre_draft() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert (
        MATCHES_COLUMN_AVAILABILITY["game_version_id"]
        == InformationAvailability.PRE_DRAFT
    )


def test_player_match_won_is_post_match_not_a_pre_or_post_draft_feature() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert PLAYER_MATCH_COLUMN_AVAILABILITY["won"] == InformationAvailability.POST_MATCH
    assert PLAYER_MATCH_COLUMN_AVAILABILITY["hero_id"] == InformationAvailability.DRAFT
    pre_draft = columns_allowed_for_stage("player_match", SnapshotStage.PRE_DRAFT)
    assert "won" not in pre_draft
    assert "hero_id" not in pre_draft
    post_draft = columns_allowed_for_stage("player_match", SnapshotStage.POST_DRAFT)
    assert "hero_id" in post_draft
    assert "won" not in post_draft
    with pytest.raises(FeatureAvailabilityError, match="won"):
        assert_columns_allowed_for_stage(
            "player_match", SnapshotStage.POST_DRAFT, ["player_id", "won"]
        )


def test_observed_position_lane_role_are_post_match_not_current_features() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    for view, availability in (
        ("match_players", MATCH_PLAYERS_COLUMN_AVAILABILITY),
        ("player_match", PLAYER_MATCH_COLUMN_AVAILABILITY),
        ("player_position_state", PLAYER_POSITION_STATE_COLUMN_AVAILABILITY),
    ):
        for column in ("position", "lane", "role"):
            assert availability[column] == InformationAvailability.POST_MATCH
            pre_draft = columns_allowed_for_stage(view, SnapshotStage.PRE_DRAFT)
            post_draft = columns_allowed_for_stage(view, SnapshotStage.POST_DRAFT)
            assert column not in pre_draft
            assert column not in post_draft
            with pytest.raises(FeatureAvailabilityError, match=column):
                assert_columns_allowed_for_stage(
                    view, SnapshotStage.PRE_DRAFT, ["match_id", column]
                )
            with pytest.raises(FeatureAvailabilityError, match=column):
                assert_columns_allowed_for_stage(
                    view, SnapshotStage.POST_DRAFT, [column]
                )


def test_slot_in_side_remains_pre_draft_and_is_not_position() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert (
        MATCH_PLAYERS_COLUMN_AVAILABILITY["slot_in_side"]
        == InformationAvailability.PRE_DRAFT
    )
    assert (
        PLAYER_MATCH_COLUMN_AVAILABILITY["slot_in_side"]
        == InformationAvailability.PRE_DRAFT
    )
    assert "slot_in_side" in columns_allowed_for_stage(
        "match_players", SnapshotStage.PRE_DRAFT
    )
    assert "position" not in columns_allowed_for_stage(
        "match_players", SnapshotStage.PRE_DRAFT
    )


def test_historical_position_metrics_are_pre_draft_current_position_is_not() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert (
        PLAYER_POSITION_STATE_COLUMN_AVAILABILITY["position"]
        == InformationAvailability.POST_MATCH
    )
    assert (
        PLAYER_POSITION_STATE_COLUMN_AVAILABILITY["prior_games_position_1"]
        == InformationAvailability.PRE_DRAFT
    )
    pre_draft = columns_allowed_for_stage(
        "player_position_state", SnapshotStage.PRE_DRAFT
    )
    assert "prior_games_position_1" in pre_draft
    assert "historical_modal_position" in pre_draft
    assert "recent_position_stability" in pre_draft
    assert "position" not in pre_draft
    assert "won" not in pre_draft
    with pytest.raises(FeatureAvailabilityError, match="position"):
        assert_columns_allowed_for_stage(
            "player_position_state",
            SnapshotStage.PRE_DRAFT,
            ["player_id", "position"],
        )


def test_expected_position_is_pre_draft_observed_position_is_not() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert (
        EXPECTED_POSITION_COLUMN_AVAILABILITY["expected_position"]
        == InformationAvailability.PRE_DRAFT
    )
    assert (
        EXPECTED_POSITION_COLUMN_AVAILABILITY["observed_position"]
        == InformationAvailability.POST_MATCH
    )
    pre_draft = columns_allowed_for_stage("expected_position", SnapshotStage.PRE_DRAFT)
    assert "expected_position" in pre_draft
    assert "assigned_position_score" in pre_draft
    assert "previous_explicit_position" in pre_draft
    assert "evidence_tier" in pre_draft
    assert "observed_position" not in pre_draft
    with pytest.raises(FeatureAvailabilityError, match="observed_position"):
        assert_columns_allowed_for_stage(
            "expected_position",
            SnapshotStage.PRE_DRAFT,
            ["player_id", "observed_position"],
        )


def test_player_hero_position_metrics_are_draft_not_pre_draft() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert (
        PLAYER_HERO_POSITION_COLUMN_AVAILABILITY["expected_position"]
        == InformationAvailability.PRE_DRAFT
    )
    assert (
        PLAYER_HERO_POSITION_COLUMN_AVAILABILITY["hero_id"]
        == InformationAvailability.DRAFT
    )
    assert (
        PLAYER_HERO_POSITION_COLUMN_AVAILABILITY["prior_games_on_hero_at_expected_position"]
        == InformationAvailability.DRAFT
    )
    assert (
        PLAYER_HERO_POSITION_COLUMN_AVAILABILITY["observed_position"]
        == InformationAvailability.POST_MATCH
    )
    pre_draft = columns_allowed_for_stage(
        "player_hero_position", SnapshotStage.PRE_DRAFT
    )
    post_draft = columns_allowed_for_stage(
        "player_hero_position", SnapshotStage.POST_DRAFT
    )
    assert "expected_position" in pre_draft
    assert "hero_id" not in pre_draft
    assert "prior_games_on_hero" not in pre_draft
    assert "observed_position" not in pre_draft
    assert "hero_id" in post_draft
    assert "prior_games_on_hero_at_expected_position" in post_draft
    assert "observed_position" not in post_draft
    with pytest.raises(FeatureAvailabilityError, match="observed_position"):
        assert_columns_allowed_for_stage(
            "player_hero_position",
            SnapshotStage.POST_DRAFT,
            ["player_id", "hero_id", "observed_position"],
        )


def test_hero_state_metrics_are_pre_draft_and_do_not_expose_current_outcome() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert (
        HERO_STATE_COLUMN_AVAILABILITY["hero_id"] == InformationAvailability.PRE_DRAFT
    )
    assert (
        HERO_STATE_COLUMN_AVAILABILITY["hero_prior_matches"]
        == InformationAvailability.PRE_DRAFT
    )
    assert (
        HERO_STATE_COLUMN_AVAILABILITY["hero_position_1_share"]
        == InformationAvailability.PRE_DRAFT
    )
    for column in (
        "radiant_win",
        "won",
        "position",
        "observed_position",
        "expected_position",
    ):
        assert column not in HERO_STATE_COLUMN_AVAILABILITY

    pre_draft = columns_allowed_for_stage("hero_state", SnapshotStage.PRE_DRAFT)
    post_draft = columns_allowed_for_stage("hero_state", SnapshotStage.POST_DRAFT)
    for column in HERO_STATE_METRIC_COLUMNS:
        assert column in pre_draft
        assert column in post_draft
        assert (
            HERO_STATE_COLUMN_AVAILABILITY[column]
            == InformationAvailability.PRE_DRAFT
        )
    assert "hero_id" in pre_draft
    assert "hero_name" in pre_draft

    # Existing current-match restrictions are not loosened.
    assert MATCH_PLAYERS_COLUMN_AVAILABILITY["hero_id"] == InformationAvailability.DRAFT
    assert (
        MATCH_PLAYERS_COLUMN_AVAILABILITY["position"]
        == InformationAvailability.POST_MATCH
    )
    assert (
        PLAYER_HERO_POSITION_COLUMN_AVAILABILITY["hero_id"]
        == InformationAvailability.DRAFT
    )


def test_player_hero_meta_metrics_are_draft_not_pre_draft() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert (
        PLAYER_HERO_META_COLUMN_AVAILABILITY["expected_position"]
        == InformationAvailability.PRE_DRAFT
    )
    assert PLAYER_HERO_META_COLUMN_AVAILABILITY["hero_id"] == InformationAvailability.DRAFT
    assert (
        PLAYER_HERO_META_COLUMN_AVAILABILITY["prior_games_on_hero"]
        == InformationAvailability.DRAFT
    )
    assert (
        PLAYER_HERO_META_COLUMN_AVAILABILITY["player_hero_recent_role_compatibility"]
        == InformationAvailability.DRAFT
    )
    assert (
        PLAYER_HERO_META_COLUMN_AVAILABILITY["hero_recent_50_contest_rate"]
        == InformationAvailability.DRAFT
    )
    assert (
        PLAYER_HERO_META_COLUMN_AVAILABILITY["observed_position"]
        == InformationAvailability.POST_MATCH
    )
    pre_draft = columns_allowed_for_stage("player_hero_meta", SnapshotStage.PRE_DRAFT)
    post_draft = columns_allowed_for_stage("player_hero_meta", SnapshotStage.POST_DRAFT)
    assert "expected_position" in pre_draft
    assert "hero_id" not in pre_draft
    assert "prior_games_on_hero" not in pre_draft
    assert "player_hero_recent_20_matches" not in pre_draft
    assert "observed_position" not in pre_draft
    assert "hero_id" in post_draft
    assert "player_hero_recent_role_compatibility" in post_draft
    assert "hero_position_share_at_expected_position" in post_draft
    assert "observed_position" not in post_draft
    with pytest.raises(FeatureAvailabilityError, match="observed_position"):
        assert_columns_allowed_for_stage(
            "player_hero_meta",
            SnapshotStage.POST_DRAFT,
            ["player_id", "hero_id", "observed_position"],
        )
    # Slice 5's own view is still PRE_DRAFT; Slice 6 does not loosen it.
    assert (
        HERO_STATE_COLUMN_AVAILABILITY["hero_recent_50_contest_rate"]
        == InformationAvailability.PRE_DRAFT
    )
    assert MATCH_PLAYERS_COLUMN_AVAILABILITY["hero_id"] == InformationAvailability.DRAFT
    assert (
        MATCH_PLAYERS_COLUMN_AVAILABILITY["position"]
        == InformationAvailability.POST_MATCH
    )
    for column in PLAYER_HERO_META_METRIC_COLUMNS:
        assert (
            PLAYER_HERO_META_COLUMN_AVAILABILITY[column]
            == InformationAvailability.DRAFT
        )


def test_player_hero_elo_metrics_are_draft_not_pre_draft() -> None:
    from dota_predictor.data.canonical_schema import InformationAvailability

    assert PLAYER_HERO_ELO_COLUMN_AVAILABILITY["hero_id"] == InformationAvailability.DRAFT
    assert (
        PLAYER_HERO_ELO_COLUMN_AVAILABILITY["player_id"]
        == InformationAvailability.PRE_DRAFT
    )
    assert (
        PLAYER_HERO_ELO_COLUMN_AVAILABILITY["prior_games_on_hero"]
        == InformationAvailability.DRAFT
    )
    assert (
        PLAYER_HERO_ELO_COLUMN_AVAILABILITY["shrunk_outcome_residual_on_hero"]
        == InformationAvailability.DRAFT
    )
    pre_draft = columns_allowed_for_stage("player_hero_elo", SnapshotStage.PRE_DRAFT)
    post_draft = columns_allowed_for_stage("player_hero_elo", SnapshotStage.POST_DRAFT)
    assert "player_id" in pre_draft
    assert "hero_id" not in pre_draft
    assert "prior_games_on_hero" not in pre_draft
    assert "mean_outcome_residual_on_hero" not in pre_draft
    assert "hero_id" in post_draft
    assert "shrunk_outcome_residual_on_hero" in post_draft
    for column in PLAYER_HERO_ELO_METRIC_COLUMNS:
        assert (
            PLAYER_HERO_ELO_COLUMN_AVAILABILITY[column]
            == InformationAvailability.DRAFT
        )
