"""Step 3 information-availability contract for the DuckDB feature layer.

Maps the column names of the analytical relations
(`matches`/`draft_events`/`match_players`, plus the optional reference
views `heroes`/`game_versions` -- see `duckdb_layer.py`) onto
`SnapshotStage`, so a future feature builder cannot casually select a
column that was not actually knowable at the stage it claims to be
building features for.

This module reuses `canonical_schema.InformationAvailability` verbatim
(see `.cursor/rules/ml.mdc`) rather than inventing a parallel
classification: `PRE_DRAFT`/`DRAFT`/`POST_MATCH` mean exactly what they
mean in `canonical_schema.py`. The per-column dictionaries here only
translate that existing classification onto the *Parquet column names*
of the analytical views, including columns that don't exist verbatim on
`CanonicalMatch` (e.g. `radiant_player_0_id` is the pivoted form of
`radiant_player_ids`; `match_players.team_id` is the side-derived form
of `radiant_team_id`/`dire_team_id`; `match_players.hero_id` is the
played-hero column from `match_players.parquet`).

`PROVENANCE_COLUMNS` (`mapper_version`, `canonicalized_at`) is the one
genuinely new category, and it is deliberately NOT expressed as
`InformationAvailability.POST_MATCH`: these columns are not a fact about
the game's outcome, they are metadata about when/how a row was written
by the ingestion pipeline. Reclassifying pipeline metadata as "post
match" would be a contradictory duplicate semantic -- instead it is
excluded from every stage directly, orthogonally to the game-timeline
classification.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from dota_predictor.data.canonical_schema import InformationAvailability
from dota_predictor.features.expected_position import EXPECTED_POSITION_EVIDENCE_COLUMNS
from dota_predictor.features.player_position import PLAYER_POSITION_STATE_METRIC_COLUMNS

__all__ = [
    "DRAFT_EVENTS_COLUMN_AVAILABILITY",
    "EXPECTED_POSITION_COLUMN_AVAILABILITY",
    "GAME_VERSIONS_COLUMN_AVAILABILITY",
    "HEROES_COLUMN_AVAILABILITY",
    "MATCHES_COLUMN_AVAILABILITY",
    "MATCH_PLAYERS_COLUMN_AVAILABILITY",
    "PLAYER_MATCH_COLUMN_AVAILABILITY",
    "PLAYER_POSITION_STATE_COLUMN_AVAILABILITY",
    "PROVENANCE_COLUMNS",
    "STAGE_ALLOWED_AVAILABILITY",
    "FeatureAvailabilityError",
    "SnapshotStage",
    "assert_columns_allowed_for_stage",
    "columns_allowed_for_stage",
]


class FeatureAvailabilityError(ValueError):
    """Raised when a column is not available at a requested `SnapshotStage`."""


class SnapshotStage(str, Enum):
    """A point in a match's lifecycle at which a prediction snapshot may
    be taken.

    Step 3A only defines the stages; it does not build feature matrices
    for either of them yet (see module docstring of `duckdb_layer.py`).

    * PRE_DRAFT: before the first draft action. Only PRE_DRAFT-classified
      information is available (identity, scheduling, competition/series
      context, rosters, patch -- never draft or outcome data).
    * POST_DRAFT: once the draft has concluded. PRE_DRAFT and DRAFT
      information is available (the completed draft becomes readable),
      but outcome data (`InformationAvailability.POST_MATCH`) is still
      never available -- the game has not been played yet at this
      snapshot point.
    """

    PRE_DRAFT = "PRE_DRAFT"
    POST_DRAFT = "POST_DRAFT"


STAGE_ALLOWED_AVAILABILITY: dict[SnapshotStage, frozenset[InformationAvailability]] = {
    SnapshotStage.PRE_DRAFT: frozenset({InformationAvailability.PRE_DRAFT}),
    SnapshotStage.POST_DRAFT: frozenset(
        {InformationAvailability.PRE_DRAFT, InformationAvailability.DRAFT}
    ),
}

# Pipeline write-provenance, not game state. Never available as a feature
# input at any stage -- see module docstring.
PROVENANCE_COLUMNS: frozenset[str] = frozenset({"mapper_version", "canonicalized_at"})

_PLAYER_SLOT_AVAILABILITY = {
    f"{side}_player_{slot}_id": InformationAvailability.PRE_DRAFT
    for side in ("radiant", "dire")
    for slot in range(5)
}

# Column availability for the `matches` view (== matches.parquet,
# unmodified). Mirrors `canonical_schema.FIELD_INFORMATION_AVAILABILITY`
# for every field that maps 1:1 onto a Parquet column; the pivoted
# `{side}_player_{slot}_id` columns inherit the PRE_DRAFT classification
# of the `radiant_player_ids`/`dire_player_ids` fields they were pivoted
# from. `draft_events` is not a `matches` column (it is its own view).
MATCHES_COLUMN_AVAILABILITY: dict[str, InformationAvailability] = {
    "match_id": InformationAvailability.PRE_DRAFT,
    "league_id": InformationAvailability.PRE_DRAFT,
    "start_time": InformationAvailability.PRE_DRAFT,
    "league_name": InformationAvailability.PRE_DRAFT,
    "series_id": InformationAvailability.PRE_DRAFT,
    "series_type": InformationAvailability.PRE_DRAFT,
    "game_number_in_series": InformationAvailability.PRE_DRAFT,
    "game_version_id": InformationAvailability.PRE_DRAFT,
    "radiant_team_id": InformationAvailability.PRE_DRAFT,
    "radiant_team_name_observed": InformationAvailability.PRE_DRAFT,
    "dire_team_id": InformationAvailability.PRE_DRAFT,
    "dire_team_name_observed": InformationAvailability.PRE_DRAFT,
    **_PLAYER_SLOT_AVAILABILITY,
    "radiant_win": InformationAvailability.POST_MATCH,
    "duration_seconds": InformationAvailability.POST_MATCH,
    # mapper_version / canonicalized_at are intentionally absent: see
    # `PROVENANCE_COLUMNS`, which excludes them regardless of this map.
}

# Column availability for the `draft_events` view (== draft_events.parquet,
# unmodified). Every non-key column is DRAFT, matching
# `canonical_schema.FIELD_INFORMATION_AVAILABILITY["draft_events"]`
# (the whole event sequence is DRAFT information; there is no separate
# POST_DRAFT classification in `canonical_schema`, since "post draft" is
# just the terminal state of the DRAFT sequence -- see that module's
# `InformationAvailability` docstring).
DRAFT_EVENTS_COLUMN_AVAILABILITY: dict[str, InformationAvailability] = {
    "match_id": InformationAvailability.PRE_DRAFT,
    "sequence": InformationAvailability.DRAFT,
    "action": InformationAvailability.DRAFT,
    "side": InformationAvailability.DRAFT,
    "hero_id": InformationAvailability.DRAFT,
    "was_successful": InformationAvailability.DRAFT,
}

# Column availability for the `match_players` view (`match_players.parquet`
# joined to `matches.start_time` -- see `duckdb_layer.py`). Roster and
# team-membership facts are PRE_DRAFT (the same classification as
# `radiant_player_ids`/`dire_player_ids`/`radiant_team_id`/`dire_team_id`
# on `matches`). Played `hero_id` is DRAFT: it is knowable once the draft
# is complete, never before the first draft action.
MATCH_PLAYERS_COLUMN_AVAILABILITY: dict[str, InformationAvailability] = {
    "match_id": InformationAvailability.PRE_DRAFT,
    "start_time": InformationAvailability.PRE_DRAFT,
    "side": InformationAvailability.PRE_DRAFT,
    "slot_in_side": InformationAvailability.PRE_DRAFT,
    "player_id": InformationAvailability.PRE_DRAFT,
    "team_id": InformationAvailability.PRE_DRAFT,
    "hero_id": InformationAvailability.DRAFT,
    # Observed STRATZ replay-parse labels for THIS match. Historical
    # rows may later contribute to state for a future match M only when
    # `H.start_time < M.start_time`. They are never PRE_DRAFT or
    # POST_DRAFT features of the same match.
    "position": InformationAvailability.POST_MATCH,
    "lane": InformationAvailability.POST_MATCH,
    "role": InformationAvailability.POST_MATCH,
}

# Derived player-match fact (`features.player_match.player_match_sql`):
# match_players joined to matches for series/patch/won. Not a Parquet
# file and not registered by `connect()`. `won` is POST_MATCH relative
# to the row's own match; historical aggregations of past `won` are a
# later layer's concern.
PLAYER_MATCH_COLUMN_AVAILABILITY: dict[str, InformationAvailability] = {
    "match_id": InformationAvailability.PRE_DRAFT,
    "player_id": InformationAvailability.PRE_DRAFT,
    "start_time": InformationAvailability.PRE_DRAFT,
    "game_version_id": InformationAvailability.PRE_DRAFT,
    "series_id": InformationAvailability.PRE_DRAFT,
    "team_id": InformationAvailability.PRE_DRAFT,
    "side": InformationAvailability.PRE_DRAFT,
    "hero_id": InformationAvailability.DRAFT,
    "won": InformationAvailability.POST_MATCH,
    "slot_in_side": InformationAvailability.PRE_DRAFT,
    "position": InformationAvailability.POST_MATCH,
    "lane": InformationAvailability.POST_MATCH,
    "role": InformationAvailability.POST_MATCH,
}

# Derived player × position historical state. Current-row `position` /
# `lane` / `role` remain POST_MATCH parse labels. Aggregates over
# strictly earlier explicit positions are PRE_DRAFT historical state for
# the current match and must not be confused with expected current
# position. They are not training features until a later slice selects
# them into a snapshot.
PLAYER_POSITION_STATE_COLUMN_AVAILABILITY: dict[str, InformationAvailability] = {
    **PLAYER_MATCH_COLUMN_AVAILABILITY,
    **{
        column: InformationAvailability.PRE_DRAFT
        for column in PLAYER_POSITION_STATE_METRIC_COLUMNS
    },
}

# Expected-position assignment. Historical evidence and the inferred
# `expected_position` are PRE_DRAFT. `observed_position` is the current
# match STRATZ parse label and remains POST_MATCH evaluation-only.
EXPECTED_POSITION_COLUMN_AVAILABILITY: dict[str, InformationAvailability] = {
    "match_id": InformationAvailability.PRE_DRAFT,
    "player_id": InformationAvailability.PRE_DRAFT,
    "start_time": InformationAvailability.PRE_DRAFT,
    "game_version_id": InformationAvailability.PRE_DRAFT,
    "team_id": InformationAvailability.PRE_DRAFT,
    "side": InformationAvailability.PRE_DRAFT,
    "expected_position": InformationAvailability.PRE_DRAFT,
    **{
        column: InformationAvailability.PRE_DRAFT
        for column in EXPECTED_POSITION_EVIDENCE_COLUMNS
    },
    "observed_position": InformationAvailability.POST_MATCH,
}

# Column availability for the optional `heroes` reference view. This is
# static catalog metadata (id -> display name). It does not reveal which
# hero a player chose in a match: `match_players.hero_id` and
# `draft_events.hero_id` remain DRAFT. Classified PRE_DRAFT because the
# catalog is knowable before any draft action.
HEROES_COLUMN_AVAILABILITY: dict[str, InformationAvailability] = {
    "hero_id": InformationAvailability.PRE_DRAFT,
    "name": InformationAvailability.PRE_DRAFT,
}

# Column availability for the optional `game_versions` reference view.
# Patch identity is PRE_DRAFT context, matching `matches.game_version_id`.
# Names are not denormalized onto the fact views.
GAME_VERSIONS_COLUMN_AVAILABILITY: dict[str, InformationAvailability] = {
    "game_version_id": InformationAvailability.PRE_DRAFT,
    "name": InformationAvailability.PRE_DRAFT,
    "as_of_datetime": InformationAvailability.PRE_DRAFT,
}

_VIEW_COLUMN_AVAILABILITY: dict[str, dict[str, InformationAvailability]] = {
    "matches": MATCHES_COLUMN_AVAILABILITY,
    "draft_events": DRAFT_EVENTS_COLUMN_AVAILABILITY,
    "match_players": MATCH_PLAYERS_COLUMN_AVAILABILITY,
    "player_match": PLAYER_MATCH_COLUMN_AVAILABILITY,
    "player_position_state": PLAYER_POSITION_STATE_COLUMN_AVAILABILITY,
    "expected_position": EXPECTED_POSITION_COLUMN_AVAILABILITY,
    "heroes": HEROES_COLUMN_AVAILABILITY,
    "game_versions": GAME_VERSIONS_COLUMN_AVAILABILITY,
}


def columns_allowed_for_stage(view: str, stage: SnapshotStage) -> frozenset[str]:
    """Return the columns of `view` that are safe to read at `stage`.

    Fails closed: a column absent from the relevant
    `*_COLUMN_AVAILABILITY` map (e.g. a future Parquet column this module
    has not yet classified) is never included, and `PROVENANCE_COLUMNS`
    is excluded regardless of stage.
    """
    try:
        column_availability = _VIEW_COLUMN_AVAILABILITY[view]
    except KeyError as exc:
        raise FeatureAvailabilityError(
            f"unknown view {view!r}; expected one of {sorted(_VIEW_COLUMN_AVAILABILITY)}"
        ) from exc

    allowed_availability = STAGE_ALLOWED_AVAILABILITY[stage]
    return frozenset(
        column
        for column, availability in column_availability.items()
        if column not in PROVENANCE_COLUMNS and availability in allowed_availability
    )


def assert_columns_allowed_for_stage(
    view: str, stage: SnapshotStage, columns: Iterable[str]
) -> None:
    """Raise `FeatureAvailabilityError` if any of `columns` is unsafe to
    read from `view` at `stage` (including provenance and unclassified
    columns)."""
    allowed = columns_allowed_for_stage(view, stage)
    disallowed = sorted(set(columns) - allowed)
    if disallowed:
        raise FeatureAvailabilityError(
            f"columns {disallowed} of view {view!r} are not available at "
            f"stage {stage.value}"
        )
