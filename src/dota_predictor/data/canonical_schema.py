"""Canonical historical match schema.

This module defines the canonical internal representation of ONE historical
professional Dota 2 game (not a series). It preserves historical facts as
they occurred, translated from a raw source (currently STRATZ, see
`stratz_mapping.py`) into a stable, typed, validated internal shape.

Scope and intent
-----------------
A `CanonicalMatch` is allowed to contain information that would not have
been available before or during the match (the complete draft, the winner,
match duration). That is intentional: this module records what happened
historically. It does NOT decide what is safe to use as a model input at a
given prediction time.

Later feature-building code is responsible for enforcing the prediction-time
boundary (see `.cursor/rules/ml.mdc`) by consulting `FIELD_INFORMATION_AVAILABILITY`
(and, for the draft, slicing `draft_events`) to expose only information that
was knowable at the relevant timestamp. This module intentionally does not
implement that slicing/consumption logic; it only preserves enough structure
to make it possible.

Draft format flexibility
-------------------------
Competitive Dota pick/ban structure (Captains Mode and other formats) has
changed across patches. This module does not assume any fixed total number
of draft events, nor any single historical pick/ban phase ordering. The only
structural invariant enforced on `draft_events` is that they form a
deterministic, gap-free, zero-indexed sequence, and that exactly five heroes
were actually picked per side -- a fixed rule of the game (5v5) rather than
an assumption about draft-format phase structure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

__all__ = [
    "EXPLICIT_DOTA_POSITIONS",
    "FIELD_INFORMATION_AVAILABILITY",
    "MATCH_PLAYER_BOX_SCORE_COLUMNS",
    "PLAYER_BOX_SCORE_FIELD_MAP",
    "STRATZ_PLAYER_BOX_SCORE_FIELDS",
    "CanonicalMatch",
    "CanonicalMatchError",
    "DraftAction",
    "DraftEvent",
    "HeroId",
    "InformationAvailability",
    "LeagueId",
    "MatchId",
    "MatchLane",
    "MatchPlayerBoxScore",
    "MatchPlayerPosition",
    "MatchPlayerRole",
    "PlayerId",
    "SeriesId",
    "Side",
    "TeamId",
]

# Type aliases document intent; STRATZ (and Dota generally) use plain
# integer identifiers for all of these.
MatchId = int
LeagueId = int
SeriesId = int
# `TeamId`/`PlayerId` are STRATZ's own team id / `steamAccountId`. They
# are the stable source identifiers this project joins/aggregates on --
# not a guarantee that they map 1:1 to a permanent real-world
# organization or person (team ids can be reused/rebranded by STRATZ;
# player accounts are not verified real-world identity). Treat them as
# "stable within STRATZ", not as a stronger identity claim, unless a
# future richer identity model is introduced with explicit evidence.
TeamId = int
PlayerId = int
HeroId = int


class CanonicalMatchError(ValueError):
    """Raised when a canonical match record fails structural validation.

    Structurally invalid data raises this explicitly; it is never silently
    coerced or warned-and-skipped. Genuinely optional/unavailable source
    data is instead modeled as `None` on the relevant field.
    """


class InformationAvailability(str, Enum):
    """When a piece of information about a match becomes knowable.

    * PRE_DRAFT: known before the first draft action (identity, scheduling,
      competition/series context, rosters, patch).
    * DRAFT: revealed progressively during the draft. The complete
      `CanonicalMatch.draft_events` sequence is DRAFT information. A future
      live-draft consumer at draft step `t` must expose only the prefix
      `draft_events[:t]`. The terminal state -- the full sequence -- is the
      post-draft prediction point. There is intentionally no separate
      POST_DRAFT class: post-draft is simply the terminal state of DRAFT
      information.
    * POST_MATCH: known only once the game has concluded (winner, duration).
    """

    PRE_DRAFT = "PRE_DRAFT"
    DRAFT = "DRAFT"
    POST_MATCH = "POST_MATCH"


class Side(str, Enum):
    """A team side in one game. Source-native Radiant/Dire orientation."""

    RADIANT = "RADIANT"
    DIRE = "DIRE"


class DraftAction(str, Enum):
    """The type of a single draft event."""

    PICK = "PICK"
    BAN = "BAN"


class MatchPlayerPosition(str, Enum):
    """STRATZ `MatchPlayerPositionType` on a match player object.

    `POSITION_1`–`POSITION_5` are the Dota positions. `UNKNOWN` is an
    explicit STRATZ value, distinct from a missing/null field.
    `FILTERED` and `ALL` are members of the same GraphQL enum (typically
    query sentinels); they are preserved if STRATZ returns them on a
    player row and are never coerced into a 1–5 position.
    """

    POSITION_1 = "POSITION_1"
    POSITION_2 = "POSITION_2"
    POSITION_3 = "POSITION_3"
    POSITION_4 = "POSITION_4"
    POSITION_5 = "POSITION_5"
    UNKNOWN = "UNKNOWN"
    FILTERED = "FILTERED"
    ALL = "ALL"


class MatchLane(str, Enum):
    """STRATZ `MatchLaneType` on a match player object."""

    ROAMING = "ROAMING"
    SAFE_LANE = "SAFE_LANE"
    MID_LANE = "MID_LANE"
    OFF_LANE = "OFF_LANE"
    JUNGLE = "JUNGLE"
    UNKNOWN = "UNKNOWN"


class MatchPlayerRole(str, Enum):
    """STRATZ `MatchPlayerRoleType` on a match player object."""

    CORE = "CORE"
    LIGHT_SUPPORT = "LIGHT_SUPPORT"
    HARD_SUPPORT = "HARD_SUPPORT"
    UNKNOWN = "UNKNOWN"


# STRATZ `MatchPlayerType` scalar name -> canonical/storage column.
# Raw JSON keeps the STRATZ names; Postgres/Parquet use the snake_case
# names. These are observed post-match box-score values, not features.
PLAYER_BOX_SCORE_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("kills", "kills"),
    ("deaths", "deaths"),
    ("assists", "assists"),
    ("goldPerMinute", "gold_per_minute"),
    ("experiencePerMinute", "experience_per_minute"),
    ("numLastHits", "num_last_hits"),
    ("numDenies", "num_denies"),
    ("networth", "networth"),
    ("heroDamage", "hero_damage"),
    ("towerDamage", "tower_damage"),
    ("heroHealing", "hero_healing"),
    ("level", "level"),
)
STRATZ_PLAYER_BOX_SCORE_FIELDS: tuple[str, ...] = tuple(
    stratz_name for stratz_name, _ in PLAYER_BOX_SCORE_FIELD_MAP
)
MATCH_PLAYER_BOX_SCORE_COLUMNS: tuple[str, ...] = tuple(
    canonical_name for _, canonical_name in PLAYER_BOX_SCORE_FIELD_MAP
)


@dataclass(frozen=True, slots=True)
class MatchPlayerBoxScore:
    """Raw observed post-match player statistics for one lobby slot.

    Null means the source omitted the field. Zero is a real observation
    and is never rewritten as null, nor is null rewritten as zero.
    Values are not ratios, residuals, or model features.
    """

    kills: int | None = None
    deaths: int | None = None
    assists: int | None = None
    gold_per_minute: int | None = None
    experience_per_minute: int | None = None
    num_last_hits: int | None = None
    num_denies: int | None = None
    networth: int | None = None
    hero_damage: int | None = None
    tower_damage: int | None = None
    hero_healing: int | None = None
    level: int | None = None

    def __post_init__(self) -> None:
        for name in MATCH_PLAYER_BOX_SCORE_COLUMNS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise CanonicalMatchError(
                    f"MatchPlayerBoxScore.{name} must be int or None, got {value!r}"
                )


# Dota 1–5 only. `UNKNOWN` / `FILTERED` / `ALL` / NULL are not positions.
EXPLICIT_DOTA_POSITIONS: frozenset[MatchPlayerPosition] = frozenset(
    {
        MatchPlayerPosition.POSITION_1,
        MatchPlayerPosition.POSITION_2,
        MatchPlayerPosition.POSITION_3,
        MatchPlayerPosition.POSITION_4,
        MatchPlayerPosition.POSITION_5,
    }
)

_UNSET_SIDE_POSITIONS: tuple[
    MatchPlayerPosition | None,
    MatchPlayerPosition | None,
    MatchPlayerPosition | None,
    MatchPlayerPosition | None,
    MatchPlayerPosition | None,
] = (None, None, None, None, None)
_UNSET_SIDE_LANES: tuple[
    MatchLane | None,
    MatchLane | None,
    MatchLane | None,
    MatchLane | None,
    MatchLane | None,
] = (None, None, None, None, None)
_UNSET_SIDE_ROLES: tuple[
    MatchPlayerRole | None,
    MatchPlayerRole | None,
    MatchPlayerRole | None,
    MatchPlayerRole | None,
    MatchPlayerRole | None,
] = (None, None, None, None, None)
_EMPTY_BOX_SCORE = MatchPlayerBoxScore()
_UNSET_SIDE_BOX_SCORES: tuple[
    MatchPlayerBoxScore,
    MatchPlayerBoxScore,
    MatchPlayerBoxScore,
    MatchPlayerBoxScore,
    MatchPlayerBoxScore,
] = (
    _EMPTY_BOX_SCORE,
    _EMPTY_BOX_SCORE,
    _EMPTY_BOX_SCORE,
    _EMPTY_BOX_SCORE,
    _EMPTY_BOX_SCORE,
)


# Documentation-as-code: which `CanonicalMatch` field maps to which
# information-availability class. This is not enforced at construction time
# -- `CanonicalMatch` legitimately stores POST_MATCH/DRAFT fields alongside
# PRE_DRAFT ones, by design (see module docstring). Later feature-building
# code should consult this mapping (or reimplement the equivalent policy)
# when deciding what is safe to expose at a given prediction boundary.
FIELD_INFORMATION_AVAILABILITY: dict[str, InformationAvailability] = {
    "match_id": InformationAvailability.PRE_DRAFT,
    "start_time": InformationAvailability.PRE_DRAFT,
    "league_id": InformationAvailability.PRE_DRAFT,
    "league_name": InformationAvailability.PRE_DRAFT,
    "series_id": InformationAvailability.PRE_DRAFT,
    "series_type": InformationAvailability.PRE_DRAFT,
    "game_number_in_series": InformationAvailability.PRE_DRAFT,
    "game_version_id": InformationAvailability.PRE_DRAFT,
    "radiant_team_id": InformationAvailability.PRE_DRAFT,
    "radiant_team_name_observed": InformationAvailability.PRE_DRAFT,
    "radiant_player_ids": InformationAvailability.PRE_DRAFT,
    "dire_team_id": InformationAvailability.PRE_DRAFT,
    "dire_team_name_observed": InformationAvailability.PRE_DRAFT,
    "dire_player_ids": InformationAvailability.PRE_DRAFT,
    # Played heroes are known once the draft is complete. They are a
    # per-player fact (`players[].heroId`), not inferred from pick order
    # or lobby slot. Distinct from PRE_DRAFT roster identity.
    "radiant_hero_ids": InformationAvailability.DRAFT,
    "dire_hero_ids": InformationAvailability.DRAFT,
    "draft_events": InformationAvailability.DRAFT,
    "radiant_win": InformationAvailability.POST_MATCH,
    "duration_seconds": InformationAvailability.POST_MATCH,
    # Observed STRATZ replay-parse labels on the player object for THIS
    # match. They must not be used as PRE_DRAFT/POST_DRAFT features of
    # the same match. Historical rows may contribute to later state only
    # when `H.start_time < M.start_time`.
    "radiant_positions": InformationAvailability.POST_MATCH,
    "radiant_lanes": InformationAvailability.POST_MATCH,
    "radiant_roles": InformationAvailability.POST_MATCH,
    "dire_positions": InformationAvailability.POST_MATCH,
    "dire_lanes": InformationAvailability.POST_MATCH,
    "dire_roles": InformationAvailability.POST_MATCH,
    # Observed STRATZ post-match box-score scalars for THIS match.
    # Historical rows may contribute to later research only when
    # `H.start_time < M.start_time`. They are never PRE_DRAFT or
    # POST_DRAFT features of the same match.
    "radiant_box_scores": InformationAvailability.POST_MATCH,
    "dire_box_scores": InformationAvailability.POST_MATCH,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class DraftEvent:
    """One action (pick or ban) within a game's draft, in occurrence order.

    Attributes:
        sequence: Zero-based position of this event within the normalized
            canonical draft. This is a canonical position assigned by
            enumerating events after sorting and de-duplicating by
            whatever ordering field a source provides -- it is NOT
            required to equal any raw source ordering value verbatim.
            See `stratz_mapping.canonical_match_from_stratz` for how this
            is derived from STRATZ's `order` field.
        action: Whether this event was a pick or a ban.
        side: The side that performed the action.
        hero_id: The hero targeted by the action.
        was_successful: For bans, whether the ban actually took effect
            (STRATZ's `wasBannedSuccessfully`). `None` when unknown/not
            applicable (always the case for picks). A ban explicitly marked
            unsuccessful did not remove a hero from the pool, so it is
            excluded from the "actual action" hero-uniqueness check.
    """

    sequence: int
    action: DraftAction
    side: Side
    hero_id: HeroId
    was_successful: bool | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise CanonicalMatchError(
                f"draft event sequence must be >= 0, got {self.sequence}"
            )
        if not isinstance(self.action, DraftAction):
            raise CanonicalMatchError(f"unsupported draft action type: {self.action!r}")
        if not isinstance(self.side, Side):
            raise CanonicalMatchError(f"invalid acting side/team value: {self.side!r}")
        if self.hero_id <= 0:
            raise CanonicalMatchError(
                f"hero_id must be a positive integer, got {self.hero_id}"
            )

    @property
    def is_actual(self) -> bool:
        """Whether this event actually changed draft state.

        Picks always count. Bans count unless explicitly marked as a failed
        attempt (`was_successful is False`); an unknown/unset value for a
        ban is conservatively treated as actual.
        """
        if self.action is DraftAction.PICK:
            return True
        return self.was_successful is not False


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise CanonicalMatchError(f"{name} must be a positive integer, got {value}")


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalMatch:
    """Canonical historical record of one professional Dota 2 game.

    One instance represents one GAME, not one series. Series context
    (`series_id`, `series_type`, `game_number_in_series`) identifies which
    series a game belongs to and, together with `start_time`, preserves
    enough ordering information for a later feature-building layer to
    reconstruct series score before this game -- that derived value is
    intentionally not stored here.
    """

    # --- Identity / timing (PRE_DRAFT) ---
    match_id: MatchId
    start_time: datetime

    # --- Competition (PRE_DRAFT) ---
    league_id: LeagueId
    league_name: str | None = None

    # --- Series (PRE_DRAFT) ---
    series_id: SeriesId | None = None
    series_type: str | None = None
    game_number_in_series: int | None = None

    # --- Game environment (PRE_DRAFT) ---
    # This is the raw source game-version identifier (STRATZ
    # `gameVersionId`), preserved verbatim. It is an opaque, source-native
    # foreign key at hotfix granularity (e.g. 7.22c and 7.22f are
    # different ids), NOT a human-readable patch string. A separate
    # id -> human-readable-version lookup (STRATZ `constants.gameVersions`)
    # exists but is intentionally not joined in here; that belongs to a
    # future ingestion-layer lookup table, not this single-record mapping.
    game_version_id: int | None = None

    # --- Radiant (PRE_DRAFT) ---
    radiant_team_id: TeamId
    # What the source reported as this team's name for THIS match, at the
    # time it was observed -- not this team's current/best-known display
    # name. `team_id` is the stable join key across matches; this field is
    # an immutable per-match fact and must never be overwritten to reflect
    # a "latest known" name (that is an entity-level, derived concern for
    # a future team registry/dataset-build layer, not this dataclass).
    radiant_team_name_observed: str | None = None
    radiant_player_ids: tuple[PlayerId, PlayerId, PlayerId, PlayerId, PlayerId]
    # Played hero per Radiant slot, aligned with `radiant_player_ids`
    # (same `slot_in_side` index). Source is STRATZ `players[].heroId`,
    # not draft pick order and not lobby slot. `slot_in_side` is lobby
    # order and is not Dota position 1-5.
    radiant_hero_ids: tuple[HeroId, HeroId, HeroId, HeroId, HeroId]
    # Observed STRATZ `players[].position` / `lane` / `role` for this
    # match, aligned with `radiant_player_ids` by `slot_in_side`.
    # POST_MATCH parse labels. Missing and `UNKNOWN` are stored as-is
    # and are never inferred from `slot_in_side`, lane, role, or farm.
    radiant_positions: tuple[
        MatchPlayerPosition | None,
        MatchPlayerPosition | None,
        MatchPlayerPosition | None,
        MatchPlayerPosition | None,
        MatchPlayerPosition | None,
    ] = _UNSET_SIDE_POSITIONS
    radiant_lanes: tuple[
        MatchLane | None,
        MatchLane | None,
        MatchLane | None,
        MatchLane | None,
        MatchLane | None,
    ] = _UNSET_SIDE_LANES
    radiant_roles: tuple[
        MatchPlayerRole | None,
        MatchPlayerRole | None,
        MatchPlayerRole | None,
        MatchPlayerRole | None,
        MatchPlayerRole | None,
    ] = _UNSET_SIDE_ROLES
    # Observed STRATZ post-match box-score scalars, aligned with
    # `radiant_player_ids` by `slot_in_side`. POST_MATCH. Missing stays
    # None; zero is preserved. Never inferred or ratioed here.
    radiant_box_scores: tuple[
        MatchPlayerBoxScore,
        MatchPlayerBoxScore,
        MatchPlayerBoxScore,
        MatchPlayerBoxScore,
        MatchPlayerBoxScore,
    ] = _UNSET_SIDE_BOX_SCORES

    # --- Dire (PRE_DRAFT) ---
    dire_team_id: TeamId
    dire_team_name_observed: str | None = None
    dire_player_ids: tuple[PlayerId, PlayerId, PlayerId, PlayerId, PlayerId]
    dire_hero_ids: tuple[HeroId, HeroId, HeroId, HeroId, HeroId]
    dire_positions: tuple[
        MatchPlayerPosition | None,
        MatchPlayerPosition | None,
        MatchPlayerPosition | None,
        MatchPlayerPosition | None,
        MatchPlayerPosition | None,
    ] = _UNSET_SIDE_POSITIONS
    dire_lanes: tuple[
        MatchLane | None,
        MatchLane | None,
        MatchLane | None,
        MatchLane | None,
        MatchLane | None,
    ] = _UNSET_SIDE_LANES
    dire_roles: tuple[
        MatchPlayerRole | None,
        MatchPlayerRole | None,
        MatchPlayerRole | None,
        MatchPlayerRole | None,
        MatchPlayerRole | None,
    ] = _UNSET_SIDE_ROLES
    dire_box_scores: tuple[
        MatchPlayerBoxScore,
        MatchPlayerBoxScore,
        MatchPlayerBoxScore,
        MatchPlayerBoxScore,
        MatchPlayerBoxScore,
    ] = _UNSET_SIDE_BOX_SCORES

    # --- Draft (DRAFT) ---
    draft_events: tuple[DraftEvent, ...]

    # --- Outcome (POST_MATCH) ---
    radiant_win: bool
    duration_seconds: int

    def __post_init__(self) -> None:
        _require_positive("match_id", self.match_id)
        _require_positive("league_id", self.league_id)
        if self.series_id is not None:
            _require_positive("series_id", self.series_id)
        if self.game_number_in_series is not None and self.game_number_in_series < 1:
            raise CanonicalMatchError(
                f"game_number_in_series must be >= 1, got {self.game_number_in_series}"
            )
        if self.game_version_id is not None:
            _require_positive("game_version_id", self.game_version_id)

        if self.start_time.tzinfo is None or self.start_time.utcoffset() != timedelta(
            0
        ):
            raise CanonicalMatchError("start_time must be an explicit UTC datetime")

        _require_positive("radiant_team_id", self.radiant_team_id)
        _require_positive("dire_team_id", self.dire_team_id)
        if self.radiant_team_id == self.dire_team_id:
            raise CanonicalMatchError("radiant_team_id and dire_team_id must differ")

        self._validate_side_players("radiant", self.radiant_player_ids)
        self._validate_side_players("dire", self.dire_player_ids)
        if set(self.radiant_player_ids) & set(self.dire_player_ids):
            raise CanonicalMatchError(
                "a player id appears on both radiant and dire sides"
            )

        self._validate_draft_events()
        self._validate_side_heroes("radiant", self.radiant_hero_ids)
        self._validate_side_heroes("dire", self.dire_hero_ids)
        self._validate_side_enum_tuple(
            "radiant_positions", self.radiant_positions, MatchPlayerPosition
        )
        self._validate_side_enum_tuple(
            "dire_positions", self.dire_positions, MatchPlayerPosition
        )
        self._validate_side_enum_tuple("radiant_lanes", self.radiant_lanes, MatchLane)
        self._validate_side_enum_tuple("dire_lanes", self.dire_lanes, MatchLane)
        self._validate_side_enum_tuple(
            "radiant_roles", self.radiant_roles, MatchPlayerRole
        )
        self._validate_side_enum_tuple("dire_roles", self.dire_roles, MatchPlayerRole)
        self._validate_side_box_scores("radiant_box_scores", self.radiant_box_scores)
        self._validate_side_box_scores("dire_box_scores", self.dire_box_scores)

        if self.duration_seconds <= 0:
            raise CanonicalMatchError(
                f"duration_seconds must be a positive integer, got {self.duration_seconds}"
            )

    @staticmethod
    def _validate_side_players(
        side_name: str, player_ids: tuple[PlayerId, ...]
    ) -> None:
        if len(player_ids) != 5:
            raise CanonicalMatchError(
                f"{side_name}_player_ids must contain exactly 5 players, got {len(player_ids)}"
            )
        for player_id in player_ids:
            _require_positive(f"{side_name}_player_ids entry", player_id)
        if len(set(player_ids)) != len(player_ids):
            raise CanonicalMatchError(
                f"{side_name}_player_ids contains duplicate player ids"
            )

    @staticmethod
    def _validate_side_enum_tuple(
        field_name: str,
        values: tuple[object, ...],
        enum_cls: type[Enum],
    ) -> None:
        # Length must match the five lobby slots. Duplicate, missing, and
        # UNKNOWN values are allowed: they are observed-source anomalies,
        # not canonicalization failures, and must not be repaired here.
        if len(values) != 5:
            raise CanonicalMatchError(
                f"{field_name} must contain exactly 5 entries, got {len(values)}"
            )
        for value in values:
            if value is not None and not isinstance(value, enum_cls):
                raise CanonicalMatchError(
                    f"{field_name} entry must be {enum_cls.__name__} or None, "
                    f"got {value!r}"
                )

    @staticmethod
    def _validate_side_box_scores(
        field_name: str, values: tuple[MatchPlayerBoxScore, ...]
    ) -> None:
        if len(values) != 5:
            raise CanonicalMatchError(
                f"{field_name} must contain exactly 5 entries, got {len(values)}"
            )
        for value in values:
            if not isinstance(value, MatchPlayerBoxScore):
                raise CanonicalMatchError(
                    f"{field_name} entry must be MatchPlayerBoxScore, got {value!r}"
                )

    def _validate_side_heroes(
        self, side_name: str, hero_ids: tuple[HeroId, ...]
    ) -> None:
        if len(hero_ids) != 5:
            raise CanonicalMatchError(
                f"{side_name}_hero_ids must contain exactly 5 heroes, got {len(hero_ids)}"
            )
        for hero_id in hero_ids:
            _require_positive(f"{side_name}_hero_ids entry", hero_id)
        if len(set(hero_ids)) != len(hero_ids):
            raise CanonicalMatchError(
                f"{side_name}_hero_ids contains duplicate hero ids"
            )
        side = Side.RADIANT if side_name == "radiant" else Side.DIRE
        pick_hero_ids = set(self._final_hero_ids(side))
        if set(hero_ids) != pick_hero_ids:
            raise CanonicalMatchError(
                f"{side_name} player hero_id set does not match successful PICK set"
            )

    def _validate_draft_events(self) -> None:
        # Enforces a deterministic, gap-free, zero-indexed ordering without
        # assuming any particular total event count (draft formats vary
        # historically). Also catches duplicate/out-of-order `sequence`
        # values, which would indicate malformed or non-deterministic
        # source ordering.
        for index, event in enumerate(self.draft_events):
            if event.sequence != index:
                raise CanonicalMatchError(
                    "draft_events must be ordered with sequence == position; "
                    f"expected sequence {index} at position {index}, got {event.sequence}"
                )

        actual_hero_ids = [
            event.hero_id for event in self.draft_events if event.is_actual
        ]
        duplicates = sorted(
            hero_id for hero_id, count in Counter(actual_hero_ids).items() if count > 1
        )
        if duplicates:
            raise CanonicalMatchError(
                f"hero id(s) {duplicates} appear in more than one actual draft action"
            )

        # Exactly five heroes picked per side is a fixed rule of the game
        # (5v5), not an assumption about draft-format phase structure, so
        # it is safe to validate regardless of historical ban-count
        # variation. This held for every completed Tier 1/Tier 2 match in
        # a 265-match verification sample spanning 2019-2025. It currently
        # represents the schema for completed, training-eligible
        # professional games; a future ingestion layer may need a
        # separate way to represent legitimate-but-incomplete historical
        # records (abandons, remakes, etc.) that fail this invariant --
        # that distinction is not implemented here.
        radiant_picks = len(self.radiant_final_hero_ids)
        if radiant_picks != 5:
            raise CanonicalMatchError(
                f"expected exactly 5 actual radiant picks, got {radiant_picks}"
            )
        dire_picks = len(self.dire_final_hero_ids)
        if dire_picks != 5:
            raise CanonicalMatchError(
                f"expected exactly 5 actual dire picks, got {dire_picks}"
            )

    def _final_hero_ids(self, side: Side) -> tuple[HeroId, ...]:
        return tuple(
            event.hero_id
            for event in self.draft_events
            if event.side is side
            and event.action is DraftAction.PICK
            and event.is_actual
        )

    @property
    def radiant_final_hero_ids(self) -> tuple[HeroId, ...]:
        """The five heroes Radiant ended the draft with, in pick order.

        Derived from `draft_events` rather than stored separately, to avoid
        duplicating information that the ordered draft already encodes.
        """
        return self._final_hero_ids(Side.RADIANT)

    @property
    def dire_final_hero_ids(self) -> tuple[HeroId, ...]:
        """The five heroes Dire ended the draft with, in pick order."""
        return self._final_hero_ids(Side.DIRE)
