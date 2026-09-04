"""Single-record mapping from raw STRATZ match data to `CanonicalMatch`.

This module translates ONE raw STRATZ GraphQL `MatchType` payload (the shape
produced by `scripts/probe_stratz_graphql.py`, see
`data/raw/stratz_probe_matches.json`) into a `CanonicalMatch`. It does not
call the STRATZ API itself and performs no bulk ingestion or backfill --
wiring this into an actual ingestion pipeline is future work.

Known STRATZ source-mapping caveats (see task reports for full detail;
verified against a 265-match Tier 1/Tier 2 sample, not just the small
committed probe file):

* `durationSeconds` is reliably populated on completed professional
  matches and is required here (POST_MATCH information is always
  knowable once a match has a result). `gameVersionId` is optional,
  preserved verbatim as `game_version_id`, and passed through if present.
  It is a STRATZ-internal, hotfix-granularity opaque id, NOT a
  human-readable patch string (e.g. "7.31c") -- resolving it to a
  human-readable version via STRATZ's `constants.gameVersions` lookup is
  intentionally left to a future ingestion-layer lookup table, not this
  single-record mapping.
* For pick/ban rows, STRATZ populates `heroId` for both picks and bans in
  most samples (duplicating `bannedHeroId` for bans), but `heroId` is
  `null` on a meaningful minority of real ban rows (~15% in the
  verification sample) with `bannedHeroId` populated instead. `heroId`
  and `bannedHeroId` were never observed both non-null and disagreeing.
  `bannedHeroId` is used only as a fallback if `heroId` is absent.
* `playerIndex`, `isCaptain`, and `letter` on pick/ban rows are not mapped
  here: their real-world population is patchy and inconsistent across
  tournaments/eras, their semantics are not confirmed from documentation
  or samples alone, and they are not needed to reconstruct draft order or
  side. See the task reports for details.
* STRATZ does not expose a distinct "side selection" / "first pick"
  field separate from the draft sequence itself on the match/league query
  surface used here (a `isRadiantFirstPick` field exists only on an
  unrelated replay-upload type). The acting side of `draft_events[0]` is
  the only available proxy for this and is preserved implicitly by
  preserving the full ordered draft; no separate field is invented here.
* Canonical `DraftEvent.sequence` is a normalized canonical position, not
  a copy of STRATZ's raw `order` value: rows are sorted by `order` (after
  requiring `order` to be present and pairwise distinct, since duplicate
  `order` values make source ordering ambiguous), then `sequence` is
  assigned by enumerating the sorted rows from 0. In every real sample
  observed so far STRATZ's `order` already happens to be zero-based and
  gap-free, but the mapper does not depend on that; it only depends on
  `order` producing an unambiguous sort.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from dota_predictor.data.canonical_schema import (
    PLAYER_BOX_SCORE_FIELD_MAP,
    CanonicalMatch,
    CanonicalMatchError,
    DraftAction,
    DraftEvent,
    HeroId,
    MatchLane,
    MatchPlayerBoxScore,
    MatchPlayerPosition,
    MatchPlayerRole,
    PlayerId,
    Side,
)

__all__ = [
    "CANONICAL_MAPPER_VERSION",
    "canonical_match_from_stratz",
    "draft_event_from_stratz_pick_ban",
]

# Monotonic version of this module's mapping *logic*, not the package
# version. Bump this whenever a change to `canonical_match_from_stratz` or
# `draft_event_from_stratz_pick_ban` would change the output for
# already-ingested raw payloads (e.g. a new fallback rule, a fixed bug in
# field selection). Persisted alongside each canonical row
# (`storage.schema.MATCHES.c.mapper_version`) so a future reprocessing job
# can select rows with `mapper_version < CANONICAL_MAPPER_VERSION` instead
# of reprocessing everything or nothing.
CANONICAL_MAPPER_VERSION = 4


def _require(raw: Mapping[str, Any], key: str, *, context: str) -> Any:
    value = raw.get(key)
    if value is None:
        raise CanonicalMatchError(f"{context}: missing required field '{key}'")
    return value


def _side_from_is_radiant(is_radiant: bool) -> Side:
    return Side.RADIANT if is_radiant else Side.DIRE


def draft_event_from_stratz_pick_ban(
    raw: Mapping[str, Any], *, sequence: int
) -> DraftEvent:
    """Map one STRATZ `MatchStatsPickBanType` row to a `DraftEvent`.

    `sequence` is the row's position in the *canonical* draft sequence, as
    assigned by `canonical_match_from_stratz` after sorting and validating
    raw rows by STRATZ's `order` field (see `_sorted_pick_ban_rows`). It is
    a caller-supplied canonical position, not derived from the raw row
    itself: canonical `sequence` means "position in the normalized draft",
    not "the literal STRATZ `order` value".
    """
    is_pick = raw.get("isPick")
    if is_pick is None:
        raise CanonicalMatchError(
            f"pick/ban row at sequence={sequence}: missing 'isPick'"
        )

    is_radiant = raw.get("isRadiant")
    if is_radiant is None:
        raise CanonicalMatchError(
            f"pick/ban row at sequence={sequence}: missing 'isRadiant'"
        )

    hero_id: HeroId | None = raw.get("heroId")
    if hero_id is None:
        hero_id = raw.get("bannedHeroId")
    if hero_id is None:
        raise CanonicalMatchError(
            f"pick/ban row at sequence={sequence}: missing both 'heroId' and 'bannedHeroId'"
        )

    action = DraftAction.PICK if is_pick else DraftAction.BAN
    was_successful = (
        raw.get("wasBannedSuccessfully") if action is DraftAction.BAN else None
    )

    return DraftEvent(
        sequence=sequence,
        action=action,
        side=_side_from_is_radiant(is_radiant),
        hero_id=hero_id,
        was_successful=was_successful,
    )


def _sorted_pick_ban_rows(
    pick_bans: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Sort raw STRATZ pick/ban rows by their source `order` field.

    This only establishes an unambiguous source ordering; it does NOT
    assign canonical `sequence` (that is done by the caller via
    `enumerate` over this function's result -- see
    `canonical_match_from_stratz`). Raw `order` is required to be present
    and pairwise distinct, but is intentionally NOT required to be
    zero-based or gap-free: STRATZ's `order` and canonical `sequence` are
    different things that happen to coincide numerically in every sample
    observed so far, and the mapper should not depend on that coincidence.
    """
    orders: list[int] = []
    for row in pick_bans:
        order = row.get("order")
        if order is None:
            raise CanonicalMatchError("pick/ban row: missing required field 'order'")
        orders.append(order)

    duplicates = sorted(order for order, count in Counter(orders).items() if count > 1)
    if duplicates:
        raise CanonicalMatchError(
            f"pick/ban rows: duplicate STRATZ 'order' value(s) {duplicates}; "
            "source draft ordering is ambiguous"
        )

    return sorted(pick_bans, key=lambda row: row["order"])


@dataclass(frozen=True, slots=True)
class _MappedSidePlayer:
    player_id: PlayerId
    hero_id: HeroId
    position: MatchPlayerPosition | None
    lane: MatchLane | None
    role: MatchPlayerRole | None
    box_score: MatchPlayerBoxScore


_EnumT = TypeVar("_EnumT", bound=Enum)


def _optional_int(value: Any, *, context: str) -> int | None:
    """Map a STRATZ scalar to int, preserving null and zero.

    Missing/null stays None. Zero is a real observation. Bool and
    non-integer values fail closed so schema drift is visible.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalMatchError(f"{context}: expected integer or null, got {value!r}")
    return int(value)


def _box_score_from_player(
    player: Mapping[str, Any], *, player_id: int
) -> MatchPlayerBoxScore:
    kwargs: dict[str, int | None] = {}
    for stratz_name, canonical_name in PLAYER_BOX_SCORE_FIELD_MAP:
        kwargs[canonical_name] = _optional_int(
            player.get(stratz_name),
            context=f"player {player_id} {stratz_name}",
        )
    return MatchPlayerBoxScore(**kwargs)


def _optional_enum(
    enum_cls: type[_EnumT], value: Any, *, context: str
) -> _EnumT | None:
    """Map a STRATZ enum string to `enum_cls`, preserving null and UNKNOWN.

    Missing/null stays None. `UNKNOWN` is a real source value, not None.
    Unexpected members fail closed so schema drift is visible.
    """
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise CanonicalMatchError(
            f"{context}: unsupported {enum_cls.__name__} value {value!r}"
        ) from exc


def _side_roster(
    players: Sequence[Mapping[str, Any]], *, is_radiant: bool
) -> tuple[_MappedSidePlayer, ...]:
    """Return mapped players for one side in lobby-slot order.

    Order follows STRATZ `playerSlot` (canonical `slot_in_side`), not
    draft pick order and not `pickBans.playerIndex`.     `heroId`,
    `position`, `lane`, `role`, and post-match box-score scalars are
    taken from the player row itself. `proSteamAccount` is ignored.
    Missing position/lane/role/box-score fields stay None and do not
    fail mapping.
    """
    side_players = sorted(
        (player for player in players if player.get("isRadiant") == is_radiant),
        key=lambda player: player.get("playerSlot") or 0,
    )
    roster: list[_MappedSidePlayer] = []
    for player in side_players:
        steam_account_id = player.get("steamAccountId")
        if steam_account_id is None:
            raise CanonicalMatchError(
                "player row: missing required field 'steamAccountId'"
            )
        hero_id = player.get("heroId")
        if hero_id is None:
            raise CanonicalMatchError("player row: missing required field 'heroId'")
        if int(hero_id) <= 0:
            raise CanonicalMatchError(
                f"player row: heroId must be a positive integer, got {hero_id}"
            )
        player_id = int(steam_account_id)
        roster.append(
            _MappedSidePlayer(
                player_id=player_id,
                hero_id=int(hero_id),
                position=_optional_enum(
                    MatchPlayerPosition,
                    player.get("position"),
                    context=f"player {player_id}",
                ),
                lane=_optional_enum(
                    MatchLane,
                    player.get("lane"),
                    context=f"player {player_id}",
                ),
                role=_optional_enum(
                    MatchPlayerRole,
                    player.get("role"),
                    context=f"player {player_id}",
                ),
                box_score=_box_score_from_player(player, player_id=player_id),
            )
        )
    return tuple(roster)


def canonical_match_from_stratz(raw: Mapping[str, Any]) -> CanonicalMatch:
    """Build a `CanonicalMatch` from one raw STRATZ `MatchType` payload."""
    match_id = _require(raw, "id", context="match")
    start_unix = _require(raw, "startDateTime", context="match")
    start_time = datetime.fromtimestamp(start_unix, tz=UTC)

    league = raw.get("league") or {}
    league_id = raw.get("leagueId") or league.get("id")
    if league_id is None:
        raise CanonicalMatchError("match: missing required field 'leagueId'")
    league_name = league.get("displayName") or league.get("name")

    series = raw.get("series") or {}
    series_id = raw.get("seriesId") or series.get("id")
    series_type = series.get("type")
    # STRATZ does not expose a direct "game number within series" field on
    # MatchType; deriving it from series.matches ordering is future
    # ingestion-layer work, out of scope here.
    game_number_in_series = None

    radiant_team = raw.get("radiantTeam") or {}
    dire_team = raw.get("direTeam") or {}
    radiant_team_id = raw.get("radiantTeamId") or radiant_team.get("id")
    dire_team_id = raw.get("direTeamId") or dire_team.get("id")
    if radiant_team_id is None:
        raise CanonicalMatchError("match: missing required field 'radiantTeamId'")
    if dire_team_id is None:
        raise CanonicalMatchError("match: missing required field 'direTeamId'")

    players = raw.get("players") or []
    radiant_roster = _side_roster(players, is_radiant=True)
    dire_roster = _side_roster(players, is_radiant=False)
    radiant_player_ids = tuple(player.player_id for player in radiant_roster)
    dire_player_ids = tuple(player.player_id for player in dire_roster)
    radiant_hero_ids = tuple(player.hero_id for player in radiant_roster)
    dire_hero_ids = tuple(player.hero_id for player in dire_roster)
    radiant_positions = tuple(player.position for player in radiant_roster)
    dire_positions = tuple(player.position for player in dire_roster)
    radiant_lanes = tuple(player.lane for player in radiant_roster)
    dire_lanes = tuple(player.lane for player in dire_roster)
    radiant_roles = tuple(player.role for player in radiant_roster)
    dire_roles = tuple(player.role for player in dire_roster)
    radiant_box_scores = tuple(player.box_score for player in radiant_roster)
    dire_box_scores = tuple(player.box_score for player in dire_roster)

    pick_bans = raw.get("pickBans")
    draft_events: tuple[DraftEvent, ...]
    draft_complete: bool
    if not pick_bans:
        # No source draft rows at all: represent the match canonically
        # with an absent draft rather than dropping it. Draft data is
        # never fabricated.
        draft_events = ()
        draft_complete = False
    else:
        try:
            draft_events = tuple(
                draft_event_from_stratz_pick_ban(row, sequence=sequence)
                for sequence, row in enumerate(_sorted_pick_ban_rows(pick_bans))
            )
            draft_complete = True
        except CanonicalMatchError:
            # The source draft is present but malformed (missing/duplicate
            # ordering, missing fields). A professional match should not
            # disappear from the canonical census because one optional
            # analytical component is unavailable: record it with an
            # absent draft and let draft-dependent consumers exclude it.
            draft_events = ()
            draft_complete = False

    duration_seconds = _require(raw, "durationSeconds", context="match")

    radiant_win = raw.get("didRadiantWin")
    if radiant_win is None:
        raise CanonicalMatchError("match: missing required field 'didRadiantWin'")

    try:
        return CanonicalMatch(
            match_id=match_id,
            start_time=start_time,
            league_id=league_id,
            league_name=league_name,
            series_id=series_id,
            series_type=series_type,
            game_number_in_series=game_number_in_series,
            game_version_id=raw.get("gameVersionId"),
            radiant_team_id=radiant_team_id,
            radiant_team_name_observed=radiant_team.get("name"),
            radiant_player_ids=radiant_player_ids,
            dire_team_id=dire_team_id,
            dire_team_name_observed=dire_team.get("name"),
            dire_player_ids=dire_player_ids,
            radiant_hero_ids=radiant_hero_ids,
            dire_hero_ids=dire_hero_ids,
            radiant_positions=radiant_positions,
            dire_positions=dire_positions,
            radiant_lanes=radiant_lanes,
            dire_lanes=dire_lanes,
            radiant_roles=radiant_roles,
            dire_roles=dire_roles,
            radiant_box_scores=radiant_box_scores,
            dire_box_scores=dire_box_scores,
            draft_events=draft_events,
            draft_complete=draft_complete,
            radiant_win=radiant_win,
            duration_seconds=duration_seconds,
        )
    except CanonicalMatchError:
        if not draft_complete:
            raise
        # The draft built from the source rows fails a draft-completeness
        # invariant (e.g. a partial draft with fewer than five picks per
        # side). The match itself is a real completed professional game, so
        # retry with the draft represented as absent. Identity/team/player
        # validations run again in this retry, so any genuine identity
        # failure still surfaces rather than being masked by the absent
        # draft.
        return CanonicalMatch(
            match_id=match_id,
            start_time=start_time,
            league_id=league_id,
            league_name=league_name,
            series_id=series_id,
            series_type=series_type,
            game_number_in_series=game_number_in_series,
            game_version_id=raw.get("gameVersionId"),
            radiant_team_id=radiant_team_id,
            radiant_team_name_observed=radiant_team.get("name"),
            radiant_player_ids=radiant_player_ids,
            dire_team_id=dire_team_id,
            dire_team_name_observed=dire_team.get("name"),
            dire_player_ids=dire_player_ids,
            radiant_hero_ids=radiant_hero_ids,
            dire_hero_ids=dire_hero_ids,
            radiant_positions=radiant_positions,
            dire_positions=dire_positions,
            radiant_lanes=radiant_lanes,
            dire_lanes=dire_lanes,
            radiant_roles=radiant_roles,
            dire_roles=dire_roles,
            radiant_box_scores=radiant_box_scores,
            dire_box_scores=dire_box_scores,
            draft_events=(),
            draft_complete=False,
            radiant_win=radiant_win,
            duration_seconds=duration_seconds,
        )
