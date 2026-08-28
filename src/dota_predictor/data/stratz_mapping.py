"""Single-record mapping from raw STRATZ match data to `CanonicalMatch`.

This module translates ONE raw STRATZ GraphQL `MatchType` payload (the shape
produced by `scripts/probe_stratz_graphql.py`, see
`data/raw/stratz_probe_matches.json`) into a `CanonicalMatch`. It does not
call the STRATZ API itself and performs no bulk ingestion or backfill --
wiring this into an actual ingestion pipeline is future work.

Known STRATZ source-mapping caveats (see task report for full detail):

* `durationSeconds` and `gameVersionId` exist on STRATZ's `MatchType` per
  schema introspection, but were not part of the field selection used by
  the existing probe script, so their real-world population has not been
  verified against a live response. `durationSeconds` is required here
  (POST_MATCH information is always knowable once a match has a result);
  `gameVersionId` (mapped to `patch`) is optional and simply passed through
  if present.
* For pick/ban rows, STRATZ populates `heroId` for both picks and bans in
  observed samples (duplicating `bannedHeroId` for bans). `bannedHeroId` is
  used only as a fallback if `heroId` is absent.
* `playerIndex`, `isCaptain`, and `letter` on pick/ban rows are not mapped
  here: their semantics are not confirmed from documentation or samples
  alone, and they are not needed to reconstruct draft order or side. See
  the task report for details.
* STRATZ does not expose a distinct "side selection" / "first pick"
  field separate from the draft sequence itself. The acting side of
  `draft_events[0]` is the only available proxy for this and is preserved
  implicitly by preserving the full ordered draft; no separate field is
  invented here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from dota_predictor.data.canonical_schema import (
    CanonicalMatch,
    CanonicalMatchError,
    DraftAction,
    DraftEvent,
    HeroId,
    PlayerId,
    Side,
)

__all__ = ["canonical_match_from_stratz", "draft_event_from_stratz_pick_ban"]


def _require(raw: Mapping[str, Any], key: str, *, context: str) -> Any:
    value = raw.get(key)
    if value is None:
        raise CanonicalMatchError(f"{context}: missing required field '{key}'")
    return value


def _side_from_is_radiant(is_radiant: bool) -> Side:
    return Side.RADIANT if is_radiant else Side.DIRE


def draft_event_from_stratz_pick_ban(raw: Mapping[str, Any]) -> DraftEvent:
    """Map one STRATZ `MatchStatsPickBanType` row to a `DraftEvent`."""
    order = raw.get("order")
    if order is None:
        raise CanonicalMatchError("pick/ban row: missing required field 'order'")

    is_pick = raw.get("isPick")
    if is_pick is None:
        raise CanonicalMatchError(f"pick/ban row at order={order}: missing 'isPick'")

    is_radiant = raw.get("isRadiant")
    if is_radiant is None:
        raise CanonicalMatchError(f"pick/ban row at order={order}: missing 'isRadiant'")

    hero_id: HeroId | None = raw.get("heroId")
    if hero_id is None:
        hero_id = raw.get("bannedHeroId")
    if hero_id is None:
        raise CanonicalMatchError(
            f"pick/ban row at order={order}: missing both 'heroId' and 'bannedHeroId'"
        )

    action = DraftAction.PICK if is_pick else DraftAction.BAN
    was_successful = (
        raw.get("wasBannedSuccessfully") if action is DraftAction.BAN else None
    )

    return DraftEvent(
        sequence=order,
        action=action,
        side=_side_from_is_radiant(is_radiant),
        hero_id=hero_id,
        was_successful=was_successful,
    )


def _side_player_ids(
    players: Sequence[Mapping[str, Any]], *, is_radiant: bool
) -> tuple[PlayerId, ...]:
    side_players = sorted(
        (player for player in players if player.get("isRadiant") == is_radiant),
        key=lambda player: player.get("playerSlot") or 0,
    )
    player_ids: list[PlayerId] = []
    for player in side_players:
        steam_account_id = player.get("steamAccountId")
        if steam_account_id is None:
            raise CanonicalMatchError(
                "player row: missing required field 'steamAccountId'"
            )
        player_ids.append(steam_account_id)
    return tuple(player_ids)


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
    radiant_player_ids = _side_player_ids(players, is_radiant=True)
    dire_player_ids = _side_player_ids(players, is_radiant=False)

    pick_bans = raw.get("pickBans") or []
    draft_events = tuple(
        draft_event_from_stratz_pick_ban(row)
        for row in sorted(pick_bans, key=lambda row: row.get("order") or 0)
    )

    duration_seconds = _require(raw, "durationSeconds", context="match")

    radiant_win = raw.get("didRadiantWin")
    if radiant_win is None:
        raise CanonicalMatchError("match: missing required field 'didRadiantWin'")

    return CanonicalMatch(
        match_id=match_id,
        start_time=start_time,
        league_id=league_id,
        league_name=league_name,
        series_id=series_id,
        series_type=series_type,
        game_number_in_series=game_number_in_series,
        patch=raw.get("gameVersionId"),
        radiant_team_id=radiant_team_id,
        radiant_team_name=radiant_team.get("name"),
        radiant_player_ids=radiant_player_ids,
        dire_team_id=dire_team_id,
        dire_team_name=dire_team.get("name"),
        dire_player_ids=dire_player_ids,
        draft_events=draft_events,
        radiant_win=radiant_win,
        duration_seconds=duration_seconds,
    )
