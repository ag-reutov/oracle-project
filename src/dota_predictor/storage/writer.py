"""Write a `CanonicalMatch` to Postgres.

Implements the write pattern decided for canonical child tables: `matches`
is upserted (one row per match, safe to update in place), while
`match_players`/`draft_events` are fully replaced (delete then insert) in
the same transaction. See `storage.schema` module docstring for why plain
upsert is insufficient for those two child tables.

Before writing `matches`/`match_players`, this module ensures every
referenced `team_id`/`player_id` has a row in the `teams`/`players`
identity registries, via a plain `INSERT ... ON CONFLICT DO NOTHING`. This
is intentionally NOT an upsert with derived/update logic: `teams`/`players`
only need to guarantee referenced ids exist (see `storage.schema` module
docstring) -- there is nothing to update once a row exists.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy import Connection, Engine
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dota_predictor.data.canonical_schema import CanonicalMatch, PlayerId, Side, TeamId
from dota_predictor.data.stratz_mapping import CANONICAL_MAPPER_VERSION
from dota_predictor.storage.schema import (
    DRAFT_EVENTS,
    MATCH_PLAYERS,
    MATCHES,
    PLAYERS,
    TEAMS,
)

__all__ = ["write_canonical_match"]


def _ensure_teams_exist(conn: Connection, team_ids: set[TeamId]) -> None:
    if not team_ids:
        return
    stmt = pg_insert(TEAMS).values([{"team_id": tid} for tid in team_ids])
    conn.execute(stmt.on_conflict_do_nothing(index_elements=[TEAMS.c.team_id]))


def _ensure_players_exist(conn: Connection, player_ids: set[PlayerId]) -> None:
    if not player_ids:
        return
    stmt = pg_insert(PLAYERS).values([{"player_id": pid} for pid in player_ids])
    conn.execute(stmt.on_conflict_do_nothing(index_elements=[PLAYERS.c.player_id]))


def write_canonical_match(engine: Engine, match: CanonicalMatch) -> None:
    """Write one `CanonicalMatch`, replacing any prior state for it.

    The caller is responsible for having already ensured `match.league_id`
    is present in `ingestion_leagues` (e.g. via the fetch loop only ever
    pulling matches for allowlisted leagues). This function does not
    silently skip out-of-scope matches -- an out-of-scope `league_id`
    surfaces loudly as a foreign key violation, since that indicates a bug
    upstream (a match slipped through scope gating), not a case to handle
    quietly here.
    """
    canonicalized_at = datetime.now(UTC)

    match_values = {
        "league_id": match.league_id,
        "start_time": match.start_time,
        "league_name": match.league_name,
        "series_id": match.series_id,
        "series_type": match.series_type,
        "game_number_in_series": match.game_number_in_series,
        "game_version_id": match.game_version_id,
        "radiant_team_id": match.radiant_team_id,
        "radiant_team_name_observed": match.radiant_team_name_observed,
        "dire_team_id": match.dire_team_id,
        "dire_team_name_observed": match.dire_team_name_observed,
        "radiant_win": match.radiant_win,
        "duration_seconds": match.duration_seconds,
        "mapper_version": CANONICAL_MAPPER_VERSION,
        "canonicalized_at": canonicalized_at,
    }

    player_rows = [
        {
            "match_id": match.match_id,
            "side": side,
            "slot_in_side": slot,
            "player_id": player_id,
            "hero_id": hero_id,
            "position": position,
            "lane": lane,
            "role": role,
            **asdict(box_score),
        }
        for side, player_ids, hero_ids, positions, lanes, roles, box_scores in (
            (
                Side.RADIANT,
                match.radiant_player_ids,
                match.radiant_hero_ids,
                match.radiant_positions,
                match.radiant_lanes,
                match.radiant_roles,
                match.radiant_box_scores,
            ),
            (
                Side.DIRE,
                match.dire_player_ids,
                match.dire_hero_ids,
                match.dire_positions,
                match.dire_lanes,
                match.dire_roles,
                match.dire_box_scores,
            ),
        )
        for slot, (player_id, hero_id, position, lane, role, box_score) in enumerate(
            zip(
                player_ids,
                hero_ids,
                positions,
                lanes,
                roles,
                box_scores,
                strict=True,
            )
        )
    ]

    draft_rows = [
        {
            "match_id": match.match_id,
            "sequence": event.sequence,
            "action": event.action,
            "side": event.side,
            "hero_id": event.hero_id,
            "was_successful": event.was_successful,
        }
        for event in match.draft_events
    ]

    with engine.begin() as conn:
        _ensure_teams_exist(conn, {match.radiant_team_id, match.dire_team_id})
        _ensure_players_exist(
            conn, set(match.radiant_player_ids) | set(match.dire_player_ids)
        )

        upsert_matches = pg_insert(MATCHES).values(
            match_id=match.match_id, **match_values
        )
        upsert_matches = upsert_matches.on_conflict_do_update(
            index_elements=[MATCHES.c.match_id],
            set_=match_values,
        )
        conn.execute(upsert_matches)

        # Full replacement, not upsert: see module/schema docstrings for why.
        conn.execute(
            MATCH_PLAYERS.delete().where(MATCH_PLAYERS.c.match_id == match.match_id)
        )
        conn.execute(
            DRAFT_EVENTS.delete().where(DRAFT_EVENTS.c.match_id == match.match_id)
        )

        if player_rows:
            conn.execute(MATCH_PLAYERS.insert(), player_rows)
        if draft_rows:
            conn.execute(DRAFT_EVENTS.insert(), draft_rows)
