"""Shared seeding helpers for research-layer tests.

Seeds the canonical base tables (`leagues`, `ingestion_leagues`, `teams`,
`players`, `matches`, `match_players`, `draft_events`,
`match_classifications`) the way the warehouse writes them, so the research
views can be tested against realistic relational state.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Connection
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dota_predictor.storage.schema import (
    DRAFT_EVENTS,
    INGESTION_LEAGUES,
    LEAGUES,
    MATCH_CLASSIFICATIONS,
    MATCH_PLAYERS,
    MATCHES,
    PLAYERS,
    TEAMS,
)

__all__ = [
    "DIRE_TEAM",
    "RADIANT_TEAM",
    "seed_draft_events",
    "seed_league",
    "seed_match",
]

RADIANT_TEAM = 8261500
DIRE_TEAM = 9247354

# 10 player ids per match; reuse a fresh offset per match to keep the
# `(match_id, player_id)` unique constraint satisfied.
PLAYER_BASE_IDS = [898754153, 137129583, 129958758, 157475523, 94296097]
DIRE_PLAYER_BASE_IDS = [10366616, 100058342, 898455820, 183719386, 25907144]


def seed_league(
    conn: Connection,
    league_id: int,
    *,
    name: str = "Test League",
    tier: str = "T1",
    in_scope: bool = True,
    window_filter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    conn.execute(
        LEAGUES.insert().values(
            league_id=league_id,
            name=name,
            liquipedia_tier=tier,
            in_scope=in_scope,
            fetch_mode="league",
            window_filter=window_filter,
            start_date=start_date,
            end_date=end_date,
        )
    )
    if in_scope:
        conn.execute(INGESTION_LEAGUES.insert().values(league_id=league_id))


def seed_match(
    conn: Connection,
    *,
    match_id: int,
    league_id: int,
    start_time: datetime,
    radiant_win: bool = True,
    draft_complete: bool = True,
    game_version_id: int = 178,
    duration_seconds: int = 2400,
    with_draft: bool = True,
) -> None:
    """Seed one canonical match + its 10 `match_players` rows.

    `with_draft` controls whether `draft_events` rows are also seeded (the
    research views do not validate draft internals; they only expose them).
    """
    conn.execute(pg_insert(TEAMS).values(team_id=RADIANT_TEAM).on_conflict_do_nothing())
    conn.execute(pg_insert(TEAMS).values(team_id=DIRE_TEAM).on_conflict_do_nothing())
    player_ids = PLAYER_BASE_IDS + DIRE_PLAYER_BASE_IDS
    for player_id in player_ids:
        conn.execute(
            pg_insert(PLAYERS).values(player_id=player_id).on_conflict_do_nothing()
        )

    conn.execute(
        MATCHES.insert().values(
            match_id=match_id,
            league_id=league_id,
            start_time=start_time,
            league_name="Test League",
            game_version_id=game_version_id,
            radiant_team_id=RADIANT_TEAM,
            radiant_team_name_observed="Radiant Test",
            dire_team_id=DIRE_TEAM,
            dire_team_name_observed="Dire Test",
            radiant_win=radiant_win,
            duration_seconds=duration_seconds,
            draft_complete=draft_complete,
            mapper_version=1,
            canonicalized_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )

    for slot, player_id in enumerate(PLAYER_BASE_IDS):
        conn.execute(
            MATCH_PLAYERS.insert().values(
                match_id=match_id,
                side="RADIANT",
                slot_in_side=slot,
                player_id=player_id,
                hero_id=slot + 1,
                position="POSITION_1" if slot == 0 else None,
                kills=3,
                deaths=2,
                assists=5,
            )
        )
    for slot, player_id in enumerate(DIRE_PLAYER_BASE_IDS):
        conn.execute(
            MATCH_PLAYERS.insert().values(
                match_id=match_id,
                side="DIRE",
                slot_in_side=slot,
                player_id=player_id,
                hero_id=slot + 11,
                position="POSITION_1" if slot == 0 else None,
                kills=4,
                deaths=3,
                assists=6,
            )
        )

    if with_draft and draft_complete:
        seed_draft_events(conn, match_id)


def seed_draft_events(conn: Connection, match_id: int) -> None:
    """Seed a deterministic 24-event draft (12 bans + 10 picks + 2 bans)."""
    for sequence in range(24):
        action = "BAN" if sequence < 12 or sequence >= 22 else "PICK"
        conn.execute(
            DRAFT_EVENTS.insert().values(
                match_id=match_id,
                sequence=sequence,
                action=action,
                side="RADIANT" if sequence % 2 == 0 else "DIRE",
                hero_id=sequence + 1,
                was_successful=None if action == "PICK" else True,
            )
        )


def classify_match(
    conn: Connection,
    match_id: int,
    *,
    event: str,
    tier: str,
    source: str = "test",
) -> None:
    conn.execute(
        MATCH_CLASSIFICATIONS.insert().values(
            match_id=match_id,
            liquipedia_event=event,
            liquipedia_tier=tier,
            source=source,
        )
    )
