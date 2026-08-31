"""Unit tests for match-ID discovery (no STRATZ/OpenDota network)."""

from __future__ import annotations

from typing import Any

from dota_predictor.ingestion.discovery import (
    dedupe_match_ids,
    discover_league_match_ids,
    discover_match_ids_from_team_matches,
)


class MockTeamFetcher:
    def __init__(self, pages: dict[tuple[int, int], list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.calls: list[tuple[int, int, int]] = []

    def fetch_team_league_match_ids_page(
        self,
        team_id: int,
        *,
        league_id: int,
        skip: int,
        take: int,
    ) -> list[dict[str, Any]]:
        self.calls.append((team_id, skip, take))
        return list(self.pages.get((team_id, skip), []))


def test_dedupe_match_ids_preserves_first_order() -> None:
    assert dedupe_match_ids([3, 1, 3, 2, 1]) == (3, 1, 2)


def test_team_walk_bfs_collects_opponents() -> None:
    fetcher = MockTeamFetcher(
        {
            (10, 0): [
                {
                    "id": 100,
                    "leagueId": 17419,
                    "radiantTeamId": 10,
                    "direTeamId": 20,
                }
            ],
            (20, 0): [
                {
                    "id": 100,
                    "leagueId": 17419,
                    "radiantTeamId": 10,
                    "direTeamId": 20,
                },
                {
                    "id": 101,
                    "leagueId": 17419,
                    "radiantTeamId": 20,
                    "direTeamId": 30,
                },
            ],
            (30, 0): [
                {
                    "id": 101,
                    "leagueId": 17419,
                    "radiantTeamId": 20,
                    "direTeamId": 30,
                }
            ],
        }
    )
    match_ids, teams = discover_match_ids_from_team_matches(
        fetcher, 17419, [10], page_size=100
    )
    assert match_ids == frozenset({100, 101})
    assert teams == frozenset({10, 20, 30})


def test_team_walk_ignores_other_league_rows() -> None:
    fetcher = MockTeamFetcher(
        {
            (10, 0): [
                {
                    "id": 1,
                    "leagueId": 17419,
                    "radiantTeamId": 10,
                    "direTeamId": 11,
                },
                {
                    "id": 2,
                    "leagueId": 999,
                    "radiantTeamId": 10,
                    "direTeamId": 99,
                },
            ],
            (11, 0): [],
        }
    )
    match_ids, teams = discover_match_ids_from_team_matches(fetcher, 17419, [10])
    assert match_ids == frozenset({1})
    assert 99 not in teams


def test_discover_league_match_ids_unions_and_notes_gap() -> None:
    fetcher = MockTeamFetcher(
        {
            (10, 0): [
                {
                    "id": 100,
                    "leagueId": 17419,
                    "radiantTeamId": 10,
                    "direTeamId": 20,
                }
            ],
            (20, 0): [],
        }
    )
    result = discover_league_match_ids(
        17419,
        team_fetcher=fetcher,
        seed_team_ids=[10],
        skip_opendota=True,
    )
    assert result.match_ids == (100,)
    assert result.opendota_match_ids == frozenset()
    assert any("team.matches walk" in note for note in result.notes)
