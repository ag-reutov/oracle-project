"""GraphQL query strings for STRATZ league-match ingestion."""

from __future__ import annotations

MATCH_SELECTION = """
id
didRadiantWin
durationSeconds
startDateTime
endDateTime
tournamentId
tournamentRound
leagueId
league { id name displayName tier region startDateTime endDateTime lastMatchDate prizePool }
seriesId
series {
  id
  type
  teamOneId
  teamTwoId
  leagueId
  teamOneWinCount
  teamTwoWinCount
  winningTeamId
  lastMatchDateTime
  matches { id startDateTime }
}
gameVersionId
radiantTeamId
direTeamId
radiantTeam { id name tag }
direTeam { id name tag }
players {
  steamAccountId
  isRadiant
  playerSlot
  heroId
  position
  lane
  role
  kills
  deaths
  assists
  goldPerMinute
  experiencePerMinute
  numLastHits
  numDenies
  networth
  heroDamage
  towerDamage
  heroHealing
  level
}
pickBans {
  isPick
  heroId
  order
  bannedHeroId
  isRadiant
  playerIndex
  wasBannedSuccessfully
  isCaptain
  letter
}
"""

LEAGUE_MATCHES_QUERY = f"""
query LeagueMatchesIngest($id: Int!, $request: LeagueMatchesRequestType!) {{
  league(id: $id) {{
    id
    name
    matches(request: $request) {{
      {MATCH_SELECTION}
    }}
  }}
}}
"""

# Individual match fetch for leagues whose `league(id)` catalog entry is
# null. Nested `league {{ ... }}` is still requested (same selection as
# the league path) but is allowed to come back null; canonicalization
# keys off `leagueId`.
MATCH_BY_ID_QUERY = f"""
query MatchByIdIngest($id: Long!) {{
  match(id: $id) {{
    {MATCH_SELECTION}
  }}
}}
"""

# Lightweight fetch for observed match-player parse labels. Used by the
# position backfill so existing raw payloads are not replaced wholesale.
MATCH_PLAYER_POSITION_SELECTION = """
id
players { steamAccountId isRadiant playerSlot heroId position lane role }
"""

MATCH_PLAYER_POSITIONS_QUERY = f"""
query MatchPlayerPositions($id: Long!) {{
  match(id: $id) {{
    {MATCH_PLAYER_POSITION_SELECTION}
  }}
}}
"""

# Lightweight fetch for observed post-match box-score scalars. Used by
# the performance backfill so existing raw payloads are not replaced
# wholesale. Does not request `stats.*` time-series, IMP, award, or
# heroAverage.
MATCH_PLAYER_PERFORMANCE_SELECTION = """
id
players {
  steamAccountId
  isRadiant
  playerSlot
  heroId
  kills
  deaths
  assists
  goldPerMinute
  experiencePerMinute
  numLastHits
  numDenies
  networth
  heroDamage
  towerDamage
  heroHealing
  level
}
"""

MATCH_PLAYER_PERFORMANCE_QUERY = f"""
query MatchPlayerPerformance($id: Long!) {{
  match(id: $id) {{
    {MATCH_PLAYER_PERFORMANCE_SELECTION}
  }}
}}
"""

# Lightweight discovery only -- ids and team ids, not canonical payloads.
TEAM_LEAGUE_MATCH_IDS_QUERY = """
query TeamLeagueMatchIds($teamId: Int!, $request: TeamMatchesRequestType!) {
  team(teamId: $teamId) {
    id
    matches(request: $request) {
      id
      leagueId
      radiantTeamId
      direTeamId
    }
  }
}
"""

# Identity-only hero catalog. `displayName` is the human-readable name;
# STRATZ `name` (npc_dota_hero_*), aliases, roles, stats, facets, talents,
# abilities, localization, and `gameVersionId` are intentionally omitted.
HEROES_QUERY = """
query HeroesReference {
  constants {
    heroes {
      id
      displayName
    }
  }
}
"""

# Source-native game-version catalog. `name` is the STRATZ patch label
# (e.g. "7.38", "7.40b"); `asOfDateTime` is a Unix-seconds timestamp.
GAME_VERSIONS_QUERY = """
query GameVersionsReference {
  constants {
    gameVersions {
      id
      name
      asOfDateTime
    }
  }
}
"""
