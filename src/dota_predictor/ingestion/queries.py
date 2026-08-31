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
players { steamAccountId isRadiant playerSlot heroId }
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
