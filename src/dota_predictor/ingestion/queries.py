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
