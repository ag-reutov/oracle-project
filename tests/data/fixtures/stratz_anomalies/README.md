# STRATZ evidence / regression fixtures

Raw, source-faithful STRATZ `MatchType` payloads preserved during the Tier
1/Tier 2 STRATZ verification probe (see the task report). Each file is one
unmodified match record (probe-internal bookkeeping keys prefixed
`_sample_` stripped only) captured because it demonstrates a real,
reproducible behavior worth guarding against regressions -- not because
the match is corrupt or invalid. None of these currently fail
`canonical_match_from_stratz`; they are preserved as evidence for future
mapper/schema work.

- `4986461644_ti2019_22event_draft_and_nonchronological_series_matches.json`
  The International 2019 (patch 7.22 era). Has a 22-event draft (vs. 24 in
  patch >= 7.30 samples), demonstrating that total draft-event count is
  not a fixed historical invariant on real data, not just in synthetic
  tests. Also belongs to a 4-game `BEST_OF_FIVE` series whose
  `series.matches` array is returned in **descending** `startDateTime`
  order (most recent game first), not chronological ascending order --
  evidence that a future `game_number_in_series` derivation must sort
  `series.matches` by `startDateTime` explicitly and must not assume the
  array's own order is chronological.

- `8461956309_ti2025_tournamentid_and_tournamentround_null.json`
  The International 2025. A recent, fully-populated, successfully-mapped
  match where `tournamentId` and `tournamentRound` are both `null`. Across
  the entire 265-match verification sample, `tournamentId` was `null` on
  every single match (0/265 populated) regardless of era or tournament
  tier, and `tournamentRound` likewise (0/265). Evidence that these two
  `MatchType` fields should not be relied on for professional-match
  identity or bracket-stage information.

- `4984038549_ban_row_heroid_null_bannedheroid_fallback.json`
  A real professional match containing at least one ban row where
  `heroId` is `null` and `bannedHeroId` is populated -- the mapper's
  `heroId`-preferred-with-`bannedHeroId`-fallback path
  (`draft_event_from_stratz_pick_ban`) is exercised by real data here, not
  only by the synthetic unit-test fixture. Across the full sample, 566 of
  3630 ban rows (~15.6%) were `heroId`-null in this shape; `heroId` and
  `bannedHeroId` were never observed both non-null and disagreeing.
