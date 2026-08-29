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
  `bannedHeroId` were never observed both non-null and disagreeing. In
  this particular match, *every* ban row is `heroId`-null (not just a
  minority), so it also demonstrates the fallback holding up when it is
  exercised on 100% of a match's bans, not only a partial subset.

- `8461854486_ti2025_radiant_dire_side_swap_within_series.json`
  Game 4 of the *same* series as `8461956309` above (`seriesId` 1010717,
  The International 2025 grand final). Team Falcons is Radiant here but
  Dire in game 5 (and vice versa for Xtreme Gaming) -- confirmed by
  comparing `radiantTeamId`/`direTeamId` and the `players[].steamAccountId`
  sets between the two files. This pair is live re-verification evidence
  (fetched directly from STRATZ, not derived from the older 265-match
  sample) that Radiant/Dire side is a **per-game** fact, not a fixed
  per-team/per-series one, and that the mapper correctly reads it from
  each match's own `radiantTeamId`/`direTeamId`/`players[].isRadiant`
  rather than assuming any team keeps the same side across a series.
  Cross-checked against each player's `isVictory` field in both raw
  payloads: it agrees with `didRadiantWin` combined with that player's
  `isRadiant` in every case, with no Radiant/Dire inversion in either
  game.   See `test_paired_matches_confirm_radiant_dire_side_swap_not_inverted`
  in `tests/data/test_stratz_mapping.py`.

## Rejected fixtures (`rejected/`)

Payloads under `rejected/` are preserved as regression evidence for shapes
that must **not** map successfully. They are excluded from
`test_real_anomaly_fixtures_still_map_successfully` and covered instead by
`test_rejected_anomaly_fixtures_fail_canonicalization`.

- `7929545363_ti2024_pickbans_null_completed_match.json`
  The International 2024 (league 16935). A completed match with full
  rosters, result, and duration, but `pickBans: null` from STRATZ. The
  GraphQL field is requested in ingestion; the null is source data, not a
  field-selection omission. Without draft sequence, the match cannot satisfy
  the canonical draft-complete prediction boundary.

- `7929556224_ti2024_pickbans_null_completed_match.json`
  Same league and anomaly shape as above (second of two such matches in the
  TI 2024 smoke ingest). Also demonstrates that `players[].heroId` can be
  populated even when `pickBans` is absent -- final lineups alone are not
  sufficient to reconstruct ordered `draft_events`.
