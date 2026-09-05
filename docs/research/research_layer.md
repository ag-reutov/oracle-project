# Research layer (`research` PostgreSQL schema)

A thin, authoritative analytical layer over the canonical Dota warehouse.
Everything here is a plain PostgreSQL **view** over the existing canonical
tables in `public` (`matches`, `match_players`, `draft_events`, `leagues`,
`match_classifications`). There is no duplicated storage, no materialized
view, no ETL job, and no second database.

The purpose is to let ordinary research questions be answered from SQL,
Metabase, or Python **without manually reconstructing league / event / tier /
qualifier semantics every time**.

## Design decision (internal)

What remains authoritative in base tables/config:

* `leagues` (+ `config/leagues.yaml`) — the curated STRATZ league registry:
  league_id -> default league name + default Liquipedia tier
  (`leagues.liquipedia_tier`), `in_scope`, `window_filter` + main-event
  window.
* `match_classifications` (+ `config/event_match_assignments.yaml`) — match-
  level overrides for shared/multi-tier STRATZ leagues (the 9 ACL 2025 finals
  matches inside T3 league 17875, and the four 1win events inside T2 umbrella
  league 16427: Spring/Summer/Fall T2 and Punch T3).
* `matches`, `match_players`, `draft_events` — the canonical facts.
* `ingestion_leagues` — the allowlist; only allowlisted leagues can have
  canonical matches, which is why qualifier-only leagues can never appear.

What the research views merely expose/derive:

* Effective tier / event — the single derivation below, never a second
  classification system.
* `match_date` / `year` / `month` — derived from UTC `start_time`.
* `winner_side` / `player_win` — derived from the canonical `radiant_win`.
* `team_id` / `team_name` per player — derived from the match's radiant/dire
  team and side.
* Population membership booleans and the population views.

How effective tier/event is computed (single definition, lives only in
`research.matches`):

```sql
effective_tier  = coalesce(match_classifications.liquipedia_tier,
                           leagues.liquipedia_tier)
effective_event = coalesce(match_classifications.liquipedia_event,
                           leagues.name)
classification_source = CASE WHEN match_classifications.match_id IS NOT NULL
                            THEN 'match-level override'
                            ELSE 'league default' END
```

How main-event population membership is computed:

* Qualifiers never reach the canonical layer: `matches.league_id` is a FK to
  `ingestion_leagues` (allowlist only), qualifier leagues are `in_scope =
  false`, and window-filtered leagues ingest only their main-event window.
  So a canonical match is already a main-event match; the research layer does
  not need to re-derive qualifier windows.
* A match belongs to the T1/T2 main-event population iff
  `effective_tier IN ('T1','T2')` and `match_date >= 2024-01-01`
  (`is_t12_main_event`), and to the T1/T2/T3 professional population iff
  `effective_tier IN ('T1','T2','T3')` and `match_date >= 2024-01-01`
  (`is_t123_main_event`).

Hard cases the derivation relies on being materialized as
`match_classifications` (they are — see `config/event_match_assignments.yaml`):

* ACL 2025: the 9 finals matches in STRATZ league 17875 (default T3) are
  classified T2. The other 55 matches in 17875 stay T3.
* 1win umbrella: STRATZ league 16427 (default T2) contains four distinct
  Liquipedia events, each classified by its exact event window via
  `match_classifications` -- 1win Spring (T2), Summer (T2), Fall (T2), and
  Punch (T3). All 86 league-16427 matches resolve to their exact event
  name/tier; nothing collapses to the umbrella name, and the T3 Punch
  matches never leak into the T1/T2 corpus.

## Relations

### `research.matches`

* Grain: one row per canonical match (`matches.match_id` unique).
* Purpose: the main analytical match relation.
* Identity: `match_id`, `start_time`, `match_date`, `year`, `month`.
* Storage event identity: `league_id`, `league_name` (curated registry name),
  `default_tier` (league default Liquipedia tier).
* Effective event identity: `effective_event`, `effective_tier`,
  `classification_source` (`'league default'` | `'match-level override'`).
* Teams/result: `radiant_team_id`, `radiant_team_name`, `dire_team_id`,
  `dire_team_name`, `radiant_win` (canonical), `winning_side` (derived).
* Context: `duration_seconds`, `game_version_id` (patch identifier),
  `draft_complete`, `series_id`, `series_type`, `game_number_in_series`.
* Population booleans: `is_main_event`, `is_t12_main_event`,
  `is_t123_main_event`.

### `research.player_matches`

* Grain: one canonical player appearance per match (`match_id`, `side`,
  `slot_in_side`; `(match_id, player_id)` is also unique).
* Purpose: player-level research (heroes, positions, box scores).
* Match context: `match_id`, `start_time`, `match_date`, `year`, `month`,
  `effective_event`, `effective_tier`, `league_id`, `game_version_id`,
  `draft_complete`.
* Identity: `player_id`, `team_id`, `team_name`, `side`, `slot_in_side`,
  `hero_id`, `position`, `lane`, `role` (observed STRATZ labels; NULL /
  `UNKNOWN` preserved).
* Result: `player_win` (bool; true iff the player's team won, derived from
  `radiant_win`).
* Box scores: `kills`, `deaths`, `assists`, `gold_per_minute`,
  `experience_per_minute`, `num_last_hits`, `num_denies`, `networth`,
  `hero_damage`, `tower_damage`, `hero_healing`, `level`. These are the
  observed STRATZ scalars, exposed verbatim; no composite/normalized scores
  are added.

### `research.players`

* Grain: one row per canonical player (`players.player_id` unique) — the
  Slice 2 canonical player universe.
* Purpose: player identity foundation for player rankings, roster history,
  player research pages, hero-pool research, and historical player
  comparisons.
* Identity: `player_id` (the canonical STRATZ `steamAccountId`; identity is
  stable across team/role/nickname changes), `display_name` (deterministic
  best-known name; NULL while the corpus has no player-name observations).
* Observation summary (derived fresh from canonical facts, never cached):
  `first_seen_at`, `last_seen_at`, `match_count`.
* Deliberately absent: current team, position, rating, form, hero pool, and
  any other time-varying / analytical attribute — canonical identity must
  not leak future state into historical research.
* Orphan registry ids (in `players` but referenced by no match) are excluded
  and reported by `scripts/audit_player_identity.py`.

### `research.draft_events`

* Grain: one canonical draft event (`match_id`, `sequence`).
* Purpose: draft research (pick/ban analysis).
* Columns: `match_id`, `start_time`, `match_date`, `year`, `month`,
  `effective_event`, `effective_tier`, `league_id`, `game_version_id`,
  `draft_complete`, `sequence`, `action`, `side`, `hero_id`, `was_successful`.

### `research.leagues`

* Grain: one row per curated league (`leagues.league_id` unique) — the
  Slice 3 canonical league/event identity view.
* Purpose: named-competition identity with explicit provenance, so league
  research never needs to reverse-engineer `config/leagues.yaml`.
* Identity: `league_id` (stable STRATZ league id), `league_name` (curated
  canonical name), `stratz_tier` (raw STRATZ `LeagueTier` — source
  identity, cross-check signal only) vs `liquipedia_tier` (our curated
  Liquipedia classification). The two are deliberately never conflated:
  our T1/T2/T3 label is a curated property, not an intrinsic STRATZ one.
* Provenance: `curation_source` (where the curation decision came from),
  `curated_at` (when it was made), plus `in_scope`, `fetch_mode`,
  `start_date`, `end_date`, `window_filter`.

## Reference entities (Slice 3)

The canonical reference-entity layer maps the three stable ids used by the
canonical facts to named, provenance-bearing entities, with exactly one
canonical path per relationship:

* `match_players.hero_id` / `draft_events.hero_id` → **hero**
  (`heroes.parquet`, the DuckDB `heroes` view). Since reference schema v2
  each row exposes `name` (canonical display name), `short_name`, `aliases`
  (STRATZ-supplied), `source` and `retrieved_at`. See
  `dota_predictor.data.reference_identity.HeroIdentity`.
* `matches.league_id` → **league/event** (`research.leagues`, the curated
  `leagues` registry). See `LeagueIdentity`.
* `matches.game_version_id` → **patch/version** (`game_versions.parquet`,
  the DuckDB `game_versions` view). Each row exposes the human-readable
  patch label `name` (e.g. "7.38"), STRATZ's authoritative release
  timestamp `as_of_datetime`, `source` and `retrieved_at`. This makes
  "which matches were played on 7.39e" answerable by label instead of an
  opaque numeric id.

Storage decision: heroes and game versions stay Parquet reference catalogs
(consumed via `register_reference_views`), leagues stay in the curated
PostgreSQL registry — the existing architecture. No parallel tables were
created. First-seen-in-corpus for a game version is a corpus-derived
observation reported by the audit, explicitly labelled as such, and is
never used as a release date. Regions were investigated but intentionally
deferred (STRATZ exposes server regions only; no canonical entity
references a region id).

The reproducible census is `scripts/audit_reference_entities.py`
(`dota_predictor.data.reference_identity.audit_reference_entities`):
hero/league/game-version counts, referenced vs resolved vs unresolved ids,
duplicate/conflict counts, and regions status.

## Roster history (Slice 4)

Observed roster history is a thin descriptive layer over the canonical match
facts — no new storage. It answers "who actually represented each team in
each historical match, and what sequence of teams has each player been
observed representing?" without pretending match appearances provide exact
contractual transfer dates.

* `research.player_matches` already **is** the canonical roster-appearance
  relation (`match_id`, `start_time`, `player_id`, `team_id`, `side`), so no
  separate `roster_appearances` view is created.
* `research.team_match_lineups` — one row per `(match_id, team_id)`: the
  players observed for that team in that match as `lineup_player_ids`
  (sorted canonical player ids) with a deterministic `lineup_key` (same ids,
  comma-joined), plus an explicit cardinality audit: `n_players`,
  `n_resolved_players`, `n_null_player_ids`, `n_distinct_players`,
  `has_duplicate_players`, `has_fewer_than_five`, `has_more_than_five`,
  `has_exactly_five`, `is_complete_five`, and `team_is_match_team`. Malformed
  lineups are reported, never forced into a five-player shape. (`team_id` is
  derived from the parent match's radiant/dire teams by side, so
  `team_is_match_team` is structurally always TRUE.)
* `research.player_team_spells` — one row per `(player_id, spell_index)`: a
  player's maximal run of matches observed for one team, in chronological
  order. A new spell begins only when the observed `team_id` changes; a later
  return to a previous team is a new spell (`A -> B -> A` is three spells); a
  time gap with no intervening team observation does not split a spell.
  `first_seen_at` / `last_seen_at` are observed match times only — never
  invented `joined_at` / `left_at` dates, and never a claim of continuous
  contractual membership between observations. Exposes `team_id`,
  `spell_index`, `first_seen_at`, `first_match_id`, `last_seen_at`,
  `last_match_id`, `observed_match_count`.

The reproducible census is `scripts/audit_roster_history.py`
(`dota_predictor.data.roster_history.audit_roster_history`): observation,
lineup-cardinality, spell, and integrity counts, with unresolved/duplicate/
contradictory anomalies reported explicitly. One-match spells and
short/returning spells are exposed descriptively (`observed_match_count = 1`,
previous/next spell via `spell_index`); no semantic `standin` label is ever
assigned. The audit states that official contractual roster history is a
separate future data-source problem — no authoritative roster/transfer
source exists in the repository today.

## Historical roster state (Slice 5)

Slice 5 turns the Slice 4 observations into a **strictly causal pre-match
roster state** at grain `(team_id, match_id, start_time)` — one team in one
match, evaluated immediately before that match. It answers *"what was known
about each team's current five-player lineup from competitive observations
strictly before that match?"* without any future roster observation, future
spell boundary, result, or post-match statistic. It is descriptive
historical-state construction, never team-strength modeling.

* `research.player_team_state` — one row per `(player_id, team_id,
  match_id)`: the player's observed relationship to a team immediately
  before that match. `prior_team_match_count` and the first/last prior
  same-team times use only observations with `start_time` strictly before
  the current match; `previous_observed_team_id` / `previous_observed_match_id`
  / `previous_observed_match_at` describe the player's most recent strictly
  earlier match **for any team** (tie-broken by `match_id DESC` only among
  already-strictly-prior rows). The flags
  (`is_first_observed_match_for_team`, `is_returning_to_team`,
  `is_continuing_with_team`) are mutually exclusive observational
  classifications (`A -> B -> A` is "returning"; a long gap with no
  intervening team is still "continuing"); no transfer / signing /
  stand-in label is ever assigned. `consecutive_prior_team_appearances`
  is spell-so-far (the causal run length before this match) — never the
  eventual spell length, which would leak the future. Timing
  (`days_since_player_previous_match`, `days_since_player_previous_team_match`)
  is descriptive; a long gap is never interpreted as retirement,
  benching, or a transfer.
* `research.team_roster_state` — one row per `(match_id, team_id)`: the
  team's historical roster state immediately before that match. The
  current lineup is reused verbatim from `research.team_match_lineups`
  (Slice 4) — no second lineup definition is reconstructed.
  `previous_match_id` / `previous_match_at` /
  `previous_lineup_player_ids` describe the team's most recent strictly
  earlier observed match (no arbitrary time cutoff). For complete-five
  current and previous lineups,
  `players_retained_from_previous_match` / `players_changed_from_previous_match`
  / `same_lineup_as_previous_match` are defined; for malformed lineups
  they are NULL and the malformation stays explicit via the cardinality
  flags. `prior_exact_lineup_match_count` /
  `last_exact_lineup_match_id` / `last_exact_lineup_at` answer "has this
  exact five-player lineup played together before?" using strictly
  earlier complete-five matches of the same team with the identical
  `lineup_key`. Team composition (`continuing_player_count`,
  `first_observed_for_team_count`, `returning_player_count`) reconciles
  to the resolved lineup players (5 for a complete lineup); unresolved
  players are never fabricated a classification. `days_since_team_previous_match`
  is descriptive timing.

Equal timestamps are strictly non-causal: a match sharing the current
`start_time` is never prior evidence, even when its `match_id` sorts
first. The future-deletion invariant (deleting all observations after any
time `T` leaves every state at `T` bit-identical) is enforced by
construction and verified by
`dota_predictor.data.roster_state.check_future_deletion_invariant` and the
test suite.

The reproducible census is `scripts/audit_roster_state.py`
(`dota_predictor.data.roster_state.audit_roster_state`): team-match /
player-team state counts, retained/changed/exact-lineup distributions,
first-observed / returning / continuing observations, and integrity
anomalies (incomplete lineups, impossible aggregate counts, future-deletion
invariant violations).

## Team strength & ranking (Slice 6)

Slice 6 establishes a canonical, inspectable **historical Elo state** of each
canonical/source `team_id` under the existing production Elo definition. It
consolidates, validates and exposes the existing Team Elo infrastructure -- it
does **not** invent a second rating system, and it does **not** expose a
leaderboard.

### What Slice 6 provides

1. **`research.team_strength_state`** — causal historical raw team-ID Elo, a
   persisted deterministic derived table at grain `(team_id, match_id)`:
   `elo_pre` (entering the match), `elo_post` (bookkeeping, only after the
   result), a strictly-prior descriptive record (`prior_match_count` /
   `prior_win_count` / `prior_loss_count` / `prior_win_rate`,
   `previous_match_id` / `previous_match_at`, `days_since_previous_match`,
   `is_first_observed_match`). It is idempotently rebuilt in one transaction
   from the canonical `matches` facts by
   `scripts/rebuild_team_strength.py`, reusing the exact production
   `features.team_elo` definition (initial rating 1500.0, K-factor 32.0,
   expected-score formula, chronological replay with equal-`start_time`
   mutual blindness). Canonical match facts remain the sole source of truth;
   there is deliberately no mutable "current Elo" column on `teams`.
2. **`research.team_strength_build`** — deterministic rebuild provenance /
   staleness marker: source corpus snapshot, count/extrema diagnostics, and a
   deterministic SHA-256 `source_fingerprint` over the canonical match fields
   that determine the derived state. `dota_predictor.data.team_strength.check_freshness`
   is the reusable freshness check (detects an old result/team/time
   correction even when count and corpus extrema are unchanged).
3. **`research.raw_team_elo_latest`** — latest raw Elo **state** per source
   `team_id` (one row per team_id, exposing the terminal post-match rating
   derived as `elo_pre + SUM(elo_post - elo_pre)` over the latest group,
   plus `last_match_at`, `observed_match_count`, wins/losses, organization
   metadata, latest known lineup, `as_of_at`, and
   `days_since_last_match_as_of_corpus_end`). It exposes **no ordinal `rank`
   and no global ordering**; a plain `SELECT *` must not imply a ranking.

### What Slice 6 deliberately does NOT provide

- a global team ranking
- a current team ranking
- an active-team ranking
- competitive-team identity
- team lineage
- disbandment status
- a T1-only / T1+T2 power rating
- activity eligibility

> Comparative global ranking is intentionally absent from Slice 6 because
> competitive-team lineage, eligibility, and rating population have not yet
> been defined.

Raw `team_id` Elo is useful historical state but raw team IDs are not yet
suitable objects for a global ordinal ranking because competitive identity is
fragmented across multiple `team_id`s (e.g. PARIVISION/PVISION, duplicate
Nigma identities, 1win/1w), inactive/disbanded identities remain present, the
population is ~60% Tier 3, and no active-team eligibility or competitive-lineage
definition exists. These are limitations of entity definition, population
definition, and ranking eligibility -- not bugs in the Elo recurrence.

Strict temporal semantics (identical to the production Elo layer): for a
match at time `T`, strength entering that match uses only matches with
`historical.start_time < T`. Equal timestamps never create causal precedence
through `match_id`: matches sharing a `start_time` are one temporal group
whose members read the same pre-group rating and never influence one another.
`match_id` is never used for ordering. `elo_pre` is available before the
match; `elo_post` is historical bookkeeping only and must never be treated as
a PRE_DRAFT feature for the same match. The future-deletion invariant holds
by construction (`dota_predictor.data.team_strength.check_future_deletion_invariant`
verifies it), and team strength depends only on match history/results, never
on future roster information.

Cross-check: for the current canonical corpus,
`research.team_strength_state.elo_pre` equals the production pre-draft Elo
(`features.team_elo.compute_team_elo_features`, the Step 3C
`radiant_team_elo`/`dire_team_elo` columns) for every team-match row. Slice 6
is the research-layer counterpart of the existing production Elo definition;
no production feature code was rewritten.

### Slice 6 diagnostics (evidence for Slice 7, no leaderboard)

The reproducible census is `scripts/audit_team_strength.py`. Its default
output contains NO leaderboard and NO Top-20; it reports:

* **Historical Elo state** (`audit_team_strength`): canonical matches
  processed, team-match states, teams observed, `elo_pre` min/median/max, and
  latest Elo min/median/max -- never an ordinal ranking.
* **Integrity**: production-Elo cross-check mismatches, future-deletion
  violations, equal-timestamp violations, missing canonical team references.
* **Freshness**: stored vs current source fingerprint and fresh/stale.
* **Activity** (`audit_activity_distribution`): days since last observed
  match at corpus end bucketed (<=30, 31-60, 61-90, 91-180, >180 days).
  Nobody is filtered out.
* **Population** (`audit_elo_population`): T1/T2/T3 count and share of Elo
  updates (plus other categories as present) and the count of predominantly
  Tier-3 teams.
* **Identity fragmentation** (`audit_identity_fragmentation`): a conservative
  diagnostic of candidate `team_id` pairs/groups that may share a competitive
  lineage (same curated organization, shared normalized observed name,
  identical/overlapping observed five-player roster, sequential
  non-overlapping activity). Nothing is merged; the output is evidence for
  human/research review and for Slice 7 lineage resolution.

An **opt-in** debugging flag `--show-raw-elo` prints the raw latest team-ID
Elo values sorted by Elo for inspection only (`audit_raw_elo_latest`), with no
ordinal `rank` and a prominent "DEBUGGING/DIAGNOSTIC OUTPUT — raw canonical
team IDs, not globally comparable current competitive teams" notice. It is
never part of the default audit.

Relationship to the pre-draft roster-continuity feature: the production
`features.pre_draft_snapshot` `radiant/dire_roster_players_retained`
columns compute the same retained count in DuckDB with the same strict `<`
boundary. Slice 5's `players_retained_from_previous_match` is the canonical
research-state equivalent (see
`tests/features/test_roster_continuity_cross_check.py`). The only
definitional difference is the incomplete-lineup edge case: the pre-draft
`COUNT(*)` can return a partial integer, while Slice 5 returns NULL and
keeps the malformation explicit. The production feature is intentionally
not modified; Slice 5 provides the cleaner canonical definition that can
later validate or replace the duplicated feature implementation.

## Population views

* `research.t12_matches` — `research.matches WHERE is_t12_main_event`:
  intended Liquipedia Tier 1 + Tier 2 main-event corpus, 2024-01-01 onward,
  qualifiers excluded, using effective tier semantics.
* `research.pro_matches` — `research.matches WHERE is_t123_main_event`:
  all intended T1/T2/T3 main-event professional matches from 2024 onward.
* `research.t12_draft_matches` — `research.t12_matches WHERE draft_complete`:
  the safe population for draft-dependent research. Draft-incomplete games
  remain in `research.matches` / `research.player_matches`.

## Usage

```sql
SELECT effective_tier, count(*)
FROM research.matches
GROUP BY effective_tier;

SELECT hero_id, count(*)
FROM research.player_matches
WHERE effective_tier IN ('T1', 'T2')
  AND start_time >= DATE '2024-01-01'
GROUP BY hero_id
ORDER BY count(*) DESC;
```

Researchers do not need to know about league 17875, league 16427, qualifier
windows, match-level tier overrides, or draft-canonicalization exceptions.
The population views already embody them.

## Known census-level verification issue

The following three historical events could not be independently reverified
against Liquipedia (HTTP 429 during the census repair) and retain the
warehouse's current authoritative classification:

* PGL Wallachia Season 4 (league 18058, T1)
* AsiaPro League Season 2 (league 17622, T2)
* Games of the Future 2025 Abu Dhabi (league 18937, T2)

The research layer exposes the authoritative classification it is given; it
does not reclassify them. See `docs/research/t12_census_audit.md`.

## Permissions

The read-only `metabase_reader` role is granted `USAGE` on the `research`
schema and `SELECT` on every research view. Future research views created by
the app role are granted via default privileges. See `docs/metabase.md`.

## Reproducible smoke-test queries

Six example research queries that exercise the layer (results as of the
2026-09-04 census in `docs/research/t12_census_audit.md`):

```sql
-- 1. T1/T2 main-event match count by year
SELECT year, count(*) FROM research.t12_matches GROUP BY year ORDER BY year;

-- 2. T1/T2 match count by game version / patch
SELECT game_version_id, count(*) AS n_matches
FROM research.t12_matches
GROUP BY game_version_id ORDER BY n_matches DESC;

-- 3. median match duration by year (minutes)
SELECT year, percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_seconds) / 60.0
FROM research.t12_matches
GROUP BY year ORDER BY year;

-- 4. top 15 heroes by player appearances in T1/T2
SELECT hero_id, count(*) AS appearances
FROM research.player_matches
WHERE effective_tier IN ('T1', 'T2') AND start_time >= DATE '2024-01-01'
GROUP BY hero_id ORDER BY appearances DESC LIMIT 15;

-- 5. total player appearances + unique players in T1/T2
SELECT count(*) AS total_appearances, count(DISTINCT player_id) AS unique_players
FROM research.player_matches
WHERE effective_tier IN ('T1', 'T2') AND start_time >= DATE '2024-01-01';

-- 6. draft-complete vs draft-incomplete T1/T2 match counts
SELECT draft_complete, count(*) FROM research.t12_matches GROUP BY draft_complete;
```