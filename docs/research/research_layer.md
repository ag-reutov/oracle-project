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