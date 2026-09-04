# T1/T2 main-event census audit

`scripts/audit_t12_census.py` compares the intended Liquipedia Tier 1 /
Tier 2 main-event universe against the registry and the canonical
warehouse, so the corpus cannot silently decay.

## Why it exists

The warehouse's intended population is:

> All Liquipedia Tier 1 and Tier 2 MAIN-EVENT professional Dota 2 matches
> from 2024-01-01 through 2026-09-04 (qualifiers out of scope).

Events come from Liquipedia tier pages (`Tier_1_Tournaments`,
`Tier_2_Tournaments`), recorded in `config/liquipedia_events_2024plus.yaml`.
Because that manifest is manually curated, it can go stale relative to
Liquipedia. This audit re-checks the manifest against the registry and the
database and fails loudly when an intended in-window event is absent or
unresolved.

A registry row marked `COMPLETE` with zero matches does not prove
completeness on its own: the audit re-checks the canonical `matches`
count directly for every in-window event.

## How to run it

```bash
# Offline (registry + DB only; requires DATABASE_URL / .env)
uv run python scripts/audit_t12_census.py

# Include OpenDota match-ID reconciliation (network, throttled)
uv run python scripts/audit_t12_census.py --opendota

# Override the window
uv run python scripts/audit_t12_census.py --end 2026-09-04
```

Exit status is non-zero when any in-window intended T1/T2 event is
absent or unresolved (no resolvable STRATZ id, no in_scope registry row,
or zero canonical matches).

## What it reports

- **intended in-window events** — every manifest T1/T2 event in the
  window, with its resolved STRATZ league, canonical match count, and any
  problems.
- **missing registry events** — intended event with no `in_scope` league row.
- **unresolved STRATZ ids** — intended event with no resolvable
  `league_id` (no `STRATZ_ID_MAP` entry and no manifest `league_id`).
- **events with zero matches** — intended event that resolves but has no
  canonical rows.
- **raw-only matches** — matches present in `stratz_raw_matches` for an
  intended event's league but absent from canonical `matches` (e.g. the
  corrupted-team match 8794593044 in ESL Challenger China S3 x ACL 2026).
- **tier / classification** — event-level assignments applied from
  `config/event_match_assignments.yaml` (e.g. ACL 2025).
- **unexpected registry events** — in-scope T1/T2 registry leagues that
  are not in the intended manifest (possible reclassification, e.g.
  PGL Wallachia Season 4, AsiaPro League Season 2, Games of the Future
  2025 Abu Dhabi).
- **coverage** — canonical intended-event matches by tier and year.

With `--opendota`, each event's expected main-event match set (OpenDota
`matches WHERE leagueid = N`, optionally windowed) is compared to the DB
canonical set. This is diagnostic only: OpenDota has known enumeration
gaps (DB-only matches are reported but never fail the audit).

## Shared-league events

When one STRATZ league contains matches from more than one Liquipedia
event or tier, a league row alone cannot represent the whole league. Such
events are resolved by an event-level assignment:

- `config/event_match_assignments.yaml` maps `(league_id, date window)`
  to an event/tier. Example: the 9 Asian Champions League 2025 finals
  matches inside T3 league 17875 are classified T2.
- `scripts/apply_event_classifications.py` materializes those assignments
  into the `match_classifications` table (idempotent upsert).

The effective tier of a match is
`coalesce(match_classifications.liquipedia_tier, leagues.liquipedia_tier)`.

## Keeping the manifest current

`scripts/validate_liquipedia_events_manifest.py` diffs the manifest
against the live Liquipedia tier pages. Run it after Liquipedia adds or
reclassifies tournaments, then update the manifest and re-run this audit.