"""Build config/leagues.yaml from Liquipedia event list + STRATZ ID map.

Liquipedia Tier_1_Tournaments and Tier_2_Tournaments are the authoritative
source for which events exist and their tier. This script maps those events
to STRATZ league_ids for ingestion.

Usage:
    uv run python scripts/build_leagues_yaml_from_liquipedia.py > config/leagues.yaml
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LIQUIPEDIA_EVENTS_PATH = REPO_ROOT / "config" / "liquipedia_events_2024plus.yaml"
LEAGUES_YAML_PATH = REPO_ROOT / "config" / "leagues.yaml"

HEADER = dedent(
    """
    # Curated professional Dota 2 league registry (Liquipedia T1/T2/T3).
    #
    # EVENT LIST (which tournaments, which tier): Liquipedia
    #   - https://liquipedia.net/dota2/Tier_1_Tournaments
    #   - https://liquipedia.net/dota2/Tier_2_Tournaments
    #   - https://liquipedia.net/dota2/Tier_3_Tournaments/{year}
    #   - Canonical manifest: config/liquipedia_events_2024plus.yaml
    #
    # STRATZ league_id: T1/T2 resolved via Dotabuff / Spectral / match leagueId
    # probes (see STRATZ_ID_MAP below). T3 IDs from Liquipedia |leagueid=.
    #   Do NOT use STRATZ LeagueTier to decide T1/T2/T3 scope.
    #
    # Scope: Liquipedia T1/T2/T3 main events from 2024-01-01 through present.
    # Pre-2024 DPC-era rows kept for audit only (in_scope: false).
    # Qualifiers listed but excluded (in_scope: false).
    # T1/T2 labels are never overwritten by T3 discovery.
    #
    # fetch_mode (optional, default league):
    #   league     — STRATZ league(id) { matches } pagination (historical default)
    #   match_ids  — discover match IDs, then STRATZ match(id)
    # Independent of in_scope. Catalog-null post-TI leagues use match_ids so
    # `ingest_stratz_leagues --all` never calls the broken league(id) endpoint.
    # window_filter (optional): restrict match_ids discovery/ingest to the
    # league's start_date..end_date main-event window (for leagues whose
    # STRATZ league also contains qualifiers).
    #
    # Sync: scripts/load_league_registry.py → leagues / ingestion_leagues tables.
    #
    # Canonical Parquet carries league_id / league_name, not liquipedia_tier.
    # Later T1/T2 vs T1/T2+T3 experiments must join matches.league_id to this
    # file or to leagues.liquipedia_tier. Do not rerun Slices 9–29 on the
    # expanded corpus without that filter; Slice 9 asserts
    # FROZEN_DEVELOPMENT_MATCH_COUNT == 5967.
    """
).strip()

# liquipedia_name -> (league_id, stratz_display_name, stratz_tier, in_scope, extra_notes)
# in_scope false when STRATZ league(id) GraphQL returns null (use fetch_mode match_ids).
STRATZ_ID_MAP: dict[str, tuple[int, str, str, bool, str | None]] = {
    # 2024 T1
    "BetBoom Dacha Dubai 2024": (16169, "BetBoom Dacha Dubai 2024", "PROFESSIONAL", True, None),
    "DreamLeague Season 22": (16201, "DreamLeague Season 22 powered by Intel", "PROFESSIONAL", True, None),
    "Elite League Season 1": (16483, "Elite League by FISSURE and ESB", "PROFESSIONAL", True, None),
    "ESL One Birmingham 2024": (16518, "ESL One Birmingham 2024 Powered by Intel", "PROFESSIONAL", True, None),
    "DreamLeague Season 23": (16632, "DreamLeague Season 23 powered by Intel", "PROFESSIONAL", True, None),
    "PGL Wallachia Season 1": (16669, "PGL Wallachia 2024 Season 1", "PROFESSIONAL", True, None),
    "Clavision Snow Ruyi": (16901, "Clavision DOTA League S1 : Snow-Ruyi", "PROFESSIONAL", True, None),
    "Riyadh Masters 2024": (16881, "Riyadh Masters 2024 at Esports World Cup", "PROFESSIONAL", True, None),
    "The International 2024": (16935, "The International 2024", "INTERNATIONAL", True, None),
    "BetBoom Dacha Belgrade 2024": (17126, "BetBoom Dacha Belgrade 2024", "PROFESSIONAL", True, None),
    "DreamLeague Season 24": (17272, "DreamLeague Season 24 powered by Intel", "PROFESSIONAL", True, None),
    "BLAST Slam I": (17414, "BLAST SLAM I", "PROFESSIONAL", True, None),
    "ESL One Bangkok 2024": (17509, "ESL One Bangkok 2024 powered by Intel", "PROFESSIONAL", True, None),
    # 2024 T2
    "1win Series Dota 2 Spring": (16427, "1WIN SERIES DOTA 2", "PROFESSIONAL", True, "Liquipedia 1win Spring 2024; STRATZ umbrella also contains Fall 2024 matches."),
    "FISSURE Universe Episode 2": (16730, "FISSURE Tournament", "PROFESSIONAL", True, None),
    "1win Series Dota 2 Summer": (16446, "1win Series Dota 2 Summer", "PROFESSIONAL", True, None),
    "Elite League Season 2": (16905, "Elite League Season 2 Main Event – presented by ESB", "PROFESSIONAL", True, None),
    "FISSURE Universe Episode 3": (16846, "FISSURE Universe: Episode 3", "PROFESSIONAL", True, None),
    "PGL Wallachia Season 2": (17119, "PGL Wallachia 2024 Season 2", "PROFESSIONAL", True, None),
    "Games of the Future 2024": (15981, "Games of Future 2024", "PROFESSIONAL", True, None),
    # 2025 T1
    "FISSURE PLAYGROUND 1": (17588, "FISSURE PLAYGROUND 1 - Dota", "PROFESSIONAL", True, None),
    "BLAST Slam II": (17417, "SLAM II", "PROFESSIONAL", True, None),
    "DreamLeague Season 25": (17765, "DreamLeague Season 25 powered by Intel", "PROFESSIONAL", True, None),
    "PGL Wallachia Season 3": (17891, "PGL Wallachia 2025 Season 3", "PROFESSIONAL", True, None),
    "FISSURE Universe Episode 4": (17907, "FISSURE Universe Episode 4", "PROFESSIONAL", True, None),
    "PGL Wallachia Season 4": (18058, "PGL Wallachia 2025 Season 4", "PROFESSIONAL", True, None),
    "ESL One Raleigh 2025": (17795, "ESL One Raleigh 2025", "PROFESSIONAL", True, None),
    "BLAST Slam III": (17418, "SLAM III", "PROFESSIONAL", True, None),
    "PGL Wallachia Season 5": (18358, "PGL Wallachia 2025 Season 5", "PROFESSIONAL", True, None),
    "DreamLeague Season 26": (18111, "DreamLeague Season 26", "PROFESSIONAL", True, None),
    "Clavision Masters 2025 Snow-Ruyi": (18359, "Clavision DOTA2 Masters 2025: Snow-Ruyi", "PROFESSIONAL", True, None),
    "Esports World Cup 2025": (18375, "Esports World Cup 2025", "PROFESSIONAL", True, None),
    "The International 2025": (18324, "The International 2025", "INTERNATIONAL", True, None),
    "FISSURE Universe Episode 6": (18433, "FISSURE Universe Episode 6", "PROFESSIONAL", True, None),
    "BLAST Slam IV": (17419, "SLAM IV", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "FISSURE PLAYGROUND 2": (18863, "FISSURE PLAYGROUND 2", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "PGL Wallachia Season 6": (18920, "PGL Wallachia 2025 Season 6", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "BLAST Slam V": (17420, "SLAM V", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "DreamLeague Season 27": (18988, "DreamLeague Season 27", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    # 2025 T2
    "FISSURE Special": (18046, "FISSURE Special", "PROFESSIONAL", True, None),
    "FISSURE Universe Episode 5": (18107, "FISSURE Universe Episode 5", "PROFESSIONAL", True, None),
    "FISSURE Universe Episode 7": (18633, "FISSURE Universe Episode 7", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "AsiaPro League Season 2": (17622, "AsiaPro League S2", "PROFESSIONAL", True, None),
    "Games of the Future 2025 Abu Dhabi": (18937, "Games of the Future 2025 Abu Dhabi", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    # 2026
    "FISSURE Universe Episode 8": (19239, "FISSURE Universe Episode 8", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "BLAST SLAM VI": (19099, "BLAST SLAM VI", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "DreamLeague Season 28": (19269, "DreamLeague Season 28", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "PGL Wallachia Season 7": (19435, "PGL Wallachia 2026 Season 7", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "ESL One Birmingham 2026": (19422, "ESL One Birmingham 2026", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "PGL Wallachia Season 8": (19543, "PGL Wallachia 2026 Season 8", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "DreamLeague Season 29": (19696, "DreamLeague Season 29", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "BLAST SLAM VII": (19101, "BLAST SLAM VII", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "Esports World Cup 2026": (19785, "Esports World Cup 2026", "PROFESSIONAL", True, "STRATZ league(id) null; ingested via match(id)."),
    "The International 2026": (19719, "The International 2026", "INTERNATIONAL", True, "STRATZ league(id) null; ingest via match(id) after ID discovery."),
    # 2026 events added 2026-09-04 census (all catalog-null; match_ids mode)
    "ESL Challenger China Season 2": (19130, "ESL Challenger China Season 2", "PROFESSIONAL", True, "STRATZ league(id) null; window_filter main-event window 2026-01-30..2026-02-01 excludes 18 qualifier matches."),
    "DreamLeague Division 2 Season 3": (19290, "DreamLeague Division 2 Season 3", "PROFESSIONAL", True, "STRATZ league(id) null; all 89 matches are main event (2026-02-04..02-12)."),
    "PREMIER SERIES": (19255, "PREMIER SERIES", "PROFESSIONAL", True, "STRATZ league(id) null; window_filter main-event window 2026-04-01..04-11 excludes 77 earlier-stage matches."),
    "ESL Challenger China Season 3 x ACL 2026": (19575, "ESL Challenger China Season 3 x ACL 2026", "PROFESSIONAL", True, "STRATZ league(id) null; window_filter main-event window 2026-05-01..05-03 excludes 17 earlier-stage matches."),
    "1win Essence I": (19656, "1win Essence I", "PROFESSIONAL", True, "STRATZ league(id) null; all 87 matches are main event (2026-05-02..05-11)."),
    "Games of the Future 2026": (19917, "Games of the Future 2026", "PROFESSIONAL", True, "STRATZ league(id) null; all 72 matches are main event (2026-07-31..08-05)."),
    "1win Essence II": (20009, "1win Essence II", "PROFESSIONAL", True, "STRATZ league(id) null; all 60 matches are main event (2026-07-30..08-05)."),
}

COVERED_BY: dict[str, str] = {
    # Same STRATZ league_id as Spring; do not emit a duplicate yaml row.
    "1win Series Dota 2 Fall": "1win Series Dota 2 Spring",
    "1win Series Dota 2 Summer": "1win Series Dota 2 Spring",
}

EXCLUDED_QUALIFIERS = [
    (17299, "ESL One Bangkok 2024 Qualifiers powered by Intel", "QUALIFIER", "Qualifier for ESL One Bangkok 2024.", "PROFESSIONAL"),
    (17525, "FISSURE PLAYGROUND - Closed Qualifiers - Americas", "QUALIFIER", "Qualifier for FISSURE PLAYGROUND 1.", "PROFESSIONAL"),
    (17628, "DreamLeague Season 25 Qualifiers powered by Intel", "QUALIFIER", "Qualifier for DreamLeague Season 25.", "PROFESSIONAL"),
    (17629, "ESL One Raleigh 2025 Qualifiers", "QUALIFIER", "Qualifier for ESL One Raleigh 2025.", "PROFESSIONAL"),
    (17874, "DreamLeague Season 26 Qualifiers", "QUALIFIER", "Qualifier for DreamLeague Season 26.", "PROFESSIONAL"),
    (18210, "Esports World Cup 2025 Qualifiers", "QUALIFIER", "Qualifier for Esports World Cup 2025.", "PROFESSIONAL"),
    (18629, "DreamLeague Season 27 Qualifiers", "QUALIFIER", "Qualifier for DreamLeague Season 27; STRATZ league(id) null.", "PROFESSIONAL"),
    (16776, "Elite League Season 2 EEU Closed Qualifiers – presented by ESB", "QUALIFIER", "Qualifier for Elite League Season 2.", "PROFESSIONAL"),
    (19089, "DreamLeague Season 28 Qualifiers", "QUALIFIER", "Qualifier for DreamLeague Season 28; STRATZ league(id) null.", "PROFESSIONAL"),
    (19090, "ESL One Birmingham 2026 Qualifiers", "QUALIFIER", "Qualifier for ESL One Birmingham 2026; STRATZ league(id) null.", "PROFESSIONAL"),
]

AUDIT_ONLY = [
    (14892, "DPC 2023 WEU Winter Tour Division I – presented by PGL", "T2", "Pre-2024; out of scope.", "DPC_LEAGUE"),
    (14050, "DPC NA Division II Spring Tour - 2021/2022 - ESL One Spring presented by Intel", "T2", "Pre-2024; out of scope.", "DPC_LEAGUE"),
    (15693, "Road to TI 2023 - WEU Regional Qualifiers", "QUALIFIER", "Pre-2024 DPC qualifier.", "DPC_LEAGUE_QUALIFIER"),
    (10979, "StarLadder ImbaTV Dota2 Minor #2", "MINOR", "Pre-2024; out of scope.", "MINOR"),
    (17807, "RD2L Season 38", "EXCLUDED", "Amateur; not on Liquipedia T1/T2.", "AMATEUR"),
    (16446, "1win Series Dota 2 Summer (dead STRATZ mapping)", "EXCLUDED", "STRATZ league 16446 has 0 matches; 1win Summer matches live under the 16427 umbrella. Kept for audit only.", "PROFESSIONAL"),
]

# Catalog-null STRATZ leagues: `ingest_stratz_leagues --all` uses match(id).
MATCH_ID_FETCH_LEAGUE_IDS = {
    17419,
    17420,
    18633,
    18863,
    18920,
    18937,
    18988,
    19099,
    19101,
    19130,
    19239,
    19255,
    19269,
    19290,
    19422,
    19435,
    19543,
    19575,
    19656,
    19696,
    19719,
    19785,
    19917,
    20009,
}

# Leagues whose STRATZ league also contains qualifiers/earlier stages that
# share the league id. `window_filter` restricts match-ID ingest to the
# Liquipedia main-event date window. Dates are the Liquipedia main event.
WINDOW_FILTER_LEAGUE_DATES: dict[int, tuple[str, str]] = {
    19130: ("2026-01-30", "2026-02-01"),
    19255: ("2026-04-01", "2026-04-11"),
    19290: ("2026-02-04", "2026-02-12"),
    19575: ("2026-05-01", "2026-05-03"),
    19656: ("2026-05-02", "2026-05-11"),
    19917: ("2026-07-31", "2026-08-05"),
    20009: ("2026-07-30", "2026-08-05"),
}


def fmt_entry(
    league_id: int,
    name: str,
    tier: str,
    in_scope: bool,
    notes: str | None,
    stratz_tier: str,
) -> str:
    lines = [
        f"  - league_id: {league_id}",
        f"    name: \"{name}\"",
        f"    stratz_tier: \"{stratz_tier}\"",
        f"    liquipedia_tier: \"{tier}\"",
        f"    in_scope: {'true' if in_scope else 'false'}",
    ]
    if league_id in MATCH_ID_FETCH_LEAGUE_IDS:
        lines.append("    fetch_mode: match_ids")
    if league_id in WINDOW_FILTER_LEAGUE_DATES:
        start, end = WINDOW_FILTER_LEAGUE_DATES[league_id]
        lines.append("    window_filter: true")
        lines.append(f"    start_date: {start}")
        lines.append(f"    end_date: {end}")
    if notes:
        lines.append(f"    notes: \"{notes}\"")
    return "\n".join(lines)


def fmt_preserved_entry(entry: dict) -> str:
    """Emit an existing leagues.yaml row (used to preserve T3 on regenerate)."""
    in_scope = bool(entry.get("in_scope", False))
    lines = [
        f"  - league_id: {entry['league_id']}",
        f"    name: \"{entry['name']}\"",
        f"    stratz_tier: \"{entry.get('stratz_tier') or 'PROFESSIONAL'}\"",
        f"    liquipedia_tier: \"{entry['liquipedia_tier']}\"",
        f"    in_scope: {'true' if in_scope else 'false'}",
    ]
    fetch_mode = entry.get("fetch_mode")
    if fetch_mode and fetch_mode != "league":
        lines.append(f"    fetch_mode: {fetch_mode}")
    if entry.get("window_filter"):
        lines.append("    window_filter: true")
    if entry.get("source"):
        lines.append(f"    source: \"{entry['source']}\"")
    start = entry.get("start_date")
    end = entry.get("end_date")
    if start is not None:
        lines.append(f"    start_date: {start}")
    if end is not None:
        lines.append(f"    end_date: {end}")
    if entry.get("notes"):
        notes = str(entry["notes"]).replace('"', '\\"')
        lines.append(f"    notes: \"{notes}\"")
    return "\n".join(lines)


def load_preserved_t3_entries() -> list[dict]:
    if not LEAGUES_YAML_PATH.is_file():
        return []
    raw = yaml.safe_load(LEAGUES_YAML_PATH.read_text(encoding="utf-8")) or {}
    return [
        entry
        for entry in (raw.get("leagues") or [])
        if entry.get("liquipedia_tier") == "T3"
    ]


def main() -> None:
    liquipedia = yaml.safe_load(LIQUIPEDIA_EVENTS_PATH.read_text(encoding="utf-8"))
    events = liquipedia.get("events") or []

    print(HEADER)
    print("\nleagues:")
    print("\n  # --- Liquipedia Tier 1 / Tier 2 main events (2024+) ---")

    pending: list[str] = []
    shared: list[str] = []
    seen_ids: set[int] = set()

    for ev in events:
        lp_name = ev["liquipedia_name"]
        if lp_name in COVERED_BY:
            continue

        lp_tier = ev["liquipedia_tier"]
        if lp_tier == "T3":
            continue
        year = ev.get("year")
        extra = ev.get("notes")

        mapped = STRATZ_ID_MAP.get(lp_name)
        if mapped is None:
            # Events that declare their own STRATZ league_id in the
            # manifest (e.g. ACL 2025 sharing 17875 with a T3 league) are
            # resolved by event-level assignment, not a league row --
            # emitting a second leagues.yaml row for the same league_id
            # would duplicate the id. Everything else without a mapping
            # stays PENDING until a STRATZ id is resolved.
            if ev.get("league_id"):
                shared.append(f"{year} {lp_tier} {lp_name} (league_id={ev['league_id']})")
            else:
                pending.append(f"{year} {lp_tier} {lp_name}")
            continue

        league_id, stratz_name, stratz_tier, in_scope, map_notes = mapped
        if league_id in seen_ids:
            notes_parts = [f"Liquipedia: {lp_name} ({year}).", map_notes or "", extra or ""]
        else:
            seen_ids.add(league_id)
            notes_parts = [f"Liquipedia: {lp_name} ({year}).", extra or "", map_notes or ""]
        notes = " ".join(p for p in notes_parts if p).strip()

        print()
        print(fmt_entry(league_id, stratz_name, lp_tier, in_scope, notes, stratz_tier))

    if shared:
        print("\n  # --- Shared-league events (event-level assignment, no league row) ---")
        for line in shared:
            print(f"  # {line}")

    if pending:
        print("\n  # --- Liquipedia events without STRATZ league_id (add to STRATZ_ID_MAP) ---")
        for line in pending:
            print(f"  # PENDING: {line}")

    print("\n  # --- Qualifiers (Liquipedia; excluded from ingestion) ---")
    for league_id, name, tier, notes, stratz_tier in EXCLUDED_QUALIFIERS:
        print()
        print(fmt_entry(league_id, name, tier, False, notes, stratz_tier))

    print("\n  # --- Pre-2024 / non-Liquipedia-T1/T2 (audit only) ---")
    for league_id, name, tier, notes, stratz_tier in AUDIT_ONLY:
        print()
        print(fmt_entry(league_id, name, tier, False, notes, stratz_tier))

    t3_entries = load_preserved_t3_entries()
    if t3_entries:
        print("\n  # --- Liquipedia Tier 3 professional events (preserved) ---")
        for entry in t3_entries:
            print()
            print(fmt_preserved_entry(entry))


if __name__ == "__main__":
    main()
