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

HEADER = dedent(
    """
    # Curated Tier 1 / Tier 2 professional Dota 2 league registry.
    #
    # EVENT LIST (which tournaments, which tier): Liquipedia
    #   - https://liquipedia.net/dota2/Tier_1_Tournaments
    #   - https://liquipedia.net/dota2/Tier_2_Tournaments
    #   - Canonical manifest: config/liquipedia_events_2024plus.yaml
    #
    # STRATZ league_id: resolved via Dotabuff / Spectral / match leagueId probes
    #   (see scripts/build_leagues_yaml_from_liquipedia.py STRATZ_ID_MAP).
    #   Do NOT use STRATZ LeagueTier to decide T1/T2 scope.
    #
    # Scope: Liquipedia T1/T2 main events from 2024-01-01 through present.
    # Pre-2024 DPC-era rows kept for audit only (in_scope: false).
    # Qualifiers listed but excluded (in_scope: false).
    #
    # Sync: scripts/load_league_registry.py → leagues / ingestion_leagues tables.
    """
).strip()

# liquipedia_name -> (league_id, stratz_display_name, stratz_tier, in_scope, extra_notes)
# in_scope false when STRATZ league(id) GraphQL returns null (ingest blocked).
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
    "BLAST Slam IV": (17419, "SLAM IV", "PROFESSIONAL", False, "STRATZ league(id) null but match leagueId 17419 confirmed; ingest blocked until API exposes league."),
    "FISSURE PLAYGROUND 2": (18863, "FISSURE PLAYGROUND 2", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "PGL Wallachia Season 6": (18920, "PGL Wallachia 2025 Season 6", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "BLAST Slam V": (17420, "SLAM V", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "DreamLeague Season 27": (18988, "DreamLeague Season 27", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    # 2025 T2
    "FISSURE Special": (18046, "FISSURE Special", "PROFESSIONAL", True, None),
    "FISSURE Universe Episode 5": (18107, "FISSURE Universe Episode 5", "PROFESSIONAL", True, None),
    "FISSURE Universe Episode 7": (18633, "FISSURE Universe Episode 7", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "AsiaPro League Season 2": (17622, "AsiaPro League S2", "PROFESSIONAL", True, None),
    "Games of the Future 2025 Abu Dhabi": (18937, "Games of the Future 2025 Abu Dhabi", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    # 2026
    "FISSURE Universe Episode 8": (19239, "FISSURE Universe Episode 8", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "BLAST SLAM VI": (19099, "BLAST SLAM VI", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "DreamLeague Season 28": (19269, "DreamLeague Season 28", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "PGL Wallachia Season 7": (19435, "PGL Wallachia 2026 Season 7", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "ESL One Birmingham 2026": (19422, "ESL One Birmingham 2026", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "PGL Wallachia Season 8": (19543, "PGL Wallachia 2026 Season 8", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "DreamLeague Season 29": (19696, "DreamLeague Season 29", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
    "BLAST SLAM VII": (19101, "BLAST SLAM VII", "PROFESSIONAL", False, "STRATZ league(id) null but match leagueId likely 19101; ingest blocked until API exposes league."),
    "Esports World Cup 2026": (19785, "Esports World Cup 2026", "PROFESSIONAL", False, "STRATZ league(id) null at verification; pending ingest."),
}

COVERED_BY: dict[str, str] = {
    # Same STRATZ league_id as Spring; do not emit a duplicate yaml row.
    "1win Series Dota 2 Fall": "1win Series Dota 2 Spring",
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
]


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
    if notes:
        lines.append(f"    notes: \"{notes}\"")
    return "\n".join(lines)


def main() -> None:
    liquipedia = yaml.safe_load(LIQUIPEDIA_EVENTS_PATH.read_text(encoding="utf-8"))
    events = liquipedia.get("events") or []

    print(HEADER)
    print("\nleagues:")
    print("\n  # --- Liquipedia Tier 1 / Tier 2 main events (2024+) ---")

    pending: list[str] = []
    seen_ids: set[int] = set()

    for ev in events:
        lp_name = ev["liquipedia_name"]
        if lp_name in COVERED_BY:
            continue

        lp_tier = ev["liquipedia_tier"]
        year = ev.get("year")
        extra = ev.get("notes")

        mapped = STRATZ_ID_MAP.get(lp_name)
        if mapped is None:
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


if __name__ == "__main__":
    main()
