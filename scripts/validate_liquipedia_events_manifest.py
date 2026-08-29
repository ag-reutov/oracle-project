"""Compare config/liquipedia_events_2024plus.yaml against Liquipedia tier pages.

Liquipedia Tier_1_Tournaments and Tier_2_Tournaments are authoritative for
which events belong in each tier. This script fetches those pages via the
Liquipedia MediaWiki API (gzip required) and reports gaps vs our manifest.

Usage:
    uv run python scripts/validate_liquipedia_events_manifest.py
    uv run python scripts/validate_liquipedia_events_manifest.py --years 2024 2025
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "config" / "liquipedia_events_2024plus.yaml"
USER_AGENT = "dota-predictor-liquipedia-validate/0.1"
API_BASE = "https://liquipedia.net/dota2/api.php"

COVERED_ALIASES = {
    # Covered by another manifest row (same STRATZ league or umbrella event).
    "1win Series Dota 2 Fall": "1win Series Dota 2 Spring",
}


def normalize_name(name: str) -> str:
    text = name.lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"[:/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def names_match(manifest_name: str, liquipedia_name: str) -> bool:
    a, b = normalize_name(manifest_name), normalize_name(liquipedia_name)
    return a in b or b in a or a.replace(" ", "") in b.replace(" ", "")


def fetch_parsed_html(page: str, retries: int = 3) -> str:
    page_slug = page.replace(" ", "_")
    url = f"{API_BASE}?action=parse&page={page_slug}&prop=text&format=json"
    cmd = [
        "curl",
        "-sL",
        "-H",
        "Accept-Encoding: gzip",
        "-A",
        USER_AGENT,
        "--compressed",
        url,
    ]
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = subprocess.check_output(cmd, text=True)
            if not raw.strip():
                raise ValueError("empty Liquipedia API response")
            payload = json.loads(raw)
            return payload["parse"]["text"]["*"]
        except (json.JSONDecodeError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                subprocess.run(["sleep", "2"], check=False)
    raise RuntimeError(f"Failed to fetch Liquipedia page {page}: {last_error}")


def section_chunk(html: str, year: str) -> str:
    match = re.search(rf"<h3 id=\"{year}\">{year}</h3>", html)
    if not match:
        return ""
    start = match.end()
    next_year = re.search(r"<h3 id=\"\d{4}\">", html[start:])
    end = start + next_year.start() if next_year else len(html)
    return html[start:end]


def liquipedia_events_in_section(chunk: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"data-sort-value=\"([^\"]+)\"", chunk):
        if re.match(
            r"\$|Dec |Jan |Feb |Mar |Apr |May |Jun |Jul |Aug |Sep |Oct |Nov ",
            raw,
        ):
            continue
        if raw in seen:
            continue
        seen.add(raw)
        names.append(raw)
    return names


def load_manifest() -> dict[tuple[int, str], list[dict]]:
    data = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    events = data.get("events") or []
    grouped: dict[tuple[int, str], list[dict]] = {}
    for event in events:
        key = (int(event["year"]), event["liquipedia_tier"])
        grouped.setdefault(key, []).append(event)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Liquipedia event manifest.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2024, 2025, 2026],
        help="Years to compare (default: 2024 2025 2026).",
    )
    args = parser.parse_args()

    manifest_by = load_manifest()
    pages = {
        "T1": "Tier_1_Tournaments",
        "T2": "Tier_2_Tournaments",
    }

    exit_code = 0
    for tier, page in pages.items():
        html = fetch_parsed_html(page)
        for year in args.years:
            lp_names = liquipedia_events_in_section(section_chunk(html, str(year)))
            ours = manifest_by.get((year, tier), [])
            our_names = [e["liquipedia_name"] for e in ours]

            missing_from_manifest = [
                lp_name
                for lp_name in lp_names
                if not any(names_match(m, lp_name) for m in our_names)
            ]
            missing_from_lp = []
            for event in ours:
                name = event["liquipedia_name"]
                if name in COVERED_ALIASES:
                    continue
                if not any(names_match(name, lp_name) for lp_name in lp_names):
                    missing_from_lp.append(name)

            print(f"\n{tier} {year}: liquipedia={len(lp_names)} manifest={len(our_names)}")
            if lp_names:
                print("  liquipedia:", lp_names)
            if missing_from_manifest:
                exit_code = 1
                print("  ADD to manifest:", missing_from_manifest)
            if missing_from_lp:
                exit_code = 1
                print("  NOT on Liquipedia tier page:", missing_from_lp)

    if exit_code:
        print("\nManifest drift detected — update config/liquipedia_events_2024plus.yaml.")
    else:
        print("\nManifest matches Liquipedia tier pages for requested years.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
