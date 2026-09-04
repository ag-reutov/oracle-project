"""Discover Liquipedia Tier 3 events and resolve Valve/STRATZ league IDs.

Reads the year pages Tier_3_Tournaments/{2024,2025,2026} via the MediaWiki
API, then fetches each tournament page's wikitext for ``|leagueid=``.

Resumable: writes progress after every page to
``data/interim/tier3_discovery_with_ids.json``.

Usage:
    uv run python scripts/discover_tier3_league_ids.py
    uv run python scripts/discover_tier3_league_ids.py --resume
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "interim" / "tier3_discovery_with_ids.json"
LIST_PATH = REPO_ROOT / "data" / "interim" / "tier3_discovery.json"
UA = "dota-predictor-research/1.0 (tier3-discovery; https://github.com/local)"
MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        1,
    )
}


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept-Encoding": "gzip"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return json.loads(data.decode())


def parse_date_range(text: str) -> tuple[str | None, str | None]:
    text = text.replace("\xa0", " ").strip().replace("–", "-").replace("—", "-")
    m = re.match(
        r"([A-Z][a-z]{2})\s+(\d{1,2})\s*-\s*([A-Z][a-z]{2})\s+(\d{1,2}),\s*(\d{4})",
        text,
    )
    if m:
        y = int(m.group(5))
        return (
            f"{y:04d}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}",
            f"{y:04d}-{MONTHS[m.group(3)]:02d}-{int(m.group(4)):02d}",
        )
    m = re.match(
        r"([A-Z][a-z]{2})\s+(\d{1,2})\s*-\s*(\d{1,2}),\s*(\d{4})",
        text,
    )
    if m:
        y = int(m.group(4))
        mo = MONTHS[m.group(1)]
        return (
            f"{y:04d}-{mo:02d}-{int(m.group(2)):02d}",
            f"{y:04d}-{mo:02d}-{int(m.group(3)):02d}",
        )
    m = re.match(
        r"([A-Z][a-z]{2})\s+(\d{1,2}),\s*(\d{4})\s*-\s*([A-Z][a-z]{2})\s+(\d{1,2}),\s*(\d{4})",
        text,
    )
    if m:
        return (
            f"{int(m.group(3)):04d}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}",
            f"{int(m.group(6)):04d}-{MONTHS[m.group(4)]:02d}-{int(m.group(5)):02d}",
        )
    return None, None


def discover_event_list(delay_s: float) -> list[dict]:
    events: list[dict] = []
    for year in (2024, 2025, 2026):
        url = (
            "https://liquipedia.net/dota2/api.php?action=parse"
            f"&page=Tier_3_Tournaments/{year}&prop=text&format=json"
        )
        doc = _http_get_json(url)
        html = doc["parse"]["text"]["*"]
        rows = re.findall(r'<tr class="[^"]*row--body[^"]*">(.*?)</tr>', html, re.S)
        for row in rows:
            tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
            if len(tds) < 3:
                continue
            link = re.search(r'<a href="/dota2/([^"]+)"[^>]*>([^<]+)</a>', tds[1])
            if not link:
                continue
            page = link.group(1)
            name = link.group(2).strip()
            if not name or "(page does not exist)" in name:
                continue
            date_text = re.sub("<[^>]+>", "", tds[2]).strip()
            start, end = parse_date_range(date_text)
            prize = None
            prize_text = re.sub("<[^>]+>", "", tds[3]).strip() if len(tds) > 3 else ""
            pm = re.search(r"\$([0-9,]+)", prize_text)
            if pm:
                prize = int(pm.group(1).replace(",", ""))
            events.append(
                {
                    "year": year,
                    "liquipedia_name": name,
                    "page": page,
                    "startdate": start,
                    "enddate": end,
                    "date_text": date_text,
                    "prize_usd": prize,
                    "liquipedia_tier": "T3",
                }
            )
        time.sleep(delay_s)

    # Scope: start on/after 2024-01-01, start on/before 2026-09-03.
    scoped = [
        e
        for e in events
        if e["startdate"]
        and "2024-01-01" <= e["startdate"] <= "2026-09-03"
    ]
    seen: set[str] = set()
    dedup: list[dict] = []
    for e in sorted(scoped, key=lambda x: (x["startdate"] or "", x["liquipedia_name"])):
        if e["page"] in seen:
            continue
        seen.add(e["page"])
        dedup.append(e)
    LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIST_PATH.write_text(json.dumps(dedup, indent=2), encoding="utf-8")
    return dedup


def extract_leagueid(wikitext: str) -> tuple[int | None, str | None]:
    m = re.search(r"\|\s*leagueid\s*=\s*(\d+)", wikitext, re.I)
    if m:
        return int(m.group(1)), "leagueid"
    for field in ("valve-id", "valveid", "valve_id"):
        m = re.search(rf"\|\s*{field}\s*=\s*(\d+)", wikitext, re.I)
        if m:
            return int(m.group(1)), field
    return None, None


def fetch_wikitext(page: str) -> str:
    """Fetch raw wikitext via action=raw (lighter than action=parse)."""
    url = (
        "https://liquipedia.net/dota2/index.php?title="
        + urllib.request.quote(page)
        + "&action=raw"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept-Encoding": "gzip"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return data.decode("utf-8", errors="replace")


def resolve_ids(events: list[dict], *, delay_s: float, resume: bool) -> list[dict]:
    by_page: dict[str, dict] = {}
    if resume and OUT_PATH.exists():
        for row in json.loads(OUT_PATH.read_text(encoding="utf-8")):
            # Keep Liquipedia-sourced IDs; allow retry of stratz-only / errors.
            src = row.get("league_id_source") or ""
            if row.get("league_id") and (
                src == "leagueid" or src.startswith("leagueid")
            ):
                by_page[row["page"]] = row

    results: list[dict] = []
    for i, event in enumerate(events):
        if event["page"] in by_page:
            results.append(by_page[event["page"]])
            print(
                f"[{i+1}/{len(events)}] SKIP {by_page[event['page']]['league_id']} "
                f"{event['liquipedia_name']}",
                flush=True,
            )
            continue

        row = None
        for attempt in range(1, 4):
            try:
                wt = fetch_wikitext(event["page"])
                lid, src = extract_leagueid(wt)
                row = {**event, "league_id": lid, "league_id_source": src}
                status = "OK" if lid else "NO_ID"
                print(
                    f"[{i+1}/{len(events)}] {status} {lid} {event['liquipedia_name']}",
                    flush=True,
                )
                break
            except urllib.error.HTTPError as exc:
                wait = delay_s * (3 ** attempt)
                print(
                    f"[{i+1}/{len(events)}] ERR {event['liquipedia_name']}: "
                    f"HTTP {exc.code} (attempt {attempt}/3, sleep {wait:.0f}s)",
                    flush=True,
                )
                if exc.code == 429 and attempt < 3:
                    time.sleep(wait)
                    continue
                row = {
                    **event,
                    "league_id": None,
                    "league_id_source": None,
                    "error": f"HTTP {exc.code}",
                }
                if exc.code == 429:
                    time.sleep(max(delay_s * 4, 30.0))
                break
            except Exception as exc:  # noqa: BLE001 - resumable discovery
                row = {
                    **event,
                    "league_id": None,
                    "league_id_source": None,
                    "error": str(exc),
                }
                print(
                    f"[{i+1}/{len(events)}] ERR {event['liquipedia_name']}: {exc}",
                    flush=True,
                )
                break
        if row is None:
            row = {
                **event,
                "league_id": None,
                "league_id_source": None,
                "error": "exhausted retries",
            }

        results.append(row)
        OUT_PATH.write_text(
            json.dumps(_merge_progress(events, results, by_page), indent=2),
            encoding="utf-8",
        )
        time.sleep(delay_s)

    final = _merge_progress(events, results, by_page)
    OUT_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def _merge_progress(
    events: list[dict], results: list[dict], prior: dict[str, dict]
) -> list[dict]:
    by_page = dict(prior)
    for row in results:
        by_page[row["page"]] = row
    out = []
    for event in events:
        if event["page"] in by_page:
            out.append(by_page[event["page"]])
        else:
            out.append({**event, "league_id": None, "league_id_source": None})
    return out


def summarize(rows: list[dict]) -> None:
    with_id = [r for r in rows if r.get("league_id")]
    missing = [r for r in rows if not r.get("league_id")]
    print("---")
    print(f"total={len(rows)} with_id={len(with_id)} missing={len(missing)}")
    print("years", Counter(r["year"] for r in rows))
    dups: dict[int, list[str]] = defaultdict(list)
    for r in with_id:
        dups[int(r["league_id"])].append(r["liquipedia_name"])
    multi = {k: v for k, v in dups.items() if len(v) > 1}
    print(f"duplicate_league_ids={len(multi)}")
    for lid, names in sorted(multi.items()):
        print(f"  {lid}: {names}")
    if missing:
        print("missing:")
        for r in missing:
            print(f"  {r['liquipedia_name']} ({r.get('error')})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between requests")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="Only refresh event list")
    args = parser.parse_args()

    if LIST_PATH.exists() and args.resume:
        events = json.loads(LIST_PATH.read_text(encoding="utf-8"))
        print(f"Loaded {len(events)} events from {LIST_PATH}")
    else:
        print("Discovering Tier 3 event list from Liquipedia year pages...")
        events = discover_event_list(args.delay)
        print(f"Wrote {len(events)} events to {LIST_PATH}")

    if args.list_only:
        return 0

    print(f"Resolving league IDs (delay={args.delay}s, resume={args.resume})...")
    rows = resolve_ids(events, delay_s=args.delay, resume=args.resume)
    summarize(rows)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
