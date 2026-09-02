"""Live STRATZ sample for Slice 11 player box-score landing.

Fetches the lightweight performance selection only. Does not persist.
Includes the three known parse-missing-position matches plus one
ordinary development match.

Usage:
    uv run python scripts/validate_slice11_player_performance.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from dota_predictor.data.canonical_schema import STRATZ_PLAYER_BOX_SCORE_FIELDS
from dota_predictor.data.stratz_mapping import canonical_match_from_stratz
from dota_predictor.ingestion.client import StratzClient
from dota_predictor.ingestion.config import (
    MissingStratzTokenError,
    load_ingestion_config,
)
from dota_predictor.utils.env import load_project_env

POSITION_MISSING_MATCH_IDS = (7570622406, 7570633219, 7704591293)
ORDINARY_DEVELOPMENT_MATCH_ID = 8461956309


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _summarize(players: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for field in STRATZ_PLAYER_BOX_SCORE_FIELDS:
        values = [player.get(field) for player in players]
        present = [value for value in values if value is not None]
        types = sorted({type(value).__name__ for value in present})
        nested = sum(1 for value in present if isinstance(value, (dict, list)))
        summary[field] = {
            "nulls": sum(value is None for value in values),
            "zeros": sum(value == 0 for value in present),
            "n": len(values),
            "min": min(present) if present else None,
            "max": max(present) if present else None,
            "types": types,
            "nested_objects": nested,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-id",
        type=int,
        action="append",
        dest="match_ids",
        help="Override the default sample match ids (repeatable).",
    )
    args = parser.parse_args()
    load_project_env(_project_root())
    try:
        config = load_ingestion_config()
    except MissingStratzTokenError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    match_ids = args.match_ids or [
        *POSITION_MISSING_MATCH_IDS,
        ORDINARY_DEVELOPMENT_MATCH_ID,
    ]
    fetched: list[int] = []
    all_players: list[dict[str, Any]] = []
    per_match: dict[int, dict[str, Any]] = {}

    with StratzClient(config) as client:
        for match_id in match_ids:
            payload = client.fetch_match_player_performance(match_id)
            if payload is None or not isinstance(payload.get("players"), list):
                print(
                    f"match {match_id}: STRATZ returned no player performance payload"
                )
                continue
            players = payload["players"]
            fetched.append(match_id)
            all_players.extend(players)
            per_match[match_id] = {
                "players": len(players),
                "summary": _summarize(players),
            }
            # Full MATCH_SELECTION parse on one ordinary/sample match.
            if match_id == match_ids[-1]:
                full = client.fetch_match(match_id)
                if full is None:
                    print(f"match {match_id}: full MATCH_SELECTION returned null")
                else:
                    canonical = canonical_match_from_stratz(full)
                    print(
                        f"full MATCH_SELECTION mapped match {match_id}: "
                        f"kills={canonical.radiant_box_scores[0].kills!r} "
                        f"level={canonical.radiant_box_scores[0].level!r}"
                    )

    print(f"matches fetched: {len(fetched)} {fetched}")
    print(f"player rows inspected: {len(all_players)}")
    combined = _summarize(all_players)
    for field, stats in combined.items():
        print(
            f"{field}: nulls={stats['nulls']}/{stats['n']} "
            f"zeros={stats['zeros']} min={stats['min']} max={stats['max']} "
            f"types={stats['types']} nested={stats['nested_objects']}"
        )
    for match_id in POSITION_MISSING_MATCH_IDS:
        if match_id not in per_match:
            continue
        kills_null = per_match[match_id]["summary"]["kills"]["nulls"]
        print(
            f"position-missing match {match_id}: "
            f"players={per_match[match_id]['players']} "
            f"kills_nulls={kills_null}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
