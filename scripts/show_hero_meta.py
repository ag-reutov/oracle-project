"""Inspect descriptive hero-meta state immediately before a match.

Developer-facing only. Reads the canonical Parquet dataset and prints
the top heroes by same-version contest rate. Does not write files, does
not change training features, and is not a production API.

Usage:
    uv run python scripts/show_hero_meta.py --match-id 1234567890
    uv run python scripts/show_hero_meta.py --match-id 1234567890 --top 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from dota_predictor.features import (
    MATCHES_VIEW,
    build_hero_meta,
    connect,
    load_feature_store_config,
    load_reference_store_config,
    rank_hero_meta,
    register_reference_views,
)
from dota_predictor.utils.env import load_project_env

_DISPLAY_COLUMNS = (
    "hero_id",
    "hero_name",
    "same_version_prior_matches",
    "same_version_prior_picks",
    "same_version_prior_bans",
    "same_version_contest_rate",
    "same_version_win_rate",
    "recent_90d_contest_rate",
    "recent_90d_win_rate",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show the descriptive hero meta immediately before a canonical match."
        )
    )
    parser.add_argument(
        "--match-id",
        type=int,
        required=True,
        help="Canonical match_id whose pre-match hero meta to inspect.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of heroes to print (default: 15).",
    )
    return parser


def _format_rate(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return f"{float(value):.3f}"


def _display_frame(ranked: pd.DataFrame, *, top: int) -> pd.DataFrame:
    frame = ranked.head(top)
    columns = [c for c in _DISPLAY_COLUMNS if c in frame.columns]
    display = frame[columns].copy()
    if "hero_name" in display.columns:
        display["hero_name"] = display["hero_name"].fillna("")
    for column in (
        "same_version_contest_rate",
        "same_version_win_rate",
        "recent_90d_contest_rate",
        "recent_90d_win_rate",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_rate)
    return display


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.top < 1:
        print("--top must be >= 1", file=sys.stderr)
        return 1
    if args.match_id <= 0:
        print("--match-id must be a positive integer", file=sys.stderr)
        return 1

    root = _project_root()
    load_project_env(root)
    config = load_feature_store_config(root=root)
    if not config.matches_path.is_file():
        print(f"Canonical matches file not found: {config.matches_path}", file=sys.stderr)
        return 1

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 24)

    reference_config = load_reference_store_config(root=root)
    catalog_available = (
        reference_config.heroes_path.is_file()
        and reference_config.game_versions_path.is_file()
    )

    with connect(config) as store:
        if catalog_available:
            register_reference_views(store, reference_config)
        match_rows = store.sql(
            f"""
            SELECT match_id, start_time, game_version_id
            FROM {MATCHES_VIEW}
            WHERE match_id = {int(args.match_id)}
            """
        ).df()
        if match_rows.empty:
            print(
                f"match_id={args.match_id} is not in {config.matches_path}",
                file=sys.stderr,
            )
            return 1
        frame = build_hero_meta(store, match_id=args.match_id).to_frame()

    info = match_rows.iloc[0]
    ranked = rank_hero_meta(frame)
    n_heroes = len(ranked)
    n_null_contest = int(ranked["same_version_contest_rate"].isna().sum())
    prior_matches = (
        int(ranked["same_version_prior_matches"].iloc[0]) if n_heroes else 0
    )

    print("=== Hero meta immediately before match ===")
    print(f"match_id: {int(info['match_id'])}")
    print(f"start_time: {info['start_time']}")
    print(f"game_version_id: {info['game_version_id']}")
    print(f"hero universe rows: {n_heroes}")
    print(f"same_version_prior_matches (context size): {prior_matches}")
    print(
        f"heroes with NULL same_version_contest_rate (no same-version history): "
        f"{n_null_contest}"
    )
    print(
        "historical hero draft/result facts → aggregated over past matches only "
        "→ PRE_DRAFT historical state for this match"
    )
    print("sort: same_version_contest_rate desc, then prior_matches, then hero_id")
    print()
    print(_display_frame(ranked, top=args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
