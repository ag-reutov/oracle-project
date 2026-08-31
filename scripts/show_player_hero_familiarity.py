"""Inspect player × hero historical familiarity immediately before a match.

Developer-facing only. Reads the canonical Parquet dataset and prints each
player's currently drafted hero plus leakage-safe historical hero
familiarity. Does not write files, does not change training features, and
is not a production API.

Usage:
    uv run python scripts/show_player_hero_familiarity.py --match-id 1234567890
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from dota_predictor.features import (
    MATCHES_VIEW,
    build_player_hero,
    connect,
    load_feature_store_config,
    load_reference_store_config,
    register_reference_views,
)
from dota_predictor.utils.env import load_project_env

_DISPLAY_COLUMNS = (
    "side",
    "player_id",
    "hero_id",
    "hero_name",
    "prior_games_on_hero",
    "prior_wins_on_hero",
    "prior_win_rate_on_hero",
    "prior_player_games",
    "prior_hero_share",
    "same_version_games_on_hero",
    "same_version_win_rate_on_hero",
    "recent_90d_games_on_hero",
    "recent_90d_hero_share",
    "days_since_last_played_hero",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show each player's drafted hero and historical hero familiarity "
            "immediately before a canonical match."
        )
    )
    parser.add_argument(
        "--match-id",
        type=int,
        required=True,
        help="Canonical match_id whose player×hero history to inspect.",
    )
    return parser


def _format_rate(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return f"{float(value):.3f}"


def _format_days(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return f"{float(value):.1f}"


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["side", "player_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    columns = [c for c in _DISPLAY_COLUMNS if c in ordered.columns]
    display = ordered[columns].copy()
    if "hero_name" in display.columns:
        display["hero_name"] = display["hero_name"].fillna("")
    for column in (
        "prior_win_rate_on_hero",
        "prior_hero_share",
        "same_version_win_rate_on_hero",
        "recent_90d_hero_share",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_rate)
    if "days_since_last_played_hero" in display.columns:
        display["days_since_last_played_hero"] = display[
            "days_since_last_played_hero"
        ].map(_format_days)
    return display


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.match_id <= 0:
        print("--match-id must be a positive integer", file=sys.stderr)
        return 1

    root = _project_root()
    load_project_env(root)
    config = load_feature_store_config(root=root)
    if not config.matches_path.is_file():
        print(f"Canonical matches file not found: {config.matches_path}", file=sys.stderr)
        return 1

    pd.set_option("display.width", 240)
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
        frame = build_player_hero(store, match_id=args.match_id).to_frame()

    info = match_rows.iloc[0]
    n_zero_hero = int((frame["prior_games_on_hero"] == 0).sum())
    n_null_rate = int(frame["prior_win_rate_on_hero"].isna().sum())
    n_null_days = int(frame["days_since_last_played_hero"].isna().sum())

    print("=== Player × hero familiarity immediately before match ===")
    print(f"match_id: {int(info['match_id'])}")
    print(f"start_time: {info['start_time']}")
    print(f"game_version_id: {info['game_version_id']}")
    print(f"player rows: {len(frame)}")
    print(f"players with zero prior games on this hero: {n_zero_hero}")
    print(f"NULL prior_win_rate_on_hero (zero hero games): {n_null_rate}")
    print(f"NULL days_since_last_played_hero (never played this hero): {n_null_days}")
    print(
        "historical player×hero facts → aggregated over past matches only "
        "→ current-draft familiarity for this match"
    )
    print("slot_in_side is lobby order, not Dota position 1-5")
    print()
    print(_display_frame(frame).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
