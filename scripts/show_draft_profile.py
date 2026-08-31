"""Inspect Radiant and Dire draft profiles immediately before a match.

Developer-facing only. Reads the canonical Parquet dataset and prints a
side-by-side summary of each side's five drafted heroes via leakage-safe
Player × Hero, Team × Hero, and Hero Meta state. Does not write files,
does not change training features, and is not a production API.

Usage:
    uv run python scripts/show_draft_profile.py --match-id 1234567890
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from dota_predictor.features import (
    DRAFT_PROFILE_METRIC_COLUMNS,
    MATCHES_VIEW,
    build_draft_profile,
    connect,
    load_feature_store_config,
    load_reference_store_config,
    register_reference_views,
)
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show Radiant and Dire draft profiles immediately before a "
            "canonical match."
        )
    )
    parser.add_argument(
        "--match-id",
        type=int,
        required=True,
        help="Canonical match_id whose side-level draft profile to inspect.",
    )
    return parser


def _format_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number.is_integer() and abs(number) < 1e12:
            return str(int(number))
        return f"{number:.3f}"
    return str(value)


def _side_by_side(frame: pd.DataFrame) -> pd.DataFrame:
    radiant = frame.loc[frame["side"] == "RADIANT"]
    dire = frame.loc[frame["side"] == "DIRE"]
    if len(radiant) != 1 or len(dire) != 1:
        raise ValueError(
            f"expected exactly one Radiant and one Dire row, got "
            f"Radiant={len(radiant)} Dire={len(dire)}"
        )
    rad = radiant.iloc[0]
    dire_row = dire.iloc[0]
    rows = [
        {
            "metric": column,
            "RADIANT": _format_value(rad[column]),
            "DIRE": _format_value(dire_row[column]),
        }
        for column in DRAFT_PROFILE_METRIC_COLUMNS
        if column in frame.columns
    ]
    return pd.DataFrame(rows)


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

    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.max_colwidth", 40)

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
            SELECT match_id, start_time, game_version_id,
                   radiant_team_id, dire_team_id
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
        frame = build_draft_profile(store, match_id=args.match_id).to_frame()

    info = match_rows.iloc[0]
    radiant = frame.loc[frame["side"] == "RADIANT"].iloc[0]
    dire = frame.loc[frame["side"] == "DIRE"].iloc[0]
    n_null_sv = int(frame["mean_same_version_contest_rate"].isna().sum())
    n_null_share = int(frame["mean_player_prior_hero_share"].isna().sum())

    print("=== Draft profile immediately before match ===")
    print(f"match_id: {int(info['match_id'])}")
    print(f"start_time: {info['start_time']}")
    print(f"game_version_id: {info['game_version_id']}")
    print(f"rows: {len(frame)} (expected 2)")
    print(
        f"Radiant team_id: {int(radiant['team_id'])}  "
        f"Dire team_id: {int(dire['team_id'])}"
    )
    print(f"NULL mean_same_version_contest_rate sides: {n_null_sv}")
    print(f"NULL mean_player_prior_hero_share sides: {n_null_share}")
    print(
        "five drafted heroes per side → Player × Hero / Team × Hero / Hero Meta "
        "→ side-level profile; current match result is not used"
    )
    print()
    print(_side_by_side(frame).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
