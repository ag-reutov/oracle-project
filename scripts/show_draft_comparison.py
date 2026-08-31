"""Inspect Radiant − Dire draft-profile differences for a match.

Developer-facing only. Reads the canonical Parquet dataset, reuses the
leakage-safe side-level Draft Profile, and prints Radiant − Dire
differences. Does not write files, does not change training features,
does not compute a composite draft score, and is not a production API.

Usage:
    uv run python scripts/show_draft_comparison.py --match-id 1234567890
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
    draft_comparison_from_profile,
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
            "Show Radiant − Dire draft-profile differences immediately "
            "before a canonical match."
        )
    )
    parser.add_argument(
        "--match-id",
        type=int,
        required=True,
        help="Canonical match_id whose Radiant − Dire comparison to inspect.",
    )
    return parser


def _format_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NULL"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number.is_integer() and abs(number) < 1e12:
            return str(int(number))
        return f"{number:.3f}"
    return str(value)


def _format_signed_diff(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NULL"
    number = float(value)
    if number.is_integer() and abs(number) < 1e12:
        integer = int(number)
        if integer > 0:
            return f"+{integer}"
        return str(integer)
    if number > 0:
        return f"+{number:.3f}"
    return f"{number:.3f}"


def _orientation_label(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NULL (either side missing)"
    number = float(value)
    if number > 0:
        return "Radiant higher"
    if number < 0:
        return "Dire higher"
    return "equal"


def _side_and_diff_table(
    profile: pd.DataFrame, comparison: pd.DataFrame
) -> pd.DataFrame:
    radiant = profile.loc[profile["side"] == "RADIANT"]
    dire = profile.loc[profile["side"] == "DIRE"]
    if len(radiant) != 1 or len(dire) != 1 or len(comparison) != 1:
        raise ValueError(
            f"expected 1 Radiant, 1 Dire, 1 comparison row; got "
            f"Radiant={len(radiant)} Dire={len(dire)} "
            f"comparison={len(comparison)}"
        )
    rad = radiant.iloc[0]
    dire_row = dire.iloc[0]
    diff_row = comparison.iloc[0]
    rows = []
    for column in DRAFT_PROFILE_METRIC_COLUMNS:
        diff_column = f"{column}_diff"
        diff_value = diff_row[diff_column]
        rows.append(
            {
                "metric": column,
                "RADIANT": _format_value(rad[column]),
                "DIRE": _format_value(dire_row[column]),
                "Radiant − Dire": _format_signed_diff(diff_value),
                "orientation": _orientation_label(diff_value),
            }
        )
    return pd.DataFrame(rows)


def _team_label(team_id: object, name: object) -> str:
    team_text = _format_value(team_id)
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return team_text
    name_text = str(name).strip()
    if not name_text:
        return team_text
    return f"{team_text} ({name_text})"


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
        print(
            f"Canonical matches file not found: {config.matches_path}",
            file=sys.stderr,
        )
        return 1

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.max_colwidth", 48)

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
                   radiant_team_id, dire_team_id,
                   radiant_team_name_observed, dire_team_name_observed
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
        profile = build_draft_profile(store, match_id=args.match_id).to_frame()

    comparison = draft_comparison_from_profile(profile)
    info = match_rows.iloc[0]
    diff_row = comparison.iloc[0]
    n_null_sv = int(pd.isna(diff_row["mean_same_version_contest_rate_diff"]))
    n_null_share = int(pd.isna(diff_row["mean_player_prior_hero_share_diff"]))

    print("=== Draft comparison (Radiant − Dire) ===")
    print(f"match_id: {int(info['match_id'])}")
    print(f"start_time: {info['start_time']}")
    print(f"game_version_id: {info['game_version_id']}")
    print(f"rows: {len(comparison)} (expected 1)")
    print(
        "Radiant team: "
        + _team_label(
            diff_row["radiant_team_id"], info["radiant_team_name_observed"]
        )
    )
    print(
        "Dire team: "
        + _team_label(diff_row["dire_team_id"], info["dire_team_name_observed"])
    )
    print(
        "Sign convention: diff = Radiant − Dire. "
        "Positive → Radiant higher; negative → Dire higher. "
        "Count fields (e.g. players with zero prior games) are not "
        "sign-flipped to make 'good' positive."
    )
    print(f"NULL mean_same_version_contest_rate_diff: {n_null_sv}")
    print(f"NULL mean_player_prior_hero_share_diff: {n_null_share}")
    print(
        "side-level Draft Profile → Radiant − Dire; "
        "no new historical aggregation; current match result is not used"
    )
    print()
    print(_side_and_diff_table(profile, comparison).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
