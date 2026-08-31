"""Print the current team Elo leaderboard from the canonical dataset.

Replays matches through the production Elo implementation
(`dota_predictor.features.team_elo`) and reports the terminal per-team
state. Does not change Elo parameters or update rules.

Usage:
    uv run python scripts/show_team_elo_rankings.py
    uv run python scripts/show_team_elo_rankings.py --top 30 --active-days 90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from dota_predictor.features import (
    DEFAULT_ACTIVE_DAYS,
    DEFAULT_ELO_CONFIG,
    MATCHES_VIEW,
    active_team_elo_cutoff,
    compute_team_elo_state,
    connect,
    filter_active_team_elo,
    load_feature_store_config,
    rank_team_elo_state,
    team_elo_trajectories,
)
from dota_predictor.utils.env import load_project_env

_MATCH_COLUMNS = (
    "match_id",
    "start_time",
    "radiant_team_id",
    "dire_team_id",
    "radiant_win",
    "radiant_team_name_observed",
    "dire_team_name_observed",
)

_LEADERBOARD_COLUMNS = (
    "rank",
    "team_id",
    "team_name",
    "elo",
    "n_matches",
    "wins",
    "losses",
    "last_match_timestamp",
    "elo_before_last_match",
    "elo_after_last_match",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show the latest team Elo leaderboard from the canonical dataset."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of rows to print for each leaderboard (default: 20).",
    )
    parser.add_argument(
        "--active-days",
        type=int,
        default=DEFAULT_ACTIVE_DAYS,
        help=(
            "A team is active if it played at least one match within this many "
            "days of the dataset's maximum start_time (default: "
            f"{DEFAULT_ACTIVE_DAYS}). Uses the dataset clock, not wall-clock now."
        ),
    )
    return parser


def latest_observed_team_names(matches: pd.DataFrame) -> pd.DataFrame:
    """Most recently observed display name per `team_id`.

    Names come from per-match `*_team_name_observed` columns -- not a
    stable registry. Missing names stay missing; nothing is invented.
    """
    required = ("radiant_team_name_observed", "dire_team_name_observed")
    empty = pd.DataFrame(
        columns=["team_id", "team_name", "n_distinct_observed_names"]
    )
    if any(column not in matches.columns for column in required):
        return empty

    radiant = matches[
        ["start_time", "radiant_team_id", "radiant_team_name_observed"]
    ].rename(
        columns={
            "radiant_team_id": "team_id",
            "radiant_team_name_observed": "team_name",
        }
    )
    dire = matches[["start_time", "dire_team_id", "dire_team_name_observed"]].rename(
        columns={"dire_team_id": "team_id", "dire_team_name_observed": "team_name"}
    )
    stacked = pd.concat([radiant, dire], ignore_index=True)
    stacked = stacked.sort_values("start_time", kind="stable")

    def _last_non_null(series: pd.Series) -> object:
        valid = series.dropna()
        valid = valid[valid.astype(str).str.strip() != ""]
        if valid.empty:
            return None
        return valid.iloc[-1]

    def _n_distinct_names(series: pd.Series) -> int:
        cleaned = series.dropna().astype(str).str.strip()
        cleaned = cleaned[cleaned != ""]
        return int(cleaned.nunique())

    return (
        stacked.groupby("team_id", sort=False)
        .agg(
            team_name=("team_name", _last_non_null),
            n_distinct_observed_names=("team_name", _n_distinct_names),
        )
        .reset_index()
    )


def _format_timestamp(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return ""
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def _with_rank(ranked: pd.DataFrame) -> pd.DataFrame:
    out = ranked.copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def _display_frame(ranked: pd.DataFrame, *, top: int) -> pd.DataFrame:
    frame = _with_rank(ranked).head(top)
    display = frame[[c for c in _LEADERBOARD_COLUMNS if c in frame.columns]].copy()
    for column in ("elo", "elo_before_last_match", "elo_after_last_match"):
        if column in display.columns:
            display[column] = display[column].map(lambda x: f"{float(x):.1f}")
    if "last_match_timestamp" in display.columns:
        display["last_match_timestamp"] = display["last_match_timestamp"].map(
            _format_timestamp
        )
    if "team_name" in display.columns:
        display["team_name"] = display["team_name"].fillna("")
    return display


def _print_section(title: str, frame: pd.DataFrame) -> None:
    print(f"=== {title} ===")
    if frame.empty:
        print("(none)")
    else:
        print(frame.to_string(index=False))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.top < 1:
        print("--top must be >= 1", file=sys.stderr)
        return 1
    if args.active_days < 0:
        print("--active-days must be >= 0", file=sys.stderr)
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

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 40)

    with connect(config) as store:
        available = set(store.relation(MATCHES_VIEW).columns)
        selected = [c for c in _MATCH_COLUMNS if c in available]
        matches = store.sql(
            f"SELECT {', '.join(selected)} FROM {MATCHES_VIEW}"
        ).df()

    if matches.empty:
        print("Canonical matches dataset is empty.", file=sys.stderr)
        return 1

    dataset_max = matches["start_time"].max()
    n_matches = len(matches)
    n_tied_start_times = int(
        matches.groupby("start_time").size().gt(1).sum()
    )

    state = compute_team_elo_state(matches, config=DEFAULT_ELO_CONFIG)
    names = latest_observed_team_names(matches)
    if names.empty:
        state["team_name"] = pd.NA
        state["n_distinct_observed_names"] = pd.NA
        names_note = (
            "team names are not present in the loaded match columns; "
            "leaderboard uses team_id only"
        )
    else:
        state = state.merge(names, on="team_id", how="left")
        names_note = (
            "team_name is the most recently observed "
            "`radiant_team_name_observed`/`dire_team_name_observed` "
            "for that team_id (not a stable registry)"
        )

    ranked_all = rank_team_elo_state(state)
    active = filter_active_team_elo(
        state,
        dataset_max_timestamp=dataset_max,
        active_days=args.active_days,
    )
    ranked_active = rank_team_elo_state(active)

    elo = ranked_all["elo"]
    median_elo = float(elo.median())
    initial = DEFAULT_ELO_CONFIG.initial_rating
    low_sample_extreme = ranked_all[
        (ranked_all["n_matches"] < 5)
        & ((ranked_all["elo"] - initial).abs() > 100.0)
    ]
    renamed_teams = ranked_all[
        ranked_all["n_distinct_observed_names"].fillna(0) > 1
    ] if "n_distinct_observed_names" in ranked_all.columns else ranked_all.iloc[0:0]
    tied_last_group = ranked_all[ranked_all["last_group_n_matches"] > 1]
    duplicate_ids = ranked_all["team_id"].duplicated().any()

    print("=== Team Elo rankings (canonical dataset) ===")
    print(f"matches file: {config.matches_path}")
    print(f"rated maps: {n_matches}")
    print(f"dataset max start_time: {_format_timestamp(dataset_max)}")
    print(
        f"Elo config: initial_rating={DEFAULT_ELO_CONFIG.initial_rating}, "
        f"k_factor={DEFAULT_ELO_CONFIG.k_factor}"
    )
    print(
        "expected score: 1 / (1 + 10 ** ((opponent - rating) / 400)); "
        "update: k * (actual - expected); actual is 1.0/0.0 per map"
    )
    print(
        "no league/tier/recency/inactivity adjustments; teams keyed by "
        "canonical team_id; unseen teams start at initial_rating"
    )
    print(f"same-start_time groups (mutually blind): {n_tied_start_times}")
    print(f"name mapping: {names_note}")
    print()

    print("=== Summary ===")
    print(f"rated teams: {len(ranked_all)}")
    print(
        f"active teams (last match within {args.active_days}d of dataset max): "
        f"{len(ranked_active)}"
    )
    cutoff = active_team_elo_cutoff(dataset_max, active_days=args.active_days)
    print(f"active cutoff (inclusive): {_format_timestamp(cutoff)}")
    print(f"highest Elo: {float(elo.max()):.1f}")
    print(f"lowest Elo: {float(elo.min()):.1f}")
    print(f"median Elo: {median_elo:.1f}")
    print(f"mean Elo: {float(elo.mean()):.1f}")
    print(
        f"centered around initial {initial:.1f}? "
        f"median offset {median_elo - initial:+.1f}, "
        f"mean offset {float(elo.mean()) - initial:+.1f}"
    )
    print(f"duplicate team_id rows: {duplicate_ids}")
    print(
        f"teams with last_group_n_matches > 1 (tied last timestamp): "
        f"{len(tied_last_group)}"
    )
    print(
        f"teams with <5 maps and |elo-initial|>100: {len(low_sample_extreme)}"
    )
    print(f"teams with more than one observed name: {len(renamed_teams)}")
    print()

    _print_section(
        f"Top {args.top} active teams (last {args.active_days}d of dataset max)",
        _display_frame(ranked_active, top=args.top),
    )
    all_time_n = min(10, args.top)
    _print_section(
        f"Top {all_time_n} all-time / latest-state",
        _display_frame(ranked_all, top=all_time_n),
    )

    print("=== Top 5 active trajectories ===")
    top5 = team_elo_trajectories(ranked_active, n=5)
    if top5.empty:
        print("(none)")
    else:
        display = top5.copy()
        for column in ("starting_elo", "current_elo", "peak_elo", "lowest_elo"):
            if column in display.columns:
                display[column] = display[column].map(lambda x: f"{float(x):.1f}")
        if "peak_after_match_timestamp" in display.columns:
            display["peak_after_match"] = display["peak_after_match_timestamp"].map(
                _format_timestamp
            )
            display = display.drop(columns=["peak_after_match_timestamp"])
        if "peak_after_match_id" in display.columns:
            display["peak_after_match_id"] = display["peak_after_match_id"].map(
                lambda x: "" if pd.isna(x) else int(x)
            )
        if "team_name" in display.columns:
            display["team_name"] = display["team_name"].fillna("")
        print(display.to_string(index=False))
    print()

    if not low_sample_extreme.empty:
        print("=== Low-sample extreme ratings (inspection only) ===")
        print(
            _display_frame(rank_team_elo_state(low_sample_extreme), top=15).to_string(
                index=False
            )
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
