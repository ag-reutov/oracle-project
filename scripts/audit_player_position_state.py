"""Audit historical player × position state on the processed dataset.

Does not infer missing positions or train a model.

Usage:
    uv run python scripts/audit_player_position_state.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from dota_predictor.features.config import load_feature_store_config
from dota_predictor.features.duckdb_layer import connect
from dota_predictor.features.player_position import (
    EXPLICIT_POSITION_LABELS,
    build_player_position_state,
)
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{100.0 * part / whole:.2f}%"


def _describe(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return "empty"
    return (
        f"n={len(clean)} mean={clean.mean():.3f} p25={clean.quantile(0.25):.3f} "
        f"median={clean.median():.3f} p75={clean.quantile(0.75):.3f} "
        f"p90={clean.quantile(0.90):.3f}"
    )


def main() -> int:
    load_project_env(_project_root())
    config = load_feature_store_config(root=_project_root())
    with connect(config) as store:
        frame = build_player_position_state(store).to_frame()

    n_rows = len(frame)
    explicit_current = frame["position"].isin(EXPLICIT_POSITION_LABELS).sum()
    latest = (
        frame.sort_values(["player_id", "start_time"], kind="mergesort")
        .groupby("player_id", as_index=False)
        .tail(1)
    )

    print("PLAYER x POSITION STATE")
    print(f"  player-match rows: {n_rows}")
    print(
        f"  rows with explicit current observed position: {explicit_current} "
        f"({_pct(int(explicit_current), n_rows)})"
    )
    print(f"  unique players: {frame['player_id'].nunique()}")

    distinct = latest["historical_distinct_positions"].fillna(0).astype(int)
    print("PLAYERS BY DISTINCT HISTORICAL POSITIONS (at latest match)")
    for n in range(6):
        count = int((distinct == n).sum())
        print(f"  {n}: {count} ({_pct(count, len(latest))})")

    print("HISTORICAL MODAL-POSITION SHARE (rows with a unique mode)")
    print(f"  {_describe(frame['historical_modal_position_share'])}")
    print("RECENT POSITION STABILITY (last-10 modal share)")
    print(f"  {_describe(frame['recent_position_stability'])}")

    explicit_rows = frame[frame["position"].isin(EXPLICIT_POSITION_LABELS)]
    switched = explicit_rows[
        explicit_rows["previous_explicit_position"].notna()
        & (explicit_rows["position"] != explicit_rows["previous_explicit_position"])
    ]
    print("POSITION SWITCHES (current explicit != previous explicit)")
    print(
        f"  {len(switched)} / {len(explicit_rows)} explicit rows "
        f"({_pct(len(switched), len(explicit_rows))})"
    )

    modal_changes = (
        frame.dropna(subset=["historical_modal_position"])
        .groupby("player_id")["historical_modal_position"]
        .nunique()
    )
    changed = int((modal_changes > 1).sum())
    print(
        f"PLAYERS WHOSE HISTORICAL MODAL CHANGES OVER TIME: {changed} / "
        f"{frame['player_id'].nunique()} ({_pct(changed, frame['player_id'].nunique())})"
    )

    multi = latest[latest["historical_distinct_positions"] >= 3]
    print(f"PLAYERS WITH 3+ DISTINCT HISTORICAL POSITIONS AT LATEST MATCH: {len(multi)}")

    print("OBSERVED CURRENT POSITION BY GAME_VERSION_ID")
    by_version = (
        frame.groupby(["game_version_id", "position"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    print(by_version.to_string())

    examples = [
        (898754153, "stable POSITION_1 candidate"),
        (106573901, "stable POSITION_2 candidate"),
        (10366616, "stable POSITION_5 candidate"),
        (87063175, "mostly POSITION_4 / NULL row candidate"),
        (352545711, "transfer / POSITION_2 continuity candidate"),
    ]
    print("EXAMPLE PLAYERS (latest row)")
    for player_id, label in examples:
        subset = frame[frame["player_id"] == player_id].sort_values("start_time")
        if subset.empty:
            print(f"  {player_id} {label}: not in dataset")
            continue
        last = subset.iloc[-1]
        hist = Counter(
            pos
            for pos in subset["position"]
            if pos in EXPLICIT_POSITION_LABELS
        )
        print(
            f"  {player_id} {label}: matches={len(subset)} last_team={last['team_id']} "
            f"last_pos={last['position']} modal={last['historical_modal_position']} "
            f"distinct={last['historical_distinct_positions']} "
            f"career_explicit={dict(hist)} "
            f"last10_modal={last['recent_10_modal_position']} "
            f"stability={last['recent_position_stability']}"
        )

    null_rows = frame[frame["position"].isna()]
    print(f"NULL CURRENT POSITION ROWS: {len(null_rows)}")
    if not null_rows.empty:
        sample = null_rows.sort_values("start_time").iloc[0]
        print(
            f"  example match_id={sample['match_id']} player_id={sample['player_id']} "
            f"prior_games={sample['prior_games']} "
            f"prior_explicit={sample['prior_explicit_position_games']} "
            f"prior_pos1={sample['prior_games_position_1']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
