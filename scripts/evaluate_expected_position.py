"""Audit naive roster uniqueness and compare expected-position baselines.

Observed current-match STRATZ position is used only as the evaluation
target after assignments are generated.

Usage:
    uv run python scripts/evaluate_expected_position.py
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from dota_predictor.features.config import load_feature_store_config
from dota_predictor.features.duckdb_layer import connect
from dota_predictor.features.expected_position import (
    EXPECTED_POSITION_METHODS,
    NAIVE_PREFERENCE_SOURCES,
    assign_expected_positions,
    audit_naive_roster_uniqueness,
    evaluate_expected_position,
)
from dota_predictor.features.player_position import (
    EXPLICIT_POSITION_LABELS,
    build_player_position_state,
)
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pct(part: float, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{100.0 * part / whole:.2f}%"


def _print_uniqueness(audit: pd.DataFrame, state: pd.DataFrame) -> None:
    n_sides = len(audit)
    print("NAIVE ROSTER UNIQUENESS (independent preferences, no joint solver)")
    print(f"  sides: {n_sides}")
    for source in NAIVE_PREFERENCE_SOURCES:
        counts = audit[source].value_counts()
        unique = int(counts.get("unique_1_to_5", 0))
        print(f"  {source}: unique {{1..5}} {unique}/{n_sides} ({_pct(unique, n_sides)})")
        for status in ("duplicate", "missing", "missing_and_duplicate", "other"):
            n = int(counts.get(status, 0))
            if n:
                print(f"    {status}: {n} ({_pct(n, n_sides)})")
        dup_col = f"{source}_duplicate_positions"
        if dup_col in audit.columns:
            collided = audit[audit[source].isin(("duplicate", "missing_and_duplicate"))]
            pos_counts: dict[str, int] = {}
            for values in collided[dup_col]:
                for label in values:
                    pos_counts[label] = pos_counts.get(label, 0) + 1
            if pos_counts:
                ordered = ", ".join(
                    f"{label}={pos_counts[label]}"
                    for label in EXPLICIT_POSITION_LABELS
                    if label in pos_counts
                )
                print(f"    colliding positions (sides): {ordered}")

    print("  previous_explicit unique {1..5} by game_version_id:")
    by_version = (
        audit.groupby("game_version_id", sort=False)
        .agg(
            n_sides=("previous_explicit_position", "size"),
            unique=(
                "previous_explicit_position",
                lambda s: int((s == "unique_1_to_5").sum()),
            ),
        )
        .reset_index()
        .sort_values("n_sides", ascending=False)
    )
    for row in by_version.itertuples(index=False):
        print(
            f"    version {row.game_version_id}: {row.unique}/{row.n_sides} "
            f"({_pct(row.unique, int(row.n_sides))})"
        )

    side_missing = state.groupby(["match_id", "side"], sort=False)[
        "prior_explicit_position_games"
    ].apply(lambda s: int((s.fillna(0) == 0).sum()))
    print("COLD-START / WEAK HISTORY SIDES")
    print(f"  sides with 0 debut players: {int((side_missing == 0).sum())}")
    for n in range(1, 6):
        count = int((side_missing == n).sum())
        print(
            f"  sides with {n} player(s) lacking explicit history: {count} "
            f"({_pct(count, n_sides)})"
        )
    player_debuts = int((state["prior_explicit_position_games"].fillna(0) == 0).sum())
    print(
        f"  player-match rows with 0 prior explicit positions: {player_debuts} "
        f"({_pct(player_debuts, len(state))})"
    )


def _history_bucket(n: object) -> str:
    value = 0 if pd.isna(n) else int(n)
    if value == 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 10:
        return "3-10"
    if value <= 50:
        return "11-50"
    return "51+"


def _print_eval(name: str, assigned: pd.DataFrame, state: pd.DataFrame) -> dict:
    report = evaluate_expected_position(assigned)
    players = assigned[assigned["observed_position"].isin(EXPLICIT_POSITION_LABELS)].copy()
    players["correct"] = players["expected_position"] == players["observed_position"]
    players["history_bucket"] = players["prior_explicit_position_games"].map(_history_bucket)

    print(f"METHOD {name}")
    print(
        f"  player accuracy: {report['player_accuracy']:.4f} "
        f"({report['n_eligible_players']} eligible rows)"
    )
    print(
        f"  side exact 5/5: {report['side_exact_accuracy']:.4f} "
        f"({report['n_eligible_sides']} eligible sides)"
    )
    print(f"  side correct-count: {report['side_correct_counts']}")
    print("  confusion (rows=observed, cols=expected):")
    print(report["confusion"].to_string())

    print("  accuracy by observed position:")
    for label in EXPLICIT_POSITION_LABELS:
        subset = players[players["observed_position"] == label]
        acc = float(subset["correct"].mean()) if len(subset) else float("nan")
        print(f"    {label}: {acc:.4f} n={len(subset)}")
    print("  accuracy by expected position:")
    for label in EXPLICIT_POSITION_LABELS:
        subset = players[players["expected_position"] == label]
        acc = float(subset["correct"].mean()) if len(subset) else float("nan")
        print(f"    {label}: {acc:.4f} n={len(subset)}")
    print("  accuracy by prior explicit history:")
    for bucket in ("0", "1-2", "3-10", "11-50", "51+"):
        subset = players[players["history_bucket"] == bucket]
        acc = float(subset["correct"].mean()) if len(subset) else float("nan")
        print(f"    {bucket}: {acc:.4f} n={len(subset)}")

    stability = players["recent_position_stability"]
    print("  accuracy by recent_position_stability:")
    bins = [
        ("null", stability.isna()),
        ("<0.8", stability.notna() & (stability < 0.8)),
        ("0.8-0.99", stability.notna() & (stability >= 0.8) & (stability < 1.0)),
        ("1.0", stability.notna() & (stability >= 1.0)),
    ]
    for label, mask in bins:
        subset = players[mask]
        acc = float(subset["correct"].mean()) if len(subset) else float("nan")
        print(f"    {label}: {acc:.4f} n={len(subset)}")

    print(
        f"  role-switch rows (observed != previous explicit): "
        f"{report['switch_player_accuracy']:.4f} n={report['n_switch_eligible_players']}"
    )

    lagged = state.sort_values(["player_id", "start_time"], kind="mergesort").copy()
    lagged["prev_team_id"] = lagged.groupby("player_id")["team_id"].shift(1)
    transfer_keys = set(
        zip(
            lagged.loc[
                lagged["prev_team_id"].notna()
                & (lagged["team_id"] != lagged["prev_team_id"]),
                "match_id",
            ],
            lagged.loc[
                lagged["prev_team_id"].notna()
                & (lagged["team_id"] != lagged["prev_team_id"]),
                "player_id",
            ],
        )
    )
    transfer_mask = [
        (row.match_id, row.player_id) in transfer_keys
        for row in players.itertuples(index=False)
    ]
    transfers = players[transfer_mask]
    acc_t = float(transfers["correct"].mean()) if len(transfers) else float("nan")
    print(f"  transfer rows (team_id changed vs prior match): {acc_t:.4f} n={len(transfers)}")

    print(
        f"  assigned_score mean (eligible): "
        f"{float(players['assigned_position_score'].mean()):.3f}"
    )
    print(
        f"  player_score_margin mean: "
        f"{float(players['player_score_margin'].mean()):.3f}"
    )
    print(
        f"  roster_assignment_margin mean: "
        f"{float(players['roster_assignment_margin'].mean()):.3f}"
    )
    if "evidence_tier" in players.columns:
        print("  evidence_tier counts / accuracy:")
        for tier, subset in players.groupby("evidence_tier", dropna=False):
            acc = float(subset["correct"].mean()) if len(subset) else float("nan")
            print(f"    {tier}: {acc:.4f} n={len(subset)}")
    return report


def main() -> int:
    load_project_env(_project_root())
    config = load_feature_store_config(root=_project_root())
    t0 = perf_counter()
    with connect(config) as store:
        state = build_player_position_state(store).to_frame()
    state_s = perf_counter() - t0
    print(f"player_position_state: {len(state)} rows in {state_s:.2f}s")

    t1 = perf_counter()
    uniqueness = audit_naive_roster_uniqueness(state)
    print(f"uniqueness audit in {perf_counter() - t1:.2f}s")
    _print_uniqueness(uniqueness, state)

    reports: dict[str, dict] = {}
    assigned_by_method: dict[str, pd.DataFrame] = {}
    for method in EXPECTED_POSITION_METHODS:
        t2 = perf_counter()
        assigned = assign_expected_positions(state, method=method)
        elapsed = perf_counter() - t2
        print(f"\nassignment {method} in {elapsed:.2f}s")
        reports[method] = _print_eval(method, assigned, state)
        assigned_by_method[method] = assigned

    print("\nMETHOD COMPARISON (eligible players / exact sides)")
    for method in EXPECTED_POSITION_METHODS:
        report = reports[method]
        print(
            f"  {method}: player={report['player_accuracy']:.4f} "
            f"side_exact={report['side_exact_accuracy']:.4f} "
            f"switch={report['switch_player_accuracy']:.4f}"
        )

    pairs = (
        ("previous", "hierarchical"),
        ("previous", "recent_10"),
        ("recent_10", "hierarchical"),
        ("career", "hierarchical"),
        ("same_version", "career"),
    )
    print("\nMETHOD DISAGREEMENTS")
    for left, right in pairs:
        merged = assigned_by_method[left].merge(
            assigned_by_method[right],
            on=["match_id", "player_id"],
            suffixes=(f"_{left}", f"_{right}"),
        )
        disagree = merged[
            merged[f"expected_position_{left}"] != merged[f"expected_position_{right}"]
        ]
        eligible = merged[
            merged[f"observed_position_{left}"].isin(EXPLICIT_POSITION_LABELS)
        ]
        disagree_eligible = eligible[
            eligible[f"expected_position_{left}"] != eligible[f"expected_position_{right}"]
        ]
        print(
            f"  {left} vs {right}: {len(disagree)} / {len(merged)} rows "
            f"({_pct(len(disagree), len(merged))}); "
            f"eligible {len(disagree_eligible)} / {len(eligible)}"
        )
    print(f"total runtime including state: {perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
