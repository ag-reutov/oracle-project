"""Compare unconditioned Player × Hero with expected-position conditioning.

Usage:
    uv run python scripts/compare_player_hero_position.py
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from dota_predictor.features.config import (
    load_feature_store_config,
    load_reference_store_config,
)
from dota_predictor.features.duckdb_layer import connect, register_reference_views
from dota_predictor.features.player_hero_position import (
    build_player_hero_position,
    summarize_player_hero_position,
)
from dota_predictor.utils.env import load_project_env


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pct(part: float, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{100.0 * part / whole:.2f}%"


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):.4f}"


def _print_row(row: pd.Series, *, patch_names: dict[int, str]) -> None:
    key = row["key"]
    if row["scope"] == "game_version_id":
        name = patch_names.get(int(key), "")
        label = f"version {key}" + (f" ({name})" if name else "")
    else:
        label = str(key)
    n = int(row["n_rows"])
    print(
        f"  {label}: n={n} "
        f"uncond_cov={_fmt(row['unconditioned_coverage'])} "
        f"cond_cov={_fmt(row['conditioned_coverage'])} "
        f"hero_not_at_expected={_fmt(row['played_hero_not_at_expected_position'])} "
        f"mean_games {_fmt(row['mean_prior_games_on_hero'])}→"
        f"{_fmt(row['mean_prior_games_at_expected_position'])} "
        f"|ΔWR|={_fmt(row['mean_abs_win_rate_delta'])} "
        f"corr={_fmt(row['win_rate_correlation'])} "
        f"both_wr={int(row['n_both_win_rates'])}"
    )


def main() -> int:
    load_project_env(_project_root())
    config = load_feature_store_config(root=_project_root())
    reference = load_reference_store_config(root=_project_root())
    t0 = perf_counter()
    with connect(config) as store:
        try:
            register_reference_views(store, reference)
        except FileNotFoundError:
            pass
        frame = build_player_hero_position(store).to_frame()
        patch_names: dict[int, str] = {}
        tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
        if "game_versions" in tables:
            versions = store.sql(
                "SELECT game_version_id, name FROM game_versions"
            ).df()
            patch_names = {
                int(row.game_version_id): str(row.name)
                for row in versions.itertuples(index=False)
            }
    elapsed = perf_counter() - t0
    print(f"player_hero_position: {len(frame)} rows in {elapsed:.2f}s")
    print(f"method: {frame['expected_position_method'].iloc[0]}")

    summary = summarize_player_hero_position(frame)
    print("UNCONDITIONED vs EXPECTED-POSITION-CONDITIONED PLAYER × HERO")
    print("overall:")
    _print_row(summary[summary["scope"] == "overall"].iloc[0], patch_names=patch_names)
    print("by game_version_id:")
    for _, row in summary[summary["scope"] == "game_version_id"].iterrows():
        _print_row(row, patch_names=patch_names)
    print("by expected_position:")
    for _, row in summary[summary["scope"] == "expected_position"].iterrows():
        _print_row(row, patch_names=patch_names)

    eligible = frame[frame["observed_position"].notna()]
    agree = eligible["expected_position"] == eligible["observed_position"]
    print("COVERAGE WHEN EXPECTED MATCHES / MISSES OBSERVED (eval-only split)")
    for label, mask in (("expected==observed", agree), ("expected!=observed", ~agree)):
        subset = eligible[mask]
        uncond = float((subset["prior_games_on_hero"].fillna(0) > 0).mean())
        cond = float(
            (subset["prior_games_on_hero_at_expected_position"].fillna(0) > 0).mean()
        )
        print(
            f"  {label}: n={len(subset)} uncond_cov={uncond:.4f} cond_cov={cond:.4f} "
            f"({_pct(len(subset), len(eligible))})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
