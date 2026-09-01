"""Audit leakage-safe expanding hero meta state on the processed dataset.

Descriptive diagnosis only. Does not train a model or add win-model features.

Usage:
    uv run python scripts/audit_hero_meta_state.py
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from time import perf_counter

import pandas as pd

from dota_predictor.features.config import (
    load_feature_store_config,
    load_reference_store_config,
)
from dota_predictor.features.duckdb_layer import (
    GAME_VERSIONS_VIEW,
    connect,
    register_reference_views,
)
from dota_predictor.features.hero_state import (
    POSITION_NUMBERS,
    build_hero_state,
    summarize_hero_state,
)
from dota_predictor.utils.env import load_project_env

_MIN_VERSION_MATCHES = 20
_MIN_PICKS_FOR_WIN_RATE = 8
_MIN_EXPLICIT_FOR_POSITION = 8
_EXAMPLE_LIMIT = 8


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pct(part: float, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{100.0 * part / whole:.2f}%"


def _fmt_rate(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_share(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _describe(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "empty"
    return (
        f"n={len(clean)} mean={clean.mean():.3f} p25={clean.quantile(0.25):.3f} "
        f"median={clean.median():.3f} p75={clean.quantile(0.75):.3f} "
        f"p90={clean.quantile(0.90):.3f} max={clean.max():.3f}"
    )


def _hero_label(row: pd.Series) -> str:
    name = row.get("hero_name")
    if pd.isna(name) or not str(name).strip():
        return f"hero {int(row['hero_id'])}"
    return f"{name} ({int(row['hero_id'])})"


def _modal_position(row: pd.Series, *, prefix: str) -> tuple[int | None, float]:
    shares = []
    for position in POSITION_NUMBERS:
        value = row.get(f"{prefix}position_{position}_share")
        shares.append(
            (position, float(value)) if pd.notna(value) else (position, float("nan"))
        )
    finite = [(pos, share) for pos, share in shares if pd.notna(share)]
    if not finite:
        return None, float("nan")
    finite.sort(key=lambda item: (-item[1], item[0]))
    return finite[0]


def _load_patch_names(store) -> dict[int, str]:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    if GAME_VERSIONS_VIEW not in tables:
        return {}
    versions = store.sql(
        f"SELECT game_version_id, name FROM {GAME_VERSIONS_VIEW}"
    ).df()
    return {
        int(row.game_version_id): str(row.name)
        for row in versions.itertuples(index=False)
    }


def _end_of_version_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["hero_id", "start_time", "match_id"], kind="mergesort"
    )
    return ordered.groupby(["hero_id", "game_version_id"], as_index=False).tail(1)


def _version_order(frame: pd.DataFrame) -> list[int]:
    starts = (
        frame.groupby("game_version_id", as_index=False)["start_time"]
        .min()
        .sort_values("start_time", kind="mergesort")
    )
    return [int(value) for value in starts["game_version_id"].tolist()]


def _consecutive_shifts(
    snapshots: pd.DataFrame,
    *,
    version_order: list[int],
    patch_names: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_hero = {
        hero_id: subset.set_index("game_version_id")
        for hero_id, subset in snapshots.groupby("hero_id")
    }
    for hero_id, subset in by_hero.items():
        for previous, current in pairwise(version_order):
            if previous not in subset.index or current not in subset.index:
                continue
            before = subset.loc[previous]
            after = subset.loc[current]
            if isinstance(before, pd.DataFrame):
                before = before.iloc[-1]
            if isinstance(after, pd.DataFrame):
                after = after.iloc[-1]
            before_modal, before_share = _modal_position(
                before, prefix="hero_same_version_"
            )
            after_modal, after_share = _modal_position(
                after, prefix="hero_same_version_"
            )
            position_l1 = 0.0
            have_shares = True
            for position in POSITION_NUMBERS:
                left = before.get(f"hero_same_version_position_{position}_share")
                right = after.get(f"hero_same_version_position_{position}_share")
                if pd.isna(left) or pd.isna(right):
                    have_shares = False
                    break
                position_l1 += abs(float(right) - float(left))
            rows.append(
                {
                    "hero_id": int(hero_id),
                    "hero_name": before.get("hero_name"),
                    "from_version": previous,
                    "to_version": current,
                    "from_patch": patch_names.get(previous, str(previous)),
                    "to_patch": patch_names.get(current, str(current)),
                    "from_matches": int(before["hero_same_version_prior_matches"]),
                    "to_matches": int(after["hero_same_version_prior_matches"]),
                    "from_picks": int(before["hero_same_version_pick_count"]),
                    "to_picks": int(after["hero_same_version_pick_count"]),
                    "from_contest": before["hero_same_version_contest_rate"],
                    "to_contest": after["hero_same_version_contest_rate"],
                    "contest_delta": (
                        after["hero_same_version_contest_rate"]
                        - before["hero_same_version_contest_rate"]
                    ),
                    "from_win_rate": before["hero_same_version_win_rate"],
                    "to_win_rate": after["hero_same_version_win_rate"],
                    "win_rate_delta": (
                        after["hero_same_version_win_rate"]
                        - before["hero_same_version_win_rate"]
                    ),
                    "from_expanding_contest": before["hero_contest_rate"],
                    "to_expanding_contest": after["hero_contest_rate"],
                    "from_recent50_contest": before["hero_recent_50_contest_rate"],
                    "to_recent50_contest": after["hero_recent_50_contest_rate"],
                    "from_explicit": int(
                        before["hero_same_version_position_explicit_count"]
                    ),
                    "to_explicit": int(
                        after["hero_same_version_position_explicit_count"]
                    ),
                    "from_modal_position": before_modal,
                    "to_modal_position": after_modal,
                    "from_modal_share": before_share,
                    "to_modal_share": after_share,
                    "position_l1": position_l1 if have_shares else float("nan"),
                }
            )
    return pd.DataFrame.from_records(rows)


def _print_coverage(frame: pd.DataFrame) -> None:
    n = len(frame)
    n_matches = int(frame["match_id"].nunique())
    n_heroes = int(frame["hero_id"].nunique())
    picks = frame["hero_pick_count"].fillna(0)
    matches = frame["hero_prior_matches"].fillna(0)
    explicit = frame["hero_position_explicit_count"].fillna(0)
    print("COVERAGE / COLD START")
    print(f"  rows (match, hero): {n}")
    print(f"  matches: {n_matches}")
    print(f"  heroes: {n_heroes}")
    print(
        f"  rows with prior matches: {int((matches > 0).sum())} "
        f"({_pct(float((matches > 0).mean()), 1)})"
    )
    print(
        f"  rows with prior picks: {int((picks > 0).sum())} "
        f"({_pct(float((picks > 0).mean()), 1)})"
    )
    print(
        f"  cold-start (zero prior picks): {int((picks == 0).sum())} "
        f"({_pct(float((picks == 0).mean()), 1)})"
    )
    print(
        f"  rows with positional evidence: {int((explicit > 0).sum())} "
        f"({_pct(float((explicit > 0).mean()), 1)})"
    )
    print(
        f"  rows with days_since_last_pick: "
        f"{int(frame['hero_days_since_last_pick'].notna().sum())} "
        f"({_pct(float(frame['hero_days_since_last_pick'].notna().mean()), 1)})"
    )


def _print_distributions(frame: pd.DataFrame) -> None:
    print("PRIOR SAMPLE SIZES")
    print(f"  hero_prior_matches: {_describe(frame['hero_prior_matches'])}")
    print(f"  hero_pick_count:    {_describe(frame['hero_pick_count'])}")
    picked = frame[frame["hero_pick_count"] > 0]
    print(f"  hero_pick_count | picks>0: {_describe(picked['hero_pick_count'])}")
    print("PICK / BAN / CONTEST RATES (NULL when no prior matches)")
    print(f"  pick_rate:    {_describe(frame['hero_pick_rate'])}")
    print(f"  ban_rate:     {_describe(frame['hero_ban_rate'])}")
    print(f"  contest_rate: {_describe(frame['hero_contest_rate'])}")
    print("WIN RATE (NULL when no prior picks)")
    print(f"  hero_prior_win_rate: {_describe(frame['hero_prior_win_rate'])}")
    print("POSITION SHARES (NULL when no explicit historical positions)")
    for position in POSITION_NUMBERS:
        column = f"hero_position_{position}_share"
        print(f"  {column}: {_describe(frame[column])}")


def _print_by_patch(
    frame: pd.DataFrame, *, patch_names: dict[int, str]
) -> None:
    print("BY-PATCH SUMMARIES (row-level expanding state)")
    grouped = frame.groupby("game_version_id", sort=False)
    order = _version_order(frame)
    for version in order:
        if version not in grouped.groups:
            continue
        subset = grouped.get_group(version)
        name = patch_names.get(version, "")
        label = f"{version}" + (f" ({name})" if name else "")
        print(
            f"  {label}: n={len(subset)} "
            f"mean_picks={subset['hero_pick_count'].mean():.1f} "
            f"mean_contest={_fmt_rate(subset['hero_contest_rate'].mean())} "
            f"mean_wr={_fmt_rate(subset['hero_prior_win_rate'].mean())} "
            f"pos_evidence={_pct(float((subset['hero_position_explicit_count'] > 0).mean()), 1)}"
        )


def _print_shift_examples(
    shifts: pd.DataFrame,
    *,
    title: str,
    score: str,
    formatter,
    minimum_mask: pd.Series,
) -> None:
    print(title)
    eligible = shifts[minimum_mask & shifts[score].notna()].copy()
    if eligible.empty:
        print("  none")
        return
    eligible["abs_score"] = eligible[score].abs()
    top = eligible.sort_values("abs_score", kind="mergesort", ascending=False).head(
        _EXAMPLE_LIMIT
    )
    for row in top.itertuples(index=False):
        print(f"  {formatter(row)}")


def _contest_line(row: object) -> str:
    return (
        f"{_hero_label(pd.Series(row._asdict()))}: contest rate "
        f"{_fmt_rate(row.from_contest)} → {_fmt_rate(row.to_contest)} "
        f"({row.from_patch} → {row.to_patch}; "
        f"same-version matches {row.from_matches}→{row.to_matches})"
    )


def _win_line(row: object) -> str:
    return (
        f"{_hero_label(pd.Series(row._asdict()))}: win rate "
        f"{_fmt_rate(row.from_win_rate)} → {_fmt_rate(row.to_win_rate)} "
        f"({row.from_patch} → {row.to_patch}; "
        f"same-version picks {row.from_picks}→{row.to_picks})"
    )


def _position_line(row: object) -> str:
    from_pos = (
        f"position {int(row.from_modal_position)}"
        if pd.notna(row.from_modal_position)
        else "no modal position"
    )
    to_pos = (
        f"position {int(row.to_modal_position)}"
        if pd.notna(row.to_modal_position)
        else "no modal position"
    )
    return (
        f"{_hero_label(pd.Series(row._asdict()))}: {from_pos} share "
        f"{_fmt_share(row.from_modal_share)} → {to_pos} share "
        f"{_fmt_share(row.to_modal_share)} "
        f"({row.from_patch} → {row.to_patch}; "
        f"explicit {row.from_explicit}→{row.to_explicit})"
    )


def main() -> int:
    load_project_env(_project_root())
    config = load_feature_store_config(root=_project_root())
    reference = load_reference_store_config(root=_project_root())
    started = perf_counter()
    with connect(config) as store:
        try:
            register_reference_views(store, reference)
        except FileNotFoundError:
            pass
        frame = build_hero_state(store).to_frame()
        patch_names = _load_patch_names(store)
    elapsed = perf_counter() - started

    print("HERO META STATE (Slice 5)")
    print(f"  runtime: {elapsed:.1f}s")
    _print_coverage(frame)
    _print_distributions(frame)
    _print_by_patch(frame, patch_names=patch_names)

    snapshots = _end_of_version_snapshots(frame)
    versions = _version_order(frame)
    shifts = _consecutive_shifts(
        snapshots, version_order=versions, patch_names=patch_names
    )
    enough_matches = (shifts["from_matches"] >= _MIN_VERSION_MATCHES) & (
        shifts["to_matches"] >= _MIN_VERSION_MATCHES
    )
    enough_picks = (shifts["from_picks"] >= _MIN_PICKS_FOR_WIN_RATE) & (
        shifts["to_picks"] >= _MIN_PICKS_FOR_WIN_RATE
    )
    enough_positions = (shifts["from_explicit"] >= _MIN_EXPLICIT_FOR_POSITION) & (
        shifts["to_explicit"] >= _MIN_EXPLICIT_FOR_POSITION
    )
    role_changed = (
        shifts["from_modal_position"].notna()
        & shifts["to_modal_position"].notna()
        & (shifts["from_modal_position"] != shifts["to_modal_position"])
    )

    _print_shift_examples(
        shifts,
        title=(
            "LARGEST SAME-VERSION CONTEST-RATE SHIFTS "
            f"(end-of-patch, ≥{_MIN_VERSION_MATCHES} matches each side)"
        ),
        score="contest_delta",
        formatter=_contest_line,
        minimum_mask=enough_matches,
    )
    _print_shift_examples(
        shifts,
        title=(
            "LARGEST SAME-VERSION WIN-RATE SHIFTS "
            f"(end-of-patch, ≥{_MIN_PICKS_FOR_WIN_RATE} picks each side)"
        ),
        score="win_rate_delta",
        formatter=_win_line,
        minimum_mask=enough_matches & enough_picks,
    )
    _print_shift_examples(
        shifts,
        title=(
            "LARGEST POSITION-ROLE SHIFTS "
            f"(same-version modal position changed, ≥{_MIN_EXPLICIT_FOR_POSITION} "
            "explicit observations each side)"
        ),
        score="position_l1",
        formatter=_position_line,
        minimum_mask=enough_positions,
    )

    if not shifts.empty:
        contest_eligible = shifts[enough_matches & shifts["contest_delta"].notna()]
        print("PATCH-SHIFT SCALE")
        if contest_eligible.empty:
            print("  no eligible contest-rate shifts")
        else:
            abs_delta = contest_eligible["contest_delta"].abs()
            print(
                f"  |same-version contest delta| among eligible transitions: "
                f"{_describe(abs_delta)}"
            )
            print(
                f"  transitions with |contest delta| ≥ 0.10: "
                f"{int((abs_delta >= 0.10).sum())} / {len(contest_eligible)}"
            )
            print(
                f"  transitions with |contest delta| ≥ 0.20: "
                f"{int((abs_delta >= 0.20).sum())} / {len(contest_eligible)}"
            )
        win_eligible = shifts[
            enough_matches & enough_picks & shifts["win_rate_delta"].notna()
        ]
        if not win_eligible.empty:
            print(
                f"  |same-version win-rate delta| among eligible transitions: "
                f"{_describe(win_eligible['win_rate_delta'].abs())}"
            )
        role_eligible = shifts[enough_positions & role_changed]
        print(
            f"  modal-position changes among eligible position transitions: "
            f"{len(role_eligible)} / {int(enough_positions.sum())}"
        )

    summary = summarize_hero_state(frame)
    overall = summary[
        (summary["scope"] == "overall") & (summary["stat"] == "coverage")
    ].iloc[0]
    print("SUMMARY TABLE (overall coverage)")
    print(
        f"  prior_match_coverage={_fmt_rate(overall['prior_match_coverage'])} "
        f"prior_pick_coverage={_fmt_rate(overall['prior_pick_coverage'])} "
        f"cold_start={_fmt_rate(overall['cold_start_no_picks'])} "
        f"position_evidence={_fmt_rate(overall['position_evidence_coverage'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
