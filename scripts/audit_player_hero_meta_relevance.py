"""Audit meta-relevant Player × Hero state on the processed dataset.

Descriptive diagnosis only. Does not train a model, fit weights, or add
win-model features.

Answers: does conditioning Player × Hero history on recency, game version,
and current hero-role meta distinguish cases that career Player × Hero
treats as equivalent?

Usage:
    uv run python scripts/audit_player_hero_meta_relevance.py
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
from dota_predictor.features.hero_state import POSITION_NUMBERS
from dota_predictor.features.player_hero_meta import (
    PREFERRED_HERO_META_WINDOW,
    build_player_hero_meta,
    summarize_player_hero_meta,
)
from dota_predictor.utils.env import load_project_env

_EXAMPLE_LIMIT = 8
_MIN_CAREER_FOR_MISMATCH = 10
_MIN_EXPLICIT_FOR_POSITION = 8
_SUPPORT_POSITIONS = frozenset({"POSITION_4", "POSITION_5"})
_CORE_POSITIONS = frozenset({"POSITION_1", "POSITION_2", "POSITION_3"})
_CAREER_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1-4", 1, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    ("20+", 20, None),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pct(part: float, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{100.0 * part / whole:.2f}%"


def _fmt(value: object, *, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


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


def _player_label(row: pd.Series) -> str:
    return f"player {int(row['player_id'])}"


def _patch_label(row: pd.Series, patch_names: dict[int, str]) -> str:
    version = int(row["game_version_id"])
    name = patch_names.get(version)
    return f"{name}" if name else str(version)


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


def _corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 2:
        return float("nan")
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return float(value) if pd.notna(value) else float("nan")


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


def _print_coverage(frame: pd.DataFrame) -> None:
    n = len(frame)
    career = frame["prior_games_on_hero"].fillna(0)
    r20 = frame["player_hero_recent_20_matches"].fillna(0)
    r50 = frame["player_hero_recent_50_matches"].fillna(0)
    r100 = frame["player_hero_recent_100_matches"].fillna(0)
    same = frame["player_hero_same_version_matches"].fillna(0)
    player_pos = frame["player_hero_position_explicit_games"].fillna(0)
    hero_pos = frame[
        f"hero_recent_{PREFERRED_HERO_META_WINDOW}_position_explicit_count"
    ].fillna(0)
    compat = frame["player_hero_recent_role_compatibility"]
    print("COVERAGE")
    print(f"  rows (match, player): {n}")
    print(f"  matches: {int(frame['match_id'].nunique())}")
    print(f"  players: {int(frame['player_id'].nunique())}")
    print(
        f"  career Player × Hero history (>0 games): "
        f"{int((career > 0).sum())} ({_pct(float((career > 0).mean()), 1)})"
    )
    print(
        f"  recent-20 history: {int((r20 > 0).sum())} "
        f"({_pct(float((r20 > 0).mean()), 1)})"
    )
    print(
        f"  recent-50 history: {int((r50 > 0).sum())} "
        f"({_pct(float((r50 > 0).mean()), 1)})"
    )
    print(
        f"  recent-100 history: {int((r100 > 0).sum())} "
        f"({_pct(float((r100 > 0).mean()), 1)})"
    )
    print(
        f"  same-version history: {int((same > 0).sum())} "
        f"({_pct(float((same > 0).mean()), 1)})"
    )
    print(
        f"  player positional hero history: {int((player_pos > 0).sum())} "
        f"({_pct(float((player_pos > 0).mean()), 1)})"
    )
    print(
        f"  usable current hero-meta positional state "
        f"(recent-{PREFERRED_HERO_META_WINDOW}): "
        f"{int((hero_pos > 0).sum())} ({_pct(float((hero_pos > 0).mean()), 1)})"
    )
    print(
        f"  non-NULL recent role compatibility: "
        f"{int(compat.notna().sum())} ({_pct(float(compat.notna().mean()), 1)})"
    )


def _print_career_vs_recent(frame: pd.DataFrame) -> None:
    print("CAREER VS RECENT (rows with substantial career experience)")
    career = frame["prior_games_on_hero"].fillna(0)
    for label, lo, hi in _CAREER_BUCKETS:
        if hi is None:
            subset = frame[career >= lo]
        else:
            subset = frame[(career >= lo) & (career <= hi)]
        n = len(subset)
        if n == 0:
            print(f"  career {label}: n=0")
            continue
        r20 = subset["player_hero_recent_20_matches"].fillna(0)
        r50 = subset["player_hero_recent_50_matches"].fillna(0)
        r100 = subset["player_hero_recent_100_matches"].fillna(0)
        print(
            f"  career {label}: n={n} "
            f"recent20>0={_pct(float((r20 > 0).mean()), 1)} "
            f"recent20=0={_pct(float((r20 == 0).mean()), 1)} "
            f"mean_r20={r20.mean():.2f} "
            f"recent50>0={_pct(float((r50 > 0).mean()), 1)} "
            f"recent100>0={_pct(float((r100 > 0).mean()), 1)}"
        )


def _print_career_vs_same_version(
    frame: pd.DataFrame, *, patch_names: dict[int, str]
) -> None:
    print("CAREER VS SAME-VERSION")
    career = frame["prior_games_on_hero"].fillna(0)
    same = frame["player_hero_same_version_matches"].fillna(0)
    has_career = frame[career > 0]
    if has_career.empty:
        print("  no career history")
        return
    ratio = (
        has_career["player_hero_same_version_matches"].fillna(0)
        / has_career["prior_games_on_hero"]
    )
    print(
        f"  among career>0: same-version=0: "
        f"{int((same[career > 0] == 0).sum())} / {len(has_career)} "
        f"({_pct(float((same[career > 0] == 0).mean()), 1)})"
    )
    print(f"  same-version / career ratio: {_describe(ratio)}")
    examples = (
        has_career.assign(
            career_games=has_career["prior_games_on_hero"],
            same_version_games=has_career["player_hero_same_version_matches"].fillna(0),
        )
        .sort_values(
            ["same_version_games", "career_games"],
            ascending=[True, False],
            kind="mergesort",
        )
        .head(_EXAMPLE_LIMIT)
    )
    print("  examples (high career, low same-version):")
    for row in examples.itertuples(index=False):
        series = pd.Series(row._asdict())
        print(
            f"    {_player_label(series)} {_hero_label(series)} "
            f"{_patch_label(series, patch_names)} "
            f"match {int(series['match_id'])}: "
            f"career={int(series['prior_games_on_hero'])} "
            f"same-version={int(series['player_hero_same_version_matches'])}"
        )


def _print_role_compatibility(
    frame: pd.DataFrame, *, patch_names: dict[int, str]
) -> None:
    compat = frame["player_hero_recent_role_compatibility"]
    print("ROLE-COMPATIBILITY DISTRIBUTION")
    print(f"  recent-{PREFERRED_HERO_META_WINDOW}: {_describe(compat)}")
    print(
        f"  same-version: "
        f"{_describe(frame['player_hero_same_version_role_compatibility'])}"
    )
    defined = frame[compat.notna()]
    if defined.empty:
        print("  no defined compatibility rows")
        return
    high_career = defined[
        defined["prior_games_on_hero"].fillna(0) >= _MIN_CAREER_FOR_MISMATCH
    ]
    print(
        f"  high-career (≥{_MIN_CAREER_FOR_MISMATCH}) rows with compatibility: "
        f"{len(high_career)}"
    )
    if high_career.empty:
        return
    cutoff = high_career["player_hero_recent_role_compatibility"].quantile(0.10)
    bottom = high_career[
        high_career["player_hero_recent_role_compatibility"] <= cutoff
    ].sort_values(
        "player_hero_recent_role_compatibility", kind="mergesort"
    )
    print(
        f"  diagnostic bottom-decile cutoff among high-career rows: "
        f"{_fmt(cutoff)} (n={len(bottom)}; not a feature threshold)"
    )
    print("  high-career, low recent-role-compatibility examples:")
    for row in bottom.head(_EXAMPLE_LIMIT).itertuples(index=False):
        series = pd.Series(row._asdict())
        expected = series["expected_position"]
        print(
            f"    {_player_label(series)} {_hero_label(series)} "
            f"{_patch_label(series, patch_names)} "
            f"match {int(series['match_id'])} expected={expected}: "
            f"career={int(series['prior_games_on_hero'])} "
            f"recent20={int(series['player_hero_recent_20_matches'])} "
            f"compat={_fmt(series['player_hero_recent_role_compatibility'])} "
            f"player@expected={_fmt(series['player_hero_share_at_expected_position'])} "
            f"hero@expected={_fmt(series['hero_position_share_at_expected_position'])}"
        )


def _hero_shift_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive hero-level grouping from Slice 5 columns already on the row.

    A hero is 'role_shifted' when consecutive same-version snapshots
    change modal position (enough explicit observations). 'contest_shifted'
    if |contest-rate delta| ≥ 0.20. 'stable' if eligible and neither.
    Not a classifier.
    """
    snapshots = (
        frame.sort_values(["hero_id", "start_time", "match_id"], kind="mergesort")
        .groupby(["hero_id", "game_version_id"], as_index=False)
        .tail(1)
    )
    starts = (
        frame.groupby("game_version_id", as_index=False)["start_time"]
        .min()
        .sort_values("start_time", kind="mergesort")
    )
    version_order = [int(value) for value in starts["game_version_id"].tolist()]
    by_hero = {
        hero_id: subset.set_index("game_version_id")
        for hero_id, subset in snapshots.groupby("hero_id")
    }
    rows: list[dict[str, object]] = []
    for hero_id, subset in by_hero.items():
        role_ever = False
        contest_ever = False
        eligible = False
        for previous, current in pairwise(version_order):
            if previous not in subset.index or current not in subset.index:
                continue
            before = subset.loc[previous]
            after = subset.loc[current]
            if isinstance(before, pd.DataFrame):
                before = before.iloc[-1]
            if isinstance(after, pd.DataFrame):
                after = after.iloc[-1]
            from_explicit = int(
                before.get("hero_same_version_position_explicit_count", 0) or 0
            )
            to_explicit = int(
                after.get("hero_same_version_position_explicit_count", 0) or 0
            )
            from_contest = before.get("hero_same_version_contest_rate")
            to_contest = after.get("hero_same_version_contest_rate")
            enough_contest = pd.notna(from_contest) and pd.notna(to_contest)
            enough_pos = (
                from_explicit >= _MIN_EXPLICIT_FOR_POSITION
                and to_explicit >= _MIN_EXPLICIT_FOR_POSITION
            )
            if enough_contest or enough_pos:
                eligible = True
            contest_delta = (
                abs(float(to_contest) - float(from_contest))
                if enough_contest
                else 0.0
            )
            before_modal, _ = _modal_position(before, prefix="hero_same_version_")
            after_modal, _ = _modal_position(after, prefix="hero_same_version_")
            if (
                enough_pos
                and before_modal is not None
                and after_modal is not None
                and before_modal != after_modal
            ):
                role_ever = True
            if enough_contest and contest_delta >= 0.20:
                contest_ever = True
        rows.append(
            {
                "hero_id": int(hero_id),
                "shift_group": (
                    "role_shifted"
                    if role_ever
                    else (
                        "contest_shifted"
                        if contest_ever
                        else ("stable" if eligible else "unclassified")
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _print_stable_vs_shifted(
    frame: pd.DataFrame, *, patch_names: dict[int, str]
) -> None:
    print("STABLE VS SHIFTED HEROES (descriptive Slice 5 grouping)")
    groups = _hero_shift_groups(frame)
    merged = frame.merge(groups, on="hero_id", how="left")
    for label in ("stable", "role_shifted", "contest_shifted", "unclassified"):
        subset = merged[merged["shift_group"] == label]
        n = len(subset)
        if n == 0:
            print(f"  {label}: n=0")
            continue
        career = subset["prior_games_on_hero"].fillna(0)
        high = subset[career >= _MIN_CAREER_FOR_MISMATCH]
        print(
            f"  {label}: n_rows={n} heroes={int(subset['hero_id'].nunique())} "
            f"mean_compat={_fmt(subset['player_hero_recent_role_compatibility'].mean())} "
            f"high-career n={len(high)} "
            f"high-career mean_compat={_fmt(high['player_hero_recent_role_compatibility'].mean())} "
            f"mean_recent20={subset['player_hero_recent_20_matches'].fillna(0).mean():.2f}"
        )
    role_ids = set(groups.loc[groups["shift_group"] == "role_shifted", "hero_id"])
    contest_ids = set(groups.loc[groups["shift_group"] == "contest_shifted", "hero_id"])
    if role_ids:
        names = (
            merged.loc[merged["hero_id"].isin(role_ids), ["hero_id", "hero_name"]]
            .drop_duplicates("hero_id")
            .sort_values("hero_id")
        )
        labeled = ", ".join(
            _hero_label(pd.Series(row._asdict()))
            for row in names.itertuples(index=False)
        )
        print(f"  role-shifted heroes: {labeled}")
        interesting = merged[
            (merged["hero_id"].isin(role_ids))
            & (merged["prior_games_on_hero"].fillna(0) >= _MIN_CAREER_FOR_MISMATCH)
            & merged["player_hero_recent_role_compatibility"].notna()
        ].sort_values("player_hero_recent_role_compatibility", kind="mergesort")
        print("  role-shifted, high-career, lowest-compatibility examples:")
        for row in interesting.head(_EXAMPLE_LIMIT).itertuples(index=False):
            series = pd.Series(row._asdict())
            print(
                f"    {_player_label(series)} {_hero_label(series)} "
                f"{_patch_label(series, patch_names)} "
                f"match {int(series['match_id'])}: "
                f"career={int(series['prior_games_on_hero'])} "
                f"compat={_fmt(series['player_hero_recent_role_compatibility'])} "
                f"player@expected={_fmt(series['player_hero_share_at_expected_position'])} "
                f"hero@expected={_fmt(series['hero_position_share_at_expected_position'])}"
            )
    if contest_ids:
        interesting = merged[
            (merged["hero_id"].isin(contest_ids))
            & (merged["prior_games_on_hero"].fillna(0) >= _MIN_CAREER_FOR_MISMATCH)
            & merged["player_hero_recent_role_compatibility"].notna()
        ].sort_values("player_hero_recent_role_compatibility", kind="mergesort")
        print("  contest-shifted, high-career, lowest-compatibility examples:")
        for row in interesting.head(_EXAMPLE_LIMIT).itertuples(index=False):
            series = pd.Series(row._asdict())
            print(
                f"    {_player_label(series)} {_hero_label(series)} "
                f"{_patch_label(series, patch_names)} "
                f"match {int(series['match_id'])}: "
                f"career={int(series['prior_games_on_hero'])} "
                f"compat={_fmt(series['player_hero_recent_role_compatibility'])} "
                f"recent20={int(series['player_hero_recent_20_matches'])}"
            )


def _print_position_relevance(
    frame: pd.DataFrame, *, patch_names: dict[int, str]
) -> None:
    print("POSITION RELEVANCE (many career games, little at expected_position)")
    career = frame["prior_games_on_hero"].fillna(0)
    share = frame["player_hero_share_at_expected_position"]
    eligible = frame[(career >= _MIN_CAREER_FOR_MISMATCH) & share.notna()]
    if eligible.empty:
        print("  none")
        return
    low = eligible[eligible["player_hero_share_at_expected_position"] <= 0.20]
    support = low[low["expected_position"].isin(_SUPPORT_POSITIONS)]
    core = low[low["expected_position"].isin(_CORE_POSITIONS)]
    print(
        f"  high-career rows with player share at expected ≤ 0.20: "
        f"{len(low)} / {len(eligible)} "
        f"(expected 4/5: {len(support)}; expected 1–3: {len(core)})"
    )
    focus = pd.concat([support, core]).drop_duplicates(
        subset=["match_id", "player_id"]
    )
    focus = focus.sort_values(
        ["player_hero_share_at_expected_position", "prior_games_on_hero"],
        ascending=[True, False],
        kind="mergesort",
    )
    print("  examples:")
    for row in focus.head(_EXAMPLE_LIMIT).itertuples(index=False):
        series = pd.Series(row._asdict())
        print(
            f"    {_player_label(series)} {_hero_label(series)} "
            f"{_patch_label(series, patch_names)} "
            f"match {int(series['match_id'])} expected={series['expected_position']}: "
            f"career={int(series['prior_games_on_hero'])} "
            f"player@expected={_fmt(series['player_hero_share_at_expected_position'])} "
            f"hero@expected={_fmt(series['hero_position_share_at_expected_position'])} "
            f"compat={_fmt(series['player_hero_recent_role_compatibility'])}"
        )


def _print_correlations(frame: pd.DataFrame) -> None:
    print("CORRELATIONS (descriptive, not model selection)")
    print(
        f"  career matches vs recent-20 matches: "
        f"{_fmt(_corr(frame['prior_games_on_hero'], frame['player_hero_recent_20_matches']))}"
    )
    print(
        f"  career matches vs recent-50 matches: "
        f"{_fmt(_corr(frame['prior_games_on_hero'], frame['player_hero_recent_50_matches']))}"
    )
    print(
        f"  career matches vs same-version matches: "
        f"{_fmt(_corr(frame['prior_games_on_hero'], frame['player_hero_same_version_matches']))}"
    )
    both_recent_wr = frame[
        frame["prior_win_rate_on_hero"].notna()
        & frame["player_hero_recent_20_win_rate"].notna()
    ]
    print(
        f"  career WR vs recent-20 WR (both defined, n={len(both_recent_wr)}): "
        f"{_fmt(_corr(both_recent_wr['prior_win_rate_on_hero'], both_recent_wr['player_hero_recent_20_win_rate']))}"
    )
    both_sv_wr = frame[
        frame["prior_win_rate_on_hero"].notna()
        & frame["player_hero_same_version_win_rate"].notna()
    ]
    print(
        f"  career WR vs same-version WR (both defined, n={len(both_sv_wr)}): "
        f"{_fmt(_corr(both_sv_wr['prior_win_rate_on_hero'], both_sv_wr['player_hero_same_version_win_rate']))}"
    )
    both_pos = frame[
        frame["player_hero_position_explicit_games"].fillna(0) > 0
    ]
    both_pos = both_pos[
        both_pos[
            f"hero_recent_{PREFERRED_HERO_META_WINDOW}_position_explicit_count"
        ].fillna(0)
        > 0
    ]
    if both_pos.empty:
        print("  player vs hero-meta position shares: n/a")
        return
    corrs = []
    for position in POSITION_NUMBERS:
        corrs.append(
            _corr(
                both_pos[f"player_hero_position_{position}_share"],
                both_pos[
                    f"hero_recent_{PREFERRED_HERO_META_WINDOW}_position_{position}_share"
                ],
            )
        )
    print(
        f"  player vs hero-meta position-share correlations (p1–p5, n={len(both_pos)}): "
        + ", ".join(_fmt(value) for value in corrs)
    )
    print(
        f"  recent role compatibility vs career matches: "
        f"{_fmt(_corr(frame['player_hero_recent_role_compatibility'], frame['prior_games_on_hero']))}"
    )
    print(
        f"  recent role compatibility vs player share at expected: "
        f"{_fmt(_corr(frame['player_hero_recent_role_compatibility'], frame['player_hero_share_at_expected_position']))}"
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
        frame = build_player_hero_meta(store).to_frame()
        patch_names = _load_patch_names(store)
    elapsed = perf_counter() - started

    print("PLAYER × HERO META RELEVANCE (Slice 6)")
    print(f"  runtime: {elapsed:.1f}s")
    print(
        f"  preferred current-meta window: recent-{PREFERRED_HERO_META_WINDOW} "
        "player-appearance / hero-match trailing windows"
    )
    _print_coverage(frame)
    _print_career_vs_recent(frame)
    _print_career_vs_same_version(frame, patch_names=patch_names)
    _print_role_compatibility(frame, patch_names=patch_names)
    _print_stable_vs_shifted(frame, patch_names=patch_names)
    _print_position_relevance(frame, patch_names=patch_names)
    _print_correlations(frame)

    summary = summarize_player_hero_meta(frame)
    overall = summary[summary["scope"] == "overall"].iloc[0]
    print("SUMMARY TABLE (overall coverage)")
    print(
        f"  career={_fmt(overall['career_coverage'])} "
        f"recent20={_fmt(overall['recent_20_coverage'])} "
        f"same_version={_fmt(overall['same_version_coverage'])} "
        f"player_pos={_fmt(overall['player_position_coverage'])} "
        f"hero_pos={_fmt(overall['hero_meta_position_coverage'])} "
        f"compat={_fmt(overall['role_compatibility_coverage'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
