"""Leakage-safe meta-relevant Player × Hero state (Slice 6).

Grain
-----
One row per current `(match_id, player_id)`: the player on their
**currently drafted hero**, with career familiarity kept intact and
additional histories that ask whether that familiarity is relevant to
the hero's **current competitive meta**.

This is descriptive feature state, not a training feature. It is not
added to `FEATURE_COLUMNS`, PRE_DRAFT snapshot SQL, or any win-model
matrix. It does not fit weights, thresholds, or a composite
"meta strength" / familiarity score.

This module does not replace `features.player_hero`,
`features.player_hero_position`, or `features.hero_state`. Career
Player × Hero semantics stay exactly those of Slice 0's player×hero
layer; Slice 6 exposes alternative / contextualized histories
alongside that baseline.

Conceptual alternatives
-----------------------
1. Career Player × Hero (unconditioned expanding history).
2. Recent Player × Hero (trailing competitive appearances).
3. Same-version Player × Hero.
4. Current hero-meta context (Slice 5, joined on the drafted hero).
5. Player-history × current-role-distribution compatibility.

Temporal integrity
------------------
Every historical match `H` contributing to current match `M` satisfies
`H.start_time < M.start_time` via `RANGE ... CURRENT ROW EXCLUDE GROUP`
(and trailing `list` / `list_slice` over that same player window).
Equal timestamps are mutually blind. History is never ordered by
`match_id`. The current match's own result, draft, and observed
STRATZ `position` / `lane` / `role` never enter that row's metrics.

Recent windows
--------------
Recent Player × Hero counts use the player's last N **strictly prior
competitive appearances** (player-match rows), then count how many of
those were the currently drafted hero. Window widths match Slice 5
(20 / 50 / 100). This is **not** "last N times this player happened
to play the hero", and it is **not** a dense global match×hero grid:
a given player appears in a small fraction of concurrent professional
matches, so a global last-N-match slot would be almost always empty.
Player-appearance trailing windows are the closest leakage-safe analog
at this grain.

A hero not played in the window has zero matches and NULL win rate
(never fabricated 0% / 50%).

Same-version history
--------------------
Only prior matches with the same `game_version_id` contribute.
Patch-opening rows are sparse or zero; they are not backfilled from
later matches or previous versions.

Current hero-meta context
-------------------------
Slice 5 `hero_state` for the currently drafted hero is joined on
`(match_id, hero_id)`. Rates and position shares are the historical
Slice 5 primitives, not a handcrafted score. `expected_position`
(Slice 3) selects `hero_position_share_at_expected_position` from
that historical distribution. Current observed position is never
used. Missing positional evidence is NULL.

Player historical role distribution
-----------------------------------
Shares of observed historical STRATZ positions on this player×hero
(`H.start_time < M.start_time`). NULL / UNKNOWN / FILTERED / ALL
increment no bucket. `player_hero_position_explicit_games` is the
explicit-position sample size.

Role-distribution compatibility
-------------------------------
`player_hero_recent_role_compatibility` is the unweighted dot product
of the player's historical hero-position shares and the hero's
preferred current-meta (recent-50) position shares:

    Σ player_share_p × meta_share_p    for p = 1..5

`player_hero_same_version_role_compatibility` uses same-version hero
position shares when both sides have evidence. NULL if either
distribution has no explicit positional evidence. No weights, no
threshold, no extra product of the expected-position primitives.

Availability
------------
`expected_position` is PRE_DRAFT. Current `hero_id` and every metric
keyed by the drafted hero (including joined Slice 5 columns and
compatibility) are DRAFT. `observed_position` remains POST_MATCH
evaluation-only. This relation does not loosen Slice 0–5 rules.

This module never writes Parquet, never bumps schema versions, and
never alters Elo or expected-position inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    HEROES_VIEW,
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.expected_position import (
    DEFAULT_EXPECTED_POSITION_METHOD,
    build_expected_position,
)
from dota_predictor.features.hero_state import (
    POSITION_NUMBERS,
    RECENT_HERO_MATCH_WINDOWS,
    build_hero_state,
)
from dota_predictor.features.player_position import EXPLICIT_POSITION_LABELS
from dota_predictor.features.temporal import STRICT_PRIOR_RANGE_SQL

__all__ = [
    "PLAYER_HERO_META_COLUMNS",
    "PLAYER_HERO_META_IDENTITY_COLUMNS",
    "PLAYER_HERO_META_METRIC_COLUMNS",
    "PREFERRED_HERO_META_WINDOW",
    "RECENT_PLAYER_HERO_MATCH_WINDOWS",
    "PlayerHeroMetaState",
    "build_player_hero_meta",
    "player_hero_meta_sql",
    "role_compatibility",
    "summarize_player_hero_meta",
]

MATCH_ID_COLUMN = "match_id"
PLAYER_ID_COLUMN = "player_id"
HERO_ID_COLUMN = "hero_id"

# Slice 5-compatible widths; trailing *player appearances*, not a global
# match×hero grid (see module docstring).
RECENT_PLAYER_HERO_MATCH_WINDOWS: tuple[int, ...] = RECENT_HERO_MATCH_WINDOWS
PREFERRED_HERO_META_WINDOW: int = 50

_POSITION_INDEX: dict[str, int] = {
    label: i for i, label in enumerate(EXPLICIT_POSITION_LABELS)
}

_APPEARANCE_STRUCT = "STRUCT(hero_id BIGINT, was_win BIGINT)"
_EMPTY_APPEARANCES = f"CAST([] AS {_APPEARANCE_STRUCT}[])"

PLAYER_HERO_META_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    PLAYER_ID_COLUMN,
    "start_time",
    "game_version_id",
    "team_id",
    "side",
    HERO_ID_COLUMN,
    "hero_name",
    "slot_in_side",
    "expected_position",
    "expected_position_method",
)


def _player_recent_metric_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for window in RECENT_PLAYER_HERO_MATCH_WINDOWS:
        columns.extend(
            [
                f"player_hero_recent_{window}_matches",
                f"player_hero_recent_{window}_wins",
                f"player_hero_recent_{window}_win_rate",
            ]
        )
    return tuple(columns)


def _player_position_metric_columns() -> tuple[str, ...]:
    shares = tuple(
        f"player_hero_position_{position}_share" for position in POSITION_NUMBERS
    )
    return shares + ("player_hero_position_explicit_games",)


def _hero_rate_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for window in RECENT_HERO_MATCH_WINDOWS:
        prefix = f"hero_recent_{window}_"
        columns.extend(
            [
                f"{prefix}contest_rate",
                f"{prefix}pick_rate",
                f"{prefix}ban_rate",
            ]
        )
    columns.extend(
        [
            "hero_same_version_contest_rate",
            "hero_same_version_pick_rate",
            "hero_same_version_ban_rate",
        ]
    )
    return tuple(columns)


def _hero_position_context_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for window in RECENT_HERO_MATCH_WINDOWS:
        prefix = f"hero_recent_{window}_"
        columns.append(f"{prefix}position_explicit_count")
        columns.extend(
            f"{prefix}position_{position}_share" for position in POSITION_NUMBERS
        )
    columns.append("hero_same_version_position_explicit_count")
    columns.extend(
        f"hero_same_version_position_{position}_share"
        for position in POSITION_NUMBERS
    )
    return tuple(columns)


HERO_META_CONTEXT_COLUMNS: tuple[str, ...] = (
    _hero_rate_columns() + _hero_position_context_columns()
)

PLAYER_HERO_META_METRIC_COLUMNS: tuple[str, ...] = (
    (
        "prior_games_on_hero",
        "prior_wins_on_hero",
        "prior_win_rate_on_hero",
        "prior_player_games",
    )
    + _player_recent_metric_columns()
    + (
        "player_hero_same_version_matches",
        "player_hero_same_version_wins",
        "player_hero_same_version_win_rate",
    )
    + _player_position_metric_columns()
    + HERO_META_CONTEXT_COLUMNS
    + (
        "player_hero_recent_role_compatibility",
        "player_hero_same_version_role_compatibility",
        "player_hero_share_at_expected_position",
        "hero_position_share_at_expected_position",
        "hero_meta_share_at_expected_position",
    )
)

PLAYER_HERO_META_COLUMNS: tuple[str, ...] = (
    PLAYER_HERO_META_IDENTITY_COLUMNS
    + PLAYER_HERO_META_METRIC_COLUMNS
    + ("observed_position",)
)


def _rate_sql(numerator: str, denominator: str) -> str:
    return (
        f"CASE WHEN {denominator} > 0 "
        f"THEN {numerator}::DOUBLE / {denominator} "
        f"ELSE NULL END"
    )


def _hero_name_select(*, catalog_registered: bool) -> str:
    if catalog_registered:
        return f"{HEROES_VIEW}.name AS hero_name"
    return "CAST(NULL AS VARCHAR) AS hero_name"


def _hero_name_join(*, catalog_registered: bool) -> str:
    if catalog_registered:
        return f"LEFT JOIN {HEROES_VIEW} ON {HEROES_VIEW}.hero_id = s.hero_id"
    return ""


def _position_flag_sql() -> str:
    parts = [
        (
            f"CASE WHEN position = '{label}' THEN 1 ELSE 0 END "
            f"AS at_position_{index}"
        )
        for index, label in enumerate(EXPLICIT_POSITION_LABELS, start=1)
    ]
    return ",\n        ".join(parts)


def _recent_slice_sql(list_expr: str, window: int) -> str:
    return (
        f"CASE WHEN {list_expr} IS NULL OR len({list_expr}) = 0 THEN "
        f"{_EMPTY_APPEARANCES} ELSE list_slice({list_expr}, "
        f"GREATEST(len({list_expr}) - {window - 1}, 1), len({list_expr})) END"
    )


def _recent_match_sql(list_expr: str) -> str:
    return (
        f"COALESCE(len(list_filter({list_expr}, "
        f"x -> x.hero_id = s.hero_id)), 0)::BIGINT"
    )


def _recent_wins_sql(list_expr: str) -> str:
    filtered = f"list_filter({list_expr}, x -> x.hero_id = s.hero_id)"
    return (
        f"COALESCE(list_sum(list_transform({filtered}, x -> x.was_win)), 0)"
        "::BIGINT"
    )


def player_hero_meta_sql(
    *, catalog_registered: bool = True, match_id: int | None = None
) -> str:
    """SQL for career / recent / same-version Player × Hero plus position shares.

    Does not mention `expected_position`. Slice 5 hero-meta context and
    expected-position selection happen after this relation is built.
    Recent windows are trailing player appearances (see module docstring).
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE s.match_id = {int(match_id)}"

    hero_name = _hero_name_select(catalog_registered=catalog_registered)
    hero_join = _hero_name_join(catalog_registered=catalog_registered)
    position_counts = ",\n        ".join(
        (
            f"COALESCE(SUM(at_position_{position}) OVER w_ph, 0)::BIGINT "
            f"AS player_hero_position_{position}_count"
        )
        for position in POSITION_NUMBERS
    )
    recent_slices = ",\n        ".join(
        f"{_recent_slice_sql('w.prior_appearances', window)} "
        f"AS recent_{window}_appearances"
        for window in RECENT_PLAYER_HERO_MATCH_WINDOWS
    )
    recent_metrics = []
    for window in RECENT_PLAYER_HERO_MATCH_WINDOWS:
        matches = _recent_match_sql(f"s.recent_{window}_appearances")
        wins = _recent_wins_sql(f"s.recent_{window}_appearances")
        recent_metrics.extend(
            [
                f"{matches} AS player_hero_recent_{window}_matches",
                f"{wins} AS player_hero_recent_{window}_wins",
                (
                    f"{_rate_sql(wins, matches)} "
                    f"AS player_hero_recent_{window}_win_rate"
                ),
            ]
        )
    sliced_explicit = " + ".join(
        f"s.player_hero_position_{position}_count" for position in POSITION_NUMBERS
    )
    position_shares = []
    for position in POSITION_NUMBERS:
        count = f"s.player_hero_position_{position}_count"
        share = f"player_hero_position_{position}_share"
        position_shares.append(
            f"{_rate_sql(count, f'({sliced_explicit})')} AS {share}"
        )

    return f"""
WITH appearances AS (
    SELECT
        mp.match_id,
        mp.start_time,
        m.game_version_id,
        mp.player_id,
        mp.hero_id,
        mp.side,
        mp.team_id,
        mp.slot_in_side,
        mp.position,
        CASE
            WHEN mp.side = 'RADIANT' THEN m.radiant_win
            ELSE NOT m.radiant_win
        END AS player_won
    FROM {MATCH_PLAYERS_VIEW} mp
    JOIN {MATCHES_VIEW} m ON m.match_id = mp.match_id
),

flagged AS (
    SELECT
        match_id,
        start_time,
        game_version_id,
        player_id,
        hero_id,
        side,
        team_id,
        slot_in_side,
        position,
        CASE WHEN player_won THEN 1 ELSE 0 END AS was_win,
        {_position_flag_sql()}
    FROM appearances
),

windowed AS (
    SELECT
        match_id,
        start_time,
        game_version_id,
        player_id,
        hero_id,
        side,
        team_id,
        slot_in_side,
        position,
        COALESCE(COUNT(*) OVER w_ph, 0)::BIGINT AS prior_games_on_hero,
        COALESCE(SUM(was_win) OVER w_ph, 0)::BIGINT AS prior_wins_on_hero,
        COALESCE(COUNT(*) OVER w_player, 0)::BIGINT AS prior_player_games,
        COALESCE(COUNT(*) OVER w_sv, 0)::BIGINT AS player_hero_same_version_matches,
        COALESCE(SUM(was_win) OVER w_sv, 0)::BIGINT AS player_hero_same_version_wins,
        list(struct_pack(hero_id := hero_id, was_win := was_win))
            OVER w_player AS prior_appearances,
        {position_counts}
    FROM flagged
    WINDOW
        w_player AS (
            PARTITION BY player_id
            ORDER BY start_time
            {STRICT_PRIOR_RANGE_SQL}
        ),
        w_ph AS (
            PARTITION BY player_id, hero_id
            ORDER BY start_time
            {STRICT_PRIOR_RANGE_SQL}
        ),
        w_sv AS (
            PARTITION BY player_id, hero_id, game_version_id
            ORDER BY start_time
            {STRICT_PRIOR_RANGE_SQL}
        )
),

sliced AS (
    SELECT
        w.*,
        {recent_slices}
    FROM windowed AS w
)

SELECT
    s.match_id,
    s.player_id,
    s.start_time,
    s.game_version_id,
    s.team_id,
    s.side,
    s.hero_id,
    {hero_name},
    s.slot_in_side,
    s.position AS observed_position,
    s.prior_games_on_hero,
    s.prior_wins_on_hero,
    {_rate_sql('s.prior_wins_on_hero', 's.prior_games_on_hero')}
        AS prior_win_rate_on_hero,
    s.prior_player_games,
    {", ".join(recent_metrics)},
    s.player_hero_same_version_matches,
    s.player_hero_same_version_wins,
    {_rate_sql('s.player_hero_same_version_wins', 's.player_hero_same_version_matches')}
        AS player_hero_same_version_win_rate,
    {", ".join(position_shares)},
    ({sliced_explicit})::BIGINT AS player_hero_position_explicit_games
FROM sliced AS s
{hero_join}
{output_filter}
"""


def _heroes_view_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return HEROES_VIEW in tables


def role_compatibility(
    player_shares: pd.DataFrame | np.ndarray,
    meta_shares: pd.DataFrame | np.ndarray,
    *,
    player_explicit: pd.Series | np.ndarray,
    meta_explicit: pd.Series | np.ndarray,
) -> pd.Series:
    """Unweighted dot product of two 5-way position distributions.

    NULL when either side has no explicit positional evidence. Does not
    fit weights or apply a threshold.
    """
    player = np.asarray(player_shares, dtype=float)
    meta = np.asarray(meta_shares, dtype=float)
    product = np.sum(player * meta, axis=1)
    player_n = pd.to_numeric(pd.Series(player_explicit), errors="coerce")
    meta_n = pd.to_numeric(pd.Series(meta_explicit), errors="coerce")
    finite = np.isfinite(product)
    valid = (player_n.to_numpy() > 0) & (meta_n.to_numpy() > 0) & finite
    index = getattr(player_shares, "index", None)
    result = pd.Series(product, index=index, dtype=float)
    return result.where(valid)


def _pick_at_expected(
    frame: pd.DataFrame, expected: pd.Series, *, prefix: str
) -> pd.Series:
    columns = [f"{prefix}{position}_share" for position in POSITION_NUMBERS]
    matrix = frame[columns].to_numpy(dtype=float)
    indexes = expected.map(_POSITION_INDEX)
    if indexes.isna().any():
        raise ValueError("expected_position must be an explicit POSITION_1–5 label")
    chosen = matrix[np.arange(len(frame)), indexes.to_numpy(dtype=np.intp)]
    return pd.Series(chosen, index=frame.index)


def attach_meta_relevance(
    windowed: pd.DataFrame, expected: pd.DataFrame, hero_state: pd.DataFrame
) -> pd.DataFrame:
    """Join Slice 3 expected position and Slice 5 hero state; derive compatibility."""
    keys = expected[
        ["match_id", "player_id", "expected_position", "method"]
    ].rename(columns={"method": "expected_position_method"})
    merged = windowed.merge(
        keys, on=["match_id", "player_id"], how="inner", validate="1:1"
    )
    hero_keep = ["match_id", "hero_id", *HERO_META_CONTEXT_COLUMNS]
    missing = set(hero_keep) - set(hero_state.columns)
    if missing:
        raise ValueError(f"hero_state is missing Slice 5 columns: {sorted(missing)}")
    merged = merged.merge(
        hero_state[hero_keep],
        on=["match_id", "hero_id"],
        how="left",
        validate="m:1",
    )
    expected_col = merged["expected_position"]
    merged["player_hero_share_at_expected_position"] = _pick_at_expected(
        merged, expected_col, prefix="player_hero_position_"
    )
    preferred = f"hero_recent_{PREFERRED_HERO_META_WINDOW}_"
    merged["hero_position_share_at_expected_position"] = _pick_at_expected(
        merged, expected_col, prefix=preferred + "position_"
    )
    merged["hero_meta_share_at_expected_position"] = merged[
        "hero_position_share_at_expected_position"
    ]
    player_share_cols = [
        f"player_hero_position_{position}_share" for position in POSITION_NUMBERS
    ]
    recent_meta_cols = [
        f"{preferred}position_{position}_share" for position in POSITION_NUMBERS
    ]
    sv_meta_cols = [
        f"hero_same_version_position_{position}_share" for position in POSITION_NUMBERS
    ]
    merged["player_hero_recent_role_compatibility"] = role_compatibility(
        merged[player_share_cols],
        merged[recent_meta_cols],
        player_explicit=merged["player_hero_position_explicit_games"],
        meta_explicit=merged[f"{preferred}position_explicit_count"],
    )
    merged["player_hero_same_version_role_compatibility"] = role_compatibility(
        merged[player_share_cols],
        merged[sv_meta_cols],
        player_explicit=merged["player_hero_position_explicit_games"],
        meta_explicit=merged["hero_same_version_position_explicit_count"],
    )
    return merged[list(PLAYER_HERO_META_COLUMNS)]


@dataclass(frozen=True)
class PlayerHeroMetaState:
    """Materialized meta-relevant Player × Hero state."""

    frame: pd.DataFrame
    method: str

    def to_frame(self) -> pd.DataFrame:
        ordered = self.frame[list(PLAYER_HERO_META_COLUMNS)]
        return ordered.sort_values(
            [MATCH_ID_COLUMN, PLAYER_ID_COLUMN], kind="mergesort"
        ).reset_index(drop=True)


def build_player_hero_meta(
    store: FeatureDuckDBConnection,
    *,
    method: str = DEFAULT_EXPECTED_POSITION_METHOD,
    match_id: int | None = None,
) -> PlayerHeroMetaState:
    """Build leakage-safe meta-relevant Player × Hero state.

    Independent of PRE_DRAFT snapshot SQL, Elo, and the win-model
    feature matrix. Does not modify Player × Hero, Player × Hero ×
    Position, expected-position inference, or Slice 5 hero-state
    semantics.
    """
    expected = build_expected_position(
        store, method=method, match_id=match_id
    ).to_frame()
    windowed = store.sql(
        player_hero_meta_sql(
            catalog_registered=_heroes_view_registered(store),
            match_id=match_id,
        )
    ).df()
    hero = build_hero_state(store, match_id=match_id).to_frame()
    attached = attach_meta_relevance(windowed, expected, hero)
    return PlayerHeroMetaState(frame=attached, method=method)


def _scope_summary(frame: pd.DataFrame, *, scope: str, key: object) -> dict[str, object]:
    n = len(frame)
    career = frame["prior_games_on_hero"].fillna(0)
    recent20 = frame["player_hero_recent_20_matches"].fillna(0)
    recent50 = frame["player_hero_recent_50_matches"].fillna(0)
    recent100 = frame["player_hero_recent_100_matches"].fillna(0)
    same_version = frame["player_hero_same_version_matches"].fillna(0)
    player_pos = frame["player_hero_position_explicit_games"].fillna(0)
    hero_pos = frame[
        f"hero_recent_{PREFERRED_HERO_META_WINDOW}_position_explicit_count"
    ].fillna(0)
    compat = frame["player_hero_recent_role_compatibility"]
    return {
        "scope": scope,
        "key": key,
        "n_rows": n,
        "career_coverage": float((career > 0).mean()) if n else float("nan"),
        "recent_20_coverage": float((recent20 > 0).mean()) if n else float("nan"),
        "recent_50_coverage": float((recent50 > 0).mean()) if n else float("nan"),
        "recent_100_coverage": float((recent100 > 0).mean()) if n else float("nan"),
        "same_version_coverage": (
            float((same_version > 0).mean()) if n else float("nan")
        ),
        "player_position_coverage": (
            float((player_pos > 0).mean()) if n else float("nan")
        ),
        "hero_meta_position_coverage": (
            float((hero_pos > 0).mean()) if n else float("nan")
        ),
        "role_compatibility_coverage": (
            float(compat.notna().mean()) if n else float("nan")
        ),
        "mean_career_games": float(career.mean()) if n else float("nan"),
        "mean_recent_20_matches": float(recent20.mean()) if n else float("nan"),
        "mean_same_version_matches": float(same_version.mean()) if n else float("nan"),
        "mean_recent_role_compatibility": (
            float(compat.mean()) if compat.notna().any() else float("nan")
        ),
    }


def summarize_player_hero_meta(frame: pd.DataFrame) -> pd.DataFrame:
    """Coverage of career / recent / same-version / compatibility state."""
    rows = [_scope_summary(frame, scope="overall", key="all")]
    for version, subset in frame.groupby("game_version_id", sort=True):
        rows.append(_scope_summary(subset, scope="game_version_id", key=version))
    for label, subset in frame.groupby("expected_position", sort=True):
        rows.append(_scope_summary(subset, scope="expected_position", key=label))
    return pd.DataFrame.from_records(rows)
