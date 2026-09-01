"""Leakage-safe Player × Hero × expected-position historical state.

Grain
-----
One row per current `(match_id, player_id)`: the player on their currently
drafted hero, with Player × Hero familiarity both unconditioned and
restricted to the Slice 3 PRE_DRAFT `expected_position`.

This is descriptive feature state, not a training feature. It is not
added to `FEATURE_COLUMNS` or PRE_DRAFT snapshot SQL.

Observed vs expected
--------------------
Historical Player × Hero × position counts use **observed** STRATZ
`position` on strictly earlier matches (`H.start_time < M.start_time`).
Those labels are POST_MATCH facts of completed games.

The current match selects which of those five historical buckets to
expose as `*_at_expected_position` using **expected_position**, never
the current match's observed `position` / `lane` / `role`.

Current `hero_id` is the DRAFT lookup key, as in unconditioned
Player × Hero. Current observed position is copied as
`observed_position` for evaluation only.

Temporal integrity
------------------
Windows use `RANGE ... CURRENT ROW EXCLUDE GROUP` so equal timestamps
are mutually blind and the current row never enters its own counts.
History is never ordered by `match_id`. NULL historical positions occupy
no POSITION_1–5 bucket.

Availability
------------
`expected_position` is PRE_DRAFT. Current `hero_id` and the hero-keyed
metrics are DRAFT (knowable only after the pick). `observed_position`
remains POST_MATCH.
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
from dota_predictor.features.player_hero import RECENT_WINDOW_DAYS
from dota_predictor.features.player_position import EXPLICIT_POSITION_LABELS
from dota_predictor.features.temporal import STRICT_PRIOR_RANGE_SQL

__all__ = [
    "PLAYER_HERO_POSITION_COLUMNS",
    "PLAYER_HERO_POSITION_IDENTITY_COLUMNS",
    "PLAYER_HERO_POSITION_METRIC_COLUMNS",
    "PlayerHeroPositionState",
    "build_player_hero_position",
    "player_hero_position_sql",
    "summarize_player_hero_position",
]

MATCH_ID_COLUMN = "match_id"
PLAYER_ID_COLUMN = "player_id"
_MICROSECONDS_PER_DAY = 86_400_000_000.0
_RECENT_90D_RANGE = (
    f"RANGE BETWEEN INTERVAL {RECENT_WINDOW_DAYS} DAY PRECEDING "
    f"AND CURRENT ROW EXCLUDE GROUP"
)
_POSITION_INDEX: dict[str, int] = {
    label: i for i, label in enumerate(EXPLICIT_POSITION_LABELS)
}

PLAYER_HERO_POSITION_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    PLAYER_ID_COLUMN,
    "start_time",
    "game_version_id",
    "team_id",
    "side",
    "hero_id",
    "hero_name",
    "slot_in_side",
    "expected_position",
    "expected_position_method",
)

PLAYER_HERO_POSITION_METRIC_COLUMNS: tuple[str, ...] = (
    "prior_games_on_hero",
    "prior_wins_on_hero",
    "prior_losses_on_hero",
    "prior_win_rate_on_hero",
    "same_version_games_on_hero",
    "same_version_wins_on_hero",
    "same_version_win_rate_on_hero",
    "recent_90d_games_on_hero",
    "recent_90d_wins_on_hero",
    "recent_90d_win_rate_on_hero",
    "prior_player_games",
    "prior_hero_share",
    "recent_90d_player_games",
    "recent_90d_hero_share",
    "days_since_last_played_hero",
    "prior_games_on_hero_at_expected_position",
    "prior_wins_on_hero_at_expected_position",
    "prior_losses_on_hero_at_expected_position",
    "prior_win_rate_on_hero_at_expected_position",
    "same_version_games_on_hero_at_expected_position",
    "same_version_wins_on_hero_at_expected_position",
    "same_version_win_rate_on_hero_at_expected_position",
    "recent_90d_games_on_hero_at_expected_position",
    "recent_90d_wins_on_hero_at_expected_position",
    "recent_90d_win_rate_on_hero_at_expected_position",
    "prior_hero_share_at_expected_position",
    "prior_position_share_on_hero",
    "days_since_last_played_hero_at_expected_position",
)

PLAYER_HERO_POSITION_COLUMNS: tuple[str, ...] = (
    PLAYER_HERO_POSITION_IDENTITY_COLUMNS
    + PLAYER_HERO_POSITION_METRIC_COLUMNS
    + ("observed_position",)
)


def _suffix(label: str) -> str:
    return label.lower()


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
        return f"LEFT JOIN {HEROES_VIEW} ON {HEROES_VIEW}.hero_id = w.hero_id"
    return ""


def _days_since_sql(earlier: str, later: str) -> str:
    return (
        f"CASE WHEN {earlier} IS NULL THEN NULL ELSE "
        f"date_diff('microsecond', {earlier}, {later})::DOUBLE "
        f"/ {_MICROSECONDS_PER_DAY} END"
    )


def _position_window_select() -> str:
    """Historical Player × Hero counts split by observed position.

    FILTER uses historical `position` inside player×hero windows. The
    current row is omitted by EXCLUDE GROUP, so current observed
    position cannot increment these counts.
    """
    parts: list[str] = []
    for label in EXPLICIT_POSITION_LABELS:
        suffix = _suffix(label)
        parts.extend(
            [
                (
                    "COALESCE(COUNT(*) FILTER (WHERE position = "
                    f"'{label}') OVER w_ph, 0)::BIGINT "
                    f"AS prior_games_on_hero_{suffix}"
                ),
                (
                    "COALESCE(SUM(CASE WHEN position = "
                    f"'{label}' THEN was_win ELSE 0 END) OVER w_ph, 0)"
                    f"::BIGINT AS prior_wins_on_hero_{suffix}"
                ),
                (
                    "COALESCE(COUNT(*) FILTER (WHERE position = "
                    f"'{label}') OVER w_sv, 0)::BIGINT "
                    f"AS same_version_games_on_hero_{suffix}"
                ),
                (
                    "COALESCE(SUM(CASE WHEN position = "
                    f"'{label}' THEN was_win ELSE 0 END) OVER w_sv, 0)"
                    f"::BIGINT AS same_version_wins_on_hero_{suffix}"
                ),
                (
                    "COALESCE(COUNT(*) FILTER (WHERE position = "
                    f"'{label}') OVER w_90_ph, 0)::BIGINT "
                    f"AS recent_90d_games_on_hero_{suffix}"
                ),
                (
                    "COALESCE(SUM(CASE WHEN position = "
                    f"'{label}' THEN was_win ELSE 0 END) OVER w_90_ph, 0)"
                    f"::BIGINT AS recent_90d_wins_on_hero_{suffix}"
                ),
                (
                    f"MAX(CASE WHEN position = '{label}' THEN start_time END) "
                    f"OVER w_ph AS last_played_hero_at_{suffix}"
                ),
            ]
        )
    return ",\n        ".join(parts)


def player_hero_position_sql(
    *, catalog_registered: bool = True, match_id: int | None = None
) -> str:
    """SQL for unconditioned Player × Hero plus per-position historical buckets.

    Does not mention `expected_position`. Conditioning happens after the
    Slice 3 assignment is joined in Python.
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE w.match_id = {int(match_id)}"

    win_rate = _rate_sql("w.prior_wins_on_hero", "w.prior_games_on_hero")
    win_rate_sv = _rate_sql(
        "w.same_version_wins_on_hero", "w.same_version_games_on_hero"
    )
    win_rate_r90 = _rate_sql("w.recent_90d_wins_on_hero", "w.recent_90d_games_on_hero")
    hero_share = _rate_sql("w.prior_games_on_hero", "w.prior_player_games")
    hero_share_r90 = _rate_sql("w.recent_90d_games_on_hero", "w.recent_90d_player_games")
    days_since = _days_since_sql("w.last_played_hero_at", "w.start_time")
    hero_name = _hero_name_select(catalog_registered=catalog_registered)
    hero_join = _hero_name_join(catalog_registered=catalog_registered)
    position_select = _position_window_select()
    position_days = ",\n    ".join(
        _days_since_sql(f"w.last_played_hero_at_{_suffix(label)}", "w.start_time")
        + f" AS days_since_last_played_hero_{_suffix(label)}"
        for label in EXPLICIT_POSITION_LABELS
    )
    position_out = ",\n    ".join(
        f"w.prior_games_on_hero_{_suffix(label)},\n    "
        f"w.prior_wins_on_hero_{_suffix(label)},\n    "
        f"w.same_version_games_on_hero_{_suffix(label)},\n    "
        f"w.same_version_wins_on_hero_{_suffix(label)},\n    "
        f"w.recent_90d_games_on_hero_{_suffix(label)},\n    "
        f"w.recent_90d_wins_on_hero_{_suffix(label)}"
        for label in EXPLICIT_POSITION_LABELS
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
        CASE WHEN player_won = FALSE THEN 1 ELSE 0 END AS was_loss
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
        COALESCE(SUM(was_loss) OVER w_ph, 0)::BIGINT AS prior_losses_on_hero,
        COALESCE(COUNT(*) OVER w_sv, 0)::BIGINT AS same_version_games_on_hero,
        COALESCE(SUM(was_win) OVER w_sv, 0)::BIGINT AS same_version_wins_on_hero,
        COALESCE(COUNT(*) OVER w_90_ph, 0)::BIGINT AS recent_90d_games_on_hero,
        COALESCE(SUM(was_win) OVER w_90_ph, 0)::BIGINT AS recent_90d_wins_on_hero,
        COALESCE(COUNT(*) OVER w_player, 0)::BIGINT AS prior_player_games,
        COALESCE(COUNT(*) OVER w_90_player, 0)::BIGINT AS recent_90d_player_games,
        MAX(start_time) OVER w_ph AS last_played_hero_at,
        {position_select}
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
        ),
        w_90_player AS (
            PARTITION BY player_id
            ORDER BY start_time
            {_RECENT_90D_RANGE}
        ),
        w_90_ph AS (
            PARTITION BY player_id, hero_id
            ORDER BY start_time
            {_RECENT_90D_RANGE}
        )
)

SELECT
    w.match_id,
    w.player_id,
    w.start_time,
    w.game_version_id,
    w.team_id,
    w.side,
    w.hero_id,
    {hero_name},
    w.slot_in_side,
    w.position AS observed_position,
    w.prior_games_on_hero,
    w.prior_wins_on_hero,
    w.prior_losses_on_hero,
    {win_rate} AS prior_win_rate_on_hero,
    w.same_version_games_on_hero,
    w.same_version_wins_on_hero,
    {win_rate_sv} AS same_version_win_rate_on_hero,
    w.recent_90d_games_on_hero,
    w.recent_90d_wins_on_hero,
    {win_rate_r90} AS recent_90d_win_rate_on_hero,
    w.prior_player_games,
    {hero_share} AS prior_hero_share,
    w.recent_90d_player_games,
    {hero_share_r90} AS recent_90d_hero_share,
    {days_since} AS days_since_last_played_hero,
    {position_out},
    {position_days}
FROM windowed AS w
{hero_join}
{output_filter}
"""


def _heroes_view_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return HEROES_VIEW in tables


def _pick_at_expected(
    frame: pd.DataFrame, expected: pd.Series, *, prefix: str
) -> pd.Series:
    columns = [f"{prefix}_{_suffix(label)}" for label in EXPLICIT_POSITION_LABELS]
    matrix = frame[columns].to_numpy()
    indexes = expected.map(_POSITION_INDEX)
    if indexes.isna().any():
        raise ValueError("expected_position must be an explicit POSITION_1–5 label")
    chosen = matrix[np.arange(len(frame)), indexes.to_numpy(dtype=np.intp)]
    return pd.Series(chosen, index=frame.index)


def _rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    games = pd.to_numeric(denominator, errors="coerce")
    wins = pd.to_numeric(numerator, errors="coerce")
    return wins.where(games > 0) / games.where(games > 0)


def attach_expected_position_metrics(
    windowed: pd.DataFrame, expected: pd.DataFrame
) -> pd.DataFrame:
    """Select per-position buckets using PRE_DRAFT expected_position."""
    keys = expected[
        ["match_id", "player_id", "expected_position", "method"]
    ].rename(columns={"method": "expected_position_method"})
    merged = windowed.merge(keys, on=["match_id", "player_id"], how="inner", validate="1:1")
    expected_col = merged["expected_position"]
    games = _pick_at_expected(merged, expected_col, prefix="prior_games_on_hero")
    wins = _pick_at_expected(merged, expected_col, prefix="prior_wins_on_hero")
    sv_games = _pick_at_expected(
        merged, expected_col, prefix="same_version_games_on_hero"
    )
    sv_wins = _pick_at_expected(
        merged, expected_col, prefix="same_version_wins_on_hero"
    )
    r90_games = _pick_at_expected(
        merged, expected_col, prefix="recent_90d_games_on_hero"
    )
    r90_wins = _pick_at_expected(
        merged, expected_col, prefix="recent_90d_wins_on_hero"
    )
    merged["prior_games_on_hero_at_expected_position"] = pd.to_numeric(games)
    merged["prior_wins_on_hero_at_expected_position"] = pd.to_numeric(wins)
    merged["prior_losses_on_hero_at_expected_position"] = (
        merged["prior_games_on_hero_at_expected_position"]
        - merged["prior_wins_on_hero_at_expected_position"]
    )
    merged["prior_win_rate_on_hero_at_expected_position"] = _rate(wins, games)
    merged["same_version_games_on_hero_at_expected_position"] = pd.to_numeric(sv_games)
    merged["same_version_wins_on_hero_at_expected_position"] = pd.to_numeric(sv_wins)
    merged["same_version_win_rate_on_hero_at_expected_position"] = _rate(
        sv_wins, sv_games
    )
    merged["recent_90d_games_on_hero_at_expected_position"] = pd.to_numeric(r90_games)
    merged["recent_90d_wins_on_hero_at_expected_position"] = pd.to_numeric(r90_wins)
    merged["recent_90d_win_rate_on_hero_at_expected_position"] = _rate(
        r90_wins, r90_games
    )
    merged["prior_hero_share_at_expected_position"] = _rate(
        merged["prior_games_on_hero_at_expected_position"],
        merged["prior_player_games"],
    )
    merged["prior_position_share_on_hero"] = _rate(
        merged["prior_games_on_hero_at_expected_position"],
        merged["prior_games_on_hero"],
    )
    merged["days_since_last_played_hero_at_expected_position"] = _pick_at_expected(
        merged, expected_col, prefix="days_since_last_played_hero"
    )
    return merged[list(PLAYER_HERO_POSITION_COLUMNS)]


@dataclass(frozen=True)
class PlayerHeroPositionState:
    """Materialized Player × Hero × expected-position state."""

    frame: pd.DataFrame
    method: str

    def to_frame(self) -> pd.DataFrame:
        ordered = self.frame[list(PLAYER_HERO_POSITION_COLUMNS)]
        return ordered.sort_values(
            [MATCH_ID_COLUMN, PLAYER_ID_COLUMN], kind="mergesort"
        ).reset_index(drop=True)


def build_player_hero_position(
    store: FeatureDuckDBConnection,
    *,
    method: str = DEFAULT_EXPECTED_POSITION_METHOD,
    match_id: int | None = None,
) -> PlayerHeroPositionState:
    """Build unconditioned and expected-position-conditioned Player × Hero state."""
    expected = build_expected_position(store, method=method, match_id=match_id).to_frame()
    windowed = store.sql(
        player_hero_position_sql(
            catalog_registered=_heroes_view_registered(store),
            match_id=match_id,
        )
    ).df()
    attached = attach_expected_position_metrics(windowed, expected)
    return PlayerHeroPositionState(frame=attached, method=method)


def _scope_summary(frame: pd.DataFrame, *, scope: str, key: object) -> dict[str, object]:
    uncond = frame["prior_games_on_hero"].fillna(0)
    cond = frame["prior_games_on_hero_at_expected_position"].fillna(0)
    both_rates = frame[
        (uncond > 0)
        & (cond > 0)
        & frame["prior_win_rate_on_hero"].notna()
        & frame["prior_win_rate_on_hero_at_expected_position"].notna()
    ]
    wr_un = both_rates["prior_win_rate_on_hero"]
    wr_cond = both_rates["prior_win_rate_on_hero_at_expected_position"]
    corr = wr_un.corr(wr_cond) if len(both_rates) > 1 else float("nan")
    return {
        "scope": scope,
        "key": key,
        "n_rows": len(frame),
        "unconditioned_coverage": float((uncond > 0).mean()) if len(frame) else float("nan"),
        "conditioned_coverage": float((cond > 0).mean()) if len(frame) else float("nan"),
        "played_hero_not_at_expected_position": (
            float(((uncond > 0) & (cond == 0)).mean()) if len(frame) else float("nan")
        ),
        "mean_prior_games_on_hero": float(uncond.mean()) if len(frame) else float("nan"),
        "mean_prior_games_at_expected_position": (
            float(cond.mean()) if len(frame) else float("nan")
        ),
        "n_both_win_rates": len(both_rates),
        "mean_abs_win_rate_delta": (
            float((wr_un - wr_cond).abs().mean()) if len(both_rates) else float("nan")
        ),
        "win_rate_correlation": float(corr) if pd.notna(corr) else float("nan"),
        "mean_unconditioned_win_rate": (
            float(wr_un.mean()) if len(both_rates) else float("nan")
        ),
        "mean_conditioned_win_rate": (
            float(wr_cond.mean()) if len(both_rates) else float("nan")
        ),
    }


def summarize_player_hero_position(frame: pd.DataFrame) -> pd.DataFrame:
    """Coverage and win-rate comparison overall, by patch, and by expected position."""
    rows = [_scope_summary(frame, scope="overall", key="all")]
    for version, subset in frame.groupby("game_version_id", sort=True):
        rows.append(_scope_summary(subset, scope="game_version_id", key=version))
    for label, subset in frame.groupby("expected_position", sort=True):
        rows.append(_scope_summary(subset, scope="expected_position", key=label))
    return pd.DataFrame.from_records(rows)
