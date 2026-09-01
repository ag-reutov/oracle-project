"""Match-level Radiant − Dire aggregation of Slice 6 Player × Hero meta.

Grain
-----
Slice 6 is ``(match_id, player_id)``. Prediction specs need one row per
``match_id``. This module does **not** invent new scores. It applies the
existing career Player × Hero draft-profile aggregation to named Slice 6
columns, then subtracts Dire from Radiant the same way
``draft_comparison`` does.

Aggregation (mirrors ``draft_profile`` Player × Hero)
-----------------------------------------------------
For a **count** column (never NULL; 0 is observed zero history):

* ``mean_{column}`` — SQL ``AVG`` / pandas ``mean``
* ``min_{column}`` — SQL ``MIN`` / pandas ``min``
* ``players_with_zero_{column}`` — count of players with value ``== 0``

For a **rate / share / compatibility** column (NULL = no evidence):

* ``mean_{column}`` — SQL ``AVG``, which skips NULL rather than treating
  it as 0%. A side whose five players all lack evidence is NULL.

Win-rate missingness
--------------------
Zero same-version / recent-20 matches is **not** a 0% win rate. The
count stays 0 and the rate stays NULL. After aggregation, a side with
no observed rates has a NULL mean rate. Observed 0% (positive count,
zero wins) remains 0.0. Preprocessing for the walk-forward logistic
(median impute + ``__was_missing``) is applied later on TRAIN only; this
module does not impute.

This layer is evaluation plumbing. It is not part of ``FEATURE_COLUMNS``
or PRE_DRAFT snapshot SQL. It never writes Parquet.
"""

from __future__ import annotations

import pandas as pd

from dota_predictor.features.draft_profile import SIDE_COLUMN

__all__ = [
    "MATCH_ID_COLUMN",
    "SLICE7_COMPARISON_COLUMNS",
    "SLICE7_COUNT_SOURCE_COLUMNS",
    "SLICE7_RATE_SOURCE_COLUMNS",
    "SLICE7_RECENT20_COUNT_DIFF_COLUMNS",
    "SLICE7_RECENT20_RATE_DIFF_COLUMNS",
    "SLICE7_ROLE_DIFF_COLUMNS",
    "SLICE7_SAME_VERSION_COUNT_DIFF_COLUMNS",
    "SLICE7_SAME_VERSION_RATE_DIFF_COLUMNS",
    "SLICE7_SIDE_METRIC_COLUMNS",
    "count_side_metric_columns",
    "player_hero_meta_comparison_from_players",
    "player_hero_meta_side_profile",
    "rate_side_metric_columns",
    "slice7_diff_column",
]

MATCH_ID_COLUMN = "match_id"
_RADIANT_SIDE = "RADIANT"
_DIRE_SIDE = "DIRE"

SLICE7_COUNT_SOURCE_COLUMNS: tuple[str, ...] = (
    "player_hero_same_version_matches",
    "player_hero_recent_20_matches",
)

SLICE7_RATE_SOURCE_COLUMNS: tuple[str, ...] = (
    "player_hero_same_version_win_rate",
    "player_hero_recent_20_win_rate",
    "player_hero_recent_role_compatibility",
    "player_hero_share_at_expected_position",
    "hero_meta_share_at_expected_position",
)


def count_side_metric_columns(source: str) -> tuple[str, str, str]:
    """Mean, min, and zero-history count names for one count column."""
    return (
        f"mean_{source}",
        f"min_{source}",
        f"players_with_zero_{source}",
    )


def rate_side_metric_columns(source: str) -> tuple[str, ...]:
    """Mean-only name for one rate/share/compatibility column."""
    return (f"mean_{source}",)


def slice7_diff_column(side_metric: str) -> str:
    return f"{side_metric}_diff"


def _side_metric_columns(
    *,
    count_columns: tuple[str, ...],
    rate_columns: tuple[str, ...],
) -> tuple[str, ...]:
    columns: list[str] = []
    for source in count_columns:
        columns.extend(count_side_metric_columns(source))
    for source in rate_columns:
        columns.extend(rate_side_metric_columns(source))
    return tuple(columns)


SLICE7_SIDE_METRIC_COLUMNS: tuple[str, ...] = _side_metric_columns(
    count_columns=SLICE7_COUNT_SOURCE_COLUMNS,
    rate_columns=SLICE7_RATE_SOURCE_COLUMNS,
)

SLICE7_COMPARISON_COLUMNS: tuple[str, ...] = tuple(
    slice7_diff_column(metric) for metric in SLICE7_SIDE_METRIC_COLUMNS
)

SLICE7_SAME_VERSION_COUNT_DIFF_COLUMNS: tuple[str, ...] = tuple(
    slice7_diff_column(name)
    for name in count_side_metric_columns("player_hero_same_version_matches")
)
SLICE7_SAME_VERSION_RATE_DIFF_COLUMNS: tuple[str, ...] = tuple(
    slice7_diff_column(name)
    for name in rate_side_metric_columns("player_hero_same_version_win_rate")
)
SLICE7_RECENT20_COUNT_DIFF_COLUMNS: tuple[str, ...] = tuple(
    slice7_diff_column(name)
    for name in count_side_metric_columns("player_hero_recent_20_matches")
)
SLICE7_RECENT20_RATE_DIFF_COLUMNS: tuple[str, ...] = tuple(
    slice7_diff_column(name)
    for name in rate_side_metric_columns("player_hero_recent_20_win_rate")
)
SLICE7_ROLE_DIFF_COLUMNS: tuple[str, ...] = tuple(
    slice7_diff_column(name)
    for source in (
        "player_hero_recent_role_compatibility",
        "player_hero_share_at_expected_position",
        "hero_meta_share_at_expected_position",
    )
    for name in rate_side_metric_columns(source)
)


_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    SIDE_COLUMN,
    "team_id",
)


def _n_zero(series: pd.Series) -> int:
    """Count observed zeros; NULL is not treated as zero history."""
    return int((series == 0).sum())


def player_hero_meta_side_profile(
    players: pd.DataFrame,
    *,
    count_columns: tuple[str, ...] = SLICE7_COUNT_SOURCE_COLUMNS,
    rate_columns: tuple[str, ...] = SLICE7_RATE_SOURCE_COLUMNS,
) -> pd.DataFrame:
    """Aggregate player-grain Slice 6 rows to ``(match_id, side)``.

    Does not recompute history. Each side row uses only that match's
    already leakage-safe player rows.
    """
    required = [
        *_IDENTITY_COLUMNS,
        *count_columns,
        *rate_columns,
    ]
    missing = [column for column in required if column not in players.columns]
    if missing:
        raise ValueError(
            "player_hero_meta frame is missing required columns: "
            f"{missing}"
        )

    grouped = players.groupby(
        list(_IDENTITY_COLUMNS), dropna=False, sort=False
    )
    aggregations: dict[str, tuple[str, str] | tuple[str, object]] = {}
    for source in count_columns:
        mean_name, min_name, zero_name = count_side_metric_columns(source)
        aggregations[mean_name] = (source, "mean")
        aggregations[min_name] = (source, "min")
        aggregations[zero_name] = (source, _n_zero)
    for source in rate_columns:
        (mean_name,) = rate_side_metric_columns(source)
        aggregations[mean_name] = (source, "mean")

    side = grouped.agg(**aggregations).reset_index()
    metric_columns = _side_metric_columns(
        count_columns=count_columns, rate_columns=rate_columns
    )
    return side[list(_IDENTITY_COLUMNS) + list(metric_columns)]


def player_hero_meta_comparison_from_players(
    players: pd.DataFrame,
    *,
    count_columns: tuple[str, ...] = SLICE7_COUNT_SOURCE_COLUMNS,
    rate_columns: tuple[str, ...] = SLICE7_RATE_SOURCE_COLUMNS,
) -> pd.DataFrame:
    """One row per match: Radiant − Dire of the Slice 6 side profile.

    NULL on either side yields a NULL difference (same as
    ``draft_comparison_from_profile``). Count fields are not sign-flipped.
    """
    side = player_hero_meta_side_profile(
        players, count_columns=count_columns, rate_columns=rate_columns
    )
    metric_columns = _side_metric_columns(
        count_columns=count_columns, rate_columns=rate_columns
    )
    radiant = side.loc[side[SIDE_COLUMN] == _RADIANT_SIDE]
    dire = side.loc[side[SIDE_COLUMN] == _DIRE_SIDE]
    if radiant[MATCH_ID_COLUMN].duplicated().any():
        raise ValueError("expected exactly one Radiant row per match_id")
    if dire[MATCH_ID_COLUMN].duplicated().any():
        raise ValueError("expected exactly one Dire row per match_id")

    merged = radiant.merge(
        dire,
        on=MATCH_ID_COLUMN,
        how="inner",
        suffixes=("_radiant", "_dire"),
        validate="one_to_one",
    )
    frame = pd.DataFrame(
        {
            MATCH_ID_COLUMN: merged[MATCH_ID_COLUMN],
            "start_time": merged["start_time_radiant"],
            "game_version_id": merged["game_version_id_radiant"],
            "radiant_team_id": merged["team_id_radiant"],
            "dire_team_id": merged["team_id_dire"],
        }
    )
    for metric in metric_columns:
        frame[slice7_diff_column(metric)] = (
            merged[f"{metric}_radiant"] - merged[f"{metric}_dire"]
        )
    diff_columns = tuple(slice7_diff_column(metric) for metric in metric_columns)
    ordered = [
        MATCH_ID_COLUMN,
        "start_time",
        "game_version_id",
        "radiant_team_id",
        "dire_team_id",
        *diff_columns,
    ]
    return (
        frame[ordered]
        .sort_values(["start_time", MATCH_ID_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )
