"""Match-level Radiant − Dire draft-profile comparison (not a feature yet).

Grain
-----
One row per `match_id`. Input is the existing side-level Draft Profile,
which has exactly `(match_id, RADIANT)` and `(match_id, DIRE)` for every
match. This layer does not recompute Hero Meta, Player × Hero, or
Team × Hero history. It only joins those two already leakage-safe side
rows and subtracts.

Every difference is:

    diff = Radiant - Dire

Radiant higher → positive; Dire higher → negative. Count fields where a
higher value is undesirable (`players_with_zero_prior_games_on_hero`,
`heroes_never_played_by_team`, …) still use raw Radiant − Dire. The
sign is mathematical direction, not a judgment that positive means
"better".

NULL semantics
--------------
If either side's underlying value is NULL, the difference is NULL.
No zero-fill, imputation, smoothing, or missingness indicators.

This module never writes Parquet, never bumps schema versions, never
adds columns to the fact/reference files, and is not part of the
training feature matrix. `radiant_win` is not a comparison column;
outcome may be joined separately for descriptive evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from dota_predictor.features.draft_profile import (
    DRAFT_PROFILE_COLUMNS,
    DRAFT_PROFILE_METRIC_COLUMNS,
    MATCH_ID_COLUMN,
    SIDE_COLUMN,
    draft_profile_sql,
)
from dota_predictor.features.duckdb_layer import (
    HEROES_VIEW,
    FeatureDuckDBConnection,
)

__all__ = [
    "DRAFT_COMPARISON_COLUMNS",
    "DRAFT_COMPARISON_IDENTITY_COLUMNS",
    "DRAFT_COMPARISON_METRIC_COLUMNS",
    "MATCH_ID_COLUMN",
    "DraftComparison",
    "build_draft_comparison",
    "draft_comparison_from_profile",
    "draft_comparison_sql",
]


def _diff_column(metric: str) -> str:
    return f"{metric}_diff"


DRAFT_COMPARISON_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    "radiant_team_id",
    "dire_team_id",
)

DRAFT_COMPARISON_METRIC_COLUMNS: tuple[str, ...] = tuple(
    _diff_column(metric) for metric in DRAFT_PROFILE_METRIC_COLUMNS
)

DRAFT_COMPARISON_COLUMNS: tuple[str, ...] = (
    DRAFT_COMPARISON_IDENTITY_COLUMNS + DRAFT_COMPARISON_METRIC_COLUMNS
)

_RADIANT_SIDE = "RADIANT"
_DIRE_SIDE = "DIRE"


def _diff_select_sql() -> str:
    lines = [
        (
            f"(radiant.{metric} - dire.{metric}) "
            f"AS {_diff_column(metric)}"
        )
        for metric in DRAFT_PROFILE_METRIC_COLUMNS
    ]
    return ",\n    ".join(lines)


def draft_comparison_sql(
    *, catalog_registered: bool = True, match_id: int | None = None
) -> str:
    """SQL for a leakage-safe one-row-per-match Radiant − Dire comparison.

    Wraps `draft_profile_sql` in a `MATERIALIZED` CTE so the three
    historical layers run once, then inner-joins the Radiant row to the
    Dire row. Optional `match_id` is forwarded to the inner profile
    (output filter only; inner windows still see full history).
    """
    profile_sql = draft_profile_sql(
        catalog_registered=catalog_registered, match_id=match_id
    )
    return f"""
WITH profile AS MATERIALIZED (
    SELECT * FROM ({profile_sql}) AS profile_inner
)

SELECT
    radiant.match_id,
    radiant.start_time,
    radiant.game_version_id,
    radiant.team_id AS radiant_team_id,
    dire.team_id AS dire_team_id,
    {_diff_select_sql()}
FROM profile AS radiant
INNER JOIN profile AS dire
    ON radiant.match_id = dire.match_id
    AND radiant.side = '{_RADIANT_SIDE}'
    AND dire.side = '{_DIRE_SIDE}'
"""


def draft_comparison_from_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Radiant − Dire from an already-materialized side-level profile.

    Does not recompute history. SQL `NULL` arithmetic is mirrored:
    NULL on either side yields a NULL difference. Count fields are not
    sign-flipped.
    """
    missing = [
        column
        for column in DRAFT_PROFILE_COLUMNS
        if column not in profile.columns
    ]
    if missing:
        raise ValueError(
            f"profile is missing required columns: {missing}"
        )

    working = profile[list(DRAFT_PROFILE_COLUMNS)].copy()
    radiant = working.loc[working[SIDE_COLUMN] == _RADIANT_SIDE]
    dire = working.loc[working[SIDE_COLUMN] == _DIRE_SIDE]
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
    for metric in DRAFT_PROFILE_METRIC_COLUMNS:
        frame[_diff_column(metric)] = (
            merged[f"{metric}_radiant"] - merged[f"{metric}_dire"]
        )
    return (
        frame[list(DRAFT_COMPARISON_COLUMNS)]
        .sort_values(["start_time", MATCH_ID_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )


def _heroes_view_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return HEROES_VIEW in tables


@dataclass(frozen=True)
class DraftComparison:
    """Lazy one-row-per-match Radiant − Dire relation.

    Nothing is materialized until `to_frame` is called. The owning
    `FeatureDuckDBConnection` must stay open for that call.
    """

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per match in `DRAFT_COMPARISON_COLUMNS` order."""
        frame = self.relation.df()
        ordered = frame[list(DRAFT_COMPARISON_COLUMNS)].copy()
        return (
            ordered.sort_values(
                ["start_time", MATCH_ID_COLUMN], kind="mergesort"
            )
            .reset_index(drop=True)
        )


def build_draft_comparison(
    store: FeatureDuckDBConnection, *, match_id: int | None = None
) -> DraftComparison:
    """Build a descriptive Radiant − Dire comparison from registered views.

    Independent of `build_pre_draft_snapshot`. Reuses `draft_profile_sql`
    as-is (which itself reuses Player × Hero, Team × Hero, and Hero Meta).
    Optional `match_id` filters aggregated output after inner windows run
    over full history.
    """
    sql = draft_comparison_sql(
        catalog_registered=_heroes_view_registered(store),
        match_id=match_id,
    )
    return DraftComparison(relation=store.sql(sql))
