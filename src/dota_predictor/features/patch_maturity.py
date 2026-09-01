"""Descriptive patch-age and professional-match maturity (not a feature).

Grain
-----
One row per `match_id`. Calendar patch age is days between STRATZ
``game_versions.as_of_datetime`` and the match ``start_time``.
Professional-match maturity is the count of dataset matches with the
same ``game_version_id`` and strictly earlier ``start_time``.

These are diagnostic context for robustness analysis. They are not
training features, are not part of ``FEATURE_COLUMNS`` / PRE_DRAFT
snapshot SQL, and must not be joined into a model matrix.

Temporal integrity
------------------
``prior_matches_in_game_version`` uses the same
``RANGE ... CURRENT ROW EXCLUDE GROUP`` rule as the other historical
layers: equal ``start_time`` peers are mutually blind, and later
matches never contribute. ``days_since_game_version_start`` is a
reference-catalog offset, not a count of future matches.

``as_of_datetime`` is not assumed to be the first professional match.
Negative calendar age is preserved and flagged; source metadata is
never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isnan

import duckdb
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    GAME_VERSIONS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)

__all__ = [
    "CALENDAR_AGE_BIN_ORDER",
    "MISSING_CALENDAR_AGE_BIN",
    "NEGATIVE_CALENDAR_AGE_BIN",
    "PATCH_MATURITY_COLUMNS",
    "PATCH_MATURITY_IDENTITY_COLUMNS",
    "PATCH_MATURITY_METRIC_COLUMNS",
    "PRIOR_MATCH_BIN_ORDER",
    "PatchMaturity",
    "assign_calendar_age_bin",
    "assign_prior_match_bin",
    "build_patch_maturity",
    "patch_age_sanity_table",
    "patch_maturity_sql",
]

MATCH_ID_COLUMN = "match_id"

PATCH_MATURITY_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    "game_version_name",
    "as_of_datetime",
)

PATCH_MATURITY_METRIC_COLUMNS: tuple[str, ...] = (
    "days_since_game_version_start",
    "prior_matches_in_game_version",
)

PATCH_MATURITY_COLUMNS: tuple[str, ...] = (
    PATCH_MATURITY_IDENTITY_COLUMNS + PATCH_MATURITY_METRIC_COLUMNS
)

NEGATIVE_CALENDAR_AGE_BIN = "negative (as_of after match)"
MISSING_CALENDAR_AGE_BIN = "missing as_of"

CALENDAR_AGE_BIN_ORDER: tuple[str, ...] = (
    NEGATIVE_CALENDAR_AGE_BIN,
    "0–7 days",
    "8–21 days",
    "22–45 days",
    "46+ days",
    MISSING_CALENDAR_AGE_BIN,
)

PRIOR_MATCH_BIN_ORDER: tuple[str, ...] = (
    "0–49 prior matches",
    "50–199",
    "200–499",
    "500+",
)

_STRICT_PRIOR_RANGE = (
    "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP"
)


def assign_calendar_age_bin(days: float | None) -> str:
    """Map calendar patch age to a predefined bin.

    Bins are on ``floor(days)`` and are not chosen from model metrics.
    Negative and missing ages are separate labels, never clipped into
    ``0–7 days``.
    """
    if days is None:
        return MISSING_CALENDAR_AGE_BIN
    value = float(days)
    if isnan(value):
        return MISSING_CALENDAR_AGE_BIN
    if value < 0.0:
        return NEGATIVE_CALENDAR_AGE_BIN
    whole_days = floor(value)
    if whole_days <= 7:
        return "0–7 days"
    if whole_days <= 21:
        return "8–21 days"
    if whole_days <= 45:
        return "22–45 days"
    return "46+ days"


def assign_prior_match_bin(prior_matches: int) -> str:
    """Map professional-match maturity to a predefined count bin."""
    count = int(prior_matches)
    if count < 0:
        raise ValueError(
            "prior_matches_in_game_version cannot be negative, "
            f"got {count}"
        )
    if count <= 49:
        return "0–49 prior matches"
    if count <= 199:
        return "50–199"
    if count <= 499:
        return "200–499"
    return "500+"


def patch_maturity_sql() -> str:
    """SQL for leakage-safe calendar age and prior-match maturity.

    Requires ``matches`` and ``game_versions`` views. Left-joins the
    catalog so an unmatched ``game_version_id`` yields NULL calendar
    age rather than dropping the match. Prior-match counts only use
    ``matches`` and do not depend on ``as_of_datetime``.
    """
    return f"""
WITH matches_ordered AS (
    SELECT
        m.match_id,
        m.start_time,
        m.game_version_id,
        COUNT(*) OVER (
            PARTITION BY m.game_version_id
            ORDER BY m.start_time
            {_STRICT_PRIOR_RANGE}
        )::BIGINT AS prior_matches_in_game_version
    FROM {MATCHES_VIEW} AS m
)

SELECT
    o.match_id,
    o.start_time,
    o.game_version_id,
    gv.name AS game_version_name,
    gv.as_of_datetime,
    CASE
        WHEN gv.as_of_datetime IS NULL THEN NULL
        ELSE (
            EXTRACT(EPOCH FROM o.start_time)
            - EXTRACT(EPOCH FROM gv.as_of_datetime)
        ) / 86400.0
    END AS days_since_game_version_start,
    o.prior_matches_in_game_version
FROM matches_ordered AS o
LEFT JOIN {GAME_VERSIONS_VIEW} AS gv
    ON gv.game_version_id = o.game_version_id
"""


def _game_versions_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return GAME_VERSIONS_VIEW in tables


@dataclass(frozen=True)
class PatchMaturity:
    """Lazy one-row-per-match patch-age relation."""

    relation: duckdb.DuckDBPyRelation

    def to_frame(self) -> pd.DataFrame:
        """Materialize one row per match in ``PATCH_MATURITY_COLUMNS`` order."""
        frame = self.relation.df()
        ordered = frame[list(PATCH_MATURITY_COLUMNS)].copy()
        return (
            ordered.sort_values(
                ["start_time", MATCH_ID_COLUMN], kind="mergesort"
            )
            .reset_index(drop=True)
        )


def build_patch_maturity(store: FeatureDuckDBConnection) -> PatchMaturity:
    """Build descriptive patch-age state from registered views.

    Requires ``game_versions``. Does not write Parquet and does not
    alter PRE_DRAFT / training contracts.
    """
    if not _game_versions_registered(store):
        raise ValueError(
            "build_patch_maturity requires the game_versions reference "
            "view; call register_reference_views first"
        )
    return PatchMaturity(relation=store.sql(patch_maturity_sql()))


def patch_age_sanity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-version coverage of calendar age and professional maturity.

    Flags negative calendar ages and unmatched catalog rows. Does not
    rewrite ``as_of_datetime`` or clip negative days.
    """
    working = frame.copy()
    grouped = working.groupby("game_version_id", dropna=False)
    rows: list[dict[str, object]] = []
    for version_id, subset in grouped:
        days = subset["days_since_game_version_start"]
        n_negative = int((days < 0).sum())
        n_missing = int(days.isna().sum())
        name = subset["game_version_name"].dropna()
        as_of = subset["as_of_datetime"].dropna()
        rows.append(
            {
                "game_version_id": version_id,
                "name": name.iloc[0] if len(name) else pd.NA,
                "as_of_datetime": as_of.iloc[0] if len(as_of) else pd.NaT,
                "match_count": len(subset),
                "min_days_since_game_version_start": (
                    float(days.min()) if days.notna().any() else float("nan")
                ),
                "median_days_since_game_version_start": (
                    float(days.median()) if days.notna().any() else float("nan")
                ),
                "max_days_since_game_version_start": (
                    float(days.max()) if days.notna().any() else float("nan")
                ),
                "max_prior_matches_in_game_version": int(
                    subset["prior_matches_in_game_version"].max()
                ),
                "n_negative_calendar_age": n_negative,
                "n_missing_as_of": n_missing,
                "flagged": bool(n_negative > 0 or n_missing > 0),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("game_version_id", kind="stable")
        .reset_index(drop=True)
    )
