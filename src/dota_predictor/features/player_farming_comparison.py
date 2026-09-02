"""Side-level and Radiant − Dire Slice 15 player farming comparison.

Grain
-----
Slice 14 player state is ``(match_id, player_id)``. Match prediction
needs one row per ``match_id``. This module does not invent a composite
score. It averages the five rostered players per side, then subtracts
Dire from Radiant.

PRE_DRAFT
---------
Farming state is keyed by ``player_id`` only. The current drafted
``hero_id`` is not a lookup key. Roster identity is knowable before the
first draft action, so side means and Radiant − Dire diffs are
PRE_DRAFT historical state. Current-match last hits, duration, position,
hero, and result never enter these columns.

The comparison reads already-computed Slice 14 *prior* state:

* ``farming_shrunk_b`` — strength (cold-start is exactly 0)
* ``farming_prior_mean_b`` — unshrunk prior mean (NULL when ``n = 0``)
* ``farming_prior_n`` / ``farming_shrinkage_weight`` — evidence, not
  strength

``farming_causal_b`` is a POST_MATCH observation of the *current*
appearance and is not a comparison input.

Lifetime volume is evidence, not strength
-----------------------------------------
Mean / min prior ``n`` and the zero-prior count describe how much
farming history the side brought in. They are **not** treated as
positive strength. Strength is the mean shrunk causal-B state. The
named candidate feature is ``mean_farming_shrunk_b_diff``.

This layer is not part of production ``FEATURE_COLUMNS`` or PRE_DRAFT
snapshot SQL. It never writes Parquet.
"""

from __future__ import annotations

import pandas as pd

from dota_predictor.features.draft_profile import SIDE_COLUMN

__all__ = [
    "FARMING_CAUSAL_B_COLUMN",
    "MATCH_ID_COLUMN",
    "PLAYER_FARMING_COMPARISON_COLUMNS",
    "PLAYER_FARMING_COMPARISON_IDENTITY_COLUMNS",
    "PLAYER_FARMING_COMPARISON_METRIC_COLUMNS",
    "PLAYER_FARMING_FEATURE_COLUMNS",
    "PLAYER_FARMING_REQUIRED_COLUMNS",
    "PLAYER_FARMING_SIDE_COLUMNS",
    "PLAYER_FARMING_SIDE_EVIDENCE_COLUMNS",
    "PLAYER_FARMING_SIDE_IDENTITY_COLUMNS",
    "PLAYER_FARMING_SIDE_METRIC_COLUMNS",
    "PLAYER_FARMING_SIDE_STRENGTH_COLUMNS",
    "PLAYER_FARMING_STATE_FEATURE_COLUMNS",
    "PLAYER_FARMING_STATE_IDENTITY_COLUMNS",
    "merge_player_farming_comparison",
    "player_farming_comparison_from_players",
    "player_farming_comparison_from_side",
    "player_farming_diff_column",
    "player_farming_side_profile",
]


MATCH_ID_COLUMN = "match_id"
FARMING_CAUSAL_B_COLUMN = "farming_causal_b"
_RADIANT_SIDE = "RADIANT"
_DIRE_SIDE = "DIRE"

PLAYER_FARMING_STATE_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "player_id",
    "start_time",
    "game_version_id",
    "team_id",
    SIDE_COLUMN,
    "slot_in_side",
)

# Historical prior state knowable at PRE_DRAFT. ``farming_causal_b`` is
# deliberately absent: it uses the current appearance's last hits.
PLAYER_FARMING_STATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "farming_residualizer_n",
    "farming_prior_n",
    "farming_prior_sum_b",
    "farming_prior_mean_b",
    "farming_shrinkage_weight",
    "farming_shrunk_b",
)

PLAYER_FARMING_SIDE_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    SIDE_COLUMN,
    "team_id",
)

PLAYER_FARMING_SIDE_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "mean_farming_prior_n",
    "min_farming_prior_n",
    "players_with_zero_farming_prior_n",
    "mean_farming_shrinkage_weight",
)

PLAYER_FARMING_SIDE_STRENGTH_COLUMNS: tuple[str, ...] = (
    "mean_farming_prior_mean_b",
    "mean_farming_shrunk_b",
)

PLAYER_FARMING_SIDE_METRIC_COLUMNS: tuple[str, ...] = (
    PLAYER_FARMING_SIDE_EVIDENCE_COLUMNS + PLAYER_FARMING_SIDE_STRENGTH_COLUMNS
)

PLAYER_FARMING_SIDE_COLUMNS: tuple[str, ...] = (
    PLAYER_FARMING_SIDE_IDENTITY_COLUMNS + PLAYER_FARMING_SIDE_METRIC_COLUMNS
)

PLAYER_FARMING_REQUIRED_COLUMNS: tuple[str, ...] = (
    *PLAYER_FARMING_SIDE_IDENTITY_COLUMNS,
    "farming_prior_n",
    "farming_prior_mean_b",
    "farming_shrinkage_weight",
    "farming_shrunk_b",
)

PLAYER_FARMING_COMPARISON_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    "radiant_team_id",
    "dire_team_id",
)


def player_farming_diff_column(side_metric: str) -> str:
    return f"{side_metric}_diff"


PLAYER_FARMING_COMPARISON_METRIC_COLUMNS: tuple[str, ...] = tuple(
    player_farming_diff_column(metric) for metric in PLAYER_FARMING_SIDE_METRIC_COLUMNS
)

PLAYER_FARMING_COMPARISON_COLUMNS: tuple[str, ...] = (
    PLAYER_FARMING_COMPARISON_IDENTITY_COLUMNS
    + PLAYER_FARMING_COMPARISON_METRIC_COLUMNS
)

# The single candidate feature: Radiant − Dire of mean shrunk farming B.
PLAYER_FARMING_FEATURE_COLUMNS: tuple[str, ...] = ("mean_farming_shrunk_b_diff",)


def player_farming_side_profile(players: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(match_id, side)`` from five rostered player rows.

    Count/evidence means include zeros. Mean raw prior B skips NULL
    (no history). Mean shrunk B includes cold-start zeros.
    """
    missing = [
        column
        for column in PLAYER_FARMING_REQUIRED_COLUMNS
        if column not in players.columns
    ]
    if missing:
        raise ValueError(f"player frame is missing required columns: {missing}")

    grouped = players.groupby(
        list(PLAYER_FARMING_SIDE_IDENTITY_COLUMNS),
        dropna=False,
        sort=False,
    )
    rows: list[dict[str, object]] = []
    for keys, subset in grouped:
        match_id, start_time, game_version_id, side, team_id = keys
        prior_n = pd.to_numeric(subset["farming_prior_n"], errors="coerce")
        raw = pd.to_numeric(subset["farming_prior_mean_b"], errors="coerce")
        rows.append(
            {
                MATCH_ID_COLUMN: match_id,
                "start_time": start_time,
                "game_version_id": game_version_id,
                SIDE_COLUMN: side,
                "team_id": team_id,
                "mean_farming_prior_n": float(prior_n.mean()),
                "min_farming_prior_n": float(prior_n.min()),
                "players_with_zero_farming_prior_n": int((prior_n == 0).sum()),
                "mean_farming_shrinkage_weight": float(
                    pd.to_numeric(
                        subset["farming_shrinkage_weight"], errors="coerce"
                    ).mean()
                ),
                "mean_farming_prior_mean_b": (
                    float(raw.mean()) if raw.notna().any() else float("nan")
                ),
                "mean_farming_shrunk_b": float(
                    pd.to_numeric(subset["farming_shrunk_b"], errors="coerce").mean()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(PLAYER_FARMING_SIDE_COLUMNS))
    side_order = frame[SIDE_COLUMN].map({_RADIANT_SIDE: 0, _DIRE_SIDE: 1})
    return (
        frame[list(PLAYER_FARMING_SIDE_COLUMNS)]
        .assign(_side_order=side_order)
        .sort_values([MATCH_ID_COLUMN, "_side_order"], kind="mergesort")
        .drop(columns="_side_order")
        .reset_index(drop=True)
    )


def player_farming_comparison_from_side(profile: pd.DataFrame) -> pd.DataFrame:
    """Radiant − Dire from an already-aggregated side profile."""
    missing = [
        column
        for column in PLAYER_FARMING_SIDE_COLUMNS
        if column not in profile.columns
    ]
    if missing:
        raise ValueError(f"side profile is missing required columns: {missing}")

    radiant = profile.loc[profile[SIDE_COLUMN] == _RADIANT_SIDE]
    dire = profile.loc[profile[SIDE_COLUMN] == _DIRE_SIDE]
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
    for metric in PLAYER_FARMING_SIDE_METRIC_COLUMNS:
        frame[player_farming_diff_column(metric)] = (
            merged[f"{metric}_radiant"] - merged[f"{metric}_dire"]
        )
    return (
        frame[list(PLAYER_FARMING_COMPARISON_COLUMNS)]
        .sort_values(["start_time", MATCH_ID_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )


def player_farming_comparison_from_players(players: pd.DataFrame) -> pd.DataFrame:
    """Match-level Radiant − Dire farming comparison from player rows."""
    return player_farming_comparison_from_side(player_farming_side_profile(players))


def merge_player_farming_comparison(
    matches: pd.DataFrame, comparison: pd.DataFrame
) -> pd.DataFrame:
    """Left-join match-level farming diffs onto an existing match frame."""
    missing = [
        column
        for column in (MATCH_ID_COLUMN, *PLAYER_FARMING_COMPARISON_METRIC_COLUMNS)
        if column not in comparison.columns
    ]
    if missing:
        raise ValueError(f"comparison is missing required columns: {missing}")
    if MATCH_ID_COLUMN not in matches.columns:
        raise ValueError("match frame is missing match_id")
    extra = comparison.loc[
        :, [MATCH_ID_COLUMN, *PLAYER_FARMING_COMPARISON_METRIC_COLUMNS]
    ]
    return matches.merge(extra, on=MATCH_ID_COLUMN, how="left", validate="one_to_one")
