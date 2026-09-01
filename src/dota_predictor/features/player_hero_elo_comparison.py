"""Side-level and Radiant − Dire Slice 10 Elo-adjusted Player × Hero.

Grain
-----
Player-level Slice 10 state is ``(match_id, player_id, hero_id)``. Match
prediction plumbing needs one row per ``match_id``. This module does not
invent a composite score. It averages the five drafted player×hero rows
per side, then subtracts Dire from Radiant.

Lifetime volume is evidence, not strength
-----------------------------------------
``mean`` / ``min`` prior games and the zero-prior count describe how much
history the side brought into the draft. They are **not** treated as
positive strength. Strength is the mean Elo-adjusted residual (raw and
shrunk). Cold-start players contribute a NULL raw residual (skipped by
the side mean) and a **zero** shrunk residual (no claimed overperformance).

This layer is not part of production ``FEATURE_COLUMNS``, PRE_DRAFT
snapshot SQL, or any Slice 7–9 specification. It never writes Parquet.
"""

from __future__ import annotations

import pandas as pd

from dota_predictor.features.draft_profile import SIDE_COLUMN
from dota_predictor.features.player_hero_elo import (
    MATCH_ID_COLUMN,
    PLAYER_HERO_ELO_COLUMNS,
)

__all__ = [
    "MATCH_ID_COLUMN",
    "PLAYER_HERO_ELO_COMPARISON_COLUMNS",
    "PLAYER_HERO_ELO_COMPARISON_IDENTITY_COLUMNS",
    "PLAYER_HERO_ELO_COMPARISON_METRIC_COLUMNS",
    "PLAYER_HERO_ELO_SIDE_COLUMNS",
    "PLAYER_HERO_ELO_SIDE_EVIDENCE_COLUMNS",
    "PLAYER_HERO_ELO_SIDE_IDENTITY_COLUMNS",
    "PLAYER_HERO_ELO_SIDE_METRIC_COLUMNS",
    "PLAYER_HERO_ELO_SIDE_STRENGTH_COLUMNS",
    "player_hero_elo_comparison_from_players",
    "player_hero_elo_comparison_from_side",
    "player_hero_elo_diff_column",
    "player_hero_elo_side_profile",
]


_RADIANT_SIDE = "RADIANT"
_DIRE_SIDE = "DIRE"

PLAYER_HERO_ELO_SIDE_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    SIDE_COLUMN,
    "team_id",
)

PLAYER_HERO_ELO_SIDE_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "mean_player_hero_prior_games",
    "min_player_hero_prior_games",
    "players_with_zero_player_hero_prior_games",
    "mean_player_hero_shrinkage_weight",
)

PLAYER_HERO_ELO_SIDE_STRENGTH_COLUMNS: tuple[str, ...] = (
    "mean_player_hero_prior_wins",
    "mean_player_hero_prior_elo_expected_wins",
    "mean_player_hero_wins_minus_expected",
    "mean_player_hero_outcome_residual",
    "mean_player_hero_shrunk_residual",
)

PLAYER_HERO_ELO_SIDE_METRIC_COLUMNS: tuple[str, ...] = (
    PLAYER_HERO_ELO_SIDE_EVIDENCE_COLUMNS + PLAYER_HERO_ELO_SIDE_STRENGTH_COLUMNS
)

PLAYER_HERO_ELO_SIDE_COLUMNS: tuple[str, ...] = (
    PLAYER_HERO_ELO_SIDE_IDENTITY_COLUMNS + PLAYER_HERO_ELO_SIDE_METRIC_COLUMNS
)

PLAYER_HERO_ELO_COMPARISON_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    "radiant_team_id",
    "dire_team_id",
)


def player_hero_elo_diff_column(side_metric: str) -> str:
    return f"{side_metric}_diff"


PLAYER_HERO_ELO_COMPARISON_METRIC_COLUMNS: tuple[str, ...] = tuple(
    player_hero_elo_diff_column(metric)
    for metric in PLAYER_HERO_ELO_SIDE_METRIC_COLUMNS
)

PLAYER_HERO_ELO_COMPARISON_COLUMNS: tuple[str, ...] = (
    PLAYER_HERO_ELO_COMPARISON_IDENTITY_COLUMNS
    + PLAYER_HERO_ELO_COMPARISON_METRIC_COLUMNS
)


def player_hero_elo_side_profile(players: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(match_id, side)`` from five player×hero Slice 10 rows.

    Count/evidence means include zeros. Mean raw residual skips NULL
    (no history). Mean shrunk residual includes cold-start zeros.
    """
    missing = [
        column
        for column in PLAYER_HERO_ELO_COLUMNS
        if column not in players.columns
    ]
    if missing:
        raise ValueError(f"player frame is missing required columns: {missing}")

    grouped = players.groupby(
        [MATCH_ID_COLUMN, "start_time", "game_version_id", SIDE_COLUMN, "team_id"],
        dropna=False,
        sort=False,
    )
    rows: list[dict[str, object]] = []
    for keys, subset in grouped:
        match_id, start_time, game_version_id, side, team_id = keys
        games = pd.to_numeric(subset["prior_games_on_hero"], errors="coerce")
        residual = pd.to_numeric(
            subset["mean_outcome_residual_on_hero"], errors="coerce"
        )
        rows.append(
            {
                MATCH_ID_COLUMN: match_id,
                "start_time": start_time,
                "game_version_id": game_version_id,
                SIDE_COLUMN: side,
                "team_id": team_id,
                "mean_player_hero_prior_games": float(games.mean()),
                "min_player_hero_prior_games": float(games.min()),
                "players_with_zero_player_hero_prior_games": int((games == 0).sum()),
                "mean_player_hero_shrinkage_weight": float(
                    pd.to_numeric(
                        subset["shrinkage_weight_on_hero"], errors="coerce"
                    ).mean()
                ),
                "mean_player_hero_prior_wins": float(
                    pd.to_numeric(
                        subset["prior_wins_on_hero"], errors="coerce"
                    ).mean()
                ),
                "mean_player_hero_prior_elo_expected_wins": float(
                    pd.to_numeric(
                        subset["prior_elo_expected_wins_on_hero"], errors="coerce"
                    ).mean()
                ),
                "mean_player_hero_wins_minus_expected": float(
                    pd.to_numeric(
                        subset["prior_wins_minus_expected_on_hero"], errors="coerce"
                    ).mean()
                ),
                "mean_player_hero_outcome_residual": (
                    float(residual.mean()) if residual.notna().any() else float("nan")
                ),
                "mean_player_hero_shrunk_residual": float(
                    pd.to_numeric(
                        subset["shrunk_outcome_residual_on_hero"], errors="coerce"
                    ).mean()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(PLAYER_HERO_ELO_SIDE_COLUMNS))
    side_order = frame[SIDE_COLUMN].map({_RADIANT_SIDE: 0, _DIRE_SIDE: 1})
    return (
        frame[list(PLAYER_HERO_ELO_SIDE_COLUMNS)]
        .assign(_side_order=side_order)
        .sort_values([MATCH_ID_COLUMN, "_side_order"], kind="mergesort")
        .drop(columns="_side_order")
        .reset_index(drop=True)
    )


def player_hero_elo_comparison_from_side(profile: pd.DataFrame) -> pd.DataFrame:
    """Radiant − Dire from an already-aggregated side profile."""
    missing = [
        column
        for column in PLAYER_HERO_ELO_SIDE_COLUMNS
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
    for metric in PLAYER_HERO_ELO_SIDE_METRIC_COLUMNS:
        frame[player_hero_elo_diff_column(metric)] = (
            merged[f"{metric}_radiant"] - merged[f"{metric}_dire"]
        )
    return (
        frame[list(PLAYER_HERO_ELO_COMPARISON_COLUMNS)]
        .sort_values(["start_time", MATCH_ID_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )


def player_hero_elo_comparison_from_players(players: pd.DataFrame) -> pd.DataFrame:
    """Match-level Radiant − Dire Slice 10 comparison from player rows."""
    return player_hero_elo_comparison_from_side(player_hero_elo_side_profile(players))
