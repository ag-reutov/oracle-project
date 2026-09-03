"""Side-level and Radiant − Dire Slice 19 player combat comparison.

Grain
-----
Slice 18 player state is ``(match_id, player_id)``. Match prediction
needs one row per ``match_id``. This module does not invent a composite
score. It averages the five rostered players per side, then subtracts
Dire from Radiant.

PRE_DRAFT
---------
Combat state is keyed by ``player_id`` only. The current drafted
``hero_id`` and current ``position`` are not lookup keys. Roster
identity is knowable before the first draft action, so side means and
the Radiant − Dire diff are PRE_DRAFT historical state. Current-match
hero damage, kills, assists, deaths, duration, last hits, networth,
hero, position, and result never enter these columns.

The comparison reads already-computed Slice 18 *prior* state:

* ``combat_shrunk_c`` — strength (cold-start is exactly 0)
* ``combat_prior_n`` — evidence, not strength

``combat_causal_c`` is a POST_MATCH observation of the *current*
appearance and is not a comparison input.

Lifetime volume is evidence, not strength
-----------------------------------------
Prior ``n`` sums and cold-start counts describe how much combat history
the side brought in. They are **not** treated as positive strength.
Strength is the arithmetic mean of shrunk causal-C state. The named
candidate feature is ``mean_combat_shrunk_c_diff``.

Roster integrity
----------------
Canonical matches already require exactly five distinct Radiant player
ids and five distinct Dire player ids, with no cross-side overlap.
This module asserts those invariants rather than repairing incomplete
sides. It never averages fewer than five players.

This layer is not part of production ``FEATURE_COLUMNS`` or PRE_DRAFT
snapshot SQL. It never writes Parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dota_predictor.features.draft_profile import SIDE_COLUMN

__all__ = [
    "COMBAT_CAUSAL_C_COLUMN",
    "COMBAT_ROSTER_SIDE_SIZE",
    "MATCH_ID_COLUMN",
    "PLAYER_COMBAT_COMPARISON_COLUMNS",
    "PLAYER_COMBAT_COMPARISON_EVIDENCE_COLUMNS",
    "PLAYER_COMBAT_COMPARISON_IDENTITY_COLUMNS",
    "PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS",
    "PLAYER_COMBAT_FEATURE_COLUMNS",
    "PLAYER_COMBAT_REQUIRED_COLUMNS",
    "PLAYER_COMBAT_SIDE_COLUMNS",
    "PLAYER_COMBAT_SIDE_EVIDENCE_COLUMNS",
    "PLAYER_COMBAT_SIDE_IDENTITY_COLUMNS",
    "PLAYER_COMBAT_SIDE_METRIC_COLUMNS",
    "PLAYER_COMBAT_SIDE_STRENGTH_COLUMNS",
    "PLAYER_COMBAT_STATE_FEATURE_COLUMNS",
    "PLAYER_COMBAT_STATE_IDENTITY_COLUMNS",
    "CombatRosterError",
    "assert_combat_roster",
    "diagnose_combat_roster",
    "match_combat_roster_flags",
    "merge_player_combat_comparison",
    "player_combat_comparison_from_players",
    "player_combat_comparison_from_side",
    "player_combat_side_profile",
]


MATCH_ID_COLUMN = "match_id"
COMBAT_CAUSAL_C_COLUMN = "combat_causal_c"
COMBAT_ROSTER_SIDE_SIZE = 5
_RADIANT_SIDE = "RADIANT"
_DIRE_SIDE = "DIRE"

PLAYER_COMBAT_STATE_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "player_id",
    "start_time",
    "game_version_id",
    "team_id",
    SIDE_COLUMN,
    "slot_in_side",
)

# Historical prior state knowable at PRE_DRAFT. ``combat_causal_c`` is
# deliberately absent: it uses the current appearance's hero damage.
PLAYER_COMBAT_STATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "combat_prior_n",
    "combat_prior_sum_c",
    "combat_prior_mean_c",
    "combat_shrinkage_weight",
    "combat_shrunk_c",
)

PLAYER_COMBAT_SIDE_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    SIDE_COLUMN,
    "team_id",
)

PLAYER_COMBAT_SIDE_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "combat_prior_n_sum",
    "combat_cold_start_count",
)

PLAYER_COMBAT_SIDE_STRENGTH_COLUMNS: tuple[str, ...] = ("mean_combat_shrunk_c",)

PLAYER_COMBAT_SIDE_METRIC_COLUMNS: tuple[str, ...] = (
    PLAYER_COMBAT_SIDE_STRENGTH_COLUMNS + PLAYER_COMBAT_SIDE_EVIDENCE_COLUMNS
)

PLAYER_COMBAT_SIDE_COLUMNS: tuple[str, ...] = (
    PLAYER_COMBAT_SIDE_IDENTITY_COLUMNS + PLAYER_COMBAT_SIDE_METRIC_COLUMNS
)

PLAYER_COMBAT_REQUIRED_COLUMNS: tuple[str, ...] = (
    *PLAYER_COMBAT_SIDE_IDENTITY_COLUMNS,
    "player_id",
    "combat_prior_n",
    "combat_shrunk_c",
)

PLAYER_COMBAT_COMPARISON_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    "radiant_team_id",
    "dire_team_id",
)

PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS: tuple[str, ...] = (
    "radiant_mean_combat_shrunk_c",
    "dire_mean_combat_shrunk_c",
    "mean_combat_shrunk_c_diff",
)

PLAYER_COMBAT_COMPARISON_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "radiant_combat_prior_n_sum",
    "dire_combat_prior_n_sum",
    "radiant_combat_cold_start_count",
    "dire_combat_cold_start_count",
)

PLAYER_COMBAT_COMPARISON_COLUMNS: tuple[str, ...] = (
    PLAYER_COMBAT_COMPARISON_IDENTITY_COLUMNS
    + PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS
    + PLAYER_COMBAT_COMPARISON_EVIDENCE_COLUMNS
)

# The single candidate feature: Radiant − Dire of mean shrunk combat C.
PLAYER_COMBAT_FEATURE_COLUMNS: tuple[str, ...] = ("mean_combat_shrunk_c_diff",)


class CombatRosterError(ValueError):
    """A match does not have a complete five-player combat roster."""


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, label: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def match_combat_roster_flags(players: pd.DataFrame) -> pd.DataFrame:
    """One row per ``match_id`` with roster and state-join integrity flags."""
    _require_columns(
        players,
        (MATCH_ID_COLUMN, SIDE_COLUMN, "player_id"),
        label="player frame",
    )
    if players.empty:
        return pd.DataFrame(
            columns=[
                MATCH_ID_COLUMN,
                "n_rows",
                "n_radiant",
                "n_dire",
                "five_per_side",
                "unique_within_side",
                "no_cross_side_duplicate",
                "complete_roster",
                "complete_state",
                "complete_state_join",
                "radiant_cold_start_count",
                "dire_cold_start_count",
                "any_cold_start",
            ]
        )

    shrunk = (
        pd.to_numeric(players["combat_shrunk_c"], errors="coerce")
        if "combat_shrunk_c" in players.columns
        else pd.Series(np.nan, index=players.index, dtype=float)
    )
    prior_n = (
        pd.to_numeric(players["combat_prior_n"], errors="coerce")
        if "combat_prior_n" in players.columns
        else pd.Series(np.nan, index=players.index, dtype=float)
    )
    work = players.assign(
        _missing_id=players["player_id"].isna(),
        _missing_state=~np.isfinite(shrunk.to_numpy(dtype=float)),
        _cold_start=prior_n.fillna(-1.0) == 0.0,
    )
    rows: list[dict[str, object]] = []
    for match_id, subset in work.groupby(MATCH_ID_COLUMN, sort=False):
        radiant = subset.loc[subset[SIDE_COLUMN] == _RADIANT_SIDE]
        dire = subset.loc[subset[SIDE_COLUMN] == _DIRE_SIDE]
        other = int((~subset[SIDE_COLUMN].isin((_RADIANT_SIDE, _DIRE_SIDE))).sum())
        r_ids = radiant["player_id"]
        d_ids = dire["player_id"]
        r_unique = int(r_ids.nunique(dropna=True))
        d_unique = int(d_ids.nunique(dropna=True))
        r_missing = bool(radiant["_missing_id"].any())
        d_missing = bool(dire["_missing_id"].any())
        five_per_side = (
            len(radiant) == COMBAT_ROSTER_SIDE_SIZE
            and len(dire) == COMBAT_ROSTER_SIDE_SIZE
            and other == 0
            and len(subset) == 2 * COMBAT_ROSTER_SIDE_SIZE
        )
        unique_within_side = (
            r_unique == len(radiant)
            and d_unique == len(dire)
            and not r_missing
            and not d_missing
        )
        overlap = bool(set(r_ids.dropna()) & set(d_ids.dropna()))
        complete_roster = (
            five_per_side
            and unique_within_side
            and r_unique == COMBAT_ROSTER_SIDE_SIZE
            and d_unique == COMBAT_ROSTER_SIDE_SIZE
            and not overlap
        )
        complete_state = not bool(subset["_missing_state"].any())
        r_cold = int(radiant["_cold_start"].sum())
        d_cold = int(dire["_cold_start"].sum())
        rows.append(
            {
                MATCH_ID_COLUMN: match_id,
                "n_rows": len(subset),
                "n_radiant": len(radiant),
                "n_dire": len(dire),
                "five_per_side": five_per_side,
                "unique_within_side": unique_within_side,
                "no_cross_side_duplicate": not overlap,
                "complete_roster": complete_roster,
                "complete_state": complete_state,
                "complete_state_join": complete_roster and complete_state,
                "radiant_cold_start_count": r_cold,
                "dire_cold_start_count": d_cold,
                "any_cold_start": (r_cold + d_cold) > 0,
            }
        )
    return pd.DataFrame(rows)


def diagnose_combat_roster(players: pd.DataFrame) -> dict[str, int]:
    """Match-level roster integrity counts. Does not repair rows."""
    flags = match_combat_roster_flags(players)
    if flags.empty:
        return {
            "n_matches": 0,
            "n_complete_10_player_roster": 0,
            "n_complete_combat_state_join": 0,
            "n_wrong_side_size_matches": 0,
            "n_duplicate_within_side_matches": 0,
            "n_both_sides_matches": 0,
            "n_missing_state_matches": 0,
            "n_missing_player_id_rows": int(players["player_id"].isna().sum())
            if "player_id" in players.columns
            else 0,
        }
    return {
        "n_matches": len(flags),
        "n_complete_10_player_roster": int(flags["complete_roster"].sum()),
        "n_complete_combat_state_join": int(flags["complete_state_join"].sum()),
        "n_wrong_side_size_matches": int((~flags["five_per_side"]).sum()),
        "n_duplicate_within_side_matches": int((~flags["unique_within_side"]).sum()),
        "n_both_sides_matches": int((~flags["no_cross_side_duplicate"]).sum()),
        "n_missing_state_matches": int((~flags["complete_state"]).sum()),
        "n_missing_player_id_rows": int(players["player_id"].isna().sum()),
    }


def assert_combat_roster(players: pd.DataFrame) -> None:
    """Require a complete 5+5 roster with a finite combat state per player.

    Canonical match rosters already guarantee five distinct ids per
    side and no cross-side duplicates. Fail loudly rather than averaging
    fewer than five players.
    """
    flags = match_combat_roster_flags(players)
    if flags.empty:
        return
    bad = flags.loc[~flags["complete_state_join"]]
    if bad.empty:
        return
    sample = bad.iloc[0]
    raise CombatRosterError(
        "incomplete or malformed combat roster for match_id="
        f"{sample[MATCH_ID_COLUMN]}: n_rows={sample['n_rows']}, "
        f"n_radiant={sample['n_radiant']}, n_dire={sample['n_dire']}, "
        f"five_per_side={sample['five_per_side']}, "
        f"unique_within_side={sample['unique_within_side']}, "
        f"no_cross_side_duplicate={sample['no_cross_side_duplicate']}, "
        f"complete_state={sample['complete_state']}"
    )


def player_combat_side_profile(players: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(match_id, side)`` from five rostered player rows.

    Mean shrunk C includes cold-start zeros and always divides by five.
    Evidence sums include zeros. Does not weight by ``combat_prior_n``.
    """
    _require_columns(players, PLAYER_COMBAT_REQUIRED_COLUMNS, label="player frame")
    assert_combat_roster(players)

    work = players.loc[:, list(PLAYER_COMBAT_REQUIRED_COLUMNS)].copy()
    work["combat_shrunk_c"] = pd.to_numeric(work["combat_shrunk_c"], errors="coerce")
    work["combat_prior_n"] = pd.to_numeric(work["combat_prior_n"], errors="coerce")
    work["combat_cold_start"] = (work["combat_prior_n"].fillna(-1.0) == 0.0).astype(int)
    grouped = work.groupby(
        list(PLAYER_COMBAT_SIDE_IDENTITY_COLUMNS),
        dropna=False,
        sort=False,
    )
    frame = grouped.agg(
        mean_combat_shrunk_c=("combat_shrunk_c", "mean"),
        combat_prior_n_sum=("combat_prior_n", "sum"),
        combat_cold_start_count=("combat_cold_start", "sum"),
        n_players=("player_id", "size"),
    ).reset_index()
    if not (frame["n_players"] == COMBAT_ROSTER_SIDE_SIZE).all():
        raise CombatRosterError(
            "side aggregation did not receive exactly five players per side"
        )
    frame = frame.drop(columns="n_players")
    if frame.empty:
        return pd.DataFrame(columns=list(PLAYER_COMBAT_SIDE_COLUMNS))
    side_order = frame[SIDE_COLUMN].map({_RADIANT_SIDE: 0, _DIRE_SIDE: 1})
    return (
        frame[list(PLAYER_COMBAT_SIDE_COLUMNS)]
        .assign(_side_order=side_order)
        .sort_values([MATCH_ID_COLUMN, "_side_order"], kind="mergesort")
        .drop(columns="_side_order")
        .reset_index(drop=True)
    )


def player_combat_comparison_from_side(profile: pd.DataFrame) -> pd.DataFrame:
    """Radiant − Dire from an already-aggregated side profile."""
    _require_columns(profile, PLAYER_COMBAT_SIDE_COLUMNS, label="side profile")

    radiant = profile.loc[profile[SIDE_COLUMN] == _RADIANT_SIDE]
    dire = profile.loc[profile[SIDE_COLUMN] == _DIRE_SIDE]
    if radiant[MATCH_ID_COLUMN].duplicated().any():
        raise CombatRosterError("expected exactly one Radiant row per match_id")
    if dire[MATCH_ID_COLUMN].duplicated().any():
        raise CombatRosterError("expected exactly one Dire row per match_id")
    radiant_ids = set(radiant[MATCH_ID_COLUMN].tolist())
    dire_ids = set(dire[MATCH_ID_COLUMN].tolist())
    if radiant_ids != dire_ids:
        raise CombatRosterError("match is missing a Radiant or Dire combat side")

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
            "radiant_mean_combat_shrunk_c": merged["mean_combat_shrunk_c_radiant"],
            "dire_mean_combat_shrunk_c": merged["mean_combat_shrunk_c_dire"],
            "mean_combat_shrunk_c_diff": (
                merged["mean_combat_shrunk_c_radiant"]
                - merged["mean_combat_shrunk_c_dire"]
            ),
            "radiant_combat_prior_n_sum": merged["combat_prior_n_sum_radiant"],
            "dire_combat_prior_n_sum": merged["combat_prior_n_sum_dire"],
            "radiant_combat_cold_start_count": merged[
                "combat_cold_start_count_radiant"
            ],
            "dire_combat_cold_start_count": merged["combat_cold_start_count_dire"],
        }
    )
    return (
        frame[list(PLAYER_COMBAT_COMPARISON_COLUMNS)]
        .sort_values(["start_time", MATCH_ID_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )


def player_combat_comparison_from_players(players: pd.DataFrame) -> pd.DataFrame:
    """Match-level Radiant − Dire combat comparison from player rows."""
    return player_combat_comparison_from_side(player_combat_side_profile(players))


def merge_player_combat_comparison(
    matches: pd.DataFrame, comparison: pd.DataFrame
) -> pd.DataFrame:
    """Left-join match-level combat diffs onto an existing match frame."""
    needed = (MATCH_ID_COLUMN, *PLAYER_COMBAT_FEATURE_COLUMNS)
    _require_columns(comparison, needed, label="comparison")
    if MATCH_ID_COLUMN not in matches.columns:
        raise ValueError("match frame is missing match_id")
    extra = comparison.loc[:, [MATCH_ID_COLUMN, *PLAYER_COMBAT_FEATURE_COLUMNS]]
    return matches.merge(extra, on=MATCH_ID_COLUMN, how="left", validate="one_to_one")
