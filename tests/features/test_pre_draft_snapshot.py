"""Tests for the Step 3B PRE_DRAFT historical training snapshot.

All fixtures build real canonical Parquet files via `pre_draft_helpers`
(itself built on the real Step 2 transform functions) -- no PostgreSQL,
no synthetic DataFrame shortcuts around the actual Parquet contract.

The main scenario (`multi_match_snapshot`) intentionally exercises, in
one fixture:

* a team appearing on both Radiant and Dire across matches (TeamA,
  TeamB) -- proving win/loss attribution follows team identity, not
  side;
* a player (`PLAYER_1`) moving from one team to a brand-new team
  (TeamA -> TeamD) between matches -- proving player history is
  tracked independently of team history;
* brand-new (zero-history) teams and players at every position a
  zero-history entity can occur;
* partial and full roster continuity (5/5, 4/5) and the "no prior
  match" NULL case;
* match_ids assigned in an order unrelated to `start_time`, so any
  code that accidentally used `match_id` as a time proxy would produce
  wrong numbers here, not just fail a narrow unit test.

All expected values below are computed by hand in the comments and
cross-checked against the module docstring's algorithm description.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pre_draft_helpers import build_feature_store_config, match_row, player_rows

from dota_predictor.features.availability import MATCHES_COLUMN_AVAILABILITY
from dota_predictor.features.duckdb_layer import DRAFT_EVENTS_VIEW, connect
from dota_predictor.features.pre_draft_snapshot import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    PRE_DRAFT_SNAPSHOT_SQL,
    SNAPSHOT_COLUMNS,
    TARGET_COLUMN,
    build_pre_draft_snapshot,
)
from dota_predictor.features.team_elo import (
    DIRE_TEAM_ELO_COLUMN,
    RADIANT_TEAM_ELO_COLUMN,
    TEAM_ELO_DELTA_COLUMN,
    TEAM_ELO_FEATURE_COLUMNS,
    EloConfig,
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)
T4 = datetime(2024, 4, 1, tzinfo=UTC)

TEAM_A, TEAM_B, TEAM_C, TEAM_D, TEAM_E = 1, 2, 3, 4, 5

# Match ids are deliberately assigned in an order unrelated to
# start_time (M1 < M2 < M3 < M4 chronologically, but
# 4001 > 1002, 3003 > 2004, etc.) -- see module docstring.
M1, M2, M3, M4 = 4001, 1002, 3003, 2004

PLAYER_1 = 1  # appears on TeamA (M1-M3) then TeamD (M4)


def _build_multi_match_matches() -> tuple[list[dict], list[dict]]:
    matches = [
        match_row(
            M1,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        match_row(
            M2,
            start_time=T2,
            radiant_team_id=TEAM_B,
            dire_team_id=TEAM_A,
            radiant_win=True,
        ),
        match_row(
            M3,
            start_time=T3,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_C,
            radiant_win=False,
        ),
        match_row(
            M4,
            start_time=T4,
            radiant_team_id=TEAM_D,
            dire_team_id=TEAM_E,
            radiant_win=True,
        ),
    ]
    players = (
        player_rows(M1, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
        + player_rows(M2, radiant_ids=(6, 7, 8, 9, 999), dire_ids=(1, 2, 3, 4, 5))
        + player_rows(M3, radiant_ids=(1, 2, 3, 4, 777), dire_ids=(11, 12, 13, 14, 15))
        + player_rows(
            M4, radiant_ids=(1, 20, 21, 22, 23), dire_ids=(24, 25, 26, 27, 28)
        )
    )
    return matches, players


@pytest.fixture
def multi_match_snapshot(tmp_path: Path) -> pd.DataFrame:
    matches, players = _build_multi_match_matches()
    config = build_feature_store_config(tmp_path, matches=matches, players=players)
    with connect(config) as store:
        return build_pre_draft_snapshot(store).to_frame().set_index("match_id")


# --- grain / identity ----------------------------------------------------


def test_exactly_one_row_per_input_match(multi_match_snapshot: pd.DataFrame) -> None:
    assert len(multi_match_snapshot) == 4
    assert set(multi_match_snapshot.index) == {M1, M2, M3, M4}


def test_identity_columns_present(multi_match_snapshot: pd.DataFrame) -> None:
    for column in IDENTITY_COLUMNS:
        if column == "match_id":
            continue  # used as the index above
        assert column in multi_match_snapshot.columns


# --- team history: exact counts/wins/losses, zero history -----------------


def test_zero_history_team_counts_are_zero_and_win_rate_is_null(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    row = multi_match_snapshot.loc[M1]
    for column in (
        "radiant_team_prior_matches",
        "radiant_team_prior_wins",
        "radiant_team_prior_losses",
        "dire_team_prior_matches",
        "dire_team_prior_wins",
        "dire_team_prior_losses",
    ):
        assert row[column] == 0, column
    assert pd.isna(row["radiant_team_prior_win_rate"])
    assert pd.isna(row["dire_team_prior_win_rate"])
    assert pd.isna(row["team_prior_win_rate_delta"])


def test_team_history_across_radiant_and_dire_appearances(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    """TeamA (radiant in M1, dire in M2) and TeamB (dire in M1, radiant
    in M2): win/loss must follow team identity, not side."""
    row = multi_match_snapshot.loc[M2]

    # TeamB is now radiant; its only history is M1 as dire, where it lost.
    assert row["radiant_team_prior_matches"] == 1
    assert row["radiant_team_prior_wins"] == 0
    assert row["radiant_team_prior_losses"] == 1
    assert row["radiant_team_prior_win_rate"] == pytest.approx(0.0)

    # TeamA is now dire; its only history is M1 as radiant, where it won.
    assert row["dire_team_prior_matches"] == 1
    assert row["dire_team_prior_wins"] == 1
    assert row["dire_team_prior_losses"] == 0
    assert row["dire_team_prior_win_rate"] == pytest.approx(1.0)

    assert row["team_prior_matches_delta"] == 0
    assert row["team_prior_wins_delta"] == -1
    assert row["team_prior_losses_delta"] == 1
    assert row["team_prior_win_rate_delta"] == pytest.approx(-1.0)


def test_team_history_accumulates_across_multiple_prior_matches(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    """TeamA by M3 has played twice (M1 win as radiant, M2 loss as
    dire): 2 prior matches, 1 win, 1 loss, 0.5 win rate."""
    row = multi_match_snapshot.loc[M3]
    assert row["radiant_team_prior_matches"] == 2
    assert row["radiant_team_prior_wins"] == 1
    assert row["radiant_team_prior_losses"] == 1
    assert row["radiant_team_prior_win_rate"] == pytest.approx(0.5)

    # TeamC is brand new.
    assert row["dire_team_prior_matches"] == 0
    assert pd.isna(row["dire_team_prior_win_rate"])
    assert row["team_prior_matches_delta"] == 2
    assert pd.isna(row["team_prior_win_rate_delta"])


def test_brand_new_teams_never_seen_before_are_all_zero_or_null(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    row = multi_match_snapshot.loc[M4]
    for column in (
        "radiant_team_prior_matches",
        "radiant_team_prior_wins",
        "radiant_team_prior_losses",
        "dire_team_prior_matches",
        "dire_team_prior_wins",
        "dire_team_prior_losses",
    ):
        assert row[column] == 0, column
    assert pd.isna(row["radiant_team_prior_win_rate"])
    assert pd.isna(row["dire_team_prior_win_rate"])
    assert row["team_prior_matches_delta"] == 0
    assert pd.isna(row["team_prior_win_rate_delta"])


# --- player history: across team changes, aggregation, zero history -------


def test_player_history_persists_across_a_team_change(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    """PLAYER_1 played M1 (TeamA, radiant, won), M2 (TeamA, dire,
    lost), M3 (TeamA, radiant, lost) -- then moved to brand-new TeamD
    for M4. Team-level TeamD history must be 0 (never played before),
    but PLAYER_1's own history (3 matches, 1 win, 2 losses) must still
    show up in TeamD's *player* aggregates -- proving player history is
    tracked on player identity, independent of team identity."""
    row = multi_match_snapshot.loc[M4]

    assert row["radiant_team_prior_matches"] == 0  # TeamD itself is brand new

    # radiant players are (PLAYER_1, 20, 21, 22, 23): PLAYER_1 has 3
    # prior matches (1 win, 2 losses -> win_rate 1/3); the other four
    # are brand-new players with 0 matches each.
    assert row["radiant_players_prior_matches_max"] == 3
    assert row["radiant_players_prior_matches_min"] == 0
    assert row["radiant_players_prior_matches_mean"] == pytest.approx(
        (3 + 0 + 0 + 0 + 0) / 5
    )
    assert row["radiant_players_prior_win_rate_mean"] == pytest.approx(1 / 3)
    assert row["radiant_players_zero_prior_matches_count"] == 4


def test_player_aggregation_mean_min_max_and_zero_count(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    """M2 radiant = TeamB (6, 7, 8, 9, 999): the first four each have 1
    prior match (M1, dire, lost); player 999 is brand new."""
    row = multi_match_snapshot.loc[M2]

    assert row["radiant_players_prior_matches_mean"] == pytest.approx(0.8)
    assert row["radiant_players_prior_matches_min"] == 0
    assert row["radiant_players_prior_matches_max"] == 1
    assert row["radiant_players_prior_win_rate_mean"] == pytest.approx(0.0)
    assert row["radiant_players_zero_prior_matches_count"] == 1

    # M2 dire = TeamA (1, 2, 3, 4, 5): all five won their only prior
    # match (M1, radiant).
    assert row["dire_players_prior_matches_mean"] == pytest.approx(1.0)
    assert row["dire_players_prior_matches_min"] == 1
    assert row["dire_players_prior_matches_max"] == 1
    assert row["dire_players_prior_win_rate_mean"] == pytest.approx(1.0)
    assert row["dire_players_zero_prior_matches_count"] == 0

    assert row["players_prior_matches_mean_delta"] == pytest.approx(0.8 - 1.0)
    assert row["players_zero_prior_matches_count_delta"] == 1


def test_player_win_rate_mean_ignores_zero_history_players_not_zero(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    """If every current player on a side is brand new, the mean prior
    win rate must be NULL (no players with a defined rate), not 0."""
    row = multi_match_snapshot.loc[M3]
    assert row["dire_players_prior_win_rate_mean"] is None or pd.isna(
        row["dire_players_prior_win_rate_mean"]
    )
    assert row["dire_players_zero_prior_matches_count"] == 5


# --- roster continuity ----------------------------------------------------


def test_roster_continuity_full_retention(multi_match_snapshot: pd.DataFrame) -> None:
    """TeamA's M2 dire roster (1,2,3,4,5) is identical to its M1
    radiant roster: 5/5 retained."""
    assert multi_match_snapshot.loc[M2, "dire_roster_players_retained"] == 5


def test_roster_continuity_partial_retention(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    """TeamB's M2 radiant roster (6,7,8,9,999) drops player 10 from its
    M1 dire roster (6,7,8,9,10): 4/5 retained."""
    assert multi_match_snapshot.loc[M2, "radiant_roster_players_retained"] == 4
    assert multi_match_snapshot.loc[M3, "radiant_roster_players_retained"] == 4


def test_roster_continuity_values_are_within_0_to_5(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    for column in ("radiant_roster_players_retained", "dire_roster_players_retained"):
        values = multi_match_snapshot[column].dropna()
        assert values.between(0, 5).all()


def test_team_with_no_prior_match_has_null_roster_continuity(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    assert pd.isna(multi_match_snapshot.loc[M1, "radiant_roster_players_retained"])
    assert pd.isna(multi_match_snapshot.loc[M1, "dire_roster_players_retained"])
    # TeamC (M3, dire) and TeamD/TeamE (M4) are also brand new.
    assert pd.isna(multi_match_snapshot.loc[M3, "dire_roster_players_retained"])
    assert pd.isna(multi_match_snapshot.loc[M4, "radiant_roster_players_retained"])
    assert pd.isna(multi_match_snapshot.loc[M4, "dire_roster_players_retained"])
    assert pd.isna(multi_match_snapshot.loc[M3, "roster_players_retained_delta"])


# --- temporal correctness: explicit invariants from the task spec ---------


def test_current_match_cannot_contribute_to_its_own_statistics(tmp_path: Path) -> None:
    matches = [
        match_row(
            555,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        )
    ]
    players = player_rows(555, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
    config = build_feature_store_config(tmp_path, matches=matches, players=players)

    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame()

    row = df.iloc[0]
    assert row["radiant_team_prior_matches"] == 0
    assert row["dire_team_prior_matches"] == 0
    assert row["radiant_players_prior_matches_mean"] == 0
    assert pd.isna(row["radiant_roster_players_retained"])


def test_future_match_cannot_contribute(tmp_path: Path) -> None:
    """An earlier match's features must not reflect a later match for
    the same team, even though both are present in the dataset."""
    matches = [
        match_row(
            10,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        match_row(
            20,
            start_time=T2,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_C,
            radiant_win=True,
        ),
    ]
    players = player_rows(
        10, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10)
    ) + player_rows(20, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(11, 12, 13, 14, 15))
    config = build_feature_store_config(tmp_path, matches=matches, players=players)

    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame().set_index("match_id")

    # Match 10 (the earlier match) must show TeamA with zero history,
    # even though TeamA plays again (and wins) in the later match 20.
    assert df.loc[10, "radiant_team_prior_matches"] == 0
    assert pd.isna(df.loc[10, "radiant_team_prior_win_rate"])


def test_equal_start_time_matches_do_not_contribute_to_each_other(
    tmp_path: Path,
) -> None:
    """Two matches sharing the exact same `start_time` must each see
    the other as having zero history -- a tie is never historical."""
    matches = [
        match_row(
            1,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        match_row(
            2,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=False,
        ),
    ]
    players = player_rows(
        1, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10)
    ) + player_rows(2, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
    config = build_feature_store_config(tmp_path, matches=matches, players=players)

    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame().set_index("match_id")

    for match_id in (1, 2):
        assert df.loc[match_id, "radiant_team_prior_matches"] == 0
        assert df.loc[match_id, "dire_team_prior_matches"] == 0
        assert pd.isna(df.loc[match_id, "radiant_roster_players_retained"])


def test_multiple_equal_timestamp_historical_matches_all_contribute(
    tmp_path: Path,
) -> None:
    """Two strictly-earlier matches that happen to share a timestamp
    with EACH OTHER (but are both earlier than the current match) must
    both count."""
    matches = [
        match_row(
            1,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        match_row(
            2,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_C,
            radiant_win=False,
        ),
        match_row(
            3,
            start_time=T2,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_D,
            radiant_win=True,
        ),
    ]
    players = (
        player_rows(1, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
        + player_rows(2, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(11, 12, 13, 14, 15))
        + player_rows(3, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(16, 17, 18, 19, 20))
    )
    config = build_feature_store_config(tmp_path, matches=matches, players=players)

    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame().set_index("match_id")

    # TeamA played both tied t1 matches (1 win, 1 loss) before match 3.
    assert df.loc[3, "radiant_team_prior_matches"] == 2
    assert df.loc[3, "radiant_team_prior_wins"] == 1
    assert df.loc[3, "radiant_team_prior_losses"] == 1


def test_match_id_ordering_does_not_affect_feature_values(tmp_path: Path) -> None:
    """Re-assigning match_ids (still nonchronologically) must not
    change any computed feature value keyed by team/start_time."""
    matches_a, players_a = _build_multi_match_matches()

    # A different, still-nonchronological permutation of match ids.
    remap = {M1: 90001, M2: 90002, M3: 90003, M4: 90004}
    matches_b = [{**row, "match_id": remap[row["match_id"]]} for row in matches_a]
    players_b = [{**row, "match_id": remap[row["match_id"]]} for row in players_a]

    config_a = build_feature_store_config(
        tmp_path / "a", matches=matches_a, players=players_a
    )
    config_b = build_feature_store_config(
        tmp_path / "b", matches=matches_b, players=players_b
    )

    with connect(config_a) as store_a:
        df_a = build_pre_draft_snapshot(store_a).to_frame().sort_values("start_time")
    with connect(config_b) as store_b:
        df_b = build_pre_draft_snapshot(store_b).to_frame().sort_values("start_time")

    feature_and_target = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    pd.testing.assert_frame_equal(
        df_a[feature_and_target].reset_index(drop=True),
        df_b[feature_and_target].reset_index(drop=True),
    )


def test_roster_continuity_uses_start_time_not_match_id_for_recency(
    tmp_path: Path,
) -> None:
    """A team's chronologically-earlier match is given a LARGER
    match_id than its chronologically-later (but still prior) match, to
    catch any code that sorts "most recent" by match_id instead of
    start_time."""
    matches = [
        # Chronologically earliest for TeamA, but the largest match_id.
        match_row(
            9999,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        # Chronologically the most recent PRIOR match for TeamA, with a
        # small match_id.
        match_row(
            1,
            start_time=T2,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_C,
            radiant_win=True,
        ),
        # Current match.
        match_row(
            5000,
            start_time=T3,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_D,
            radiant_win=True,
        ),
    ]
    players = (
        player_rows(9999, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
        + player_rows(1, radiant_ids=(1, 2, 3, 4, 99), dire_ids=(11, 12, 13, 14, 15))
        + player_rows(5000, radiant_ids=(1, 2, 3, 4, 99), dire_ids=(16, 17, 18, 19, 20))
    )
    config = build_feature_store_config(tmp_path, matches=matches, players=players)

    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame().set_index("match_id")

    # The correct "most recent prior roster" is match_id=1's roster
    # (1,2,3,4,99), which is identical to the current roster: 5/5.
    # If match_id (9999) were used instead, the comparison would be
    # against (1,2,3,4,5), giving only 4/5.
    assert df.loc[5000, "radiant_roster_players_retained"] == 5


def test_swapping_radiant_dire_in_historical_matches_does_not_break_win_attribution(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    # TeamA won M1 as radiant, then lost M2 as dire: net 1-1 heading
    # into M3, not "2 wins" (which double side-blind counting could
    # produce) and not "0 wins" (which side-swap confusion could
    # produce).
    row = multi_match_snapshot.loc[M3]
    assert row["radiant_team_prior_wins"] == 1
    assert row["radiant_team_prior_losses"] == 1


# --- availability / feature contract --------------------------------------


def test_target_is_excluded_from_feature_columns() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in IDENTITY_COLUMNS


def test_target_present_in_full_frame_but_not_in_feature_frame(
    tmp_path: Path,
) -> None:
    matches, players = _build_multi_match_matches()
    config = build_feature_store_config(tmp_path, matches=matches, players=players)

    with connect(config) as store:
        snapshot = build_pre_draft_snapshot(store)
        full = snapshot.to_frame()
        features = snapshot.feature_frame()
        target = snapshot.target_series()

    assert TARGET_COLUMN in full.columns
    assert TARGET_COLUMN not in features.columns
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert target.name == TARGET_COLUMN
    assert set(target.unique()) <= {True, False}


def test_no_draft_information_is_available_in_pre_draft_snapshot() -> None:
    draft_only_columns = {"sequence", "action", "hero_id", "was_successful"}
    assert draft_only_columns.isdisjoint(SNAPSHOT_COLUMNS)
    # Structural guard: the query must never even reference the draft
    # events view.
    assert DRAFT_EVENTS_VIEW not in PRE_DRAFT_SNAPSHOT_SQL


def test_no_post_match_information_other_than_the_separated_target() -> None:
    post_match_columns = {
        column
        for column, availability in MATCHES_COLUMN_AVAILABILITY.items()
        if availability.value == "POST_MATCH"
    }
    assert "duration_seconds" in post_match_columns  # sanity on the source map
    disallowed = post_match_columns - {TARGET_COLUMN}
    assert disallowed.isdisjoint(SNAPSHOT_COLUMNS)
    assert "duration_seconds" not in SNAPSHOT_COLUMNS


def test_provenance_columns_are_absent() -> None:
    assert "mapper_version" not in SNAPSHOT_COLUMNS
    assert "canonicalized_at" not in SNAPSHOT_COLUMNS


def test_observed_team_names_and_player_ids_are_not_exposed() -> None:
    assert "radiant_team_name_observed" not in SNAPSHOT_COLUMNS
    assert "dire_team_name_observed" not in SNAPSHOT_COLUMNS
    for column in SNAPSHOT_COLUMNS:
        assert "player_id" not in column


# --- structural / static guards on the SQL --------------------------------


def test_sql_never_uses_match_id_as_a_historical_ordering_predicate() -> None:
    """Every historical-eligibility join condition must compare
    `start_time`. The single legitimate use of `match_id` in a temporal
    ordering context is the `ORDER BY ..., h.match_id DESC` tie-breaker
    inside `prior_team_match_candidates`, which only orders rows
    already filtered to `start_time <` -- never used as a substitute
    filter condition itself."""
    # Exactly 4 real historical-eligibility join predicates (radiant
    # team history, dire team history, player history, roster
    # continuity's candidate search); a 5th textual match is just the
    # explanatory SQL comment above the roster-continuity join, not a
    # predicate.
    assert PRE_DRAFT_SNAPSHOT_SQL.count("AND h.start_time <") == 4
    assert "match_id <" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "match_id >" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "match_id <=" not in PRE_DRAFT_SNAPSHOT_SQL
    assert "match_id >=" not in PRE_DRAFT_SNAPSHOT_SQL


def test_feature_columns_are_disjoint_from_identity_and_target() -> None:
    assert set(FEATURE_COLUMNS).isdisjoint(IDENTITY_COLUMNS)
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert (
        tuple(IDENTITY_COLUMNS) + tuple(FEATURE_COLUMNS) + (TARGET_COLUMN,)
        == SNAPSHOT_COLUMNS
    )


# --- Step 3C: team Elo integration -----------------------------------------


def test_team_elo_columns_are_included_in_feature_columns_and_excluded_elsewhere() -> (
    None
):
    assert set(TEAM_ELO_FEATURE_COLUMNS).issubset(FEATURE_COLUMNS)
    assert set(TEAM_ELO_FEATURE_COLUMNS).isdisjoint(IDENTITY_COLUMNS)
    assert TARGET_COLUMN not in TEAM_ELO_FEATURE_COLUMNS


def test_team_elo_present_in_full_frame_and_feature_frame_but_not_target(
    tmp_path: Path,
) -> None:
    matches, players = _build_multi_match_matches()
    config = build_feature_store_config(tmp_path, matches=matches, players=players)

    with connect(config) as store:
        snapshot = build_pre_draft_snapshot(store)
        full = snapshot.to_frame()
        features = snapshot.feature_frame()
        target = snapshot.target_series()

    for column in TEAM_ELO_FEATURE_COLUMNS:
        assert column in full.columns
        assert column in features.columns
    assert TARGET_COLUMN not in features.columns
    assert TARGET_COLUMN not in TEAM_ELO_FEATURE_COLUMNS
    assert target.name == TARGET_COLUMN


def test_first_appearance_teams_both_get_default_elo_rating(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    row = multi_match_snapshot.loc[M1]
    assert row[RADIANT_TEAM_ELO_COLUMN] == 1500.0
    assert row[DIRE_TEAM_ELO_COLUMN] == 1500.0
    assert row[TEAM_ELO_DELTA_COLUMN] == 0.0


def test_team_elo_delta_is_exactly_radiant_minus_dire(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    diff = (
        multi_match_snapshot[RADIANT_TEAM_ELO_COLUMN]
        - multi_match_snapshot[DIRE_TEAM_ELO_COLUMN]
    )
    assert (multi_match_snapshot[TEAM_ELO_DELTA_COLUMN] == diff).all()


def test_team_elo_output_has_exactly_one_row_per_canonical_match(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    assert len(multi_match_snapshot) == 4
    for column in TEAM_ELO_FEATURE_COLUMNS:
        assert multi_match_snapshot[column].notna().all()


def test_team_elo_side_swap_preserves_rating_across_the_full_pipeline(
    tmp_path: Path,
) -> None:
    """TeamA wins M1 as radiant, then appears as DIRE in a later match:
    its rating must carry over through the full `PreDraftSnapshot`
    pipeline (SQL history features + Python Elo merge), not just the
    standalone `team_elo` module."""
    matches = [
        match_row(
            1,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        match_row(
            2,
            start_time=T2,
            radiant_team_id=TEAM_C,
            dire_team_id=TEAM_A,
            radiant_win=False,
        ),
    ]
    players = player_rows(
        1, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10)
    ) + player_rows(2, radiant_ids=(11, 12, 13, 14, 15), dire_ids=(1, 2, 3, 4, 5))
    config = build_feature_store_config(tmp_path, matches=matches, players=players)

    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame().set_index("match_id")

    expected_rating = 1500.0 + EloConfig().k_factor * 0.5
    assert df.loc[2, DIRE_TEAM_ELO_COLUMN] == pytest.approx(expected_rating)


def test_team_elo_equal_start_time_matches_do_not_influence_each_other(
    tmp_path: Path,
) -> None:
    matches = [
        match_row(
            1,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        match_row(
            2,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_C,
            radiant_win=True,
        ),
    ]
    players = player_rows(
        1, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10)
    ) + player_rows(2, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(11, 12, 13, 14, 15))
    config = build_feature_store_config(tmp_path, matches=matches, players=players)

    with connect(config) as store:
        df = build_pre_draft_snapshot(store).to_frame().set_index("match_id")

    assert df.loc[1, RADIANT_TEAM_ELO_COLUMN] == 1500.0
    assert df.loc[2, RADIANT_TEAM_ELO_COLUMN] == 1500.0


def test_team_elo_match_id_permutation_within_equal_timestamps_is_stable(
    tmp_path: Path,
) -> None:
    """Re-assigning match_ids among two equal-`start_time` matches must
    not change any team's computed Elo feature values, mirroring
    `test_match_id_ordering_does_not_affect_feature_values` for Step 3C."""
    matches_a = [
        match_row(
            111,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        match_row(
            222,
            start_time=T1,
            radiant_team_id=TEAM_C,
            dire_team_id=TEAM_D,
            radiant_win=False,
        ),
        match_row(
            333,
            start_time=T2,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_C,
            radiant_win=True,
        ),
    ]
    players_a = (
        player_rows(111, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(6, 7, 8, 9, 10))
        + player_rows(
            222, radiant_ids=(11, 12, 13, 14, 15), dire_ids=(16, 17, 18, 19, 20)
        )
        + player_rows(333, radiant_ids=(1, 2, 3, 4, 5), dire_ids=(11, 12, 13, 14, 15))
    )

    remap = {111: 999, 222: 111, 333: 222}
    matches_b = [{**row, "match_id": remap[row["match_id"]]} for row in matches_a]
    players_b = [{**row, "match_id": remap[row["match_id"]]} for row in players_a]

    config_a = build_feature_store_config(
        tmp_path / "a", matches=matches_a, players=players_a
    )
    config_b = build_feature_store_config(
        tmp_path / "b", matches=matches_b, players=players_b
    )

    with connect(config_a) as store_a:
        df_a = build_pre_draft_snapshot(store_a).to_frame().sort_values("start_time")
    with connect(config_b) as store_b:
        df_b = build_pre_draft_snapshot(store_b).to_frame().sort_values("start_time")

    pd.testing.assert_frame_equal(
        df_a[list(TEAM_ELO_FEATURE_COLUMNS)].reset_index(drop=True),
        df_b[list(TEAM_ELO_FEATURE_COLUMNS)].reset_index(drop=True),
    )


def test_full_snapshot_column_order_matches_snapshot_columns(
    multi_match_snapshot: pd.DataFrame,
) -> None:
    assert list(multi_match_snapshot.reset_index().columns) == list(SNAPSHOT_COLUMNS)
