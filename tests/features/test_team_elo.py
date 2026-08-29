"""Tests for the Step 3C historical team Elo feature layer
(`features.team_elo`).

Pure, in-memory tests against `compute_team_elo_features` directly --
no DuckDB connection, no Parquet fixtures, no PostgreSQL. Integration
with `PreDraftSnapshot`/`FEATURE_COLUMNS` is covered separately in
`test_pre_draft_snapshot.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from dota_predictor.features.team_elo import (
    DIRE_TEAM_ELO_COLUMN,
    RADIANT_TEAM_ELO_COLUMN,
    TEAM_ELO_DELTA_COLUMN,
    TEAM_ELO_FEATURE_COLUMNS,
    EloConfig,
    InvalidTeamIdError,
    compute_team_elo_features,
    expected_score,
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)

TEAM_A, TEAM_B, TEAM_C, TEAM_D = 1, 2, 3, 4


def _match(
    match_id: int,
    *,
    start_time: datetime,
    radiant_team_id: int,
    dire_team_id: int,
    radiant_win: bool,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "start_time": start_time,
        "radiant_team_id": radiant_team_id,
        "dire_team_id": dire_team_id,
        "radiant_win": radiant_win,
    }


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- first appearance / default rating ------------------------------------


def test_both_teams_receive_default_initial_rating_on_first_appearance() -> None:
    frame = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            )
        ]
    )
    result = compute_team_elo_features(frame).set_index("match_id")

    assert result.loc[1, RADIANT_TEAM_ELO_COLUMN] == 1500.0
    assert result.loc[1, DIRE_TEAM_ELO_COLUMN] == 1500.0
    assert result.loc[1, TEAM_ELO_DELTA_COLUMN] == 0.0


def test_custom_initial_rating_is_honored() -> None:
    config = EloConfig(initial_rating=1000.0, k_factor=32.0)
    frame = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            )
        ]
    )
    result = compute_team_elo_features(frame, config=config).set_index("match_id")

    assert result.loc[1, RADIANT_TEAM_ELO_COLUMN] == 1000.0
    assert result.loc[1, DIRE_TEAM_ELO_COLUMN] == 1000.0


# --- winner gains / loser loses the expected amount, subsequent match ----


def test_winner_gains_and_loser_loses_expected_amount_in_the_next_match() -> None:
    """Two brand-new teams: expected score is exactly 0.5 each, so a
    win/loss moves the rating by exactly `k_factor * 0.5`."""
    config = EloConfig()
    frame = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
            # Fresh opponents at T2 so we can read TEAM_A/TEAM_B's
            # updated ratings directly off the next match's snapshot.
            _match(
                2,
                start_time=T2,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_C,
                radiant_win=True,
            ),
            _match(
                3,
                start_time=T2,
                radiant_team_id=TEAM_D,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
        ]
    )
    result = compute_team_elo_features(frame, config=config).set_index("match_id")

    expected_gain = config.k_factor * 0.5
    assert result.loc[2, RADIANT_TEAM_ELO_COLUMN] == pytest.approx(
        config.initial_rating + expected_gain
    )
    assert result.loc[3, DIRE_TEAM_ELO_COLUMN] == pytest.approx(
        config.initial_rating - expected_gain
    )


def test_subsequent_match_sees_the_updated_rating_not_the_default() -> None:
    frame = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
            _match(
                2,
                start_time=T2,
                radiant_team_id=TEAM_B,
                dire_team_id=TEAM_C,
                radiant_win=False,
            ),
        ]
    )
    result = compute_team_elo_features(frame).set_index("match_id")

    # TEAM_B lost match 1 as dire, so its rating entering match 2 (as
    # radiant) must be below the default -- not reset to 1500.
    assert result.loc[2, RADIANT_TEAM_ELO_COLUMN] < 1500.0


# --- Radiant/Dire side swap preserves team identity/rating ---------------


def test_side_swap_preserves_team_rating_history() -> None:
    """TEAM_A wins match 1 as radiant, then appears as DIRE in match 2
    against a fresh opponent: its updated rating must carry over
    regardless of side."""
    frame = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
            _match(
                2,
                start_time=T2,
                radiant_team_id=TEAM_C,
                dire_team_id=TEAM_A,
                radiant_win=False,
            ),
        ]
    )
    result = compute_team_elo_features(frame).set_index("match_id")

    expected_rating = 1500.0 + EloConfig().k_factor * 0.5
    assert result.loc[2, DIRE_TEAM_ELO_COLUMN] == pytest.approx(expected_rating)


# --- multiple sequential wins accumulate correctly ------------------------


def test_multiple_sequential_wins_accumulate_correctly() -> None:
    """Reproduces the same trajectory using the public `expected_score`
    helper as an independent reference, rather than hard-coded magic
    numbers."""
    config = EloConfig()
    opponents = [TEAM_B, TEAM_C, TEAM_D]
    rows = [
        _match(
            i + 1,
            start_time=T1 + timedelta(days=i),
            radiant_team_id=TEAM_A,
            dire_team_id=opponent,
            radiant_win=True,
        )
        for i, opponent in enumerate(opponents)
    ]
    result = compute_team_elo_features(_frame(rows), config=config).set_index(
        "match_id"
    )

    reference_rating = config.initial_rating
    for i in range(1, len(opponents) + 1):
        assert result.loc[i, RADIANT_TEAM_ELO_COLUMN] == pytest.approx(reference_rating)
        assert result.loc[i, DIRE_TEAM_ELO_COLUMN] == pytest.approx(
            config.initial_rating
        )
        gain = config.k_factor * (
            1.0 - expected_score(reference_rating, config.initial_rating)
        )
        reference_rating += gain

    assert result[RADIANT_TEAM_ELO_COLUMN].is_monotonic_increasing


# --- current outcome cannot affect current Elo features ------------------


def test_current_match_outcome_does_not_affect_its_own_snapshot() -> None:
    """Two otherwise-identical histories that only differ in the
    *current* match's own `radiant_win` must produce identical Elo
    features for that match -- the outcome only affects future rows."""
    prior = _match(
        1,
        start_time=T1,
        radiant_team_id=TEAM_A,
        dire_team_id=TEAM_B,
        radiant_win=True,
    )
    current_win = _match(
        2,
        start_time=T2,
        radiant_team_id=TEAM_A,
        dire_team_id=TEAM_B,
        radiant_win=True,
    )
    current_loss = {**current_win, "radiant_win": False}

    result_win = compute_team_elo_features(_frame([prior, current_win])).set_index(
        "match_id"
    )
    result_loss = compute_team_elo_features(_frame([prior, current_loss])).set_index(
        "match_id"
    )

    for column in TEAM_ELO_FEATURE_COLUMNS:
        assert result_win.loc[2, column] == pytest.approx(result_loss.loc[2, column])


# --- future-match exclusion -----------------------------------------------


def test_future_match_cannot_influence_an_earlier_matchs_features() -> None:
    without_future = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            )
        ]
    )
    with_future = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
            _match(
                2,
                start_time=T2,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
        ]
    )

    result_without = compute_team_elo_features(without_future).set_index("match_id")
    result_with = compute_team_elo_features(with_future).set_index("match_id")

    for column in TEAM_ELO_FEATURE_COLUMNS:
        assert result_without.loc[1, column] == pytest.approx(
            result_with.loc[1, column]
        )


# --- equal-start_time matches cannot influence one another ----------------


def test_equal_start_time_matches_read_the_same_pre_group_rating() -> None:
    """TEAM_A plays two matches at the exact same `start_time`: both
    must see TEAM_A's pre-group rating (1500.0), neither influenced by
    the other."""
    frame = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
            _match(
                2,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_C,
                radiant_win=True,
            ),
        ]
    )
    result = compute_team_elo_features(frame).set_index("match_id")

    assert result.loc[1, RADIANT_TEAM_ELO_COLUMN] == 1500.0
    assert result.loc[2, RADIANT_TEAM_ELO_COLUMN] == 1500.0


def test_equal_start_time_group_updates_are_applied_as_one_independent_batch() -> None:
    """After the tied group resolves, a later match must see TEAM_A's
    rating reflecting BOTH tied wins added independently to the
    pre-group rating -- not one win compounding on top of the other."""
    config = EloConfig()
    frame = _frame(
        [
            _match(
                1,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_B,
                radiant_win=True,
            ),
            _match(
                2,
                start_time=T1,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_C,
                radiant_win=True,
            ),
            _match(
                3,
                start_time=T2,
                radiant_team_id=TEAM_A,
                dire_team_id=TEAM_D,
                radiant_win=True,
            ),
        ]
    )
    result = compute_team_elo_features(frame, config=config).set_index("match_id")

    gain_vs_b = config.k_factor * (
        1.0 - expected_score(config.initial_rating, config.initial_rating)
    )
    gain_vs_c = config.k_factor * (
        1.0 - expected_score(config.initial_rating, config.initial_rating)
    )
    expected_rating_after_group = config.initial_rating + gain_vs_b + gain_vs_c

    assert result.loc[3, RADIANT_TEAM_ELO_COLUMN] == pytest.approx(
        expected_rating_after_group
    )


# --- match_id permutation within equal timestamps does not change features


def test_match_id_permutation_within_a_tied_group_does_not_change_features() -> None:
    content_rows = [
        _match(
            100,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        _match(
            200,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_C,
            radiant_win=False,
        ),
    ]
    swapped_rows = [
        {**content_rows[1], "match_id": 100},
        {**content_rows[0], "match_id": 200},
    ]

    result_original = compute_team_elo_features(_frame(content_rows)).set_index(
        "match_id"
    )
    result_swapped = compute_team_elo_features(_frame(swapped_rows)).set_index(
        "match_id"
    )

    # match_id 100's content (TEAM_A vs TEAM_B, radiant win) must
    # produce the same features no matter which match_id label or row
    # position it was assigned.
    for column in TEAM_ELO_FEATURE_COLUMNS:
        assert result_original.loc[100, column] == pytest.approx(
            result_swapped.loc[200, column]
        )
        assert result_original.loc[200, column] == pytest.approx(
            result_swapped.loc[100, column]
        )


def test_row_order_within_a_tied_group_does_not_change_features() -> None:
    rows = [
        _match(
            1,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        _match(
            2,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_C,
            radiant_win=False,
        ),
    ]
    result_forward = compute_team_elo_features(_frame(rows)).set_index("match_id")
    result_reversed = compute_team_elo_features(_frame(list(reversed(rows)))).set_index(
        "match_id"
    )

    pd.testing.assert_frame_equal(
        result_forward.sort_index(), result_reversed.sort_index()
    )


# --- output grain / exactness ----------------------------------------------


def test_output_has_exactly_one_row_per_input_match() -> None:
    matches, _ = _build_five_match_frame()
    result = compute_team_elo_features(matches)

    assert len(result) == len(matches)
    assert set(result["match_id"]) == set(matches["match_id"])


def test_team_elo_delta_is_exactly_radiant_minus_dire() -> None:
    matches, _ = _build_five_match_frame()
    result = compute_team_elo_features(matches)

    diff = result[RADIANT_TEAM_ELO_COLUMN] - result[DIRE_TEAM_ELO_COLUMN]
    assert (result[TEAM_ELO_DELTA_COLUMN] == diff).all()


def _build_five_match_frame() -> tuple[pd.DataFrame, EloConfig]:
    config = EloConfig()
    rows = [
        _match(
            10,
            start_time=T1,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
        _match(
            20,
            start_time=T1,
            radiant_team_id=TEAM_C,
            dire_team_id=TEAM_D,
            radiant_win=False,
        ),
        _match(
            30,
            start_time=T2,
            radiant_team_id=TEAM_B,
            dire_team_id=TEAM_C,
            radiant_win=True,
        ),
        _match(
            40,
            start_time=T2,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_D,
            radiant_win=False,
        ),
        _match(
            50,
            start_time=T3,
            radiant_team_id=TEAM_A,
            dire_team_id=TEAM_B,
            radiant_win=True,
        ),
    ]
    return _frame(rows), config


# --- missing/invalid team ids ----------------------------------------------


@pytest.mark.parametrize("bad_team_id", [None, float("nan"), 0, -5])
def test_missing_or_invalid_radiant_team_id_raises(bad_team_id: object) -> None:
    frame = _frame(
        [
            {
                "match_id": 1,
                "start_time": T1,
                "radiant_team_id": bad_team_id,
                "dire_team_id": TEAM_B,
                "radiant_win": True,
            }
        ]
    )
    with pytest.raises(InvalidTeamIdError):
        compute_team_elo_features(frame)


def test_missing_dire_team_id_raises() -> None:
    frame = _frame(
        [
            {
                "match_id": 1,
                "start_time": T1,
                "radiant_team_id": TEAM_A,
                "dire_team_id": None,
                "radiant_win": True,
            }
        ]
    )
    with pytest.raises(InvalidTeamIdError):
        compute_team_elo_features(frame)


def test_missing_radiant_win_raises_instead_of_silently_skipping_update() -> None:
    frame = _frame(
        [
            {
                "match_id": 1,
                "start_time": T1,
                "radiant_team_id": TEAM_A,
                "dire_team_id": TEAM_B,
                "radiant_win": None,
            }
        ]
    )
    with pytest.raises(InvalidTeamIdError):
        compute_team_elo_features(frame)


def test_missing_required_column_raises_value_error() -> None:
    frame = pd.DataFrame([{"match_id": 1, "start_time": T1, "radiant_team_id": TEAM_A}])
    with pytest.raises(ValueError, match="missing required columns"):
        compute_team_elo_features(frame)
