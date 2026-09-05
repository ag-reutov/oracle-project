"""Tests for the Slice 6 team-strength derivation (`data.team_strength`).

Pure, in-memory tests against `derive_team_strength_state` /
`derive_latest_ratings` / `check_future_deletion_invariant` /
`source_fingerprint` -- no database. Covers the Section 15 requirements:
canonical initial rating, zero-sum updates, favorite-vs-upset magnitude,
causal elo_pre/record, equal-timestamp mutual blindness, match-id
independence from chronology, future-deletion invariance, deterministic
latest-Elo state (no ranking) with corpus as-of time, the production-Elo
cross-check, and the deterministic source-fingerprint (detects an old
result/team/time correction even when count and corpus extrema are
unchanged). Slice 6 deliberately exposes no ordinal ranking.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from dota_predictor.data.team_strength import (
    MatchFact,
    TeamStrengthState,
    check_future_deletion_invariant,
    derive_latest_ratings,
    derive_team_strength_state,
    source_fingerprint,
)
from dota_predictor.features.team_elo import (
    DEFAULT_ELO_CONFIG,
    EloConfig,
    compute_team_elo_features,
    expected_score,
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)

TEAM_A, TEAM_B, TEAM_C, TEAM_D = 1, 2, 3, 4

CONFIG = EloConfig()


def _fact(
    match_id: int,
    *,
    start_time: datetime,
    radiant_team_id: int,
    dire_team_id: int,
    radiant_win: bool,
    radiant_name: str | None = None,
    dire_name: str | None = None,
) -> MatchFact:
    return MatchFact(
        match_id=match_id,
        start_time=start_time,
        radiant_team_id=radiant_team_id,
        dire_team_id=dire_team_id,
        radiant_win=radiant_win,
        radiant_team_name=radiant_name,
        dire_team_name=dire_name,
    )


def _state_for(
    states: list[TeamStrengthState], match_id: int, team_id: int
) -> TeamStrengthState:
    return next(s for s in states if s.match_id == match_id and s.team_id == team_id)


def _side_state(states: list[TeamStrengthState], match_id: int, side: str) -> TeamStrengthState:
    return next(s for s in states if s.match_id == match_id and s.side == side)


# --- 1. new teams begin at the canonical initial rating --------------------


def test_new_teams_begin_at_initial_rating() -> None:
    states = derive_team_strength_state(
        [
            _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True)
        ],
        config=CONFIG,
    )
    assert len(states) == 2
    assert _side_state(states, 1, "RADIANT").elo_pre == DEFAULT_ELO_CONFIG.initial_rating
    assert _side_state(states, 1, "DIRE").elo_pre == DEFAULT_ELO_CONFIG.initial_rating


# --- 2/3. winner gains / loser loses ----------------------------------------


def test_winner_gains_and_loser_loses() -> None:
    states = derive_team_strength_state(
        [
            _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True)
        ],
        config=CONFIG,
    )
    gain = CONFIG.k_factor * 0.5
    assert _side_state(states, 1, "RADIANT").elo_post == pytest.approx(
        CONFIG.initial_rating + gain
    )
    assert _side_state(states, 1, "DIRE").elo_post == pytest.approx(
        CONFIG.initial_rating - gain
    )


# --- 4. update is zero-sum (true of the existing definition) ---------------


def test_update_is_zero_sum() -> None:
    states = derive_team_strength_state(
        [
            _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True)
        ],
        config=CONFIG,
    )
    total_post = _side_state(states, 1, "RADIANT").elo_post + _side_state(
        states, 1, "DIRE"
    ).elo_post
    assert total_post == pytest.approx(2 * CONFIG.initial_rating)


# --- 5. favorite win changes less than upset win ----------------------------


def test_favorite_win_changes_less_than_upset_win() -> None:
    # Build up a favorite: TEAM_A beats two fresh teams.
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    states = derive_team_strength_state(rows, config=CONFIG)
    favorite_pre = _state_for(states, 2, TEAM_A).elo_post

    # Favorite (TEAM_A) beats a fresh team vs a fresh team beating TEAM_A.
    favorite_win = derive_team_strength_state(
        rows
        + [_fact(3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_D, radiant_win=True)],
        config=CONFIG,
    )
    upset = derive_team_strength_state(
        rows
        + [_fact(3, start_time=T3, radiant_team_id=TEAM_D, dire_team_id=TEAM_A, radiant_win=True)],
        config=CONFIG,
    )
    favorite_delta = abs(
        _state_for(favorite_win, 3, TEAM_A).elo_post - favorite_pre
    )
    upset_delta = abs(_state_for(upset, 3, TEAM_A).elo_post - favorite_pre)
    assert upset_delta > favorite_delta


# --- 6. current result cannot affect current elo_pre -------------------------


def test_current_result_does_not_affect_own_elo_pre() -> None:
    prior = _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True)
    current_win = _fact(
        2, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True
    )
    current_loss = _fact(
        2, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=False
    )
    win_states = derive_team_strength_state([prior, current_win], config=CONFIG)
    loss_states = derive_team_strength_state([prior, current_loss], config=CONFIG)
    assert _state_for(win_states, 2, TEAM_A).elo_pre == pytest.approx(
        _state_for(loss_states, 2, TEAM_A).elo_pre
    )
    assert _state_for(win_states, 2, TEAM_B).elo_pre == pytest.approx(
        _state_for(loss_states, 2, TEAM_B).elo_pre
    )


# --- 7. previous result affects next strictly-later match --------------------


def test_previous_result_affects_next_strictly_later_match() -> None:
    states = derive_team_strength_state(
        [
            _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
            _fact(2, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
        ],
        config=CONFIG,
    )
    assert _state_for(states, 2, TEAM_A).elo_pre == pytest.approx(
        CONFIG.initial_rating + CONFIG.k_factor * 0.5
    )


# --- 8. match_id order cannot replace chronological order --------------------


def test_match_id_order_cannot_replace_chronological_order() -> None:
    # A has a higher match_id but an EARLIER start_time.
    rows = [
        _fact(200, start_time=T2, radiant_team_id=TEAM_B, dire_team_id=TEAM_C, radiant_win=True),
        _fact(100, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
    ]
    states = derive_team_strength_state(rows, config=CONFIG)
    # TEAM_B lost match 100 (T1) then won match 200 (T2). Its rating entering
    # match 200 must reflect the loss, so it is below initial.
    assert _state_for(states, 200, TEAM_B).elo_pre < CONFIG.initial_rating


# --- 9. equal timestamps do not create causal precedence ---------------------


def test_equal_timestamps_do_not_create_causal_precedence() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    states = derive_team_strength_state(rows, config=CONFIG)
    for match_id in (1, 2):
        s = _state_for(states, match_id, TEAM_A)
        assert s.elo_pre == CONFIG.initial_rating
        assert s.prior_match_count == 0
        assert s.previous_match_id is None
        assert s.is_first_observed_match is True


def test_equal_timestamp_group_updates_apply_as_one_independent_batch() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
        _fact(3, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_D, radiant_win=True),
    ]
    states = derive_team_strength_state(rows, config=CONFIG)
    gain = CONFIG.k_factor * (
        1.0 - expected_score(CONFIG.initial_rating, CONFIG.initial_rating)
    )
    assert _state_for(states, 3, TEAM_A).elo_pre == pytest.approx(
        CONFIG.initial_rating + 2 * gain
    )


# --- 10. future deletion leaves earlier elo_pre unchanged --------------------


def test_future_deletion_leaves_earlier_state_unchanged() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=False),
        _fact(3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    full = derive_team_strength_state(rows, config=CONFIG)
    for match_id in (1, 2):
        before = _state_for(full, match_id, TEAM_A)
        truncated = derive_team_strength_state(
            [m for m in rows if m.start_time <= before.start_time], config=CONFIG
        )
        after = _state_for(truncated, match_id, TEAM_A)
        assert before.elo_pre == pytest.approx(after.elo_pre)
        assert before.elo_post == pytest.approx(after.elo_post)
        assert before.prior_match_count == after.prior_match_count

    result = check_future_deletion_invariant(rows, config=CONFIG)
    assert result["violations"] == []


# --- 11/12. prior record is causal; first match has zero prior record --------


def test_prior_record_is_causal_and_first_match_zero() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=False),
        _fact(3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    states = derive_team_strength_state(rows, config=CONFIG)
    first = _state_for(states, 1, TEAM_A)
    assert (first.prior_match_count, first.prior_win_count, first.prior_loss_count) == (0, 0, 0)
    assert first.prior_win_rate is None
    assert first.is_first_observed_match is True

    second = _state_for(states, 2, TEAM_A)
    assert (second.prior_match_count, second.prior_win_count, second.prior_loss_count) == (1, 1, 0)
    assert second.prior_win_rate == pytest.approx(1.0)
    assert second.previous_match_id == 1
    assert second.days_since_previous_match == pytest.approx(31.0, rel=0.01)

    third = _state_for(states, 3, TEAM_A)
    assert (third.prior_match_count, third.prior_win_count, third.prior_loss_count) == (2, 1, 1)
    assert third.prior_win_rate == pytest.approx(0.5)
    assert third.previous_match_id == 2


# --- 13. latest rating equals the state after the final observed result -----


def test_latest_rating_equals_final_observed_state() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
    ]
    states = derive_team_strength_state(rows, config=CONFIG)
    latest = derive_latest_ratings(rows, config=CONFIG)
    a = next(r for r in latest if r.team_id == TEAM_A)
    assert a.rating == pytest.approx(_state_for(states, 2, TEAM_A).elo_post)
    assert a.last_match_id == 2
    assert a.observed_match_count == 2
    assert a.wins == 2
    assert a.losses == 0


# --- 14. latest-Elo state is deterministic (no ranking) ----------------------


def test_latest_ratings_are_deterministic() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T1, radiant_team_id=TEAM_C, dire_team_id=TEAM_D, radiant_win=False),
    ]
    latest1 = derive_latest_ratings(rows, config=CONFIG)
    latest2 = derive_latest_ratings(rows, config=CONFIG)
    assert [(r.team_id, r.rating) for r in latest1] == [
        (r.team_id, r.rating) for r in latest2
    ]
    # One latest-state row per distinct team_id (no rank, no merging).
    assert [r.team_id for r in latest1] == sorted(r.team_id for r in latest1)
    assert len(latest1) == 4


# --- 15. latest-Elo state exposes corpus as-of time ---------------------------


def test_latest_ratings_expose_corpus_as_of_time() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
    ]
    latest = derive_latest_ratings(rows, config=CONFIG)
    assert all(r.as_of_at == T1 for r in latest)


# --- 16. team ids are never merged (one latest-state row per team_id) ---------


def test_team_ids_are_not_merged_by_identity() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=False),
    ]
    latest = derive_latest_ratings(rows, config=CONFIG)
    team_ids = {r.team_id for r in latest}
    # TEAM_A and TEAM_B remain two distinct latest-state rows regardless of
    # any shared identity/lineage that a later (Slice 7) layer might infer.
    assert team_ids == {TEAM_A, TEAM_B}
    assert len(latest) == 2


# --- 17. production-Elo cross-check -------------------------------------------


def test_elo_pre_cross_checks_against_production_features() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_B, dire_team_id=TEAM_C, radiant_win=True),
        _fact(3, start_time=T2, radiant_team_id=TEAM_D, dire_team_id=TEAM_A, radiant_win=False),
        _fact(4, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
    ]
    states = derive_team_strength_state(rows, config=CONFIG)
    frame = pd.DataFrame(
        [
            {
                "match_id": m.match_id,
                "start_time": m.start_time,
                "radiant_team_id": m.radiant_team_id,
                "dire_team_id": m.dire_team_id,
                "radiant_win": m.radiant_win,
            }
            for m in rows
        ]
    )
    prod = compute_team_elo_features(frame, config=CONFIG).set_index("match_id")
    for s in states:
        expected = (
            prod.loc[s.match_id, "radiant_team_elo"]
            if s.side == "RADIANT"
            else prod.loc[s.match_id, "dire_team_elo"]
        )
        assert s.elo_pre == pytest.approx(float(expected))

# --- 18. source fingerprint is deterministic and content-sensitive -----------


def test_source_fingerprint_is_deterministic_and_order_independent() -> None:
    rows = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_B, dire_team_id=TEAM_C, radiant_win=False),
    ]
    fp1 = source_fingerprint(rows)
    fp2 = source_fingerprint(list(reversed(rows)))
    fp3 = source_fingerprint(rows)
    assert fp1 == fp2 == fp3
    assert isinstance(fp1, str) and len(fp1) == 64


def test_source_fingerprint_detects_result_change_even_with_unchanged_count_and_extrema() -> None:
    base = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_B, dire_team_id=TEAM_C, radiant_win=False),
        _fact(3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    # Same match count, same min/max start_time, but match 1's result flips.
    altered = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=False),
        _fact(2, start_time=T2, radiant_team_id=TEAM_B, dire_team_id=TEAM_C, radiant_win=False),
        _fact(3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    assert source_fingerprint(base) != source_fingerprint(altered)


def test_source_fingerprint_detects_team_change_even_with_unchanged_count_and_extrema() -> None:
    base = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_B, dire_team_id=TEAM_C, radiant_win=False),
        _fact(3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    # Same count/extrema; match 2's dire team changes.
    altered = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_B, dire_team_id=TEAM_D, radiant_win=False),
        _fact(3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    assert source_fingerprint(base) != source_fingerprint(altered)


def test_source_fingerprint_detects_time_correction_even_with_unchanged_count_and_extrema() -> None:
    base = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=T2, radiant_team_id=TEAM_B, dire_team_id=TEAM_C, radiant_win=False),
        _fact(3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    # Same count and same corpus extrema (T1 and T3), but match 2's start
    # time is corrected within the range.
    altered = [
        _fact(1, start_time=T1, radiant_team_id=TEAM_A, dire_team_id=TEAM_B, radiant_win=True),
        _fact(2, start_time=datetime(2024, 2, 15, tzinfo=UTC), radiant_team_id=TEAM_B, dire_team_id=TEAM_C, radiant_win=False),
        _fact(3, start_time=T3, radiant_team_id=TEAM_A, dire_team_id=TEAM_C, radiant_win=True),
    ]
    assert source_fingerprint(base) != source_fingerprint(altered)
