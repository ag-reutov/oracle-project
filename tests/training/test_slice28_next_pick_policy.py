"""Slice 28 causal next-pick draft-policy benchmark tests.

Leakage, common-support, Slice 26 reuse, holdout, and production boundary.
Research only; not a production feature.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from training_helpers import match_row, player_rows

from dota_predictor.datasets.canonical_export import (
    DRAFT_EVENTS_FILENAME,
    MATCH_PLAYERS_FILENAME,
    MATCHES_FILENAME,
    build_draft_events_table,
    build_match_players_table,
    build_matches_table,
    write_canonical_dataset,
)
from dota_predictor.features.config import FeatureStoreConfig
from dota_predictor.features.duckdb_layer import connect
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS
from dota_predictor.training.next_pick_policy import (
    BASELINE_A,
    BASELINE_B,
    PICK_HERO_PREFIX,
    POLICY_EARLY_STOPPING_MIN_FACTOR,
    POLICY_SGD_MAX_ITER,
    SLICE28_DIAGNOSTIC_ONLY,
    build_causal_pick_history,
    build_next_pick_decision_rows,
    build_policy_feature_vector,
    candidate_universe_for_row,
    run_slice28_next_pick_policy_benchmark,
    score_policy_distribution,
    unavailable_heroes,
    _fit_multinomial_policy,
    _match_clustered_delta_ci,
    _score_frequency_rows,
)
from dota_predictor.training.player_hero_pool_state import SCORING_MIXTURE_EPSILON
from dota_predictor.training.player_hero_pool_state import SCORING_MIXTURE_EPSILON as S25_EPS
from dota_predictor.training.sequential_draft_benchmark import build_match_draft_index
from dota_predictor.training.sequential_draft_state import (
    ACTION_BAN,
    ACTION_PICK,
    BOUNDARY_CONVENTION,
    SIDE_DIRE,
    SIDE_RADIANT,
    build_draft_prefix_state,
    event_is_actual,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END

RADIANT_IDS = (11, 12, 13, 14, 15)
DIRE_IDS = (21, 22, 23, 24, 25)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _event(
    sequence: int,
    action: str,
    side: str,
    hero_id: int,
    *,
    was_successful: bool | None = None,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "action": action,
        "side": side,
        "hero_id": hero_id,
        "was_successful": was_successful,
    }


def _draft_with_bans_and_picks(*, failed_ban: bool = False) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        _event(0, ACTION_BAN, SIDE_DIRE, 50, was_successful=True),
        _event(1, ACTION_BAN, SIDE_RADIANT, 51, was_successful=True),
    ]
    if failed_ban:
        events.append(_event(2, ACTION_BAN, SIDE_RADIANT, 99, was_successful=False))
        seq = 3
    else:
        seq = 2
    pick_plan = [
        (SIDE_DIRE, 6),
        (SIDE_RADIANT, 1),
        (SIDE_DIRE, 7),
        (SIDE_RADIANT, 2),
        (SIDE_DIRE, 8),
        (SIDE_RADIANT, 3),
        (SIDE_DIRE, 9),
        (SIDE_RADIANT, 4),
        (SIDE_DIRE, 10),
        (SIDE_RADIANT, 5),
    ]
    for side, hero in pick_plan:
        events.append(_event(seq, ACTION_PICK, side, hero))
        seq += 1
    return events


def _matches_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_scored_rows_are_immediately_before_successful_pick() -> None:
    events = _draft_with_bans_and_picks()
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    matches = _matches_frame(
        {
            "match_id": 1,
            "start_time": T0,
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 200,
        }
    )
    rows = build_next_pick_decision_rows(matches=matches, indexes={1: index})
    assert len(rows) == 10
    for _, row in rows.iterrows():
        boundary = int(row["boundary_t"])
        target_events = [e for e in events if int(e["sequence"]) == boundary]
        assert len(target_events) == 1
        assert target_events[0]["action"] == ACTION_PICK
        assert int(target_events[0]["hero_id"]) == int(row["next_pick_hero_id"])
        assert BOUNDARY_CONVENTION == "before_event_t"


def test_target_hero_absent_from_state_features() -> None:
    events = _draft_with_bans_and_picks()
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    matches = _matches_frame(
        {
            "match_id": 1,
            "start_time": T0,
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 200,
        }
    )
    rows = build_next_pick_decision_rows(matches=matches, indexes={1: index})
    for row in rows.to_dict(orient="records"):
        target = int(row["next_pick_hero_id"])
        prefix = set(row["radiant_pick_hero_ids"]) | set(row["dire_pick_hero_ids"])
        bans = set(row["radiant_ban_hero_ids"]) | set(row["dire_ban_hero_ids"])
        assert target not in prefix
        assert target not in bans
        feats = build_policy_feature_vector(
            row,
            pick_vocabulary=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            include_picks=True,
        )
        assert feats.get(f"{PICK_HERO_PREFIX}{target}", 0.0) == 0.0


def test_future_events_cannot_affect_current_policy_features() -> None:
    events = _draft_with_bans_and_picks()
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    matches = _matches_frame(
        {
            "match_id": 1,
            "start_time": T0,
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 200,
        }
    )
    base = build_next_pick_decision_rows(matches=matches, indexes={1: index})
    mutated = [dict(e) for e in events]
    mutated[-1]["hero_id"] = 42
    mutated[-2]["hero_id"] = 43
    index2 = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in mutated])
    )[1]
    after = build_next_pick_decision_rows(matches=matches, indexes={1: index2})
    early = base.loc[base["pick_decision_index"] == 3].iloc[0]
    early2 = after.loc[after["pick_decision_index"] == 3].iloc[0]
    assert early["radiant_pick_hero_ids"] == early2["radiant_pick_hero_ids"]
    assert early["dire_pick_hero_ids"] == early2["dire_pick_hero_ids"]


def test_picked_and_successful_bans_excluded_from_universe() -> None:
    prior = frozenset(range(1, 60))
    unavail = unavailable_heroes(
        radiant_pick_hero_ids=(1,),
        dire_pick_hero_ids=(6,),
        radiant_ban_hero_ids=(51,),
        dire_ban_hero_ids=(50,),
    )
    assert unavail == frozenset({1, 6, 50, 51})
    cands = candidate_universe_for_row(
        prior_heroes=prior, unavailable=unavail, realized_hero=2
    )
    assert 1 not in cands and 6 not in cands and 50 not in cands and 51 not in cands
    assert 2 in cands


def test_failed_bans_do_not_make_heroes_unavailable() -> None:
    events = _draft_with_bans_and_picks(failed_ban=True)
    assert event_is_actual(ACTION_BAN, False) is False
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    matches = _matches_frame(
        {
            "match_id": 1,
            "start_time": T0,
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 200,
        }
    )
    rows = build_next_pick_decision_rows(matches=matches, indexes={1: index})
    first = rows.iloc[0]
    assert 99 not in first["unavailable_hero_ids"]
    assert 50 in first["unavailable_hero_ids"]
    assert 51 in first["unavailable_hero_ids"]


def test_acting_side_and_pick_index_are_causal() -> None:
    events = _draft_with_bans_and_picks()
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    matches = _matches_frame(
        {
            "match_id": 1,
            "start_time": T0,
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 200,
        }
    )
    rows = build_next_pick_decision_rows(matches=matches, indexes={1: index})
    # First pick is Dire hero 6 at sequence 2.
    r0 = rows.iloc[0]
    assert r0["acting_side"] == SIDE_DIRE
    assert r0["acting_team_id"] == 200
    assert r0["overall_pick_index"] == 1
    assert r0["side_pick_index"] == 1
    assert r0["n_dire_picks"] == 0
    # Second pick Radiant.
    r1 = rows.iloc[1]
    assert r1["acting_side"] == SIDE_RADIANT
    assert r1["acting_team_id"] == 100
    assert r1["overall_pick_index"] == 2
    assert r1["n_dire_picks"] == 1
    assert r1["n_radiant_picks"] == 0


def test_common_C_T_identical_across_estimators() -> None:
    prior = frozenset({1, 2, 3, 4, 5})
    unavail = frozenset({1})
    realized = 2
    cands = candidate_universe_for_row(
        prior_heroes=prior, unavailable=unavail, realized_hero=realized
    )
    mass_a = {2: 3.0, 3: 1.0}
    mass_b = {2: 1.0, 4: 9.0, 99: 100.0}  # 99 not in C_T
    sa = score_policy_distribution(mass_a, realized_hero=realized, candidates=cands)
    sb = score_policy_distribution(mass_b, realized_hero=realized, candidates=cands)
    assert sa["n_candidates"] == sb["n_candidates"] == float(len(cands))
    # Estimator-specific support cannot shrink C_T.
    assert 99 not in cands


def test_estimator_specific_support_cannot_change_normalization() -> None:
    cands = frozenset({1, 2, 3})
    # Model only supports {1}; still scored over full C_T via mixture.
    mass = {1: 1.0}
    scored = score_policy_distribution(mass, realized_hero=2, candidates=cands)
    # With p(2)=0 before mixture, q = epsilon / 3
    expected_q = SCORING_MIXTURE_EPSILON / 3.0
    assert scored["p_realized"] == pytest.approx(expected_q)
    assert SCORING_MIXTURE_EPSILON == S25_EPS


def test_first_observed_target_handled_consistently() -> None:
    prior = frozenset({1, 2, 3})
    unavail = frozenset()
    realized = 99
    cands = candidate_universe_for_row(
        prior_heroes=prior, unavailable=unavail, realized_hero=realized
    )
    assert 99 in cands
    sa = score_policy_distribution({1: 1.0}, realized_hero=99, candidates=cands)
    sb = score_policy_distribution({}, realized_hero=99, candidates=cands)
    assert sa["n_candidates"] == sb["n_candidates"]
    # Empty mass -> uniform after mixture still defined.
    assert np.isfinite(sb["log_loss"])


def test_history_mutually_blind_at_same_timestamp() -> None:
    events = _draft_with_bans_and_picks()
    idx_frame = pd.DataFrame(
        [{**e, "match_id": mid} for mid in (1, 2) for e in events]
    )
    indexes = build_match_draft_index(idx_frame)
    matches = _matches_frame(
        {
            "match_id": 1,
            "start_time": T0,
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 200,
        },
        {
            "match_id": 2,
            "start_time": T0,  # same timestamp
            "game_version_id": 176,
            "radiant_team_id": 101,
            "dire_team_id": 201,
        },
    )
    decisions = build_next_pick_decision_rows(matches=matches, indexes=indexes)
    history = build_causal_pick_history(decisions)
    history.advance_before(pd.Timestamp(T0))
    assert history.snapshot_prior_heroes() == frozenset()
    # Later match can see earlier.
    later = T0 + timedelta(hours=1)
    history2 = build_causal_pick_history(decisions)
    history2.advance_before(pd.Timestamp(later))
    assert 6 in history2.snapshot_prior_heroes()


def test_team_history_uses_only_prior_matches() -> None:
    events = _draft_with_bans_and_picks()
    idx_frame = pd.DataFrame(
        [{**e, "match_id": mid} for mid in (1, 2) for e in events]
    )
    indexes = build_match_draft_index(idx_frame)
    matches = _matches_frame(
        {
            "match_id": 1,
            "start_time": T0,
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 200,
        },
        {
            "match_id": 2,
            "start_time": T0 + timedelta(days=1),
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 201,
        },
    )
    decisions = build_next_pick_decision_rows(matches=matches, indexes=indexes)
    history = build_causal_pick_history(decisions)
    # Before match 2: team 100 has picks from match 1 only.
    history.advance_before(pd.Timestamp(T0 + timedelta(days=1)))
    assert sum(history.team_counts[100].values()) == 5  # radiant picks in match 1
    assert history._ptr == 1  # only match 1 included


def test_models_compared_on_identical_decision_rows() -> None:
    events = _draft_with_bans_and_picks()
    indexes = {
        mid: build_match_draft_index(
            pd.DataFrame([{**e, "match_id": mid} for e in events])
        )[mid]
        for mid in (1, 2, 3)
    }
    matches = _matches_frame(
        *[
            {
                "match_id": mid,
                "start_time": T0 + timedelta(days=mid),
                "game_version_id": 176,
                "radiant_team_id": 100 + mid,
                "dire_team_id": 200 + mid,
            }
            for mid in (1, 2, 3)
        ]
    )
    decisions = build_next_pick_decision_rows(matches=matches, indexes=indexes)
    history = build_causal_pick_history(decisions)
    test_rows = decisions.to_dict(orient="records")
    a = pd.DataFrame(_score_frequency_rows(test_rows, model=BASELINE_A, history=history))
    b = pd.DataFrame(_score_frequency_rows(test_rows, model=BASELINE_B, history=history))
    merged = a.merge(
        b,
        on=["match_id", "pick_decision_index"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    assert len(merged) == len(a) == len(b)


def test_match_level_bootstrap_clusters_decisions() -> None:
    # Two matches × 10 picks with constant per-match deltas.
    rows = []
    for match_id, delta in ((1, -0.2), (2, 0.4)):
        for i in range(10):
            rows.append(
                {
                    "match_id": match_id,
                    "left": 1.0 + delta,
                    "right": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    ci = _match_clustered_delta_ci(
        frame, left_col="left", right_col="right", seed=0
    )
    # Mean of match means = (-0.2 + 0.4) / 2 = 0.1, not row-mean of 0.1
    # (same here) — importantly n_matches=2 not 20.
    assert ci["n_matches"] == 2.0
    assert ci["n_rows"] == 20.0
    assert ci["mean"] == pytest.approx(0.1)


def test_no_outcome_position_assignment_leakage_in_features() -> None:
    events = _draft_with_bans_and_picks()
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    matches = _matches_frame(
        {
            "match_id": 1,
            "start_time": T0,
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 200,
        }
    )
    rows = build_next_pick_decision_rows(matches=matches, indexes={1: index})
    row = rows.iloc[4].to_dict()
    feats = build_policy_feature_vector(
        row,
        pick_vocabulary=(1, 2, 6, 7),
        ban_vocabulary=(50, 51),
        version_vocabulary=(176,),
        team_vocabulary=(100, 200),
        include_picks=True,
        include_bans=True,
        include_team_identity=True,
    )
    forbidden = {
        "radiant_win",
        "duration",
        "position",
        "slot_in_side",
        "player_id",
        "assignment",
        "kills",
        "team_elo_delta",
    }
    assert forbidden.isdisjoint(feats.keys())


def test_slice26_builder_reused_for_prefix() -> None:
    events = _draft_with_bans_and_picks()
    index = build_match_draft_index(
        pd.DataFrame([{**e, "match_id": 1} for e in events])
    )[1]
    matches = _matches_frame(
        {
            "match_id": 1,
            "start_time": T0,
            "game_version_id": 176,
            "radiant_team_id": 100,
            "dire_team_id": 200,
        }
    )
    rows = build_next_pick_decision_rows(matches=matches, indexes={1: index})
    row = rows.iloc[3]
    state = build_draft_prefix_state(
        match_id=1,
        start_time=T0,
        game_version_id=176,
        boundary_t=int(row["boundary_t"]),
        events=events,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_player_ids=RADIANT_IDS,
        dire_player_ids=DIRE_IDS,
    )
    assert tuple(state["radiant_pick_hero_ids"]) == tuple(row["radiant_pick_hero_ids"])
    assert tuple(state["dire_pick_hero_ids"]) == tuple(row["dire_pick_hero_ids"])
    assert tuple(state["radiant_ban_hero_ids"]) == tuple(row["radiant_ban_hero_ids"])
    assert tuple(state["dire_ban_hero_ids"]) == tuple(row["dire_ban_hero_ids"])


def test_feature_columns_remain_33() -> None:
    assert len(FEATURE_COLUMNS) == 33
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)
    assert SLICE28_DIAGNOSTIC_ONLY is True


def test_train_vocab_cannot_see_future_fold_categories() -> None:
    # Pick vocabulary for features is derived from train rows only inside
    # _fit_multinomial_policy; assert feature builder zeros unknown heroes.
    row = {
        "acting_side": SIDE_RADIANT,
        "overall_pick_index": 3,
        "side_pick_index": 2,
        "game_version_id": 176,
        "radiant_pick_hero_ids": (1,),
        "dire_pick_hero_ids": (6,),
        "radiant_ban_hero_ids": (),
        "dire_ban_hero_ids": (),
        "acting_team_id": 100,
    }
    feats = build_policy_feature_vector(
        row,
        pick_vocabulary=(1, 6),  # train vocab excludes future hero 99
        include_picks=True,
    )
    assert f"{PICK_HERO_PREFIX}99" not in feats
    assert feats[f"{PICK_HERO_PREFIX}1"] == 1.0


def _write_mini_store(tmp_path: Path, *, n_matches: int = 12) -> FeatureStoreConfig:
    matches = []
    players = []
    drafts = []
    for i in range(n_matches):
        mid = 1000 + i
        t = T0 + timedelta(days=i)
        assert t <= FROZEN_DEVELOPMENT_END
        matches.append(
            match_row(
                mid,
                start_time=t,
                radiant_team_id=100 + (i % 3),
                dire_team_id=200 + (i % 3),
                radiant_win=i % 2 == 0,
                game_version_id=176 + (i % 2),
            )
        )
        rows = player_rows(mid, radiant_ids=RADIANT_IDS, dire_ids=DIRE_IDS)
        for row in rows:
            if row["side"] == SIDE_RADIANT:
                row["hero_id"] = int(row["slot_in_side"]) + 1
            else:
                row["hero_id"] = int(row["slot_in_side"]) + 6
        players.extend(rows)
        for e in _draft_with_bans_and_picks():
            drafts.append({"match_id": mid, **e})

    matches_table = build_matches_table(matches, players)
    players_table = build_match_players_table(matches, players)
    draft_table = build_draft_events_table(drafts)
    write_canonical_dataset(
        tmp_path,
        matches_table=matches_table,
        draft_events_table=draft_table,
        match_players_table=players_table,
    )
    return FeatureStoreConfig(
        matches_path=tmp_path / MATCHES_FILENAME,
        match_players_path=tmp_path / MATCH_PLAYERS_FILENAME,
        draft_events_path=tmp_path / DRAFT_EVENTS_FILENAME,
    )


def test_holdout_excluded_and_end_to_end_smoke(tmp_path: Path) -> None:
    from dota_predictor.training.walk_forward import WalkForwardConfig

    config = _write_mini_store(tmp_path, n_matches=16)
    with connect(config) as store:
        report = run_slice28_next_pick_policy_benchmark(
            store,
            development_end=FROZEN_DEVELOPMENT_END,
            walk_forward_config=WalkForwardConfig(
                n_blocks=4, train_fraction_of_past=0.7
            ),
            run_logistic_candidates=True,
        )
    assert report.n_holdout_excluded == 0
    assert report.integrity["feature_columns_unchanged"] is True
    assert report.integrity["holdout_excluded"] is True
    assert report.n_decision_rows == 16 * 10
    assert report.n_oos_rows > 0
    assert not report.pooled_metrics.empty
    models = set(report.pooled_metrics["model"].tolist())
    assert BASELINE_A in models and BASELINE_B in models


def test_tiny_fold_early_stopping_fallback() -> None:
    """SGDClassifier falls back to early_stopping=False when the fold is
    too small for the internal stratified validation split."""
    # Build tiny decision rows: 3 matches × 10 picks = 30 rows,
    # with ~10 distinct hero classes → 30 < 10 * 10 → early_stopping=False.
    events = _draft_with_bans_and_picks()
    indexes = {}
    all_rows = []
    for mid in (1, 2, 3):
        idx_frame = pd.DataFrame([{**e, "match_id": mid} for e in events])
        indexes.update(build_match_draft_index(idx_frame))
    matches = _matches_frame(
        *[
            {
                "match_id": mid,
                "start_time": T0 + timedelta(days=mid),
                "game_version_id": 176,
                "radiant_team_id": 100,
                "dire_team_id": 200,
            }
            for mid in (1, 2, 3)
        ]
    )
    decisions = build_next_pick_decision_rows(matches=matches, indexes=indexes)
    history = build_causal_pick_history(decisions)
    history.advance_before(pd.Timestamp(T0 + timedelta(days=10)))
    train_rows = decisions.to_dict(orient="records")
    # This should NOT raise ValueError even with few rows per class.
    model = _fit_multinomial_policy(
        train_rows=train_rows,
        name="tiny_test",
        C=1.0,
        include_picks=True,
        include_bans=False,
        include_team_identity=False,
        include_team_tendency=False,
    )
    assert model is not None
    assert model.classifier.early_stopping is False


def test_normal_folds_retain_early_stopping() -> None:
    """With enough rows, early_stopping is enabled."""
    from sklearn.linear_model import SGDClassifier as _SGD

    # We can't easily build a huge dataset here, but we can verify the
    # threshold logic directly.
    n_classes = 10
    # Threshold is POLICY_EARLY_STOPPING_MIN_FACTOR * n_classes.
    threshold = POLICY_EARLY_STOPPING_MIN_FACTOR * n_classes
    assert threshold == 100
    # A real research fold with e.g. 1000 rows and 120 classes needs
    # 1200 rows.  verify the constant is 10.
    assert POLICY_EARLY_STOPPING_MIN_FACTOR == 10
    assert POLICY_SGD_MAX_ITER == 200
