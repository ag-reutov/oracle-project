"""Development-only Slice 10 diagnostics: TRAIN/history semantics for k."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from dota_predictor.features.player_hero_elo import DEFAULT_SHRINKAGE_K
from dota_predictor.training.player_hero_elo_diagnostics import (
    MIN_GAMES_FOR_K_ESTIMATE,
    estimate_shrinkage_k,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END


def _cell_rows(
    *,
    player_id: int,
    hero_id: int,
    start_time: datetime,
    n: int,
    y: float,
    expected: float,
) -> list[dict[str, object]]:
    return [
        {
            "match_id": player_id * 1000 + hero_id * 10 + i,
            "start_time": start_time,
            "game_version_id": 176,
            "player_id": player_id,
            "hero_id": hero_id,
            "side": "RADIANT",
            "team_id": 1,
            "y": y,
            "elo_expected_win": expected,
        }
        for i in range(n)
    ]


def test_estimate_shrinkage_k_excludes_timestamps_after_development_end() -> None:
    end = FROZEN_DEVELOPMENT_END
    before = end - timedelta(days=10)
    after = end + timedelta(days=30)
    development_rows: list[dict[str, object]] = []
    for player_id in range(1, 12):
        development_rows.extend(
            _cell_rows(
                player_id=player_id,
                hero_id=1,
                start_time=before,
                n=MIN_GAMES_FOR_K_ESTIMATE,
                y=1.0 if player_id % 2 == 0 else 0.0,
                expected=0.5,
            )
        )
    future_rows = _cell_rows(
        player_id=99,
        hero_id=99,
        start_time=after,
        n=40,
        y=1.0,
        expected=0.1,
    )
    mixed = pd.DataFrame(development_rows + future_rows)
    development_only = pd.DataFrame(development_rows)

    mixed_estimate = estimate_shrinkage_k(mixed, development_end=end)
    development_estimate = estimate_shrinkage_k(
        development_only, development_end=end
    )
    with_future_as_development = estimate_shrinkage_k(
        mixed, development_end=after
    )

    assert mixed_estimate.used_for_state is False
    assert mixed_estimate.development_end == end
    assert mixed_estimate.min_games_for_cell == MIN_GAMES_FOR_K_ESTIMATE
    assert mixed_estimate.n_cells == development_estimate.n_cells
    assert mixed_estimate.n_appearances == development_estimate.n_appearances
    assert mixed_estimate.k == pytest.approx(development_estimate.k)
    assert with_future_as_development.n_cells == mixed_estimate.n_cells + 1
    assert with_future_as_development.n_appearances > mixed_estimate.n_appearances
    assert with_future_as_development.k != pytest.approx(mixed_estimate.k)
    assert DEFAULT_SHRINKAGE_K == 40.0


def test_estimate_shrinkage_k_never_uses_holdout_to_set_state_k() -> None:
    end = FROZEN_DEVELOPMENT_END
    before = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for player_id in range(1, 10):
        rows.extend(
            _cell_rows(
                player_id=player_id,
                hero_id=2,
                start_time=before,
                n=10,
                y=0.0,
                expected=0.5,
            )
        )
    estimate = estimate_shrinkage_k(pd.DataFrame(rows), development_end=end)
    assert estimate.used_for_state is False
    assert estimate.development_end == end
