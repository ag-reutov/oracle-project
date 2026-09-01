"""Fixture builders for expected-position tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from player_position_helpers import (
    assign_positions,
    draft_and_player_rows,
    player_position_state_frame,
)

from dota_predictor.features.expected_position import assign_expected_positions

__all__ = [
    "assign_expected_positions_frame",
    "assign_positions",
    "draft_and_player_rows",
    "player_position_state_frame",
]


def assign_expected_positions_frame(
    tmp_path: Path,
    *,
    matches: list[dict[str, Any]],
    players: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    method: str,
    match_id: int | None = None,
) -> pd.DataFrame:
    """Build Slice 2 state then jointly assign expected positions."""
    state = player_position_state_frame(
        tmp_path,
        matches=matches,
        players=players,
        drafts=drafts,
        match_id=match_id,
    )
    return assign_expected_positions(state, method=method)
