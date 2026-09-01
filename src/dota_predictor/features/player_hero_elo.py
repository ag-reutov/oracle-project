"""Leakage-safe Elo-adjusted Player × Hero performance state (Slice 10).

Grain
-----
One row per current ``(match_id, player_id, hero_id)``: the player on
their currently drafted hero, with *performance relative to pre-match
team Elo* computed from strictly earlier appearances of that same
player×hero. This is historical state keyed by the current draft, not
a dense player×hero grid and not a training feature.

Lifetime Player × Hero *volume* is not strength
-----------------------------------------------
Slice 0 career Player × Hero counts (games on the drafted hero) remain
available as evidence of sample size. They are **not** interpreted here
as positive strength. Strength is whether those appearances beat the
player's team's pre-match Elo expected win probability. Prior-games
volume enters only as the sample size that shrinks that residual toward
zero.

This module does not replace ``features.player_hero``. Existing Slice
0–9 code, Career volume columns, and production ``FEATURE_COLUMNS`` are
unchanged. Slice 10 is not added to any win-model specification.

Residual
--------
For a historical appearance ``h`` of player ``P`` on hero ``H``:

* ``y`` = 1 if ``P``'s side won ``h``, else 0.
* ``e`` = Elo expected win for ``P``'s team in ``h``, from the
  **pre-match** team-Elo snapshot (same sequential replay as
  ``compute_team_elo_features``). Same-``start_time`` matches are
  mutually blind in Elo and in this window.
* Residual contribution = ``y - e``.

At current match ``c``, sums run over ``h.start_time < c.start_time``
(``RANGE ... EXCLUDE GROUP``). The current match's own result and its
own Elo update never enter that row.

    prior_games
    prior_wins                  sum of y
    prior_elo_expected_wins     sum of e
    prior_wins_minus_expected   sum(y) - sum(e)
    mean_outcome_residual       (sum(y)-sum(e)) / n   NULL if n = 0
    shrunk_outcome_residual     (n / (n + k)) * mean  0 if n = 0
    shrinkage_weight            n / (n + k)           0 if n = 0

Shrinkage
---------
Empirical-Bayes / precision-weighted shrinkage of the mean residual
toward **zero** (league-average residual under a calibrated Elo book):

    shrunk = (n / (n + k)) * mean_residual

``k = σ² / τ²`` with a statistical prior, not a metric search:

* ``σ² ≈ 0.24`` — Bernoulli residual variance ``E[e(1-e)]`` near
  professional Elo expected values.
* ``τ ≈ 0.08`` — an 8 percentage-point true player×hero residual is
  already a large persistent effect.
* ``k = 0.24 / 0.08² = 37.5``, rounded to ``DEFAULT_SHRINKAGE_K = 40``.

``k`` is a frozen state-layer constant. It is not estimated from TI
2026, not chosen to minimize a holdout loss, and not a feature gate.
Volume ``n`` only scales how far the residual may move away from zero.

Temporal integrity
------------------
Every historical match ``h`` contributing to current match ``c``
satisfies ``h.start_time < c.start_time``. Equal timestamps are
mutually blind. History is never ordered by ``match_id``. Elo expected
wins use the pre-match snapshot of ``h``, never ``h``'s own rating
update and never ``c``'s outcome.

This layer does not expose in-game events, duration, gold/xp, or the
current match result. It is not part of ``FEATURE_COLUMNS`` or PRE_DRAFT
snapshot SQL. It never writes Parquet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    HEROES_VIEW,
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.team_elo import (
    DEFAULT_ELO_CONFIG,
    DIRE_TEAM_ELO_COLUMN,
    MATCH_ID_COLUMN,
    RADIANT_TEAM_ELO_COLUMN,
    EloConfig,
    compute_team_elo_features,
    expected_score,
)
from dota_predictor.features.temporal import STRICT_PRIOR_RANGE_SQL

__all__ = [
    "DEFAULT_SHRINKAGE_K",
    "EVIDENCE_METRIC_COLUMNS",
    "MATCH_ID_COLUMN",
    "PLAYER_HERO_ELO_COLUMNS",
    "PLAYER_HERO_ELO_IDENTITY_COLUMNS",
    "PLAYER_HERO_ELO_METRIC_COLUMNS",
    "RESIDUAL_VARIANCE_PRIOR",
    "STRENGTH_METRIC_COLUMNS",
    "TEAM_ELO_EXPECTED_VIEW",
    "TRUE_EFFECT_SD_PRIOR",
    "PlayerHeroEloState",
    "apply_residual_shrinkage",
    "build_player_hero_elo",
    "match_elo_expected_wins",
    "player_hero_elo_sql",
    "shrinkage_weight",
    "shrunk_residual",
]


PLAYER_ID_COLUMN = "player_id"
HERO_ID_COLUMN = "hero_id"
HERO_NAME_COLUMN = "hero_name"
TEAM_ELO_EXPECTED_VIEW = "team_elo_expected"

# Statistical prior for k = σ² / τ². Not estimated from TI 2026.
RESIDUAL_VARIANCE_PRIOR = 0.24
TRUE_EFFECT_SD_PRIOR = 0.08
DEFAULT_SHRINKAGE_K = 40.0

PLAYER_HERO_ELO_IDENTITY_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "start_time",
    "game_version_id",
    PLAYER_ID_COLUMN,
    HERO_ID_COLUMN,
    HERO_NAME_COLUMN,
    "side",
    "team_id",
    "slot_in_side",
)

PLAYER_HERO_ELO_METRIC_COLUMNS: tuple[str, ...] = (
    "prior_games_on_hero",
    "prior_wins_on_hero",
    "prior_elo_expected_wins_on_hero",
    "prior_wins_minus_expected_on_hero",
    "mean_outcome_residual_on_hero",
    "shrunk_outcome_residual_on_hero",
    "shrinkage_weight_on_hero",
)

# Volume / confidence. Not strength.
EVIDENCE_METRIC_COLUMNS: tuple[str, ...] = (
    "prior_games_on_hero",
    "shrinkage_weight_on_hero",
)

# Elo-adjusted performance. Volume is not among these.
STRENGTH_METRIC_COLUMNS: tuple[str, ...] = (
    "prior_wins_on_hero",
    "prior_elo_expected_wins_on_hero",
    "prior_wins_minus_expected_on_hero",
    "mean_outcome_residual_on_hero",
    "shrunk_outcome_residual_on_hero",
)

PLAYER_HERO_ELO_COLUMNS: tuple[str, ...] = (
    PLAYER_HERO_ELO_IDENTITY_COLUMNS + PLAYER_HERO_ELO_METRIC_COLUMNS
)


def shrinkage_weight(prior_games: float, *, k: float = DEFAULT_SHRINKAGE_K) -> float:
    """Evidence fraction ``n / (n + k)``. Zero when ``n = 0``."""
    if k <= 0.0:
        raise ValueError(f"shrinkage k must be positive, got {k}")
    n = float(prior_games)
    if n <= 0.0:
        return 0.0
    return n / (n + k)


def shrunk_residual(
    mean_residual: float | None,
    prior_games: float,
    *,
    k: float = DEFAULT_SHRINKAGE_K,
) -> float:
    """Precision-weighted residual toward zero.

    ``n = 0`` or a NULL mean is no evidence: return 0.0, not a fabricated
    win rate and not a volume bonus.
    """
    weight = shrinkage_weight(prior_games, k=k)
    if weight == 0.0 or mean_residual is None or (
        isinstance(mean_residual, float) and not np.isfinite(mean_residual)
    ):
        return 0.0
    return weight * float(mean_residual)


def apply_residual_shrinkage(
    frame: pd.DataFrame, *, k: float = DEFAULT_SHRINKAGE_K
) -> pd.DataFrame:
    """Add ``shrunk_outcome_residual_on_hero`` and ``shrinkage_weight_on_hero``.

    Does not impute ``mean_outcome_residual_on_hero``. Cold-start rows
    keep a NULL mean and a zero shrunk residual.
    """
    if k <= 0.0:
        raise ValueError(f"shrinkage k must be positive, got {k}")
    out = frame.copy()
    games = pd.to_numeric(out["prior_games_on_hero"], errors="coerce").fillna(0.0)
    mean = pd.to_numeric(out["mean_outcome_residual_on_hero"], errors="coerce")
    weight = games / (games + k)
    weight = weight.mask(games <= 0.0, 0.0)
    shrunk = weight * mean
    out["shrinkage_weight_on_hero"] = weight.to_numpy(dtype=float)
    out["shrunk_outcome_residual_on_hero"] = shrunk.fillna(0.0).to_numpy(dtype=float)
    return out


def match_elo_expected_wins(
    matches: pd.DataFrame, *, config: EloConfig = DEFAULT_ELO_CONFIG
) -> pd.DataFrame:
    """Pre-match Elo expected win for each side of every match.

    Uses ``compute_team_elo_features`` snapshots. A match's own outcome
    never enters its expected-win row. Same-``start_time`` matches share
    the pre-group ratings.
    """
    elo = compute_team_elo_features(matches, config=config)
    radiant = elo[RADIANT_TEAM_ELO_COLUMN].to_numpy(dtype=float)
    dire = elo[DIRE_TEAM_ELO_COLUMN].to_numpy(dtype=float)
    radiant_expected = np.array(
        [expected_score(float(r), float(d)) for r, d in zip(radiant, dire, strict=True)],
        dtype=float,
    )
    return pd.DataFrame(
        {
            MATCH_ID_COLUMN: elo[MATCH_ID_COLUMN].to_numpy(),
            "radiant_expected_win": radiant_expected,
            "dire_expected_win": 1.0 - radiant_expected,
            RADIANT_TEAM_ELO_COLUMN: radiant,
            DIRE_TEAM_ELO_COLUMN: dire,
        }
    )


def _hero_name_select(*, catalog_registered: bool) -> str:
    if catalog_registered:
        return f"{HEROES_VIEW}.name AS hero_name"
    return "CAST(NULL AS VARCHAR) AS hero_name"


def _hero_name_join(*, catalog_registered: bool) -> str:
    if catalog_registered:
        return f"LEFT JOIN {HEROES_VIEW} ON {HEROES_VIEW}.hero_id = w.hero_id"
    return ""


def player_hero_elo_sql(
    *, catalog_registered: bool = True, match_id: int | None = None
) -> str:
    """SQL for leakage-safe Elo-adjusted player×hero sums and mean residual.

    Requires ``team_elo_expected`` to be registered on the same
    connection (see ``build_player_hero_elo``). Windows implement
    ``historical.start_time < current.start_time`` including
    same-timestamp blindness. Shrinkage is applied after materialization
    so ``k`` stays a Python constant, not a SQL-tuned cutoff.

    ``slot_in_side`` is lobby identity only and is never a window
    partition. History is keyed by ``player_id`` × ``hero_id``; changing
    ``team_id`` does not reset it.
    """
    output_filter = ""
    if match_id is not None:
        output_filter = f"WHERE match_id = {int(match_id)}"
    hero_name = _hero_name_select(catalog_registered=catalog_registered)
    hero_join = _hero_name_join(catalog_registered=catalog_registered)
    mean_residual = (
        "CASE WHEN w.prior_games_on_hero > 0 "
        "THEN w.prior_wins_minus_expected_on_hero "
        "/ w.prior_games_on_hero::DOUBLE "
        "ELSE NULL END"
    )
    return f"""
WITH appearances AS (
    SELECT
        mp.match_id,
        mp.start_time,
        m.game_version_id,
        mp.player_id,
        mp.hero_id,
        mp.side,
        mp.team_id,
        mp.slot_in_side,
        CASE
            WHEN mp.side = 'RADIANT' THEN m.radiant_win
            ELSE NOT m.radiant_win
        END AS player_won,
        CASE
            WHEN mp.side = 'RADIANT' THEN e.radiant_expected_win
            ELSE e.dire_expected_win
        END AS elo_expected_win
    FROM {MATCH_PLAYERS_VIEW} mp
    JOIN {MATCHES_VIEW} m ON m.match_id = mp.match_id
    JOIN {TEAM_ELO_EXPECTED_VIEW} e ON e.match_id = mp.match_id
),

flagged AS (
    SELECT
        match_id,
        start_time,
        game_version_id,
        player_id,
        hero_id,
        side,
        team_id,
        slot_in_side,
        CASE WHEN player_won THEN 1.0 ELSE 0.0 END AS was_win,
        elo_expected_win
    FROM appearances
),

windowed AS (
    SELECT
        match_id,
        start_time,
        game_version_id,
        player_id,
        hero_id,
        side,
        team_id,
        slot_in_side,
        COALESCE(COUNT(*) OVER w_ph, 0)::BIGINT AS prior_games_on_hero,
        COALESCE(SUM(was_win) OVER w_ph, 0)::DOUBLE AS prior_wins_on_hero,
        COALESCE(SUM(elo_expected_win) OVER w_ph, 0)::DOUBLE
            AS prior_elo_expected_wins_on_hero,
        (
            COALESCE(SUM(was_win) OVER w_ph, 0)::DOUBLE
            - COALESCE(SUM(elo_expected_win) OVER w_ph, 0)::DOUBLE
        ) AS prior_wins_minus_expected_on_hero
    FROM flagged
    WINDOW
        w_ph AS (
            PARTITION BY player_id, hero_id
            ORDER BY start_time
            {STRICT_PRIOR_RANGE_SQL}
        )
)

SELECT
    w.match_id,
    w.start_time,
    w.game_version_id,
    w.player_id,
    w.hero_id,
    {hero_name},
    w.side,
    w.team_id,
    w.slot_in_side,
    w.prior_games_on_hero,
    w.prior_wins_on_hero,
    w.prior_elo_expected_wins_on_hero,
    w.prior_wins_minus_expected_on_hero,
    {mean_residual} AS mean_outcome_residual_on_hero
FROM windowed AS w
{hero_join}
{output_filter}
"""


def _heroes_view_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return HEROES_VIEW in tables


def _register_team_elo_expected(
    store: FeatureDuckDBConnection, *, config: EloConfig
) -> pd.DataFrame:
    matches = store.sql(
        f"""
        SELECT
            match_id,
            start_time,
            radiant_team_id,
            dire_team_id,
            radiant_win
        FROM {MATCHES_VIEW}
        """
    ).df()
    expected = match_elo_expected_wins(matches, config=config)
    store.connection.register(TEAM_ELO_EXPECTED_VIEW, expected)
    return expected


@dataclass(frozen=True)
class PlayerHeroEloState:
    """Materialized ``(match_id, player_id, hero_id)`` Elo-adjusted state."""

    frame: pd.DataFrame
    shrinkage_k: float

    def to_frame(self) -> pd.DataFrame:
        """One row per current player×hero in ``PLAYER_HERO_ELO_COLUMNS`` order."""
        ordered = self.frame[list(PLAYER_HERO_ELO_COLUMNS)]
        return ordered.sort_values(
            [MATCH_ID_COLUMN, PLAYER_ID_COLUMN, HERO_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)


def build_player_hero_elo(
    store: FeatureDuckDBConnection,
    *,
    match_id: int | None = None,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> PlayerHeroEloState:
    """Build Elo-adjusted player×hero state from registered analytical views.

    Registers a temporary ``team_elo_expected`` relation from the
    production Elo replay. Independent of ``build_pre_draft_snapshot``.
    Optional ``match_id`` filters output after windows run over full
    history. ``shrinkage_k`` defaults to the frozen statistical prior.
    """
    if shrinkage_k <= 0.0:
        raise ValueError(f"shrinkage k must be positive, got {shrinkage_k}")
    _register_team_elo_expected(store, config=elo_config)
    sql = player_hero_elo_sql(
        catalog_registered=_heroes_view_registered(store),
        match_id=match_id,
    )
    raw = store.sql(sql).df()
    shrunk = apply_residual_shrinkage(raw, k=shrinkage_k)
    return PlayerHeroEloState(frame=shrunk, shrinkage_k=float(shrinkage_k))
