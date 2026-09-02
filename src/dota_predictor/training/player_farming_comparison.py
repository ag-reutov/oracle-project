"""Slice 15: leakage-safe pre-draft player-farming team comparison.

Converts frozen Slice 14 player state into one Radiant − Dire row per
match. Shrinkage ``k`` is the frozen Slice 14 constant; this module
does not re-search it and does not use match outcomes to choose it.

Research / evaluation plumbing only. Does not add columns to production
``FEATURE_COLUMNS``, does not train a win model, and does not score the
frozen holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.player_farming_comparison import (
    FARMING_CAUSAL_B_COLUMN,
    PLAYER_FARMING_COMPARISON_METRIC_COLUMNS,
    PLAYER_FARMING_FEATURE_COLUMNS,
    PLAYER_FARMING_REQUIRED_COLUMNS,
    PLAYER_FARMING_STATE_FEATURE_COLUMNS,
    player_farming_comparison_from_players,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.features.team_elo import DEFAULT_ELO_CONFIG, EloConfig
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
    SLICE15_CANDIDATE_SPEC,
    SLICE15_FROZEN_SPECS,
)
from dota_predictor.training.player_farming_state import (
    FROZEN_SHRINKAGE_K,
    SLICE14_STATE_COLUMNS,
    attach_player_farming_state,
)
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    _jsonable_value,
    _numeric,
    _std,
    build_player_performance_frame,
    restrict_development,
)
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    FROZEN_DEVELOPMENT_MATCH_COUNT,
    utc_datetime,
)

__all__ = [
    "EXPECTED_DEVELOPMENT_MATCHES",
    "Slice15DiagnosticReport",
    "build_player_farming_comparison",
    "build_player_farming_state",
    "run_player_farming_comparison_diagnostics",
    "slice15_report_to_jsonable",
]


EXPECTED_DEVELOPMENT_MATCHES = FROZEN_DEVELOPMENT_MATCH_COUNT


@dataclass(frozen=True)
class Slice15DiagnosticReport:
    development_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    frozen_k: float
    coverage: pd.DataFrame
    feature_distribution: pd.DataFrame
    integrity: dict[str, object]


def build_player_farming_state(
    appearances: pd.DataFrame, *, k: float = FROZEN_SHRINKAGE_K
) -> pd.DataFrame:
    """Attach frozen Slice 14 farming state. Does not re-search ``k``."""
    return attach_player_farming_state(appearances, k=k)


def build_player_farming_comparison(
    store: FeatureDuckDBConnection,
    *,
    k: float = FROZEN_SHRINKAGE_K,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
    development_end: datetime | None = None,
) -> pd.DataFrame:
    """Match-level PRE_DRAFT farming comparison from canonical appearances.

    When ``development_end`` is set, later matches are dropped *before*
    state is attached so holdout box scores cannot enter residualizer
    or player history. Default is the full store (still leakage-safe:
    each row uses only ``start_time < M``).
    """
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    if development_end is not None:
        appearances = restrict_development(
            appearances, development_end=utc_datetime(development_end)
        )
    state = build_player_farming_state(appearances, k=k)
    return player_farming_comparison_from_players(state)


def _coverage_table(state: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    prior_n = pd.to_numeric(state["farming_prior_n"], errors="coerce").fillna(0)
    shrunk = _numeric(state["farming_shrunk_b"])
    feature = _numeric(comparison["mean_farming_shrunk_b_diff"])
    n_matches = len(comparison)
    return pd.DataFrame(
        [
            {
                "n_player_rows": len(state),
                "n_matches": n_matches,
                "n_players": int(state["player_id"].nunique()) if len(state) else 0,
                "n_prior_0": int((prior_n == 0).sum()),
                "n_prior_ge_1": int((prior_n >= 1).sum()),
                "n_nonzero_shrunk": int((shrunk.fillna(0.0) != 0.0).sum()),
                "n_feature_null": int(feature.isna().sum()),
                "n_feature_zero": int((feature == 0.0).sum()),
                "n_feature_nonzero": int((feature.notna() & (feature != 0.0)).sum()),
                "mean_abs_feature": (
                    float(np.abs(feature.dropna().to_numpy(dtype=float)).mean())
                    if feature.notna().any()
                    else float("nan")
                ),
            }
        ]
    )


def _feature_distribution(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in PLAYER_FARMING_COMPARISON_METRIC_COLUMNS:
        values = _numeric(comparison[column]).to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            rows.append(
                {
                    "column": column,
                    "n": 0,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "median": float("nan"),
                    "p05": float("nan"),
                    "p95": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "column": column,
                "n": int(finite.size),
                "mean": float(finite.mean()),
                "std": _std(finite),
                "median": float(np.median(finite)),
                "p05": float(np.quantile(finite, 0.05)),
                "p95": float(np.quantile(finite, 0.95)),
                "min": float(finite.min()),
                "max": float(finite.max()),
            }
        )
    return pd.DataFrame(rows)


def run_player_farming_comparison_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice15DiagnosticReport:
    """Development-only Slice 15 feature construction. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    state = build_player_farming_state(development, k=FROZEN_SHRINKAGE_K)
    comparison = player_farming_comparison_from_players(state)

    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    later_than_end = bool(
        (pd.to_datetime(development["start_time"], utc=True) > pd.Timestamp(end)).any()
    )
    integrity = {
        "development_end": end.isoformat(),
        "frozen_shrinkage_k": FROZEN_SHRINKAGE_K,
        "k_re_searched": False,
        "ti2026_used_for_k": False,
        "holdout_used_for_k": False,
        "holdout_used_for_feature": False,
        "holdout_rows_in_development": later_than_end,
        "stratz_called": False,
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "causal_b_in_comparison_columns": (
            FARMING_CAUSAL_B_COLUMN in PLAYER_FARMING_COMPARISON_METRIC_COLUMNS
        ),
        "causal_b_in_required_columns": (
            FARMING_CAUSAL_B_COLUMN in PLAYER_FARMING_REQUIRED_COLUMNS
        ),
        "hero_id_in_required_columns": "hero_id" in PLAYER_FARMING_REQUIRED_COLUMNS,
        "state_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in SLICE14_STATE_COLUMNS
        ),
        "comparison_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in PLAYER_FARMING_COMPARISON_METRIC_COLUMNS
        ),
        "candidate_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in PLAYER_FARMING_FEATURE_COLUMNS
        ),
        "comparison_in_all_feature_columns": any(
            name in ALL_FEATURE_COLUMNS
            for name in PLAYER_FARMING_COMPARISON_METRIC_COLUMNS
        ),
        "state_feature_columns_exclude_causal_b": (
            FARMING_CAUSAL_B_COLUMN not in PLAYER_FARMING_STATE_FEATURE_COLUMNS
        ),
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "slice15_frozen_spec_count": len(SLICE15_FROZEN_SPECS),
        "slice15_candidate_uses_frozen_feature": (
            PLAYER_FARMING_FEATURE_COLUMNS[0] in SLICE15_CANDIDATE_SPEC.feature_columns
        ),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "player_rating_persisted": False,
        "player_farming_state_persisted": False,
        "player_farming_comparison_persisted": False,
        "model_trained": False,
        "win_model_benchmarked": False,
        "shrinkage_chosen_from_outcomes": False,
        "population_matches_expected": (
            int(development["match_id"].nunique()) == EXPECTED_DEVELOPMENT_MATCHES
        ),
        "n_holdout_excluded": len(holdout),
    }
    return Slice15DiagnosticReport(
        development_end=end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        frozen_k=FROZEN_SHRINKAGE_K,
        coverage=_coverage_table(state, comparison),
        feature_distribution=_feature_distribution(comparison),
        integrity=integrity,
    )


def slice15_report_to_jsonable(report: Slice15DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 15 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "frozen_k": report.frozen_k,
        "coverage": _jsonable_value(report.coverage),
        "feature_distribution": _jsonable_value(report.feature_distribution),
        "integrity": _jsonable_value(report.integrity),
    }
