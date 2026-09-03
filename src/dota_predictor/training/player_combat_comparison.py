"""Slice 19: leakage-safe pre-draft player-combat team comparison.

Converts frozen Slice 18 player state into one Radiant − Dire row per
match. Shrinkage ``k`` is the frozen Slice 18 constant; this module
does not re-search it and does not use match outcomes to choose it.

Research / evaluation plumbing only. Does not add columns to production
``FEATURE_COLUMNS``, does not train a win model, and does not score the
frozen holdout. ``SLICE19_FROZEN_SPECS`` is a frozen *benchmark spec*
for Slice 20, not a production-feature freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    MATCH_PLAYERS_VIEW,
    MATCHES_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.player_combat_comparison import (
    COMBAT_CAUSAL_C_COLUMN,
    PLAYER_COMBAT_COMPARISON_EVIDENCE_COLUMNS,
    PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS,
    PLAYER_COMBAT_FEATURE_COLUMNS,
    PLAYER_COMBAT_REQUIRED_COLUMNS,
    PLAYER_COMBAT_STATE_FEATURE_COLUMNS,
    diagnose_combat_roster,
    match_combat_roster_flags,
    player_combat_comparison_from_players,
)
from dota_predictor.features.player_farming_comparison import (
    PLAYER_FARMING_FEATURE_COLUMNS,
    player_farming_comparison_from_players,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.features.team_elo import (
    DEFAULT_ELO_CONFIG,
    TEAM_ELO_DELTA_COLUMN,
    TEAM_ELO_FEATURE_COLUMNS,
    EloConfig,
    compute_team_elo_features,
)
from dota_predictor.training.combat_performance_target import (
    COMBAT_C_POSITION,
    FROZEN_COMBAT_CANDIDATE,
)
from dota_predictor.training.farming_performance_target import CANDIDATE_B
from dota_predictor.training.feature_sets import (
    ALL_FEATURE_COLUMNS,
    POST_DRAFT_BLOCK_ABLATION_SPECS,
    SLICE9_FROZEN_SPECS,
    SLICE19_CANDIDATE_SPEC,
    SLICE19_FROZEN_SPECS,
)
from dota_predictor.training.player_combat_state import (
    FROZEN_COMBAT_SHRINKAGE_K,
    SLICE18_STATE_COLUMNS,
    attach_player_combat_state,
)
from dota_predictor.training.player_farming_state import (
    FROZEN_CANDIDATE_B,
    FROZEN_SHRINKAGE_K,
    attach_player_farming_state,
)
from dota_predictor.training.player_performance_target import (
    BOX_SCORE_COLUMNS,
    _jsonable_value,
    _numeric,
    _pearson,
    _spearman,
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
    "Slice19DiagnosticReport",
    "build_player_combat_comparison",
    "build_player_combat_state",
    "run_player_combat_comparison_diagnostics",
    "slice19_report_to_jsonable",
]


EXPECTED_DEVELOPMENT_MATCHES = FROZEN_DEVELOPMENT_MATCH_COUNT
_DISTRIBUTION_COLUMNS: tuple[str, ...] = PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS


@dataclass(frozen=True)
class Slice19DiagnosticReport:
    development_end: datetime
    n_development_matches: int
    n_development_player_rows: int
    n_holdout_excluded: int
    frozen_combat_k: float
    frozen_farming_k: float
    coverage: pd.DataFrame
    roster_integrity: dict[str, object]
    cold_start_radiant_distribution: pd.DataFrame
    cold_start_dire_distribution: pd.DataFrame
    prior_n_distribution: pd.DataFrame
    feature_distribution: pd.DataFrame
    farming_relationship: pd.DataFrame
    farming_joint_quartiles: pd.DataFrame
    elo_relationship: pd.DataFrame
    integrity: dict[str, object]


def build_player_combat_state(
    appearances: pd.DataFrame, *, k: float = FROZEN_COMBAT_SHRINKAGE_K
) -> pd.DataFrame:
    """Attach frozen Slice 18 combat state. Does not re-search ``k``."""
    return attach_player_combat_state(appearances, k=k)


def build_player_combat_comparison(
    store: FeatureDuckDBConnection,
    *,
    k: float = FROZEN_COMBAT_SHRINKAGE_K,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
    development_end: datetime | None = None,
) -> pd.DataFrame:
    """Match-level PRE_DRAFT combat comparison from canonical appearances.

    When ``development_end`` is set, later matches are dropped *before*
    state is attached so holdout box scores cannot enter the position
    baseline or player history. Default is the full store (still
    leakage-safe: each row uses only ``start_time < M``).
    """
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    if development_end is not None:
        appearances = restrict_development(
            appearances, development_end=utc_datetime(development_end)
        )
    state = build_player_combat_state(appearances, k=k)
    return player_combat_comparison_from_players(state)


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _distribution_row(column: str, values: pd.Series) -> dict[str, object]:
    finite = _numeric(values).to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "column": column,
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "column": column,
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": _std(finite),
        "median": float(np.median(finite)),
        "p05": _quantile(finite, 0.05),
        "p25": _quantile(finite, 0.25),
        "p75": _quantile(finite, 0.75),
        "p95": _quantile(finite, 0.95),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _count_distribution(counts: pd.Series, *, side: str) -> pd.DataFrame:
    value_counts = counts.value_counts(dropna=False).sort_index()
    rows = [
        {"side": side, "cold_start_count": int(key), "n_matches": int(value)}
        for key, value in value_counts.items()
    ]
    return pd.DataFrame(rows)


def _coverage_table(
    state: pd.DataFrame,
    comparison: pd.DataFrame,
    flags: pd.DataFrame,
) -> pd.DataFrame:
    prior_n = pd.to_numeric(state["combat_prior_n"], errors="coerce").fillna(0)
    feature = _numeric(comparison["mean_combat_shrunk_c_diff"])
    n_matches = int(state["match_id"].nunique()) if len(state) else 0
    n_complete_roster = int(flags["complete_roster"].sum()) if len(flags) else 0
    n_complete_join = int(flags["complete_state_join"].sum()) if len(flags) else 0
    n_any_cold = int(flags["any_cold_start"].sum()) if len(flags) else 0
    n_no_cold = int((~flags["any_cold_start"]).sum()) if len(flags) else 0
    return pd.DataFrame(
        [
            {
                "n_development_matches": n_matches,
                "n_complete_10_player_roster": n_complete_roster,
                "n_complete_combat_state_join": n_complete_join,
                "n_matches_any_cold_start": n_any_cold,
                "n_matches_no_cold_start": n_no_cold,
                "n_cold_start_player_appearances": int((prior_n == 0).sum()),
                "n_player_rows": len(state),
                "n_players": int(state["player_id"].nunique()) if len(state) else 0,
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
    return pd.DataFrame(
        [
            _distribution_row(column, comparison[column])
            for column in _DISTRIBUTION_COLUMNS
        ]
    )


def _history_accumulates(state: pd.DataFrame) -> dict[str, object]:
    """Repeated appearances must not reuse a player's final-period value."""
    if state.empty:
        return {
            "n_repeat_players": 0,
            "n_repeat_players_with_changing_state": 0,
            "history_accumulates": True,
        }
    grouped = state.groupby("player_id", sort=False)["combat_shrunk_c"]
    n_appearances = grouped.size()
    n_unique_state = grouped.nunique(dropna=False)
    repeats = n_appearances >= 2
    changing = repeats & (n_unique_state >= 2)
    n_repeat = int(repeats.sum())
    n_changing = int(changing.sum())
    return {
        "n_repeat_players": n_repeat,
        "n_repeat_players_with_changing_state": n_changing,
        "history_accumulates": n_repeat == 0 or n_changing > 0,
    }


def _relationship_row(
    *,
    name: str,
    left: pd.Series,
    right: pd.Series,
) -> dict[str, object]:
    return {
        "pair": name,
        "n": int((_numeric(left).notna() & _numeric(right).notna()).sum()),
        "pearson": _pearson(left, right),
        "spearman": _spearman(left, right),
        "left_mean": (
            float(_numeric(left).mean()) if left.notna().any() else float("nan")
        ),
        "right_mean": float(_numeric(right).mean())
        if right.notna().any()
        else float("nan"),
    }


def _joint_quartiles(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    pair = pd.DataFrame({"combat": _numeric(left), "farming": _numeric(right)}).dropna()
    if len(pair) < 4:
        return pd.DataFrame(columns=["combat_quartile", "farming_quartile", "n"])
    combat_q = pd.qcut(
        pair["combat"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    farming_q = pd.qcut(
        pair["farming"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    counts = (
        pd.crosstab(combat_q, farming_q)
        .rename_axis(index="combat_quartile", columns="farming_quartile")
        .stack()
        .rename("n")
        .reset_index()
    )
    return counts


def _load_development_matches(
    store: FeatureDuckDBConnection, *, development_end: datetime
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
    return restrict_development(matches, development_end=development_end)


def run_player_combat_comparison_diagnostics(
    store: FeatureDuckDBConnection,
    *,
    development_end: datetime | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> Slice19DiagnosticReport:
    """Development-only Slice 19 feature construction. Does not train a model."""
    end = utc_datetime(
        development_end if development_end is not None else FROZEN_DEVELOPMENT_END
    )
    appearances = build_player_performance_frame(store, elo_config=elo_config)
    stamp = pd.to_datetime(appearances["start_time"], utc=True)
    holdout = appearances.loc[stamp > pd.Timestamp(end)]
    development = restrict_development(appearances, development_end=end)
    combat_state = build_player_combat_state(development, k=FROZEN_COMBAT_SHRINKAGE_K)
    comparison = player_combat_comparison_from_players(combat_state)
    flags = match_combat_roster_flags(combat_state)
    roster = diagnose_combat_roster(combat_state)
    history = _history_accumulates(combat_state)

    farming_state = attach_player_farming_state(development, k=FROZEN_SHRINKAGE_K)
    farming = player_farming_comparison_from_players(farming_state)
    joined = comparison.merge(
        farming.loc[:, ["match_id", *PLAYER_FARMING_FEATURE_COLUMNS]],
        on="match_id",
        how="inner",
        validate="one_to_one",
    )
    matches = _load_development_matches(store, development_end=end)
    elo = compute_team_elo_features(matches, config=elo_config)
    with_elo = comparison.merge(elo, on="match_id", how="inner", validate="one_to_one")

    view_columns = store.relation(MATCH_PLAYERS_VIEW).columns
    later_than_end = bool(
        (pd.to_datetime(development["start_time"], utc=True) > pd.Timestamp(end)).any()
    )
    extra = set(SLICE19_CANDIDATE_SPEC.feature_columns) - set(TEAM_ELO_FEATURE_COLUMNS)
    integrity = {
        "development_end": end.isoformat(),
        "frozen_combat_candidate": FROZEN_COMBAT_CANDIDATE,
        "slice17_candidate_unchanged": FROZEN_COMBAT_CANDIDATE == COMBAT_C_POSITION,
        "frozen_combat_shrinkage_k": FROZEN_COMBAT_SHRINKAGE_K,
        "combat_k_is_20": FROZEN_COMBAT_SHRINKAGE_K == 20.0,
        "k_re_searched": False,
        "farming_candidate_b": FROZEN_CANDIDATE_B,
        "farming_candidate_b_unchanged": FROZEN_CANDIDATE_B == CANDIDATE_B,
        "farming_frozen_shrinkage_k": FROZEN_SHRINKAGE_K,
        "farming_k_is_5": FROZEN_SHRINKAGE_K == 5.0,
        "alternative_combat_aggregation_searched": False,
        "ti2026_used_for_k": False,
        "holdout_used_for_k": False,
        "holdout_used_for_feature": False,
        "holdout_rows_in_development": later_than_end,
        "stratz_called": False,
        "ingestion_modified": False,
        "schema_modified": False,
        "box_scores_in_feature_match_players_view": any(
            column in view_columns for column in BOX_SCORE_COLUMNS
        ),
        "causal_c_in_required_columns": (
            COMBAT_CAUSAL_C_COLUMN in PLAYER_COMBAT_REQUIRED_COLUMNS
        ),
        "hero_id_in_required_columns": "hero_id" in PLAYER_COMBAT_REQUIRED_COLUMNS,
        "position_in_required_columns": "position" in PLAYER_COMBAT_REQUIRED_COLUMNS,
        "state_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in SLICE18_STATE_COLUMNS
        ),
        "comparison_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in PLAYER_COMBAT_COMPARISON_METRIC_COLUMNS
        ),
        "candidate_in_feature_columns": any(
            name in FEATURE_COLUMNS for name in PLAYER_COMBAT_FEATURE_COLUMNS
        ),
        "evidence_in_candidate_spec": any(
            name in SLICE19_CANDIDATE_SPEC.feature_columns
            for name in PLAYER_COMBAT_COMPARISON_EVIDENCE_COLUMNS
        ),
        "state_feature_columns_exclude_causal_c": (
            COMBAT_CAUSAL_C_COLUMN not in PLAYER_COMBAT_STATE_FEATURE_COLUMNS
        ),
        "feature_columns_unchanged_length": len(FEATURE_COLUMNS) == 33,
        "all_feature_columns_is_feature_columns": list(ALL_FEATURE_COLUMNS)
        == list(FEATURE_COLUMNS),
        "slice9_frozen_spec_count": len(SLICE9_FROZEN_SPECS),
        "slice19_frozen_spec_count": len(SLICE19_FROZEN_SPECS),
        "slice19_candidate_uses_frozen_feature": (
            PLAYER_COMBAT_FEATURE_COLUMNS[0] in SLICE19_CANDIDATE_SPEC.feature_columns
        ),
        "slice19_candidate_extra_columns": sorted(extra),
        "post_draft_block_ablation_spec_count": len(POST_DRAFT_BLOCK_ABLATION_SPECS),
        "player_rating_persisted": False,
        "player_combat_state_persisted": False,
        "player_combat_comparison_persisted": False,
        "model_trained": False,
        "win_model_benchmarked": False,
        "holdout_scored": False,
        "shrinkage_chosen_from_outcomes": False,
        "population_matches_expected": (
            int(development["match_id"].nunique()) == EXPECTED_DEVELOPMENT_MATCHES
        ),
        "n_holdout_excluded": len(holdout),
        **history,
    }
    return Slice19DiagnosticReport(
        development_end=end,
        n_development_matches=int(development["match_id"].nunique()),
        n_development_player_rows=len(development),
        n_holdout_excluded=len(holdout),
        frozen_combat_k=FROZEN_COMBAT_SHRINKAGE_K,
        frozen_farming_k=FROZEN_SHRINKAGE_K,
        coverage=_coverage_table(combat_state, comparison, flags),
        roster_integrity=roster,
        cold_start_radiant_distribution=_count_distribution(
            flags["radiant_cold_start_count"], side="RADIANT"
        ),
        cold_start_dire_distribution=_count_distribution(
            flags["dire_cold_start_count"], side="DIRE"
        ),
        prior_n_distribution=pd.DataFrame(
            [_distribution_row("combat_prior_n", combat_state["combat_prior_n"])]
        ),
        feature_distribution=_feature_distribution(comparison),
        farming_relationship=pd.DataFrame(
            [
                _relationship_row(
                    name="mean_combat_shrunk_c_diff vs mean_farming_shrunk_b_diff",
                    left=joined["mean_combat_shrunk_c_diff"],
                    right=joined[PLAYER_FARMING_FEATURE_COLUMNS[0]],
                )
            ]
        ),
        farming_joint_quartiles=_joint_quartiles(
            joined["mean_combat_shrunk_c_diff"],
            joined[PLAYER_FARMING_FEATURE_COLUMNS[0]],
        ),
        elo_relationship=pd.DataFrame(
            [
                _relationship_row(
                    name="mean_combat_shrunk_c_diff vs team_elo_delta",
                    left=with_elo["mean_combat_shrunk_c_diff"],
                    right=with_elo[TEAM_ELO_DELTA_COLUMN],
                )
            ]
        ),
        integrity=integrity,
    )


def slice19_report_to_jsonable(report: Slice19DiagnosticReport) -> dict[str, object]:
    """JSON-safe dump of the development-only Slice 19 report."""
    return {
        "development_end": report.development_end.isoformat(),
        "n_development_matches": report.n_development_matches,
        "n_development_player_rows": report.n_development_player_rows,
        "n_holdout_excluded": report.n_holdout_excluded,
        "frozen_combat_k": report.frozen_combat_k,
        "frozen_farming_k": report.frozen_farming_k,
        "coverage": _jsonable_value(report.coverage),
        "roster_integrity": _jsonable_value(report.roster_integrity),
        "cold_start_radiant_distribution": _jsonable_value(
            report.cold_start_radiant_distribution
        ),
        "cold_start_dire_distribution": _jsonable_value(
            report.cold_start_dire_distribution
        ),
        "prior_n_distribution": _jsonable_value(report.prior_n_distribution),
        "feature_distribution": _jsonable_value(report.feature_distribution),
        "farming_relationship": _jsonable_value(report.farming_relationship),
        "farming_joint_quartiles": _jsonable_value(report.farming_joint_quartiles),
        "elo_relationship": _jsonable_value(report.elo_relationship),
        "integrity": _jsonable_value(report.integrity),
    }
