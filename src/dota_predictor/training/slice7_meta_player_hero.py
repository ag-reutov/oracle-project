"""Slice 7: walk-forward ablation of meta-aware Player × Hero features.

Evaluation only. Does not change Elo, walk-forward fold boundaries,
canonical data, Slices 0–6, or production ``FEATURE_COLUMNS``.

Career Player × Hero is the existing post-draft comparison block.
Slice 6 columns are aggregated with the same mean / min / zero-count
(for counts) and NULL-skipping mean (for rates) used by that block,
then Radiant − Dire.

Paired deltas are spec minus reference (negative = spec better).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from dota_predictor.features.duckdb_layer import (
    GAME_VERSIONS_VIEW,
    FeatureDuckDBConnection,
)
from dota_predictor.features.hero_state import POSITION_NUMBERS
from dota_predictor.features.patch_maturity import build_patch_maturity
from dota_predictor.features.player_hero_meta import build_player_hero_meta
from dota_predictor.features.player_hero_meta_comparison import (
    MATCH_ID_COLUMN,
    SLICE7_COMPARISON_COLUMNS,
    player_hero_meta_comparison_from_players,
)
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.training.dataset import ModelReadyDataset, TrainingDatasetError
from dota_predictor.training.feature_sets import (
    SLICE7_CAREER_SPEC_NAME,
    SLICE7_META_PLAYER_HERO_SPECS,
    BlockAblationSpec,
)
from dota_predictor.training.metrics import evaluate_probabilities
from dota_predictor.training.post_draft import build_post_draft_model_ready_dataset
from dota_predictor.training.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG,
    ELO_BLOCK_SPEC_NAME,
    WalkForwardConfig,
    WalkForwardReport,
    run_post_draft_walk_forward,
)

__all__ = [
    "CAREER_SAMPLE_BUCKET_ORDER",
    "COMPATIBILITY_BUCKET_ORDER",
    "CONTEST_SHIFT_ABS_DELTA",
    "LOW_N_VERSION_THRESHOLD",
    "MIN_EXPLICIT_FOR_ROLE_SHIFT",
    "PATCH_MATURITY_BIN_ORDER",
    "SLICE7_COMBINED_SPEC_NAME",
    "SLICE7_RECENT20_SPEC_NAME",
    "SLICE7_RECENT20_VOLUME_SPEC_NAME",
    "SLICE7_ROLE_SPEC_NAME",
    "SLICE7_SAME_VERSION_SPEC_NAME",
    "SLICE7_SAME_VERSION_VOLUME_SPEC_NAME",
    "Slice7Assembly",
    "Slice7BenchmarkReport",
    "assign_career_sample_bucket",
    "assign_compatibility_bucket",
    "assign_patch_maturity_bin",
    "build_slice7_model_ready_dataset",
    "describe_hero_shift_groups",
    "run_slice7_meta_player_hero_benchmark",
]

SLICE7_SAME_VERSION_VOLUME_SPEC_NAME = "logistic_elo_plus_same_version_volume"
SLICE7_SAME_VERSION_SPEC_NAME = (
    "logistic_elo_plus_same_version_volume_performance"
)
SLICE7_RECENT20_VOLUME_SPEC_NAME = "logistic_elo_plus_recent20_volume"
SLICE7_RECENT20_SPEC_NAME = "logistic_elo_plus_recent20_volume_performance"
SLICE7_ROLE_SPEC_NAME = "logistic_elo_plus_role_meta"
SLICE7_COMBINED_SPEC_NAME = "logistic_elo_plus_same_version_role"

# Slice 5 audit constants. Not fit to match outcomes.
MIN_EXPLICIT_FOR_ROLE_SHIFT = 8
CONTEST_SHIFT_ABS_DELTA = 0.20
_CAREER_EXPERIENCED_MEAN_GAMES = 10.0
LOW_N_VERSION_THRESHOLD = 50

PATCH_MATURITY_BIN_ORDER: tuple[str, ...] = (
    "opening (0–49 prior matches)",
    "early (50–199)",
    "mature (200+)",
)
CAREER_SAMPLE_BUCKET_ORDER: tuple[str, ...] = (
    "0",
    "1–4",
    "5–9",
    "10–19",
    "20+",
)
COMPATIBILITY_BUCKET_ORDER: tuple[str, ...] = (
    "NULL",
    "bottom quartile",
    "middle 50%",
    "top quartile",
)
HERO_SHIFT_GROUP_ORDER: tuple[str, ...] = (
    "stable",
    "role_shifted",
    "contest_shifted",
    "unclassified",
)


def assign_patch_maturity_bin(prior_matches: int) -> str:
    """Fixed professional-match maturity cuts for Slice 7 diagnostics.

    Thresholds match the existing ``assign_prior_match_bin`` 0–49 / 50–199
    edges; 200–499 and 500+ are collapsed to ``mature (200+)``. Cuts are
    not chosen from prediction metrics.
    """
    count = int(prior_matches)
    if count < 0:
        raise ValueError(
            "prior_matches_in_game_version cannot be negative, "
            f"got {count}"
        )
    if count <= 49:
        return PATCH_MATURITY_BIN_ORDER[0]
    if count <= 199:
        return PATCH_MATURITY_BIN_ORDER[1]
    return PATCH_MATURITY_BIN_ORDER[2]


def assign_career_sample_bucket(mean_prior_games: float) -> str:
    """Fixed buckets on match-level mean career Player × Hero games."""
    value = float(mean_prior_games)
    if value <= 0.0:
        return "0"
    if value < 5.0:
        return "1–4"
    if value < 10.0:
        return "5–9"
    if value < 20.0:
        return "10–19"
    return "20+"


def assign_compatibility_bucket(
    value: float | None, *, q25: float, q75: float
) -> str:
    """Map a compatibility score onto precomputed quartiles.

    Quartile edges must be computed from compatibility values only.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NULL"
    score = float(value)
    if np.isnan(q25) or np.isnan(q75):
        return "NULL"
    if score <= q25:
        return "bottom quartile"
    if score >= q75:
        return "top quartile"
    return "middle 50%"


def _modal_position(row: pd.Series, *, prefix: str) -> tuple[int | None, float]:
    shares: list[tuple[int, float]] = []
    for position in POSITION_NUMBERS:
        value = row.get(f"{prefix}position_{position}_share")
        if pd.notna(value):
            shares.append((position, float(value)))
    if not shares:
        return None, float("nan")
    shares.sort(key=lambda item: (-item[1], item[0]))
    return shares[0]


def describe_hero_shift_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Hero-level stable / role-shifted / contest-shifted labels.

    Reproduces Slice 5's descriptive rule from
    ``scripts/audit_hero_meta_state.py`` / the Slice 6 audit grouping:

    * Consecutive same-version end snapshots (last row per
      ``(hero_id, game_version_id)`` after sorting by start_time).
    * Role-shifted if modal same-version position changes and both
      snapshots have ≥ ``MIN_EXPLICIT_FOR_ROLE_SHIFT`` explicit
      observations.
    * Contest-shifted if |same-version contest-rate delta| ≥ 0.20.
    * Priority: role_shifted > contest_shifted > stable (eligible) >
      unclassified.

    Outcomes are not used. This is a descriptive grouping, not a feature.
    """
    snapshots = (
        frame.sort_values(["hero_id", "start_time", "match_id"], kind="mergesort")
        .groupby(["hero_id", "game_version_id"], as_index=False)
        .tail(1)
    )
    starts = (
        frame.groupby("game_version_id", as_index=False)["start_time"]
        .min()
        .sort_values("start_time", kind="mergesort")
    )
    version_order = [int(value) for value in starts["game_version_id"].tolist()]
    by_hero = {
        hero_id: subset.set_index("game_version_id")
        for hero_id, subset in snapshots.groupby("hero_id")
    }
    rows: list[dict[str, object]] = []
    for hero_id, subset in by_hero.items():
        role_ever = False
        contest_ever = False
        eligible = False
        for previous, current in pairwise(version_order):
            if previous not in subset.index or current not in subset.index:
                continue
            before = subset.loc[previous]
            after = subset.loc[current]
            if isinstance(before, pd.DataFrame):
                before = before.iloc[-1]
            if isinstance(after, pd.DataFrame):
                after = after.iloc[-1]
            from_explicit = int(
                before.get("hero_same_version_position_explicit_count", 0) or 0
            )
            to_explicit = int(
                after.get("hero_same_version_position_explicit_count", 0) or 0
            )
            from_contest = before.get("hero_same_version_contest_rate")
            to_contest = after.get("hero_same_version_contest_rate")
            enough_contest = pd.notna(from_contest) and pd.notna(to_contest)
            enough_pos = (
                from_explicit >= MIN_EXPLICIT_FOR_ROLE_SHIFT
                and to_explicit >= MIN_EXPLICIT_FOR_ROLE_SHIFT
            )
            if enough_contest or enough_pos:
                eligible = True
            contest_delta = (
                abs(float(to_contest) - float(from_contest))
                if enough_contest
                else 0.0
            )
            before_modal, _ = _modal_position(
                before, prefix="hero_same_version_"
            )
            after_modal, _ = _modal_position(after, prefix="hero_same_version_")
            if (
                enough_pos
                and before_modal is not None
                and after_modal is not None
                and before_modal != after_modal
            ):
                role_ever = True
            if enough_contest and contest_delta >= CONTEST_SHIFT_ABS_DELTA:
                contest_ever = True
        rows.append(
            {
                "hero_id": int(hero_id),
                "shift_group": (
                    "role_shifted"
                    if role_ever
                    else (
                        "contest_shifted"
                        if contest_ever
                        else ("stable" if eligible else "unclassified")
                    )
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["hero_id", "shift_group"])
    return pd.DataFrame.from_records(rows)


def _game_versions_registered(store: FeatureDuckDBConnection) -> bool:
    tables = {row[0] for row in store.sql("SHOW TABLES").fetchall()}
    return GAME_VERSIONS_VIEW in tables


def _count_zeros(series: pd.Series) -> int:
    """Observed zeros only; NULL is not treated as zero history."""
    return int((series == 0).sum())


def _side_mean(
    players: pd.DataFrame, *, value_column: str, side: str
) -> pd.DataFrame:
    name = f"{side.lower()}_mean_{value_column}"
    return (
        players.loc[players["side"] == side]
        .groupby(MATCH_ID_COLUMN, sort=False)[value_column]
        .mean()
        .rename(name)
        .reset_index()
    )


def _match_diagnostics(players: pd.DataFrame) -> pd.DataFrame:
    """Match-grain summaries of Slice 6 state for conditional evaluation."""
    grouped = players.groupby(MATCH_ID_COLUMN, sort=False)
    frame = grouped.agg(
        n_players=("player_id", "size"),
        mean_prior_games_on_hero=("prior_games_on_hero", "mean"),
        min_prior_games_on_hero=("prior_games_on_hero", "min"),
        mean_same_version_matches=("player_hero_same_version_matches", "mean"),
        min_same_version_matches=("player_hero_same_version_matches", "min"),
        max_same_version_matches=("player_hero_same_version_matches", "max"),
        mean_recent_20_matches=("player_hero_recent_20_matches", "mean"),
        mean_role_compatibility=(
            "player_hero_recent_role_compatibility",
            "mean",
        ),
        mean_player_share_at_expected_position=(
            "player_hero_share_at_expected_position",
            "mean",
        ),
        mean_hero_meta_share_at_expected_position=(
            "hero_meta_share_at_expected_position",
            "mean",
        ),
        n_zero_same_version_players=(
            "player_hero_same_version_matches",
            _count_zeros,
        ),
    ).reset_index()
    radiant_career = _side_mean(
        players, value_column="prior_games_on_hero", side="RADIANT"
    )
    dire_career = _side_mean(
        players, value_column="prior_games_on_hero", side="DIRE"
    )
    frame = frame.merge(radiant_career, on=MATCH_ID_COLUMN, how="left").merge(
        dire_career, on=MATCH_ID_COLUMN, how="left"
    )
    frame["career_sample_bucket"] = [
        assign_career_sample_bucket(value)
        for value in frame["mean_prior_games_on_hero"].fillna(0.0)
    ]
    return frame


def _match_shift_group(
    players: pd.DataFrame, hero_groups: pd.DataFrame
) -> pd.DataFrame:
    """One shift label per match from its drafted heroes.

    ``role_shifted`` if any drafted hero is role-shifted; else
    ``contest_shifted`` if any is contest-shifted; else ``stable`` if
    every drafted hero is stable; else ``unclassified``.
    """
    merged = players[[MATCH_ID_COLUMN, "hero_id"]].merge(
        hero_groups, on="hero_id", how="left"
    )
    rows: list[dict[str, object]] = []
    for match_id, subset in merged.groupby(MATCH_ID_COLUMN, sort=False):
        groups = subset["shift_group"].fillna("unclassified")
        n_role = int((groups == "role_shifted").sum())
        n_contest = int((groups == "contest_shifted").sum())
        n_stable = int((groups == "stable").sum())
        n_heroes = int(subset["hero_id"].nunique())
        if n_role > 0:
            label = "role_shifted"
        elif n_contest > 0:
            label = "contest_shifted"
        elif n_stable == len(groups) and n_heroes > 0:
            label = "stable"
        else:
            label = "unclassified"
        rows.append(
            {
                MATCH_ID_COLUMN: match_id,
                "match_shift_group": label,
                "n_role_shifted_hero_rows": n_role,
                "n_contest_shifted_hero_rows": n_contest,
                "n_stable_hero_rows": n_stable,
            }
        )
    return pd.DataFrame.from_records(rows)


def _align_diagnostics(
    diagnostics: pd.DataFrame, match_ids: pd.Series
) -> pd.DataFrame:
    return match_ids.to_frame(name=MATCH_ID_COLUMN).merge(
        diagnostics, on=MATCH_ID_COLUMN, how="left"
    )


@dataclass(frozen=True)
class Slice7Assembly:
    """Post-draft matrix plus Slice 6 comparison columns, same match rows."""

    dataset: ModelReadyDataset
    match_diagnostics: pd.DataFrame
    n_post_draft_matches: int
    n_slice6_comparison_matches: int
    n_missing_slice6_comparison: int
    n_incomplete_player_rows: int


def build_slice7_model_ready_dataset(
    store: FeatureDuckDBConnection,
) -> Slice7Assembly:
    """Elo + existing draft-comparison matrix, plus Slice 6 diffs.

    Row identity and order match ``build_post_draft_model_ready_dataset``.
    Slice 6 comparison columns are left-joined so a missing comparison
    does not drop an OOS match; NULL diffs are reported as coverage loss
    rather than a silent change of the evaluation population.
    """
    post = build_post_draft_model_ready_dataset(store)
    meta = build_player_hero_meta(store).to_frame()
    comparison = player_hero_meta_comparison_from_players(meta)

    roster_counts = meta.groupby(MATCH_ID_COLUMN).size()
    n_incomplete = int((roster_counts != 10).sum())

    match_ids = post.context[MATCH_ID_COLUMN]
    aligned = (
        comparison.set_index(MATCH_ID_COLUMN)
        .reindex(match_ids.to_numpy())
        .reset_index(drop=True)
    )
    slice7_frame = aligned[list(SLICE7_COMPARISON_COLUMNS)].copy()
    present_ids = set(comparison[MATCH_ID_COLUMN].tolist())
    n_missing_join = int((~match_ids.isin(present_ids)).sum())

    X = pd.concat([post.X.reset_index(drop=True), slice7_frame], axis=1)
    feature_columns = tuple(X.columns)
    overlap = set(FEATURE_COLUMNS) & set(SLICE7_COMPARISON_COLUMNS)
    if overlap:
        raise TrainingDatasetError(
            "Slice 7 comparison columns must not appear in FEATURE_COLUMNS: "
            f"{sorted(overlap)}"
        )
    dataset = ModelReadyDataset(
        X=X,
        y=post.y.reset_index(drop=True).copy(),
        context=post.context.reset_index(drop=True).copy(),
        feature_columns=feature_columns,
        target_column=post.target_column,
        identity_columns=post.identity_columns,
    )

    diagnostics = _match_diagnostics(meta)
    hero_groups = describe_hero_shift_groups(meta)
    if not hero_groups.empty:
        diagnostics = diagnostics.merge(
            _match_shift_group(meta, hero_groups),
            on=MATCH_ID_COLUMN,
            how="left",
        )
    else:
        diagnostics["match_shift_group"] = pd.NA

    if _game_versions_registered(store):
        maturity = build_patch_maturity(store).to_frame()
        diagnostics = diagnostics.merge(
            maturity[
                [
                    MATCH_ID_COLUMN,
                    "prior_matches_in_game_version",
                    "days_since_game_version_start",
                    "game_version_name",
                ]
            ],
            on=MATCH_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
        diagnostics["patch_maturity_bin"] = [
            assign_patch_maturity_bin(int(value)) if pd.notna(value) else pd.NA
            for value in diagnostics["prior_matches_in_game_version"]
        ]
    else:
        diagnostics["prior_matches_in_game_version"] = pd.NA
        diagnostics["days_since_game_version_start"] = pd.NA
        diagnostics["game_version_name"] = pd.NA
        diagnostics["patch_maturity_bin"] = pd.NA

    diagnostics = _align_diagnostics(diagnostics, match_ids)
    return Slice7Assembly(
        dataset=dataset,
        match_diagnostics=diagnostics,
        n_post_draft_matches=len(post),
        n_slice6_comparison_matches=len(comparison),
        n_missing_slice6_comparison=n_missing_join,
        n_incomplete_player_rows=n_incomplete,
    )


def _attach_career_deltas(oos: pd.DataFrame) -> pd.DataFrame:
    career = oos.loc[
        oos["model"] == SLICE7_CAREER_SPEC_NAME,
        [MATCH_ID_COLUMN, "sample_log_loss"],
    ].rename(columns={"sample_log_loss": "career_log_loss"})
    merged = oos.merge(
        career, on=MATCH_ID_COLUMN, how="left", validate="many_to_one"
    )
    merged["delta_vs_career"] = (
        merged["sample_log_loss"] - merged["career_log_loss"]
    )
    return merged


def _mean_or_nan(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def _subset_rows(oos: pd.DataFrame, match_ids: pd.Series) -> pd.DataFrame:
    ids = {int(value) for value in match_ids}
    return oos.loc[oos[MATCH_ID_COLUMN].isin(ids)]


def _spec_summary(subset: pd.DataFrame, spec_name: str) -> dict[str, float]:
    rows = subset.loc[subset["model"] == spec_name]
    n = len(rows)
    if n == 0:
        return {
            "n": 0.0,
            "log_loss": float("nan"),
            "delta_vs_elo": float("nan"),
            "delta_vs_career": float("nan"),
        }
    return {
        "n": float(n),
        "log_loss": _mean_or_nan(rows["sample_log_loss"]),
        "delta_vs_elo": _mean_or_nan(rows["delta_vs_elo"]),
        "delta_vs_career": _mean_or_nan(rows["delta_vs_career"]),
    }


def _wide_delta_row(
    subset: pd.DataFrame, *, extra: dict[str, object]
) -> dict[str, object]:
    elo = _spec_summary(subset, ELO_BLOCK_SPEC_NAME)
    career = _spec_summary(subset, SLICE7_CAREER_SPEC_NAME)
    same = _spec_summary(subset, SLICE7_SAME_VERSION_SPEC_NAME)
    recent = _spec_summary(subset, SLICE7_RECENT20_SPEC_NAME)
    role = _spec_summary(subset, SLICE7_ROLE_SPEC_NAME)
    combined = _spec_summary(subset, SLICE7_COMBINED_SPEC_NAME)
    return {
        **extra,
        "n": int(elo["n"]),
        "career_delta": career["delta_vs_elo"],
        "same_version_delta": same["delta_vs_elo"],
        "recent20_delta": recent["delta_vs_elo"],
        "role_delta": role["delta_vs_elo"],
        "combined_delta": combined["delta_vs_elo"],
        "career_log_loss": career["log_loss"],
        "same_version_log_loss": same["log_loss"],
        "recent20_log_loss": recent["log_loss"],
        "role_log_loss": role["log_loss"],
        "combined_log_loss": combined["log_loss"],
    }


def _pooled_with_career(
    oos: pd.DataFrame, specs: tuple[BlockAblationSpec, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        subset = oos.loc[oos["model"] == spec.name]
        full = evaluate_probabilities(subset["y_true"], subset["p_spec"])
        rows.append(
            {
                "model": spec.name,
                "label": spec.label,
                "n_features": len(spec.feature_columns),
                "n": len(subset),
                "log_loss": full.log_loss,
                "brier_score": full.brier_score,
                "accuracy_at_0.5": full.accuracy_at_0_5,
                "roc_auc": full.roc_auc,
                "ece": full.expected_calibration_error,
                "delta_vs_elo": _mean_or_nan(subset["delta_vs_elo"]),
                "delta_vs_career": _mean_or_nan(subset["delta_vs_career"]),
            }
        )
    return pd.DataFrame(rows)


def _fold_with_career(
    fold_metrics: pd.DataFrame, oos: pd.DataFrame
) -> pd.DataFrame:
    extras = (
        oos.groupby(["fold_id", "model"], sort=False)["delta_vs_career"]
        .mean()
        .reset_index()
    )
    return fold_metrics.merge(extras, on=["fold_id", "model"], how="left")


def _compatibility_edges(
    diagnostics: pd.DataFrame, oos_ids: pd.Series
) -> tuple[float, float]:
    values = diagnostics.loc[
        diagnostics[MATCH_ID_COLUMN].isin(set(oos_ids)),
        "mean_role_compatibility",
    ].dropna()
    if values.empty:
        return float("nan"), float("nan")
    return float(values.quantile(0.25)), float(values.quantile(0.75))


def _oos_labeled(
    diagnostics: pd.DataFrame, oos_ids: pd.Series
) -> tuple[pd.DataFrame, float, float]:
    q25, q75 = _compatibility_edges(diagnostics, oos_ids)
    labeled = diagnostics.copy()
    labeled["compatibility_bucket"] = [
        assign_compatibility_bucket(
            None if pd.isna(value) else float(value), q25=q25, q75=q75
        )
        for value in labeled["mean_role_compatibility"]
    ]
    oos_diag = labeled.loc[labeled[MATCH_ID_COLUMN].isin(set(oos_ids))].copy()
    return oos_diag, q25, q75


def _group_table(
    oos: pd.DataFrame,
    oos_diag: pd.DataFrame,
    *,
    column: str,
    values: tuple[str, ...],
    key_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value in values:
        ids = oos_diag.loc[oos_diag[column] == value, MATCH_ID_COLUMN]
        rows.append(
            _wide_delta_row(_subset_rows(oos, ids), extra={key_name: value})
        )
    return pd.DataFrame(rows)


@dataclass
class Slice7BenchmarkReport:
    """Walk-forward Slice 7 ablation plus conditional diagnostic tables."""

    assembly: Slice7Assembly
    walk_forward: WalkForwardReport
    oos_predictions: pd.DataFrame
    overall: pd.DataFrame
    fold_metrics: pd.DataFrame
    by_patch: pd.DataFrame
    patch_maturity: pd.DataFrame
    patch_cold: pd.DataFrame
    career_sample: pd.DataFrame
    compatibility: pd.DataFrame
    role_shift: pd.DataFrame
    count_vs_performance: pd.DataFrame
    compatibility_q25: float
    compatibility_q75: float
    n_oos: int


def run_slice7_meta_player_hero_benchmark(
    store: FeatureDuckDBConnection,
    *,
    config: WalkForwardConfig | None = None,
) -> Slice7BenchmarkReport:
    """Fit the named Slice 7 specs on the existing expanding-window folds."""
    resolved = config if config is not None else DEFAULT_WALK_FORWARD_CONFIG
    assembly = build_slice7_model_ready_dataset(store)
    wf = run_post_draft_walk_forward(
        assembly.dataset,
        config=resolved,
        specs=SLICE7_META_PLAYER_HERO_SPECS,
    )
    oos = _attach_career_deltas(wf.oos_predictions)
    overall = _pooled_with_career(oos, SLICE7_META_PLAYER_HERO_SPECS)
    fold_metrics = _fold_with_career(wf.fold_metrics, oos)

    elo_oos = oos.loc[oos["model"] == ELO_BLOCK_SPEC_NAME]
    oos_ids = elo_oos[MATCH_ID_COLUMN]
    oos_diag, q25, q75 = _oos_labeled(assembly.match_diagnostics, oos_ids)

    by_patch_rows: list[dict[str, object]] = []
    versions = sorted(
        v for v in elo_oos["game_version_id"].dropna().unique().tolist()
    )
    for version in versions:
        ids = elo_oos.loc[
            elo_oos["game_version_id"] == version, MATCH_ID_COLUMN
        ]
        n = len(ids)
        by_patch_rows.append(
            _wide_delta_row(
                _subset_rows(oos, ids),
                extra={
                    "game_version_id": version,
                    "low_n": n < LOW_N_VERSION_THRESHOLD,
                },
            )
        )
    by_patch = pd.DataFrame(by_patch_rows)

    patch_maturity = _group_table(
        oos,
        oos_diag,
        column="patch_maturity_bin",
        values=PATCH_MATURITY_BIN_ORDER,
        key_name="maturity",
    )

    patch_cold_rows = []
    cold_masks = (
        (
            "career present, same-version = 0",
            (oos_diag["mean_prior_games_on_hero"] > 0)
            & (oos_diag["mean_same_version_matches"] == 0),
        ),
        (
            f"career mean ≥{_CAREER_EXPERIENCED_MEAN_GAMES:.0f}, same-version = 0",
            (
                oos_diag["mean_prior_games_on_hero"]
                >= _CAREER_EXPERIENCED_MEAN_GAMES
            )
            & (oos_diag["mean_same_version_matches"] == 0),
        ),
    )
    for name, mask in cold_masks:
        ids = oos_diag.loc[mask, MATCH_ID_COLUMN]
        patch_cold_rows.append(
            _wide_delta_row(_subset_rows(oos, ids), extra={"population": name})
        )
    patch_cold = pd.DataFrame(patch_cold_rows)

    career_sample = _group_table(
        oos,
        oos_diag,
        column="career_sample_bucket",
        values=CAREER_SAMPLE_BUCKET_ORDER,
        key_name="career_sample_bucket",
    )

    compatibility = _group_table(
        oos,
        oos_diag,
        column="compatibility_bucket",
        values=COMPATIBILITY_BUCKET_ORDER,
        key_name="compatibility_bucket",
    )
    high_low_ids = oos_diag.loc[
        (oos_diag["mean_prior_games_on_hero"] >= _CAREER_EXPERIENCED_MEAN_GAMES)
        & (oos_diag["compatibility_bucket"] == "bottom quartile"),
        MATCH_ID_COLUMN,
    ]
    compatibility = pd.concat(
        [
            compatibility,
            pd.DataFrame(
                [
                    _wide_delta_row(
                        _subset_rows(oos, high_low_ids),
                        extra={
                            "compatibility_bucket": (
                                f"career mean ≥{_CAREER_EXPERIENCED_MEAN_GAMES:.0f} "
                                "/ bottom-quartile compatibility"
                            )
                        },
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    role_shift = _group_table(
        oos,
        oos_diag,
        column="match_shift_group",
        values=HERO_SHIFT_GROUP_ORDER,
        key_name="hero_group",
    )

    count_rows = []
    for block, count_name, plus_name in (
        (
            "same-version",
            SLICE7_SAME_VERSION_VOLUME_SPEC_NAME,
            SLICE7_SAME_VERSION_SPEC_NAME,
        ),
        (
            "recent-20",
            SLICE7_RECENT20_VOLUME_SPEC_NAME,
            SLICE7_RECENT20_SPEC_NAME,
        ),
    ):
        count_stats = _spec_summary(oos, count_name)
        plus_stats = _spec_summary(oos, plus_name)
        count_rows.append(
            {
                "block": block,
                "count_only_delta_vs_elo": count_stats["delta_vs_elo"],
                "count_plus_wr_delta_vs_elo": plus_stats["delta_vs_elo"],
                "count_only_delta_vs_career": count_stats["delta_vs_career"],
                "count_plus_wr_delta_vs_career": plus_stats["delta_vs_career"],
            }
        )

    return Slice7BenchmarkReport(
        assembly=assembly,
        walk_forward=wf,
        oos_predictions=oos,
        overall=overall,
        fold_metrics=fold_metrics,
        by_patch=by_patch,
        patch_maturity=patch_maturity,
        patch_cold=patch_cold,
        career_sample=career_sample,
        compatibility=compatibility,
        role_shift=role_shift,
        count_vs_performance=pd.DataFrame(count_rows),
        compatibility_q25=q25,
        compatibility_q75=q75,
        n_oos=len(elo_oos),
    )
