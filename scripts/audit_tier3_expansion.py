"""T1/T2 vs T3 canonical quality and population accounting.

Read-only. Does not train models. Join path for later experiments:
``matches.league_id`` -> ``leagues.liquipedia_tier`` (Parquet has
``league_id``, not tier).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import and_, func, select

from dota_predictor.data.canonical_schema import (
    EXPLICIT_DOTA_POSITIONS,
    MATCH_PLAYER_BOX_SCORE_COLUMNS,
    DraftAction,
)
from dota_predictor.datasets.canonical_export import (
    ANALYTICAL_SCHEMA_VERSION,
    DRAFT_EVENTS_FILENAME,
    MATCH_PLAYERS_FILENAME,
    MATCHES_FILENAME,
)
from dota_predictor.datasets.config import load_dataset_export_config
from dota_predictor.storage.engine import MissingDatabaseUrlError, get_engine
from dota_predictor.storage.schema import (
    DRAFT_EVENTS,
    LEAGUES,
    MATCH_INGESTION_ERRORS,
    MATCH_PLAYERS,
    MATCHES,
    STRATZ_RAW_MATCHES,
)
from dota_predictor.training.slice9_frozen_holdout import FROZEN_DEVELOPMENT_END
from dota_predictor.utils.env import load_project_env

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "interim" / "tier3_quality_audit.json"


def _tier_match_filter(tiers: tuple[str, ...]):
    return and_(
        LEAGUES.c.league_id == MATCHES.c.league_id,
        LEAGUES.c.liquipedia_tier.in_(tiers),
        LEAGUES.c.in_scope.is_(True),
    )


def _cohort_stats(conn, tiers: tuple[str, ...]) -> dict[str, Any]:
    match_n = int(
        conn.execute(
            select(func.count())
            .select_from(MATCHES)
            .join(LEAGUES, _tier_match_filter(tiers))
        ).scalar_one()
    )
    league_n = int(
        conn.execute(
            select(func.count(func.distinct(MATCHES.c.league_id)))
            .select_from(MATCHES)
            .join(LEAGUES, _tier_match_filter(tiers))
        ).scalar_one()
    )
    registered_n = int(
        conn.execute(
            select(func.count()).select_from(LEAGUES).where(
                LEAGUES.c.liquipedia_tier.in_(tiers),
                LEAGUES.c.in_scope.is_(True),
            )
        ).scalar_one()
    )
    min_start, max_start = conn.execute(
        select(func.min(MATCHES.c.start_time), func.max(MATCHES.c.start_time))
        .select_from(MATCHES)
        .join(LEAGUES, _tier_match_filter(tiers))
    ).one()
    player_n = int(
        conn.execute(
            select(func.count())
            .select_from(MATCH_PLAYERS)
            .join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
            .join(LEAGUES, _tier_match_filter(tiers))
        ).scalar_one()
    )
    explicit_n = int(
        conn.execute(
            select(func.count())
            .select_from(MATCH_PLAYERS)
            .join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
            .join(LEAGUES, _tier_match_filter(tiers))
            .where(MATCH_PLAYERS.c.position.in_(tuple(EXPLICIT_DOTA_POSITIONS)))
        ).scalar_one()
    )
    box_predicates = [
        getattr(MATCH_PLAYERS.c, col).is_not(None)
        for col in MATCH_PLAYER_BOX_SCORE_COLUMNS
    ]
    box_any = box_predicates[0]
    for pred in box_predicates[1:]:
        box_any = box_any | pred
    box_n = int(
        conn.execute(
            select(func.count())
            .select_from(MATCH_PLAYERS)
            .join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
            .join(LEAGUES, _tier_match_filter(tiers))
            .where(box_any)
        ).scalar_one()
    )
    ten_players = int(
        conn.execute(
            select(func.count()).select_from(
                select(MATCH_PLAYERS.c.match_id)
                .join(MATCHES, MATCHES.c.match_id == MATCH_PLAYERS.c.match_id)
                .join(LEAGUES, _tier_match_filter(tiers))
                .group_by(MATCH_PLAYERS.c.match_id)
                .having(func.count() == 10)
                .subquery()
            )
        ).scalar_one()
    )
    ten_picks = int(
        conn.execute(
            select(func.count()).select_from(
                select(DRAFT_EVENTS.c.match_id)
                .join(MATCHES, MATCHES.c.match_id == DRAFT_EVENTS.c.match_id)
                .join(LEAGUES, _tier_match_filter(tiers))
                .where(DRAFT_EVENTS.c.action == DraftAction.PICK)
                .group_by(DRAFT_EVENTS.c.match_id)
                .having(func.count() == 10)
                .subquery()
            )
        ).scalar_one()
    )
    gv_n = int(
        conn.execute(
            select(func.count())
            .select_from(MATCHES)
            .join(LEAGUES, _tier_match_filter(tiers))
            .where(MATCHES.c.game_version_id.is_not(None))
        ).scalar_one()
    )
    named_teams = int(
        conn.execute(
            select(func.count())
            .select_from(MATCHES)
            .join(LEAGUES, _tier_match_filter(tiers))
            .where(
                MATCHES.c.radiant_team_name_observed.is_not(None),
                MATCHES.c.dire_team_name_observed.is_not(None),
            )
        ).scalar_one()
    )
    raw_n = int(
        conn.execute(
            select(func.count())
            .select_from(STRATZ_RAW_MATCHES)
            .join(LEAGUES, LEAGUES.c.league_id == STRATZ_RAW_MATCHES.c.league_id)
            .where(
                LEAGUES.c.liquipedia_tier.in_(tiers),
                LEAGUES.c.in_scope.is_(True),
            )
        ).scalar_one()
    )
    err_n = int(
        conn.execute(
            select(func.count())
            .select_from(MATCH_INGESTION_ERRORS)
            .join(LEAGUES, LEAGUES.c.league_id == MATCH_INGESTION_ERRORS.c.league_id)
            .where(
                LEAGUES.c.liquipedia_tier.in_(tiers),
                LEAGUES.c.in_scope.is_(True),
            )
        ).scalar_one()
    )
    before_end = int(
        conn.execute(
            select(func.count())
            .select_from(MATCHES)
            .join(LEAGUES, _tier_match_filter(tiers))
            .where(MATCHES.c.start_time <= FROZEN_DEVELOPMENT_END)
        ).scalar_one()
    )
    after_end = int(
        conn.execute(
            select(func.count())
            .select_from(MATCHES)
            .join(LEAGUES, _tier_match_filter(tiers))
            .where(MATCHES.c.start_time > FROZEN_DEVELOPMENT_END)
        ).scalar_one()
    )
    per_league = [
        {
            "league_id": int(row.league_id),
            "name": row.name,
            "matches": int(row.n),
        }
        for row in conn.execute(
            select(LEAGUES.c.league_id, LEAGUES.c.name, func.count().label("n"))
            .select_from(MATCHES)
            .join(LEAGUES, _tier_match_filter(tiers))
            .group_by(LEAGUES.c.league_id, LEAGUES.c.name)
            .order_by(func.count().desc())
        )
    ]

    def _rate(num: int, den: int) -> float | None:
        return (num / den) if den else None

    return {
        "registered_in_scope_leagues": registered_n,
        "leagues_with_canonical_matches": league_n,
        "matches": match_n,
        "raw_matches": raw_n,
        "min_start_time": min_start.isoformat() if min_start else None,
        "max_start_time": max_start.isoformat() if max_start else None,
        "explicit_position_rate": _rate(explicit_n, player_n),
        "box_score_row_rate": _rate(box_n, player_n),
        "complete_draft_rate": _rate(ten_picks, match_n),
        "ten_player_rate": _rate(ten_players, match_n),
        "game_version_rate": _rate(gv_n, match_n),
        "named_team_rate": _rate(named_teams, match_n),
        "malformed_draft_rate": _rate(match_n - ten_picks, match_n),
        "ingestion_error_rate": _rate(err_n, raw_n),
        "ingestion_errors": err_n,
        "matches_before_frozen_end": before_end,
        "matches_after_frozen_end": after_end,
        "matches_per_league": per_league,
    }


def parquet_counts(output_dir: Path) -> dict[str, int | None]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"matches": None, "match_players": None, "draft_events": None}
    counts: dict[str, int | None] = {}
    for key, name in (
        ("matches", MATCHES_FILENAME),
        ("match_players", MATCH_PLAYERS_FILENAME),
        ("draft_events", DRAFT_EVENTS_FILENAME),
    ):
        path = output_dir / name
        counts[key] = int(pq.ParquetFile(path).metadata.num_rows) if path.is_file() else None
    return counts


def main() -> int:
    load_project_env(REPO_ROOT)
    try:
        engine = get_engine()
    except MissingDatabaseUrlError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with engine.connect() as conn:
        t12 = _cohort_stats(conn, ("T1", "T2"))
        t3 = _cohort_stats(conn, ("T3",))
        combined = _cohort_stats(conn, ("T1", "T2", "T3"))
        after_by_tier = [
            {"tier": row.liquipedia_tier, "matches": int(row.n)}
            for row in conn.execute(
                select(LEAGUES.c.liquipedia_tier, func.count().label("n"))
                .select_from(MATCHES)
                .join(LEAGUES, LEAGUES.c.league_id == MATCHES.c.league_id)
                .where(
                    LEAGUES.c.in_scope.is_(True),
                    MATCHES.c.start_time > FROZEN_DEVELOPMENT_END,
                )
                .group_by(LEAGUES.c.liquipedia_tier)
            )
        ]
        before_by_tier = [
            {"tier": row.liquipedia_tier, "matches": int(row.n)}
            for row in conn.execute(
                select(LEAGUES.c.liquipedia_tier, func.count().label("n"))
                .select_from(MATCHES)
                .join(LEAGUES, LEAGUES.c.league_id == MATCHES.c.league_id)
                .where(
                    LEAGUES.c.in_scope.is_(True),
                    MATCHES.c.start_time <= FROZEN_DEVELOPMENT_END,
                )
                .group_by(LEAGUES.c.liquipedia_tier)
            )
        ]

    config = load_dataset_export_config(root=REPO_ROOT)
    report = {
        "frozen_development_end": FROZEN_DEVELOPMENT_END.isoformat(),
        "analytical_schema_version": ANALYTICAL_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "join_path": (
            "matches.league_id -> leagues.liquipedia_tier "
            "(or config/leagues.yaml). Parquet has no tier column."
        ),
        "existing_t1_t2": t12,
        "added_t3": t3,
        "combined_canonical": combined,
        "combined_before_frozen_end": {
            "matches": sum(row["matches"] for row in before_by_tier),
            "by_tier": before_by_tier,
        },
        "matches_after_frozen_end": {
            "matches": sum(row["matches"] for row in after_by_tier),
            "by_tier": after_by_tier,
        },
        "parquet": parquet_counts(config.output_dir),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
