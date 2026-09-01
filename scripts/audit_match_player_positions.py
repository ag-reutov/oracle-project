"""Audit observed STRATZ position/lane/role on canonical match_players.

Does not infer or repair anomalies. `slot_in_side` is compared with
explicit POSITION_N only as a diagnostic that lobby slot is not position.

Usage:
    uv run python scripts/audit_match_player_positions.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path

import duckdb
from sqlalchemy import func, select

from dota_predictor.data.player_position_diagnostics import (
    EXPLICIT_POSITION_VALUES,
    audit_side_positions,
)
from dota_predictor.features.config import (
    load_feature_store_config,
    load_reference_store_config,
)
from dota_predictor.storage.engine import MissingDatabaseUrlError, get_engine
from dota_predictor.storage.schema import MATCH_PLAYERS, MATCHES
from dota_predictor.utils.env import load_project_env

POSITION_ORDER = (
    "POSITION_1",
    "POSITION_2",
    "POSITION_3",
    "POSITION_4",
    "POSITION_5",
    "UNKNOWN",
    None,
)
SLOT_VALUES = (0, 1, 2, 3, 4)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{100.0 * part / whole:.2f}%"


def _label(value: object) -> str:
    if value is None:
        return "NULL"
    return str(value)


def _enum_str(value: object) -> str:
    if isinstance(value, Enum):
        return value.value
    return str(value)


def _optional_enum_str(value: object) -> str | None:
    if value is None:
        return None
    return _enum_str(value)


def _print_counts(
    title: str,
    counts: Counter[object],
    *,
    total: int,
    expected: frozenset[object] | None = None,
) -> None:
    print(title)
    ordered_keys = [key for key in POSITION_ORDER if key in counts]
    other_keys = sorted(
        (key for key in counts if key not in POSITION_ORDER),
        key=lambda key: _label(key),
    )
    for key in [*ordered_keys, *other_keys]:
        n = counts[key]
        print(f"  {_label(key)}: {n} ({_pct(n, total)})")
    if expected is None:
        return
    unexpected = {
        _label(key): counts[key]
        for key in counts
        if key not in expected and key is not None
    }
    if unexpected:
        print(f"  unexpected values: {unexpected}")
    else:
        print("  unexpected values: none")


def _print_crosstab(
    title: str,
    table: dict[str, Counter[object]],
    row_keys: tuple[str, ...],
    col_keys: tuple[object, ...],
) -> None:
    print(title)
    header = "  {:>12}".format("row") + "".join(
        f" { _label(col):>16}" for col in col_keys
    )
    print(header)
    for row in row_keys:
        counts = table.get(row, Counter())
        cells = "".join(f" {counts.get(col, 0):>16d}" for col in col_keys)
        print(f"  {row:>12}{cells}")


def _compare_postgres(parquet_rows: list[dict]) -> None:
    print("POSTGRES VS PARQUET")
    try:
        engine = get_engine()
    except MissingDatabaseUrlError as exc:
        print(f"  skipped: {exc}")
        return

    with engine.connect() as conn:
        pg_player_count = conn.execute(
            select(func.count()).select_from(MATCH_PLAYERS)
        ).scalar_one()
        pg_match_count = conn.execute(
            select(func.count()).select_from(MATCHES)
        ).scalar_one()
        ten_rows = conn.execute(
            select(func.count())
            .select_from(
                select(MATCH_PLAYERS.c.match_id)
                .group_by(MATCH_PLAYERS.c.match_id)
                .having(func.count() != 10)
                .subquery()
            )
        ).scalar_one()
        duplicate_players = conn.execute(
            select(func.count())
            .select_from(
                select(MATCH_PLAYERS.c.match_id, MATCH_PLAYERS.c.player_id)
                .group_by(MATCH_PLAYERS.c.match_id, MATCH_PLAYERS.c.player_id)
                .having(func.count() > 1)
                .subquery()
            )
        ).scalar_one()
        mapper_versions = dict(
            conn.execute(
                select(MATCHES.c.mapper_version, func.count())
                .group_by(MATCHES.c.mapper_version)
                .order_by(MATCHES.c.mapper_version)
            ).all()
        )
        pg_rows = conn.execute(
            select(
                MATCH_PLAYERS.c.match_id,
                MATCH_PLAYERS.c.side,
                MATCH_PLAYERS.c.slot_in_side,
                MATCH_PLAYERS.c.player_id,
                MATCH_PLAYERS.c.hero_id,
                MATCH_PLAYERS.c.position,
                MATCH_PLAYERS.c.lane,
                MATCH_PLAYERS.c.role,
            )
        ).all()

    pg_keys = {
        (
            int(row.match_id),
            _enum_str(row.side),
            int(row.slot_in_side),
            int(row.player_id),
            int(row.hero_id),
            _optional_enum_str(row.position),
            _optional_enum_str(row.lane),
            _optional_enum_str(row.role),
        )
        for row in pg_rows
    }
    pq_keys = {
        (
            int(row["match_id"]),
            str(row["side"]),
            int(row["slot_in_side"]),
            int(row["player_id"]),
            int(row["hero_id"]),
            row["position"],
            row["lane"],
            row["role"],
        )
        for row in parquet_rows
    }
    identity_pg = {(k[0], k[1], k[2], k[3], k[4]) for k in pg_keys}
    identity_pq = {(k[0], k[1], k[2], k[3], k[4]) for k in pq_keys}

    print(f"  postgres match_players rows: {pg_player_count}")
    print(f"  postgres matches: {pg_match_count}")
    print(f"  postgres matches without exactly 10 player rows: {ten_rows}")
    print(f"  postgres duplicate (match_id, player_id): {duplicate_players}")
    print(f"  postgres mapper_version counts: {mapper_versions}")
    print(f"  parquet rows: {len(parquet_rows)}")
    print(f"  identity tuples equal: {identity_pg == identity_pq}")
    print(f"  identity+position/lane/role equal: {pg_keys == pq_keys}")
    if pg_keys != pq_keys:
        only_pg = pg_keys - pq_keys
        only_pq = pq_keys - pg_keys
        print(f"  rows only in postgres: {len(only_pg)}")
        print(f"  rows only in parquet: {len(only_pq)}")
        for sample in list(only_pg)[:3]:
            print(f"    postgres sample: {sample}")
        for sample in list(only_pq)[:3]:
            print(f"    parquet sample: {sample}")


def main() -> int:
    root = _project_root()
    load_project_env(root)
    config = load_feature_store_config(root=root)
    reference = load_reference_store_config(root=root)
    con = duckdb.connect(database=":memory:")
    con.execute(
        f"CREATE VIEW match_players AS SELECT * FROM read_parquet('{config.match_players_path.as_posix()}')"
    )
    con.execute(
        f"CREATE VIEW matches AS SELECT * FROM read_parquet('{config.matches_path.as_posix()}')"
    )
    version_name_sql = "CAST(NULL AS VARCHAR) AS game_version_name"
    if reference.game_versions_path.is_file():
        con.execute(
            f"CREATE VIEW game_versions AS SELECT * FROM read_parquet('{reference.game_versions_path.as_posix()}')"
        )
        version_name_sql = "gv.name AS game_version_name"

    rows = con.execute(
        f"""
        SELECT
            mp.match_id,
            m.league_id,
            m.league_name,
            m.game_version_id,
            {version_name_sql},
            m.start_time,
            m.mapper_version,
            mp.team_id,
            mp.side,
            mp.slot_in_side,
            mp.player_id,
            mp.hero_id,
            mp.position,
            mp.lane,
            mp.role
        FROM match_players mp
        JOIN matches m USING (match_id)
        LEFT JOIN game_versions gv ON m.game_version_id = gv.game_version_id
        ORDER BY mp.match_id, mp.side, mp.slot_in_side
        """
        if reference.game_versions_path.is_file()
        else f"""
        SELECT
            mp.match_id,
            m.league_id,
            m.league_name,
            m.game_version_id,
            {version_name_sql},
            m.start_time,
            m.mapper_version,
            mp.team_id,
            mp.side,
            mp.slot_in_side,
            mp.player_id,
            mp.hero_id,
            mp.position,
            mp.lane,
            mp.role
        FROM match_players mp
        JOIN matches m USING (match_id)
        ORDER BY mp.match_id, mp.side, mp.slot_in_side
        """
    ).fetchall()
    columns = [
        "match_id",
        "league_id",
        "league_name",
        "game_version_id",
        "game_version_name",
        "start_time",
        "mapper_version",
        "team_id",
        "side",
        "slot_in_side",
        "player_id",
        "hero_id",
        "position",
        "lane",
        "role",
    ]
    records = [dict(zip(columns, row, strict=True)) for row in rows]

    n_rows = len(records)
    n_matches = len({row["match_id"] for row in records})
    rows_per_match = Counter(row["match_id"] for row in records)
    unique_player_pairs = len({(row["match_id"], row["player_id"]) for row in records})
    null_identity = sum(
        1
        for row in records
        if any(
            row[col] is None
            for col in ("match_id", "team_id", "side", "slot_in_side", "player_id", "hero_id")
        )
    )

    position_counts: Counter[object] = Counter(row["position"] for row in records)
    lane_counts: Counter[object] = Counter(row["lane"] for row in records)
    role_counts: Counter[object] = Counter(row["role"] for row in records)

    by_match_side: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in records:
        by_match_side[(row["match_id"], row["side"])].append(row)

    duplicate_sides = 0
    missing_sides = 0
    null_or_unknown_sides = 0
    clean_sides = 0
    anomaly_sizes: Counter[int] = Counter()
    worst_sides: list[tuple[int, dict]] = []
    match_clean_sides: dict[int, int] = defaultdict(int)
    match_meta: dict[int, dict] = {}
    slot_confusion: dict[int, Counter[object]] = {slot: Counter() for slot in SLOT_VALUES}
    position_by_slot: dict[str, Counter[object]] = {
        value: Counter() for value in sorted(EXPLICIT_POSITION_VALUES)
    }
    position_role: dict[str, Counter[object]] = defaultdict(Counter)
    position_lane: dict[str, Counter[object]] = defaultdict(Counter)
    lane_by_version_null: dict[object, list[int]] = defaultdict(lambda: [0, 0])
    role_by_version_null: dict[object, list[int]] = defaultdict(lambda: [0, 0])

    for (match_id, side), side_rows in by_match_side.items():
        positions = [row["position"] for row in side_rows]
        audit = audit_side_positions(positions)
        anomaly_size = (
            audit.null_count
            + audit.unknown_count
            + audit.other_count
            + sum(positions.count(value) - 1 for value in audit.duplicate_explicit)
        )
        if audit.duplicate_explicit:
            duplicate_sides += 1
        if audit.missing_explicit:
            missing_sides += 1
        if audit.null_count or audit.unknown_count:
            null_or_unknown_sides += 1
        if audit.is_clean_unique_1_to_5:
            clean_sides += 1
            match_clean_sides[match_id] += 1
        else:
            anomaly_sizes[anomaly_size] += 1
            worst_sides.append(
                (
                    anomaly_size,
                    {
                        "match_id": match_id,
                        "side": side,
                        "positions": positions,
                        "null_count": audit.null_count,
                        "unknown_count": audit.unknown_count,
                        "duplicates": audit.duplicate_explicit,
                        "missing": audit.missing_explicit,
                        "league": side_rows[0]["league_name"],
                        "version": side_rows[0]["game_version_name"]
                        or side_rows[0]["game_version_id"],
                        "start_time": side_rows[0]["start_time"],
                    },
                )
            )
        for row in side_rows:
            position = row["position"]
            if position in EXPLICIT_POSITION_VALUES:
                slot_confusion[int(row["slot_in_side"])][position] += 1
                position_by_slot[position][int(row["slot_in_side"])] += 1
            position_role[_label(position)][row["role"]] += 1
            position_lane[_label(position)][row["lane"]] += 1
        match_meta[match_id] = side_rows[0]

        version_key = (
            side_rows[0]["game_version_id"],
            side_rows[0]["game_version_name"],
        )
        lane_by_version_null[version_key][0] += len(side_rows)
        role_by_version_null[version_key][0] += len(side_rows)
        lane_by_version_null[version_key][1] += sum(
            row["lane"] is None for row in side_rows
        )
        role_by_version_null[version_key][1] += sum(
            row["role"] is None for row in side_rows
        )

    n_sides = len(by_match_side)
    matches_clean_both = sum(1 for n in match_clean_sides.values() if n == 2)
    matches_one_anomalous = sum(
        1
        for match_id in match_meta
        if match_clean_sides.get(match_id, 0) == 1
    )
    matches_both_anomalous = sum(
        1
        for match_id in match_meta
        if match_clean_sides.get(match_id, 0) == 0
    )

    by_version_clean: dict[object, list[int]] = defaultdict(lambda: [0, 0])
    by_league_clean: dict[object, list[int]] = defaultdict(lambda: [0, 0])
    by_month_clean: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for match_id, sample in match_meta.items():
        both_clean = 1 if match_clean_sides.get(match_id, 0) == 2 else 0
        version_key = (sample["game_version_id"], sample["game_version_name"])
        by_version_clean[version_key][0] += 1
        by_version_clean[version_key][1] += both_clean
        league_key = (sample["league_id"], sample["league_name"])
        by_league_clean[league_key][0] += 1
        by_league_clean[league_key][1] += both_clean
        month = str(sample["start_time"])[:7]
        by_month_clean[month][0] += 1
        by_month_clean[month][1] += both_clean

    explicit_compared = sum(slot_confusion[slot].total() for slot in SLOT_VALUES)
    slot_agrees = 0
    for slot in SLOT_VALUES:
        expected = f"POSITION_{slot + 1}"
        slot_agrees += slot_confusion[slot][expected]
    slot_mode_purity: list[tuple[int, str, int, int]] = []
    for slot in SLOT_VALUES:
        counts = slot_confusion[slot]
        if not counts:
            continue
        mode_position, mode_n = counts.most_common(1)[0]
        slot_mode_purity.append((slot, str(mode_position), mode_n, counts.total()))

    _compare_postgres(records)

    print("CANONICAL PARQUET INVARIANTS")
    print(f"  player-match rows: {n_rows}")
    print(f"  matches: {n_matches}")
    print(f"  unique (match_id, player_id): {unique_player_pairs}")
    print(f"  matches without exactly 10 rows: {sum(1 for n in rows_per_match.values() if n != 10)}")
    print(f"  rows with null identity/hero/team/slot: {null_identity}")
    print(f"  mapper_version counts: {dict(Counter(row['mapper_version'] for row in match_meta.values()))}")

    print("POSITION COMPLETENESS")
    _print_counts(
        "  value distribution",
        position_counts,
        total=n_rows,
        expected=EXPLICIT_POSITION_VALUES | {"UNKNOWN"},
    )
    explicit_rows = sum(
        position_counts[value] for value in EXPLICIT_POSITION_VALUES
    )
    print(f"  POSITION_1-5 total: {explicit_rows} ({_pct(explicit_rows, n_rows)})")

    print("PER-SIDE STRUCTURAL VALIDITY")
    print(f"  total sides: {n_sides}")
    print(
        f"  clean sides with exactly {{1,2,3,4,5}}: {clean_sides} "
        f"({_pct(clean_sides, n_sides)})"
    )
    print(f"  sides with duplicate positions: {duplicate_sides}")
    print(f"  sides missing one or more positions: {missing_sides}")
    print(f"  sides containing NULL/UNKNOWN: {null_or_unknown_sides}")
    print(f"  anomaly-size distribution (non-clean sides): {dict(sorted(anomaly_sizes.items()))}")
    print(f"  matches clean on both sides: {matches_clean_both} ({_pct(matches_clean_both, n_matches)})")
    print(f"  matches with exactly one anomalous side: {matches_one_anomalous}")
    print(f"  matches anomalous on both sides: {matches_both_anomalous}")
    print("  worst anomalous sides:")
    if not worst_sides:
        print("    none")
    else:
        for _, example in sorted(worst_sides, key=lambda item: (-item[0], item[1]["match_id"])):
            print(
                f"    match {example['match_id']} {example['side']} "
                f"{example['start_time']} {example['league']} {example['version']} "
                f"positions={example['positions']} null={example['null_count']} "
                f"unknown={example['unknown_count']} duplicates={example['duplicates']} "
                f"missing={example['missing']}"
            )

    print("COMPLETENESS BY GAME VERSION")
    for (version_id, version_name), (total, clean) in sorted(
        by_version_clean.items(),
        key=lambda item: (item[0][0] is None, item[0][0] or 0),
    ):
        label = f"{version_id} {version_name or ''}".rstrip()
        print(f"  {label}: {clean}/{total} both-sides clean ({_pct(clean, total)})")

    print("COMPLETENESS BY LEAGUE")
    incomplete_leagues = 0
    for (league_id, league_name), (total, clean) in sorted(
        by_league_clean.items(), key=lambda item: (-item[1][0], item[0][0])
    ):
        if clean != total:
            incomplete_leagues += 1
        print(
            f"  {league_id} {league_name}: {clean}/{total} "
            f"({_pct(clean, total)})"
        )
    print(f"  leagues with any incomplete match: {incomplete_leagues}/{len(by_league_clean)}")

    print("COMPLETENESS BY MONTH")
    for month, (total, clean) in sorted(by_month_clean.items()):
        print(f"  {month}: {clean}/{total} ({_pct(clean, total)})")

    print("LANE")
    _print_counts("  value distribution", lane_counts, total=n_rows)
    print("  NULL lane by game version (only versions with any NULL):")
    any_lane_null = False
    for (version_id, version_name), (total, n_null) in sorted(
        lane_by_version_null.items(),
        key=lambda item: (item[0][0] is None, item[0][0] or 0),
    ):
        if n_null:
            any_lane_null = True
            print(
                f"    {version_id} {version_name or ''}: {n_null}/{total} "
                f"({_pct(n_null, total)})"
            )
    if not any_lane_null:
        print("    none")

    print("ROLE")
    _print_counts("  value distribution", role_counts, total=n_rows)
    print("  NULL role by game version (only versions with any NULL):")
    any_role_null = False
    for (version_id, version_name), (total, n_null) in sorted(
        role_by_version_null.items(),
        key=lambda item: (item[0][0] is None, item[0][0] or 0),
    ):
        if n_null:
            any_role_null = True
            print(
                f"    {version_id} {version_name or ''}: {n_null}/{total} "
                f"({_pct(n_null, total)})"
            )
    if not any_role_null:
        print("    none")

    role_cols = tuple(
        key
        for key, _ in role_counts.most_common()
    )
    lane_cols = tuple(key for key, _ in lane_counts.most_common())
    position_rows = (
        "POSITION_1",
        "POSITION_2",
        "POSITION_3",
        "POSITION_4",
        "POSITION_5",
        "UNKNOWN",
        "NULL",
    )
    _print_crosstab("POSITION x ROLE", position_role, position_rows, role_cols)
    _print_crosstab("POSITION x LANE", position_lane, position_rows, lane_cols)

    print("SLOT VS POSITION")
    print(
        f"  slot_in_side == POSITION_N-1 on explicit rows: "
        f"{slot_agrees}/{explicit_compared} ({_pct(slot_agrees, explicit_compared)})"
    )
    _print_crosstab(
        "  slot_in_side x position",
        {str(slot): slot_confusion[slot] for slot in SLOT_VALUES},
        tuple(str(slot) for slot in SLOT_VALUES),
        tuple(sorted(EXPLICIT_POSITION_VALUES)),
    )
    print("  most common position per lobby slot (purity):")
    for slot, mode, mode_n, total in slot_mode_purity:
        print(f"    slot {slot}: {mode} {mode_n}/{total} ({_pct(mode_n, total)})")
    max_purity = max((mode_n / total for _, _, mode_n, total in slot_mode_purity), default=0.0)
    print(
        "  deterministic slot->position mapping exists: "
        f"{'no' if max_purity < 1.0 else 'yes'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
