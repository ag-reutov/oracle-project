"""League registry loading guards for the Liquipedia T3 expansion.

No database required. These pin T3 labels, unchanged T1/T2 entries,
duplicate-id rejection, and the frozen research/schema constants that
ingestion must not alter.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from dota_predictor.datasets.canonical_export import ANALYTICAL_SCHEMA_VERSION
from dota_predictor.features.pre_draft_snapshot import FEATURE_COLUMNS
from dota_predictor.storage.schema import LIQUIPEDIA_TIERS
from dota_predictor.training.feature_sets import ALL_FEATURE_COLUMNS
from dota_predictor.training.slice9_frozen_holdout import (
    FROZEN_DEVELOPMENT_END,
    FROZEN_DEVELOPMENT_MATCH_COUNT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEAGUES_YAML = REPO_ROOT / "config" / "leagues.yaml"
LOADER_PATH = REPO_ROOT / "scripts" / "load_league_registry.py"

# In-scope T1/T2 league_id -> liquipedia_tier as of the pre-T3 registry.
HEAD_IN_SCOPE_T1_T2: dict[int, str] = {
    16169: "T1",
    16201: "T1",
    16483: "T1",
    16518: "T1",
    16632: "T1",
    16669: "T1",
    16901: "T1",
    16881: "T1",
    16935: "T1",
    17126: "T1",
    17272: "T1",
    17414: "T1",
    17509: "T1",
    16427: "T2",
    16730: "T2",
    16446: "T2",
    16905: "T2",
    16846: "T2",
    17119: "T2",
    15981: "T2",
    17588: "T1",
    17417: "T1",
    17765: "T1",
    17891: "T1",
    17907: "T1",
    18058: "T1",
    17795: "T1",
    17418: "T1",
    18358: "T1",
    18111: "T1",
    18359: "T1",
    18375: "T1",
    18324: "T1",
    18433: "T1",
    17419: "T1",
    18863: "T1",
    18920: "T1",
    17420: "T1",
    18988: "T1",
    18046: "T2",
    18107: "T2",
    18633: "T2",
    17622: "T2",
    18937: "T2",
    19239: "T2",
    19099: "T1",
    19269: "T1",
    19435: "T1",
    19422: "T1",
    19543: "T1",
    19696: "T1",
    19101: "T1",
    19785: "T1",
    19719: "T1",
}


def _load_yaml_leagues() -> list[dict]:
    raw = yaml.safe_load(LEAGUES_YAML.read_text(encoding="utf-8")) or {}
    return list(raw.get("leagues") or [])


def _load_registry_module():
    spec = importlib.util.spec_from_file_location("load_league_registry", LOADER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_t3_is_an_allowed_liquipedia_tier() -> None:
    assert "T3" in LIQUIPEDIA_TIERS


def test_t3_labels_survive_registry_loading() -> None:
    leagues = _load_yaml_leagues()
    t3 = [entry for entry in leagues if entry["liquipedia_tier"] == "T3"]
    assert t3
    assert all(entry.get("in_scope") is True for entry in t3)
    assert all(entry.get("source") == "liquipedia_tier3_2024plus" for entry in t3)
    loader = _load_registry_module()
    loaded = loader.load_registry_entries(LEAGUES_YAML)
    loaded_t3 = [entry for entry in loaded if entry["liquipedia_tier"] == "T3"]
    assert len(loaded_t3) == len(t3)
    assert {entry["league_id"] for entry in loaded_t3} == {
        entry["league_id"] for entry in t3
    }


def test_existing_t1_t2_entries_unchanged() -> None:
    by_id = {entry["league_id"]: entry for entry in _load_yaml_leagues()}
    for league_id, tier in HEAD_IN_SCOPE_T1_T2.items():
        assert league_id in by_id, f"missing pre-T3 league_id {league_id}"
        entry = by_id[league_id]
        assert entry["liquipedia_tier"] == tier
        assert entry.get("in_scope") is True
    assert by_id[16427]["liquipedia_tier"] == "T2"
    assert by_id[17622]["liquipedia_tier"] == "T2"
    t3_ids = {
        entry["league_id"]
        for entry in by_id.values()
        if entry["liquipedia_tier"] == "T3"
    }
    t12_ids = {
        entry["league_id"]
        for entry in by_id.values()
        if entry["liquipedia_tier"] in {"T1", "T2"}
    }
    assert t3_ids.isdisjoint(t12_ids)


def test_duplicate_league_ids_rejected(tmp_path: Path) -> None:
    path = tmp_path / "leagues.yaml"
    path.write_text(
        """
leagues:
  - league_id: 15520
    name: "One"
    liquipedia_tier: T3
    in_scope: true
  - league_id: 15520
    name: "Dup"
    liquipedia_tier: T3
    in_scope: true
""",
        encoding="utf-8",
    )
    loader = _load_registry_module()
    with pytest.raises(ValueError, match="Duplicate league_id 15520"):
        loader.load_registry_entries(path)


def test_registry_file_has_unique_league_ids() -> None:
    ids = [entry["league_id"] for entry in _load_yaml_leagues()]
    assert len(ids) == len(set(ids))


def test_frozen_development_boundary_unchanged() -> None:
    assert FROZEN_DEVELOPMENT_END == datetime(2026, 7, 19, 17, 49, 1, tzinfo=UTC)
    assert FROZEN_DEVELOPMENT_MATCH_COUNT == 5967


def test_feature_columns_remain_33() -> None:
    assert len(FEATURE_COLUMNS) == 33
    assert list(ALL_FEATURE_COLUMNS) == list(FEATURE_COLUMNS)


def test_analytical_schema_version_unchanged() -> None:
    assert ANALYTICAL_SCHEMA_VERSION == 4
