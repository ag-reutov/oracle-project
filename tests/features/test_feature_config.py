"""Tests for `features.config` (feature-layer Parquet path resolution)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dota_predictor.features.config import load_feature_store_config


def test_defaults_to_data_processed_under_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROCESSED_DATA_DIR", raising=False)
    config = load_feature_store_config(root=Path("/repo"))
    assert config.matches_path == Path("/repo/data/processed/matches.parquet")
    assert config.draft_events_path == Path("/repo/data/processed/draft_events.parquet")


def test_agrees_with_dataset_export_config_on_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The feature layer must read from exactly the directory Step 2
    writes to -- this is the "do not duplicate configuration logic"
    requirement, checked directly against the export config."""
    from dota_predictor.datasets.config import load_dataset_export_config

    monkeypatch.setenv("PROCESSED_DATA_DIR", "/custom/output")
    export_config = load_dataset_export_config(root=Path("/repo"))
    feature_config = load_feature_store_config(root=Path("/repo"))

    assert feature_config.matches_path.parent == export_config.output_dir
    assert feature_config.draft_events_path.parent == export_config.output_dir
