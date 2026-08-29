"""Tests for dataset export configuration resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from dota_predictor.datasets.config import load_dataset_export_config


def test_defaults_to_data_processed_under_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROCESSED_DATA_DIR", raising=False)
    config = load_dataset_export_config(root=Path("/repo"))
    assert config.output_dir == Path("/repo/data/processed")


def test_env_var_override_is_used_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROCESSED_DATA_DIR", "/custom/output")
    config = load_dataset_export_config(root=Path("/repo"))
    assert config.output_dir == Path("/custom/output")


def test_defaults_to_cwd_when_root_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROCESSED_DATA_DIR", raising=False)
    config = load_dataset_export_config()
    assert config.output_dir == Path.cwd() / "data" / "processed"
