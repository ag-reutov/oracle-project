"""Tests for the pure team-identity derivation logic.

These are database-free tests of `dota_predictor.data.team_identity`
(derivation over observation tuples, config parsing, and the
no-silent-merge / deterministic-alias properties). DB-touching behavior is
covered separately in `tests/storage/test_team_identity_storage.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dota_predictor.data.team_identity import (
    derive_team_aliases,
    derive_team_tags,
    load_team_organizations_config,
)

T1 = datetime(2024, 1, 1, tzinfo=UTC)
T2 = datetime(2024, 2, 1, tzinfo=UTC)
T3 = datetime(2024, 3, 1, tzinfo=UTC)


def test_derive_team_aliases_groups_by_team_id_and_name():
    observations = [
        (100, "Example Team", T1),
        (100, "Example Team", T3),
        (200, "Example Team", T2),
        (200, "Other Name", T1),
        (100, None, T2),  # missing observed name is not an alias
    ]
    aliases = derive_team_aliases(observations)
    by_key = {(a.team_id, a.name): a for a in aliases}
    assert len(aliases) == 3
    assert by_key[(100, "Example Team")].observation_count == 2
    assert by_key[(100, "Example Team")].first_seen_at == T1
    assert by_key[(100, "Example Team")].last_seen_at == T3
    assert by_key[(200, "Example Team")].observation_count == 1
    assert by_key[(200, "Other Name")].observation_count == 1


def test_no_silent_merge_two_team_ids_sharing_a_name_stay_distinct():
    """The core Slice 1 invariant: same name never merges source teams."""
    observations = [
        (100, "Example Team", T1),
        (200, "Example Team", T2),
    ]
    aliases = derive_team_aliases(observations)
    assert {(a.team_id, a.name) for a in aliases} == {
        (100, "Example Team"),
        (200, "Example Team"),
    }


def test_derive_team_aliases_is_deterministic_regardless_of_input_order():
    observations = [
        (100, "A", T1),
        (200, "B", T2),
        (100, "A", T3),
        (200, "C", T1),
    ]
    shuffled = list(reversed(observations))
    assert derive_team_aliases(observations) == derive_team_aliases(shuffled)


def test_derive_team_aliases_handles_empty_and_all_none():
    assert derive_team_aliases([]) == []
    assert derive_team_aliases([(100, None, T1)]) == []


def test_derive_team_tags_skips_missing_tags_and_groups_periods():
    observations = [
        (100, "VP", T1),
        (100, "VP", T3),
        (100, None, T2),  # missing tag is not an observation
        (200, "GG", T2),
    ]
    tags = derive_team_tags(observations)
    by_key = {(t.team_id, t.tag): t for t in tags}
    assert len(tags) == 2
    assert by_key[(100, "VP")].observation_count == 2
    assert by_key[(100, "VP")].first_seen_at == T1
    assert by_key[(100, "VP")].last_seen_at == T3
    assert by_key[(200, "GG")].observation_count == 1


def test_derive_team_tags_treats_multiple_tags_as_observations_not_merge():
    """A team whose tag changes over time produces multiple tag rows."""
    tags = derive_team_tags(
        [
            (100, "OLD", T1),
            (100, "NEW", T3),
        ]
    )
    assert {(t.team_id, t.tag) for t in tags} == {
        (100, "OLD"),
        (100, "NEW"),
    }


def test_derive_team_tags_is_deterministic():
    observations = [(100, "VP", T1), (200, "GG", T2), (100, "VP", T3)]
    assert derive_team_tags(observations) == derive_team_tags(
        list(reversed(observations))
    )


def test_load_team_organizations_config_parses_valid_entries(tmp_path):
    path = tmp_path / "team_organizations.yaml"
    path.write_text(
        "organizations:\n"
        "  - organization_id: 1\n"
        "    name: Virtus.pro\n"
        "    team_ids: [8724984, 9729720, 9895392]\n"
        "    source: slice1\n"
        "    reason: curated\n",
        encoding="utf-8",
    )
    entries = load_team_organizations_config(path)
    assert len(entries) == 1
    assert entries[0]["organization_id"] == 1
    assert entries[0]["team_ids"] == [8724984, 9729720, 9895392]


def test_load_team_organizations_config_rejects_duplicate_org_id(tmp_path):
    path = tmp_path / "team_organizations.yaml"
    path.write_text(
        "organizations:\n"
        "  - organization_id: 1\n"
        "    name: A\n"
        "    team_ids: [1]\n"
        "  - organization_id: 1\n"
        "    name: B\n"
        "    team_ids: [2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate organization_id"):
        load_team_organizations_config(path)


def test_load_team_organizations_config_rejects_shared_team_id(tmp_path):
    path = tmp_path / "team_organizations.yaml"
    path.write_text(
        "organizations:\n"
        "  - organization_id: 1\n"
        "    name: A\n"
        "    team_ids: [100]\n"
        "  - organization_id: 2\n"
        "    name: B\n"
        "    team_ids: [100]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="more than one organization"):
        load_team_organizations_config(path)


def test_load_team_organizations_config_rejects_empty_team_ids(tmp_path):
    path = tmp_path / "team_organizations.yaml"
    path.write_text(
        "organizations:\n"
        "  - organization_id: 1\n"
        "    name: A\n"
        "    team_ids: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no team_ids"):
        load_team_organizations_config(path)