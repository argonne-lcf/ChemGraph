"""Tests for bundled facility Transfer metadata."""

from __future__ import annotations

import pytest

from chemgraph.hpc_configs import get_facility_transfer_profile


def test_polaris_profile_uses_public_eagle_collection():
    profile = get_facility_transfer_profile(" POLARIS ")

    assert profile is not None
    assert profile.collection_name == "alcf#dtn_eagle"
    assert profile.collection_id == "05d2c76a-e867-4f67-aa57-76edeb0beda0"
    assert not profile.has_placeholder_collection_id
    assert profile.compute_path("/eagle/MyProject/staging") == (
        "/eagle/MyProject/staging"
    )


def test_aurora_profile_maps_flare_collection_paths_to_compute_paths():
    profile = get_facility_transfer_profile("aurora")

    assert profile is not None
    assert profile.collection_name == "alcf#dtn_flare"
    assert profile.has_placeholder_collection_id
    assert profile.compute_path("/MyProject/staging") == (
        "/flare/MyProject/staging"
    )


def test_profile_rejects_paths_outside_collection_root():
    profile = get_facility_transfer_profile("polaris")

    assert profile is not None
    with pytest.raises(ValueError, match="outside the polaris collection root"):
        profile.compute_path("/flare/MyProject")
    with pytest.raises(ValueError, match="Parent traversal"):
        profile.compute_path("/eagle/MyProject/../OtherProject")


def test_unknown_system_has_no_transfer_profile():
    assert get_facility_transfer_profile("unknown") is None
