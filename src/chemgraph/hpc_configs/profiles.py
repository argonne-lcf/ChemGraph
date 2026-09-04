"""Public, backend-neutral facility metadata.

This module contains identifiers and path mappings that are shared by users of
a facility.  User-specific endpoint IDs, projects, paths, and credentials do
not belong here.
"""

from __future__ import annotations

import posixpath
import uuid
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class FacilityTransferProfile:
    """Public Globus Transfer metadata for one compute system."""

    system: str
    collection_id: str
    collection_name: str
    transfer_root: str
    compute_root: str
    documentation_url: str
    verified_on: date | None

    def __post_init__(self) -> None:
        uuid.UUID(self.collection_id)
        for name in ("transfer_root", "compute_root"):
            value = getattr(self, name)
            if not value.startswith("/"):
                raise ValueError(f"{name} must be an absolute POSIX path: {value!r}")

    @property
    def has_placeholder_collection_id(self) -> bool:
        """Whether the collection UUID still needs to be supplied by a user."""
        return uuid.UUID(self.collection_id).int == 0

    def compute_path(self, transfer_path: str) -> str:
        """Translate a collection-visible path to a compute-visible path."""
        path = _normalize_absolute_path(transfer_path)
        transfer_root = _normalize_absolute_path(self.transfer_root)
        compute_root = _normalize_absolute_path(self.compute_root)

        if transfer_root == "/":
            relative = path.lstrip("/")
        elif path == transfer_root:
            relative = ""
        elif path.startswith(f"{transfer_root}/"):
            relative = path[len(transfer_root) + 1 :]
        else:
            raise ValueError(
                f"Transfer path {path!r} is outside the {self.system} "
                f"collection root {transfer_root!r}."
            )

        if not relative:
            return compute_root
        return posixpath.join(compute_root, relative)


def _normalize_absolute_path(path: str) -> str:
    if not path or not path.startswith("/"):
        raise ValueError(f"Expected an absolute POSIX path, got {path!r}.")
    if ".." in path.split("/"):
        raise ValueError(f"Parent traversal is not allowed in path {path!r}.")
    return posixpath.normpath(path)


_FACILITY_TRANSFER_PROFILES = {
    "polaris": FacilityTransferProfile(
        system="polaris",
        collection_id="05d2c76a-e867-4f67-aa57-76edeb0beda0",
        collection_name="alcf#dtn_eagle",
        transfer_root="/eagle",
        compute_root="/eagle",
        documentation_url=(
            "https://docs.alcf.anl.gov/data-management/data-transfer/using-globus/"
        ),
        verified_on=date(2026, 9, 3),
    ),
    "aurora": FacilityTransferProfile(
        system="aurora",
        # Placeholder: replace this public UUID once the Flare collection ID is
        # ready to be maintained here. The factory will not use a nil UUID.
        collection_id="f39a7a0f-5bfc-46ce-9615-ba9f8592814f",
        collection_name="alcf#dtn_flare",
        transfer_root="/",
        compute_root="/flare",
        documentation_url=(
            "https://docs.alcf.anl.gov/aurora/data-management/"
            "moving_data_to_aurora/globus/"
        ),
        verified_on=None,
    ),
}


def get_facility_transfer_profile(
    system: str,
) -> FacilityTransferProfile | None:
    """Return the public Transfer profile for *system*, if one is bundled."""
    return _FACILITY_TRANSFER_PROFILES.get(system.strip().lower())
