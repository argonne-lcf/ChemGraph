"""Error types for the Academy simulation replacement layer."""

from __future__ import annotations


class AcademySimError(RuntimeError):
    """Base class for academy_sim runtime failures."""


class AcademySimConfigError(AcademySimError):
    """Raised when an academy_sim config is invalid."""


class PeerRegistrationError(AcademySimError):
    """Raised when peer identity discovery fails."""
