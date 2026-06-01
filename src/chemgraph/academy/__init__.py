"""Academy Agents integration for ChemGraph.

Provides agent classes and utilities for deploying ChemGraph workflows
across federated HPC infrastructure using the Academy framework.

Requires the ``academy`` optional extra::

    pip install chemgraphagent[academy]

Modules that depend on ``academy-py`` (agent, screening, coordinator)
use lazy imports so that the rate limiter and config utilities remain
usable without the optional dependency.
"""

from __future__ import annotations

from chemgraph.academy.config import AcademyConfig, build_manager
from chemgraph.academy.rate_limiter import RateLimiter


def __getattr__(name: str):  # noqa: N807
    """Lazy-import Academy-dependent classes."""
    if name == "ChemGraphAgent":
        from chemgraph.academy.agent import ChemGraphAgent

        return ChemGraphAgent
    if name == "ScreeningAgent":
        from chemgraph.academy.screening import ScreeningAgent

        return ScreeningAgent
    if name == "CoordinatorAgent":
        from chemgraph.academy.coordinator import CoordinatorAgent

        return CoordinatorAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChemGraphAgent",
    "AcademyConfig",
    "build_manager",
    "RateLimiter",
    "ScreeningAgent",
    "CoordinatorAgent",
]
