"""Thin Academy replacement layer for graph-to-graph ChemGraph demos.

``chemgraph.academy_sim`` is intentionally independent of ``chemgraph.academy``.
It treats ChemGraph graphs as black boxes and uses Academy only as a transport
for peer messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chemgraph.academy_sim.config import AcademySimConfig
from chemgraph.academy_sim.config import ExchangeConfig
from chemgraph.academy_sim.config import GraphConfig
from chemgraph.academy_sim.config import load_config
from chemgraph.academy_sim.envelopes import PeerEnvelope
from chemgraph.academy_sim.envelopes import build_envelope

if TYPE_CHECKING:
    from chemgraph.academy_sim.agent import ChemGraphSimAgent


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ChemGraphSimAgent": (
        "chemgraph.academy_sim.agent",
        "ChemGraphSimAgent",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        try:
            from importlib import import_module

            module = import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Importing {name!r} from chemgraph.academy_sim requires "
                f"the 'academy' optional extra: "
                f"`pip install 'chemgraphagent[academy]'`. "
                f"Underlying error: {exc}"
            ) from exc
        return getattr(module, attr)
    raise AttributeError(
        f"module 'chemgraph.academy_sim' has no attribute {name!r}"
    )


__all__ = [
    "AcademySimConfig",
    "ChemGraphSimAgent",
    "ExchangeConfig",
    "GraphConfig",
    "PeerEnvelope",
    "build_envelope",
    "load_config",
]
