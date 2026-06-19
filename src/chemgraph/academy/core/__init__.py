"""Core ChemGraph Academy campaign contracts and agent logic.

Re-exports split into two tiers to keep the ``[academy]`` optional-dep
contract:

* **Eager** (pure stdlib + pydantic + langchain_core): the campaign
  spec types, prompt profile, and reasoning-turn helpers. These are
  what the dashboard, ``--trace-dir``, and the test collector touch
  on a CPU-only checkout.
* **Lazy** (resolved via ``__getattr__``; requires the ``[academy]``
  extra because it depends on ``academy.agent.Agent``):
  ``ChemGraphLogicalAgent``.

Without this split, importing ``chemgraph.academy.core.campaign``
would transitively run ``core/__init__.py`` and pull in
``core.agent``, which fails when ``academy-py`` is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.core.campaign import MCPServerSpec
from chemgraph.academy.core.campaign import ResourceSpec
from chemgraph.academy.core.campaign import load_campaign
from chemgraph.academy.core.campaign import resolve_campaign_resources
from chemgraph.academy.core.prompt import PromptProfile
from chemgraph.academy.core.prompt import load_prompt_profile
from chemgraph.academy.core.turn import ReasoningTurnResult
from chemgraph.academy.core.turn import run_academy_turn


if TYPE_CHECKING:
    from chemgraph.academy.core.agent import ChemGraphLogicalAgent


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ChemGraphLogicalAgent": (
        "chemgraph.academy.core.agent",
        "ChemGraphLogicalAgent",
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
                f"Importing {name!r} from chemgraph.academy.core requires "
                f"the 'academy' optional extra: "
                f"`pip install 'chemgraph[academy]'`. "
                f"Underlying error: {exc}"
            ) from exc
        return getattr(module, attr)
    raise AttributeError(
        f"module 'chemgraph.academy.core' has no attribute {name!r}"
    )


__all__ = [
    "ChemGraphAgentSpec",
    "ChemGraphCampaign",
    "ChemGraphDaemonConfig",
    "ChemGraphLogicalAgent",
    "MCPServerSpec",
    "PromptProfile",
    "ReasoningTurnResult",
    "ResourceSpec",
    "load_campaign",
    "load_prompt_profile",
    "resolve_campaign_resources",
    "run_academy_turn",
]
