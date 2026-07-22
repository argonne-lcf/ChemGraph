"""swarm: federated multi-agent operating + authoring surface.

Split out of ChemGraph on 2026-07-07. See README.md for the layout.

Public re-exports come in two tiers:

* **Eager** (pure stdlib + pydantic, always importable):
  ``ChemGraphAgentSpec``, ``ChemGraphCampaign``,
  ``ChemGraphDaemonConfig``, ``MCPServerSpec``, ``ResourceSpec``,
  ``load_campaign``, ``resolve_campaign_resources``,
  ``PromptProfile``, ``load_prompt_profile``, ``CampaignEvent``,
  ``EventLog``. Usable without ``academy-py``.
* **Lazy** (resolved via ``__getattr__`` on first access; requires
  ``academy-py``): ``ChemGraphLogicalAgent``. Access raises
  ``ImportError`` with an actionable install hint if the dependency
  is missing.
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
from chemgraph.academy.observability.event_log import CampaignEvent
from chemgraph.academy.observability.event_log import EventLog


if TYPE_CHECKING:
    from chemgraph.academy.core.agent import ChemGraphLogicalAgent


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # public name -> (module path, attribute in that module)
    "ChemGraphLogicalAgent": (
        "chemgraph.academy.core.agent",
        "ChemGraphLogicalAgent",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy resolver for academy-py-dependent re-exports.

    Called by Python only when ``name`` is not found among the eager
    imports above. On ``ImportError`` we re-raise with an actionable
    hint so the operator knows which extra to install.
    """
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        try:
            from importlib import import_module
            module = import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Importing {name!r} from swarm requires the "
                f"academy-py package: `pip install academy-py`. "
                f"Underlying error: {exc}"
            ) from exc
        return getattr(module, attr)
    raise AttributeError(
        f"module 'swarm' has no attribute {name!r}"
    )


__all__ = [
    "CampaignEvent",
    "ChemGraphAgentSpec",
    "ChemGraphCampaign",
    "ChemGraphDaemonConfig",
    "ChemGraphLogicalAgent",
    "EventLog",
    "MCPServerSpec",
    "PromptProfile",
    "ResourceSpec",
    "load_campaign",
    "load_prompt_profile",
    "resolve_campaign_resources",
]
