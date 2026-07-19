"""Academy Agents integration for ChemGraph.

Public re-exports come in two tiers so the package honours the
``[academy]`` optional-dep contract:

* **Eager** (pure stdlib + pydantic, always importable):
  ``ChemGraphAgentSpec``, ``ChemGraphCampaign``,
  ``ChemGraphDaemonConfig``, ``MCPServerSpec``, ``ResourceSpec``,
  ``load_campaign``, ``resolve_campaign_resources``,
  ``PromptProfile``, ``load_prompt_profile``, ``CampaignEvent``,
  ``EventLog``. These let the dashboard, ``--trace-dir``, and the
  observability tooling work on a checkout without ``academy-py``
  installed.
* **Lazy** (resolved via ``__getattr__`` on first access; requires
  the ``[academy]`` extra): ``ChemGraphLogicalAgent``. Importing it
  pulls in ``academy.agent``; without the extra installed, access
  raises ``ImportError`` with a hint instead of crashing the package
  import.

This split exists because ``chemgraph.cli.trace`` (single-agent
``--trace-dir`` flow) and the test collector both touch
``chemgraph.academy`` via leaf submodules; eager-importing the
academy-py-dependent ``ChemGraphLogicalAgent`` here broke those code
paths for users without the optional extra.
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
                f"Importing {name!r} from chemgraph.academy requires "
                f"the 'academy' optional extra: "
                f"`pip install 'chemgraph[academy]'`. "
                f"Underlying error: {exc}"
            ) from exc
        return getattr(module, attr)
    raise AttributeError(
        f"module 'chemgraph.academy' has no attribute {name!r}"
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
