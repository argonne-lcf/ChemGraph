"""Core ChemGraph Academy campaign contracts and agent logic."""

from chemgraph.academy.core.agent import ChemGraphLogicalAgent
from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.core.campaign import ResourceSpec
from chemgraph.academy.core.campaign import ToolSpec
from chemgraph.academy.core.campaign import load_campaign
from chemgraph.academy.core.campaign import resolve_campaign_resources
from chemgraph.academy.core.lm import LLMSettings
from chemgraph.academy.core.lm import load_lm_config
from chemgraph.academy.core.prompt import PromptProfile
from chemgraph.academy.core.prompt import load_prompt_profile
from chemgraph.academy.core.turn import ChemGraphReasoningRoundEngine
from chemgraph.academy.core.turn import ReasoningTurnResult

__all__ = [
    "ChemGraphAgentSpec",
    "ChemGraphCampaign",
    "ChemGraphDaemonConfig",
    "ChemGraphLogicalAgent",
    "ChemGraphReasoningRoundEngine",
    "LLMSettings",
    "PromptProfile",
    "ReasoningTurnResult",
    "ResourceSpec",
    "ToolSpec",
    "load_campaign",
    "load_lm_config",
    "load_prompt_profile",
    "resolve_campaign_resources",
]
