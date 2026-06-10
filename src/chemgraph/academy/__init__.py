"""Academy Agents integration for ChemGraph.

Provides agent classes and utilities for deploying ChemGraph workflows
across federated HPC infrastructure using the Academy framework.

Requires the ``academy`` optional extra.
"""

from __future__ import annotations

from chemgraph.academy.core.agent import ChemGraphLogicalAgent
from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.core.campaign import MCPServerSpec
from chemgraph.academy.core.campaign import ResourceSpec
from chemgraph.academy.core.campaign import load_campaign
from chemgraph.academy.core.campaign import resolve_campaign_resources
from chemgraph.academy.observability.event_log import CampaignEvent
from chemgraph.academy.observability.event_log import EventLog
from chemgraph.academy.core.prompt import PromptProfile
from chemgraph.academy.core.prompt import load_prompt_profile


__all__ = [
    "CampaignEvent",
    "ChemGraphAgentSpec",
    "ChemGraphCampaign",
    "ChemGraphDaemonConfig",
    "EventLog",
    "MCPServerSpec",
    "PromptProfile",
    "ResourceSpec",
    "ChemGraphLogicalAgent",
    "load_campaign",
    "load_prompt_profile",
    "resolve_campaign_resources",
]
