from __future__ import annotations

import json

import pytest

from chemgraph.academy.core.campaign import campaign_bootstrap_text
from chemgraph.academy.core.campaign import load_campaign
from chemgraph.academy.core.campaign import MCPServerSpec
from chemgraph.academy.core.campaign import validate_campaign


def test_builtin_mace_campaign_uses_star_coordinator_without_routing_policy() -> None:
    campaign = load_campaign("mace-ensemble-screening-20")

    validate_campaign(campaign, len(campaign.agents))

    assert campaign.initial_agent == "coordinator-agent"
    assert [agent.name for agent in campaign.agents] == [
        "coordinator-agent",
        "structure-agent-a",
        "structure-agent-b",
        "mace-agent",
        "assessment-agent",
    ]
    peers = {agent.name: set(agent.allowed_peers) for agent in campaign.agents}
    assert peers["coordinator-agent"] == {
        "structure-agent-a",
        "structure-agent-b",
        "mace-agent",
        "assessment-agent",
    }
    assert peers["structure-agent-a"] == {"coordinator-agent"}
    assert peers["structure-agent-b"] == {"coordinator-agent"}
    assert peers["mace-agent"] == {"coordinator-agent"}
    assert peers["assessment-agent"] == {"coordinator-agent"}

    bootstrap = json.loads(campaign_bootstrap_text(campaign))
    assert "parameters" not in bootstrap
    assert "routing_policy" not in bootstrap


def test_removed_structured_orchestration_fields_are_rejected(tmp_path) -> None:
    campaign_path = tmp_path / "campaign.jsonc"
    campaign_path.write_text(
        json.dumps(
            {
                "run_id": "stale",
                "user_task": "test",
                "prompt_profile": "prompt.json",
                "parameters": {"old": "field"},
                "routing_policy": {"type": "old"},
                "agents": [
                    {
                        "name": "agent-a",
                        "role": "Role",
                        "mission": "Do the task.",
                        "allowed_peers": [],
                        "mcp_servers": [],
                    },
                ],
                "mcp_servers": [],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="removed structured orchestration"):
        load_campaign(campaign_path)


def test_campaign_loader_accepts_jsonc_comments(tmp_path) -> None:
    campaign_path = tmp_path / "campaign.jsonc"
    campaign_path.write_text(
        """
        {
          // User-facing campaign files may include comments.
          "run_id": "commented",
          "user_task": "test",
          "prompt_profile": "prompt.json",
          "resources": {
            /* Resource options are documented in the built-in examples. */
            "input": {
              "kind": "json",
              "path": "input.json",
              "scope": "campaign_file",
              "expose_content": false
            }
          },
          "agents": [
            {
              "name": "agent-a",
              "role": "Role",
              "mission": "Do the task.",
              "allowed_peers": [],
              "mcp_servers": ["general"],
              "resources": ["input"]
            }
          ],
          "mcp_servers": [
            {
              "name": "general",
              "command": "python -m chemgraph.mcp.mcp_tools"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    campaign = load_campaign(campaign_path)

    assert campaign.run_id == "commented"
    assert campaign.resources["input"].kind == "json"
    assert campaign.mcp_servers[0].name == "general"
    assert campaign.agents[0].mcp_servers == ("general",)


def test_mcp_server_spec_validation() -> None:
    spec = MCPServerSpec.model_validate(
        {"name": "general", "command": "python -m server"},
    )
    assert spec.env == {}

    with pytest.raises(ValueError, match="field required|Field required"):
        MCPServerSpec.model_validate({"name": "general"})

    with pytest.raises(ValueError):
        MCPServerSpec.model_validate(
            {"name": "general", "command": "python -m server", "extra": "bad"},
        )


def test_resource_kind_and_scope_are_option_sets(tmp_path) -> None:
    campaign_path = tmp_path / "campaign.jsonc"
    campaign_path.write_text(
        json.dumps(
            {
                "run_id": "bad-resource",
                "user_task": "test",
                "prompt_profile": "prompt.json",
                "resources": {
                    "input": {
                        "kind": "blob",
                        "path": "input.json",
                        "scope": "somewhere",
                    },
                },
                "agents": [
                    {
                        "name": "agent-a",
                        "role": "Role",
                        "mission": "Do the task.",
                        "allowed_peers": [],
                        "mcp_servers": [],
                    },
                ],
                "mcp_servers": [],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resource kind must be one of"):
        load_campaign(campaign_path)


def test_validate_campaign_rejects_unknown_mcp_server(tmp_path) -> None:
    campaign_path = tmp_path / "campaign.jsonc"
    campaign_path.write_text(
        json.dumps(
            {
                "run_id": "bad-server",
                "user_task": "test",
                "prompt_profile": "prompt.json",
                "mcp_servers": [],
                "agents": [
                    {
                        "name": "agent-a",
                        "role": "Role",
                        "mission": "Do the task.",
                        "allowed_peers": [],
                        "mcp_servers": ["missing"],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    campaign = load_campaign(campaign_path)
    with pytest.raises(RuntimeError, match="unknown MCP servers"):
        validate_campaign(campaign, 1)


def test_validate_campaign_rejects_duplicate_mcp_server_names(tmp_path) -> None:
    campaign_path = tmp_path / "campaign.jsonc"
    campaign_path.write_text(
        json.dumps(
            {
                "run_id": "duplicate-server",
                "user_task": "test",
                "prompt_profile": "prompt.json",
                "mcp_servers": [
                    {"name": "general", "command": "python -m one"},
                    {"name": "general", "command": "python -m two"},
                ],
                "agents": [
                    {
                        "name": "agent-a",
                        "role": "Role",
                        "mission": "Do the task.",
                        "allowed_peers": [],
                        "mcp_servers": ["general"],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    campaign = load_campaign(campaign_path)
    with pytest.raises(RuntimeError, match="MCP server names must be unique"):
        validate_campaign(campaign, 1)
