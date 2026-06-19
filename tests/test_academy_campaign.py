from __future__ import annotations

import json

import pytest

# Skip the whole module when the optional 'academy' extra is absent.
# Even though this file only touches the pure-stdlib parts of
# chemgraph.academy, the import guard is applied uniformly across the
# academy test suite so pytest collection stays clean on a CPU-only
# checkout without per-test bookkeeping.
pytest.importorskip("academy")

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


def test_agent_allowed_tools_parses(tmp_path) -> None:
    campaign_path = tmp_path / "campaign.jsonc"
    campaign_path.write_text(
        json.dumps(
            {
                "run_id": "allowed-tools-ok",
                "user_task": "test",
                "prompt_profile": "prompt.json",
                "mcp_servers": [
                    {"name": "general", "command": "python -m chemgraph.mcp.mcp_tools"},
                ],
                "agents": [
                    {
                        "name": "agent-a",
                        "role": "Role",
                        "mission": "Do the task.",
                        "allowed_peers": [],
                        "mcp_servers": ["general"],
                        "allowed_tools": ["run_ase", "extract_output_json"],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    campaign = load_campaign(campaign_path)
    validate_campaign(campaign, 1)

    assert campaign.agents[0].allowed_tools == (
        "run_ase",
        "extract_output_json",
    )


def test_agent_allowed_tools_defaults_to_empty(tmp_path) -> None:
    campaign_path = tmp_path / "campaign.jsonc"
    campaign_path.write_text(
        json.dumps(
            {
                "run_id": "allowed-tools-default",
                "user_task": "test",
                "prompt_profile": "prompt.json",
                "mcp_servers": [
                    {"name": "general", "command": "python -m chemgraph.mcp.mcp_tools"},
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
    validate_campaign(campaign, 1)

    assert campaign.agents[0].allowed_tools == ()


def test_validate_campaign_rejects_duplicate_allowed_tools(tmp_path) -> None:
    campaign_path = tmp_path / "campaign.jsonc"
    campaign_path.write_text(
        json.dumps(
            {
                "run_id": "duplicate-allowed-tools",
                "user_task": "test",
                "prompt_profile": "prompt.json",
                "mcp_servers": [
                    {"name": "general", "command": "python -m chemgraph.mcp.mcp_tools"},
                ],
                "agents": [
                    {
                        "name": "agent-a",
                        "role": "Role",
                        "mission": "Do the task.",
                        "allowed_peers": [],
                        "mcp_servers": ["general"],
                        "allowed_tools": ["run_ase", "run_ase"],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    campaign = load_campaign(campaign_path)
    with pytest.raises(RuntimeError, match="duplicate allowed_tools"):
        validate_campaign(campaign, 1)


def test_validate_campaign_rejects_allowed_tools_without_servers(tmp_path) -> None:
    campaign_path = tmp_path / "campaign.jsonc"
    campaign_path.write_text(
        json.dumps(
            {
                "run_id": "allowed-tools-no-servers",
                "user_task": "test",
                "prompt_profile": "prompt.json",
                "mcp_servers": [],
                "agents": [
                    {
                        "name": "agent-a",
                        "role": "Role",
                        "mission": "Do the task.",
                        "allowed_peers": [],
                        "mcp_servers": [],
                        "allowed_tools": ["run_ase"],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    campaign = load_campaign(campaign_path)
    with pytest.raises(
        RuntimeError,
        match="allowed_tools but no mcp_servers",
    ):
        validate_campaign(campaign, 1)


# ---------------------------------------------------------------------------
# Phase B.1: filter_agents + parse_agents_selection (federated spawn-site)
# ---------------------------------------------------------------------------


def test_parse_agents_selection_handles_trimming_and_empty_segments() -> None:
    """The CLI's ``--agents`` value passes through this helper before
    reaching the daemon. Tolerate whitespace + trailing commas so
    operators don't get bitten by shell-quoting quirks."""
    from chemgraph.academy.core.campaign import parse_agents_selection
    assert parse_agents_selection(None) == ()
    assert parse_agents_selection("") == ()
    assert parse_agents_selection("worker-a") == ("worker-a",)
    assert parse_agents_selection(" worker-a , worker-b ") == ("worker-a", "worker-b")
    assert parse_agents_selection("worker-a,,worker-b,") == ("worker-a", "worker-b")


def test_filter_agents_returns_slice_in_caller_order() -> None:
    """MPI rank-to-agent mapping must match the order the operator
    asked for. Don't accidentally re-sort or use the campaign's
    declaration order."""
    from chemgraph.academy.core.campaign import filter_agents, load_campaign
    campaign = load_campaign("mace-ensemble-screening-20")
    selected = filter_agents(campaign, ["mace-agent", "structure-agent-a"])
    assert [a.name for a in selected.agents] == ["mace-agent", "structure-agent-a"]
    # initial_agent is intentionally NOT rewritten -- in the federated
    # flow it may name an agent hosted on another site.
    assert selected.initial_agent == campaign.initial_agent
    # Untouched campaign fields stay intact.
    assert selected.resources == campaign.resources
    assert selected.mcp_servers == campaign.mcp_servers


def test_filter_agents_rejects_unknown_names() -> None:
    """An unknown name almost certainly indicates an operator typo or
    a campaign-file/CLI drift. Fail closed."""
    from chemgraph.academy.core.campaign import filter_agents, load_campaign
    campaign = load_campaign("mace-ensemble-screening-20")
    with pytest.raises(RuntimeError, match="not declared on campaign"):
        filter_agents(campaign, ["mace-agent", "no-such-agent"])


def test_filter_agents_rejects_empty_selection() -> None:
    """A zero-length slice means 'launch nothing,' which is never what
    the operator means. The launcher should never even construct an
    empty selection (parse_agents_selection returns () on no input,
    and the launcher short-circuits on empty), but the helper itself
    must still fail closed if reached."""
    from chemgraph.academy.core.campaign import filter_agents, load_campaign
    campaign = load_campaign("mace-ensemble-screening-20")
    with pytest.raises(RuntimeError, match="at least one agent"):
        filter_agents(campaign, [])


def test_filter_agents_rejects_duplicate_names() -> None:
    """Duplicates would shadow each other in the post-filter campaign
    and silently confuse the rank-to-agent mapping. Fail closed."""
    from chemgraph.academy.core.campaign import filter_agents, load_campaign
    campaign = load_campaign("mace-ensemble-screening-20")
    with pytest.raises(RuntimeError, match="duplicate agent names"):
        filter_agents(campaign, ["mace-agent", "mace-agent"])


def test_validate_campaign_federated_loosens_cross_site_peer_check() -> None:
    """In a federated spawn-site slice, allowed_peers / initial_agent
    may legitimately reference agents owned by another site. Strict
    validation (the default) rejects those; ``federated=True`` lets
    them through because the daemon will discover those peers via the
    exchange at runtime instead of from this slice's agent list."""
    from chemgraph.academy.core.campaign import (
        filter_agents, load_campaign, validate_campaign,
    )
    campaign = load_campaign("federated-hello")
    slice_aurora = filter_agents(campaign, ["agent-aurora"])

    # Strict validation rejects the cross-site peer reference.
    with pytest.raises(RuntimeError, match="unknown allowed peers"):
        validate_campaign(slice_aurora, agent_count=1)

    # federated=True accepts it.
    validate_campaign(slice_aurora, agent_count=1, federated=True)


def test_validate_campaign_federated_still_rejects_self_peer() -> None:
    """The 'agent must not list itself as a peer' invariant is local
    to the slice and stays a hard error even in federated mode --
    self-peering would loop messages back to the sender, regardless
    of how many sites the campaign spans."""
    from chemgraph.academy.core.campaign import (
        ChemGraphAgentSpec, ChemGraphCampaign, validate_campaign,
    )
    import pathlib
    bad = ChemGraphCampaign(
        run_id="r", user_task="t", initial_agent="a",
        prompt_profile=pathlib.Path("p"),
        agents=(ChemGraphAgentSpec(
            name="a", role="r", mission="m",
            allowed_peers=("a",),  # <-- self-peer
        ),),
    )
    with pytest.raises(RuntimeError, match="must not list itself as a peer"):
        validate_campaign(bad, agent_count=1, federated=True)
