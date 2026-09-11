"""Hermetic tests for per-call Globus Transfer destination selection."""

from unittest.mock import MagicMock

import pytest

from chemgraph.execution.config import get_transfer_manager
from chemgraph.execution.globus_transfer import GlobusTransferManager
from chemgraph.hpc_configs import list_facility_transfer_profiles
from chemgraph.mcp.transfer_tools import register_transfer_tools
from tests.test_globus_transfer import _FakeMCP

EAGLE = "05d2c76a-e867-4f67-aa57-76edeb0beda0"
FLARE = "f39a7a0f-5bfc-46ce-9615-ba9f8592814f"
CUSTOM = "a915cc60-86aa-4dc7-b273-90bc2725fdb4"


@pytest.fixture
def staging(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "chemgraph.execution.config._load_execution_config", lambda _: {}
    )
    for name in (
        "COMPUTE_SYSTEM",
        "GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID",
        "GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID",
        "GLOBUS_TRANSFER_DESTINATION_BASE_PATH",
        "GLOBUS_TRANSFER_DESTINATION_COMPUTE_BASE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    local_file = tmp_path / "water.xyz"
    local_file.write_text("3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n")
    client = MagicMock()
    client.submit_transfer.return_value = {"task_id": "task-id"}
    client.get_task.return_value = {"status": "SUCCEEDED"}
    client.operation_ls.return_value = [{"name": "water.xyz", "type": "file"}]

    def get_client(manager):
        assert manager.allow_interactive_auth is False
        return client

    monkeypatch.setattr(GlobusTransferManager, "_get_transfer_client", get_client)
    manager = get_transfer_manager(
        system="aurora",
        source_endpoint_id="source-id",
        destination_base_path="/Project/staging",
        allow_interactive_auth=False,
    )
    return local_file, client, manager


@pytest.mark.parametrize(
    "selectors,destination,compute_root",
    [
        ({}, FLARE, "/flare"),
        ({"compute_system": "polaris"}, EAGLE, "/eagle"),
        ({"compute_system": "crux"}, EAGLE, "/eagle"),
        ({"compute_system": "aurora"}, FLARE, "/flare"),
        ({"destination_endpoint_id": EAGLE}, EAGLE, "/eagle"),
        ({"destination_endpoint_id": CUSTOM}, CUSTOM, ""),
        ({"compute_system": "crux", "destination_endpoint_id": FLARE}, FLARE, "/flare"),
        ({"compute_system": "polaris", "destination_endpoint_id": CUSTOM}, CUSTOM, ""),
        ({"source_endpoint_id": "alternate-source"}, FLARE, "/flare"),
    ],
)
def test_transfer_and_listing_use_per_call_destination(
    staging,
    selectors,
    destination,
    compute_root,
):
    local_file, client, manager = staging
    defaults = vars(manager).copy()
    mcp = _FakeMCP()
    register_transfer_tools(mcp, manager)
    result = mcp.tools["transfer_files"](
        str(local_file),
        remote_subdir="batch",
        **selectors,
    )
    submitted = client.submit_transfer.call_args.args[0]
    assert submitted["source_endpoint"] == selectors.get(
        "source_endpoint_id", "source-id"
    )
    assert submitted["destination_endpoint"] == destination
    assert submitted["DATA"][0]["source_path"] == str(local_file.resolve())
    assert (
        submitted["DATA"][0]["destination_path"] == "/Project/staging/batch/water.xyz"
    )
    assert result["status"] == "completed"
    assert result["destination_endpoint_id"] == destination
    assert result["source_endpoint_id"] == submitted["source_endpoint"]
    assert result["remote_directory"] == f"{compute_root}/Project/staging/batch"
    assert result["transfer_directory"] == "/Project/staging/batch"
    listing_selectors = {
        k: v for k, v in selectors.items() if k != "source_endpoint_id"
    }
    assert mcp.tools["list_remote_files"](
        result["transfer_directory"], **listing_selectors
    )
    client.operation_ls.assert_called_once_with(
        destination, path=result["transfer_directory"]
    )
    assert vars(manager) == defaults
    mcp.tools["transfer_files"](str(local_file), wait=False)
    assert client.submit_transfer.call_args.args[0]["destination_endpoint"] == FLARE


def test_crux_name_is_preserved_for_shared_collection(staging):
    _, _, defaults = staging
    manager = get_transfer_manager(default_manager=defaults, system="crux")
    assert manager.system == "crux"
    mcp = _FakeMCP()
    register_transfer_tools(mcp, manager)
    discovery = mcp.tools["list_transfer_facilities"]()
    assert discovery["active_system"] == "crux"
    assert [p["system"] for p in discovery["facilities"] if p["active"]] == ["crux"]


@pytest.mark.parametrize("system", ["polaris", "crux"])
def test_call_system_overrides_configured_endpoint(staging, monkeypatch, system):
    monkeypatch.setenv("GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID", "source-id")
    monkeypatch.setenv("GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID", FLARE)
    monkeypatch.setenv("GLOBUS_TRANSFER_DESTINATION_BASE_PATH", "/Project/staging")
    monkeypatch.setenv(
        "GLOBUS_TRANSFER_DESTINATION_COMPUTE_BASE_PATH", "/flare/Project/staging"
    )
    manager = get_transfer_manager(system=system)
    assert manager.destination_endpoint_id == EAGLE
    assert manager.destination_compute_base_path == "/eagle/Project/staging"


def test_custom_mapping_stays_only_with_same_destination(staging):
    defaults = GlobusTransferManager(
        source_endpoint_id="source-id",
        destination_endpoint_id=CUSTOM,
        destination_base_path="/Project/staging",
        destination_compute_base_path="/custom/Project/staging",
    )
    same = get_transfer_manager(
        default_manager=defaults, destination_endpoint_id=CUSTOM
    )
    assert same.destination_compute_base_path == "/custom/Project/staging"
    changed = get_transfer_manager(default_manager=defaults, system="crux")
    assert changed.destination_compute_base_path == "/eagle/Project/staging"


def test_calls_can_fill_missing_defaults_and_poll(staging, monkeypatch):
    local_file, client, _ = staging
    mcp = _FakeMCP()
    register_transfer_tools(mcp, None)
    with pytest.raises(ValueError, match="destination_base_path"):
        mcp.tools["transfer_files"](
            str(local_file),
            compute_system="polaris",
            source_endpoint_id="source-id",
        )
    monkeypatch.setenv("GLOBUS_TRANSFER_DESTINATION_BASE_PATH", "/Project/staging")
    with pytest.raises(ValueError, match="source_endpoint_id"):
        mcp.tools["transfer_files"](str(local_file), compute_system="polaris")
    client.submit_transfer.assert_not_called()
    result = mcp.tools["transfer_files"](
        str(local_file),
        compute_system="polaris",
        source_endpoint_id="source-id",
        wait=False,
    )
    assert result["destination_endpoint_id"] == EAGLE
    assert (
        mcp.tools["check_transfer_status"](result["task_id"])["status"] == "SUCCEEDED"
    )
    with pytest.raises(ValueError, match="source_endpoint_id"):
        mcp.tools["transfer_files"](str(local_file))


@pytest.mark.asyncio
async def test_mcp_schema_and_validation_follow_registry(staging):
    from fastmcp import Client
    from mcp.server.fastmcp import FastMCP

    local_file, transfer_client, manager = staging
    mcp = FastMCP("transfer-test")
    register_transfer_tools(mcp, manager)
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
        for name in ("transfer_files", "list_remote_files"):
            schema = tools[name].inputSchema
            assert schema["$defs"]["TransferComputeSystem"]["enum"] == [
                p.system for p in list_facility_transfer_profiles()
            ]
            assert "compute_system" not in schema["required"]
            assert "destination_endpoint_id" not in schema["required"]
            for selectors in (
                {"compute_system": "unsupported"},
                {"compute_system": "unsupported", "destination_endpoint_id": CUSTOM},
                {"destination_endpoint_id": " "},
            ):
                arguments = (
                    {"source_paths": str(local_file)}
                    if name == "transfer_files"
                    else {"remote_path": "/Project/staging"}
                )
                result = await client.call_tool(
                    name, {**arguments, **selectors}, raise_on_error=False
                )
                assert result.is_error
        transfer_client.submit_transfer.assert_not_called()
        transfer_client.operation_ls.assert_not_called()


@pytest.mark.asyncio
async def test_scripted_agent_stages_to_system_without_uuid(staging):
    from fastmcp import Client
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_mcp_adapters.tools import load_mcp_tools
    from mcp.server.fastmcp import FastMCP

    from chemgraph.graphs.deep_agent import construct_deep_agent_graph
    from tests.test_main_agent import _ScriptedChatModel

    local_file, client, manager = staging
    mcp = FastMCP("transfer-test")
    register_transfer_tools(mcp, manager)
    model = _ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "transfer_files",
                        "id": "stage",
                        "type": "tool_call",
                        "args": {
                            "source_paths": str(local_file),
                            "compute_system": "polaris",
                        },
                    }
                ],
            ),
            AIMessage(content="Files staged."),
        ]
    )
    async with Client(mcp) as connection:
        graph = construct_deep_agent_graph(
            model,
            tools=await load_mcp_tools(connection.session),
            checkpointer=None,
        )
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content=f"Stage {local_file} to Polaris"),
                ]
            }
        )
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "success"
    assert "/eagle/Project/staging" in str(tool_message.content)
    assert client.submit_transfer.call_args.args[0]["destination_endpoint"] == EAGLE
