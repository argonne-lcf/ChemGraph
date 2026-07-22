from __future__ import annotations

from dataclasses import dataclass

import pytest

from chemgraph.mcp.globus_transfer_mcp import create_globus_transfer_mcp
from chemgraph.mcp.transfer_tools import register_transfer_tools


@dataclass
class _TransferResult:
    task_id: str
    remote_directory: str
    file_mapping: dict[str, str]


class _FakeTransferManager:
    def __init__(self) -> None:
        self.local_paths: list[str] = []

    def transfer_files(self, *, local_paths, remote_subdir=None, label=None):
        self.local_paths = local_paths
        return _TransferResult(
            task_id="task-1",
            remote_directory="/destination/demo",
            file_mapping={path: f"/destination/demo/{path.rsplit('/', 1)[-1]}" for path in local_paths},
        )

    def wait_for_transfer(self, task_id):
        assert task_id == "task-1"
        return {
            "status": "SUCCEEDED",
            "bytes_transferred": 42,
            "files_transferred": 1,
        }

    def check_transfer_status(self, task_id):
        return {"task_id": task_id, "status": "ACTIVE"}

    def list_remote_directory(self, path):
        return [{"name": "ethanol.xyz", "type": "file", "size": 42}]


class _FakeMCP:
    def __init__(self) -> None:
        self.tools = {}

    def add_tool(self, function, *, name, description):
        self.tools[name] = function


def test_transfer_tool_returns_destination_mapping_after_wait(tmp_path):
    source = tmp_path / "ethanol.xyz"
    source.write_text("1\nethanol\nH 0 0 0\n", encoding="utf-8")
    manager = _FakeTransferManager()
    mcp = _FakeMCP()
    register_transfer_tools(mcp, manager)

    result = mcp.tools["transfer_files"](str(source), wait=True)

    resolved = str(source.resolve())
    assert manager.local_paths == [resolved]
    assert result["status"] == "completed"
    assert result["file_mapping"] == {
        resolved: "/destination/demo/ethanol.xyz"
    }
    assert result["files_transferred"] == 1


def test_register_transfer_tools_exposes_complete_staging_surface():
    mcp = _FakeMCP()
    register_transfer_tools(mcp, _FakeTransferManager())

    assert set(mcp.tools) == {
        "transfer_files",
        "check_transfer_status",
        "list_remote_files",
    }


def test_dedicated_transfer_mcp_exposes_only_transfer_tools():
    mcp = create_globus_transfer_mcp(_FakeTransferManager())

    assert set(mcp._tool_manager._tools) == {
        "transfer_files",
        "check_transfer_status",
        "list_remote_files",
    }


def test_dedicated_transfer_mcp_requires_configuration(monkeypatch):
    monkeypatch.setattr(
        "chemgraph.mcp.globus_transfer_mcp.get_transfer_manager",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID"):
        create_globus_transfer_mcp()
