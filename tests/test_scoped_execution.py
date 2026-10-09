"""Hermetic embedding contracts; no models, downloads, or remote jobs."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from chemgraph.execution.scoped import (
    ARTIFACT_KEY,
    ExecutionContext,
    current_execution,
    decode_result,
    execute_scoped_tool,
    execution_context,
    result_artifacts,
    scoped_calculator,
    scoped_path,
)
from chemgraph.registry.tools import ToolSpec
from chemgraph.tools.ase_core import _resolve_existing_path, _resolve_path


class Controller:
    def __init__(self):
        self.calls = []
        self.saved = {}

    async def execute_tool(self, context, call, invoke):
        self.calls.append((context.owner, call["id"]))
        if call["id"] not in self.saved:
            self.saved[call["id"]] = await invoke()
        return self.saved[call["id"]]


def scope(tmp_path, owner="alice", controller=None):
    return ExecutionContext(
        owner,
        owner + "-thread",
        owner + "-task",
        tmp_path / owner,
        controller or Controller(),
    )


def test_scoped_paths_and_standalone_compatibility(tmp_path, monkeypatch):
    context = scope(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("private")
    (context.workspace / "escape").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    with execution_context(context):
        assert _resolve_existing_path("secret.txt") == str(
            context.workspace / "secret.txt"
        )
        assert _resolve_path("out.json") == str(context.workspace / "out.json")
        for path in (
            outside,
            "../secret.txt",
            "escape/secret.txt",
            ".chemgraph-artifacts/x",
        ):
            with pytest.raises(PermissionError):
                scoped_path(path)
    assert current_execution() is None
    assert scoped_path(outside) is None
    assert _resolve_existing_path(str(outside)) == str(outside)


def test_calculator_resources_are_operator_controlled(tmp_path):
    context = scope(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    context.model_roots = (models,)
    with execution_context(context):
        assert scoped_calculator({"directory": "."})["directory"] == str(
            context.workspace
        )
        assert scoped_calculator({"model": "medium"}) == {"model": "medium"}
        assert scoped_calculator({"model": str(models / "weights.model")})[
            "model"
        ].endswith("weights.model")
        for params in (
            {"command": "touch /tmp/unsafe"},
            {"profile": {"command": "x"}},
            {"model": "/tmp/weights.model"},
            {"model": "https://host/weights"},
        ):
            with pytest.raises(PermissionError):
                scoped_calculator(params)


@pytest.mark.asyncio
async def test_controller_precedes_execution_and_artifacts_survive_overwrite(tmp_path):
    controller = Controller()
    context = scope(tmp_path, controller=controller)
    spec = ToolSpec("writer", "write an output", None)

    async def handler(request):
        assert controller.calls[-1] == ("alice", request.tool_call["id"])
        Path(_resolve_path("result.txt")).write_text(request.tool_call["args"]["value"])
        return ToolMessage(content="written", tool_call_id=request.tool_call["id"])

    messages = []
    with execution_context(context):
        for identifier, value in (
            ("one", "first"),
            ("two", "second"),
            ("one", "first"),
        ):
            request = SimpleNamespace(
                tool_call={"id": identifier, "name": "writer", "args": {"value": value}}
            )
            messages.append(await execute_scoped_tool(spec, request, handler))
    assert (context.workspace / "result.txt").read_text() == "second"
    artifacts = result_artifacts({"messages": messages}, "alice-task")
    assert [Path(a["path"]).read_text() for a in artifacts] == ["first", "second"]
    assert all(a["media_type"] == "text/plain" for a in artifacts)
    restored = decode_result(controller.saved["one"]["result"])
    assert restored.additional_kwargs[ARTIFACT_KEY][0] == artifacts[0]
    assert (
        result_artifacts({"messages": [m.model_dump() for m in messages]}, "bob-task")
        == []
    )


@pytest.mark.asyncio
async def test_controller_failure_never_invokes_tool(tmp_path):
    class Unavailable:
        async def execute_tool(self, context, call, invoke):
            raise OSError("journal unavailable")

    async def handler(request):
        pytest.fail("execution preceded acceptance")

    with execution_context(scope(tmp_path, controller=Unavailable())):
        with pytest.raises(OSError, match="journal unavailable"):
            await execute_scoped_tool(
                ToolSpec("writer", "", None),
                SimpleNamespace(tool_call={"id": "one"}),
                handler,
            )


def test_rag_state_is_scoped_without_loading_embeddings(tmp_path):
    from chemgraph.tools.rag_tools import _stores, get_loaded_documents

    alice, bob = scope(tmp_path), scope(tmp_path, "bob")
    with execution_context(alice):
        _stores()["alice.txt"] = object()
        _stores()["__latest__"] = "alice.txt"
    with execution_context(bob):
        assert get_loaded_documents() == []
        assert _stores().get("__latest__") is None
    with execution_context(alice):
        assert get_loaded_documents() == ["alice.txt"]
