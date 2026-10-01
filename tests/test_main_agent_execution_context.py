"""Restart and isolation regressions for the execution context behind approvals."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.agent.main_session import MainAgentSession, IncompatibleCheckpointError
from chemgraph.cli import commands
from chemgraph.cli.checkpoint_runtime import CheckpointRuntime
from chemgraph.graphs.workspace import create_cli_workspace_backend
from chemgraph.memory.store import SessionStore
from chemgraph.models.endpoints import PreparedModel
from chemgraph.models.endpoints.identity import describe_model_endpoint
from chemgraph.registry.tools import ToolRegistry
from chemgraph.utils.artifacts import artifact_context, artifact_directory
from tests.test_main_agent import _ScriptedChatModel


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name, "type": "tool_call"}])


@pytest.fixture
def model_loader(monkeypatch, tmp_path):
    monkeypatch.setattr("chemgraph.memory.store.DEFAULT_DB_DIR", str(tmp_path))
    monkeypatch.setattr("chemgraph.memory.store.DEFAULT_DB_PATH", str(tmp_path / "sessions.db"))
    requests = []
    responses = []

    def load(**kwargs):
        requests.append(kwargs)
        model = _ScriptedChatModel(responses=list(responses))
        endpoint = kwargs.get("endpoint")
        prepared = PreparedModel(endpoint_name="test", protocol="openai_compatible",
                                 client_kwargs={"model": kwargs["model_name"], "base_url": kwargs.get("base_url")})
        if endpoint is not None:
            prepared = replace(prepared, endpoint_descriptor=endpoint)
        return model, prepared

    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", load)
    monkeypatch.setattr(commands, "check_api_keys", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(commands.Confirm, "ask", lambda *args, **kwargs: True)
    monkeypatch.setattr(commands.time, "sleep", lambda *_args: None)
    return responses, requests


@pytest.mark.parametrize("startup", [False, True])
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("worker", [False, True])
def test_sqlite_cli_restores_chemistry_artifacts_and_endpoint(monkeypatch, tmp_path, model_loader,
                                                            startup, decision, worker):
    responses, requests = model_loader
    from ase import io
    executions = []
    write_atoms = io.write
    def record_write(path, *args, **kwargs):
        executions.append(str(path))
        return write_atoms(path, *args, **kwargs)
    monkeypatch.setattr(io, "write", record_write)
    saved, current = tmp_path / "saved", tmp_path / "current"
    saved.mkdir()
    current.mkdir()
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(saved))
    monkeypatch.setenv("VLLM_BASE_URL", "https://saved.example/v1")
    write = call("save_atomsdata_to_file", atomsdata={"numbers": [1], "positions": [[0, 0, 0]]}, fname="approved.xyz")
    responses[:] = ([call("load_agents", names=["single_agent"]),
                     call("task", subagent_type="single_agent", description="save a molecule"),
                     call("smiles_to_coordinate_file", smiles="O", output_file="approved.xyz")]
                    if worker else [call("load_tools", names=["save_atomsdata_to_file"]), write])
    database = str(tmp_path / "checkpoints.db")
    with CheckpointRuntime() as runtime:
        agent = ChemGraph(model_name="test", workflow_type="main_agent", base_url="https://saved.example/v1",
                          log_dir=str(saved), discover_skills=False,
                          tool_registry=ToolRegistry([ToolRegistry().get_spec("save_atomsdata_to_file")]),
                          subagent_names=["single_agent"] if worker else [],
                          checkpointer=runtime.open_sqlite(database))
        session = commands.create_main_agent_session(agent, thread_id="artifact", checkpoint_db=database)
        assert runtime.run(lambda: session.run("save the atom")).status == "waiting_for_user"
        config = agent.main_agent_metadata.graph_config
        assert config.cli_restorable
    assert SessionStore().get_session("artifact").log_dir == str(saved.resolve())
    monkeypatch.chdir(current)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(current))
    monkeypatch.setenv("VLLM_BASE_URL", "https://current.example/v1")
    responses[:] = [AIMessage(content="worker done"), AIMessage(content="done")]
    replies = iter([decision, "/quit"] if startup else ["test", "main_agent", "/resume artifact", decision, "/quit"])
    monkeypatch.setattr(commands.Prompt, "ask", lambda *args, **kwargs: next(replies))
    with commands.console.capture() as output:
        commands.interactive_mode(model="test", workflow="main_agent", base_url="https://current.example/v1",
                                  resume_session="artifact" if startup else None, checkpoint_db=database)
    assert "could not" not in output.get().lower()
    assert (saved / "approved.xyz").exists() == (decision == "approve"), output.get()
    assert not (current / "approved.xyz").exists()
    assert executions == ([str(saved / "approved.xyz")] if decision == "approve" else [])
    assert requests[-1]["endpoint"] == config.model_endpoint
    assert SessionStore().get_session("artifact").status == "completed"
    assert artifact_directory() == str(current)


def test_canonical_context_and_endpoint_identity(monkeypatch, tmp_path, model_loader):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "artifacts"
    root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)

    def config(directory, url="https://first.example/v1", **kwargs):
        return ChemGraph(workflow_type="main_agent", log_dir=str(directory), base_url=url,
                         enable_memory=False, discover_skills=False, subagent_names=[], **kwargs).runtime_config.saved

    original = config("artifacts")
    assert original.topology_fingerprint == config(link).topology_fingerprint
    assert original.topology_fingerprint != config(tmp_path / "different").topology_fingerprint
    assert original.topology_fingerprint != config(root, "https://second.example/v1").topology_fingerprint
    for url in ("https://user:secret@example.test/v1", "https://example.test/v1?token=secret", "https://example.test/v1#secret"):
        opaque = config(root, url)
        assert not opaque.cli_restorable and opaque.requires_configuration_id
        assert "secret" not in opaque.model_dump_json()
        with pytest.raises(ValueError, match="Python"):
            commands._main_agent_options(opaque)


@pytest.mark.asyncio
async def test_interleaved_tools_inherit_context_without_environment_mutation(monkeypatch, tmp_path):
    from chemgraph.tools.ase_tools import save_atomsdata_to_file
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "ambient"))
    barrier = asyncio.Barrier(2)

    async def write(directory):
        with artifact_context(str(directory)):
            await barrier.wait()
            await save_atomsdata_to_file.ainvoke({"atomsdata": {"numbers": [1], "positions": [[0, 0, 0]]},
                                                 "fname": "atom.xyz"})
    await asyncio.gather(write(tmp_path / "one"), write(tmp_path / "two"))
    assert (tmp_path / "one/atom.xyz").exists() and (tmp_path / "two/atom.xyz").exists()
    assert not (tmp_path / "ambient/atom.xyz").exists()
    assert artifact_directory() == str(tmp_path / "ambient")


def test_shell_backend_is_bound_per_session(monkeypatch, tmp_path, model_loader):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "ambient"))
    backend = create_cli_workspace_backend(tmp_path)
    agents = [ChemGraph(workflow_type="main_agent", enable_memory=False, backend=backend,
                        discover_skills=False, log_dir=str(tmp_path / name)) for name in ("one", "two")]
    assert backend._env["CHEMGRAPH_LOG_DIR"] == str(tmp_path / "ambient")
    for agent in agents:
        result = agent.backend.execute('printf "%s" "$CHEMGRAPH_LOG_DIR"')
        assert result.output == agent.log_dir
        assert agent.runtime_config.saved.cli_restorable


def test_loader_pins_resolved_endpoint_without_changing_request_defaults(monkeypatch):
    from chemgraph.models import loader
    from chemgraph.models.endpoints import openai_direct
    built = []

    def build(kwargs):
        built.append(kwargs)
        import os
        return SimpleNamespace(root_client=SimpleNamespace(base_url=kwargs.get("base_url") or os.environ["OPENAI_BASE_URL"]))

    spec = replace(openai_direct.SPEC, protocol_build=build)
    monkeypatch.setattr(loader, "_select_endpoint", lambda _request: spec)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://saved.example/v1")
    _, prepared = loader.load_chat_model_prepared("gpt-4o-mini", api_key="old-secret")
    endpoint = prepared.endpoint_descriptor
    assert endpoint.base_url == "https://saved.example/v1" and not endpoint.configured_base_url
    monkeypatch.setenv("OPENAI_BASE_URL", "https://changed.example/v1")
    _, restored = loader.load_chat_model_prepared("gpt-4o-mini", endpoint=endpoint, api_key="new-secret")
    assert restored.endpoint_descriptor == endpoint
    assert built[1]["base_url"] == "https://saved.example/v1"
    assert built[0]["max_tokens"] == built[1]["max_tokens"]
    assert built[1]["api_key"] == "new-secret"
    assert "secret" not in endpoint.model_dump_json()


def test_actual_client_route_is_described():
    prepared = PreparedModel(endpoint_name="openai_direct", protocol="openai_compatible", client_kwargs={"model": "test"})
    client = SimpleNamespace(root_client=SimpleNamespace(base_url="https://EXAMPLE.test:443/v1/"))
    assert describe_model_endpoint(prepared, client, "test").base_url == "https://example.test/v1"


@pytest.mark.asyncio
async def test_changed_context_rejected_before_pending_action(tmp_path, model_loader):
    from langgraph.checkpoint.memory import InMemorySaver
    responses, _ = model_loader
    responses[:] = [call("write_file", file_path="/test.txt", content="approved")]
    saver = InMemorySaver()
    first = ChemGraph(workflow_type="main_agent", enable_memory=False, discover_skills=False,
                      log_dir=str(tmp_path / "one"), checkpointer=saver)
    session = MainAgentSession(first.workflow, thread_id="context", session_metadata=first.main_agent_metadata)
    await session.run("write")
    changed = ChemGraph(workflow_type="main_agent", enable_memory=False, discover_skills=False,
                        log_dir=str(tmp_path / "two"), checkpointer=saver)
    with pytest.raises(IncompatibleCheckpointError):
        await MainAgentSession(changed.workflow, thread_id="context", session_metadata=changed.main_agent_metadata).restore()


@pytest.mark.asyncio
async def test_retry_restores_context_after_failure(monkeypatch, tmp_path):
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from chemgraph.tools.ase_core import _resolve_path
    from pathlib import Path

    attempts = []
    def write(_state):
        attempts.append(artifact_directory())
        if len(attempts) == 1:
            raise RuntimeError("try again")
        Path(_resolve_path("retried.txt")).write_text("done")
        return {"messages": [AIMessage(content="done")]}
    graph = StateGraph(MessagesState)
    graph.add_node("write", write)
    graph.add_edge(START, "write")
    graph.add_edge("write", END)
    saved, current = str(tmp_path / "saved"), str(tmp_path / "current")
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", saved)
    session = MainAgentSession(graph.compile(checkpointer=InMemorySaver()))
    with pytest.raises(RuntimeError, match="try again"):
        await session.run("write")
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", current)
    assert (await session.retry()).status == "completed"
    assert attempts == [saved, saved]
    assert (tmp_path / "saved/retried.txt").read_text() == "done"
    assert artifact_directory() == current


def test_missing_execution_context_is_rejected_before_initialization(monkeypatch, tmp_path, model_loader):
    agent = ChemGraph(workflow_type="main_agent", enable_memory=False, log_dir=str(tmp_path))
    monkeypatch.setattr(commands, "initialize_agent", lambda *args, **kwargs: pytest.fail("must fail before construction"))
    for changes in ({"artifact_directory": None}, {"model_endpoint": None}, {"graph_schema_version": 3}):
        with pytest.raises(ValueError, match="Start a new session"):
            commands._initialize_saved_main_agent(agent.runtime_config.saved.model_copy(update=changes),
                                                  return_option="state", checkpointer=None)
