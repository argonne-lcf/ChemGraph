"""CLI configuration changes reuse the normalized main-agent runtime."""

from chemgraph.cli import commands
from chemgraph.registry.tools import ToolRegistry
from tests.test_main_agent_execution_context import model_loader as model_loader


def test_switches_reuse_resolved_context(monkeypatch, tmp_path, model_loader):
    original = tmp_path / "original"
    original.mkdir()
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(original))
    agents = []
    initialize = commands.initialize_agent

    def record(*args, **kwargs):
        agent = initialize(*args, **kwargs)
        assert agent is not None
        agents.append(agent)
        return agent

    replies = iter(["test", "main_agent", "/model test-two", "/workflow single_agent",
                    "/workflow main_agent", "/quit"])
    def ask(*args, **kwargs):
        answer = next(replies)
        if answer.startswith("/model"):
            monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "changed"))
        return answer

    monkeypatch.setattr(commands, "initialize_agent", record)
    monkeypatch.setattr(commands.Prompt, "ask", ask)
    with commands.console.capture():
        commands.interactive_mode(
            workflow="main_agent", checkpoint_db=str(tmp_path / "checkpoints.db"),
            workspace=str(tmp_path), base_url="https://original.example/v1", discover_skills=False,
            subagent_names=[], tool_registry=ToolRegistry([]),
        )
    mains = [agent.runtime_config.saved for agent in agents if agent.workflow_type == "main_agent"]
    assert [config.model_name for config in mains] == ["test", "test-two", "test-two"]
    assert len({agent.session_id for agent in agents}) == len(agents)
    for config in mains:
        assert config.artifact_directory == str(original)
        assert config.model_endpoint.base_url == "https://original.example/v1"
        assert config.workspace == str(tmp_path)
        assert config.configured_subagent_names == config.registry_tool_names == ()
        assert config.discover_skills is False
