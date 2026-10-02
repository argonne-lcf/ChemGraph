"""The Deep Agent evaluation path formats its own trace before deterministic scoring."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from chemgraph.agent.usage import summarize_usage
from chemgraph.eval import deepagent
from chemgraph.eval.cli import build_config_from_args, parse_args
from chemgraph.eval.config import BenchmarkConfig
from chemgraph.eval.runner import ModelBenchmarkRunner
from chemgraph.graphs.deep_agent import _normalize_backend
from chemgraph.skills.runtime import prepare_skill_backend
from chemgraph.tools.ase_tools import file_to_atomsdata, save_atomsdata_to_file


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_output", [True, False], ids=["scored-answer", "formatting-failure"])
async def test_deepagent_evaluation(tmp_path, monkeypatch, valid_output):
    expected = {"smiles": ["O"]}
    dataset = tmp_path / "groundtruth.json"
    dataset.write_text(json.dumps([{
        "id": "water", "query": "Find the SMILES for water", "category": "lookup",
        "answer": {"structured_output": expected, "result": "GROUND_TRUTH_ONLY"},
    }]))
    profile = tmp_path / "config.toml"
    profile.write_text('[eval]\ndefault_profile = "test"\n'
                       '[eval.profiles.test]\nbase_url = "https://unused.example/v1"\n')
    proxy = "http://127.0.0.1:20219/v1"
    config = build_config_from_args(parse_args([
        "--config", str(profile), "--base-url", proxy,
        "--models", "fake", "--dataset", str(dataset), "--workflows", "deep_agent",
        "--judge-type", "structured", "--query-ids", "water",
        "--deepagent-auto-approve",
        "--output-dir", str(tmp_path / "results"),
    ]))
    state = {"messages": [{"type": "human", "content": "Find the SMILES for water"},
                          {"type": "ai", "content": "Water has SMILES O."}]}
    execution = AsyncMock(return_value=state)
    execution_usage = summarize_usage([{
        "counts": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, "complete": True,
    }])

    def make_agent(**options):
        assert options["base_url"] == proxy
        assert options["workflow_type"] == "deep_agent"
        assert options["deepagent_auto_approve"] and not options["deepagent_discover_skills"]
        assert not options["enable_memory"] and not options["structured_output"]
        assert os.environ["CHEMGRAPH_LOG_DIR"] == options["log_dir"]
        workspace = Path(options["log_dir"])
        assert workspace == tmp_path / "results/logs/fake/deep_agent/thread_water"
        assert str(workspace) in options["prompts"].deepagent

        # Exercise the graph's backend wrapping and real chemistry file tools.
        backend, _, _ = prepare_skill_backend(
            _normalize_backend(options["deepagent_backend"]), (), discover_skills=False,
        )
        written = backend.write(str(workspace / "eval-water.xyz"),
                                "3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n")
        assert written.error is None
        atoms = file_to_atomsdata.invoke({"fname": "eval-water.xyz"})
        assert atoms.numbers == [8, 1, 1]
        save_atomsdata_to_file.invoke({"atomsdata": atoms, "fname": "eval-water-copy.xyz"})
        saved = backend.read(str(workspace / "eval-water-copy.xyz"))
        assert saved.error is None and "Properties=species" in saved.file_data["content"]
        command = [sys.executable, "-c", (
            "import os; from pathlib import Path; "
            "assert Path.cwd() == Path(os.environ['CHEMGRAPH_LOG_DIR']).resolve(); "
            "print(Path('eval-water-copy.xyz').read_text())"
        )]
        shell = backend.execute(
            subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        )
        assert shell.exit_code == 0 and "Properties=species" in shell.output
        return SimpleNamespace(run=execution, last_usage=execution_usage)

    formatter = FakeMessagesListChatModel(responses=[AIMessage(
        content=[{"type": "text", "text": json.dumps(expected) if valid_output else "invalid JSON"}],
        usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
    )])
    original_invoke = formatter.ainvoke
    invocations = []

    async def format_trace(messages, **kwargs):
        assert "GROUND_TRUTH_ONLY" not in json.dumps(messages)
        assert "Water has SMILES O." in json.dumps(messages)
        invocations.append(messages)
        return await original_invoke(messages, **kwargs)

    def make_formatter(**options):
        assert options["base_url"] == proxy
        return SimpleNamespace(ainvoke=format_trace)

    monkeypatch.setattr(deepagent, "ChemGraph", make_agent)
    monkeypatch.setattr(deepagent, "load_chat_model", make_formatter)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", "original-logs")
    runner = ModelBenchmarkRunner(config)
    result = (await runner.run_all())["fake"]["deep_agent"]
    runner.report("all")

    aggregate = result["structured_judge_aggregate"]
    raw = result["raw_tool_calls"][0]
    assert raw["status"] == ("completed" if valid_output else "formatting_error"), raw.get("error")
    assert aggregate["n_queries"] == 1
    assert aggregate["n_correct"] == int(valid_output)
    assert raw["state"] == state and raw["result"] == "Water has SMILES O."
    assert len(invocations) == (1 if valid_output else 2)
    assert raw["usage"]["total"]["total_tokens"] == 12 + 25 * len(invocations)
    report = json.loads(next((tmp_path / "results").glob("benchmark_*.json")).read_text())
    assert report["results"]["fake"]["deep_agent"]["structured_judge_aggregate"] == aggregate
    assert os.environ["CHEMGRAPH_LOG_DIR"] == "original-logs"

    config.resume = True
    resumed = (await ModelBenchmarkRunner(config).run_all())["fake"]["deep_agent"]
    assert resumed == result
    execution.assert_awaited_once()
    assert len(invocations) == (1 if valid_output else 2)


@pytest.mark.parametrize("setting", [
    "execution_url", "judge_url", "credentials",
    "VLLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE",
    "ANTHROPIC_BASE_URL", "ANTHROPIC_API_URL", "GROQ_BASE_URL", "GROQ_API_BASE", "OLLAMA_HOST",
    "GOOGLE_GEMINI_BASE_URL", "GOOGLE_VERTEX_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "CHEMGRAPH_ARGO_MODEL_FORMAT",
])
def test_deepagent_checkpoint_endpoint_identity(tmp_path, monkeypatch, setting):
    dataset = tmp_path / "groundtruth.json"
    dataset.write_text(json.dumps([{"id": "water", "query": "Find the SMILES for water"}]))
    profile = tmp_path / "config.toml"
    execution_url = "https://execution.example/v1"
    judge_url = "https://judge.example/v1"

    def make_runner():
        profile.write_text(
            f'[api.openai]\nbase_url = "{execution_url}"\n'
            f'[api.anthropic]\nbase_url = "{judge_url}"\n'
        )
        return ModelBenchmarkRunner(BenchmarkConfig(
            models=["gpt-4o-mini"], workflow_types=["deep_agent"],
            judge_model="claude-sonnet-4-20250514", judge_type="llm",
            dataset=str(dataset), config_file=str(profile),
            output_dir=str(tmp_path / "results"), resume=True,
        ))

    monkeypatch.setattr("chemgraph.eval.runner.load_judge_model", lambda *args, **kwargs: object())
    if setting.isupper():
        monkeypatch.delenv(setting, raising=False)
    runner = make_runner()
    query_result = {"raw": {"result": "O"}, "judge": None, "structured_judge": None}
    runner._save_query_checkpoint("gpt-4o-mini", "deep_agent", "water", 0, query_result)
    assert make_runner()._load_checkpoint("gpt-4o-mini", "deep_agent") == {"water": query_result}

    changed_url = "https://changed.example/v1?token=dummy-endpoint-secret"
    if setting == "execution_url":
        execution_url = changed_url
    elif setting == "judge_url":
        judge_url = changed_url
    elif setting == "credentials":
        for name in ("OPENAI_API_KEY", "VLLM_API_KEY", "ANTHROPIC_API_KEY", "UNRELATED_SETTING"):
            monkeypatch.setenv(name, "dummy-rotated-secret")
    else:
        monkeypatch.setenv(setting, {
            "GOOGLE_GENAI_USE_VERTEXAI": "true", "GOOGLE_CLOUD_PROJECT": "changed-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1", "CHEMGRAPH_ARGO_MODEL_FORMAT": "wire",
        }.get(setting, changed_url))

    changed = make_runner()
    if setting == "credentials":
        assert changed._load_checkpoint("gpt-4o-mini", "deep_agent") == {"water": query_result}
    else:
        with pytest.raises(ValueError, match="Incompatible Deep Agent checkpoint"):
            changed._load_checkpoint("gpt-4o-mini", "deep_agent")

    # Only the digest is persisted, even when an endpoint embeds sensitive data.
    changed._clear_checkpoint("gpt-4o-mini", "deep_agent")
    changed._save_query_checkpoint("gpt-4o-mini", "deep_agent", "water", 0, query_result)
    checkpoint = Path(changed._checkpoint_path("gpt-4o-mini", "deep_agent")).read_text()
    assert "dummy-endpoint-secret" not in checkpoint
    assert "dummy-rotated-secret" not in checkpoint
