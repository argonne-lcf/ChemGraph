"""The Deep Agent evaluation path formats its own trace before deterministic scoring."""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from chemgraph.agent.usage import summarize_usage
from chemgraph.eval import deepagent
from chemgraph.eval.cli import build_config_from_args, parse_args
from chemgraph.eval.runner import ModelBenchmarkRunner


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_output", [True, False], ids=["scored-answer", "formatting-failure"])
async def test_deepagent_evaluation(tmp_path, monkeypatch, valid_output):
    expected = {"smiles": ["O"]}
    dataset = tmp_path / "groundtruth.json"
    dataset.write_text(json.dumps([{
        "id": "water", "query": "Find the SMILES for water", "category": "lookup",
        "answer": {"structured_output": expected, "result": "GROUND_TRUTH_ONLY"},
    }]))
    config = build_config_from_args(parse_args([
        "--models", "fake", "--dataset", str(dataset), "--workflows", "deep_agent",
        "--judge-type", "structured", "--query-ids", "water",
        "--deepagent-workspace", str(tmp_path / "workspaces"), "--deepagent-auto-approve",
        "--output-dir", str(tmp_path / "results"),
    ]))
    state = {"messages": [{"type": "human", "content": "Find the SMILES for water"},
                          {"type": "ai", "content": "Water has SMILES O."}]}
    execution = AsyncMock(return_value=state)
    execution_usage = summarize_usage([{
        "counts": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, "complete": True,
    }])

    def make_agent(**options):
        assert options["workflow_type"] == "deep_agent"
        assert options["deepagent_auto_approve"] and not options["deepagent_discover_skills"]
        assert not options["enable_memory"] and not options["structured_output"]
        assert os.environ["CHEMGRAPH_LOG_DIR"] == options["log_dir"]
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

    monkeypatch.setattr(deepagent, "ChemGraph", make_agent)
    monkeypatch.setattr(deepagent, "load_chat_model", lambda **kwargs: SimpleNamespace(ainvoke=format_trace))
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", "original-logs")
    runner = ModelBenchmarkRunner(config)
    result = (await runner.run_all())["fake"]["deep_agent"]
    runner.report("all")

    aggregate = result["structured_judge_aggregate"]
    assert aggregate["n_queries"] == 1
    assert aggregate["n_correct"] == int(valid_output)
    raw = result["raw_tool_calls"][0]
    assert raw["state"] == state and raw["result"] == "Water has SMILES O."
    assert raw["status"] == ("completed" if valid_output else "formatting_error")
    assert len(invocations) == (1 if valid_output else 2)
    assert raw["usage"]["total"]["total_tokens"] == 12 + 25 * len(invocations)
    report = json.loads(next((tmp_path / "results").glob("benchmark_*.json")).read_text())
    assert report["results"]["fake"]["deep_agent"]["structured_judge_aggregate"] == aggregate
    assert os.environ["CHEMGRAPH_LOG_DIR"] == "original-logs"
