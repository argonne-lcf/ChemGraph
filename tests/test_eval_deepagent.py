"""The Deep Agent evaluation path formats its own trace before deterministic scoring."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from chemgraph.agent.usage import summarize_usage
from chemgraph.eval import deepagent
from chemgraph.eval.cli import build_config_from_args, parse_args, run_eval
from chemgraph.eval.config import BenchmarkConfig
from chemgraph.eval.runner import ModelBenchmarkRunner
from chemgraph.graphs.deep_agent import _normalize_backend
from chemgraph.models.endpoints import PreparedModel
from chemgraph.skills.runtime import prepare_skill_backend
from chemgraph.tools.ase_tools import file_to_atomsdata, save_atomsdata_to_file
from tests.test_main_agent import _ScriptedChatModel
from tests.test_registry_middleware import call


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_output", [True, False], ids=["scored-answer", "formatting-failure"])
@pytest.mark.parametrize("judge_type", ["structured", "llm", "both"])
async def test_deepagent_evaluation(tmp_path, monkeypatch, valid_output, judge_type):
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
        "--judge-type", judge_type, "--judge-model", "fake-judge", "--query-ids", "water",
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
        return SimpleNamespace(run=execution, last_usage=execution_usage, reasoning_effort=None)

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
    judge = AsyncMock(return_value={"score": 1, "rationale": "Correct answer", "parse_error": None})
    monkeypatch.setattr("chemgraph.eval.runner.load_judge_model", lambda *args, **kwargs: object())
    monkeypatch.setattr("chemgraph.eval.runner.judge_single_query", judge)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", "original-logs")
    runner = ModelBenchmarkRunner(config)
    result = (await runner.run_all())["fake"]["deep_agent"]
    runner.report("all")

    raw = result["raw_tool_calls"][0]
    assert raw["status"] == ("completed" if valid_output else "formatting_error"), raw.get("error")
    if judge_type in ("structured", "both"):
        aggregate = result["structured_judge_aggregate"]
        assert aggregate["n_queries"] == 1
        assert aggregate["n_correct"] == int(valid_output)
    if judge_type in ("llm", "both"):
        assert result["judge_aggregate"]["n_queries"] == 1
        assert result["judge_aggregate"]["n_correct"] == 1
        judge.assert_awaited_once()
        assert judge.await_args.kwargs["model_result"] == "Water has SMILES O."
        assert judge.await_args.kwargs["model_tool_calls"] == raw["tool_calls"]
    else:
        judge.assert_not_awaited()
    assert result["execution_summary"]["n_errors"] == int(not valid_output)
    if not valid_output:
        assert raw["error"] == raw["structured_output"]["_parse_error"]
    assert raw["state"] == state and raw["result"] == "Water has SMILES O."
    assert len(invocations) == (1 if valid_output else 2)
    assert raw["usage"]["total"]["total_tokens"] == 12 + 25 * len(invocations)
    report = json.loads(next((tmp_path / "results").glob("benchmark_*.json")).read_text())
    assert report["results"]["fake"]["deep_agent"] == result
    assert os.environ["CHEMGRAPH_LOG_DIR"] == "original-logs"

    config.resume = True
    resumed = (await ModelBenchmarkRunner(config).run_all())["fake"]["deep_agent"]
    assert resumed == result
    execution.assert_awaited_once()
    assert len(invocations) == (1 if valid_output else 2)
    assert judge.await_count == int(judge_type in ("llm", "both"))


@pytest.fixture
def evaluation_dataset(tmp_path):
    dataset = tmp_path / "groundtruth.json"
    dataset.write_text(json.dumps([{
        "id": "water", "query": "Find the SMILES for water",
        "answer": {"structured_output": {"smiles": ["O"]}, "result": "Water has SMILES O."},
    }]))
    return dataset


@pytest.mark.asyncio
@pytest.mark.parametrize("judge_type", ["llm", "both"])
@pytest.mark.parametrize("failure", [
    "initialization", "execution", "formatter_initialization", "formatter_invocation",
])
async def test_deepagent_failure_scoring(tmp_path, monkeypatch, evaluation_dataset, judge_type, failure):
    config = BenchmarkConfig(
        models=["fake"], workflow_types=["deep_agent"], dataset=str(evaluation_dataset),
        judge_type=judge_type, judge_model="fake-judge", deepagent_auto_approve=True,
        output_dir=str(tmp_path / "results"),
    )
    state = {"messages": [{"type": "ai", "content": "Water has SMILES O."}]}
    execution_usage = summarize_usage([{
        "counts": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, "complete": True,
    }])
    agent = SimpleNamespace(
        run=AsyncMock(return_value=state), aget_state=AsyncMock(return_value=state),
        last_usage=execution_usage, reasoning_effort=None,
    )
    make_agent = Mock(return_value=agent)
    formatter = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError(failure)))
    make_formatter = Mock(return_value=formatter)
    if failure == "initialization":
        make_agent.side_effect = RuntimeError(failure)
    elif failure == "execution":
        agent.run.side_effect = RuntimeError(failure)
    elif failure == "formatter_initialization":
        make_formatter.side_effect = RuntimeError(failure)
    judge = AsyncMock(return_value={"score": 1, "rationale": "Correct answer", "parse_error": None})
    monkeypatch.setattr(deepagent, "ChemGraph", make_agent)
    monkeypatch.setattr(deepagent, "load_chat_model", make_formatter)
    monkeypatch.setattr("chemgraph.eval.runner.load_judge_model", lambda *args, **kwargs: object())
    monkeypatch.setattr("chemgraph.eval.runner.judge_single_query", judge)
    previous_log_dir = "original-logs" if judge_type == "both" else None
    if previous_log_dir is None:
        monkeypatch.delenv("CHEMGRAPH_LOG_DIR", raising=False)
    else:
        monkeypatch.setenv("CHEMGRAPH_LOG_DIR", previous_log_dir)

    result = (await ModelBenchmarkRunner(config).run_all())["fake"]["deep_agent"]
    raw = result["raw_tool_calls"][0]
    executed = failure.startswith("formatter_")
    assert raw["status"] == ("formatting_error" if executed else f"{failure}_error")
    assert raw["error"] == f"RuntimeError: {failure}"
    assert raw["structured_output"]["_parse_error"] == raw["error"]
    assert result["judge_aggregate"]["n_queries"] == 1
    assert result["judge_aggregate"]["n_correct"] == int(executed)
    if executed:
        judge.assert_awaited_once()
        assert judge.await_args.kwargs["model_result"] == "Water has SMILES O."
    else:
        judge.assert_not_awaited()
    if judge_type == "both":
        assert result["structured_judge_aggregate"]["n_queries"] == 1
        assert result["structured_judge_aggregate"]["n_correct"] == 0
    if failure != "initialization":
        assert raw["state"] == state and raw["result"] == "Water has SMILES O."
        assert raw["usage"]["execution"] == execution_usage
    assert raw["usage"]["total"]["total_tokens"] == (0 if failure == "initialization" else 12)
    assert result["execution_summary"]["n_errors"] == 1
    assert formatter.ainvoke.await_count == int(failure == "formatter_invocation")
    assert os.environ.get("CHEMGRAPH_LOG_DIR") == previous_log_dir


@pytest.mark.parametrize("workflows", [["deep_agent"], ["single_agent", "deep_agent"]])
@pytest.mark.parametrize("approve", [False, True])
def test_deepagent_config_requires_approval(evaluation_dataset, workflows, approve):
    options = dict(
        models=["fake"], workflow_types=workflows, dataset=str(evaluation_dataset),
        judge_type="structured", deepagent_auto_approve=approve,
    )
    if approve:
        assert BenchmarkConfig(**options).deepagent_auto_approve is True
    else:
        with pytest.raises(ValueError, match="requires --deepagent-auto-approve"):
            BenchmarkConfig(**options)


@pytest.mark.parametrize("source", ["cli", "profile"])
@pytest.mark.parametrize("approve", [False, True])
def test_deepagent_cli_approval_before_runner(tmp_path, monkeypatch, evaluation_dataset, source, approve):
    argv = ["--models", "fake", "--dataset", str(evaluation_dataset), "--judge-type", "structured"]
    if source == "profile":
        profile = tmp_path / "config.toml"
        profile.write_text(
            '[eval]\ndefault_profile = "test"\n[eval.profiles.test]\n'
            'workflow_types = ["single_agent", "deep_agent"]\n'
            f'deepagent_auto_approve = {str(approve).lower()}\n'
        )
        argv += ["--config", str(profile)]
    else:
        argv += ["--workflows", "single_agent", "deep_agent"]
        if approve:
            argv.append("--deepagent-auto-approve")
    args = parse_args(argv)
    runner = Mock()
    monkeypatch.setattr("chemgraph.eval.cli.ModelBenchmarkRunner", runner)
    if approve:
        config = build_config_from_args(args)
        assert config.deepagent_auto_approve is True
        assert config.workflow_types == ["single_agent", "deep_agent"]
    else:
        with pytest.raises(ValueError, match="requires --deepagent-auto-approve") as exc:
            run_eval(args)
        assert "deepagent_auto_approve = true" in str(exc.value)
        assert "no interactive approval handler" in str(exc.value)
    runner.assert_not_called()


@pytest.mark.parametrize("setting", ["max_queries", "query_ids", "code_sha256"])
def test_deepagent_checkpoint_requires_same_selection_and_code(tmp_path, evaluation_dataset, setting):
    options = dict(
        models=["fake"], workflow_types=["deep_agent"], dataset=str(evaluation_dataset),
        judge_type="structured", deepagent_auto_approve=True, max_queries=1,
        output_dir=str(tmp_path / "results"), resume=True,
    )
    runner = ModelBenchmarkRunner(BenchmarkConfig(**options))
    query_result = {"raw": {"result": "O"}, "judge": None, "structured_judge": None}
    runner._save_query_checkpoint("fake", "deep_agent", "water", 0, query_result)
    unchanged = ModelBenchmarkRunner(BenchmarkConfig(**options))
    assert unchanged._load_checkpoint("fake", "deep_agent") == {"water": query_result}

    if setting == "max_queries":
        options["max_queries"] = 2
    elif setting == "query_ids":
        options["query_ids"] = ["water"]
    # Selection options must match even if the effective query list is identical.
    with pytest.raises(ValueError, match="Incompatible Deep Agent checkpoint"):
        changed = ModelBenchmarkRunner(BenchmarkConfig(**options))
        if setting == "code_sha256":
            changed._provenance["code_sha256"] = "changed-code-checksum"
        changed._load_checkpoint("fake", "deep_agent")


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
            deepagent_auto_approve=True,
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

    if setting == "credentials":
        changed = make_runner()
        assert changed._load_checkpoint("gpt-4o-mini", "deep_agent") == {"water": query_result}
    else:
        with pytest.raises(ValueError, match="Incompatible Deep Agent checkpoint"):
            make_runner()  # Incompatible resume is rejected before loading the judge.

    # Only the digest is persisted, even when an endpoint embeds sensitive data.
    runner._clear_checkpoint("gpt-4o-mini", "deep_agent")
    changed = make_runner()
    changed._save_query_checkpoint("gpt-4o-mini", "deep_agent", "water", 0, query_result)
    checkpoint = Path(changed._checkpoint_path("gpt-4o-mini", "deep_agent")).read_text()
    assert "dummy-endpoint-secret" not in checkpoint
    assert "dummy-rotated-secret" not in checkpoint


@pytest.fixture
def evaluation_config(tmp_path, evaluation_dataset):
    return BenchmarkConfig(
        models=["fake"], workflow_types=["deep_agent"], dataset=str(evaluation_dataset),
        judge_type="both", judge_model="fake-judge", deepagent_auto_approve=True,
        output_dir=str(tmp_path / "results"),
    )


@pytest.fixture
def eval_models(monkeypatch):
    state = {"messages": [{"type": "ai", "content": "Water has SMILES O."}]}
    agent = SimpleNamespace(
        run=AsyncMock(return_value=state), last_usage=summarize_usage([]),
        reasoning_effort=None,
    )
    make_agent = Mock(return_value=agent)
    formatter = FakeMessagesListChatModel(responses=[AIMessage(content='{"smiles": ["O"]}')])
    make_formatter = Mock(return_value=formatter)
    make_judge = Mock(return_value=object())
    judge = AsyncMock(return_value={"score": 1, "rationale": "Correct", "parse_error": None})
    monkeypatch.setattr(deepagent, "ChemGraph", make_agent)
    monkeypatch.setattr(deepagent, "load_chat_model", make_formatter)
    monkeypatch.setattr("chemgraph.eval.runner.load_judge_model", make_judge)
    monkeypatch.setattr("chemgraph.eval.runner.judge_single_query", judge)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", "original-logs")
    return SimpleNamespace(
        agent=agent, make_agent=make_agent, make_formatter=make_formatter,
        make_judge=make_judge, judge=judge,
    )


def add_second_query(config):
    dataset = Path(config.dataset)
    items = json.loads(dataset.read_text())
    items.append({**items[0], "id": "second"})
    dataset.write_text(json.dumps(items))


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["custom_root", "model_directory", "thread_directory"])
async def test_deepagent_workspace_setup_failure(evaluation_config, eval_models, location):
    config = evaluation_config
    workspace = Path(config.deepagent_workspace) / "fake/deep_agent/thread_water"
    blocked = {
        "custom_root": Path(config.output_dir).parent / "custom-workspace",
        "model_directory": workspace.parent.parent,
        "thread_directory": workspace,
    }[location]
    if location == "custom_root":
        config.deepagent_workspace = str(blocked)
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("preserve this file")

    runner = ModelBenchmarkRunner(config)
    result = (await runner.run_all())["fake"]["deep_agent"]
    raw = result["raw_tool_calls"][0]
    assert raw["status"] == "initialization_error"
    assert raw["query_id"] == "water" and raw["query"]
    assert raw["error"] == raw["structured_output"]["_parse_error"]
    assert raw["elapsed_seconds"] >= 0
    assert raw["usage"]["total"]["call_count"] == 0
    assert result["judge_aggregate"]["n_correct"] == 0
    assert result["structured_judge_aggregate"]["n_correct"] == 0
    assert result["execution_summary"]["n_errors"] == 1
    eval_models.make_agent.assert_not_called()
    eval_models.make_formatter.assert_not_called()
    eval_models.judge.assert_not_awaited()
    assert blocked.read_text() == "preserve this file"
    assert os.environ["CHEMGRAPH_LOG_DIR"] == "original-logs"


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True], ids=["fresh-rerun", "interrupted-resume"])
async def test_deepagent_stale_preflight_preserves_results(evaluation_config, eval_models, resume):
    config = evaluation_config
    add_second_query(config)
    runner = ModelBenchmarkRunner(config)
    await runner.run_all()
    runner.report("all")
    if resume:
        # Simulate interruption after the second query wrote artifacts, before checkpointing it.
        checkpoint = Path(runner._checkpoint_path("fake", "deep_agent"))
        first_record = checkpoint.read_text().splitlines()[0]
        checkpoint.write_text(first_record + "\n")
    stale = Path(config.deepagent_workspace) / "fake/deep_agent/thread_second/previous.txt"
    stale.write_text("previous result")
    output = Path(config.output_dir)
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    for mock in (eval_models.make_agent, eval_models.make_formatter, eval_models.make_judge, eval_models.judge):
        mock.reset_mock()
    config.resume = resume

    with pytest.raises(ValueError, match="[Ww]orkspace.*not empty") as exc:
        ModelBenchmarkRunner(config)
    assert str(stale.parent) in str(exc.value)
    assert "--output-dir" in str(exc.value) and "--deepagent-workspace" in str(exc.value)
    after = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert after == before
    eval_models.make_agent.assert_not_called()
    eval_models.make_formatter.assert_not_called()
    eval_models.make_judge.assert_not_called()
    eval_models.judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_deepagent_late_artifacts_fail_query_and_continue(evaluation_config, eval_models):
    config = evaluation_config
    add_second_query(config)
    runner = ModelBenchmarkRunner(config)
    stale = Path(config.deepagent_workspace) / "fake/deep_agent/thread_water/late.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("appeared after preflight")

    result = (await runner.run_all())["fake"]["deep_agent"]
    assert [raw["status"] for raw in result["raw_tool_calls"]] == ["initialization_error", "completed"]
    assert result["judge_aggregate"]["n_correct"] == 1
    assert result["structured_judge_aggregate"]["n_correct"] == 1
    assert result["execution_summary"]["n_errors"] == 1
    assert stale.read_text() == "appeared after preflight"
    eval_models.make_agent.assert_called_once()
    eval_models.judge.assert_awaited_once()
    assert os.environ["CHEMGRAPH_LOG_DIR"] == "original-logs"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["adapter", "usage_finalization"])
async def test_deepagent_escaped_failure_has_unknown_usage(
    evaluation_config, eval_models, monkeypatch, failure,
):
    if failure == "adapter":
        monkeypatch.setattr(deepagent, "run_query", AsyncMock(side_effect=RuntimeError(failure)))
    else:
        monkeypatch.setattr(deepagent, "combine_usage", Mock(side_effect=RuntimeError(failure)))
    runner = ModelBenchmarkRunner(evaluation_config)
    result = (await runner.run_all())["fake"]["deep_agent"]
    runner.report("all")
    raw = result["raw_tool_calls"][0]
    assert raw["status"] == "execution_error"
    assert raw["query_id"] == "water" and raw["query"]
    assert raw["error"] == f"RuntimeError: {failure}"
    assert raw["elapsed_seconds"] >= 0
    assert result["judge_aggregate"]["n_correct"] == 0
    assert result["structured_judge_aggregate"]["n_correct"] == 0
    assert result["execution_summary"]["n_errors"] == 1
    usage = result["execution_summary"]["usage"]
    assert usage["total_tokens"] is None and usage["partial"] is True
    assert usage["call_count"] == 0
    eval_models.judge.assert_not_awaited()
    assert os.environ["CHEMGRAPH_LOG_DIR"] == "original-logs"


@pytest.mark.asyncio
async def test_deepagent_retry_exception_preserves_reply_and_usage(evaluation_config, eval_models):
    formatter = _ScriptedChatModel(responses=[
        AIMessage(content="invalid first reply", usage_metadata={
            "input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
        }),
        RuntimeError("retry failed"),
    ])
    eval_models.make_formatter.return_value = formatter
    runner = ModelBenchmarkRunner(evaluation_config)
    result = (await runner.run_all())["fake"]["deep_agent"]
    runner.report("all")
    raw = result["raw_tool_calls"][0]
    assert raw["status"] == "formatting_error"
    assert raw["formatter_responses"] == ["invalid first reply"]
    assert raw["error"] == "RuntimeError: retry failed"
    assert raw["result"] == "Water has SMILES O."
    assert raw["usage"]["formatting"]["call_count"] == 2
    assert raw["usage"]["formatting"]["total_tokens"] == 25
    assert raw["usage"]["formatting"]["partial"] is True
    assert result["structured_judge_aggregate"]["n_correct"] == 0
    assert result["judge_aggregate"]["n_correct"] == 1
    checkpoint = json.loads(Path(runner._checkpoint_path("fake", "deep_agent")).read_text())
    assert checkpoint["raw"]["formatter_responses"] == raw["formatter_responses"]
    assert os.environ["CHEMGRAPH_LOG_DIR"] == "original-logs"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name,effort", [("fake", None), ("argo:gpt-5.6-sol", "none")])
async def test_deepagent_real_graph_formats_native_trace(
    evaluation_config, monkeypatch, model_name, effort,
):
    config = evaluation_config
    config.models = [model_name]
    config.judge_type = "structured"
    workspace = Path(config.deepagent_workspace) / model_name.replace(":", "_") / "deep_agent/thread_water"
    workspace.mkdir(parents=True)  # Existing empty directories are valid.
    structure = workspace / "water.xyz"
    model = _ScriptedChatModel(responses=[
        call("write_file", file_path=str(structure), content="3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n"),
        call("load_tools", names=["file_to_atomsdata"]),
        call("file_to_atomsdata", fname=str(structure)),
        AIMessage(content="Water has SMILES O."),
    ])
    for response in model.responses:
        response.usage_metadata = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    formatter = _ScriptedChatModel(responses=[AIMessage(
        content='{"smiles": ["O"]}',
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )])
    loader = Mock(return_value=(model, PreparedModel(
        endpoint_name="test", protocol="openai_compatible", client_kwargs={},
    )))
    formatter_loader = Mock(return_value=formatter)
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", loader)
    monkeypatch.setattr(deepagent, "load_chat_model", formatter_loader)
    monkeypatch.delenv("CHEMGRAPH_LOG_DIR", raising=False)
    runner = ModelBenchmarkRunner(config)
    result = (await runner.run_all())[model_name]["deep_agent"]
    raw = result["raw_tool_calls"][0]
    assert raw["status"] == "completed", raw.get("error")
    assert raw["result"] == "Water has SMILES O."
    assert raw["tool_calls"][-1] == {"file_to_atomsdata": {"fname": str(structure)}}
    native_reply = next(m for m in raw["state"]["messages"] if m.get("name") == "file_to_atomsdata")
    assert "numbers=[8, 1, 1]" in native_reply["content"]
    assert structure.is_file()
    assert result["structured_judge_aggregate"]["accuracy"] == 1.0
    assert raw["usage"]["total"]["total_tokens"] == 25
    assert loader.call_args.kwargs["reasoning_effort"] == effort
    assert formatter_loader.call_args.kwargs.get("reasoning_effort") == effort
    assert "CHEMGRAPH_LOG_DIR" not in os.environ
    config.resume = True
    assert (await ModelBenchmarkRunner(config).run_all())[model_name]["deep_agent"] == result
    loader.assert_called_once()
    formatter_loader.assert_called_once()
