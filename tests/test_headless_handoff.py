"""Regression tests for CLI clarification handoff across process restarts."""

import io
import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

from chemgraph.agent.main_session import MainAgentSession, PendingInterrupt
from chemgraph.cli import commands, headless as driver
from chemgraph.memory.store import SessionStore
from chemgraph.registry.tools import ToolRegistry
from tests.test_approval_autonomy import artifacts as artifacts, headless as headless
from tests.test_main_agent import _ScriptedChatModel
from tests.test_main_agent_cli import cli_main, _run_args
from tests.test_main_agent_execution_context import model_loader as model_loader, call


def test_file_handoff_inspects_and_answers_current_questions(headless, tmp_path, monkeypatch):
    run, responses, _ = headless
    result, answers = tmp_path / "result.json", tmp_path / "answers.json"
    registry = ToolRegistry([ToolRegistry().get_spec("ask_human")])
    responses[:] = [call("load_tools", names=["ask_human"]), call("ask_human", question="Which calculator?")]
    assert run(query="calculate", human_supervised=True, tool_registry=registry, result_file=result) == 3
    first = json.loads(result.read_text(encoding="utf-8"))
    assert first["version"] == 1 and first["status"] == "waiting_for_user" and first["error"] is None
    assert first["exit_code"] == 3 and first["error_code"] is None
    pending = first["interrupts"][0]
    assert pending["kind"] == "clarification" and pending["payload"] == {"question": "Which calculator?"}
    session_id = first["session_id"]

    with monkeypatch.context() as inspection:
        inspection.setattr(_ScriptedChatModel, "_generate", lambda *a, **k: pytest.fail("inspection called model"))
        result = tmp_path / "inspection.json"
        assert run(resume_session=session_id, result_file=result) == 3
        assert json.loads(result.read_text(encoding="utf-8")) == first
        answers.write_text('{"stale-id": "EMT"}', encoding="utf-8")
        result = tmp_path / "stale.json"
        assert run(resume_session=session_id, answers_file=answers, result_file=result) == 2
        assert json.loads(result.read_text(encoding="utf-8"))["interrupts"] == first["interrupts"]

    answers.write_text(json.dumps({pending["id"]: "EMT"}), encoding="utf-8")
    responses[:] = [AIMessage(content="", tool_calls=[
        {"name": "ask_human", "id": "charge", "args": {"question": "Charge?"}, "type": "tool_call"},
        {"name": "ask_human", "id": "spin", "args": {"question": "Spin?"}, "type": "tool_call"},
    ])]
    result = tmp_path / "second.json"
    assert run(resume_session=session_id, answers_file=answers, result_file=result) == 3
    second = json.loads(result.read_text(encoding="utf-8"))
    ids = {item["id"] for item in second["interrupts"]}
    assert len(ids) == 2 and pending["id"] not in ids
    with monkeypatch.context() as validation:
        validation.setattr(_ScriptedChatModel, "_generate", lambda *a, **k: pytest.fail("invalid answer called model"))
        for index, invalid in enumerate(({pending["id"]: "EMT"}, {next(iter(ids)): "0"}, {**dict.fromkeys(ids, "0"), "extra": "1"})):
            answers.write_text(json.dumps(invalid), encoding="utf-8")
            result = tmp_path / f"invalid-{index}.json"
            assert run(resume_session=session_id, answers_file=answers, result_file=result) == 2
            rejected = json.loads(result.read_text(encoding="utf-8"))
            assert rejected["status"] == "waiting_for_user" and rejected["error"]
            assert rejected["exit_code"] == 2 and rejected["error_code"] == "stale_answers"
            assert rejected["interrupts"] == second["interrupts"]

    answers.write_text(json.dumps(dict.fromkeys(ids, "0")), encoding="utf-8")
    responses[:] = [AIMessage(content="Done — H₂O.")]
    output = tmp_path / "answer.txt"
    result = tmp_path / "completed.json"
    assert run(resume_session=session_id, answers_file=answers, result_file=result,
               return_option="last_message", output_file=output) == 0
    assert json.loads(result.read_text(encoding="utf-8")) == {
        "version": 1, "session_id": session_id, "status": "completed", "assistant_response": "Done — H₂O.",
        "interrupts": [], "error": None, "error_code": None, "exit_code": 0,
    }
    assert output.read_text(encoding="utf-8") == "Done — H₂O."
    result = tmp_path / "not-waiting.json"
    assert run(resume_session=session_id, answers_file=answers, result_file=result) == 2
    receipt = json.loads(result.read_text(encoding="utf-8"))
    assert receipt["error_code"] == "not_waiting" and receipt["exit_code"] == 2
    assert receipt["status"] == "completed"
    saved = SessionStore().get_session(session_id)
    assert saved.status == "completed" and saved.query_count == 1


@pytest.mark.parametrize("content", [None, '{', '[]', '{}', '{"id": " "}', '{"id": null}',
                                      '{"id": {"question": "EMT"}}', '{"id": "EMT", "id": "MACE"}'])
def test_invalid_answer_file_does_not_start_session(headless, tmp_path, content):
    run, _, requests = headless
    answers, result = tmp_path / "answers.json", tmp_path / "result.json"
    if content is not None:
        answers.write_text(content, encoding="utf-8")
    assert run(resume_session="unknown", answers_file=answers, result_file=result) == 2
    receipt = json.loads(result.read_text(encoding="utf-8"))
    assert receipt["session_id"] is None and receipt["status"] is None and receipt["error"]
    assert receipt["error_code"] == "invalid_answers" and receipt["exit_code"] == 2
    assert not requests


def test_surrogate_in_duplicate_answer_id_keeps_error_receipt(headless, tmp_path, monkeypatch):
    answers, result = tmp_path / "answers.json", tmp_path / "result.json"
    answers.write_text('{"\\ud800": "EMT", "\\ud800": "MACE"}', encoding="utf-8")
    with io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict") as stream:
        monkeypatch.setattr(driver, "console", Console(file=stream))
        assert headless[0](resume_session="unknown", answers_file=answers, result_file=result) == 2
    receipt = json.loads(result.read_text(encoding="utf-8"))
    assert receipt["exit_code"] == 2 and receipt["error_code"] == "invalid_answers"
    assert "Duplicate answer ID: ?" in receipt["error"]


def test_cli_parses_and_forwards_handoff_paths(monkeypatch):
    received = {}
    monkeypatch.setattr(cli_main, "run_headless_main_agent", lambda **k: received.update(k))
    args = cli_main.create_argument_parser().parse_args([
        "run", "-w", "main_agent", "--dangerously-skip-approvals", "--resume", "saved",
        "--answers-file", "answers.json", "--result-file", "result.json",
    ])
    cli_main._handle_run(args)
    assert received["answers_file"] == "answers.json" and received["result_file"] == "result.json"


@pytest.mark.parametrize("changes", [{"workflow": "single_agent"}, {"interactive": True}])
def test_handoff_rejects_other_cli_modes(monkeypatch, tmp_path, changes):
    monkeypatch.setattr(cli_main, "run_headless_main_agent", lambda **k: pytest.fail("must reject invalid mode"))
    result = tmp_path / "result.json"
    with pytest.raises(SystemExit) as exc:
        cli_main._handle_run(_run_args(result_file=result, **changes))
    assert exc.value.code == 2
    receipt = json.loads(result.read_text(encoding="utf-8"))
    assert receipt["error_code"] == "config" and receipt["exit_code"] == 2


def test_running_receipt_precedes_new_turn_answers_and_followup(headless, tmp_path, monkeypatch):
    run, responses, _ = headless
    result, answers = tmp_path / "result.json", tmp_path / "answers.json"
    observed = []
    generate = _ScriptedChatModel._generate

    def inspect_before_model(*args, **kwargs):
        receipt = json.loads(result.read_text(encoding="utf-8"))
        assert receipt["status"] == "running" and receipt["session_id"]
        assert receipt["exit_code"] is None and receipt["error_code"] is None and receipt["error"] is None
        assert receipt["assistant_response"] == "" and receipt["interrupts"] == []
        observed.append(receipt["session_id"])
        return generate(*args, **kwargs)

    monkeypatch.setattr(_ScriptedChatModel, "_generate", inspect_before_model)
    responses[:] = [call("load_tools", names=["ask_human"]), call("ask_human", question="Method?")]
    assert run(query="calculate", result_file=result, human_supervised=True,
               tool_registry=ToolRegistry([ToolRegistry().get_spec("ask_human")])) == 3
    waiting = json.loads(result.read_text(encoding="utf-8"))
    answers.write_text(json.dumps({waiting["interrupts"][0]["id"]: "EMT"}), encoding="utf-8")
    responses[:] = [AIMessage(content="done")]
    result = tmp_path / "answered.json"
    assert run(resume_session=waiting["session_id"], answers_file=answers, result_file=result) == 0
    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "completed"
    result = tmp_path / "followup.json"
    assert run(resume_session=waiting["session_id"], query="follow up", result_file=result) == 0
    assert observed == [waiting["session_id"]] * 4


@pytest.mark.parametrize("startup_error", [False, True])
def test_existing_result_database_keeps_checkpoints(headless, tmp_path, monkeypatch, startup_error):
    run, responses, _ = headless
    responses[:] = [AIMessage(content="saved")]
    assert run(query="calculate") == 0
    result = tmp_path / "checkpoints.db"
    with sqlite3.connect(result) as db:
        checkpoints = db.execute("SELECT thread_id, checkpoint_id, checkpoint FROM checkpoints ORDER BY checkpoint_id").fetchall()
    assert checkpoints
    monkeypatch.setattr(_ScriptedChatModel, "_generate", lambda *a, **k: pytest.fail("must not execute"))
    if startup_error:
        with pytest.raises(SystemExit) as exc:
            cli_main._handle_run(_run_args(query="calculate", result_file=result))
        assert exc.value.code == 1
    else:
        assert run(query="calculate again", result_file=result) == 1
    with sqlite3.connect(result) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("SELECT thread_id, checkpoint_id, checkpoint FROM checkpoints ORDER BY checkpoint_id").fetchall() == checkpoints


@pytest.mark.parametrize("failure", ["missing_parent", "directory", "permission"])
def test_unusable_receipt_prevents_execution(headless, tmp_path, monkeypatch, failure):
    result = tmp_path / "result.json"
    if failure == "missing_parent":
        result = tmp_path / "missing" / "result.json"
    elif failure == "directory":
        result.mkdir()
    else:
        def denied(*args, **kwargs):
            raise PermissionError("receipt is read-only")
        monkeypatch.setattr(driver.tempfile, "NamedTemporaryFile", denied)
    monkeypatch.setattr(_ScriptedChatModel, "_generate", lambda *a, **k: pytest.fail("must not execute"))
    assert headless[0](query="calculate", result_file=result) == 1
    assert not SessionStore().list_sessions()


@pytest.mark.parametrize("failure_at", [1, 2])
def test_atomic_receipt_failure_preserves_last_write(headless, tmp_path, monkeypatch, failure_at):
    run, responses, _ = headless
    responses[:] = [AIMessage(content="done")]
    result = tmp_path / "result.json"
    replace_file, writes = driver.os.replace, []

    def fail_replace(source, destination):
        writes.append(destination)
        if len(writes) == failure_at:
            raise OSError("replace failed")
        return replace_file(source, destination)

    monkeypatch.setattr(driver.os, "replace", fail_replace)
    assert run(query="calculate", result_file=result) == 1
    assert not list(tmp_path.glob(".result.json-*"))
    if failure_at == 1:
        assert not SessionStore().list_sessions()
    else:
        receipt = json.loads(result.read_text(encoding="utf-8"))
        assert receipt["status"] == "running" and receipt["exit_code"] is None
        assert SessionStore().get_session(receipt["session_id"]).status == "completed"


@pytest.mark.parametrize("payload_kind", ["datetime", "bytes", "nan", "keys", "cycle", "unicode"])
def test_unencodable_interrupt_preserves_waiting_outcome(headless, tmp_path, monkeypatch, payload_kind):
    run, responses, _ = headless
    cycle = []
    cycle.append(cycle)
    payload = {"datetime": datetime(2026, 1, 1), "bytes": b"payload", "nan": float("nan"),
               "keys": {(1, 2): "value"}, "cycle": cycle, "unicode": "\ud800"}[payload_kind]
    original = commands.run_main_agent_query

    def with_payload(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, interrupts=(PendingInterrupt("unsupported-id", payload),))

    monkeypatch.setattr(commands, "run_main_agent_query", with_payload)
    responses[:] = [call("load_tools", names=["ask_human"]), call("ask_human", question="Method?")]
    result = tmp_path / "result.json"
    assert run(query="calculate", result_file=result, human_supervised=True,
               tool_registry=ToolRegistry([ToolRegistry().get_spec("ask_human")])) == 3
    receipt = json.loads(result.read_text(encoding="utf-8"))
    assert receipt["status"] == "waiting_for_user" and receipt["exit_code"] == 3
    assert receipt["error"] is None and receipt["error_code"] is None
    pending = receipt["interrupts"][0]
    assert pending["id"] == "unsupported-id" and pending["kind"] == "unsupported"
    assert "inspect interactively" in pending["payload"]["message"]


@pytest.mark.parametrize("kind", ["tool_review", "stale_tool_review", "unsupported", "missing_id"])
def test_answers_reject_nonclarifications_without_advancing(headless, tmp_path, monkeypatch, kind):
    run, responses, _ = headless
    result, answers = tmp_path / "result.json", tmp_path / "answers.json"
    responses[:] = [call("load_tools", names=["ask_human"]), call("ask_human", question="Method?")]
    assert run(query="calculate", result_file=result, human_supervised=True,
               tool_registry=ToolRegistry([ToolRegistry().get_spec("ask_human")])) == 3
    waiting = json.loads(result.read_text(encoding="utf-8"))
    pending_id = waiting["interrupts"][0]["id"]
    answers.write_text(json.dumps({"stale" if kind == "stale_tool_review" else pending_id: "approve"}), encoding="utf-8")
    payload_kind = "tool_review" if kind == "stale_tool_review" else kind
    original = commands.restore_main_agent_session

    def restore(*args, **kwargs):
        restored = original(*args, **kwargs)
        payload = {"tool_review": {"action_requests": [], "review_configs": []},
                   "unsupported": {"message": "review"}, "missing_id": {"question": "Method?"}}[payload_kind]
        return replace(restored, interrupts=(PendingInterrupt("" if kind == "missing_id" else pending_id, payload),))

    monkeypatch.setattr(commands, "restore_main_agent_session", restore)
    monkeypatch.setattr(MainAgentSession, "resume", lambda *a, **k: pytest.fail("must not resume"))
    if kind == "missing_id":
        with driver.console.capture() as captured:
            assert run(resume_session=waiting["session_id"], result_file=tmp_path / "anonymous.json") == 3
        assert "Or submit answers" not in captured.get()
    result = tmp_path / "rejected.json"
    assert run(resume_session=waiting["session_id"], answers_file=answers, result_file=result) == 2
    receipt = json.loads(result.read_text(encoding="utf-8"))
    assert receipt["error_code"] == "unsupported_interrupt" and receipt["status"] == "waiting_for_user"
    assert receipt["interrupts"][0]["kind"] == ("clarification" if kind == "missing_id" else payload_kind)


@pytest.mark.parametrize("outcome", ["execution", "config", "export", "export_serialization", "interrupted", "initialization"])
def test_receipt_classifies_failures(headless, tmp_path, monkeypatch, outcome):
    run, responses, _ = headless
    responses[:] = [RuntimeError("unavailable")] if outcome == "execution" else [AIMessage(content="done")]
    options = {"query": "calculate", "result_file": tmp_path / "result.json"}
    if outcome == "config":
        options["query"] = " "
    elif outcome == "export":
        options["output_file"] = tmp_path / "missing" / "output.txt"
    elif outcome == "export_serialization":
        options["output_file"] = tmp_path / "output.json"
        original = commands.run_main_agent_query
        monkeypatch.setattr(commands, "run_main_agent_query",
                            lambda *a, **k: replace(original(*a, **k), state={"nan": float("nan")}))
    elif outcome == "interrupted":
        def cancel(*args, **kwargs):
            raise KeyboardInterrupt
        monkeypatch.setattr(commands, "run_main_agent_query", cancel)
    elif outcome == "initialization":
        monkeypatch.setattr(commands, "initialize_agent", lambda **k: None)
    code = {"config": 2, "interrupted": 130}.get(outcome, 1)
    assert run(**options) == code
    receipt = json.loads(options["result_file"].read_text(encoding="utf-8"))
    assert receipt["exit_code"] == code and receipt["error"]
    assert receipt["error_code"] == {"initialization": "execution", "export_serialization": "export"}.get(outcome, outcome)
    if outcome.startswith("export"):
        assert receipt["status"] == "completed" and receipt["assistant_response"] == "done"


@pytest.mark.parametrize("waiting", [False, True])
def test_failed_continuation_does_not_report_previous_result(headless, tmp_path, waiting):
    run, responses, _ = headless
    result, answers = tmp_path / "result.json", tmp_path / "answers.json"
    responses[:] = ([call("load_tools", names=["ask_human"]), call("ask_human", question="Method?")]
                    if waiting else [AIMessage(content="previous answer")])
    assert run(query="calculate", result_file=result, human_supervised=True,
               tool_registry=ToolRegistry([ToolRegistry().get_spec("ask_human")])) == (3 if waiting else 0)
    previous = json.loads(result.read_text(encoding="utf-8"))
    options = {"query": "follow up"}
    if waiting:
        answers.write_text(json.dumps({previous["interrupts"][0]["id"]: "EMT"}), encoding="utf-8")
        options = {"answers_file": answers}
    responses[:] = [RuntimeError("unavailable")]
    result = tmp_path / "failed.json"
    assert run(resume_session=previous["session_id"], result_file=result, **options) == 1
    receipt = json.loads(result.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed" and receipt["error_code"] == "execution"
    assert receipt["assistant_response"] == "" and receipt["interrupts"] == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM semantics")
def test_sigterm_leaves_running_receipt(tmp_path):
    program = textwrap.dedent('''
        import sys
        import threading
        from pathlib import Path
        import pytest
        from tests.test_approval_autonomy import headless
        from tests.test_main_agent_execution_context import model_loader
        from tests.test_main_agent import _ScriptedChatModel
        root = Path(sys.argv[1])
        patch = pytest.MonkeyPatch()
        patch.setenv("CHEMGRAPH_LOG_DIR", str(root / "artifacts"))
        model = model_loader.__wrapped__(patch, root)
        run, _, _ = headless.__wrapped__(patch, root, model)
        def block(*args, **kwargs):
            (root / "executing").touch()
            threading.Event().wait(60)
        patch.setattr(_ScriptedChatModel, "_generate", block)
        run(query="calculate", result_file=root / "result.json")
    ''')
    # A real model stub runs inside the real checkpoint runtime in a separate process.
    with (tmp_path / "process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-c", program, str(tmp_path)],
                                   cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "executing").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert (tmp_path / "executing").exists(), (tmp_path / "process.log").read_text(encoding="utf-8")
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) == -signal.SIGTERM
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    receipt = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "running" and receipt["session_id"]
    assert receipt["exit_code"] is None and receipt["error_code"] is None
    assert SessionStore(db_path=str(tmp_path / "sessions.db")).get_session(receipt["session_id"]) is not None
