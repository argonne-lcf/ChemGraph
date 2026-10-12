"""Regression tests for CLI clarification handoff across process restarts."""

import json

import pytest
from langchain_core.messages import AIMessage

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
    first = json.loads(result.read_text())
    assert first["version"] == 1 and first["status"] == "waiting_for_user" and first["error"] is None
    pending = first["interrupts"][0]
    assert pending["kind"] == "clarification" and pending["payload"] == {"question": "Which calculator?"}
    session_id = first["session_id"]

    with monkeypatch.context() as inspection:
        inspection.setattr(_ScriptedChatModel, "_generate", lambda *a, **k: pytest.fail("inspection called model"))
        assert run(resume_session=session_id, result_file=result) == 3
        assert json.loads(result.read_text()) == first
        answers.write_text('{"stale-id": "EMT"}')
        assert run(resume_session=session_id, answers_file=answers, result_file=result) == 2
        assert json.loads(result.read_text())["interrupts"] == first["interrupts"]

    answers.write_text(json.dumps({pending["id"]: "EMT"}))
    responses[:] = [AIMessage(content="", tool_calls=[
        {"name": "ask_human", "id": "charge", "args": {"question": "Charge?"}, "type": "tool_call"},
        {"name": "ask_human", "id": "spin", "args": {"question": "Spin?"}, "type": "tool_call"},
    ])]
    assert run(resume_session=session_id, answers_file=answers, result_file=result) == 3
    second = json.loads(result.read_text())
    ids = {item["id"] for item in second["interrupts"]}
    assert len(ids) == 2 and pending["id"] not in ids
    with monkeypatch.context() as validation:
        validation.setattr(_ScriptedChatModel, "_generate", lambda *a, **k: pytest.fail("invalid answer called model"))
        for invalid in ({pending["id"]: "EMT"}, {next(iter(ids)): "0"}, {**dict.fromkeys(ids, "0"), "extra": "1"}):
            answers.write_text(json.dumps(invalid))
            assert run(resume_session=session_id, answers_file=answers, result_file=result) == 2
            rejected = json.loads(result.read_text())
            assert rejected["status"] == "waiting_for_user" and rejected["error"]
            assert rejected["interrupts"] == second["interrupts"]

    answers.write_text(json.dumps(dict.fromkeys(ids, "0")))
    responses[:] = [AIMessage(content="Done — H₂O.")]
    output = tmp_path / "answer.txt"
    assert run(resume_session=session_id, answers_file=answers, result_file=result,
               return_option="last_message", output_file=output) == 0
    assert json.loads(result.read_text()) == {
        "version": 1, "session_id": session_id, "status": "completed", "assistant_response": "Done — H₂O.",
        "interrupts": [], "error": None,
    }
    assert output.read_text() == "Done — H₂O."
    assert run(resume_session=session_id, answers_file=answers, result_file=result) == 2
    saved = SessionStore().get_session(session_id)
    assert saved.status == "completed" and saved.query_count == 1


@pytest.mark.parametrize("content", ['{', '[]', '{}', '{"id": " "}', '{"id": null}',
                                      '{"id": {"question": "EMT"}}', '{"id": "EMT", "id": "MACE"}'])
def test_invalid_answer_file_does_not_start_session(headless, tmp_path, content):
    run, _, requests = headless
    answers, result = tmp_path / "answers.json", tmp_path / "result.json"
    answers.write_text(content)
    assert run(resume_session="unknown", answers_file=answers, result_file=result) == 2
    receipt = json.loads(result.read_text())
    assert receipt["session_id"] is None and receipt["status"] is None and receipt["error"]
    assert not requests


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
def test_handoff_rejects_other_cli_modes(monkeypatch, changes):
    monkeypatch.setattr(cli_main, "run_headless_main_agent", lambda **k: pytest.fail("must reject invalid mode"))
    with pytest.raises(SystemExit) as exc:
        cli_main._handle_run(_run_args(result_file="unused", **changes))
    assert exc.value.code == 2
