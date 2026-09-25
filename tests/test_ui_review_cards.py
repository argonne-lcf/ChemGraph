"""Tests for Deep Agent action-review cards and the chat approval flow."""

import queue
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ui import review_cards
from ui._pages import main_interface as main_ui


def _review(actions, allowed=("approve", "reject")):
    names = {a["name"] for a in actions}
    return {
        "action_requests": list(actions),
        "review_configs": [
            {"action_name": name, "allowed_decisions": list(allowed)} for name in sorted(names)
        ],
    }


# ---------------------------------------------------------------------------
# action_preview / review_summary
# ---------------------------------------------------------------------------


def test_execute_preview_shows_command_as_bash():
    info = review_cards.action_preview({"name": "execute", "args": {"command": "ls -la\x1b[2K"}})
    assert info["summary"] == "Tool: execute"
    # Terminal controls are made visible, exactly as in the CLI panel.
    assert info["blocks"] == [("Command", "ls -la\\x1b[2K", "bash")]
    assert info["truncated"] is False


def test_write_file_preview_reports_path_and_truncates():
    content = "before\n" * 50 + "middle\n" + "after\n" * 50
    info = review_cards.action_preview(
        {"name": "write_file", "args": {"file_path": "/tmp/x.txt", "content": content}}
    )
    assert info["summary"] == "Tool: write_file | Path: /tmp/x.txt"
    labels = [block[0] for block in info["blocks"]]
    assert labels == ["Arguments", "Content"]
    assert info["truncated"] is True
    assert "characters omitted" in info["blocks"][1][1]
    assert '"file_path": "/tmp/x.txt"' in info["arguments"]


def test_edit_file_preview_renders_diff():
    info = review_cards.action_preview(
        {
            "name": "edit_file",
            "args": {
                "file_path": "a.py",
                "old_string": "x = 1\n",
                "new_string": "x = 2\n",
                "replace_all": True,
            },
        }
    )
    label, diff, language = info["blocks"][-1]
    assert label == "Proposed replacement (applies to all occurrences)"
    assert language == "diff"
    assert "-x = 1" in diff and "+x = 2" in diff


def test_unknown_tool_preview_falls_back_to_json_arguments():
    info = review_cards.action_preview({"name": "run_ase", "args": {"steps": 5}})
    assert info["blocks"] == [("Arguments", '{\n  "steps": 5\n}', "json")]


def test_allowed_decisions_and_summary():
    payload = _review(
        [{"name": "execute", "args": {"command": "a"}}, {"name": "execute", "args": {"command": "b"}}],
        allowed=("approve",),
    )
    assert review_cards.allowed_decisions(payload, "execute") == ["approve"]
    assert review_cards.allowed_decisions(payload, "other") == []
    assert review_cards.review_summary(payload) == "Review 2 Deep Agent actions: execute, execute"


# ---------------------------------------------------------------------------
# Resume value construction
# ---------------------------------------------------------------------------


def test_text_answer_rejects_reviews_with_instructions_and_answers_questions():
    records = [
        {"id": "i1", "payload": _review([{"name": "execute", "args": {"command": "rm -rf"}}])},
        {"id": "i2", "payload": {"question": "Which molecule?"}},
    ]
    answers = main_ui._text_answers(records, "Use EMT instead")
    assert answers == [
        {"decisions": [{"type": "reject", "message": "Use EMT instead"}]},
        "Use EMT instead",
    ]
    assert main_ui._build_resume_value(records, answers) == {
        "i1": answers[0],
        "i2": "Use EMT instead",
    }


def test_typed_text_never_approves_when_reject_is_not_allowed():
    records = [{"id": "", "payload": _review([{"name": "execute", "args": {}}], allowed=("approve",))}]
    with pytest.raises(ValueError, match="cannot be rejected"):
        main_ui._text_answers(records, "no, do not run this")


def test_single_pending_interrupt_resumes_with_bare_answer():
    records = [{"id": "", "payload": _review([{"name": "execute", "args": {}}])}]
    answers = main_ui._decision_answers(records, {(0, 0): {"type": "approve"}})
    assert main_ui._build_resume_value(records, answers) == {"decisions": [{"type": "approve"}]}


def test_multiple_pending_without_ids_cannot_resume():
    records = [
        {"id": "", "payload": {"question": "a"}},
        {"id": "x", "payload": {"question": "b"}},
    ]
    with pytest.raises(ValueError, match="stable IDs"):
        main_ui._build_resume_value(records, ["1", "2"])


def test_decision_answers_validate_allowed_and_completeness():
    payload = _review(
        [{"name": "execute", "args": {"command": "a"}}, {"name": "execute", "args": {"command": "a"}}],
        allowed=("approve",),
    )
    records = [{"id": "", "payload": payload}]
    with pytest.raises(ValueError, match="Every action needs a decision"):
        main_ui._decision_answers(records, {(0, 0): {"type": "approve"}})
    with pytest.raises(ValueError, match="not allowed"):
        main_ui._decision_answers(
            records, {(0, 0): {"type": "approve"}, (0, 1): {"type": "reject"}}
        )
    answers = main_ui._decision_answers(
        records, {(0, 0): {"type": "approve"}, (0, 1): {"type": "approve"}}
    )
    assert answers == [{"decisions": [{"type": "approve"}, {"type": "approve"}]}]


# ---------------------------------------------------------------------------
# Streaming: pending interrupts come from the checkpoint with ids
# ---------------------------------------------------------------------------


class _Interrupt:
    def __init__(self, value, id="placeholder-id"):
        self.value = value
        self.id = id


def test_stream_workflow_reports_checkpointed_review_records():
    payload = _review([{"name": "execute", "args": {"command": "pwd"}}])

    async def astream(_input, stream_mode, config):
        yield {"messages": []}
        yield {"__interrupt__": [_Interrupt(payload)]}

    def get_state(config):
        return SimpleNamespace(
            interrupts=[_Interrupt(payload, id="int-42")], tasks=[]
        )

    agent = SimpleNamespace(workflow=SimpleNamespace(astream=astream, get_state=get_state))
    q = queue.Queue()
    main_ui._stream_workflow({"messages": []}, {"configurable": {"thread_id": "1"}}, agent, q)
    events = []
    while not q.empty():
        events.append(q.get())
    assert events[-1][0] == "interrupt"
    assert events[-1][1] == [{"id": "int-42", "payload": payload}]
    assert main_ui._pending_is_review(events[-1][1])
    assert main_ui._pending_summary(events[-1][1]) == "Review 1 Deep Agent action: execute"


def test_stream_workflow_plain_question_keeps_text_summary():
    async def astream(_input, stream_mode, config):
        yield {"__interrupt__": [_Interrupt({"question": "Which solvent?"})]}

    agent = SimpleNamespace(
        workflow=SimpleNamespace(astream=astream, get_state=lambda c: None)
    )
    q = queue.Queue()
    main_ui._stream_workflow({}, {}, agent, q)
    kind, records = q.get()
    assert kind == "interrupt"
    assert records == [{"id": "", "payload": {"question": "Which solvent?"}}]
    assert not main_ui._pending_is_review(records)
    assert main_ui._pending_summary(records) == "Which solvent?"


# ---------------------------------------------------------------------------
# Rendering: Approve button resumes with decisions
# ---------------------------------------------------------------------------


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _ReviewStreamlit:
    def __init__(self, pressed=()):
        self.session_state = _SessionState()
        self.calls = []
        self.pressed = set(pressed)

    def info(self, text, icon=None):
        self.calls.append(("info", text))

    def error(self, text):
        self.calls.append(("error", text))

    def caption(self, text):
        self.calls.append(("caption", text))

    def markdown(self, text):
        self.calls.append(("markdown", text))

    def code(self, text, language=None):
        self.calls.append(("code", text, language))

    def container(self, border=False):
        return nullcontext()

    def expander(self, label, expanded=False):
        return nullcontext()

    def columns(self, spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [nullcontext() for _ in range(count)]

    def button(self, label, key=None, **kwargs):
        self.calls.append(("button", label, key))
        return key in self.pressed

    def radio(self, label, options, index=0, key=None, **kwargs):
        self.calls.append(("radio", key, tuple(options)))
        return options[index]


def test_single_action_approve_button_resumes_with_decision(monkeypatch):
    payload = _review([{"name": "execute", "args": {"command": "pwd"}}])
    fake_st = _ReviewStreamlit(pressed={"review_7_0_0_approve"})
    fake_st.session_state.review_nonce = 7
    fake_st.session_state.pending_interrupt_thread_id = 3
    monkeypatch.setattr(main_ui, "st", fake_st)
    monkeypatch.setattr(review_cards, "st", fake_st)
    resumed = []
    monkeypatch.setattr(
        main_ui, "_resume_pending_interrupts", lambda *args: resumed.append(args)
    )

    main_ui._render_review_cards([{"id": "", "payload": payload}])

    assert resumed == [({"decisions": [{"type": "approve"}]}, "Approved", 3, None)]
    labels = [call[1] for call in fake_st.calls if call[0] == "button"]
    assert labels == ["\u2705 Approve", "\u274c Reject"]
    assert any(call[0] == "code" and call[1] == "pwd" and call[2] == "bash" for call in fake_st.calls)


def test_multiple_actions_use_per_action_choice_and_submit(monkeypatch):
    payload = _review(
        [{"name": "execute", "args": {"command": "a"}}, {"name": "execute", "args": {"command": "a"}}]
    )
    fake_st = _ReviewStreamlit(pressed={"review_2_reject_all"})
    fake_st.session_state.review_nonce = 2
    fake_st.session_state.pending_interrupt_thread_id = 1
    monkeypatch.setattr(main_ui, "st", fake_st)
    monkeypatch.setattr(review_cards, "st", fake_st)
    resumed = []
    monkeypatch.setattr(
        main_ui, "_resume_pending_interrupts", lambda *args: resumed.append(args)
    )

    main_ui._render_review_cards([{"id": "int-1", "payload": payload}])

    radios = [call for call in fake_st.calls if call[0] == "radio"]
    assert [call[1] for call in radios] == ["review_2_0_0_choice", "review_2_0_1_choice"]
    assert resumed == [
        (
            {"decisions": [{"type": "reject"}, {"type": "reject"}]},
            "Rejected all 2 actions",
            1,
            None,
        )
    ]


def test_typed_reply_during_review_rejects_with_instructions(monkeypatch):
    payload = _review([{"name": "write_file", "args": {"file_path": "x", "content": "y"}}])
    fake_st = _ReviewStreamlit()
    fake_st.session_state.pending_interrupts = [{"id": "", "payload": payload}]
    fake_st.session_state.pending_human_question = "Review 1 Deep Agent action: write_file"
    monkeypatch.setattr(main_ui, "st", fake_st)
    resumed = []
    monkeypatch.setattr(
        main_ui, "_resume_pending_interrupts", lambda *args: resumed.append(args)
    )

    main_ui._handle_human_response("Write it to out/ instead", 5, None)

    assert resumed == [
        (
            {"decisions": [{"type": "reject", "message": "Write it to out/ instead"}]},
            "Write it to out/ instead",
            5,
            None,
        )
    ]


# ---------------------------------------------------------------------------
# End to end: a real Deep Agent graph paused and resumed through the UI path
# ---------------------------------------------------------------------------


def _drain(q):
    events = []
    while not q.empty():
        events.append(q.get())
    return events


def test_ui_stream_and_resume_drive_real_deep_agent_review(tmp_path):
    from deepagents.backends import LocalShellBackend
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from chemgraph.graphs.deep_agent import construct_deep_agent_graph
    from tests.test_deep_agent import _RecordingChatModel

    feedback = "Write revised.txt instead; keep blocked.txt untouched."
    model = _RecordingChatModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "write_file", "id": name,
            "args": {"file_path": f"/workspace/{name}.txt", "content": name},
        }])
        for name in ("blocked", "revised")
    ] + [AIMessage(content="Done")])
    worker = construct_deep_agent_graph(
        model, backend=LocalShellBackend(root_dir=tmp_path, env={}),
        discover_skills=False, checkpointer=InMemorySaver(),
    )
    agent = SimpleNamespace(workflow=worker)
    config = {"configurable": {"thread_id": "ui-review"}}

    # 1. First write pauses for review; the UI receives the checkpointed record.
    q = queue.Queue()
    main_ui._stream_workflow(
        {"messages": [HumanMessage(content="Write a file")]}, config, agent, q
    )
    kind, records = _drain(q)[-1]
    assert kind == "interrupt"
    assert main_ui._pending_is_review(records)
    assert records[0]["payload"]["action_requests"][0]["name"] == "write_file"
    assert records[0]["payload"]["action_requests"][0]["args"]["file_path"] == "/workspace/blocked.txt"
    assert not (tmp_path / "blocked.txt").exists()

    # 2. Typed instructions reject that action with the feedback message.
    resume = main_ui._build_resume_value(records, main_ui._text_answers(records, feedback))
    q = queue.Queue()
    main_ui._stream_workflow(Command(resume=resume), config, agent, q)
    kind, records = _drain(q)[-1]
    assert kind == "interrupt"
    assert records[0]["payload"]["action_requests"][0]["args"]["file_path"] == "/workspace/revised.txt"
    assert not (tmp_path / "blocked.txt").exists()
    assert any(m.type == "tool" and feedback in str(m.content) for m in model.received_messages)

    # 3. The Approve button's decision completes the run and writes the file.
    resume = main_ui._build_resume_value(
        records, main_ui._decision_answers(records, {(0, 0): {"type": "approve"}})
    )
    q = queue.Queue()
    main_ui._stream_workflow(Command(resume=resume), config, agent, q)
    kind, state = _drain(q)[-1]
    assert kind == "done"
    assert state["messages"][-1].content == "Done"
    assert (tmp_path / "revised.txt").read_text() == "revised"
    assert not (tmp_path / "blocked.txt").exists()


def test_refused_typed_reply_keeps_the_review_pending(monkeypatch):
    payload = _review([{"name": "execute", "args": {"command": "rm -rf build"}}], allowed=("approve",))
    fake_st = _ReviewStreamlit()
    records = [{"id": "", "payload": payload}]
    fake_st.session_state.pending_interrupts = records
    fake_st.session_state.pending_human_question = "Review 1 Deep Agent action: execute"
    monkeypatch.setattr(main_ui, "st", fake_st)
    resumed = []
    monkeypatch.setattr(main_ui, "_resume_pending_interrupts", lambda *a: resumed.append(a))

    main_ui._handle_human_response("no, do not run this", 1, None)

    assert resumed == []
    assert any(call[0] == "error" and "cannot be rejected" in call[1] for call in fake_st.calls)
    assert fake_st.session_state.pending_interrupts == records
    assert fake_st.session_state.pending_human_question is not None


def test_card_identity_line_is_literal_not_markdown(monkeypatch):
    fake_st = _ReviewStreamlit()
    monkeypatch.setattr(review_cards, "st", fake_st)
    path = "/tmp/ok.txt` ![x](https://example.invalid/p.png) `/home/u/.ssh/authorized_keys"
    review_cards.render_action_card(
        {"name": "write_file", "args": {"file_path": path, "content": "k"}}, 1, 1
    )
    markdown = [call[1] for call in fake_st.calls if call[0] == "markdown"]
    assert markdown == ["**Review action 1 of 1**"]
    codes = [call for call in fake_st.calls if call[0] == "code"]
    assert codes[0] == ("code", f"Tool: write_file | Path: {path}", "text")


def test_set_agent_log_dir_updates_process_and_deep_agent_shell(tmp_path, monkeypatch):
    from chemgraph.agent.deepagent_backend import create_host_shell_backend

    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "chat"))
    backend = create_host_shell_backend(str(tmp_path))
    agent = SimpleNamespace(deepagent_backend=backend)
    turn = str(tmp_path / "chat" / "turn_003_beef")
    main_ui._set_agent_log_dir(agent, turn)
    assert main_ui.os.environ["CHEMGRAPH_LOG_DIR"] == turn
    assert backend.execute("echo $CHEMGRAPH_LOG_DIR").output.strip() == turn
    main_ui._set_agent_log_dir(SimpleNamespace(), str(tmp_path))  # non-deep agents: env only
    assert main_ui.os.environ["CHEMGRAPH_LOG_DIR"] == str(tmp_path)


def test_attachment_note_reaches_rejections_and_multi_interrupt_answers():
    note = "\n\n[Attached files: /t/replacement.xyz]"
    assert main_ui._append_attachment_note("use it", note) == "use it" + note
    review = {"decisions": [{"type": "approve"}, {"type": "reject", "message": "Use the attached replacement"}]}
    assert main_ui._append_attachment_note(review, note) == {
        "decisions": [{"type": "approve"}, {"type": "reject", "message": "Use the attached replacement" + note}]
    }
    assert review["decisions"][1]["message"] == "Use the attached replacement"  # not mutated
    multi = {"id-a": "answer", "id-b": {"decisions": [{"type": "reject", "message": "m"}]}}
    assert main_ui._append_attachment_note(multi, note) == {
        "id-a": "answer" + note,
        "id-b": {"decisions": [{"type": "reject", "message": "m" + note}]},
    }
    assert main_ui._append_attachment_note(review, "") is review
