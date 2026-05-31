from contextlib import nullcontext
import os

from ui._pages import main_interface as main_ui


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStreamlit:
    def __init__(self):
        self.session_state = _SessionState()

    def spinner(self, _message):
        return nullcontext()


def test_argo_structured_output_is_disabled_after_model_selection():
    argo_model = next(iter(main_ui.supported_argo_models))

    structured, notice = main_ui._resolve_structured_output_for_model(
        argo_model, True
    )

    assert structured is False
    assert "Structured output is disabled" in notice


def test_non_argo_structured_output_is_preserved():
    structured, notice = main_ui._resolve_structured_output_for_model(
        "gpt-4o-mini", True
    )

    assert structured is True
    assert notice is None


def test_failed_agent_initialization_is_not_cached(monkeypatch, tmp_path):
    fake_st = _FakeStreamlit()
    fake_st.session_state.agent = None
    fake_st.session_state.last_config = ("previous",)
    monkeypatch.setattr(main_ui, "st", fake_st)
    monkeypatch.setattr(main_ui, "_ensure_chat_log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(main_ui, "initialize_agent", lambda *args, **kwargs: None)

    main_ui._auto_initialize_agent(
        {"general": {"recursion_limit": 20}, "api": {"openai": {}}},
        "gpt-4o-mini",
        "single_agent",
        False,
        "state",
        False,
        False,
        None,
    )

    assert fake_st.session_state.agent is None
    assert fake_st.session_state.last_config is None


def test_active_session_metadata_prefers_ui_overrides(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.config = {
        "general": {"model": "gpt-4o-mini", "workflow": "single_agent"}
    }
    fake_st.session_state.active_model = "custom-model"
    fake_st.session_state.active_workflow = "python_repl"
    fake_st.session_state.pending_interrupt_model = None
    fake_st.session_state.pending_interrupt_workflow = None
    monkeypatch.setattr(main_ui, "st", fake_st)

    model, workflow = main_ui._active_session_metadata()

    assert model == "custom-model"
    assert workflow == "python_relp"


def test_active_session_metadata_prefers_pending_interrupt(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.config = {
        "general": {"model": "gpt-4o-mini", "workflow": "single_agent"}
    }
    fake_st.session_state.active_model = "new-model"
    fake_st.session_state.active_workflow = "multi_agent"
    fake_st.session_state.pending_interrupt_model = "original-model"
    fake_st.session_state.pending_interrupt_workflow = "single_agent"
    monkeypatch.setattr(main_ui, "st", fake_st)

    model, workflow = main_ui._active_session_metadata()

    assert model == "original-model"
    assert workflow == "single_agent"


def test_latest_artifact_path_uses_newest_shallow_match(tmp_path):
    older = tmp_path / "ir_spectrum_old.png"
    newer = tmp_path / "ir_spectrum_new.png"
    older.write_text("old")
    newer.write_text("new")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert main_ui._latest_artifact_path(str(tmp_path), "ir_spectrum*.png") == str(
        newer
    )


def test_latest_artifact_path_does_not_fallback_to_global_env(monkeypatch, tmp_path):
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    stale_file = stale_dir / "ir_spectrum_old.png"
    stale_file.write_text("old")
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(stale_dir))

    assert main_ui._latest_artifact_path(None, "ir_spectrum*.png") is None


def test_resolve_artifact_path_uses_run_directory_for_relative_paths(tmp_path):
    assert main_ui._resolve_artifact_path("mol_vib.1.traj", str(tmp_path)) == str(
        tmp_path / "mol_vib.1.traj"
    )


def test_artifact_log_dir_prefers_conversation_entry():
    messages = [{"content": "saved to /old/run/output.json"}]
    entry = {"log_dir": "/current/chat"}

    assert main_ui._artifact_log_dir(messages, entry) == "/current/chat"


def test_start_new_chat_clears_history_agent_and_log_dir(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.conversation_history = [{"query": "old"}]
    fake_st.session_state.current_session_id = "abc123"
    fake_st.session_state.session_created = True
    fake_st.session_state.query_input = "draft"
    fake_st.session_state.last_run_error = RuntimeError("old error")
    fake_st.session_state.last_run_result = {"messages": ["old"]}
    fake_st.session_state.last_run_query = "old query"
    fake_st.session_state._pending_example_query = "example"
    fake_st.session_state.agent = object()
    fake_st.session_state.last_config = ("old",)
    fake_st.session_state.current_chat_log_dir = "/tmp/old-chat"
    fake_st.session_state.pending_human_question = "question"
    fake_st.session_state.pending_interrupt_config = {"configurable": {"thread_id": "1"}}
    fake_st.session_state.pending_interrupt_query = "interrupted"
    fake_st.session_state.pending_interrupt_thread_id = 1
    fake_st.session_state.pending_interrupt_prev_msg_count = 3
    fake_st.session_state.pending_interrupt_model = "old-model"
    fake_st.session_state.pending_interrupt_workflow = "single_agent"
    fake_st.session_state.pending_interrupt_log_dir = "/tmp/old-chat"
    fake_st.session_state.interrupt_count = 2
    fake_st.session_state.interrupt_exchanges = [{"question": "q", "answer": "a"}]
    monkeypatch.setattr(main_ui, "st", fake_st)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", "/tmp/old-chat")

    main_ui._start_new_chat()

    assert fake_st.session_state.conversation_history == []
    assert fake_st.session_state.current_session_id is None
    assert fake_st.session_state.session_created is False
    assert fake_st.session_state.query_input == ""
    assert fake_st.session_state.last_run_error is None
    assert fake_st.session_state.last_run_result is None
    assert fake_st.session_state.last_run_query is None
    assert "_pending_example_query" not in fake_st.session_state
    assert fake_st.session_state.agent is None
    assert fake_st.session_state.last_config is None
    assert fake_st.session_state.current_chat_log_dir is None
    assert "CHEMGRAPH_LOG_DIR" not in os.environ
    assert fake_st.session_state.pending_human_question is None
    assert fake_st.session_state.pending_interrupt_config is None
    assert fake_st.session_state.pending_interrupt_query is None
    assert fake_st.session_state.pending_interrupt_thread_id is None
    assert fake_st.session_state.pending_interrupt_prev_msg_count == 0
    assert fake_st.session_state.pending_interrupt_model is None
    assert fake_st.session_state.pending_interrupt_workflow is None
    assert fake_st.session_state.pending_interrupt_log_dir is None
    assert fake_st.session_state.interrupt_count == 0
    assert fake_st.session_state.interrupt_exchanges == []
