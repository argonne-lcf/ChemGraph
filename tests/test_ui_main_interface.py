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


def test_failed_agent_initialization_is_not_cached(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.agent = None
    fake_st.session_state.last_config = ("previous",)
    monkeypatch.setattr(main_ui, "st", fake_st)
    monkeypatch.setattr(main_ui, "initialize_agent", lambda *args: None)

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


def test_resolve_artifact_path_uses_run_directory_for_relative_paths(tmp_path):
    assert main_ui._resolve_artifact_path("mol_vib.1.traj", str(tmp_path)) == str(
        tmp_path / "mol_vib.1.traj"
    )
