from contextlib import nullcontext
import os

from ui._pages import main_interface as main_ui
from ui.message_utils import extract_molecular_structure, normalize_latex_delimiters


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


class _FakeSidebar:
    def __init__(self):
        self.calls = []

    def expander(self, label, expanded=False):
        self.calls.append(("expander", label, expanded))
        return nullcontext()


class _FakeStreamlitWithSidebar(_FakeStreamlit):
    def __init__(self):
        super().__init__()
        self.sidebar = _FakeSidebar()
        self.calls = []

    def caption(self, text):
        self.calls.append(("caption", text))

    def success(self, text):
        self.calls.append(("success", text))

    def markdown(self, text):
        self.calls.append(("markdown", text))

    def latex(self, text):
        self.calls.append(("latex", text))


class _FakeComponentsV1:
    def __init__(self, calls):
        self.calls = calls

    def html(self, html, height=None, scrolling=False):
        self.calls.append(("html", html, height, scrolling))


class _FakeComponents:
    def __init__(self, calls):
        self.v1 = _FakeComponentsV1(calls)


class _FakeStreamlitRich(_FakeStreamlit):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.components = _FakeComponents(self.calls)

    def expander(self, label, expanded=False):
        self.calls.append(("expander", label, expanded))
        return nullcontext()

    def write(self, text):
        self.calls.append(("write", text))

    def markdown(self, text):
        self.calls.append(("markdown", text))

    def latex(self, text):
        self.calls.append(("latex", text))

    def code(self, text, language=None):
        self.calls.append(("code", text, language))

    def download_button(self, label, data, file_name, mime, key=None):
        self.calls.append(("download_button", label, data, file_name, mime, key))

    def warning(self, text):
        self.calls.append(("warning", text))

    def error(self, text):
        self.calls.append(("error", text))


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


def test_available_calculators_sidebar_replaces_quick_settings(monkeypatch):
    fake_st = _FakeStreamlitWithSidebar()
    monkeypatch.setattr(main_ui, "st", fake_st)
    monkeypatch.setattr(
        main_ui,
        "get_available_calculator_names",
        lambda: ["MaceCalc", "TBLiteCalc", "EMTCalc"],
    )
    monkeypatch.setattr(main_ui, "get_default_calculator_name", lambda: "MaceCalc")

    main_ui._render_available_calculators_sidebar()

    assert fake_st.sidebar.calls == [
        ("expander", "\U0001f9ee Available Calculators", True)
    ]
    rendered_text = "\n".join(text for _, text in fake_st.calls)
    assert "Mace (default)" in rendered_text
    assert "TBLite (xTB, GFN1-xTB, GFN2-xTB)" in rendered_text
    assert "Quick Settings" not in rendered_text


def test_extract_structure_ignores_non_dict_answer_without_crashing():
    # A JSON message whose "answer" is prose/list that merely mentions the
    # words numbers/positions must not raise (previously TypeError).
    assert (
        extract_molecular_structure(
            '{"answer": "the numbers and positions are unknown"}'
        )
        is None
    )
    assert (
        extract_molecular_structure('{"answer": ["numbers", "positions"]}') is None
    )
    # Null structure fields should not be reported as a structure.
    assert extract_molecular_structure('{"numbers": null, "positions": null}') is None


def test_extract_structure_parses_valid_payloads():
    flat = extract_molecular_structure(
        '{"numbers": [1, 8, 1], "positions": [[0,0,0],[0,0,1],[0,1,0]]}'
    )
    assert flat == {
        "atomic_numbers": [1, 8, 1],
        "positions": [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    }
    nested = extract_molecular_structure(
        '{"answer": {"atomic_numbers": [8, 1, 1], '
        '"positions": [[0,0,0],[0,0,1],[0,1,0]]}}'
    )
    assert nested == {
        "atomic_numbers": [8, 1, 1],
        "positions": [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    }


def test_parenthetical_prose_is_not_mangled_into_math():
    # Prose with a bare subscript must stay untouched...
    assert (
        normalize_latex_delimiters("The energy is -76.4 eV (a_1 symmetry).")
        == "The energy is -76.4 eV (a_1 symmetry)."
    )
    assert normalize_latex_delimiters("Rate (k_B T) term.") == "Rate (k_B T) term."
    # ...while genuine equations still convert to inline math.
    assert normalize_latex_delimiters("Momentum (E = mc^2) here.") == (
        "Momentum $E = mc^2$ here."
    )


def test_num_nonvibrational_modes_handles_linear_and_small_systems():
    # No geometry available -> conservative default of 6 for polyatomics.
    assert main_ui._num_nonvibrational_modes(9, None) == 6
    # Diatomic is always linear (5 non-vibrational modes).
    assert main_ui._num_nonvibrational_modes(6, None) == 5
    # Single atom has no vibrations.
    assert main_ui._num_nonvibrational_modes(3, None) == 3


def test_is_linear_geometry_detects_linear_molecules():
    from ase import Atoms

    co2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    h2o = Atoms("H2O", positions=[[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]])
    assert main_ui._is_linear_geometry(co2) is True
    assert main_ui._is_linear_geometry(h2o) is False


def test_latex_delimiters_are_normalized_for_streamlit_markdown():
    raw = (
        "Using the combustion reaction\n\n"
        "[ \\mathrm{CH_4 + 2\\,O_2 \\rightarrow CO_2 + 2\\,H_2O} ]\n\n"
        "(H(\\mathrm{CH_4}) = -21.88) eV\n\n"
        "Per mole, using (1\\ \\text{eV} = 96.485\\ \\text{kJ mol}^{-1}):"
    )

    normalized = normalize_latex_delimiters(raw)

    assert "$$\n\\mathrm{CH_4 + 2\\,O_2 \\rightarrow CO_2 + 2\\,H_2O}\n$$" in normalized
    assert "$H(\\mathrm{CH_4}) = -21.88$ eV" in normalized
    assert (
        "Per mole, using $1\\ \\text{eV} = "
        "96.485\\ \\text{kJ mol}^{-1}$:"
    ) in normalized


def test_nested_square_brackets_in_display_math_are_preserved():
    raw = (
        "Reaction enthalpy:\n\n"
        "[ \\Delta H_\\mathrm{rxn} = \\left[H(\\mathrm{CO_2}) + "
        "2H(\\mathrm{H_2O})\\right]\n\n"
        "\\left[H(\\mathrm{CH_4}) + 2H(\\mathrm{O_2})\\right] ]"
    )

    blocks = main_ui.split_markdown_latex_blocks(raw)

    assert blocks == [
        ("markdown", "Reaction enthalpy:"),
        (
            "latex",
            "\\Delta H_\\mathrm{rxn} = \\left[H(\\mathrm{CO_2}) + "
            "2H(\\mathrm{H_2O})\\right]\n\n"
            "\\left[H(\\mathrm{CH_4}) + 2H(\\mathrm{O_2})\\right]",
        ),
    ]


def test_markdown_with_math_uses_streamlit_latex_for_display_blocks(monkeypatch):
    fake_st = _FakeStreamlitRich()
    monkeypatch.setattr(main_ui, "st", fake_st)

    main_ui._render_markdown_with_math(
        "Using reaction\n\n"
        "[ \\mathrm{CH_4 + 2\\,O_2 \\rightarrow CO_2 + 2\\,H_2O} ]\n\n"
        "Done"
    )

    assert ("markdown", "Using reaction") in fake_st.calls
    assert (
        "latex",
        "\\mathrm{CH_4 + 2\\,O_2 \\rightarrow CO_2 + 2\\,H_2O}",
    ) in fake_st.calls
    assert ("markdown", "Done") in fake_st.calls


def test_multiline_latex_blocks_are_wrapped_for_katex():
    prepared = main_ui._prepare_latex_block("a = b\n\nc = d")

    assert prepared == "\\begin{aligned}\na = b \\\\\nc = d\n\\end{aligned}"


def test_verbose_info_shows_pretty_raw_result_only(monkeypatch):
    fake_st = _FakeStreamlitRich()
    fake_st.session_state.last_run_query = "query"
    fake_st.session_state.last_run_error = None
    fake_st.session_state.last_run_result = {"messages": [{"type": "ai", "content": "x"}]}
    monkeypatch.setattr(main_ui, "st", fake_st)

    main_ui._render_verbose_info(
        1,
        [{"type": "ai", "content": "x"}],
        {"query": "query", "result": {"messages": [{"type": "ai", "content": "x"}]}},
    )

    code_calls = [call for call in fake_st.calls if call[0] == "code"]
    assert len(code_calls) == 1
    assert "\n" in code_calls[0][1]
    assert not any("Message 1" in str(call) for call in fake_st.calls)


def test_html_report_includes_download_button(monkeypatch, tmp_path):
    fake_st = _FakeStreamlitRich()
    monkeypatch.setattr(main_ui, "st", fake_st)
    report_path = tmp_path / "report.html"
    report_path.write_text("<html><body>Report</body></html>", encoding="utf-8")

    main_ui._render_html_report(
        2,
        "report.html",
        [{"content": "report.html"}],
        {"log_dir": str(tmp_path)},
    )

    download_calls = [call for call in fake_st.calls if call[0] == "download_button"]
    assert download_calls == [
        (
            "download_button",
            "Download HTML Report",
            "<html><body>Report</body></html>",
            "report.html",
            "text/html",
            "download_report_2",
        )
    ]


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
