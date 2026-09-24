"""End-to-end smoke tests for the Streamlit app using st.testing.

These run the real app script (navigation, first-run setup, chat page)
in-process without a browser. Provider credentials and config paths are
isolated so results do not depend on the developer's environment.
"""

from pathlib import Path

import pytest

_APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py")

_CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "ARGO_USER",
    "ALCF_ACCESS_TOKEN",
    "CHEMGRAPH_LOG_DIR",
)


@pytest.fixture()
def isolated_app(monkeypatch, tmp_path):
    """Return an AppTest for the app with credentials/config isolated."""
    from streamlit.testing.v1 import AppTest

    for var in _CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)

    import ui.alcf_auth as alcf_auth
    import ui.codex_auth as codex_auth
    import ui.config as ui_config

    # Never spawn the Codex app-server from tests; default to "logged out".
    monkeypatch.setattr(
        codex_auth,
        "account_status",
        lambda use_cache=True: codex_auth.CodexStatus(
            codex_auth.STATE_LOGGED_OUT, "No Codex login is available."
        ),
    )
    monkeypatch.setattr(
        ui_config, "_DEFAULT_CONFIG_PATH", str(tmp_path / "config.toml")
    )
    monkeypatch.setattr(
        alcf_auth, "CHEMGRAPH_TOKENS_PATH", str(tmp_path / "tokens.json")
    )
    monkeypatch.setattr(
        alcf_auth, "HELPER_TOKENS_PATH", str(tmp_path / "helper.json")
    )
    return AppTest.from_file(_APP_PATH, default_timeout=60)


def test_first_run_setup_renders_without_credentials(isolated_app):
    at = isolated_app.run()

    assert not at.exception
    assert any("Welcome" in info.value for info in at.info)
    labels = [b.label for b in at.button]
    assert "Use Argo" in labels
    assert "Skip setup for now" in labels


def test_argo_setup_persists_username_and_enters_chat(isolated_app, tmp_path):
    at = isolated_app.run()

    at.text_input(key="setup_argo_user").set_value("aturing")
    at.run()
    at.button(key="setup_argo_go").click()
    at.run()

    assert not at.exception
    config = at.session_state["config"]
    assert config["api"]["argo"]["argo_user"] == "aturing"
    assert config["general"]["model"].startswith("argo:")
    # The wizard is gone; the chat input is available.
    assert len(at.chat_input) == 1
    # And the choice was persisted to disk.
    assert (tmp_path / "config.toml").exists()


def test_chat_page_renders_with_provider_key(isolated_app, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    at = isolated_app.run()

    assert not at.exception
    # No wizard; chat input is present immediately.
    assert not any("Welcome" in info.value for info in at.info)
    assert len(at.chat_input) == 1


def test_codex_setup_tab_offers_sign_in_when_logged_out(isolated_app):
    at = isolated_app.run()

    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Sign in with ChatGPT" in label for label in labels)
    assert not any(b.key == "setup_codex_go" for b in at.button)


def test_codex_setup_uses_codex_model_when_signed_in(isolated_app, monkeypatch, tmp_path):
    import ui.codex_auth as codex_auth

    monkeypatch.setattr(
        codex_auth,
        "account_status",
        lambda use_cache=True: codex_auth.CodexStatus(
            codex_auth.STATE_CHATGPT, "Signed in to Codex with ChatGPT.", "chemist@example.com"
        ),
    )
    at = isolated_app.run()
    assert not at.exception
    at.button(key="setup_codex_go").click()
    at.run()

    assert not at.exception
    config = at.session_state["config"]
    assert config["general"]["model"] == "codex:gpt-5"
    assert len(at.chat_input) == 1
    # A Log out action is offered on the Configuration page for a live login.
    conf = _configuration_apptest()
    conf.run()
    assert not conf.exception
    assert any(b.key == "config_codex_logout" for b in conf.button)


def _configuration_apptest():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string(
        "from ui._pages.configuration import render\nrender()",
        default_timeout=60,
    )


@pytest.fixture
def configuration_app(isolated_app):
    """Reuse credential/path isolation while exercising the editor directly."""
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string(
        "from ui._pages.configuration import render\nrender()",
        default_timeout=60,
    )


def test_configuration_save_preserves_automatic_selection(configuration_app, tmp_path):
    import toml

    at = configuration_app.run()
    assert not at.exception
    selector = next(s for s in at.selectbox if s.label == "Default Calculator")
    assert selector.value is None
    assert "default" not in at.session_state["_config_draft"]["chemistry"]["calculators"]
    next(b for b in at.button if "Save Configuration" in b.label).click().run()
    assert not at.exception
    assert "default" not in toml.load(tmp_path / "config.toml")["chemistry"]["calculators"]


def test_configuration_recursion_limit_defaults_and_persists(configuration_app, tmp_path):
    import toml

    at = configuration_app.run()
    assert not at.exception
    limit = next(widget for widget in at.number_input if widget.label == "Recursion Limit")
    assert limit.value == 200
    limit.set_value(350).run()
    next(b for b in at.button if "Save Configuration" in b.label).click().run()
    assert not at.exception
    assert toml.load(tmp_path / "config.toml")["general"]["recursion_limit"] == 350


def test_raw_toml_can_restore_automatic_selection(configuration_app, tmp_path):
    import toml

    at = configuration_app.run()
    next(s for s in at.selectbox if s.label == "Default Calculator").select("mace_mp").run()
    next(b for b in at.button if "Save Configuration" in b.label).click().run()
    at.run()
    assert not at.exception
    assert toml.load(tmp_path / "config.toml")["chemistry"]["calculators"]["default"] == "mace_mp"
    previous_nonce = at.session_state["_config_widget_nonce"]
    toml_area = next(t for t in at.text_area if t.label == "TOML Content")
    toml_area.set_value('[chemistry.calculators]\nfallback = "emt"\n')
    at.button(key="update_from_toml").click().run()
    assert not at.exception
    assert at.session_state["_config_widget_nonce"] > previous_nonce
    assert next(s for s in at.selectbox if s.label == "Default Calculator").value is None
    assert "default" not in at.session_state["_config_draft"]["chemistry"]["calculators"]
    assert at.session_state["config"]["chemistry"]["calculators"]["default"] == "mace_mp"
    next(b for b in at.button if "Save Configuration" in b.label).click().run()
    assert not at.exception
    assert "default" not in toml.load(tmp_path / "config.toml")["chemistry"]["calculators"]


def test_deep_agent_tab_persists_cli_compatible_keys(configuration_app, tmp_path):
    import toml

    workspace = tmp_path / "ws"
    workspace.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()

    at = configuration_app.run()
    assert not at.exception
    workflow = next(s for s in at.selectbox if s.label == "Workflow")
    assert "deep_agent" in workflow.options
    workflow.select("deep_agent").run()

    next(t for t in at.text_input if t.label == "Workspace directory").set_value(
        str(workspace)
    ).run()
    next(t for t in at.text_area if t.label == "One directory per line").set_value(
        f"{skills}\n"
    ).run()
    next(c for c in at.checkbox if c.label.startswith("Discover personal")).uncheck().run()
    next(c for c in at.checkbox if c.label.startswith("Restrict the catalog")).check().run()
    tools = next(m for m in at.multiselect if m.label == "Tools")
    assert "run_ase" in tools.options
    tools.select("run_ase").run()
    next(b for b in at.button if "Save Configuration" in b.label).click().run()
    assert not at.exception

    general = toml.load(tmp_path / "config.toml")["general"]
    assert general["workflow"] == "deep_agent"
    assert general["deepagent_workspace"] == str(workspace)
    assert general["deepagent_skills"] == [str(skills)]
    assert general["deepagent_discover_skills"] is False
    assert general["tools"] == ["run_ase"]

    # The acknowledgment is per-session, never written to disk.
    assert "deepagent_host_shell_acknowledged" not in general
    assert at.session_state["deepagent_host_shell_acknowledged"] is False
    next(c for c in at.checkbox if c.label.startswith("I understand the Deep Agent")).check().run()
    assert at.session_state["deepagent_host_shell_acknowledged"] is True
