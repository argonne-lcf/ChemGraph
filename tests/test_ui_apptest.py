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
    import ui.config as ui_config

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
