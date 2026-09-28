"""Tests for the web-UI launcher and its CLI wiring."""

import sys

from ui import streamlit_launcher


def test_app_path_points_at_real_file():
    path = streamlit_launcher.app_path()
    assert path.endswith("app.py")
    import os

    assert os.path.exists(path)


def test_launch_builds_streamlit_command(monkeypatch):
    captured = {}

    def fake_call(cmd, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return 0

    monkeypatch.setattr(streamlit_launcher.subprocess, "call", fake_call)

    code = streamlit_launcher.launch(
        address="0.0.0.0",
        port=9000,
        headless=True,
        extra_args=["--server.enableCORS", "false"],
    )

    assert code == 0
    cmd = captured["cmd"]
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert cmd[4] == streamlit_launcher.app_path()
    assert cmd[cmd.index("--server.address") + 1] == "0.0.0.0"
    assert cmd[cmd.index("--server.port") + 1] == "9000"
    assert cmd[cmd.index("--server.headless") + 1] == "true"
    assert cmd[-2:] == ["--server.enableCORS", "false"]


def test_cli_ui_subcommand_dispatches_to_launcher(monkeypatch):
    import importlib

    # chemgraph.cli.__init__ re-exports main() under the submodule's name,
    # so plain attribute imports would return the function.
    cli_main = importlib.import_module("chemgraph.cli.main")

    parser = cli_main.create_argument_parser()
    args = parser.parse_args(["ui", "--port", "9001", "--headless"])

    assert args.command == "ui"
    assert args.port == 9001
    assert args.headless is True


def test_deep_agent_is_off_unless_enabled_at_launch(monkeypatch, tmp_path):
    from ui.deepagent_policy import ENABLE_ENV, ROOTS_ENV

    captured = {}
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    monkeypatch.delenv(ROOTS_ENV, raising=False)
    monkeypatch.setattr(
        streamlit_launcher.subprocess,
        "call",
        lambda cmd, env=None: captured.update(env=env) or 0,
    )
    streamlit_launcher.launch()
    assert ENABLE_ENV not in captured["env"]
    assert ROOTS_ENV not in captured["env"]

    streamlit_launcher.launch(enable_deep_agent=True, deep_agent_roots=[str(tmp_path)])
    assert captured["env"][ENABLE_ENV] == "1"
    assert captured["env"][ROOTS_ENV] == str(tmp_path.resolve())


def test_cli_ui_forwards_deep_agent_flags():
    from chemgraph.cli.main import create_argument_parser

    parser = create_argument_parser()
    args = parser.parse_args(["ui", "--enable-deep-agent", "--deep-agent-root", "a", "--deep-agent-root", "b"])
    assert args.enable_deep_agent is True
    assert args.deep_agent_root == ["a", "b"]
    assert parser.parse_args(["ui"]).enable_deep_agent is False
