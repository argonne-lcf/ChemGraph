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

    def fake_call(cmd):
        captured["cmd"] = cmd
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
