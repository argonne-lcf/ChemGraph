"""Live progress cooperates with logging handlers created before the spinner."""

import io
import logging
import sys

import pytest
from rich.console import Console

from chemgraph.cli import commands


@pytest.mark.parametrize("cancelled", [False, True])
def test_progress_routes_existing_log_handlers_through_live_console(monkeypatch, tmp_path, cancelled):
    stdout, stderr, display = io.StringIO(), io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(commands, "console", Console(
        file=display, force_terminal=True, color_system=None, width=100,
    ))
    logger = logging.getLogger("progress_test")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.WARNING)
    stdout_handler = logging.StreamHandler(stdout)
    stderr_handler = logging.StreamHandler(stderr)
    file_handler = logging.FileHandler(tmp_path / "tool.log")
    for handler in (stdout_handler, stderr_handler, file_handler):
        logger.addHandler(handler)

    try:
        try:
            with commands._working_progress():
                logger.warning("calculator diagnostic")
                print("optimizer step")
                assert stdout.getvalue() == stderr.getvalue() == ""
                assert display.getvalue().count("calculator diagnostic") == 2
                assert "optimizer step" in display.getvalue()
                assert (tmp_path / "tool.log").read_text() == "calculator diagnostic\n"
                if cancelled:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            assert cancelled

        assert sys.stdout is stdout and sys.stderr is stderr
        assert stdout_handler.stream is stdout
        assert stderr_handler.stream is stderr
        logger.warning("after progress")
        assert stdout.getvalue() == stderr.getvalue() == "after progress\n"
    finally:
        file_handler.close()


def test_nonterminal_progress_keeps_log_streams(monkeypatch):
    stderr, display = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(commands, "console", Console(file=display, force_terminal=False))
    logger = logging.getLogger("progress_test")
    handler = logging.StreamHandler(stderr)
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "propagate", False)
    with commands._working_progress():
        assert handler.stream is stderr
        logger.warning("calculator diagnostic")
    assert stderr.getvalue() == "calculator diagnostic\n"
    assert "calculator diagnostic" not in display.getvalue()
