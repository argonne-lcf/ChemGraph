"""Only top-level inputs enter readline's persistent history."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from chemgraph.cli import commands, history
from chemgraph.cli.formatting import console


def test_history_excludes_defaults_and_review_retries(monkeypatch, tmp_path):
    entries, auto = [], [True]
    readline = SimpleNamespace(
        clear_history=entries.clear,
        set_history_length=lambda length: None,
        set_auto_history=lambda enabled: auto.__setitem__(0, enabled),
        read_history_file=lambda path: entries.extend(Path(path).read_text().splitlines()),
        write_history_file=lambda path: Path(path).write_text("\n".join(entries) + "\n"),
        get_current_history_length=lambda: len(entries),
        get_history_item=lambda index: entries[index - 1] if index else None,
        add_history=entries.append,
    )
    monkeypatch.setitem(sys.modules, "readline", readline)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    inputs = iter(["optimize methane", "", "n", "y"])
    def input_text(*args, **kwargs):
        try:
            value = next(inputs)
        except StopIteration:
            raise EOFError from None
        if auto[0]:
            entries.append(value)
        return value
    monkeypatch.setattr(console, "input", input_text)
    monkeypatch.setattr(commands.Prompt, "get_input", lambda *args, **kwargs: input_text())
    monkeypatch.setattr(commands, "initialize_agent", lambda *args, **kwargs: SimpleNamespace())
    review = {
        "action_requests": [{"name": "execute", "args": {"command": "test"}}],
        "review_configs": [{"action_name": "execute", "allowed_decisions": ["approve"]}],
    }
    def query(*args, **kwargs):
        for _ in range(2):
            assert commands._prompt_for_interrupt(review) == {"decisions": [{"type": "approve"}]}
    monkeypatch.setattr(commands, "run_query", query)
    with console.capture() as captured:
        commands.interactive_mode()
    assert "That response is not allowed" in captured.get()
    assert entries == ["optimize methane"] and auto[0]
    path = tmp_path / ".chemgraph" / "cli_history"
    assert path.stat().st_mode & 0o777 == 0o600
    with history.enable_input_history() as record:
        record("optimize methane")
        record("   ")
        assert entries == ["optimize methane"]
    assert path.read_text() == "optimize methane\n" and auto[0]


@pytest.mark.parametrize("mode", ["not_tty", "missing_readline", "missing_auto_history"])
def test_history_is_optional(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: mode != "not_tty"))
    monkeypatch.setitem(sys.modules, "readline", None if mode == "missing_readline" else SimpleNamespace())
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with history.enable_input_history() as record:
        record("query")
    assert not (tmp_path / ".chemgraph").exists()
