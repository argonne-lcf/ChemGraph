import json

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

from chemgraph.agent.main_session import MainAgentTurnResult
from chemgraph.cli.formatting import (
    _content_text, console, format_action_review, format_response,
)


@pytest.mark.parametrize("name,args,expected", [
    (
        "execute", {"command": "echo [red]literal[/red]\necho done"},
        ["echo [red]literal[/red]", "echo done"],
    ),
    (
        "write_file",
        {"file_path": "/new.txt", "content": "[red]literal[/red]\nsecond line"},
        ["/new.txt", "[red]literal[/red]", "second line"],
    ),
    (
        "edit_file",
        {"file_path": "/file.txt", "old_string": "old\n", "new_string": "new\n", "replace_all": True},
        ["Proposed replacement snippet", "-old", "+new", '"replace_all": true'],
    ),
    (
        "edit_file", {"old_string": "same", "new_string": "same\n"},
        ["-same", "+same", "No newline at end of file"],
    ),
    (
        "[red]custom[/red]", {"params": {"calculator": "EMT"}},
        ["[red]custom[/red]", '"calculator": "EMT"'],
    ),
    ("custom", {(1, 2): "fallback"}, ["(1, 2)", "fallback"]),
])
def test_action_preview_is_literal_and_does_not_read_files(
    monkeypatch, name, args, expected,
):
    from copy import deepcopy

    original = deepcopy(args)
    monkeypatch.setattr(
        "builtins.open", lambda *_, **__: pytest.fail("preview must not read files"),
    )
    terminal = Console(width=120)
    with terminal.capture() as capture:
        terminal.print(format_action_review({"name": name, "args": args}, 2, 3))
    output = capture.get()
    assert "Review action 2 of 3" in output
    assert all(text in output for text in expected)
    assert "\\n" not in output
    assert args == original


def test_content_text_normalizes_structured_blocks():
    content = [
        {"type": "reasoning", "reasoning": "internal"},
        {"type": "text", "text": "Optimization complete"},
        "!",
        {"type": "tool_call", "name": "run_ase"},
    ]

    assert _content_text(content) == "Optimization complete!"


def test_format_response_renders_structured_ai_content():
    result = {
        "messages": [
            {
                "type": "ai",
                "content": [{"type": "text", "text": "Dictionary response"}],
            },
            AIMessage(
                content=[{"type": "text", "text": "Optimization complete"}]
            ),
        ]
    }

    with console.capture() as capture:
        format_response(result)

    output = capture.get()
    assert "ChemGraph Response" in output
    assert "Optimization complete" in output


def test_format_response_detects_atomic_json_in_structured_content():
    structure = json.dumps(
        {
            "numbers": [8, 1, 1],
            "positions": [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        }
    )
    result = {
        "messages": [
            AIMessage(content=[{"type": "text", "text": structure}]),
            AIMessage(content=[{"type": "text", "text": "Water optimized"}]),
        ]
    }

    with console.capture() as capture:
        format_response(result)

    output = capture.get()
    assert "ChemGraph Response" in output
    assert "Water optimized" in output
    assert "Molecular Structure Data" in output


def test_format_response_renders_main_agent_turn_result():
    result = MainAgentTurnResult(
        thread_id="thread-1",
        status="waiting_for_user",
        assistant_response="Delegated calculation complete.",
        interrupts=(),
        state={},
    )

    with console.capture() as capture:
        format_response(result)

    output = capture.get()
    assert "ChemGraph Response" in output
    assert "Delegated calculation complete" in output
