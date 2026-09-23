import json

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

from chemgraph.agent.main_session import MainAgentTurnResult
from chemgraph.cli.formatting import (
    _content_text, action_review_summary, build_action_review, console,
    format_action_review, format_response, list_models,
)
from chemgraph.models.endpoints.codex import SPEC as CODEX_SPEC


def test_list_models_displays_codex_subscription():
    with console.capture() as capture:
        list_models()
    row = next(line for line in capture.get().splitlines() if "codex:<model-id>" in line)
    assert "Codex / ChatGPT" in row
    assert "Subscription" in row
    assert CODEX_SPEC.model_type == "Subscription"
    assert "Experimental" not in row


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
        ["Proposed replacement snippet", "-old", "+new", '"replace_all": true',
         "Applies to all occurrences"],
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


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("name,args", [
    ("execute", {"command": "danger\x1b[2K\x1b[1A\rharmless\x9b2K"}),
    ("write_file", {"file_path": "/a\x9b2K", "content": "danger\x1b]0;title\x07\rharmless"}),
    ("edit_file", {"old_string": "old\r\n", "new_string": "new\x1b[2K\x9b2K\n"}),
    ("custom\x1b[2K", {"path": "a\x9b2K\x7f"}),
    ("execute", {"command": " ".join(chr(n) for n in range(160) if n < 32 or n >= 127)}),
])
def test_action_reviews_escape_terminal_controls(monkeypatch, name, args, full):
    from copy import deepcopy

    action = {"name": name, "args": args}
    original = deepcopy(action)
    monkeypatch.setattr("builtins.open", lambda *_, **__: pytest.fail("no file reads"))
    terminal = Console(width=120, color_system=None)
    with terminal.capture() as capture:
        panel, _ = build_action_review(action, 1, 1, full=full)
        terminal.print(panel)
        terminal.print(action_review_summary(action))
    output = capture.get()
    assert not any(ord(c) < 32 and c not in "\n\t" or 127 <= ord(c) <= 159 for c in output)
    assert "\\x1b" in output
    if name == "edit_file":
        assert "-old\\x0d" in output  # Preserve CR before splitting diff lines.
    assert action == original


# Pytest stores IDs in PYTEST_CURRENT_TEST, which has a size limit on Windows.
@pytest.mark.parametrize("content", [
    "\n".join(f"line-{n:04d}" for n in range(5000)),
    "first " + "x" * 10000 + " hidden-middle " + "y" * 10000 + " last",
], ids=["many-lines", "single-long-line"])
def test_large_action_preview_is_bounded_and_full_view_keeps_everything(content):
    action = {"name": "write_file", "args": {"file_path": "/large.txt", "content": content}}
    terminal = Console(width=120, color_system=None)
    panel, truncated = build_action_review(action, 1, 1)
    with terminal.capture() as capture:
        terminal.print(panel)
    output = capture.get()
    assert truncated
    assert "characters omitted" in output
    assert "Type v" in output
    assert "line-2500" not in output and "hidden-middle" not in output
    assert content[:5] in output and content[-5:] in output
    assert len(output.splitlines()) < 120  # Also bounds wrapped single-line content.
    full_panel, truncated = build_action_review(action, 1, 1, full=True)
    with terminal.capture() as capture:
        terminal.print(full_panel)
    assert not truncated
    assert ("line-2500" if "line-2500" in content else "hidden-middle") in capture.get()


def test_full_edit_review_includes_context_omitted_from_compact_preview():
    context = "unchanged context\n" * 50 + "hidden-context\n" + "unchanged context\n" * 50
    action = {"name": "edit_file", "args": {
        "old_string": context + "before\n", "new_string": context + "after\n",
    }}
    terminal = Console(width=120, color_system=None)
    panel, truncated = build_action_review(action, 1, 1)
    with terminal.capture() as capture:
        terminal.print(panel)
    assert truncated and "hidden-context" not in capture.get()
    panel, _ = build_action_review(action, 1, 1, full=True)
    with terminal.capture() as capture:
        terminal.print(panel)
    output = capture.get()
    assert '"old_string"' in output and '"new_string"' in output
    assert "hidden-context" in output
    assert "Applies to all occurrences" not in output


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
