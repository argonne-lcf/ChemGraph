import json

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

from chemgraph.agent.main_session import MainAgentTurnResult
from chemgraph.cli.formatting import (
    StreamMessageRenderer, ToolEventRenderer, _content_text, action_review_summary, build_action_review, console,
    format_action_review, format_response, list_models, show_tool_output_log, summarize_tool_args,
    summarize_tool_result, tool_output_log,
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
    ("execute", {"command": "ls", "reason": "why\x1b[2J\x9b2K"}),
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


def test_tool_reason_is_shown_once_and_escaped_in_review_and_event_line():
    from chemgraph.cli.commands import make_tool_event_renderer

    terminal = Console(width=120, color_system=None)
    with terminal.capture() as capture:
        terminal.print(format_action_review(
            {"name": "execute", "args": {"command": "ls", "reason": "List the run directory."}}, 1, 1,
        ))
    output = capture.get()
    assert "Reason: List the run directory." in output
    assert '"reason"' not in output

    with console.capture() as capture:
        make_tool_event_renderer()("tool_call_started", {
            "tool_name": "execute", "arguments": "{'command': 'ls'}", "reason": "[red]why[/red]\x1b[2J",
        })
    output = capture.get()
    assert "↳ [red]why[/red]\\x1b[2J" in output
    assert "\x1b[2J" not in output


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
    assert "\n" + "─" * 10 in output and "ChemGraph Response" not in output
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
    assert "\n" + "─" * 10 in output and "ChemGraph Response" not in output
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
    assert "\n" + "─" * 10 in output and "ChemGraph Response" not in output
    assert "Delegated calculation complete" in output



@pytest.mark.parametrize("name,content,kwargs,expected", [
    ("read_file", "\n".join(f"{n:>4}  x" for n in range(1, 143)), {}, ("Read 142 lines", False)),
    ("execute", "out\n[Command failed with exit code 3]\n[Output was truncated due to size limits]", {},
     ("Exit 3 · out", True)),
    # A status line inside the command's own output is not its exit status.
    ("execute", "[Command failed with exit code 1]\nok", {}, ("[Command failed with exit code 1]\nok", False)),
    ("run_ase", "\n".join(f"line {n}" for n in range(30)), {}, ("line 0\nline 1\n… +28 lines", False)),
])
def test_tool_result_summary_is_bounded(name, content, kwargs, expected):
    assert summarize_tool_result(name, content, **kwargs) == expected


def test_tool_args_summary_is_one_bounded_escaped_line():
    summary = summarize_tool_args("write_file", str({"content": "x\x1b[2J" * 5000, "reason": "why"}))
    assert len(summary) <= 100 and "\\x1b" in summary and "\x1b" not in summary and "why" not in summary


def test_parallel_call_labels_do_not_require_call_ids():
    renderer = StreamMessageRenderer(console)
    with console.capture() as capture:
        renderer.render([AIMessage("", tool_calls=[
            {"name": "ls", "args": {"path": path}, "id": None} for path in ("/a", "/b")
        ])])
    assert "ls(/a) #1/2" in capture.get() and "ls(/b) #2/2" in capture.get()


def test_held_answer_replaced_by_structured_result_keeps_answer_rule():
    renderer = StreamMessageRenderer(console)
    with console.capture() as capture:
        renderer.render([AIMessage("Water optimized", id="a")])
        renderer.hand_off(AIMessage('{"energy": -14.2}', id="s"))
    assert "─\nWater optimized" in capture.get()


def test_event_renderer_summarizes_callback_payloads():
    import io

    from langchain_core.tools import ToolException, tool

    from chemgraph.agent.events import _AstreamEventCallback

    @tool
    def read_file(file_path: str) -> str:
        """Read a file."""
        return "\n".join(f"{n:>4}  body-{n}" for n in range(1, 301))

    @tool
    def broken(value: int) -> str:
        """Fail."""
        raise ToolException("bad value")

    broken.handle_tool_error = True
    output = io.StringIO()
    render = ToolEventRenderer(Console(file=output, width=80, color_system=None))
    config = {"callbacks": [_AstreamEventCallback(render, "t")]}
    read_file.invoke({"name": "read_file", "args": {"file_path": "/S.md"}, "id": "1", "type": "tool_call"}, config)
    broken.invoke({"name": "broken", "args": {"value": 1}, "id": "2", "type": "tool_call"}, config)
    render("tool_call_started", {"tool_name": "ask_human", "arguments": "{}", "call_id": "q"})
    render("tool_call_failed", {"error": "GraphInterrupt((Interrupt(value='?'),))", "call_id": "q"})
    assert "⎿ Read 300 lines" in output.getvalue() and "⎿ bad value" in output.getvalue()
    assert "Interrupt" not in output.getvalue()  # A pause for input is not a failure.


def test_expand_pages_the_full_output_of_the_last_turn(monkeypatch):
    from contextlib import nullcontext

    from langchain_core.messages import ToolMessage

    monkeypatch.setattr(console, "pager", lambda **_kwargs: nullcontext())
    body = "\n".join(f"{n:>4}  body-{n}" for n in range(1, 301))
    renderer = StreamMessageRenderer(console)
    tool_output_log.start_turn()
    with console.capture() as capture:
        renderer.render([
            AIMessage("", id="a", tool_calls=[{"name": "read_file", "args": {"file_path": "/S.md"}, "id": "r"}]),
            ToolMessage(body, tool_call_id="r", name="read_file", id="t"),
        ])
        show_tool_output_log()
    output = capture.get()
    assert "⎿ Read 300 lines" in output and "body-250" in output and '"file_path": "/S.md"' in output
