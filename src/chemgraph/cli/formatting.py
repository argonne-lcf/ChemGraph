"""Rich-based display helpers for the ChemGraph CLI.

This module handles all terminal rendering: banners, tables,
response formatting, and API-key status display.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import threading
from difflib import unified_diff
from typing import Any, Callable

from rich.align import Align
from rich.cells import cell_len, set_cell_size
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from chemgraph.models.endpoints.registry import CATALOG_ENDPOINTS, catalog_entries

# Shared console instance for the CLI package.
console = Console()
logger = logging.getLogger(__name__)


def format_token_usage(usage: dict) -> Text:
    """Render provider counters locally; never generate a model summary."""
    keys = ("input_tokens", "output_tokens", "total_tokens")
    history_unaccounted = usage.get("history_unaccounted", False)
    if all(usage.get(key) is None for key in keys):
        reasons = []
        if usage.get("call_count"):
            reasons.append("provider did not report usage")
        if history_unaccounted:
            reasons.append("historical usage was not recorded")
        suffix = " (" + "; ".join(reasons) + ")" if reasons else ""
        return Text("Tokens: unavailable" + suffix, style="dim")
    parts = [
        f"{usage[key]:,} {label}" if usage.get(key) is not None else f"unknown {label}"
        for key, label in zip(keys, ("input", "output", "total"), strict=True)
    ]
    prefix = "Tokens (partial): " if usage.get("partial") else "Tokens: "
    details = []
    for key, label in (("cached_input_tokens", "cached input"), ("reasoning_output_tokens", "reasoning output")):
        if usage.get(key) is not None and usage.get("call_count"):
            qualifier = "known " if usage.get("unreported_counts", {}).get(key) else ""
            details.append(f"{usage[key]:,} {qualifier}{label}")
    suffix = f" (included: {'; '.join(details)})" if details else ""
    if usage.get("incomplete_calls"):
        suffix += f"; incomplete usage for {usage['incomplete_calls']} call(s)"
    if history_unaccounted:
        suffix += "; historical usage was not recorded"
    return Text(prefix + " · ".join(parts) + suffix, style="dim")


def _safe_review_text(value: str) -> str:
    """Make terminal controls visible while retaining multiline previews."""
    return re.sub(
        r"[\x00-\x08\x0b-\x1f\x7f-\x9f]",
        lambda match: f"\\x{ord(match.group()):02x}",
        value,
    )


def _limit_review_text(
    value: str, max_lines: int = 40, max_chars: int = 8000,
) -> tuple[str, bool]:
    """Keep the head and tail within both source-line and character budgets."""
    lines = value.splitlines(keepends=True)
    if len(lines) <= max_lines and len(value) <= max_chars:
        return value, False
    head = "".join(lines[:max_lines // 2])[:max_chars // 2]
    tail = "".join(lines[-(max_lines // 2):])[-(max_chars // 2):]
    omitted = value[len(head):len(value) - len(tail)]
    note = f"… {len(omitted)} characters omitted ({omitted.count(chr(10))} line breaks) …"
    return f"{head}\n{note}\n{tail}", True


def action_review_summary(action: dict) -> Text:
    """Repeat a short, literal tool/path identity beside the decision prompt."""
    values = [f"Tool: {action.get('name', 'unknown')}"]
    args = action.get("args", {})
    if isinstance(args, dict) and "file_path" in args:
        values.append(f"Path: {args['file_path']}")
    text = _safe_review_text(" | ".join(values)).replace("\n", "\\n").replace("\t", "\\t")
    if len(text) > 240:
        text = text[:120] + " … " + text[-120:]
    return Text(text)


def build_action_review(
    action: dict, index: int, total: int, *, full: bool = False, viewable: bool = True,
) -> tuple[Panel, bool]:
    """Build a sanitized preview and report omissions, using only supplied args."""
    name = str(action.get("name", "unknown"))
    args = action.get("args", {})
    label = ""
    preview = []
    reason = None
    if isinstance(args, dict):
        args = dict(args)
        if isinstance(args.get("reason"), str) and args["reason"].strip():
            reason = args.pop("reason")
        if name == "execute" and isinstance(args.get("command"), str):
            label = "Command:"
            preview = [(args.pop("command"), "bash")]
        elif name == "write_file" and isinstance(args.get("content"), str):
            label = "Content:"
            preview = [(args.pop("content"), None)]
        elif name == "edit_file" and all(
            isinstance(args.get(key), str) for key in ("old_string", "new_string")
        ):
            before = _safe_review_text(args.pop("old_string")).splitlines(keepends=True)
            after = _safe_review_text(args.pop("new_string")).splitlines(keepends=True)
            lines = unified_diff(
                before, after, fromfile="before", tofile="after",
                n=max(len(before), len(after)),
            )
            diff = "".join(
                line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                for line in lines
            )
            label = "Proposed replacement snippet:"
            if args.get("replace_all") is True:
                label += " Applies to all occurrences."
            preview = [(diff or "(No changes)", "diff")]
    if full:
        # Include exact arguments as well as the human-readable replacement diff.
        args = action.get("args", {})
    try:
        arguments = (json.dumps(args, indent=2, ensure_ascii=False, default=str), "json")
    except (TypeError, ValueError):
        arguments = (repr(args), None)
    parts = [(f"Tool: {name}", None)]
    if reason is not None:
        parts.append((f"Reason: {reason}", None))
    if args or not preview:
        parts.append(arguments)
    if preview:
        parts.append((label, None))
        parts.extend(preview)
    details = []
    truncated = False
    lines_left, chars_left = 40, 8000
    for position, (value, lexer) in enumerate(parts):
        value = _safe_review_text(value)
        if not full:
            parts_left = len(parts) - position
            line_budget, char_budget = lines_left // parts_left, chars_left // parts_left
            lines_left -= min(len(value.splitlines()), line_budget)
            chars_left -= min(len(value), char_budget)
            value, omitted = _limit_review_text(value, line_budget, char_budget)
            truncated |= omitted
        details.append(Syntax(value, lexer, word_wrap=True) if lexer else Text(value))
    if truncated:
        details.append(Text(
            "Preview shortened. Type v to inspect the full action."
            if viewable else "Preview shortened."
        ))
    return Panel(
        Group(*details),
        title=Text(f"Review action {index} of {total}"),
        border_style="yellow",
    ), truncated


def format_action_review(action: dict, index: int, total: int) -> Panel:
    """Preview an action using only its arguments, without reading host files."""
    return build_action_review(action, index, total)[0]


# ---------------------------------------------------------------------------
# Compact tool activity
# ---------------------------------------------------------------------------

_PRIMARY_ARG_KEYS = ("file_path", "path", "command", "pattern", "description", "query")
# Only the status deepagents appends (optionally before its truncation note), not
# matching text inside the command's own output.
_EXIT_TRAILER = re.compile(
    r"\n?\[Command (?:succeeded|failed) with exit code (-?\d+)\]"
    r"(?=(?:\n\[Output was truncated[^\n]*\])?\s*\Z)"
)
_READ_ROW = re.compile(r"^\s*\d+(?:  |\t)")


def _one_line(value: str, limit: int) -> str:
    """Return a single escaped line of at most *limit* characters."""
    # Only a bounded prefix is displayed; line breaks and their indentation
    # collapse to one space (``/expand`` keeps the original layout).
    head = value[: limit * 4]
    text = re.sub(r"\s*\n\s*", " ", _safe_review_text(head)).replace("\t", " ").strip()
    if len(text) <= limit and len(head) == len(value):
        return text
    return text[: limit - 1] + "…"


def _parse_literal(value: Any) -> Any:
    """Decode small Python literals from callbacks; leave other values alone."""
    if isinstance(value, str) and len(value) <= 65536 and value.lstrip()[:1] in ("{", "["):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return value


def summarize_tool_args(name: str, args: Any, limit: int = 100) -> str:
    """Summarize tool arguments (a dict or its ``str``) on one bounded line."""
    args = _parse_literal(args)
    if not isinstance(args, dict):
        return _one_line(str(args), limit)
    args = {key: value for key, value in args.items() if key != "reason"}
    keys = ("pattern", *_PRIMARY_ARG_KEYS) if name in {"grep", "glob"} else _PRIMARY_ARG_KEYS
    for key in keys:
        if isinstance(args.get(key), str):
            return _one_line(args[key], limit)
    parts = []
    for key, value in args.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, default=str)
        parts.append(f"{key}={_one_line(value, min(40, limit))}")
    return _one_line(", ".join(parts), limit)


def tool_result_parts(value: Any) -> tuple[str, str | None, Any]:
    """Return ``(text, status, artifact)`` from a tool message or its serialized form."""
    if isinstance(value, dict) and isinstance(value.get("update"), dict):
        # A serialized Command (e.g. the ``task`` tool) carries its ToolMessage.
        messages = value["update"].get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[-1], dict):
            value = messages[-1]
    if isinstance(value, dict) and "content" in value:
        content, status, artifact = value["content"], value.get("status"), value.get("artifact")
    elif hasattr(value, "content") and hasattr(value, "type"):
        content = value.content
        status, artifact = getattr(value, "status", None), getattr(value, "artifact", None)
    else:
        content, status, artifact = value, None, None
    if isinstance(content, str):
        return content, status, artifact
    text = _content_text(content)
    if not text:
        try:
            text = json.dumps(content, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(content)
    return text, status, artifact


def _first_line(text: str) -> str:
    return next((line for line in text.splitlines() if line.strip()), "")


def _preview(text: str) -> str:
    """Show short output as-is; otherwise its first lines and how much is hidden."""
    text = text.strip()
    if not text:
        return "(no output)"
    lines = text.splitlines()
    if len(lines) <= 2 and len(text) <= 160:
        return "\n".join(_one_line(line, 160) for line in lines)
    shown = [_one_line(line, 120) for line in lines[:2]]
    shown.append(f"… +{len(lines) - 2} lines" if len(lines) > 2 else f"… {len(text):,} chars")
    return "\n".join(shown)


def _summarize_result(name: str, text: str, status: str | None, artifact: Any) -> tuple[str, bool]:
    stripped = text.strip()
    if status == "error":
        return _one_line(_first_line(stripped) or "Error", 160), True
    if name == "execute":
        code = artifact.get("exit_code") if isinstance(artifact, dict) else None
        trailer = _EXIT_TRAILER.search(text)
        if not isinstance(code, int) and trailer:
            code = int(trailer.group(1))
        body = (text[: trailer.start()] + text[trailer.end():] if trailer else text).strip()
        if isinstance(code, int) and code != 0:
            first = _first_line(body)
            return f"Exit {code}" + (f" · {_one_line(first, 120)}" if first else ""), True
        return _preview(body), False
    if name == "read_file":
        if stripped.startswith("Error"):
            return _one_line(_first_line(stripped), 160), True
        rows = sum(1 for line in text.splitlines() if _READ_ROW.match(line))
        if rows or not stripped:
            return f"Read {rows} line{'' if rows == 1 else 's'}", False
    elif name in {"ls", "glob"}:
        if stripped == "No files found":
            return "No files", False
        listing = stripped.split("\n\n", 1)[0]
        paths = _parse_literal(listing)
        if isinstance(paths, list):
            more = " (truncated)" if listing != stripped else ""
            return f"{len(paths)} path{'' if len(paths) == 1 else 's'}{more}", False
    elif name == "grep" and stripped == "No matches found":
        return "No matches", False
    return _preview(text), False


def summarize_tool_result(
    name: str, content: Any, *, status: str | None = None, artifact: Any = None,
) -> tuple[str, bool]:
    """Return a ≤3-line summary of a tool result and whether it failed."""
    text = content if isinstance(content, str) else tool_result_parts(content)[0]
    try:
        return _summarize_result(name, text, status, artifact)
    except Exception:  # noqa: BLE001 - display must never break a run.
        logger.debug("Could not summarize %s output.", name, exc_info=True)
        return _preview(str(text)), status == "error"


def tool_call_line(
    name: str, args: Any, *, subagent: str | None = None, label: str = "",
    reason: Any = None, width: int | None = None,
) -> Text:
    """``⏺ [subagent · ]name(args) [label]`` plus an optional reason line.

    The argument budget leaves room for the label so truncation never hides it.
    """
    line = Text("⏺ ", style="cyan")
    if subagent:
        line.append(_one_line(str(subagent), 40) + " · ", style="cyan")
    line.append(_one_line(str(name), 60), style="bold")
    tail = f" {label}" if label else ""
    limit = 100 if width is None else max(12, min(100, width - line.cell_len - cell_len(tail) - 2))
    summary = summarize_tool_args(name, args, limit)
    if cell_len(summary) > limit:  # Wide characters take two terminal cells.
        summary = set_cell_size(summary, limit - 1) + "…"
    line.append(f"({summary})")
    if tail:
        line.append(tail, style="dim")
    if reason:
        line.append("\n")
        line.append_text(reason_text(reason))
    return line


def tool_result_text(summary: str, *, failed: bool = False, prefix: str = "") -> Text:
    """Indent a result summary under its call with ``⎿``."""
    lines = (prefix + summary).splitlines() or [""]
    body = "\n".join(["  ⎿ " + lines[0], *("    " + line for line in lines[1:])])
    return Text(body, style="red" if failed else "dim")


def reason_text(reason: Any) -> Text:
    return Text("  ↳ " + _one_line(str(reason), 200), style="dim")


def _field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _message_identity(message: Any) -> tuple:
    if message_id := _field(message, "id"):
        return ("id", message_id)
    return (
        "content", _field(message, "type"), _content_text(_field(message, "content")),
        _field(message, "tool_call_id"),
    )


class ToolOutputLog:
    """Full tool calls and results of the latest REPL turn, for ``/expand``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[tuple[str, Any, str]] = []

    def start_turn(self) -> None:
        with self._lock:
            self._entries.clear()

    def record(self, title: str, args: Any, text: str) -> None:
        with self._lock:
            self._entries.append((title, args, text))

    def entries(self) -> list[tuple[str, Any, str]]:
        with self._lock:
            return list(self._entries)


tool_output_log = ToolOutputLog()


def show_tool_output_log(log: ToolOutputLog = tool_output_log) -> None:
    """Page the full, unshortened tool activity of the latest turn."""
    entries = log.entries()
    if not entries:
        console.print("[dim]No tool output in the last turn.[/dim]")
        return
    with console.pager(styles=False, links=False):
        for title, args, text in entries:
            args = _parse_literal(args)
            if not isinstance(args, str):
                args = json.dumps(args, indent=2, ensure_ascii=False, default=str)
            console.print(Text(f"⏺ {title}", style="bold"))
            console.print(Text("Arguments:\n" + _safe_review_text(args)))
            console.print(Text("Result:\n" + (_safe_review_text(text) or "(no output)")))
            console.print()


class StreamMessageRenderer:
    """Print new graph messages as compact tool activity.

    ``begin_turn`` seeds history from LangGraph's initial input-state snapshot.
    Assistant prose and graph-generated human messages are shown in full. The
    latest text-only AI message is held until more output follows, so a final
    answer reaches :func:`format_response` (see :meth:`hand_off`) and atomic
    JSON is left to its structure panel. With ``verbose``, new messages are
    pretty-printed instead.
    """

    def __init__(self, output: Console | None = None, *, verbose: bool = False) -> None:
        self._console = output or console
        self._verbose = verbose
        self._seen: set[Any] = set()
        self._handled: set[tuple] = set()
        self._initial = False
        self._answer: Any = None
        self._labels: dict[str, tuple[str, Any]] = {}

    def _print(self, text: Text) -> None:
        self._console.print(text, no_wrap=True, overflow="ellipsis")

    def begin_turn(self) -> None:
        self._seen.clear()
        self._handled.clear()
        self._labels.clear()
        self._answer = None
        self._initial = True

    def finish(self) -> None:
        """Show a held answer the caller did not take, e.g. after a failure."""
        self.flush()
        self._labels.clear()

    def flush(self) -> None:
        """Print the held text-only AI message before later output or a prompt."""
        message, self._answer = self._answer, None
        if message is None:
            return
        self._handled.add(_message_identity(message))
        if self._verbose and hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            text = _content_text(_field(message, "content")).strip()
            self._console.print(Markdown(_safe_review_text(text)))

    def hand_off(self, result: Any) -> None:
        """Leave the held answer to :func:`format_response` if *result* has it."""
        if self._answer is not None and _message_identity(self._answer) in {
            _message_identity(message) for message in _result_messages(result)
        }:
            self._answer = None
        self.flush()

    def excludes(self, message: Any) -> bool:
        """Exclude old history and already displayed messages from final output."""
        return _message_identity(message) in self._handled

    def render(self, messages: Any) -> None:
        if not messages:
            return
        occurrences: dict[tuple, int] = {}
        try:
            for index, message in enumerate(messages):
                identity = _message_identity(message)
                occurrences[identity] = occurrences.get(identity, 0) + 1
                # Occurrences distinguish repeated ID-less messages without tying
                # them to positions that can change when history is compacted.
                key = (identity, occurrences[identity])
                if key in self._seen:
                    continue
                self._seen.add(key)
                labels = self._record(message, record_output=not self._initial)
                if self._initial:
                    # The last input message is this turn's query. Never replay
                    # existing history, even when verbose or entering a saved thread.
                    self._handled.add(identity)
                    if self._verbose and index == len(messages) - 1 and _field(message, "type") == "human":
                        message.pretty_print()
                    continue
                text = _content_text(_field(message, "content")).strip()
                structure = _is_atomic_json(text)
                calls = _field(message, "tool_calls")
                if _field(message, "type") == "ai" and text and not structure and not calls:
                    self.flush()
                    self._answer = message
                    continue
                if not structure:
                    self._handled.add(identity)
                elif _field(message, "type") in {"human", "ai"} and not calls and not self._verbose:
                    continue  # Left to format_response's structure panel.
                self.flush()
                if self._verbose:
                    message.pretty_print()
                else:
                    self._render_one(message, labels)
        finally:
            self._initial = False

    def _record(self, message: Any, *, record_output: bool = True) -> list[str]:
        """Keep full tool activity for ``/expand`` and return its display labels."""
        kind = _field(message, "type")
        if kind == "ai":
            calls = [call for call in _field(message, "tool_calls") or [] if isinstance(call, dict)]
            labels = []
            for index, call in enumerate(calls, start=1):
                label = f"#{index}/{len(calls)}" if len(calls) > 1 else ""
                labels.append(label)
                self._labels[str(call.get("id"))] = (label, call.get("args"))
            return labels
        elif kind == "tool":
            name = str(_field(message, "name") or "tool")
            label, args = self._labels.pop(str(_field(message, "tool_call_id")), ("", None))
            if record_output:
                tool_output_log.record(f"{name} {label}".strip(), args, tool_result_parts(message)[0])
            return [label]
        return []

    def _render_one(self, message: Any, labels: list[str]) -> None:
        kind = _field(message, "type")
        if kind == "human":
            text = _content_text(_field(message, "content")).strip()
            if text:
                self._console.print(Text(_safe_review_text(text)))
            return
        if kind not in {"ai", "tool"}:
            return
        if kind == "tool":
            label = labels[0]
            name = str(_field(message, "name") or "tool")
            text, status, artifact = tool_result_parts(message)
            summary, failed = summarize_tool_result(name, text, status=status, artifact=artifact)
            self._print(tool_result_text(
                summary, failed=failed, prefix=f"{name} {label} · " if label else "",
            ))
            return
        text = _content_text(_field(message, "content")).strip()
        calls = [call for call in _field(message, "tool_calls") or [] if isinstance(call, dict)]
        if text and not _is_atomic_json(text):
            self._console.print(Markdown(_safe_review_text(text)))
        for call, label in zip(calls, labels, strict=True):
            name = str(call.get("name") or "tool")
            args = call.get("args")
            reason = args.get("reason") if isinstance(args, dict) else None
            self._print(tool_call_line(
                name, args, label=label, reason=reason, width=self._console.width,
            ))


class ToolEventRenderer:
    """Render tool callback events (main_agent and delegated workers) compactly.

    LangChain may deliver these callbacks from executor threads, so bookkeeping
    and output are serialized. Calls that overlap with another displayed call
    get a unique ``#n`` label on their start and result lines.
    """

    def __init__(
        self, output: Console | None = None, *, verbose: bool = False, subagents_only: bool = False,
    ) -> None:
        self._console = output or console
        self._verbose = verbose
        self._subagents_only = subagents_only
        self._lock = threading.Lock()
        self._batched: set[str] = set()
        self._calls: dict[str, dict[str, Any]] = {}
        self._sequence = 0
        self._closed = False

    def finish(self) -> None:
        """Release outstanding calls and ignore late callbacks after this turn."""
        with self._lock:
            self._closed = True
            self._calls.clear()
            self._batched.clear()

    def _print(self, text: Any) -> None:
        self._console.print(text, no_wrap=True, overflow="ellipsis")

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            if event == "llm_decision":
                calls = [call for call in payload.get("tool_calls") or [] if call.get("id")]
                if len(calls) > 1:
                    self._batched.update(str(call["id"]) for call in calls)
            elif event == "tool_call_started":
                self._started(payload)
            elif event in {"tool_call_finished", "tool_call_failed"}:
                self._ended(event, payload)

    @staticmethod
    def _concurrent(scope: str | None, other: str | None) -> bool:
        # A direct call that is outstanding while a worker runs is its parent
        # delegation, not a parallel sibling.
        return scope == other or (scope is not None and other is not None)

    def _started(self, payload: dict[str, Any]) -> None:
        subagent = payload.get("subagent_name")
        if self._subagents_only and not subagent:
            return
        name = str(payload.get("tool_name") or "unknown")
        arguments = payload.get("arguments", "")
        reason = payload.get("reason")
        self._sequence += 1
        batched = str(payload.get("tool_call_id")) in self._batched
        self._batched.discard(str(payload.get("tool_call_id")))
        overlapping = False
        for other in self._calls.values():
            if self._concurrent(subagent, other["scope"]):
                other["overlapped"] = overlapping = True
        label = f"#{self._sequence}" if batched or overlapping else ""
        call = {"name": name, "scope": subagent, "label": label, "args": arguments,
                "overlapped": overlapping}
        if payload.get("call_id"):
            self._calls[str(payload["call_id"])] = call
        if self._verbose:
            # Full arguments, as before compact rendering.
            self._console.print(
                (f"[dim]Subagent[/dim] [bold cyan]{escape(str(subagent))}[/bold cyan] "
                 if subagent else "[bold cyan]main_agent[/bold cyan] ")
                + f"[dim]→[/dim] [bold]{escape(name)}[/bold]({escape(str(arguments))})"
                + (f" [dim]{label}[/dim]" if label else "")
                + (f"\n[dim]  ↳ {escape(_safe_review_text(str(reason)))}[/dim]" if reason else "")
            )
            return
        self._print(tool_call_line(
            name, arguments, subagent=subagent, label=label, reason=reason,
            width=self._console.width,
        ))

    def _ended(self, event: str, payload: dict[str, Any]) -> None:
        call = self._calls.pop(str(payload.get("call_id")), None)
        if call is None:
            if self._subagents_only:
                return
            call = {"name": str(payload.get("tool_name") or "unknown"), "scope": None, "label": "", "args": ""}
            ambiguous = False
        else:
            ambiguous = call["overlapped"]
        if event == "tool_call_failed" and str(payload.get("error", "")).startswith(
            ("GraphInterrupt(", "NodeInterrupt("),
        ):
            return  # A clarification or nested approval pause, not a failure.
        name = call["name"]
        if event == "tool_call_failed":
            text, status, artifact = str(payload.get("error", "")), "error", None
        else:
            text, status, artifact = tool_result_parts(payload.get("result"))
        tool_output_log.record(
            " ".join(filter(None, [f"{call['scope']} ·" if call["scope"] else "", name, call["label"]])),
            call["args"], text,
        )
        prefix = ""
        if call["label"] or ambiguous:
            worker = f"{_one_line(str(call['scope']), 40)} · " if call["scope"] else ""
            identity = call["label"] or f"({summarize_tool_args(name, call['args'], 40)})"
            prefix = f"{worker}{name} {identity} · " if call["label"] else f"{worker}{name}{identity} · "
        if self._verbose:
            self._console.print(tool_result_text(
                _safe_review_text(text) or "(no output)", failed=status == "error", prefix=prefix,
            ))
            return
        summary, failed = summarize_tool_result(name, text, status=status, artifact=artifact)
        self._print(tool_result_text(summary, failed=failed, prefix=prefix))


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def create_session_header(
    model: str, workflow: str, approval_mode: str, *,
    workspace: str | None = None, auto: bool = False,
) -> Panel:
    """Show the effective configuration after initialization or restore."""
    text = Text(f"model {model} · workflow {workflow} · approvals {approval_mode}")
    if auto:
        text.append(" [auto]", style="yellow")
    if workspace:
        text.append(f"\nworkspace {workspace}", style="dim")
    text.append("\n/help commands · /expand tool output · ↑/↓ history · /quit exit", style="dim")
    return Panel(text, title="ChemGraph", border_style="blue", expand=False)


def create_interactive_help() -> Group:
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    for command, description in (
        ("/help", "Show commands"),
        ("/quit", "Exit interactive mode"),
        ("/clear", "Clear the screen"),
        ("/config", "Show current configuration"),
        ("/expand", "View the last turn's full tool calls and results"),
        ("/model <name>", "Change model"),
        ("/workflow <type>", "Change workflow"),
        ("/auto [on|off]", "Toggle automatic tool approvals for this session"),
        ("/history", "List recent sessions"),
        ("/show <id>", "Show a session's conversation"),
        ("/resume <id>", "Resume a previous session"),
        ("/retry", "Retry a failed main_agent operation"),
    ):
        table.add_row(Text(command), description)
    return Group(
        table,
        Text("↑/↓ recalls inputs. Exact bare aliases work without arguments; otherwise use /commands.", style="dim"),
        Text("Reviews: Enter/y approves; n rejects; 3/auto enables auto mode; type instructions to revise.", style="dim"),
        Text("main_agent /resume restores durable threads. Agent questions always require an answer.", style="dim"),
    )


def create_banner() -> Panel:
    """Create a welcome banner for ChemGraph CLI."""
    banner_text = """

    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                           ChemGraph                           ║
    ║             AI Agents for Computational Chemistry             ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    return Panel(Align.center(banner_text), style="bold blue", padding=(1, 2))


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

def list_models() -> None:
    """Display available models in a formatted table."""
    console.print(Panel("Available Models", style="bold cyan"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model Name", style="cyan", width=40)
    table.add_column("Provider", style="green")
    table.add_column("Type", style="yellow")

    entries = catalog_entries()
    for model, spec in entries:
        table.add_row(
            model,
            spec.display_name or spec.name,
            spec.model_type,
        )

    table.add_row(
        "codex:<model-id>",
        "Codex / ChatGPT",
        "Subscription",
    )

    console.print(table)
    console.print(
        f"\n[bold green]Curated models available: {len(entries)}[/bold green]"
    )


# ---------------------------------------------------------------------------
# API-key status
# ---------------------------------------------------------------------------

def check_api_keys_status() -> None:
    """Display API key availability status."""
    console.print(Panel("API Key Status", style="bold cyan"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan", width=15)
    table.add_column("Environment Variable", style="yellow", width=25)
    table.add_column("Status", style="white", width=15)
    table.add_column("Example Models", style="dim", width=30)

    api_keys = []
    seen: set[tuple[str, str | None]] = set()
    for spec in CATALOG_ENDPOINTS:
        policy = spec.credential
        identity = (spec.config_section or spec.name, policy.env_var)
        if identity in seen:
            continue
        seen.add(identity)
        api_keys.append(
            {
                "provider": spec.display_name or spec.name,
                "env_var": policy.env_var or "Not Required",
                "examples": ", ".join(spec.curated_models[:2]) or "Prefix-routed",
            }
        )

    for key_info in api_keys:
        if key_info["env_var"] == "Not Required":
            status = "[green]Available[/green]"
        else:
            is_set = bool(os.getenv(key_info["env_var"]))
            status = "[green]Set[/green]" if is_set else "[red]Missing[/red]"

        table.add_row(
            key_info["provider"], key_info["env_var"], status, key_info["examples"]
        )

    console.print(table)

    console.print("\n[bold]How to set API keys:[/bold]")
    console.print("  [cyan]Bash/Zsh:[/cyan] export OPENAI_API_KEY='your_key_here'")
    console.print("  [cyan]Fish:[/cyan] set -x OPENAI_API_KEY 'your_key_here'")
    console.print(
        "  [cyan].env file:[/cyan] Add OPENAI_API_KEY=your_key_here to a .env file"
    )

    console.print("\n[bold]Get API keys:[/bold]")
    console.print("  [cyan]OpenAI:[/cyan] https://platform.openai.com/api-keys")
    console.print("  [cyan]Anthropic:[/cyan] https://console.anthropic.com/")
    console.print("  [cyan]Google:[/cyan] https://aistudio.google.com/apikey")
    console.print("  [cyan]OpenRouter:[/cyan] https://openrouter.ai/keys")


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------


def _content_text(content: Any) -> str:
    """Return display text from string or structured message content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            text_parts.append(block["text"])
    return "".join(text_parts)


def _is_atomic_json(content: Any) -> bool:
    """Return True if *content* is a JSON string with atomic-structure keys.

    This replaces the old fragile substring check (Bug 10) with a
    proper parse attempt.

    Parameters
    ----------
    content : Any
        Candidate string or structured message content.

    Returns
    -------
    bool
        ``True`` when the parsed object contains atomic-structure keys.
    """
    content = _content_text(content)
    if not content:
        return False
    try:
        data = json.loads(content.strip())
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    atomic_keys = {"numbers", "positions", "cell", "pbc", "atomic_numbers"}
    return bool(atomic_keys & data.keys())


def _result_messages(result: Any) -> list[Any]:
    """Return the messages of an agent result, message list, or single message."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "messages" in result:
        return result["messages"]
    return [result]


def print_answer(text: str, *, output: Console | None = None) -> None:
    """Render prose consistently, including structured last-message runs."""
    destination = output or console
    answer = Table.grid(padding=(0, 1), expand=True)
    answer.add_column(width=1, no_wrap=True)
    answer.add_column(ratio=1)
    answer.add_row(Text("●", style="green"), Markdown(text))
    destination.print()
    destination.print(answer)


def format_response(
    result: Any, verbose: bool = False, *, exclude: Callable[[Any], bool] | None = None,
) -> None:
    """Format the agent response for display.

    Parameters
    ----------
    result : Any
        Agent result, message list, state dictionary, or message object.
    verbose : bool, optional
        Whether to include raw message details.
    exclude : callable, optional
        Skip messages already handled by the CLI's streamed display.
    """
    if not result:
        console.print("[red]No response received from agent.[/red]")
        return

    if hasattr(result, "assistant_response"):
        response = str(getattr(result, "assistant_response", "")).strip()
        if response:
            print_answer(response)
        elif verbose:
            console.print("[dim]Main agent returned no assistant text.[/dim]")
        return

    messages = _result_messages(result)

    # Find the final AI response
    final_answer = ""
    for message in reversed(messages):
        if exclude is not None and exclude(message):
            continue
        if hasattr(message, "content") and hasattr(message, "type"):
            content = _content_text(message.content).strip()
            if message.type == "ai" and content:
                if not _is_atomic_json(content):
                    final_answer = content
                    break
        elif isinstance(message, dict):
            content = _content_text(message.get("content", "")).strip()
            if message.get("type") == "ai" and content:
                if not _is_atomic_json(content):
                    final_answer = content
                    break

    if final_answer:
        print_answer(final_answer)

    # Check for structure data (valid JSON with atomic keys)
    for message in messages:
        if exclude is not None and exclude(message):
            continue
        content = ""
        if hasattr(message, "content"):
            content = _content_text(message.content).strip()
        elif isinstance(message, dict):
            content = _content_text(message.get("content", "")).strip()

        if content and _is_atomic_json(content):
            console.print(
                Panel(
                    Syntax(content, "json", theme="monokai"),
                    title="Molecular Structure Data",
                    style="cyan",
                )
            )

    # Verbose output
    if verbose:
        console.print(
            Panel(
                f"Messages: {len(messages)}", title="Debug Information", style="dim"
            )
        )
