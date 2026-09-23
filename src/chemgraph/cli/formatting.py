"""Rich-based display helpers for the ChemGraph CLI.

This module handles all terminal rendering: banners, tables,
response formatting, and API-key status display.
"""

from __future__ import annotations

import json
import os
import re
from difflib import unified_diff
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from chemgraph.models.endpoints.registry import CATALOG_ENDPOINTS, catalog_entries

# Shared console instance for the CLI package.
console = Console()


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
    action: dict, index: int, total: int, *, full: bool = False,
) -> tuple[Panel, bool]:
    """Build a sanitized preview and report omissions, using only supplied args."""
    name = str(action.get("name", "unknown"))
    args = action.get("args", {})
    label = ""
    preview = []
    if isinstance(args, dict):
        args = dict(args)
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
        details.append(Text("Preview shortened. Type v to inspect the full action."))
    return Panel(
        Group(*details),
        title=Text(f"Review action {index} of {total}"),
        border_style="yellow",
    ), truncated


def format_action_review(action: dict, index: int, total: int) -> Panel:
    """Preview an action using only its arguments, without reading host files."""
    return build_action_review(action, index, total)[0]


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

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


def format_response(result: Any, verbose: bool = False) -> None:
    """Format the agent response for display.

    Parameters
    ----------
    result : Any
        Agent result, message list, state dictionary, or message object.
    verbose : bool, optional
        Whether to include raw message details.
    """
    if not result:
        console.print("[red]No response received from agent.[/red]")
        return

    if hasattr(result, "assistant_response"):
        response = str(getattr(result, "assistant_response", "")).strip()
        if response:
            console.print(
                Panel(
                    Markdown(response),
                    title="ChemGraph Response",
                    style="green",
                    padding=(1, 2),
                )
            )
        elif verbose:
            console.print("[dim]Main agent returned no assistant text.[/dim]")
        return

    # Extract messages from result
    messages: list[Any] = []
    if isinstance(result, list):
        messages = result
    elif isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
    else:
        messages = [result]

    # Find the final AI response
    final_answer = ""
    for message in reversed(messages):
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
        console.print(
            Panel(
                Markdown(final_answer),
                title="ChemGraph Response",
                style="green",
                padding=(1, 2),
            )
        )

    # Check for structure data (valid JSON with atomic keys)
    for message in messages:
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
