"""Rich-based display helpers for the ChemGraph CLI.

This module handles all terminal rendering: banners, tables,
response formatting, and API-key status display.
"""

from __future__ import annotations

import json
import os
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from chemgraph.models.endpoints.registry import CATALOG_ENDPOINTS, catalog_entries

# Shared console instance for the CLI package.
console = Console()


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
        "Experimental",
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
        identity = (spec.display_name or spec.name, policy.env_var)
        if identity in seen:
            continue
        seen.add(identity)
        api_keys.append(
            {
                "provider": identity[0],
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
