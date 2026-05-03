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

from chemgraph.models.supported_models import all_supported_models

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

    # Categorize models by provider
    model_info = {
        "openai": {"provider": "OpenAI", "type": "Cloud"},
        "gpt": {"provider": "OpenAI", "type": "Cloud"},
        "claude": {"provider": "Anthropic", "type": "Cloud"},
        "gemini": {"provider": "Google", "type": "Cloud"},
        "llama": {"provider": "Meta", "type": "Local/Cloud"},
        "qwen": {"provider": "Alibaba", "type": "Local/Cloud"},
        "ollama": {"provider": "Ollama", "type": "Local"},
        "groq": {"provider": "GROQ", "type": "Cloud"},
        "argo:": {"provider": "Argo (ANL)", "type": "Cloud"},
    }

    for model in all_supported_models:
        provider = "Unknown"
        model_type = "Unknown"

        for key, info in model_info.items():
            if key.lower() in model.lower():
                provider = info["provider"]
                model_type = info["type"]
                break

        table.add_row(model, provider, model_type)

    console.print(table)
    console.print(
        f"\n[bold green]Total models available: {len(all_supported_models)}[/bold green]"
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

    api_keys = [
        {
            "provider": "OpenAI",
            "env_var": "OPENAI_API_KEY",
            "examples": "gpt-4o, gpt-4o-mini, o1",
        },
        {
            "provider": "Anthropic",
            "env_var": "ANTHROPIC_API_KEY",
            "examples": "claude-3-5-sonnet, claude-3-opus",
        },
        {
            "provider": "Google",
            "env_var": "GEMINI_API_KEY",
            "examples": "gemini-pro, gemini-2.5-pro",
        },
        {
            "provider": "GROQ",
            "env_var": "GROQ_API_KEY",
            "examples": "groq:llama-3.3-70b-versatile",
        },
        {
            "provider": "ALCF",
            "env_var": "ALCF_ACCESS_TOKEN",
            "examples": "Llama-3.1-405B, Qwen3-32B",
        },
        {
            "provider": "Local/Ollama",
            "env_var": "Not Required",
            "examples": "llama3.2, qwen2.5",
        },
    ]

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


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def _is_atomic_json(content: str) -> bool:
    """Return True if *content* is a JSON string with atomic-structure keys.

    This replaces the old fragile substring check (Bug 10) with a
    proper parse attempt.
    """
    try:
        data = json.loads(content.strip())
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    atomic_keys = {"numbers", "positions", "cell", "pbc", "atomic_numbers"}
    return bool(atomic_keys & data.keys())


def format_response(result: Any, verbose: bool = False) -> None:
    """Format the agent response for display."""
    if not result:
        console.print("[red]No response received from agent.[/red]")
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
            if message.type == "ai" and message.content.strip():
                content = message.content.strip()
                if not _is_atomic_json(content):
                    final_answer = content
                    break
        elif isinstance(message, dict):
            if message.get("type") == "ai" and message.get("content", "").strip():
                content = message["content"].strip()
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
            content = message.content
        elif isinstance(message, dict):
            content = message.get("content", "")

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
