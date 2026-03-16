#!/usr/bin/env python3
"""
ChemGraph Command Line Interface

A command-line interface for ChemGraph that provides computational chemistry
capabilities through natural language queries powered by AI agents.
"""

import argparse
import toml
import sys
import time
import os
import signal
import threading
import asyncio
import platform
from typing import Dict, Any
from contextlib import contextmanager

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.align import Align

# ChemGraph imports
from chemgraph.models.supported_models import all_supported_models
from chemgraph.utils.config_utils import (
    flatten_config,
    get_argo_user_from_flat_config,
    get_base_url_for_model_from_flat_config,
)
from chemgraph.memory.store import SessionStore

# Initialize rich console
console = Console()


@contextmanager
def timeout(seconds):
    """Context manager for timeout functionality - works on Unix and Windows."""
    if platform.system() == "Windows":
        # Signals are unavailable on Windows; no-op timeout in this context.
        yield
        return

    # Unix-based timeout using signals
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the old signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def check_api_keys(model_name: str) -> tuple[bool, str]:
    """
    Check if required API keys are available for the specified model.

    Returns:
        tuple: (is_available, error_message)
    """
    model_lower = model_name.lower()

    # Check OpenAI models
    if any(provider in model_lower for provider in ["o1", "o3", "o4"]):
        if not os.getenv("OPENAI_API_KEY"):
            return (
                False,
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.",
            )

    # Check Anthropic models
    elif "claude" in model_lower:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return (
                False,
                "Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.",
            )

    # Check Google models
    elif "gemini" in model_lower:
        if not os.getenv("GEMINI_API_KEY"):
            return (
                False,
                "Gemini API key not found. Please set GEMINI_API_KEY environment variable.",
            )
    # check GROQ models
    elif "groq" in model_lower:
        if not os.getenv("GROQ_API_KEY"):
            return (
                False,
                "GROQ API key not found. Please set GROQ_API_KEY environment variable.",
            )
    # Check local models (no API key needed)
    elif any(local in model_lower for local in ["llama", "qwen", "ollama"]):
        # For local models, we might want to check if the service is running
        # but for now, we'll assume they're available
        pass

    return True, ""


def create_banner():
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


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="ChemGraph CLI - AI Agents for Computational Chemistry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -q "What is the SMILES string for water?"
  %(prog)s -q "Optimize water molecule geometry" -m gpt-4o -w single_agent
  %(prog)s -q "Calculate CO2 vibrational frequencies" -m claude-3-sonnet-20240229 -r
  %(prog)s -q "Show me the structure of caffeine" -o last_message -s
  %(prog)s --config config.toml -q "Calculate frequencies"
  %(prog)s --interactive
  %(prog)s --list-models
  %(prog)s --check-keys

Session management:
  %(prog)s --list-sessions
  %(prog)s --show-session a3b2
  %(prog)s --delete-session a3b2c1d4
  %(prog)s -q "Optimize the geometry" --resume a3b2
        """,
    )

    # Main query argument
    parser.add_argument(
        "-q", "--query", type=str, help="The computational chemistry query to execute"
    )

    # Model selection
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )

    # Workflow type
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        choices=["single_agent", "multi_agent", "python_repl", "graspa"],
        default="single_agent",
        help="Workflow type (default: single_agent)",
    )

    # Output format
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        choices=["state", "last_message"],
        default="state",
        help="Output format (default: state)",
    )

    # Structured output
    parser.add_argument(
        "-s", "--structured", action="store_true", help="Use structured output format"
    )

    # Generate report
    parser.add_argument(
        "-r", "--report", action="store_true", help="Generate detailed report"
    )

    # Recursion limit
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=20,
        help="Recursion limit for agent workflows (default: 20)",
    )

    # Interactive mode
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive mode"
    )

    # List available models
    parser.add_argument(
        "--list-models", action="store_true", help="List all available models"
    )

    # Check API keys
    parser.add_argument(
        "--check-keys", action="store_true", help="Check API key availability"
    )

    # Session management
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List recent sessions from the memory database",
    )

    parser.add_argument(
        "--show-session",
        type=str,
        metavar="ID",
        help="Show conversation for a session (supports prefix matching)",
    )

    parser.add_argument(
        "--delete-session",
        type=str,
        metavar="ID",
        help="Delete a session from the memory database",
    )

    parser.add_argument(
        "--resume",
        type=str,
        metavar="ID",
        help="Resume from a previous session (injects context into new query)",
    )

    # Verbose output
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    # Output file
    parser.add_argument("--output-file", type=str, help="Save output to file")

    # Configuration file
    parser.add_argument("--config", type=str, help="Load configuration from TOML file")

    return parser


def list_models():
    """Display available models in a formatted table."""
    console.print(Panel("🧠 Available Models", style="bold cyan"))

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


def run_async_callable(fn):
    """Run an async callable and return its result in sync context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(fn())

    result_container = {}
    error_container = {}

    def runner():
        try:
            result_container["value"] = asyncio.run(fn())
        except Exception as exc:
            error_container["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_container:
        raise error_container["error"]
    return result_container.get("value")


def check_api_keys_status():
    """Display API key availability status."""
    console.print(Panel("🔑 API Key Status", style="bold cyan"))

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
            "examples": "gemini-pro, gemini-1.5-pro",
        },
        {
            "provider": "GROQ",
            "env_var": "GROQ_API_KEY",
            "examples": "gpt-oss-20b, gpt-oss-120b",
        },
        {
            "provider": "Local/Ollama",
            "env_var": "Not Required",
            "examples": "llama3.2, qwen2.5",
        },
    ]

    for key_info in api_keys:
        if key_info["env_var"] == "Not Required":
            status = "[green]✓ Available[/green]"
        else:
            is_set = bool(os.getenv(key_info["env_var"]))
            status = "[green]✓ Set[/green]" if is_set else "[red]✗ Missing[/red]"

        table.add_row(
            key_info["provider"], key_info["env_var"], status, key_info["examples"]
        )

    console.print(table)

    console.print("\n[bold]💡 How to set API keys:[/bold]")
    console.print("• [cyan]Bash/Zsh:[/cyan] export OPENAI_API_KEY='your_key_here'")
    console.print("• [cyan]Fish:[/cyan] set -x OPENAI_API_KEY 'your_key_here'")
    console.print(
        "• [cyan].env file:[/cyan] Add OPENAI_API_KEY=your_key_here to a .env file"
    )
    console.print(
        "• [cyan]Python:[/cyan] os.environ['OPENAI_API_KEY'] = 'your_key_here'"
    )

    console.print("\n[bold]🔗 Get API keys:[/bold]")
    console.print("• [cyan]OpenAI:[/cyan] https://platform.openai.com/api-keys")
    console.print("• [cyan]Anthropic:[/cyan] https://console.anthropic.com/")
    console.print("• [cyan]Google:[/cyan] https://aistudio.google.com/apikey")


def list_sessions(limit: int = 20, db_path: str = None):
    """Display recent sessions in a formatted table."""
    store = SessionStore(db_path=db_path)
    sessions = store.list_sessions(limit=limit)

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    console.print(Panel(f"Recent Sessions ({len(sessions)})", style="bold cyan"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Session ID", style="cyan", width=10)
    table.add_column("Title", style="white", width=40)
    table.add_column("Model", style="green", width=16)
    table.add_column("Workflow", style="yellow", width=14)
    table.add_column("Queries", style="white", justify="right", width=8)
    table.add_column("Messages", style="white", justify="right", width=9)
    table.add_column("Date", style="dim", width=16)

    for s in sessions:
        table.add_row(
            s.session_id,
            s.title or "[dim]Untitled[/dim]",
            s.model_name,
            s.workflow_type,
            str(s.query_count),
            str(s.message_count),
            s.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    console.print(
        "\n[dim]Use --show-session <id> to view a session's conversation. "
        "Prefix matching is supported (e.g. first few chars).[/dim]"
    )


def show_session(session_id: str, db_path: str = None, max_content: int = 500):
    """Display a session's full conversation."""
    store = SessionStore(db_path=db_path)
    session = store.get_session(session_id)

    if session is None:
        console.print(
            f"[red]Session '{session_id}' not found. "
            f"The ID may be ambiguous or nonexistent.[/red]"
        )
        console.print("[dim]Use --list-sessions to see available sessions.[/dim]")
        return

    # Session metadata header
    meta_table = Table(show_header=False, box=None, padding=(0, 2))
    meta_table.add_column("Key", style="bold cyan")
    meta_table.add_column("Value")
    meta_table.add_row("Session ID", session.session_id)
    meta_table.add_row("Title", session.title or "Untitled")
    meta_table.add_row("Model", session.model_name)
    meta_table.add_row("Workflow", session.workflow_type)
    meta_table.add_row("Queries", str(session.query_count))
    meta_table.add_row("Created", session.created_at.strftime("%Y-%m-%d %H:%M:%S"))
    meta_table.add_row("Updated", session.updated_at.strftime("%Y-%m-%d %H:%M:%S"))
    if session.log_dir:
        meta_table.add_row("Log Dir", session.log_dir)

    console.print(Panel(meta_table, title="Session Info", style="bold cyan"))

    if not session.messages:
        console.print("[dim]No messages in this session.[/dim]")
        return

    # Display conversation
    console.print(f"\n[bold]Conversation ({len(session.messages)} messages):[/bold]\n")

    for msg in session.messages:
        if msg.role == "human":
            label = "[bold cyan]User[/bold cyan]"
        elif msg.role == "ai":
            label = "[bold green]Assistant[/bold green]"
        elif msg.role == "tool":
            tool_label = f" ({msg.tool_name})" if msg.tool_name else ""
            label = f"[bold yellow]Tool{tool_label}[/bold yellow]"
        else:
            label = f"[dim]{msg.role}[/dim]"

        content = msg.content
        if max_content and len(content) > max_content:
            content = (
                content[:max_content]
                + f"\n... [truncated, {len(msg.content)} chars total]"
            )

        timestamp = msg.timestamp.strftime("%H:%M:%S") if msg.timestamp else ""

        console.print(f"  {label} [dim]{timestamp}[/dim]")
        console.print(f"    {content}\n")


def delete_session_cmd(session_id: str, db_path: str = None):
    """Delete a session from the database."""
    store = SessionStore(db_path=db_path)

    # Show session info before deleting
    session = store.get_session(session_id)
    if session is None:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        return

    console.print(
        f"[yellow]Deleting session: {session.session_id} "
        f"({session.title or 'Untitled'})[/yellow]"
    )

    if store.delete_session(session_id):
        console.print("[green]Session deleted.[/green]")
    else:
        console.print("[red]Failed to delete session.[/red]")


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    try:
        with open(config_file, "r") as f:
            config = toml.load(f)
        console.print(f"[green]✓[/green] Configuration loaded from {config_file}")

        flattened = flatten_config(config)

        return flattened

    except FileNotFoundError:
        console.print(f"[red]✗[/red] Configuration file not found: {config_file}")
        sys.exit(1)
    except toml.TomlDecodeError as e:
        console.print(f"[red]✗[/red] Invalid TOML in configuration file: {e}")
        sys.exit(1)


def initialize_agent(
    model_name: str,
    workflow_type: str,
    structured_output: bool,
    return_option: str,
    generate_report: bool,
    recursion_limit: int,
    base_url: str = None,
    argo_user: str = None,
    verbose: bool = False,
):
    """Initialize ChemGraph agent with progress indication."""

    if verbose:
        console.print("[blue]Initializing agent with:[/blue]")
        console.print(f"  Model: {model_name}")
        console.print(f"  Workflow: {workflow_type}")
        console.print(f"  Structured Output: {structured_output}")
        console.print(f"  Return Option: {return_option}")
        console.print(f"  Generate Report: {generate_report}")
        console.print(f"  Recursion Limit: {recursion_limit}")
        console.print(f"  Base URL: {base_url}")
        console.print(f"  Argo User: {argo_user}")

    # Check API keys before attempting initialization
    api_key_available, error_msg = check_api_keys(model_name)
    if not api_key_available:
        console.print(f"[red]✗ {error_msg}[/red]")
        console.print(
            "[dim]💡 Tip: You can set environment variables in your shell or .env file[/dim]"
        )
        console.print(
            "[dim]   Example: export OPENAI_API_KEY='your_api_key_here'[/dim]"
        )
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Initializing ChemGraph agent...", total=None)

        try:
            # Add timeout to prevent hanging
            with timeout(30):  # 30 second timeout
                from chemgraph.agent.llm_agent import ChemGraph

                agent = ChemGraph(
                    model_name=model_name,
                    workflow_type=workflow_type,
                    base_url=base_url,
                    argo_user=argo_user,
                    generate_report=generate_report,
                    return_option=return_option,
                    recursion_limit=recursion_limit,
                )

            progress.update(task, description="[green]Agent initialized successfully!")
            time.sleep(0.5)  # Brief pause to show success message

            return agent

        except TimeoutError:
            progress.update(task, description="[red]Agent initialization timed out!")
            console.print(
                "[red]✗ Agent initialization timed out after 30 seconds[/red]"
            )
            console.print(
                "[dim]💡 This might indicate network issues or invalid API credentials[/dim]"
            )
            return None
        except Exception as e:
            progress.update(task, description="[red]Agent initialization failed!")
            console.print(f"[red]✗ Error initializing agent: {e}[/red]")

            # Provide more helpful error messages
            if "authentication" in str(e).lower() or "api" in str(e).lower():
                console.print(
                    "[dim]💡 This looks like an API key issue. Please check your credentials.[/dim]"
                )
            elif "connection" in str(e).lower() or "network" in str(e).lower():
                console.print(
                    "[dim]💡 This looks like a network connectivity issue.[/dim]"
                )

            return None


def format_response(result, verbose: bool = False):
    """Format the agent response for display."""
    if not result:
        console.print("[red]No response received from agent.[/red]")
        return

    # Extract messages from result
    messages = []
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
                if not (
                    content.startswith("{")
                    and content.endswith("}")
                    and "numbers" in content
                ):
                    final_answer = content
                    break
        elif isinstance(message, dict):
            if message.get("type") == "ai" and message.get("content", "").strip():
                content = message["content"].strip()
                if not (
                    content.startswith("{")
                    and content.endswith("}")
                    and "numbers" in content
                ):
                    final_answer = content
                    break

    if final_answer:
        console.print(
            Panel(
                Markdown(final_answer),
                title="🅒🅖 ChemGraph Response",
                style="green",
                padding=(1, 2),
            )
        )

    # Check for structure data
    for message in messages:
        content = ""
        if hasattr(message, "content"):
            content = message.content
        elif isinstance(message, dict):
            content = message.get("content", "")

        if content and ("numbers" in content or "positions" in content):
            console.print(
                Panel(
                    Syntax(content, "json", theme="monokai"),
                    title="🧬 Molecular Structure Data",
                    style="cyan",
                )
            )

    # Verbose output
    if verbose:
        console.print(
            Panel(
                f"Messages: {len(messages)}", title="🔍 Debug Information", style="dim"
            )
        )


def run_query(
    agent,
    query: str,
    thread_id: int,
    verbose: bool = False,
    resume_from: str = None,
):
    """Execute a query with the agent."""
    if verbose:
        console.print(f"[blue]Executing query:[/blue] {query}")
        console.print(f"[blue]Thread ID:[/blue] {thread_id}")
        if resume_from:
            console.print(f"[blue]Resuming from session:[/blue] {resume_from}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Processing query...", total=None)

        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = run_async_callable(
                lambda: agent.run(query, config=config, resume_from=resume_from)
            )

            progress.update(task, description="[green]Query completed!")
            time.sleep(0.5)

            return result

        except Exception as e:
            progress.update(task, description="[red]Query failed!")
            console.print(f"[red]✗ Error processing query: {e}[/red]")
            return None


def interactive_mode():
    """Start interactive mode for ChemGraph CLI."""
    console.print(create_banner())
    console.print("[bold green]Welcome to ChemGraph Interactive Mode![/bold green]")
    console.print(
        "Type your queries and get AI-powered computational chemistry insights."
    )
    console.print(
        "[dim]Type 'quit', 'exit', or 'q' to exit. Type 'help' for commands.[/dim]\n"
    )

    # Get initial configuration
    model = Prompt.ask(
        "Select model (or type a custom model ID)", default="gpt-4o-mini"
    )
    workflow = Prompt.ask(
        "Select workflow",
        choices=["single_agent", "multi_agent", "python_repl", "graspa"],
        default="single_agent",
    )

    # Initialize agent
    agent = initialize_agent(model, workflow, False, "state", True, 20, verbose=True)
    if not agent:
        return

    console.print(
        "[green]✓ Ready! You can now ask computational chemistry questions.[/green]\n"
    )

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]🧪 ChemGraph[/bold cyan]")

            if query.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            elif query.lower() == "help":
                console.print(
                    Panel(
                        """
Available commands:
• quit/exit/q - Exit interactive mode
• help - Show this help message
• clear - Clear screen
• config - Show current configuration
• model <name> - Change model
• workflow <type> - Change workflow type

Session commands:
• history - List recent sessions
• show <id> - Show a session's conversation
• resume <id> - Resume from a previous session

Example queries:
• What is the SMILES string for water?
• Optimize the geometry of methane
• Calculate CO2 vibrational frequencies
• Show me the structure of caffeine
                    """,
                        title="Help",
                        style="blue",
                    )
                )
                continue
            elif query.lower() == "clear":
                console.clear()
                continue
            elif query.lower() == "config":
                console.print(f"Model: {model}")
                console.print(f"Workflow: {workflow}")
                if hasattr(agent, "session_id"):
                    console.print(f"Session ID: {agent.session_id}")
                continue
            elif query.lower() == "history":
                list_sessions()
                continue
            elif query.lower().startswith("show "):
                sid = query[5:].strip()
                if sid:
                    show_session(sid)
                else:
                    console.print("[red]Usage: show <session_id>[/red]")
                continue
            elif query.lower().startswith("resume "):
                sid = query[7:].strip()
                if not sid:
                    console.print("[red]Usage: resume <session_id>[/red]")
                    continue
                resume_query = Prompt.ask(
                    "[bold cyan]Enter query to continue with[/bold cyan]"
                )
                if resume_query.strip():
                    result = run_query(
                        agent,
                        resume_query,
                        1,
                        verbose=False,
                        resume_from=sid,
                    )
                    if result:
                        format_response(result, verbose=False)
                continue
            elif query.startswith("model "):
                new_model = query[6:].strip()
                model = new_model
                agent = initialize_agent(model, workflow, False, "state", True, 20)
                if agent:
                    console.print(f"[green]✓ Model changed to: {model}[/green]")
                continue
            elif query.startswith("workflow "):
                new_workflow = query[9:].strip()
                if new_workflow in [
                    "single_agent",
                    "multi_agent",
                    "python_repl",
                    "graspa",
                ]:
                    workflow = new_workflow
                    agent = initialize_agent(model, workflow, False, "state", True, 20)
                    if agent:
                        console.print(
                            f"[green]✓ Workflow changed to: {workflow}[/green]"
                        )
                else:
                    console.print(f"[red]✗ Invalid workflow: {new_workflow}[/red]")
                continue

            # Execute query
            result = run_query(agent, query, 1, verbose=False)
            if result:
                format_response(result, verbose=False)

        except KeyboardInterrupt:
            console.print(
                "\n[yellow]Interrupted by user. Type 'quit' to exit.[/yellow]"
            )
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")


def save_output(content: str, output_file: str):
    """Save output to file."""
    try:
        with open(output_file, "w") as f:
            f.write(content)
        console.print(f"[green]✓ Output saved to: {output_file}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error saving output: {e}[/red]")


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle special commands
    if args.list_models:
        list_models()
        return

    if args.check_keys:
        check_api_keys_status()
        return

    if args.list_sessions:
        list_sessions()
        return

    if args.show_session:
        show_session(args.show_session)
        return

    if args.delete_session:
        delete_session_cmd(args.delete_session)
        return

    if args.interactive:
        interactive_mode()
        return

    # Load configuration if specified
    config = {}
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
        # Honor config recursion_limit unless user explicitly provided CLI flag.
        if "recursion_limit" in config and "--recursion-limit" not in sys.argv:
            args.recursion_limit = config["recursion_limit"]

    base_url = (
        get_base_url_for_model_from_flat_config(args.model, config) if config else None
    )
    argo_user = get_argo_user_from_flat_config(config) if config else None

    if args.model not in all_supported_models:
        console.print(
            f"[yellow]⚠ Using custom model ID: {args.model} (not in curated list)[/yellow]"
        )

    # Require query for non-interactive mode
    if not args.query:
        console.print("[red]✗ Query is required. Use -q or --query to specify.[/red]")
        console.print(
            "Use --help for more information or --interactive for interactive mode."
        )
        sys.exit(1)

    # Show banner
    console.print(create_banner())

    # Initialize agent
    agent = initialize_agent(
        args.model,
        args.workflow,
        args.structured,
        args.output,
        args.report,
        args.recursion_limit,
        base_url=base_url,
        argo_user=argo_user,
        verbose=args.verbose,
    )

    if not agent:
        sys.exit(1)

    # Execute query
    console.print(f"[bold blue]Query:[/bold blue] {args.query}")
    if args.resume:
        console.print(f"[bold blue]Resuming from:[/bold blue] {args.resume}")
    result = run_query(agent, args.query, 1, args.verbose, resume_from=args.resume)

    if result:
        format_response(result, args.verbose)

        # Save output if requested
        if args.output_file:
            # Convert result to string format
            output_content = str(result)
            save_output(output_content, args.output_file)

    console.print("\n[dim]Thank you for using ChemGraph CLI! 🧪[/dim]")


if __name__ == "__main__":
    main()
