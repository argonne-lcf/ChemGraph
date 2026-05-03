"""Command implementations for the ChemGraph CLI.

Each public function corresponds to a CLI action: running a query,
starting interactive mode, managing sessions, etc.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from chemgraph.memory.store import SessionStore
from chemgraph.models.supported_models import (
    supported_alcf_models,
    supported_anthropic_models,
    supported_gemini_models,
    supported_ollama_models,
    supported_openai_models,
    supported_argo_models,
)
from chemgraph.utils.async_utils import run_async_callable

from chemgraph.cli.formatting import (
    console,
    create_banner,
    format_response,
)

# ---------------------------------------------------------------------------
# Workflow helpers
# ---------------------------------------------------------------------------

# All workflow types registered in ChemGraph.workflow_map
ALL_WORKFLOW_TYPES = [
    "single_agent",
    "multi_agent",
    "python_relp",
    "graspa",
    "mock_agent",
    "single_agent_mcp",
    "graspa_mcp",
    "rag_agent",
    "single_agent_xanes",
]

# Common aliases so users can type the "obvious" name.
WORKFLOW_ALIASES: Dict[str, str] = {
    "python_repl": "python_relp",
    "graspa_agent": "graspa",
}


def resolve_workflow(name: str) -> str:
    """Resolve a workflow name, applying aliases."""
    return WORKFLOW_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# API-key validation
# ---------------------------------------------------------------------------


def check_api_keys(model_name: str) -> tuple[bool, str]:
    """Check if required API keys are available for *model_name*.

    Returns ``(is_available, error_message)``.
    """
    model_lower = model_name.lower()

    # OpenAI models (including GPT family, o-series, and Argo OpenAI)
    if (
        model_name in supported_openai_models
        or model_name in supported_argo_models
        or model_lower.startswith("gpt")
        or any(prefix in model_lower for prefix in ["o1", "o3", "o4"])
    ):
        # Argo models use a different auth mechanism; skip key check.
        if model_name in supported_argo_models:
            pass
        elif not os.getenv("OPENAI_API_KEY"):
            return (
                False,
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable.",
            )

    # Anthropic models
    elif "claude" in model_lower or model_name in supported_anthropic_models:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return (
                False,
                "Anthropic API key not found. Set the ANTHROPIC_API_KEY environment variable.",
            )

    # Google models
    elif "gemini" in model_lower or model_name in supported_gemini_models:
        if not os.getenv("GEMINI_API_KEY"):
            return (
                False,
                "Gemini API key not found. Set the GEMINI_API_KEY environment variable.",
            )

    # GROQ models (groq: prefix)
    elif model_name.startswith("groq:"):
        if not os.getenv("GROQ_API_KEY"):
            return (
                False,
                "GROQ API key not found. Set the GROQ_API_KEY environment variable.",
            )

    # ALCF models (Globus OAuth access token)
    elif model_name in supported_alcf_models:
        if not os.getenv("ALCF_ACCESS_TOKEN"):
            return (
                False,
                "ALCF access token not found. To authenticate with ALCF:\n"
                "  1. pip install globus_sdk\n"
                "  2. wget https://raw.githubusercontent.com/argonne-lcf/"
                "inference-endpoints/refs/heads/main/inference_auth_token.py\n"
                "  3. python inference_auth_token.py authenticate\n"
                "  4. export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)\n"
                "\n"
                "  See: https://docs.alcf.anl.gov/services/inference-endpoints/#api-access",
            )

    # Local models (no API key needed)
    elif model_name in supported_ollama_models or any(
        local in model_lower for local in ["llama", "qwen", "ollama"]
    ):
        pass

    return True, ""


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

_INIT_TIMEOUT_SECONDS = 30


def initialize_agent(
    model_name: str,
    workflow_type: str,
    structured_output: bool,
    return_option: str,
    generate_report: bool,
    recursion_limit: int,
    base_url: Optional[str] = None,
    argo_user: Optional[str] = None,
    verbose: bool = False,
) -> Any:
    """Initialize a ChemGraph agent with progress indication.

    Uses a thread-pool executor for the timeout so it works on all
    platforms.
    """
    # Resolve workflow alias before initializing.
    workflow_type = resolve_workflow(workflow_type)

    if verbose:
        console.print("[blue]Initializing agent with:[/blue]")
        console.print(f"  Model: {model_name}")
        console.print(f"  Workflow: {workflow_type}")
        console.print(f"  Structured Output: {structured_output}")
        console.print(f"  Return Option: {return_option}")
        console.print(f"  Generate Report: {generate_report}")
        console.print(f"  Recursion Limit: {recursion_limit}")
        if base_url:
            console.print(f"  Base URL: {base_url}")
        if argo_user:
            console.print(f"  Argo User: {argo_user}")

    # Check API keys before attempting initialization
    api_key_available, error_msg = check_api_keys(model_name)
    if not api_key_available:
        console.print(f"[red]{error_msg}[/red]")
        console.print(
            "[dim]Tip: Set environment variables in your shell or .env file[/dim]"
        )
        console.print(
            "[dim]  Example: export OPENAI_API_KEY='your_api_key_here'[/dim]"
        )
        return None

    # Resolve API key for providers that need one passed explicitly.
    api_key: Optional[str] = None
    if model_name in supported_alcf_models:
        api_key = os.getenv("ALCF_ACCESS_TOKEN")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Initializing ChemGraph agent...", total=None)

        def _create_agent() -> Any:
            from chemgraph.agent.llm_agent import ChemGraph

            return ChemGraph(
                model_name=model_name,
                workflow_type=workflow_type,
                base_url=base_url,
                api_key=api_key,
                argo_user=argo_user,
                generate_report=generate_report,
                return_option=return_option,
                recursion_limit=recursion_limit,
                structured_output=structured_output,
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_create_agent)
                agent = future.result(timeout=_INIT_TIMEOUT_SECONDS)

            progress.update(task, description="[green]Agent initialized successfully!")
            time.sleep(0.5)
            return agent

        except FuturesTimeoutError:
            progress.update(task, description="[red]Agent initialization timed out!")
            console.print(
                f"[red]Agent initialization timed out after {_INIT_TIMEOUT_SECONDS}s[/red]"
            )
            console.print(
                "[dim]This might indicate network issues or invalid API credentials[/dim]"
            )
            return None
        except Exception as e:
            progress.update(task, description="[red]Agent initialization failed!")
            console.print(f"[red]Error initializing agent: {e}[/red]")

            err_str = str(e).lower()
            if "authentication" in err_str or "api" in err_str:
                console.print(
                    "[dim]This looks like an API key issue. Check your credentials.[/dim]"
                )
            elif "connection" in err_str or "network" in err_str:
                console.print(
                    "[dim]This looks like a network connectivity issue.[/dim]"
                )
            return None


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

# Thread-ID counter for interactive mode so each query gets unique state.
_thread_counter: int = 0


def _next_thread_id() -> int:
    global _thread_counter
    _thread_counter += 1
    return _thread_counter


def run_query(
    agent: Any,
    query: str,
    thread_id: Optional[int] = None,
    verbose: bool = False,
    resume_from: Optional[str] = None,
) -> Any:
    """Execute a query with the agent."""
    if thread_id is None:
        thread_id = _next_thread_id()

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
            console.print(f"[red]Error processing query: {e}[/red]")
            return None


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def list_sessions(limit: int = 20, db_path: Optional[str] = None) -> None:
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
        "\n[dim]Use 'chemgraph session show <id>' to view a session. "
        "Prefix matching is supported.[/dim]"
    )


def show_session(
    session_id: str,
    db_path: Optional[str] = None,
    max_content: int = 500,
) -> None:
    """Display a session's full conversation."""
    store = SessionStore(db_path=db_path)
    session = store.get_session(session_id)

    if session is None:
        console.print(
            f"[red]Session '{session_id}' not found. "
            f"The ID may be ambiguous or nonexistent.[/red]"
        )
        console.print("[dim]Use 'chemgraph session list' to see available sessions.[/dim]")
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


def delete_session_cmd(session_id: str, db_path: Optional[str] = None) -> None:
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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_output(content: str, output_file: str) -> None:
    """Save output to a file."""
    try:
        with open(output_file, "w") as f:
            f.write(content)
        console.print(f"[green]Output saved to: {output_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving output: {e}[/red]")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def interactive_mode(
    model: str = "gpt-4o-mini",
    workflow: str = "single_agent",
    structured: bool = False,
    return_option: str = "state",
    generate_report: bool = True,
    recursion_limit: int = 20,
    base_url: Optional[str] = None,
    argo_user: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Start interactive REPL mode for ChemGraph CLI.

    Accepts the same configuration parameters as a normal run so that
    ``--config`` and CLI flags are honoured when entering interactive
    mode.
    """
    console.print(create_banner())
    console.print("[bold green]Welcome to ChemGraph Interactive Mode![/bold green]")
    console.print(
        "Type your queries and get AI-powered computational chemistry insights."
    )
    console.print(
        "[dim]Type 'quit', 'exit', or 'q' to exit. Type 'help' for commands.[/dim]\n"
    )

    # Allow the user to override model/workflow at startup.
    model = Prompt.ask(
        "Select model (or type a custom model ID)", default=model
    )
    workflow = Prompt.ask(
        "Select workflow",
        choices=ALL_WORKFLOW_TYPES,
        default=resolve_workflow(workflow),
    )

    # Initialize agent with the full config context.
    agent = initialize_agent(
        model,
        workflow,
        structured,
        return_option,
        generate_report,
        recursion_limit,
        base_url=base_url,
        argo_user=argo_user,
        verbose=verbose,
    )
    if not agent:
        return

    console.print(
        "[green]Ready! You can now ask computational chemistry questions.[/green]\n"
    )

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]ChemGraph[/bold cyan]")

            if query.lower() in ("quit", "exit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif query.lower() == "help":
                console.print(
                    Panel(
                        """
Available commands:
  quit/exit/q        Exit interactive mode
  help               Show this help message
  clear              Clear screen
  config             Show current configuration
  model <name>       Change model
  workflow <type>    Change workflow type

Session commands:
  history            List recent sessions
  show <id>          Show a session's conversation
  resume <id>        Resume from a previous session

Example queries:
  What is the SMILES string for water?
  Optimize the geometry of methane
  Calculate CO2 vibrational frequencies
  Show me the structure of caffeine
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
                        verbose=verbose,
                        resume_from=sid,
                    )
                    if result:
                        format_response(result, verbose=verbose)
                continue
            elif query.startswith("model "):
                new_model = query[6:].strip()
                model = new_model
                agent = initialize_agent(
                    model,
                    workflow,
                    structured,
                    return_option,
                    generate_report,
                    recursion_limit,
                    base_url=base_url,
                    argo_user=argo_user,
                )
                if agent:
                    console.print(f"[green]Model changed to: {model}[/green]")
                continue
            elif query.startswith("workflow "):
                new_workflow = resolve_workflow(query[9:].strip())
                if new_workflow in ALL_WORKFLOW_TYPES:
                    workflow = new_workflow
                    agent = initialize_agent(
                        model,
                        workflow,
                        structured,
                        return_option,
                        generate_report,
                        recursion_limit,
                        base_url=base_url,
                        argo_user=argo_user,
                    )
                    if agent:
                        console.print(
                            f"[green]Workflow changed to: {workflow}[/green]"
                        )
                else:
                    console.print(f"[red]Invalid workflow: {new_workflow}[/red]")
                    console.print(
                        f"[dim]Available: {', '.join(ALL_WORKFLOW_TYPES)}[/dim]"
                    )
                continue

            # Execute query (each query gets a unique thread ID)
            result = run_query(agent, query, verbose=verbose)
            if result:
                format_response(result, verbose=verbose)
                if hasattr(agent, "session_id") and agent.session_id:
                    console.print(f"[dim]Session: {agent.session_id}[/dim]")

        except KeyboardInterrupt:
            console.print(
                "\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]"
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
