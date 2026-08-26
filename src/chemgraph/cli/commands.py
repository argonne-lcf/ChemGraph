"""Command implementations for the ChemGraph CLI.

Each public function corresponds to a CLI action: running a query,
starting interactive mode, managing sessions, etc.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rich.panel import Panel
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from chemgraph.agent.interrupts import (
    deduplicate_interrupts,
    interrupt_question as _interrupt_question,
    normalize_interrupts,
)
from chemgraph.graphs.deep_agent import _normalize_skill_sources
from chemgraph.memory.store import SessionStore
from chemgraph.memory.durable import delete_durable_session
from chemgraph.cli.checkpoint_runtime import (
    DEFAULT_CHECKPOINT_DB,
    CheckpointRuntime,
)
from chemgraph.models.endpoints import (
    ModelRequest,
    missing_credential_help,
)
from chemgraph.models.endpoints.registry import select_endpoint
from chemgraph.models.supported_models import MODELS_WITH_REASONING_EFFORT
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
    "main_agent",
    "deep_agent",
    "multi_agent",
    "python_relp",
    "graspa",
    "mock_agent",
    "graspa_mcp",
    "rag_agent",
    "single_agent_xanes",
    "molecular_docking",
    "single_agent_iri",
]

# Common aliases so users can type the "obvious" name.
WORKFLOW_ALIASES: Dict[str, str] = {
    "deepagent": "deep_agent",
    "python_repl": "python_relp",
    "graspa_agent": "graspa",
    "iri": "single_agent_iri",
}


def resolve_workflow(name: str) -> str:
    """Resolve a workflow name, applying aliases.

    Parameters
    ----------
    name : str
        Workflow name or supported alias.

    Returns
    -------
    str
        Canonical workflow name.
    """
    return WORKFLOW_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# API-key validation
# ---------------------------------------------------------------------------


def check_api_keys(
    model_name: str,
    *,
    base_url: str | None = None,
) -> tuple[bool, str]:
    """Check if required API keys are available for *model_name*.

    Parameters
    ----------
    model_name : str
        Model identifier selected for a run.

    Returns
    -------
    tuple[bool, str]
        ``(is_available, error_message)``. The message is empty when the
        required credentials are available or not required.
    """
    try:
        spec = select_endpoint(ModelRequest(model=model_name, base_url=base_url))
    except ValueError as exc:
        return False, str(exc)

    policy = spec.credential
    if not policy.required:
        return True, ""
    if policy.env_var and os.getenv(policy.env_var):
        return True, ""
    return False, missing_credential_help(policy)


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

_INIT_TIMEOUT_SECONDS = 30

_DEEPAGENT_ENV_ALLOWLIST = (
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "TMPDIR",
    "CHEMGRAPH_LOG_DIR",
)


def _create_experimental_deepagent_backend(
    workspace: str | None,
    *,
    require_confirmation: bool = True,
):
    """Create the explicitly approved development-only host-shell backend."""
    from deepagents.backends import LocalShellBackend

    root = Path(workspace or Path.cwd()).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Deep Agent workspace is not a directory: {root}")

    console.print(
        Panel(
            "The experimental Deep Agent can read and modify files under "
            f"{root} and can run arbitrary shell commands on this host. The "
            "shell is not confined to that directory. "
            + (
                "Every shell command and file mutation will require approval."
                if require_confirmation
                else "Tool approvals are disabled for this run."
            ),
            title="[bold red]Experimental host-shell access[/bold red]",
            style="red",
        )
    )
    if require_confirmation and not Confirm.ask(
        "Enable this development-only capability?",
        default=False,
    ):
        raise RuntimeError("Experimental Deep Agent access was not approved.")

    environment = {
        name: os.environ[name]
        for name in _DEEPAGENT_ENV_ALLOWLIST
        if name in os.environ
    }
    return LocalShellBackend(
        root_dir=root,
        virtual_mode=True,
        env=environment,
        inherit_env=False,
    )


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
    human_supervised: bool = False,
    tools: Optional[list] = None,
    on_event: Optional[Any] = None,
    enable_deepagent: bool = False,
    deepagent_workspace: str | None = None,
    deepagent_skills: Sequence[str] | None = None,
    deepagent_auto_approve: bool = False,
    checkpointer: Any | None = None,
    reasoning_effort: str | None = None,
    max_retries: int = 1,
    terminal_tool_names: tuple[str, ...] = (),
) -> Any:
    """Initialize a ChemGraph agent with progress indication.

    Uses a thread-pool executor for the timeout so it works on all
    platforms.

    Parameters
    ----------
    model_name : str
        LLM model identifier.
    workflow_type : str
        ChemGraph workflow name or alias.
    structured_output : bool
        Whether to request structured final output.
    return_option : str
        Agent return mode, such as ``"state"`` or ``"last_message"``.
    generate_report : bool
        Whether the agent should generate an HTML report.
    recursion_limit : int
        LangGraph recursion limit for the run.
    base_url : str, optional
        Custom model endpoint URL.
    argo_user : str, optional
        Argo username for Argo-hosted models.
    verbose : bool, optional
        Whether to print initialization details.
    human_supervised : bool, optional
        Whether to enable human-interrupt tooling.
    tools : list, optional
        Custom tools for MCP-backed workflows, or supervisor-level tools for
        ``main_agent``. The default chemistry subagent retains its built-ins.
    enable_deepagent : bool, optional
        Enable the development-only workspace worker for ``main_agent``.
    deepagent_workspace : str, optional
        Root directory exposed to the development-only local backend.
    deepagent_skills : sequence of str, optional
        Ordered backend-relative Agent Skills directories.
    deepagent_auto_approve : bool, optional
        Disable tool-review interrupts for standalone ``deep_agent``. This is
        intended only for explicitly trusted, isolated headless runs.

    Returns
    -------
    Any
        Initialized ``ChemGraph`` instance, or ``None`` when initialization
        fails.
    """
    # Resolve workflow alias before initializing.
    workflow_type = resolve_workflow(workflow_type)
    deepagent_skills = _normalize_skill_sources(deepagent_skills)
    if enable_deepagent and workflow_type != "main_agent":
        raise ValueError(
            "The experimental Deep Agent is available only with main_agent."
        )
    uses_deepagent = enable_deepagent or workflow_type == "deep_agent"
    if deepagent_workspace is not None and not uses_deepagent:
        raise ValueError(
            "deepagent_workspace requires enable_deepagent=True or the "
            "deep_agent workflow."
        )
    if deepagent_skills and not uses_deepagent:
        raise ValueError(
            "deepagent_skills requires enable_deepagent=True or the "
            "deep_agent workflow."
        )
    if deepagent_auto_approve and workflow_type != "deep_agent":
        raise ValueError(
            "deepagent_auto_approve is available only for the deep_agent workflow."
        )
    if deepagent_auto_approve and not deepagent_workspace:
        raise ValueError(
            "deepagent_auto_approve requires an explicit deepagent_workspace."
        )

    deepagent_backend = None
    if uses_deepagent:
        try:
            deepagent_backend = _create_experimental_deepagent_backend(
                deepagent_workspace,
                require_confirmation=not deepagent_auto_approve,
            )
        except (RuntimeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            return None

    if verbose:
        console.print("[blue]Initializing agent with:[/blue]")
        console.print(f"  Model: {model_name}")
        console.print(f"  Workflow: {workflow_type}")
        console.print(f"  Structured Output: {structured_output}")
        console.print(f"  Return Option: {return_option}")
        console.print(f"  Generate Report: {generate_report}")
        console.print(f"  Human Supervised: {human_supervised}")
        console.print(f"  Recursion Limit: {recursion_limit}")
        console.print(f"  Deep Agent: {uses_deepagent}")
        if deepagent_skills:
            console.print(f"  Deep Agent Skills: {len(deepagent_skills)} source(s)")
        if base_url:
            console.print(f"  Base URL: {base_url}")
        if argo_user:
            console.print(f"  Argo User: {argo_user}")
        if tools:
            console.print(f"  MCP Tools: {len(tools)} loaded")

    # Check API keys before attempting initialization
    api_key_available, error_msg = check_api_keys(model_name, base_url=base_url)
    if not api_key_available:
        console.print(f"[red]{error_msg}[/red]")
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Initializing ChemGraph agent...", total=None)

        def _create_agent() -> Any:
            """Create the ChemGraph agent inside the initialization worker.

            Returns
            -------
            Any
                Initialized ``ChemGraph`` instance.
            """
            from chemgraph.agent.llm_agent import ChemGraph

            return ChemGraph(
                model_name=model_name,
                workflow_type=workflow_type,
                base_url=base_url,
                argo_user=argo_user,
                generate_report=generate_report,
                return_option=return_option,
                recursion_limit=recursion_limit,
                structured_output=structured_output,
                human_supervised=human_supervised,
                tools=tools,
                on_event=on_event,
                enable_deepagent=enable_deepagent,
                deepagent_backend=deepagent_backend,
                deepagent_skills=deepagent_skills,
                deepagent_auto_approve=deepagent_auto_approve,
                checkpointer=checkpointer,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
                terminal_tool_names=terminal_tool_names,
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
            console.print(f"[red]Error initializing agent: {escape(str(e))}[/red]")

            err_str = str(e).lower()
            if model_name.startswith("codex:"):
                console.print(
                    "[dim]Install the Codex CLI separately, install the optional "
                    "chemgraph\\[codex] extra, then run `codex login` and choose "
                    "ChatGPT authentication.[/dim]"
                )
            elif "authentication" in err_str or "api" in err_str:
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
    """Return the next interactive-mode thread ID.

    Returns
    -------
    int
        Incremented thread ID.
    """
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
    """Execute a query with the agent.

    When the graph pauses for human input (``HumanInputRequired``), the
    spinner is stopped, the question is shown in a Rich panel, and the
    user is prompted for a response.  The graph is then resumed with the
    user's answer and the spinner restarts.  This loop repeats until the
    graph completes or a non-interrupt error occurs.

    Parameters
    ----------
    agent : Any
        Initialized ChemGraph-like agent with ``run`` and ``workflow`` methods.
    query : str
        User query to execute.
    thread_id : int, optional
        LangGraph thread identifier. A new ID is allocated when omitted.
    verbose : bool, optional
        Whether to print execution details.
    resume_from : str, optional
        Previous ChemGraph session ID to load as context.

    Returns
    -------
    Any
        Agent result, resumed graph result, or ``None`` on failure.
    """
    from langgraph.types import Command
    from chemgraph.agent.llm_agent import HumanInputRequired

    if thread_id is None:
        thread_id = _next_thread_id()

    if verbose:
        console.print(f"[blue]Executing query:[/blue] {query}")
        console.print(f"[blue]Thread ID:[/blue] {thread_id}")
        if resume_from:
            console.print(f"[blue]Resuming from session:[/blue] {resume_from}")

    config = {"configurable": {"thread_id": thread_id}}
    max_interrupts = 10  # safety guard
    interrupt_count = 0

    # --- First invocation: run the full agent.run() ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Processing query...", total=None)
        try:
            result = run_async_callable(
                lambda: agent.run(query, config=config, resume_from=resume_from)
            )
            progress.update(task, description="[green]Query completed!")
            time.sleep(0.3)
            return result
        except HumanInputRequired as hir:
            progress.update(task, description="[yellow]Agent needs your input")
            time.sleep(0.2)
            pending_interrupts = hir.interrupts
        except Exception as e:
            progress.update(task, description="[red]Query failed!")
            console.print(f"[red]Error processing query: {e}[/red]")
            return None

    # --- Interrupt-resume loop ---
    # The spinner's `with` block has exited, so the terminal is free
    # for interactive user input.
    while pending_interrupts:
        interrupt_count += len(pending_interrupts)
        if interrupt_count > max_interrupts:
            console.print(
                "[red]Exceeded maximum number of human interrupts. Aborting.[/red]"
            )
            return None

        if len(pending_interrupts) > 1:
            if any(not pending.id for pending in pending_interrupts):
                console.print(
                    "[red]Multiple pending interrupts do not expose stable "
                    "IDs and cannot be resumed safely.[/red]"
                )
                return None

        answers = [
            _prompt_for_interrupt(pending.payload)
            for pending in pending_interrupts
        ]
        if len(pending_interrupts) == 1:
            human_answer = answers[0]
        else:
            human_answer = {
                pending.id: answer
                for pending, answer in zip(
                    pending_interrupts,
                    answers,
                    strict=True,
                )
            }

        # Resume the graph, streaming messages so tool-call parameters
        # are printed just like the initial invocation.
        resume_config = dict(config)
        resume_config["recursion_limit"] = agent.recursion_limit

        async def _resume_stream():
            """Resume an interrupted graph and stream updates until completion.

            Returns
            -------
            dict or None
                Final streamed graph state.
            """
            prev_msgs: list = []
            last_st = None
            found_interrupts = []
            async for s in agent.workflow.astream(
                Command(resume=human_answer),
                stream_mode="values",
                config=resume_config,
            ):
                if "__interrupt__" in s:
                    found_interrupts.extend(
                        normalize_interrupts(s["__interrupt__"])
                    )
                if "messages" in s and s["messages"] != prev_msgs:
                    new_message = s["messages"][-1]
                    try:
                        new_message.pretty_print()
                    except Exception:
                        pass
                    prev_msgs = s["messages"]
                last_st = s
            try:
                snapshot = agent.workflow.get_state(resume_config)
                found_interrupts.extend(
                    normalize_interrupts(
                        getattr(snapshot, "interrupts", ())
                    )
                )
                for pending_task in getattr(snapshot, "tasks", ()):
                    found_interrupts.extend(
                        normalize_interrupts(
                            getattr(pending_task, "interrupts", ())
                        )
                    )
            except Exception:
                pass
            next_interrupts = deduplicate_interrupts(found_interrupts)
            if next_interrupts:
                raise HumanInputRequired(
                    _interrupt_question(next_interrupts[0].payload),
                    payload=next_interrupts[0].payload,
                    interrupts=next_interrupts,
                )
            return last_st

        try:
            result = run_async_callable(_resume_stream)

            if result is None:
                console.print("[red]Resume produced no output.[/red]")
                return None

            return agent._finalize_completed_run(
                result,
                resume_config,
                query,
            )
        except HumanInputRequired as hir:
            agent._persist_run_state(resume_config)
            pending_interrupts = hir.interrupts
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")
            return None

    return None


def create_main_agent_session(
    agent: Any,
    *,
    thread_id: str | None = None,
    checkpoint_db: str | None = None,
):
    """Create the CLI session driver for a constructed main-agent workflow."""
    from chemgraph.agent.main_session import MainAgentSession

    metadata = agent.main_agent_metadata.model_copy(deep=True)
    if checkpoint_db is not None:
        metadata.checkpoint_backend = "AsyncSqliteSaver"
        metadata.checkpoint_db = os.path.abspath(os.path.expanduser(checkpoint_db))
    return MainAgentSession(
        agent.workflow,
        thread_id=thread_id or agent.session_id,
        recursion_limit=agent.recursion_limit,
        session_store=agent.session_store,
        session_metadata=metadata,
        on_event=_render_main_agent_event,
    )


def _render_main_agent_event(event: str, payload: dict[str, Any]) -> None:
    """Render one tagged subagent tool call during interactive execution."""
    if event != "tool_call_started":
        return
    subagent_name = payload.get("subagent_name")
    if not subagent_name:
        return
    tool_name = payload.get("tool_name") or "unknown"
    arguments = payload.get("arguments", "")
    console.print(
        f"[dim]Subagent[/dim] [bold cyan]{escape(str(subagent_name))}[/bold cyan] "
        f"[dim]→[/dim] [bold]{escape(str(tool_name))}[/bold]"
        f"({escape(str(arguments))})"
    )


def _is_tool_review(payload: Any) -> bool:
    """Return whether an interrupt is a Deep Agents tool-review request."""
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("action_requests"), list)
        and isinstance(payload.get("review_configs"), list)
    )


def _prompt_for_interrupt(payload: Any) -> Any:
    """Render one interrupt and collect its resume value."""
    if not _is_tool_review(payload):
        console.print(
            Panel(
                _interrupt_question(payload),
                title="[bold yellow]Agent needs your input[/bold yellow]",
                style="yellow",
            )
        )
        return Prompt.ask("[bold cyan]Your response[/bold cyan]")

    review_configs = {
        item.get("action_name"): item
        for item in payload["review_configs"]
        if isinstance(item, dict)
    }
    decisions = []
    for action in payload["action_requests"]:
        if not isinstance(action, dict):
            raise ValueError("Invalid Deep Agent approval request.")
        name = str(action.get("name", "unknown"))
        args = action.get("args", {})
        config = review_configs.get(name, {})
        allowed = [
            item
            for item in config.get("allowed_decisions", [])
            if item in {"approve", "reject"}
        ]
        if not allowed:
            raise ValueError(
                f"Deep Agent action {name!r} does not allow approve/reject."
            )
        console.print(
            Panel(
                f"Tool: {escape(name)}\nArguments: {escape(repr(args))}",
                title="[bold red]Deep Agent approval required[/bold red]",
                style="red",
            )
        )
        decision = Prompt.ask(
            "[bold cyan]Decision[/bold cyan]",
            choices=allowed,
            default="reject" if "reject" in allowed else allowed[0],
        )
        decisions.append({"type": decision})
    return {"decisions": decisions}


def _main_agent_failure_hint(session: Any) -> None:
    if getattr(session, "failed", False):
        console.print(
            "[yellow]The checkpoint can be resumed with the `/retry` command.[/yellow]"
        )


def _run_main_agent_operation(
    session: Any,
    operation: Any,
    *,
    progress_description: str,
    checkpoint_runtime: CheckpointRuntime | None = None,
) -> Any:
    """Run one session operation and resolve nested-graph interrupts."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(progress_description, total=None)
            result = (
                checkpoint_runtime.run(operation)
                if checkpoint_runtime is not None
                else run_async_callable(operation)
            )
    except Exception as exc:
        console.print(f"[red]Error processing main-agent session: {exc}[/red]")
        _main_agent_failure_hint(session)
        return None

    interrupt_count = 0
    while result.status == "waiting_for_user" and result.interrupts:
        interrupt_count += len(result.interrupts)
        if interrupt_count > 10:
            console.print(
                "[red]Exceeded maximum number of nested clarifications.[/red]"
            )
            return None

        answers: Any
        if len(result.interrupts) == 1:
            pending = result.interrupts[0]
            answers = _prompt_for_interrupt(pending.payload)
        else:
            answers = {}
            for pending in result.interrupts:
                answers[pending.id] = _prompt_for_interrupt(pending.payload)

        try:
            result = (
                checkpoint_runtime.run(lambda: session.resume(answers))
                if checkpoint_runtime is not None
                else run_async_callable(lambda: session.resume(answers))
            )
        except Exception as exc:
            console.print(f"[red]Error resuming main-agent session: {exc}[/red]")
            _main_agent_failure_hint(session)
            return None

    return result


def run_main_agent_query(
    session: Any,
    query: str,
    verbose: bool = False,
    checkpoint_runtime: CheckpointRuntime | None = None,
) -> Any:
    """Run one main-agent turn and resolve nested clarifications."""
    if verbose:
        console.print(f"[blue]Main-agent thread:[/blue] {session.thread_id}")

    return _run_main_agent_operation(
        session,
        lambda: session.run(query),
        progress_description="Processing main-agent turn...",
        checkpoint_runtime=checkpoint_runtime,
    )


def retry_main_agent_session(
    session: Any,
    verbose: bool = False,
    checkpoint_runtime: CheckpointRuntime | None = None,
) -> Any:
    """Retry the failed operation for one main-agent session."""
    if verbose:
        console.print(f"[blue]Main-agent thread:[/blue] {session.thread_id}")
    return _run_main_agent_operation(
        session,
        session.retry,
        progress_description="Retrying main-agent operation...",
        checkpoint_runtime=checkpoint_runtime,
    )


def restore_main_agent_session(
    session: Any,
    *,
    checkpoint_runtime: CheckpointRuntime | None = None,
) -> Any:
    """Restore one durable thread and immediately resolve pending interrupts."""
    return _run_main_agent_operation(
        session,
        session.restore,
        progress_description="Restoring main-agent thread...",
        checkpoint_runtime=checkpoint_runtime,
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def list_sessions(limit: int = 20, db_path: Optional[str] = None) -> None:
    """Display recent sessions in a formatted table.

    Parameters
    ----------
    limit : int, optional
        Maximum number of sessions to display.
    db_path : str, optional
        Path to the session SQLite database.
    """
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
    table.add_column("Children", style="white", justify="right", width=8)
    table.add_column("Status", style="magenta", width=16)
    table.add_column("Date", style="dim", width=16)

    for s in sessions:
        table.add_row(
            s.session_id,
            s.title or "[dim]Untitled[/dim]",
            s.model_name,
            s.workflow_type,
            str(s.query_count),
            str(s.message_count),
            str(s.child_run_count),
            s.status,
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
    """Display a session's full conversation.

    Parameters
    ----------
    session_id : str
        Session ID or unique session prefix.
    db_path : str, optional
        Path to the session SQLite database.
    max_content : int, optional
        Maximum number of characters displayed for each message.
    """
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
    meta_table.add_row("Status", session.status)
    meta_table.add_row("Child Runs", str(session.child_run_count))
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

    for child in session.child_runs:
        child_title = (
            f"{child.agent_name} · {child.status} · {child.run_id[:8]}"
        )
        console.print(Panel(child.delegated_task, title=child_title, style="magenta"))
        if child.error_text:
            console.print(f"  [red]{escape(child.error_text)}[/red]")
        for msg in child.messages:
            tool_label = f" ({msg.tool_name})" if msg.tool_name else ""
            console.print(f"  [bold]{msg.role}{tool_label}[/bold]: {msg.content}")


def delete_session_cmd(session_id: str, db_path: Optional[str] = None) -> None:
    """Delete a session from the database.

    Parameters
    ----------
    session_id : str
        Session ID or unique session prefix to delete.
    db_path : str, optional
        Path to the session SQLite database.
    """
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

    try:
        deleted = delete_durable_session(store, session_id)
    except Exception as exc:
        console.print(
            f"[red]Could not fully delete the session: {escape(str(exc))}[/red]"
        )
        return
    if deleted:
        console.print("[green]Session deleted.[/green]")
    else:
        console.print("[red]Failed to delete session.[/red]")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_output(content: str, output_file: str) -> None:
    """Save output to a file.

    Parameters
    ----------
    content : str
        Text content to write.
    output_file : str
        Destination file path.
    """
    try:
        with open(output_file, "w") as f:
            f.write(content)
        console.print(f"[green]Output saved to: {output_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving output: {e}[/red]")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


_INTERACTIVE_COMMANDS = {
    "clear",
    "config",
    "help",
    "history",
    "model",
    "quit",
    "resume",
    "retry",
    "show",
    "workflow",
}
_INTERACTIVE_BARE_ALIASES = {
    "clear": "clear",
    "config": "config",
    "exit": "quit",
    "help": "help",
    "history": "history",
    "q": "quit",
    "quit": "quit",
    "retry": "retry",
}
_INTERACTIVE_SLASH_ALIASES = {
    "exit": "quit",
    "q": "quit",
}


def _parse_interactive_input(query: str) -> tuple[str, str] | None:
    """Return a canonical interactive command and its argument, if any."""
    stripped = query.strip()
    bare_command = _INTERACTIVE_BARE_ALIASES.get(stripped.lower())
    if bare_command is not None:
        return bare_command, ""
    if not stripped.startswith("/"):
        return None

    command_text = stripped[1:].strip()
    if not command_text:
        return "unknown", ""
    parts = command_text.split(maxsplit=1)
    name = parts[0]
    argument = parts[1] if len(parts) == 2 else ""
    name = _INTERACTIVE_SLASH_ALIASES.get(name.lower(), name.lower())
    if name not in _INTERACTIVE_COMMANDS:
        return "unknown", name
    return name, argument.strip()


def interactive_mode(
    model: str = "gpt-4o-mini",
    workflow: str = "single_agent",
    structured: bool = False,
    return_option: str = "state",
    generate_report: bool = True,
    human_supervised: bool = False,
    recursion_limit: int = 20,
    base_url: Optional[str] = None,
    argo_user: Optional[str] = None,
    verbose: bool = False,
    tools: Optional[list] = None,
    enable_deepagent: bool = False,
    deepagent_workspace: str | None = None,
    deepagent_skills: Sequence[str] | None = None,
    deepagent_auto_approve: bool = False,
    checkpoint_db: str | None = None,
    resume_session: str | None = None,
) -> None:
    """Start interactive REPL mode for ChemGraph CLI.

    Accepts the same configuration parameters as a normal run so that
    ``--config`` and CLI flags are honoured when entering interactive
    mode.

    Parameters
    ----------
    model : str, optional
        Initial model selection.
    workflow : str, optional
        Initial workflow selection.
    structured : bool, optional
        Whether structured output is requested.
    return_option : str, optional
        Agent return mode.
    generate_report : bool, optional
        Whether report generation is enabled.
    human_supervised : bool, optional
        Whether human supervision tools are enabled.
    recursion_limit : int, optional
        LangGraph recursion limit.
    base_url : str, optional
        Custom model endpoint URL.
    argo_user : str, optional
        Argo username for Argo-hosted models.
    verbose : bool, optional
        Whether to print diagnostic output.
    tools : list, optional
        Custom tool list for MCP-backed workflows.
    enable_deepagent : bool, optional
        Whether to add the experimental workspace Deep Agent whenever the
        selected workflow is ``main_agent``.
    deepagent_workspace : str, optional
        Local workspace used by the experimental host-shell backend.
    deepagent_skills : sequence of str, optional
        Ordered backend-relative Agent Skills directories.
    deepagent_auto_approve : bool, optional
        Disable action approvals for a standalone Deep Agent. The CLI rejects
        this option in interactive mode.
    """
    console.print(create_banner())
    console.print("[bold green]Welcome to ChemGraph Interactive Mode![/bold green]")
    console.print(
        "Type your queries and get AI-powered computational chemistry insights."
    )
    console.print(
        "[dim]Type '/quit' to exit or '/help' for commands. Exact bare "
        "aliases such as 'quit' and 'help' also work.[/dim]\n"
    )

    checkpoint_runtime: CheckpointRuntime | None = None
    checkpoint_saver = None
    restored_thread_id: str | None = None
    stored_graph_config = None
    reasoning_effort: str | None = None
    max_retries = 1
    terminal_tool_names: tuple[str, ...] = ()
    if resume_session:
        resolved = SessionStore().get_session_metadata(resume_session)
        if resolved is None:
            console.print(
                f"[red]Durable main-agent session '{resume_session}' was not found.[/red]"
            )
            return
        if resolved[1] is None:
            console.print(
                f"[red]Session '{resume_session}' exists but is not a durable "
                "main-agent thread.[/red]"
            )
            return
        restored_thread_id, stored_metadata = resolved
        if stored_metadata.checkpoint_backend == "memory":
            console.print(
                f"[red]Session '{resume_session}' used a process-local checkpoint "
                "and cannot be restored after its owner process exits.[/red]"
            )
            return
        if stored_metadata.checkpoint_backend not in {None, "AsyncSqliteSaver"}:
            console.print(
                f"[red]Session '{resume_session}' uses a caller-owned external "
                "checkpointer and cannot be restored by the CLI.[/red]"
            )
            return
        stored_graph_config = stored_metadata.graph_config
        model = stored_graph_config.model_name
        workflow = "main_agent"
        recursion_limit = stored_graph_config.recursion_limit
        structured = stored_graph_config.structured_output
        generate_report = stored_graph_config.generate_report
        human_supervised = stored_graph_config.human_supervised
        enable_deepagent = stored_graph_config.enable_deepagent
        deepagent_workspace = stored_graph_config.deepagent_workspace
        deepagent_skills = stored_graph_config.deepagent_skills
        reasoning_effort = stored_graph_config.reasoning_effort
        max_retries = stored_graph_config.max_retries
        terminal_tool_names = stored_graph_config.terminal_tool_names
        checkpoint_db = stored_metadata.checkpoint_db or checkpoint_db
    else:
        # Allow the user to override model/workflow at startup.
        model = Prompt.ask(
            "Select model (or type a custom model ID)", default=model
        )
        workflow = Prompt.ask(
            "Select workflow",
            choices=ALL_WORKFLOW_TYPES,
            default=resolve_workflow(workflow),
        )

    if workflow == "main_agent":
        checkpoint_runtime = CheckpointRuntime()
        try:
            checkpoint_saver = checkpoint_runtime.open_sqlite(
                checkpoint_db or DEFAULT_CHECKPOINT_DB
            )
        except Exception as exc:
            checkpoint_runtime.close()
            console.print(f"[red]Could not open checkpoint database: {exc}[/red]")
            return

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
        human_supervised=human_supervised,
        tools=tools,
        enable_deepagent=enable_deepagent and workflow == "main_agent",
        deepagent_workspace=(
            deepagent_workspace
            if workflow == "deep_agent"
            or (enable_deepagent and workflow == "main_agent")
            else None
        ),
        deepagent_skills=(
            deepagent_skills
            if workflow == "deep_agent"
            or (enable_deepagent and workflow == "main_agent")
            else None
        ),
        deepagent_auto_approve=(
            deepagent_auto_approve and workflow == "deep_agent"
        ),
        checkpointer=checkpoint_saver,
        reasoning_effort=reasoning_effort,
        max_retries=max_retries,
        terminal_tool_names=terminal_tool_names,
    )
    if not agent:
        if checkpoint_runtime is not None:
            checkpoint_runtime.close()
        return

    main_session = (
        create_main_agent_session(
            agent,
            thread_id=restored_thread_id,
            checkpoint_db=checkpoint_db or DEFAULT_CHECKPOINT_DB,
        )
        if workflow == "main_agent"
        else None
    )
    standalone_thread_id = _next_thread_id() if workflow == "deep_agent" else None

    if restored_thread_id and main_session is not None:
        result = restore_main_agent_session(
            main_session,
            checkpoint_runtime=checkpoint_runtime,
        )
        if result is None:
            if checkpoint_runtime is not None:
                checkpoint_runtime.close()
            return
        if result.status == "failed":
            console.print(
                "[yellow]The restored operation can be continued with /retry.[/yellow]"
            )
        elif result.assistant_response:
            format_response(result, verbose=verbose)

    console.print(
        "[green]Ready! You can now ask computational chemistry questions.[/green]\n"
    )

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]ChemGraph[/bold cyan]")
            parsed_command = _parse_interactive_input(query)

            if parsed_command is None:
                command = None
                argument = ""
            else:
                command, argument = parsed_command

            if command == "unknown":
                label = f"/{argument}" if argument else "/"
                console.print(
                    f"[red]Unknown interactive command: {label}[/red]"
                )
                console.print("[dim]Type /help to see available commands.[/dim]")
                continue
            if command == "quit":
                if argument:
                    console.print("[red]Usage: /quit[/red]")
                    continue
                console.print("[yellow]Goodbye![/yellow]")
                if checkpoint_runtime is not None:
                    checkpoint_runtime.close()
                break
            elif command == "help":
                if argument:
                    console.print("[red]Usage: /help[/red]")
                    continue
                console.print(
                    Panel(
                        """
Available commands:
  /quit              Exit interactive mode
  /help              Show this help message
  /clear             Clear screen
  /config            Show current configuration
  /model <name>      Change model
  /workflow <type>   Change workflow type

Session commands:
  /history           List recent sessions
  /show <id>         Show a session's conversation
  /resume <id>       Resume from a previous session
  /retry             Retry a failed main_agent operation

Exact bare aliases for commands without arguments remain supported. Commands
with arguments require the leading slash so prompts beginning with words such
as "show", "model", or "workflow" are sent to the agent.

main_agent keeps one durable checkpointed thread. `/resume <id>` restores
completed, interrupted, or retryable threads. Nested chemistry workers may
pause to request input.

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
            elif command == "clear":
                if argument:
                    console.print("[red]Usage: /clear[/red]")
                    continue
                console.clear()
                continue
            elif command == "config":
                if argument:
                    console.print("[red]Usage: /config[/red]")
                    continue
                console.print(f"Model: {model}")
                console.print(f"Workflow: {workflow}")
                console.print(
                    "Deep Agent: "
                    f"{'enabled' if workflow == 'deep_agent' or (enable_deepagent and workflow == 'main_agent') else 'disabled'}"
                )
                if main_session is not None:
                    console.print(f"Thread ID: {main_session.thread_id}")
                    console.print(f"Failed: {main_session.failed}")
                elif hasattr(agent, "session_id"):
                    console.print(f"Session ID: {agent.session_id}")
                continue
            elif command == "history":
                if argument:
                    console.print("[red]Usage: /history[/red]")
                    continue
                list_sessions()
                continue
            elif command == "show":
                if not argument:
                    console.print("[red]Usage: /show <session_id>[/red]")
                    continue
                show_session(argument)
                continue
            elif command == "resume":
                if not argument:
                    console.print("[red]Usage: /resume <session_id>[/red]")
                    continue
                target = SessionStore().get_session_metadata(argument)
                if target is None and main_session is not None:
                    console.print(f"[red]Session '{argument}' was not found.[/red]")
                    continue
                if target is not None and target[1] is not None:
                    target_id, target_metadata = target
                    if target_metadata.checkpoint_backend == "memory":
                        console.print(
                            "[red]The selected session used a process-local "
                            "checkpoint and is no longer restorable.[/red]"
                        )
                        continue
                    if target_metadata.checkpoint_backend not in {
                        None,
                        "AsyncSqliteSaver",
                    }:
                        console.print(
                            "[red]The selected session uses a caller-owned external "
                            "checkpointer and cannot be restored by the CLI.[/red]"
                        )
                        continue
                    target_config = target_metadata.graph_config
                    candidate_db = (
                        target_metadata.checkpoint_db or DEFAULT_CHECKPOINT_DB
                    )
                    previous_db = (
                        checkpoint_db or DEFAULT_CHECKPOINT_DB
                        if checkpoint_runtime is not None
                        else None
                    )
                    candidate_runtime = checkpoint_runtime or CheckpointRuntime()
                    try:
                        candidate_saver = candidate_runtime.open_sqlite(candidate_db)
                        candidate_agent = initialize_agent(
                            target_config.model_name,
                            "main_agent",
                            target_config.structured_output,
                            return_option,
                            target_config.generate_report,
                            target_config.recursion_limit,
                            base_url=base_url,
                            argo_user=argo_user,
                            verbose=verbose,
                            human_supervised=target_config.human_supervised,
                            tools=tools,
                            enable_deepagent=target_config.enable_deepagent,
                            deepagent_workspace=target_config.deepagent_workspace,
                            deepagent_skills=target_config.deepagent_skills,
                            checkpointer=candidate_saver,
                            reasoning_effort=target_config.reasoning_effort,
                            max_retries=target_config.max_retries,
                            terminal_tool_names=target_config.terminal_tool_names,
                        )
                        if candidate_agent is None:
                            raise RuntimeError("Could not recreate the stored agent.")
                        candidate_session = create_main_agent_session(
                            candidate_agent,
                            thread_id=target_id,
                            checkpoint_db=candidate_db,
                        )
                        restored = restore_main_agent_session(
                            candidate_session,
                            checkpoint_runtime=candidate_runtime,
                        )
                        if restored is None:
                            raise RuntimeError("The stored thread could not be restored.")
                    except Exception as exc:
                        if checkpoint_runtime is None:
                            candidate_runtime.close()
                        elif (
                            previous_db is not None
                            and os.path.abspath(os.path.expanduser(candidate_db))
                            != os.path.abspath(os.path.expanduser(previous_db))
                        ):
                            try:
                                candidate_runtime.close_sqlite(candidate_db)
                            except Exception:
                                pass
                        console.print(f"[red]Could not restore session: {exc}[/red]")
                        continue
                    checkpoint_runtime = candidate_runtime
                    checkpoint_saver = candidate_saver
                    agent = candidate_agent
                    main_session = candidate_session
                    model = target_config.model_name
                    workflow = "main_agent"
                    checkpoint_db = candidate_db
                    structured = target_config.structured_output
                    generate_report = target_config.generate_report
                    human_supervised = target_config.human_supervised
                    recursion_limit = target_config.recursion_limit
                    reasoning_effort = target_config.reasoning_effort
                    max_retries = target_config.max_retries
                    terminal_tool_names = target_config.terminal_tool_names
                    enable_deepagent = target_config.enable_deepagent
                    deepagent_workspace = target_config.deepagent_workspace
                    deepagent_skills = target_config.deepagent_skills
                    if (
                        previous_db is not None
                        and os.path.abspath(os.path.expanduser(previous_db))
                        != os.path.abspath(os.path.expanduser(candidate_db))
                    ):
                        try:
                            checkpoint_runtime.close_sqlite(previous_db)
                        except Exception as exc:
                            console.print(
                                "[yellow]Could not release the previous checkpoint "
                                f"database: {escape(str(exc))}[/yellow]"
                            )
                    if restored.status == "failed":
                        console.print(
                            "[yellow]The restored operation can be continued with "
                            "/retry.[/yellow]"
                        )
                    elif restored.assistant_response:
                        format_response(restored, verbose=verbose)
                    continue
                if main_session is not None:
                    console.print(
                        "[red]The selected session is not a durable main-agent thread.[/red]"
                    )
                    continue
                resume_query = Prompt.ask(
                    "[bold cyan]Enter query to continue with[/bold cyan]"
                )
                if resume_query.strip():
                    run_options = {}
                    if standalone_thread_id is not None:
                        run_options["thread_id"] = standalone_thread_id
                    result = run_query(
                        agent,
                        resume_query,
                        verbose=verbose,
                        resume_from=argument,
                        **run_options,
                    )
                    if result:
                        format_response(result, verbose=verbose)
                continue
            elif command == "retry":
                if argument:
                    console.print("[red]Usage: /retry[/red]")
                    continue
                if main_session is None:
                    console.print(
                        "[yellow]/retry is available only for the main_agent "
                        "workflow.[/yellow]"
                    )
                    continue
                if checkpoint_runtime is None:
                    result = retry_main_agent_session(main_session, verbose=verbose)
                else:
                    result = retry_main_agent_session(
                        main_session,
                        verbose=verbose,
                        checkpoint_runtime=checkpoint_runtime,
                    )
                if result:
                    format_response(result, verbose=verbose)
                    console.print(f"[dim]Thread: {main_session.thread_id}[/dim]")
                continue
            elif command == "model":
                if not argument:
                    console.print("[red]Usage: /model <name>[/red]")
                    continue
                new_model = argument
                new_reasoning_effort = (
                    reasoning_effort
                    if new_model in MODELS_WITH_REASONING_EFFORT
                    else None
                )
                new_agent = initialize_agent(
                    new_model,
                    workflow,
                    structured,
                    return_option,
                    generate_report,
                    recursion_limit,
                    base_url=base_url,
                    argo_user=argo_user,
                    human_supervised=human_supervised,
                    tools=tools,
                    enable_deepagent=enable_deepagent and workflow == "main_agent",
                    deepagent_workspace=(
                        deepagent_workspace
                        if workflow == "deep_agent"
                        or (enable_deepagent and workflow == "main_agent")
                        else None
                    ),
                    deepagent_skills=(
                        deepagent_skills
                        if workflow == "deep_agent"
                        or (enable_deepagent and workflow == "main_agent")
                        else None
                    ),
                    deepagent_auto_approve=(
                        deepagent_auto_approve and workflow == "deep_agent"
                    ),
                    checkpointer=(checkpoint_saver if workflow == "main_agent" else None),
                    reasoning_effort=new_reasoning_effort,
                    max_retries=max_retries,
                    terminal_tool_names=terminal_tool_names,
                )
                if new_agent:
                    if reasoning_effort is not None and new_reasoning_effort is None:
                        console.print(
                            "[yellow]Reasoning effort was reset because the new "
                            "model does not support it.[/yellow]"
                        )
                    model = new_model
                    reasoning_effort = new_reasoning_effort
                    agent = new_agent
                    main_session = (
                        create_main_agent_session(
                            agent,
                            checkpoint_db=checkpoint_db or DEFAULT_CHECKPOINT_DB,
                        )
                        if workflow == "main_agent"
                        else None
                    )
                    standalone_thread_id = (
                        _next_thread_id() if workflow == "deep_agent" else None
                    )
                    console.print(f"[green]Model changed to: {model}[/green]")
                continue
            elif command == "workflow":
                if not argument:
                    console.print("[red]Usage: /workflow <type>[/red]")
                    continue
                new_workflow = resolve_workflow(argument)
                if new_workflow in ALL_WORKFLOW_TYPES:
                    if new_workflow == "main_agent" and checkpoint_runtime is None:
                        candidate_runtime = None
                        try:
                            candidate_runtime = CheckpointRuntime()
                            candidate_saver = candidate_runtime.open_sqlite(
                                checkpoint_db or DEFAULT_CHECKPOINT_DB
                            )
                        except Exception as exc:
                            if candidate_runtime is not None:
                                candidate_runtime.close()
                            checkpoint_runtime = None
                            checkpoint_saver = None
                            console.print(
                                "[red]Could not open checkpoint database: "
                                f"{exc}[/red]"
                            )
                            continue
                        checkpoint_runtime = candidate_runtime
                        checkpoint_saver = candidate_saver
                    new_agent = initialize_agent(
                        model,
                        new_workflow,
                        structured,
                        return_option,
                        generate_report,
                        recursion_limit,
                        base_url=base_url,
                        argo_user=argo_user,
                        human_supervised=human_supervised,
                        tools=tools,
                        enable_deepagent=(
                            enable_deepagent and new_workflow == "main_agent"
                        ),
                        deepagent_workspace=(
                            deepagent_workspace
                            if new_workflow == "deep_agent"
                            or (enable_deepagent and new_workflow == "main_agent")
                            else None
                        ),
                        deepagent_skills=(
                            deepagent_skills
                            if new_workflow == "deep_agent"
                            or (enable_deepagent and new_workflow == "main_agent")
                            else None
                        ),
                        deepagent_auto_approve=(
                            deepagent_auto_approve and new_workflow == "deep_agent"
                        ),
                        checkpointer=(
                            checkpoint_saver if new_workflow == "main_agent" else None
                        ),
                        reasoning_effort=reasoning_effort,
                        max_retries=max_retries,
                        terminal_tool_names=terminal_tool_names,
                    )
                    if new_agent:
                        workflow = new_workflow
                        agent = new_agent
                        main_session = (
                            create_main_agent_session(
                                agent,
                                checkpoint_db=checkpoint_db or DEFAULT_CHECKPOINT_DB,
                            )
                            if workflow == "main_agent"
                            else None
                        )
                        standalone_thread_id = (
                            _next_thread_id()
                            if workflow == "deep_agent"
                            else None
                        )
                        console.print(
                            f"[green]Workflow changed to: {workflow}[/green]"
                        )
                else:
                    console.print(f"[red]Invalid workflow: {new_workflow}[/red]")
                    console.print(
                        f"[dim]Available: {', '.join(ALL_WORKFLOW_TYPES)}[/dim]"
                    )
                continue

            if main_session is not None:
                if checkpoint_runtime is None:
                    result = run_main_agent_query(
                        main_session,
                        query,
                        verbose=verbose,
                    )
                else:
                    result = run_main_agent_query(
                        main_session,
                        query,
                        verbose=verbose,
                        checkpoint_runtime=checkpoint_runtime,
                    )
            else:
                # Deep Agent keeps process-local context for this REPL; other
                # standalone workflows use a fresh thread for each query.
                run_options = {}
                if standalone_thread_id is not None:
                    run_options["thread_id"] = standalone_thread_id
                result = run_query(
                    agent,
                    query,
                    verbose=verbose,
                    **run_options,
                )
            if result:
                format_response(result, verbose=verbose)
                if main_session is not None:
                    console.print(f"[dim]Thread: {main_session.thread_id}[/dim]")
                elif hasattr(agent, "session_id") and agent.session_id:
                    console.print(f"[dim]Session: {agent.session_id}[/dim]")

        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            if checkpoint_runtime is not None:
                checkpoint_runtime.close()
            break
        except KeyboardInterrupt:
            console.print(
                "\n[yellow]Interrupted. Type '/quit' to exit.[/yellow]"
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
