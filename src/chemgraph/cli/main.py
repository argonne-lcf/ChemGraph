"""Argument parsing and main entry point for the ChemGraph CLI.

Supports three usage styles:

1. **Legacy** (no subcommand) -- ``chemgraph -q "..." -m gpt-4o``
2. **Subcommand** -- ``chemgraph run ...``, ``chemgraph eval ...``,
   ``chemgraph session ...``, ``chemgraph models``
3. **Standalone eval** -- ``chemgraph-eval`` via its own entry point.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import toml

from chemgraph.models.supported_models import all_supported_models
from chemgraph.utils.config_utils import (
    flatten_config,
    get_argo_user_from_flat_config,
    get_base_url_for_model_from_flat_config,
)

from chemgraph.cli.commands import (
    ALL_WORKFLOW_TYPES,
    WORKFLOW_ALIASES,
    resolve_workflow,
    delete_session_cmd,
    initialize_agent,
    interactive_mode,
    list_sessions,
    run_query,
    save_output,
    show_session,
)
from chemgraph.cli.formatting import (
    check_api_keys_status,
    console,
    create_banner,
    format_response,
    list_models,
)


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

# Workflow choices exposed to the user.  We include common aliases
# (e.g. ``python_repl``) so that users don't have to know the
# internal ``python_relp`` name.
_WORKFLOW_CHOICES = sorted(set(ALL_WORKFLOW_TYPES) | set(WORKFLOW_ALIASES.keys()))


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add query/run-specific arguments to *parser*.

    Used by both the ``run`` subcommand and the legacy (no subcommand)
    argument parser for backward compatibility.
    """
    parser.add_argument(
        "-q", "--query", type=str, help="The computational chemistry query to execute"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        choices=_WORKFLOW_CHOICES,
        default="single_agent",
        help="Workflow type (default: single_agent)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        choices=["state", "last_message"],
        default="state",
        help="Output format (default: state)",
    )
    parser.add_argument(
        "-s", "--structured", action="store_true", help="Use structured output format"
    )
    parser.add_argument(
        "-r", "--report", action="store_true", help="Generate detailed report"
    )
    parser.add_argument(
        "--human-supervised",
        action="store_true",
        help="Enable the ask_human tool for human-in-the-loop interaction",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=20,
        help="Recursion limit for agent workflows (default: 20)",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive mode"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List all available models"
    )
    parser.add_argument(
        "--check-keys", action="store_true", help="Check API key availability"
    )
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    parser.add_argument("--output-file", type=str, help="Save output to file")
    parser.add_argument("--config", type=str, help="Load configuration from TOML file")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for the LLM API endpoint (overrides config file)",
    )
    parser.add_argument(
        "--mcp-url",
        type=str,
        default=None,
        help="MCP server URL for streamable_http transport (e.g. http://localhost:9003/mcp/)",
    )
    parser.add_argument(
        "--mcp-command",
        type=str,
        default=None,
        help="MCP server command for stdio transport (e.g. 'python -m chemgraph.mcp.mcp_tools')",
    )
    parser.add_argument(
        "--mcp-server-name",
        type=str,
        default="ChemGraph General Tools",
        help="Display name for the MCP server connection (default: 'ChemGraph General Tools')",
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="chemgraph",
        description="ChemGraph CLI - AI Agents for Computational Chemistry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy style (still works)
  %(prog)s -q "What is the SMILES string for water?"
  %(prog)s --interactive
  %(prog)s --list-models

  # Subcommand style
  %(prog)s run -q "Optimize water geometry" -m gpt-4o
  %(prog)s eval --profile quick --models gpt-4o-mini --config config.toml
  %(prog)s eval --models gpt-4o --dataset ground_truth.json
  %(prog)s session list
  %(prog)s session show a3b2
  %(prog)s models
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # ---- "run" subcommand ------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single query or start interactive mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(run_parser)

    # ---- "eval" subcommand -----------------------------------------------
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation benchmarks against ground-truth datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Import here to avoid circular imports at module level
    from chemgraph.eval.cli import add_eval_args

    add_eval_args(eval_parser)

    # ---- "session" subcommand --------------------------------------------
    session_parser = subparsers.add_parser(
        "session",
        help="Manage conversation sessions.",
    )
    session_sub = session_parser.add_subparsers(dest="session_command")

    session_sub.add_parser("list", help="List recent sessions.")

    show_parser = session_sub.add_parser("show", help="Show a session's conversation.")
    show_parser.add_argument("id", help="Session ID (prefix matching supported).")

    delete_parser = session_sub.add_parser("delete", help="Delete a session.")
    delete_parser.add_argument("id", help="Session ID to delete.")

    # ---- "models" subcommand ---------------------------------------------
    subparsers.add_parser("models", help="List all available LLM models.")

    # ---- "dashboard" subcommands ----------------------------------------
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Serve the ChemGraph dashboard for a run directory.",
    )
    dashboard_parser.add_argument(
        "dashboard_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to chemgraph.academy.dashboard.",
    )

    dashboard_run_parser = subparsers.add_parser(
        "dashboard-run",
        help="Run a local ChemGraph workflow and write dashboard artifacts.",
    )
    dashboard_run_parser.add_argument(
        "dashboard_run_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to chemgraph.observability.local_dashboard_run.",
    )

    # ---- "academy" subcommand -------------------------------------------
    academy_parser = subparsers.add_parser(
        "academy",
        help="Run and inspect Academy-backed ChemGraph agent campaigns.",
    )
    academy_sub = academy_parser.add_subparsers(dest="academy_command")

    daemon_parser = academy_sub.add_parser(
        "mpi-daemon",
        help="Run one ChemGraph Academy agent daemon inside mpiexec.",
    )
    daemon_parser.add_argument(
        "daemon_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to chemgraph.academy.runtime.daemon.",
    )

    compute_parser = academy_sub.add_parser(
        "run-compute",
        help="Run a profile-backed ChemGraph Academy campaign in this allocation.",
    )
    compute_parser.add_argument(
        "compute_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to chemgraph.academy.runtime.compute_launcher.",
    )

    dashboard_parser = academy_sub.add_parser(
        "dashboard",
        help="Start the local dashboard launcher for a ChemGraph Academy run.",
    )
    dashboard_parser.add_argument(
        "dashboard_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to chemgraph.academy.runtime.dashboard_launcher.",
    )

    academy_sub.add_parser(
        "campaigns",
        help="List built-in ChemGraph Academy campaign specs.",
    )
    academy_sub.add_parser(
        "logical-agent-configs",
        help="List built-in ChemGraph Academy logical-agent prompt configs.",
    )

    # ---- Legacy fallback args -------------------------------------------
    # Also add run args to the top-level parser so that
    # `chemgraph -q "..."` keeps working without a subcommand.
    _add_run_args(parser)

    return parser


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_file: str) -> Dict[str, Any]:
    """Load and flatten a TOML configuration file.

    Merges missing keys from a sensible default so that partial config
    files don't crash the CLI (addresses Bug 4 -- parity with the
    Streamlit config loader).
    """
    try:
        with open(config_file, "r") as f:
            raw_config = toml.load(f)
        console.print(f"[green]Configuration loaded from {config_file}[/green]")

        # Merge defaults for required sections so partial configs work.
        _DEFAULT_SECTIONS = {
            "general": {
                "model": "gpt-4o-mini",
                "workflow": "single_agent",
                "output": "state",
                "structured": False,
                "report": False,
                "thread": 1,
                "recursion_limit": 20,
                "human_supervised": False,
                "verbose": False,
            },
            "api": {},
            "chemistry": {},
            "output": {},
            "mcp": {},
        }

        for section, defaults in _DEFAULT_SECTIONS.items():
            if section not in raw_config:
                raw_config[section] = defaults
            elif isinstance(defaults, dict):
                for key, value in defaults.items():
                    raw_config[section].setdefault(key, value)

        flat = flatten_config(raw_config)

        # Inject MCP config keys (not handled by flatten_config).
        if "mcp" in raw_config and isinstance(raw_config["mcp"], dict):
            for key, value in raw_config["mcp"].items():
                flat[f"mcp_{key}"] = value

        return flat

    except FileNotFoundError:
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        sys.exit(1)
    except toml.TomlDecodeError as e:
        console.print(f"[red]Invalid TOML in configuration file: {e}[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _handle_run(args: argparse.Namespace) -> None:
    """Handle the ``run`` subcommand (and legacy no-subcommand mode)."""
    # Handle special commands first
    if getattr(args, "list_models", False):
        list_models()
        return

    if getattr(args, "check_keys", False):
        check_api_keys_status()
        return

    if getattr(args, "list_sessions", False):
        list_sessions()
        return

    if getattr(args, "show_session", None):
        show_session(args.show_session)
        return

    if getattr(args, "delete_session", None):
        delete_session_cmd(args.delete_session)
        return

    # Load configuration if specified
    config: Dict[str, Any] = {}
    if args.config:
        config = load_config(args.config)
        # Override args with config values (only when the user hasn't
        # explicitly set them on the command line).
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
        # Honour config recursion_limit unless user gave explicit flag.
        if "recursion_limit" in config and "--recursion-limit" not in sys.argv:
            args.recursion_limit = config["recursion_limit"]

    # ---- Configure logging verbosity --------------------------------
    import logging as _logging

    from chemgraph.utils.logging_config import configure_logging

    # Start from config baseline (default: WARNING = quiet).
    _log_level_name = config.get("logging_level", "WARNING").upper() if config else "WARNING"
    _log_level = getattr(_logging, _log_level_name, _logging.WARNING)

    # CLI -v / -vv overrides the config value.
    if args.verbose >= 2:
        _log_level = _logging.DEBUG
    elif args.verbose >= 1:
        _log_level = _logging.INFO

    configure_logging(_log_level)

    base_url = args.base_url or (
        get_base_url_for_model_from_flat_config(args.model, config) if config else None
    )
    argo_user = get_argo_user_from_flat_config(config) if config else None

    # Resolve workflow alias (e.g. python_repl -> python_relp)
    args.workflow = resolve_workflow(args.workflow)

    # ---- MCP tool loading ----------------------------------------------
    mcp_tools = None
    mcp_url = getattr(args, "mcp_url", None) or config.get("mcp_url")
    mcp_command = getattr(args, "mcp_command", None) or config.get("mcp_command")
    mcp_server_name = (
        getattr(args, "mcp_server_name", None)
        or config.get("mcp_server_name", "ChemGraph General Tools")
    )

    if mcp_url or mcp_command:
        from chemgraph.cli.mcp_utils import load_mcp_tools_from_config

        mcp_tools = load_mcp_tools_from_config(
            url=mcp_url,
            command=mcp_command,
            server_name=mcp_server_name,
            verbose=(args.verbose > 0),
        )
        if mcp_tools is None:
            sys.exit(1)

    if getattr(args, "interactive", False):
        interactive_mode(
            model=args.model,
            workflow=args.workflow,
            structured=args.structured,
            return_option=args.output,
            generate_report=args.report,
            human_supervised=args.human_supervised,
            recursion_limit=args.recursion_limit,
            base_url=base_url,
            argo_user=argo_user,
            verbose=(args.verbose > 0),
            tools=mcp_tools,
        )
        return

    if args.model not in all_supported_models:
        console.print(
            f"[yellow]Using custom model ID: {args.model} (not in curated list)[/yellow]"
        )

    # Require query for non-interactive mode
    if not args.query:
        console.print("[red]Query is required. Use -q or --query to specify.[/red]")
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
        verbose=(args.verbose > 0),
        human_supervised=args.human_supervised,
        tools=mcp_tools,
    )

    if not agent:
        sys.exit(1)

    # Execute query
    console.print(f"[bold blue]Query:[/bold blue] {args.query}")
    if args.resume:
        console.print(f"[bold blue]Resuming from:[/bold blue] {args.resume}")
    result = run_query(
        agent, args.query, verbose=(args.verbose > 0), resume_from=args.resume
    )

    if result:
        format_response(result, verbose=(args.verbose > 0))

        # Save output if requested
        if args.output_file:
            output_content = str(result)
            save_output(output_content, args.output_file)

    if hasattr(agent, "session_id") and agent.session_id:
        console.print(
            f"\n[dim]Session: {agent.session_id}"
            f" | Resume: chemgraph -q \"<query>\" --resume {agent.session_id}[/dim]"
        )
    console.print("[dim]Thank you for using ChemGraph CLI![/dim]")


def _strip_remainder_separator(args: list[str]) -> list[str]:
    """Remove an optional argparse remainder separator."""
    if args and args[0] == "--":
        return args[1:]
    return args


def _run_module_main(module_name: str, argv: list[str]) -> None:
    """Run a module-level main() with forwarded command-line arguments."""
    import importlib

    module = importlib.import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [f"chemgraph {module_name.rsplit('.', 1)[-1]}", *argv]
        code = module.main()
    finally:
        sys.argv = old_argv
    if isinstance(code, int) and code:
        sys.exit(code)


def _handle_academy(args: argparse.Namespace) -> None:
    """Handle Academy-backed ChemGraph campaign commands."""
    command = getattr(args, "academy_command", None)
    if command == "mpi-daemon":
        _run_module_main(
            "chemgraph.academy.runtime.daemon",
            _strip_remainder_separator(args.daemon_args),
        )
        return
    if command == "dashboard":
        _run_module_main(
            "chemgraph.academy.runtime.dashboard_launcher",
            _strip_remainder_separator(args.dashboard_args),
        )
        return
    if command == "run-compute":
        from chemgraph.academy.runtime.compute_launcher import main as compute_main

        code = compute_main(_strip_remainder_separator(args.compute_args))
        if code:
            sys.exit(code)
        return
    if command == "campaigns":
        from chemgraph.academy.examples import list_builtin_campaigns

        for name in list_builtin_campaigns():
            console.print(name)
        return
    if command == "logical-agent-configs":
        from chemgraph.academy.examples import list_builtin_logical_agent_configs

        for name in list_builtin_logical_agent_configs():
            console.print(name)
        return
    console.print(
        "Usage: chemgraph academy "
        "{mpi-daemon,run-compute,dashboard,campaigns,"
        "logical-agent-configs}.",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main CLI entry point.

    Dispatches to the appropriate subcommand handler, or falls back
    to the legacy behaviour when no subcommand is given.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.command == "eval":
        from chemgraph.eval.cli import run_eval

        run_eval(args)

    elif args.command == "session":
        sc = getattr(args, "session_command", None)
        if sc == "list":
            list_sessions()
        elif sc == "show":
            show_session(args.id)
        elif sc == "delete":
            delete_session_cmd(args.id)
        else:
            console.print(
                "Usage: chemgraph session {list,show,delete}. Use --help for details."
            )

    elif args.command == "models":
        list_models()

    elif args.command == "dashboard":
        _run_module_main(
            "chemgraph.academy.dashboard",
            _strip_remainder_separator(args.dashboard_args),
        )

    elif args.command == "dashboard-run":
        _run_module_main(
            "chemgraph.observability.local_dashboard_run",
            _strip_remainder_separator(args.dashboard_run_args),
        )

    elif args.command == "academy":
        _handle_academy(args)

    elif args.command == "run":
        _handle_run(args)

    else:
        # No subcommand given -- legacy behaviour.
        _handle_run(args)


if __name__ == "__main__":
    main()
