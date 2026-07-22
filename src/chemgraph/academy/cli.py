"""``swarm`` command-line entry point.

Registered as the ``swarm`` script in pyproject.toml. Dispatches to
the per-subcommand main() functions inside swarm.runtime.*, matching
the surface previously exposed as ``chemgraph academy <subcommand>``
before the 2026-07-07 split-out.

Kept intentionally thin: this file owns argparse only; every
subcommand's real logic lives in its module's own main().
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence


def _strip_remainder_separator(args: Sequence[str]) -> list[str]:
    """Remove an optional argparse ``--`` remainder separator."""
    args = list(args)
    if args and args[0] == "--":
        return args[1:]
    return args


def _run_module_main(module_name: str, argv: Sequence[str]) -> None:
    """Run a module-level main() with forwarded argv."""
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [f"swarm {module_name.rsplit('.', 1)[-1]}", *argv]
        code = module.main()
    finally:
        sys.argv = old_argv
    if isinstance(code, int) and code:
        sys.exit(code)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swarm",
        description=(
            "Federated multi-agent operating + authoring surface. "
            "Every subcommand forwards its trailing argv to the "
            "underlying module."
        ),
    )
    sub = p.add_subparsers(dest="command")

    daemon = sub.add_parser("mpi-daemon", help="Run one agent daemon inside mpiexec.")
    daemon.add_argument("daemon_args", nargs=argparse.REMAINDER)

    compute = sub.add_parser("run-compute", help="Run a campaign in this allocation.")
    compute.add_argument("compute_args", nargs=argparse.REMAINDER)

    spawn = sub.add_parser(
        "spawn-site",
        help=(
            "Launch one site of a federated campaign. Like run-compute "
            "but restricted to --agents."
        ),
    )
    spawn.add_argument("spawn_site_args", nargs=argparse.REMAINDER)

    launch = sub.add_parser(
        "launch",
        help="Drive a federated launch from the operator's laptop.",
    )
    launch.add_argument("launch_args", nargs=argparse.REMAINDER)

    inject = sub.add_parser(
        "inject",
        help=(
            "Send an operator message to an agent's mailbox. Used for "
            "both campaign kickoff and mid-run nudges."
        ),
    )
    inject.add_argument("inject_args", nargs=argparse.REMAINDER)

    dashboard = sub.add_parser(
        "dashboard",
        help="Start the local dashboard launcher for a run.",
    )
    dashboard.add_argument("dashboard_args", nargs=argparse.REMAINDER)

    sub.add_parser("campaigns", help="List shipped campaign specs.")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    command = args.command

    if command == "mpi-daemon":
        _run_module_main(
            "chemgraph.academy.runtime.daemon",
            _strip_remainder_separator(args.daemon_args),
        )
    elif command == "dashboard":
        _run_module_main(
            "chemgraph.academy.runtime.dashboard_launcher",
            _strip_remainder_separator(args.dashboard_args),
        )
    elif command == "run-compute":
        from chemgraph.academy.runtime.compute_launcher import main as compute_main

        code = compute_main(_strip_remainder_separator(args.compute_args))
        if code:
            sys.exit(code)
    elif command == "spawn-site":
        from chemgraph.academy.runtime.compute_launcher import main as compute_main

        forwarded = _strip_remainder_separator(args.spawn_site_args)
        code = compute_main(forwarded)
        if code:
            sys.exit(code)
    elif command == "launch":
        from chemgraph.academy.runtime.remote.remote_launcher import main as launch_main

        code = launch_main(_strip_remainder_separator(args.launch_args))
        if code:
            sys.exit(code)
    elif command == "inject":
        from chemgraph.academy.runtime.inject import main as inject_main

        code = inject_main(_strip_remainder_separator(args.inject_args))
        if code:
            sys.exit(code)
    elif command == "campaigns":
        from chemgraph.academy.campaigns import list_campaigns

        for name in list_campaigns():
            print(name)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
