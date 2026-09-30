"""Launcher for the ChemGraph web UI.

Used by ``chemgraph ui`` (and installable as a standalone entry point)
so users can start the Streamlit app without knowing where the package
is installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

# Brand accent applied on top of the user's light/dark base theme.
PRIMARY_COLOR = "#0E8A8C"


def app_path() -> str:
    """Return the absolute path of the Streamlit app module."""
    import ui

    return str(Path(ui.__file__).resolve().parent / "app.py")


def launch(
    address: str = "localhost",
    port: int = 8501,
    headless: bool = False,
    extra_args: Sequence[str] = (),
    enable_deep_agent: bool = False,
    deep_agent_roots: Optional[Sequence[str]] = None,
) -> int:
    """Run ``streamlit run`` for the ChemGraph UI and wait for it.

    Parameters
    ----------
    address : str, optional
        Bind address for the Streamlit server.
    port : int, optional
        Server port.
    headless : bool, optional
        Do not open a browser (for servers/containers).
    extra_args : Sequence[str], optional
        Additional arguments passed through to ``streamlit run``.
    enable_deep_agent : bool, optional
        Offer the experimental host-shell ``deep_agent`` workflow in the UI
        (sets ``CHEMGRAPH_UI_DEEPAGENT=1`` for the server process).
    deep_agent_roots : sequence of str, optional
        Directories UI-entered Deep Agent workspace/skill paths must stay in
        (default: the launch directory).

    Returns
    -------
    int
        The Streamlit process exit code.
    """
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path(),
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--theme.primaryColor",
        PRIMARY_COLOR,
    ]
    if headless:
        cmd += ["--server.headless", "true"]
    cmd += list(extra_args)
    env = deep_agent_environment(os.environ, enable_deep_agent, deep_agent_roots)
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 0


def deep_agent_environment(
    base: "os._Environ[str] | dict[str, str]",
    enable_deep_agent: bool,
    deep_agent_roots: Optional[Sequence[str]],
) -> dict[str, str]:
    """Return the server environment carrying the Deep Agent policy."""
    from ui.deepagent_policy import ENABLE_ENV, ROOTS_ENV

    env = dict(base)
    if enable_deep_agent:
        env[ENABLE_ENV] = "1"
    if deep_agent_roots:
        env[ROOTS_ENV] = os.pathsep.join(
            str(Path(root).expanduser().resolve()) for root in deep_agent_roots
        )
    return env


def add_deep_agent_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the Deep Agent server-policy flags to a ``chemgraph ui`` parser."""
    parser.add_argument(
        "--enable-deep-agent",
        action="store_true",
        help=(
            "Offer the experimental deep_agent workflow, which can run shell "
            "commands on this host after in-chat approval. Off by default; "
            "only enable it for a UI that only you can reach."
        ),
    )
    parser.add_argument(
        "--deep-agent-root",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Directory the UI's Deep Agent workspace and skill paths must "
            "stay inside (repeatable; default: the launch directory)."
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point."""
    parser = argparse.ArgumentParser(
        prog="chemgraph ui",
        description="Launch the ChemGraph web UI (Streamlit).",
    )
    parser.add_argument(
        "--address",
        default="localhost",
        help="Bind address (default: localhost).",
    )
    parser.add_argument(
        "--port", type=int, default=8501, help="Port (default: 8501)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a browser window.",
    )
    add_deep_agent_arguments(parser)
    parser.add_argument(
        "streamlit_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to 'streamlit run' (prefix with --).",
    )
    args = parser.parse_args(argv)
    extra = [a for a in args.streamlit_args if a != "--"]
    raise SystemExit(
        launch(
            args.address,
            args.port,
            args.headless,
            extra,
            enable_deep_agent=args.enable_deep_agent,
            deep_agent_roots=args.deep_agent_root,
        )
    )


if __name__ == "__main__":
    main()
