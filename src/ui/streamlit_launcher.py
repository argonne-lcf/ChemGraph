"""Launcher for the ChemGraph web UI.

Used by ``chemgraph ui`` (and installable as a standalone entry point)
so users can start the Streamlit app without knowing where the package
is installed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

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
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


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
    parser.add_argument(
        "streamlit_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to 'streamlit run' (prefix with --).",
    )
    args = parser.parse_args(argv)
    extra = [a for a in args.streamlit_args if a != "--"]
    raise SystemExit(launch(args.address, args.port, args.headless, extra))


if __name__ == "__main__":
    main()
