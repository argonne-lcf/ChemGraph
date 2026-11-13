"""Readline history restricted to top-level interactive inputs."""

from __future__ import annotations

import atexit
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def enable_input_history():
    """Yield an explicit recorder; save history and restore readline on exit."""
    if not sys.stdin.isatty():
        yield lambda _value: None
        return
    try:
        import readline
    except ImportError:
        yield lambda _value: None
        return
    if not hasattr(readline, "set_auto_history"):
        yield lambda _value: None
        return

    path = Path.home() / ".chemgraph" / "cli_history"
    readline.set_auto_history(False)
    try:
        readline.clear_history()
        readline.set_history_length(1000)
        try:
            readline.read_history_file(str(path))
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not load CLI input history.", exc_info=True)

        def record(value: str) -> None:
            if not value.strip():
                return
            last = readline.get_history_item(readline.get_current_history_length())
            if value != last:
                readline.add_history(value)

        def save() -> None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
                try:
                    os.fchmod(descriptor, 0o600)
                finally:
                    os.close(descriptor)
                readline.write_history_file(str(path))
            except OSError:
                logger.debug("Could not save CLI input history.", exc_info=True)

        atexit.register(save)
        try:
            yield record
        finally:
            atexit.unregister(save)
            save()
    finally:
        readline.set_auto_history(True)
