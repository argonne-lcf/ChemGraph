"""File-system helpers for the ChemGraph Streamlit UI.

Functions for resolving output paths, finding XYZ files, checking file
recency, and extracting directory paths from agent messages.
"""

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


def resolve_output_path(path: str) -> str:
    """Resolve output paths relative to CHEMGRAPH_LOG_DIR when set."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    if log_dir:
        return os.path.join(log_dir, path)
    return path


def changed_recently(path: str = "ir_spectrum.png", window_seconds: int = 300) -> bool:
    """Return True if *path* exists and was modified within *window_seconds*."""
    p = Path(resolve_output_path(path))
    if not p.exists():
        return False

    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - mtime) <= timedelta(seconds=window_seconds)


def find_latest_xyz_file() -> Optional[str]:
    """Find the most recently modified ``.xyz`` file in the log dir or cwd."""
    search_dirs: list[str] = []
    log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    if log_dir:
        search_dirs.append(log_dir)
    search_dirs.append(os.getcwd())

    latest_path: Optional[str] = None
    latest_mtime = -1.0
    for base in search_dirs:
        if not base or not os.path.isdir(base):
            continue
        for path in Path(base).rglob("*.xyz"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = str(path)
    return latest_path


def find_latest_xyz_file_in_dir(directory: str) -> Optional[str]:
    """Find the most recently modified ``.xyz`` file under *directory*."""
    if not directory or not os.path.isdir(directory):
        return None
    latest_path: Optional[str] = None
    latest_mtime = -1.0
    for path in Path(directory).rglob("*.xyz"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = str(path)
    return latest_path


def extract_log_dir_from_messages(messages: Any) -> Optional[str]:
    """Extract a directory path from message content that references an output file."""
    if not messages:
        return None
    patterns = [
        r"(/[^\s'\"`]+?\.json)",
        r"(/[^\s'\"`]+?\.xyz)",
        r"(/[^\s'\"`]+?\.html)",
        r"(/[^\s'\"`]+?\.csv)",
    ]

    def _scan_value(value: Any) -> Optional[str]:
        if isinstance(value, str):
            for pattern in patterns:
                match = re.search(pattern, value)
                if match:
                    path = match.group(1)
                    if os.path.isabs(path):
                        return str(Path(path).parent)
        elif isinstance(value, dict):
            for v in value.values():
                found = _scan_value(v)
                if found:
                    return found
        elif isinstance(value, list):
            for v in value:
                found = _scan_value(v)
                if found:
                    return found
        return None

    for message in reversed(messages):
        content = ""
        if hasattr(message, "content"):
            content = getattr(message, "content", "")
        elif isinstance(message, dict):
            content = message.get("content", "")
        elif isinstance(message, str):
            content = message
        else:
            content = str(message)
        if not content:
            continue
        found = _scan_value(content)
        if found:
            return found

        # Also scan structured tool outputs if present
        if hasattr(message, "additional_kwargs"):
            found = _scan_value(message.additional_kwargs)
            if found:
                return found
        if isinstance(message, dict):
            found = _scan_value(message)
            if found:
                return found
    return None
