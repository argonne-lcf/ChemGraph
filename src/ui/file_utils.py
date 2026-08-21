"""File-system helpers for the ChemGraph Streamlit UI.

Functions for finding XYZ files and extracting directory paths from agent
messages.  Helpers that searched the current working directory or the
global ``CHEMGRAPH_LOG_DIR`` were removed on purpose: they picked up
stale artifacts from earlier sessions.  Per-exchange attribution lives in
:mod:`ui.artifacts`.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional


def find_latest_xyz_file_in_dir(directory: str) -> Optional[str]:
    """Find the most recently modified ``.xyz`` file under a directory.

    Parameters
    ----------
    directory : str
        Directory to search recursively.

    Returns
    -------
    str or None
        Latest XYZ file path, or ``None`` when none is found.
    """
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
    """Extract a log directory from messages that reference output files.

    Parameters
    ----------
    messages : Any
        Message object, dictionary, list, or text to scan.

    Returns
    -------
    str or None
        Parent directory of a referenced output file, or ``None``.
    """
    if not messages:
        return None
    patterns = [
        r"(/[^\s'\"`]+?\.json)",
        r"(/[^\s'\"`]+?\.xyz)",
        r"(/[^\s'\"`]+?\.html)",
        r"(/[^\s'\"`]+?\.csv)",
    ]

    def _scan_value(value: Any) -> Optional[str]:
        """Recursively scan a value for absolute output-file references.

        Parameters
        ----------
        value : Any
            Message content, mapping, list, or scalar to scan.

        Returns
        -------
        str or None
            Parent directory of a referenced file, or ``None``.
        """
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
