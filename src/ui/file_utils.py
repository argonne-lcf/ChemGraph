"""File-system helpers for the ChemGraph Streamlit UI.

Functions for finding XYZ files and extracting directory paths from agent
messages.  Helpers that searched the current working directory or the
global ``CHEMGRAPH_LOG_DIR`` were removed on purpose: they picked up
stale artifacts from earlier sessions.  Per-exchange attribution lives in
:mod:`ui.artifacts`.
"""

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
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


_WINDOWS_DRIVE_RE = re.compile(r"[A-Za-z]:[\\/]")


def _is_absolute_output_path(path: str) -> bool:
    """Return whether *path* is absolute on the platform it came from.

    Agent messages can quote paths from either path flavor (e.g. a stored
    session moved between machines), so both flavors are checked
    explicitly -- ``os.path.isabs`` would answer only for the platform
    the app happens to run on. Non-existent paths are discarded by the
    callers' existence/containment checks.

    Parameters
    ----------
    path : str
        Extracted path candidate.

    Returns
    -------
    bool
        ``True`` for POSIX-absolute or Windows drive-absolute paths.
    """
    return path.startswith("/") or bool(_WINDOWS_DRIVE_RE.match(path))


def _parent_directory(path: str) -> str:
    """Return the parent directory of *path*, honoring its path flavor.

    Uses the Pure* path classes so the result keeps the flavor of the
    input on every platform: ``pathlib.Path`` would rewrite a POSIX path
    to backslashes when running on Windows (and cannot split a Windows
    path when running on POSIX).

    Parameters
    ----------
    path : str
        Absolute file path (POSIX or Windows form).

    Returns
    -------
    str
        Parent directory in the same form.
    """
    if _WINDOWS_DRIVE_RE.match(path):
        return str(PureWindowsPath(path).parent)
    return str(PurePosixPath(path).parent)


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
    # Absolute POSIX (/run/dir/file.ext) or Windows drive
    # (C:\run\dir\file.ext, C:/run/dir/file.ext) paths. The boundary
    # prefix keeps slashes inside relative paths and URLs from matching.
    _abs = r"(?:/|[A-Za-z]:[\\/])"
    _start = r"(?:^|[\s'\"`=(,])"
    patterns = [
        rf"{_start}({_abs}[^\s'\"`]+?\.json)",
        rf"{_start}({_abs}[^\s'\"`]+?\.xyz)",
        rf"{_start}({_abs}[^\s'\"`]+?\.html)",
        rf"{_start}({_abs}[^\s'\"`]+?\.csv)",
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
                    if _is_absolute_output_path(path):
                        return _parent_directory(path)
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
