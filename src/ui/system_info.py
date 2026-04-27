"""Host and build metadata collection for the ChemGraph UI sidebar.

All system-introspection helpers are grouped here so they can be tested
independently of the Streamlit rendering layer.
"""

import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

import chemgraph as chemgraph_pkg
from chemgraph import __version__ as chemgraph_version


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_command(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 2) -> str:
    """Run a shell command and return stripped stdout; empty string on failure."""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            cwd=str(cwd) if cwd else None,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def _find_repo_root(start: Path) -> Optional[Path]:
    """Find git repo root by walking up parents from a starting path."""
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def _format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "Unknown"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return "Unknown"


def _get_total_memory_bytes() -> int:
    """Return total system memory in bytes when available."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total = int(page_size) * int(phys_pages)
        if total > 0:
            return total
    except Exception:
        pass

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
        except Exception:
            return 0
    return 0


def _get_cpu_model() -> str:
    """Try to get a human-readable CPU model name."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text().splitlines():
                if line.lower().startswith("model name"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        return parts[1].strip()
        except Exception:
            pass

    cpu_name = platform.processor().strip()
    if cpu_name:
        return cpu_name
    return platform.machine()


def _get_gpu_summary() -> str:
    """Return GPU summary from nvidia-smi when available."""
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return "No GPU detected"

    entries = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            name, mem_mib = parts[0], parts[1]
            entries.append(f"{name} ({mem_mib} MiB)")
        elif parts:
            entries.append(parts[0])
    return "; ".join(entries) if entries else "No GPU detected"


# ---------------------------------------------------------------------------
# Public API (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def get_host_info() -> Dict[str, str]:
    """Collect host metadata for sidebar display."""
    return {
        "hostname": socket.gethostname(),
        "platform": f"{platform.system()} {platform.release()}",
        "cpu_model": _get_cpu_model(),
        "cpu_cores": str(os.cpu_count() or "Unknown"),
        "memory_total": _format_bytes(_get_total_memory_bytes()),
        "gpu": _get_gpu_summary(),
    }


@st.cache_data(ttl=60)
def get_build_info() -> Dict[str, str]:
    """Collect app and repository metadata for sidebar display."""
    app_file = Path(__file__).resolve()
    chemgraph_file = Path(chemgraph_pkg.__file__).resolve()
    repo_root = _find_repo_root(app_file) or _find_repo_root(chemgraph_file)

    commit = "Unknown"
    commit_date = "Unknown"
    branch = "Unknown"

    if repo_root:
        commit = (
            _run_command(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
            or "Unknown"
        )
        commit_date = (
            _run_command(
                ["git", "show", "-s", "--format=%cd", "--date=iso", "HEAD"],
                cwd=repo_root,
            )
            or "Unknown"
        )
        branch = (
            _run_command(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root
            )
            or "Unknown"
        )

    return {
        "chemgraph_version": str(chemgraph_version),
        "commit": commit,
        "commit_date": commit_date,
        "branch": branch,
        "chemgraph_file": str(chemgraph_file),
    }


def render_sidebar_host_and_build_info() -> None:
    """Render host and build metadata blocks in the left sidebar."""
    host_info = get_host_info()
    build_info = get_build_info()

    with st.sidebar.expander("\U0001f5a5\ufe0f Host Info", expanded=False):
        st.markdown(f"**Hostname:** `{host_info['hostname']}`")
        st.markdown(f"**OS:** `{host_info['platform']}`")
        st.markdown(f"**CPU:** `{host_info['cpu_model']}`")
        st.markdown(f"**CPU Cores:** `{host_info['cpu_cores']}`")
        st.markdown(f"**Memory:** `{host_info['memory_total']}`")
        st.markdown(f"**GPU:** `{host_info['gpu']}`")

    with st.sidebar.expander("\U0001f4e6 Build Info", expanded=False):
        st.markdown(f"**ChemGraph Version:** `{build_info['chemgraph_version']}`")
        st.markdown(f"**Branch:** `{build_info['branch']}`")
        st.markdown(f"**Commit:** `{build_info['commit']}`")
        st.markdown(f"**Commit Date:** `{build_info['commit_date']}`")
        st.markdown(f"**ChemGraph File:** `{build_info['chemgraph_file']}`")
