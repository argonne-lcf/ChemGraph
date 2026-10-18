"""ChemGraph package metadata."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_DISTRIBUTION_NAMES = ("chemgraphagent", "chemgraph")


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML data from a file path.

    Parameters
    ----------
    path : pathlib.Path
        TOML file to read.

    Returns
    -------
    dict[str, Any]
        Parsed TOML data.
    """
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib

        return tomllib.loads(text)
    except ModuleNotFoundError:
        import toml

        return toml.loads(text)


def _version_from_distribution() -> str | None:
    """Return the installed package version when distribution metadata exists.

    Returns
    -------
    str or None
        Installed version string, or ``None`` when ChemGraph is running from a
        source checkout without package metadata.
    """
    for dist_name in _DISTRIBUTION_NAMES:
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue
    return None


def _version_from_pyproject() -> str | None:
    """Return the project version from a source checkout's ``pyproject.toml``.

    Returns
    -------
    str or None
        Project version string, or ``None`` when the file cannot be found or
        parsed.
    """
    start = Path(__file__).resolve()
    for directory in start.parents:
        pyproject = directory / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            project = _load_toml(pyproject).get("project", {})
        except Exception:
            return None
        project_version = project.get("version")
        return str(project_version) if project_version else None
    return None


def _patch_langgraph_sync_checkpoint_deadlock() -> None:
    """Prevent ``SyncPregelLoop`` thread-pool starvation during checkpointing.

    In ``langgraph==1.2.11``, ``SyncPregelLoop`` submits both
    ``checkpointer.put_writes(...)`` (stored in ``self._delta_write_futs``) and
    ``_checkpointer_put_after_previous(...)`` into the same bounded
    ``BackgroundExecutor`` ``ThreadPoolExecutor``. Because
    ``_checkpointer_put_after_previous`` blocks its worker thread waiting on
    ``self._delta_write_futs`` and ``prev.result()``, any synchronous run with
    small ``max_concurrency`` or multiple rapid supersteps can fill all worker
    threads with ``_checkpointer_put_after_previous`` tasks while the
    ``put_writes`` tasks they wait on sit queued behind them.

    Draining ``_delta_write_futs`` and ``_put_checkpoint_fut`` on the loop
    thread before submitting the next checkpoint task eliminates the nested
    in-pool wait while preserving checkpoint ordering.
    """
    try:
        from langgraph.pregel._loop import SyncPregelLoop
    except Exception:
        return

    if getattr(SyncPregelLoop, "_chemgraph_sync_checkpoint_patched", False):
        return

    orig_put_checkpoint = SyncPregelLoop._put_checkpoint

    def _put_checkpoint(self: Any, metadata: Any) -> Any:
        if self._delta_write_futs:
            futs, self._delta_write_futs = self._delta_write_futs, []
            for fut in futs:
                fut.result()
        if (prev := getattr(self, "_put_checkpoint_fut", None)) is not None:
            prev.result()
            self._put_checkpoint_fut = None
        return orig_put_checkpoint(self, metadata)

    SyncPregelLoop._put_checkpoint = _put_checkpoint  # type: ignore[method-assign]
    SyncPregelLoop._chemgraph_sync_checkpoint_patched = True  # type: ignore[attr-defined]


_patch_langgraph_sync_checkpoint_deadlock()
__version__ = _version_from_pyproject() or _version_from_distribution() or "unknown"
