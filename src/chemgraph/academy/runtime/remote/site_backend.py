"""Minimal SiteBackend Protocol shared by attach- and submit-mode.

Earned its keep in phase 2: the orchestrator now dispatches across
two backends, so a uniform interface beats two ``if mode == ...``
branches. Methods are intentionally small (no pre/post hooks, no
event streams) -- expand only when a future backend needs more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class SiteBackend(Protocol):
    """How one site is brought up + torn down. One instance per --site."""

    site_name: str

    async def start(self) -> None:
        """Kick the compute side -- ssh+exec for attach, qsub for submit."""
        ...

    async def wait_ready(
        self,
        *,
        local_run_dir: Path,
        timeout_s: float,
    ) -> set[str]:
        """Block until this site's agents have registered. Return their names.

        Raises TimeoutError on miss.
        """
        ...

    async def stop(self, *, force: bool = False) -> None:
        """Cancel work owned by this backend. Mode-aware:
        attach -> SIGTERM the ssh process (allocation untouched);
        submit -> qdel the PBS job.
        """
        ...
