"""Screening agent for batch molecule processing.

Wraps :class:`ChemGraphAgent` with a ``@loop`` that iterates over an
assigned list of molecules and publishes results via the Academy
exchange.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

from academy.agent import Agent, action, loop

from chemgraph.academy.agent import ChemGraphAgent

logger = logging.getLogger(__name__)


class ScreeningAgent(ChemGraphAgent):
    """Agent that screens a batch of molecules using a ChemGraph workflow.

    Parameters
    ----------
    molecules : list[str]
        SMILES strings to screen.
    query_template : str
        Query template with ``{smiles}`` placeholder, e.g.
        ``"Optimize the geometry of {smiles} and compute its energy."``.
    results_dir : str or None
        Directory to write per-molecule JSON result files for
        downstream aggregation.  If ``None``, results are only
        returned via the exchange.
    model_name, workflow_type, log_dir, rate_limiter, **chemgraph_kwargs
        Forwarded to :class:`ChemGraphAgent`.
    """

    def __init__(
        self,
        molecules: list[str],
        query_template: str,
        results_dir: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        workflow_type: str = "single_agent",
        log_dir: Optional[str] = None,
        rate_limiter: Any = None,
        **chemgraph_kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            workflow_type=workflow_type,
            log_dir=log_dir,
            rate_limiter=rate_limiter,
            **chemgraph_kwargs,
        )
        self._molecules = molecules
        self._query_template = query_template
        self._results_dir = results_dir
        self.results: list[dict[str, Any]] = []
        self.completed: int = 0
        self.failed: int = 0

    async def agent_on_startup(self) -> None:
        await super().agent_on_startup()
        if self._results_dir:
            os.makedirs(self._results_dir, exist_ok=True)
        logger.info(
            "ScreeningAgent %s: %d molecules to process",
            self._agent_uuid,
            len(self._molecules),
        )

    @action
    async def get_progress(self) -> dict[str, Any]:
        """Return screening progress."""
        return {
            "agent_uuid": self._agent_uuid,
            "total": len(self._molecules),
            "completed": self.completed,
            "failed": self.failed,
        }

    @loop
    async def screening_loop(self, shutdown: asyncio.Event) -> None:
        """Iterate over assigned molecules and run queries."""
        for smiles in self._molecules:
            if shutdown.is_set():
                logger.info(
                    "ScreeningAgent %s: shutdown requested, stopping",
                    self._agent_uuid,
                )
                break

            query = self._query_template.format(smiles=smiles)
            t0 = time.monotonic()
            try:
                result = await self.run_query(query)
                elapsed = time.monotonic() - t0
                record = {
                    "smiles": smiles,
                    "status": "success",
                    "result": result,
                    "elapsed_seconds": round(elapsed, 2),
                    "agent_uuid": self._agent_uuid,
                }
                self.completed += 1
            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.exception(
                    "ScreeningAgent %s: failed on %s",
                    self._agent_uuid,
                    smiles,
                )
                record = {
                    "smiles": smiles,
                    "status": "error",
                    "error": str(exc),
                    "elapsed_seconds": round(elapsed, 2),
                    "agent_uuid": self._agent_uuid,
                }
                self.failed += 1

            self.results.append(record)

            # Write individual result file for aggregation.
            if self._results_dir:
                safe_name = smiles.replace("/", "_").replace("\\", "_")[:50]
                path = os.path.join(
                    self._results_dir,
                    f"{self._agent_uuid}_{safe_name}.json",
                )
                with open(path, "w") as f:
                    json.dump(record, f, default=str)

        logger.info(
            "ScreeningAgent %s: finished (%d ok, %d failed)",
            self._agent_uuid,
            self.completed,
            self.failed,
        )
        # Signal that this agent is done.
        self.agent_shutdown()

    @action
    async def get_results(self) -> list[dict[str, Any]]:
        """Return all collected results so far."""
        return self.results
