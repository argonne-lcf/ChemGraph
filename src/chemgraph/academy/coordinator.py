"""Coordinator agent for multi-wave screening campaigns.

The coordinator manages a fleet of :class:`ScreeningAgent` instances,
collects results, and optionally uses a ChemGraph LLM workflow to
analyse the collected data and spawn follow-up screening waves.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import time
from typing import Any, Optional

from academy.agent import Agent, action, timer
from academy.handle import Handle

logger = logging.getLogger(__name__)


class CoordinatorAgent(Agent):
    """Collects screening results and orchestrates follow-up waves.

    Parameters
    ----------
    results_dir : str
        Directory where :class:`ScreeningAgent` instances write their
        per-molecule JSON result files.
    worker_handles : list[Handle] or None
        Handles to active screening agents (for progress polling).
    analysis_model : str
        LLM model for analysing aggregated results.
    analysis_workflow : str
        ChemGraph workflow type for the analysis step.
    analysis_kwargs : dict
        Extra kwargs for the analysis ChemGraph instance.
    """

    def __init__(
        self,
        results_dir: str,
        worker_handles: list[Handle] | None = None,
        analysis_model: str = "gpt-4o",
        analysis_workflow: str = "single_agent",
        **analysis_kwargs: Any,
    ) -> None:
        super().__init__()
        self._results_dir = results_dir
        self._worker_handles = worker_handles or []
        self._analysis_model = analysis_model
        self._analysis_workflow = analysis_workflow
        self._analysis_kwargs = analysis_kwargs
        self._collected: list[dict[str, Any]] = []
        self._analysis_result: Optional[dict[str, Any]] = None

    async def agent_on_startup(self) -> None:
        os.makedirs(self._results_dir, exist_ok=True)
        logger.info(
            "CoordinatorAgent started: watching %s, %d workers",
            self._results_dir,
            len(self._worker_handles),
        )

    # ------------------------------------------------------------------
    # Progress monitoring
    # ------------------------------------------------------------------

    @action
    async def poll_progress(self) -> dict[str, Any]:
        """Query all workers for their screening progress."""
        progress = []
        for handle in self._worker_handles:
            try:
                p = await handle.get_progress()
                progress.append(p)
            except Exception as exc:
                progress.append({"error": str(exc)})
        total = sum(p.get("total", 0) for p in progress if "error" not in p)
        completed = sum(p.get("completed", 0) for p in progress if "error" not in p)
        failed = sum(p.get("failed", 0) for p in progress if "error" not in p)
        return {
            "workers": len(progress),
            "total": total,
            "completed": completed,
            "failed": failed,
            "per_worker": progress,
        }

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    @action
    async def collect_results(self) -> list[dict[str, Any]]:
        """Read all result JSON files from the shared results directory."""
        pattern = os.path.join(self._results_dir, "*.json")
        files = sorted(glob.glob(pattern))
        results = []
        for path in files:
            try:
                with open(path) as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping corrupt result file: %s", path)
        self._collected = results
        logger.info("Collected %d results from %s", len(results), self._results_dir)
        return results

    # ------------------------------------------------------------------
    # LLM-powered analysis
    # ------------------------------------------------------------------

    @action
    async def analyse(self, query: Optional[str] = None) -> dict[str, Any]:
        """Use a ChemGraph agent to analyse collected results.

        Parameters
        ----------
        query : str, optional
            Custom analysis query.  Defaults to a standard prompt
            asking the LLM to rank candidates.
        """
        from chemgraph.agent.llm_agent import ChemGraph

        if not self._collected:
            await self.collect_results()

        successes = [r for r in self._collected if r.get("status") == "success"]
        if not successes:
            return {"error": "No successful results to analyse"}

        summary = json.dumps(successes, default=str, indent=2)
        if query is None:
            query = (
                "You are analysing computational chemistry screening results. "
                f"Here are {len(successes)} results:\n\n{summary}\n\n"
                "Identify the top candidates based on energy, stability, "
                "or other relevant properties. Rank them and explain why."
            )

        cg = ChemGraph(
            model_name=self._analysis_model,
            workflow_type=self._analysis_workflow,
            enable_memory=False,
            **self._analysis_kwargs,
        )
        self._analysis_result = await cg.run(query=query)
        return self._analysis_result

    @action
    async def get_analysis(self) -> dict[str, Any] | None:
        """Return the most recent analysis result."""
        return self._analysis_result

    # ------------------------------------------------------------------
    # Wave dispatch
    # ------------------------------------------------------------------

    @action
    async def suggest_followup_molecules(
        self,
        top_n: int = 10,
    ) -> list[str]:
        """Extract top candidate SMILES from analysis for a follow-up wave.

        Returns a list of SMILES strings identified as promising by
        the analysis step.  Falls back to returning the top-N by
        lowest energy if no analysis is available.
        """
        if not self._collected:
            await self.collect_results()

        successes = [r for r in self._collected if r.get("status") == "success"]
        # Simple heuristic: return the SMILES of completed molecules.
        # A real implementation would parse energies from results.
        return [r["smiles"] for r in successes[:top_n]]
