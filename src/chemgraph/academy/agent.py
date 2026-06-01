"""Base Academy Agent wrapping a ChemGraph instance.

Each ``ChemGraphAgent`` holds one ``ChemGraph`` object and exposes its
``run()`` method as an Academy ``@action`` so it can be invoked remotely
by peer agents, coordinators, or the Manager user handle.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Optional

from academy.agent import Agent, action

from chemgraph.agent.llm_agent import ChemGraph

logger = logging.getLogger(__name__)


class ChemGraphAgent(Agent):
    """Academy Agent wrapping a single :class:`ChemGraph` instance.

    Parameters
    ----------
    model_name : str
        LLM model to use (e.g. ``"gpt-4o"``, ``"claude-sonnet-4"``).
    workflow_type : str
        ChemGraph workflow (e.g. ``"single_agent"``, ``"multi_agent"``).
    log_dir : str or None
        Base directory for agent logs.  A per-agent subdirectory is
        created automatically.
    rate_limiter : RateLimiter or None
        Shared rate limiter for LLM API calls.
    chemgraph_kwargs : dict
        Extra keyword arguments forwarded to the :class:`ChemGraph`
        constructor (e.g. ``base_url``, ``api_key``, ``recursion_limit``).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        workflow_type: str = "single_agent",
        log_dir: Optional[str] = None,
        rate_limiter: Any = None,
        **chemgraph_kwargs: Any,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._workflow_type = workflow_type
        self._log_dir = log_dir
        self._rate_limiter = rate_limiter
        self._chemgraph_kwargs = chemgraph_kwargs
        self._cg: Optional[ChemGraph] = None
        self._agent_uuid = uuid.uuid4().hex[:8]

    async def agent_on_startup(self) -> None:
        """Initialise the ChemGraph instance on the remote worker."""
        agent_log_dir = self._log_dir
        if agent_log_dir:
            agent_log_dir = os.path.join(agent_log_dir, self._agent_uuid)
            os.makedirs(agent_log_dir, exist_ok=True)

        self._cg = ChemGraph(
            model_name=self._model_name,
            workflow_type=self._workflow_type,
            log_dir=agent_log_dir,
            enable_memory=False,
            **self._chemgraph_kwargs,
        )
        logger.info(
            "ChemGraphAgent %s started: model=%s workflow=%s",
            self._agent_uuid,
            self._model_name,
            self._workflow_type,
        )

    async def agent_on_shutdown(self) -> None:
        """Clean up resources."""
        logger.info("ChemGraphAgent %s shutting down", self._agent_uuid)
        self._cg = None

    @action
    async def run_query(
        self,
        query: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a ChemGraph query and return the result.

        Parameters
        ----------
        query : str
            The natural-language chemistry query.
        config : dict, optional
            LangGraph config (thread_id, etc.).

        Returns
        -------
        dict
            The workflow result (serialised state or last message,
            depending on the ChemGraph ``return_option``).
        """
        if self._cg is None:
            raise RuntimeError("Agent not initialised (call agent_on_startup first)")

        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(self._model_name)

        thread_cfg = config or {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}
        result = await self._cg.run(query=query, config=thread_cfg)
        return result

    @action
    async def get_info(self) -> dict[str, str]:
        """Return metadata about this agent instance."""
        return {
            "agent_uuid": self._agent_uuid,
            "model_name": self._model_name,
            "workflow_type": self._workflow_type,
        }
