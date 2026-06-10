"""Persistent logical Academy agent for ChemGraph campaigns."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from academy.agent import Agent, action
from academy.agent import loop
from academy.handle import Handle
from academy.identifier import AgentId

from chemgraph.mcp.fastmcp_client import (
    FastMCPToolInvoker,
)
from chemgraph.academy.core.peer_protocol import validate_message
from chemgraph.academy.observability.event_log import EventLog
from chemgraph.academy.observability.run_artifacts import write_status_snapshot
from chemgraph.academy.core.tools import build_chemgraph_reasoning_tools
from chemgraph.academy.core.turn import run_academy_turn
from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.lm import LLMSettings
from chemgraph.academy.core.prompt import PromptProfile


class ChemGraphLogicalAgent(Agent):
    """Persistent Academy logical agent for one ChemGraph campaign role."""

    def __init__(
        self,
        spec: ChemGraphAgentSpec,
        *,
        campaign: ChemGraphCampaign,
        llm_settings: LLMSettings,
        prompt_profile: PromptProfile,
        run_dir: Path,
        max_decisions: int,
        tool_invoker: FastMCPToolInvoker,
        peer_agent_ids: Mapping[str, AgentId[Any]] | None = None,
        placement: dict[str, Any] | None = None,
        poll_timeout_s: float = 2.0,
        idle_timeout_s: float = 120.0,
        status_interval_s: float = 5.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.campaign = campaign
        self.llm_settings = llm_settings
        self.prompt_profile = prompt_profile
        self.run_dir = run_dir
        self.max_decisions = max_decisions
        self.tool_invoker = tool_invoker
        self.peer_agent_ids = dict(peer_agent_ids or {})
        self.placement = placement or {}
        self.poll_timeout_s = poll_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.status_interval_s = status_interval_s

        self.peer_names = tuple(spec.allowed_peers)
        self.peer_handles: dict[str, Handle[Any]] = {}
        self.received_message_history: list[dict[str, Any]] = []
        self.outbox: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.final_result: dict[str, Any] | None = None
        self.round_index = 0
        self.finished = False
        self.last_error: str | None = None
        self._wake_event: asyncio.Event | None = None

    async def agent_on_startup(self) -> None:
        self._wake_event = asyncio.Event()
        self.peer_handles = {
            name: Handle(agent_id)
            for name, agent_id in self.peer_agent_ids.items()
            if name in self.peer_names
        }
        self._trace(
            'agent_started',
            {
                'role': self.spec.role,
                'tool_names': list(self.spec.tool_names),
                'allowed_peers': list(self.spec.allowed_peers),
                'placement': self.placement,
                **self.placement,
            },
        )

    @action
    async def receive_message(self, message: dict[str, Any]) -> None:
        validate_message(message)
        self.received_message_history.append(message)
        self._trace('message_received', message)
        if self._wake_event is not None:
            self._wake_event.set()

    @action
    async def get_status(self) -> dict[str, Any]:
        return await self.report_state()

    @loop
    async def deliberate(self, shutdown: asyncio.Event) -> None:
        if self._wake_event is None:
            raise RuntimeError('agent startup did not initialize wake state')

        decisions_completed = 0
        last_activity = time.monotonic()
        last_status = 0.0

        while not shutdown.is_set():
            if self._wake_event.is_set():
                self._wake_event.clear()
                decisions_completed, self_wake = await self.run_decision_turn(
                    decisions_completed,
                )
                last_activity = time.monotonic()
                if self_wake:
                    self._wake_event.set()
                await self.write_runtime_status()
                if decisions_completed >= self.max_decisions:
                    self._trace(
                        'max_decisions_reached',
                        {'decisions_completed': decisions_completed},
                    )
                    break
                continue

            now = time.monotonic()
            if now - last_status >= self.status_interval_s:
                await self.write_runtime_status()
                last_status = now

            if now - last_activity >= self.idle_timeout_s:
                self._trace(
                    'idle_timeout',
                    {
                        'idle_timeout_s': self.idle_timeout_s,
                        'decisions_completed': decisions_completed,
                    },
                )
                break

            try:
                await asyncio.wait_for(
                    self._wake_event.wait(),
                    timeout=self.poll_timeout_s,
                )
            except asyncio.TimeoutError:
                pass

        self.finished = True
        self._trace(
            'daemon_stopped',
            {
                'decisions_completed': decisions_completed,
                'shutdown_requested': shutdown.is_set(),
            },
        )
        await self.write_runtime_status()
        self.agent_shutdown()

    async def write_runtime_status(self) -> None:
        write_status_snapshot(
            run_dir=self.run_dir,
            campaign=self.campaign,
            agent_state=await self.report_state(),
            placement=self.placement,
        )

    async def run_decision_turn(self, decisions_completed: int) -> tuple[int, bool]:
        self.round_index += 1
        try:
            self_wake = await self._reasoning_round()
        except Exception as exc:
            self.last_error = repr(exc)
            self._trace('agent_error', {'error': self.last_error})
            raise
        return decisions_completed + 1, self_wake

    async def report_state(self) -> dict[str, Any]:
        return {
            'agent_name': self.spec.name,
            'role': self.spec.role,
            'status_updated_at': time.time(),
            'round': self.round_index,
            'finished': self.finished,
            'last_error': self.last_error,
            'current_activity': None,
            'recent_outbox': self.outbox[-10:],
            'belief': self.final_result or {
                'hypothesis': None,
                'confidence': 0.0,
                'supporting_message_ids': [],
                'supporting_tool_result_ids': [],
                'reason': None,
            },
        }

    async def _reasoning_round(self) -> bool:
        self._trace('round_started', {'round': self.round_index})
        tools = await build_chemgraph_reasoning_tools(
            spec=self.spec,
            run_dir=self.run_dir,
            tool_invoker=self.tool_invoker,
            peer_names=self.peer_names,
            peer_handles=self.peer_handles,
            outbox=self.outbox,
            tool_results=self.tool_results,
            get_round_index=lambda: self.round_index,
            set_final_result=self._set_final_result,
            trace=self._trace,
        )
        result = await run_academy_turn(
            campaign=self.campaign,
            spec=self.spec,
            llm_settings=self.llm_settings,
            prompt_profile=self.prompt_profile,
            run_dir=self.run_dir,
            max_decisions=self.max_decisions,
            tools=tools,
            received_message_history=self.received_message_history,
            outbox=self.outbox,
            tool_results=self.tool_results,
            get_final_result=lambda: self.final_result,
            get_round_index=lambda: self.round_index,
            trace=self._trace,
            peer_names=self.peer_names,
        )
        self._trace(
            'agent_decision',
            {
                'mode': 'mpi_daemon',
                'wake_reason': f'daemon round {self.round_index}',
                'rationale': 'LM returned the listed tool calls for this daemon turn.',
                'round': self.round_index,
                'tool_names': list(result.executed_tool_names),
                'action_tools_called': list(result.action_tools_called),
                'science_tools_called': list(result.science_tools_called),
                'thread_id': result.thread_id,
                'engine': 'chemgraph_single_agent',
                'actions': [
                    {'action': name}
                    for name in result.executed_tool_names
                ],
            },
        )
        self._trace('round_finished', {'round': self.round_index})
        if result.requested_self_wake:
            self._trace(
                'self_wake_scheduled',
                {
                    'round': self.round_index,
                    'reason': (
                        'local ChemGraph tool result is now available in '
                        'local_chemgraph_tool_results'
                    ),
                },
            )
        return result.requested_self_wake

    def _set_final_result(self, result: dict[str, Any]) -> None:
        self.final_result = result

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        EventLog(self.run_dir / 'events.jsonl').emit(
            event,  # type: ignore[arg-type]
            run_id=self.run_dir.name,
            agent_id=self.spec.name,
            role=self.spec.role,
            payload=payload,
        )
