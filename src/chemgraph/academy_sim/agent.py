"""Academy agent that exposes one ChemGraph graph as a black box."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from academy.agent import Agent, action, loop

from chemgraph.academy_sim.artifacts import emit_event, write_status
from chemgraph.academy_sim.config import AcademySimConfig, GraphConfig
from chemgraph.academy_sim.envelopes import PeerEnvelope, envelope_to_prompt
from chemgraph.agent.graph_runtime import GraphRunner


class ChemGraphSimAgent(Agent):
    """Thin Academy wrapper around a configured ChemGraph graph."""

    def __init__(
        self,
        *,
        config: AcademySimConfig,
        graph: GraphConfig,
        run_graph: GraphRunner,
        run_dir: Path,
        done_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.graph = graph
        self.run_graph = run_graph
        self.run_dir = run_dir
        self.done_event = done_event or asyncio.Event()
        self.run_count = 0
        self.last_error: str | None = None
        self.last_output: str | None = None
        self.finished = False
        self._inbox: asyncio.Queue[PeerEnvelope] = asyncio.Queue()
        self._active_runs = 0
        self._last_activity = time.monotonic()

    async def agent_on_startup(self) -> None:
        self._trace(
            'agent_started',
            {
                'allowed_peers': list(self.graph.allowed_peers),
                'site': self.graph.site,
                'workflow_type': self.graph.workflow_type,
            },
        )
        await self.write_runtime_status()

    @action
    async def receive_message(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Receive one peer envelope and enqueue it for graph processing.

        Academy ``Handle.action(...)`` is request/response. If this action runs
        ChemGraph inline, two graphs can deadlock when each graph sends a peer
        message while the caller is still waiting for the callee's action to
        return. Queueing makes delivery synchronous but graph processing
        asynchronous.
        """

        parsed = PeerEnvelope.model_validate(envelope)
        if parsed.recipient != self.graph.name:
            raise ValueError(
                f'envelope recipient {parsed.recipient!r} does not match '
                f'graph {self.graph.name!r}'
            )
        self._last_activity = time.monotonic()
        await self._inbox.put(parsed)
        self._trace(
            'peer_message_enqueued',
            {
                'message_id': parsed.message_id,
                'queue_size': self._inbox.qsize(),
                'sender': parsed.sender,
            },
        )
        await self.write_runtime_status()
        return {
            'message_id': parsed.message_id,
            'queue_size': self._inbox.qsize(),
            'status': 'accepted',
        }

    @loop
    async def process_inbox(self, shutdown: asyncio.Event) -> None:
        """Run ChemGraph for queued peer messages."""

        while not shutdown.is_set():
            try:
                envelope = await asyncio.wait_for(
                    self._inbox.get(),
                    timeout=self.graph.poll_interval_s,
                )
            except asyncio.TimeoutError:
                continue
            self._last_activity = time.monotonic()
            self.run_count += 1
            self._active_runs += 1
            self._trace(
                'peer_message_processing_started',
                {
                    'message_id': envelope.message_id,
                    'sender': envelope.sender,
                },
            )
            try:
                await self._run_envelope(envelope)
            finally:
                self._active_runs -= 1
                self._inbox.task_done()

    async def _run_envelope(self, parsed: PeerEnvelope) -> None:
        try:
            result = await self.run_graph(envelope_to_prompt(parsed))
        except Exception as exc:
            self.last_error = repr(exc)
            self._trace('graph_run_failed', {'error': self.last_error})
            await self.write_runtime_status()
            raise
        self.last_output = result.output
        if result.terminal_tool is None:
            self.done_event.set()
            self._trace(
                'graph_naturally_completed',
                {'message_id': parsed.message_id},
            )
        self._trace(
            'graph_run_finished',
            {
                'executed_tool_names': list(result.executed_tool_names),
                'message_id': parsed.message_id,
                'output_preview': result.output[:500],
                'run_count': self.run_count,
                'terminal_tool': result.terminal_tool,
            },
        )
        await self.write_runtime_status()

    @action
    async def get_status(self) -> dict[str, Any]:
        return await self.report_state()

    @loop
    async def idle_watchdog(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            idle = time.monotonic() - self._last_activity >= self.graph.idle_timeout_s
            if idle and self._inbox.empty() and self._active_runs == 0:
                self._trace(
                    'idle_timeout',
                    {'idle_timeout_s': self.graph.idle_timeout_s},
                )
                break
            try:
                await asyncio.wait_for(
                    shutdown.wait(),
                    timeout=self.graph.poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass
        self.finished = True
        await self.write_runtime_status()
        self.agent_shutdown()

    @loop
    async def completion_watchdog(self, shutdown: asyncio.Event) -> None:
        """Shutdown after ChemGraph marks the graph done and work is drained."""

        while not shutdown.is_set():
            if self.done_event.is_set() and self._inbox.empty() and self._active_runs == 0:
                self._trace('graph_marked_done', {'run_count': self.run_count})
                break
            try:
                await asyncio.wait_for(
                    shutdown.wait(),
                    timeout=self.graph.poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass
        if not shutdown.is_set():
            self.finished = True
            await self.write_runtime_status()
            self.agent_shutdown()

    async def report_state(self) -> dict[str, Any]:
        return {
            'finished': self.finished,
            'graph': self.graph.name,
            'graph_done': self.done_event.is_set(),
            'last_error': self.last_error,
            'pending_messages': self._inbox.qsize(),
            'run_count': self.run_count,
            'runs_active': self._active_runs,
            'site': self.graph.site,
            'updated_at': time.time(),
        }

    async def write_runtime_status(self) -> None:
        write_status(
            self.run_dir,
            run_id=self.config.run_id,
            graph=self.graph.name,
            status=await self.report_state(),
        )

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        emit_event(
            self.run_dir,
            event=event,
            run_id=self.config.run_id,
            graph=self.graph.name,
            payload=payload,
        )
