"""Persistent logical Academy agent for ChemGraph campaigns."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from academy.agent import Agent, action
from academy.agent import loop
from academy.handle import Handle
from academy.identifier import AgentId
from langchain_core.tools import BaseTool

from chemgraph.academy.core.peer_protocol import validate_message
from chemgraph.academy.observability.event_log import EventLog
from chemgraph.academy.observability.run_artifacts import write_status_snapshot
from chemgraph.academy.core.tools import build_chemgraph_reasoning_tools
from chemgraph.academy.core.turn import run_academy_turn
from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.prompt import PromptProfile
from chemgraph.academy.core.llm import LLMSettings

# Plug-in point: the ``turn_runner`` is a callable that takes the
# accumulated message state + tool bindings and returns the next
# decision (see swarm.core.turn.run_academy_turn for the exact
# signature). Default implementation lives in the ``chemgraph``
# extra package. Users who plug in a different runtime (React,
# DSPy, custom) can either (a) import their runner and pass it
# to ChemGraphLogicalAgent as ``turn_runner=...``, or (b) skip
# the extra entirely and pass their own callable.
#
# We import lazily inside ``on_startup`` (not at module load) so
# ``swarm`` is usable without ``chemgraphagent`` installed until
# the operator actually asks for the default runner.
_default_turn_runner = None  # resolved lazily in _get_default_turn_runner()


def _get_default_turn_runner():
    """Import chemgraph.agent.turn.run_turn on first use.

    Raises RuntimeError with an actionable message if the
    ``chemgraph`` extra is not installed. Cached for subsequent
    calls so repeated agent construction doesn't re-import.
    """
    global _default_turn_runner
    if _default_turn_runner is not None:
        return _default_turn_runner
    try:
        from chemgraph.agent.turn import run_turn as _rt
    except ImportError as exc:
        raise RuntimeError(
            "swarm's default turn_runner is chemgraph.agent.turn.run_turn "
            "(from the `chemgraphagent` package). Install swarm with the "
            "chemgraph extra (`pip install 'swarm[chemgraph]'`) or pass "
            "your own turn_runner callable to ChemGraphLogicalAgent."
        ) from exc
    _default_turn_runner = _rt
    return _rt


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
        external_tools: Sequence[BaseTool] = (),
        peer_agent_ids: Mapping[str, AgentId[Any]] | None = None,
        placement: dict[str, Any] | None = None,
        poll_timeout_s: float = 2.0,
        idle_timeout_s: float = 120.0,
        status_interval_s: float = 5.0,
        turn_runner: Any = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.campaign = campaign
        self.llm_settings = llm_settings
        self.prompt_profile = prompt_profile
        self.run_dir = run_dir
        self.max_decisions = max_decisions
        self.external_tools = list(external_tools)
        self.peer_agent_ids = dict(peer_agent_ids or {})
        self.placement = placement or {}
        self.poll_timeout_s = poll_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.status_interval_s = status_interval_s
        # Lazy default: the chemgraph adapter is imported the first
        # time we actually need it. Pass an explicit ``turn_runner``
        # to bypass entirely (e.g. a React or DSPy loop).
        self.turn_runner = turn_runner

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
        # Per-correlation-id scratchpad. Workflow states and reflex rules
        # can read/write via $state.X. Isolated per logical task so two
        # concurrent MOFs don't clobber each other.
        self.state: dict[str, dict[str, Any]] = {}

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
                'tool_names': [tool.name for tool in self.external_tools],
                'allowed_peers': list(self.spec.allowed_peers),
                'placement': self.placement,
                **self.placement,
            },
        )

    @action
    async def receive_message(self, message: dict[str, Any]) -> None:
        validate_message(message)
        first_message = not self.received_message_history
        self.received_message_history.append(message)
        self._trace('message_received', message)
        if self._wake_event is not None:
            self._wake_event.set()
        if first_message:
            # Operator-visible lifecycle landmark: the FIRST message
            # to land on this agent (usually the operator's kickoff
            # inject for initial_agent, or a peer's reply on everyone
            # else) is the canonical "kickoff arrived" signal. Use
            # print so it surfaces on stdout regardless
            # of log level configuration on the rank.
            sender = message.get('sender', '?')
            kind = message.get('kind', '?')
            tldr = message.get('tldr') or message.get('content', '')[:60]
            print(
                f"[agent {self.spec.name}] first message arrived from "
                f"{sender!r} (kind={kind}): {tldr}",
                flush=True,
            )

    @action
    async def get_status(self) -> dict[str, Any]:
        return await self.report_state()

    @loop
    async def deliberate(self, shutdown: asyncio.Event) -> None:
        if self._wake_event is None:
            raise RuntimeError('agent startup did not initialize wake state')

        decisions_completed = 0
        last_status = 0.0

        while not shutdown.is_set():
            if self._wake_event.is_set():
                self._wake_event.clear()
                decisions_completed, self_wake = await self.run_decision_turn(
                    decisions_completed,
                )
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

            # No idle timeout: PBS walltime is the sole ceiling. A
            # passive peer may legitimately sit idle for the full run
            # while its counterpart runs long tools; killing it early
            # cascades into all_agents_done=true and orphans the active
            # side.
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
        }

    async def _reasoning_round(self) -> bool:
        """Drive one LLM decision turn.

        Each wake: call the agent's configured engine once. The engine
        (chemgraph.single_agent, chemgraph.multi_agent, or a custom
        registered one) reads the mission + message history + tool
        palette and decides what to do -- including any peer messaging
        and finish_turn -- via bound action tools.
        """
        self._trace('round_started', {'round': self.round_index})
        tools = await build_chemgraph_reasoning_tools(
            spec=self.spec,
            run_dir=self.run_dir,
            external_tools=self.external_tools,
            peer_names=self.peer_names,
            peer_handles=self.peer_handles,
            outbox=self.outbox,
            tool_results=self.tool_results,
            get_round_index=lambda: self.round_index,
            set_final_result=self._set_final_result,
            trace=self._trace,
            received_message_history=self.received_message_history,
            wake_event=self._wake_event,
            agent_state=self.state,
        )
        result = await self._run_llm_turn(tools)
        self._trace('round_finished', {'round': self.round_index})
        return bool(result.requested_self_wake)

    async def _run_llm_turn(self, tools: list[BaseTool]):
        """One autonomous LangGraph turn using the agent's configured engine."""
        from chemgraph.academy.runtime.engines import resolve_engine
        engine_name = self.spec.engine
        runner = self.turn_runner or resolve_engine(engine_name)
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
            turn_runner=runner,
            peer_names=self.peer_names,
        )
        self._trace('agent_decision', {
            'mode': 'mpi_daemon',
            'wake_reason': f'daemon round {self.round_index}',
            'rationale': 'LM returned the listed tool calls for this daemon turn.',
            'round': self.round_index,
            'tool_names': list(result.executed_tool_names),
            'action_tools_called': list(result.action_tools_called),
            'science_tools_called': list(result.science_tools_called),
            'thread_id': result.thread_id,
            'engine': engine_name,
            'actions': [{'action': name} for name in result.executed_tool_names],
        })
        if result.requested_self_wake:
            self._trace('self_wake_scheduled', {
                'round': self.round_index,
                'reason': 'local ChemGraph tool result now available',
            })
        return result

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
