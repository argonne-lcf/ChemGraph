"""Drive ordinary ChemGraph instances; the transport never sees graph internals."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import AsyncExitStack
from typing import Any

from chemgraph.distributed.protocol import PeerEnvelope, PeerTurnResult
from chemgraph.utils.artifacts import artifact_context


class ChemGraphRunner:
    def __init__(self, agent, *, thread_id: str, session=None):
        self.agent, self.thread_id, self.session = agent, thread_id, session
        self._answer: asyncio.Future | None = None
        self._legacy_interrupts: tuple[dict, ...] = ()
        agent.human_input_handler = self._ask_operator

    @property
    def interrupts(self) -> tuple[dict, ...]:
        if self.session is not None:
            return tuple({"id": item.id, "payload": item.payload} for item in self.session.pending_interrupts)
        return self._legacy_interrupts

    @property
    def failed(self) -> bool:
        return bool(self.session is not None and self.session.failed)

    async def _ask_operator(self, question: str, payload: Any = None):
        self._legacy_interrupts = ({"question": question, "payload": payload},)
        self._answer = asyncio.get_running_loop().create_future()
        try:
            return await self._answer
        finally:
            self._answer = None
            self._legacy_interrupts = ()

    def _result(self, result) -> PeerTurnResult:
        return PeerTurnResult(output=result.assistant_response, status=result.status, interrupts=self.interrupts)

    async def run(self, message: PeerEnvelope) -> PeerTurnResult:
        if self.session is not None:
            return self._result(await self.session.run(message.prompt()))
        with artifact_context(self.agent.log_dir):
            result = await self.agent.run(message.prompt(), config={"thread_id": self.thread_id})
        return PeerTurnResult(output=str(getattr(result, "content", result)))

    async def resume(self, response: Any) -> PeerTurnResult | None:
        if self.session is not None:
            return self._result(await self.session.resume(response))
        if self._answer is None or self._answer.done():
            raise RuntimeError("Graph is not waiting for operator input")
        self._answer.set_result(response)
        return None

    async def retry(self) -> PeerTurnResult:
        if self.session is None:
            raise RuntimeError("Checkpoint retry is supported only by main_agent; restart other failed peers")
        return self._result(await self.session.retry())


class ConfiguredRunner:
    """Own the MCP tools and optional SQLite checkpointer for one peer."""

    def __init__(self, config, name: str, peer_tools: list, trace):
        self.config, self.name, self.peer_tools, self.trace = config, name, peer_tools, trace
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> ChemGraphRunner:
        from chemgraph.agent.llm_agent import ChemGraph, PromptConfig
        from chemgraph.agent.main_session import MainAgentSession
        from chemgraph.mcp.client import load_mcp_tools

        spec = self.config.peers[self.name]
        directory = self.config.peer_dir(self.name)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = directory / "artifacts"
        artifacts.mkdir(exist_ok=True)
        thread_id = f"{self.config.run_id}/{self.config.launch_id}/{self.name}"
        try:
            sources = [source.model_copy(update={"env": {"CHEMGRAPH_LOG_DIR": str(artifacts), **source.env}})
                       for source in spec.mcp_servers]
            tools = [*await load_mcp_tools(sources), *self.peer_tools]
            if len({tool.name for tool in tools}) != len(tools):
                raise ValueError("MCP tools conflict with the peer communication tool")
            options = dict(spec.options)
            if spec.workflow_type == "main_agent":
                import aiosqlite
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
                from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

                connection = await self._stack.enter_async_context(aiosqlite.connect(directory / "checkpoints.db"))
                saver = AsyncSqliteSaver(connection, serde=JsonPlusSerializer(
                    pickle_fallback=False, allowed_msgpack_modules=None,
                ))
                await saver.setup()
                (directory / "checkpoints.db").chmod(0o600)
                identity = hashlib.sha256(self.config.model_dump_json().encode()).hexdigest()
                options.update(checkpointer=saver, configuration_id=f"distributed-{identity}")
            agent = ChemGraph(
                model_name=spec.model_name, workflow_type=spec.workflow_type, base_url=spec.base_url,
                tools=tools, prompts=PromptConfig(**spec.prompts), on_event=self.trace,
                log_dir=str(artifacts), memory_db_path=str(directory / "sessions.db"), **options,
            )
            session = None
            if spec.workflow_type == "main_agent":
                session = MainAgentSession(agent.workflow, thread_id=thread_id,
                                           session_metadata=agent.main_agent_metadata,
                                           session_store=agent.session_store, on_event=self.trace)
                snapshot = await agent.workflow.aget_state(session.config)
                if snapshot and snapshot.created_at:
                    await session.restore()
            return ChemGraphRunner(agent, thread_id=thread_id, session=session)
        except BaseException:
            await self._stack.aclose()
            raise

    async def __aexit__(self, *_args):
        await self._stack.aclose()
