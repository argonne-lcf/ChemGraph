"""Discover and activate specialist workers without eager graph imports."""

from __future__ import annotations

import json
import re
from threading import RLock
from typing import Annotated, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from chemgraph.graphs.subagents import _adapt_subagent
from chemgraph.registry.agents import AgentRegistry
from chemgraph.registry.tools import RegistryError


class RegistryAgentState(AgentState):
    active_registry_agents: NotRequired[Annotated[list[str], PrivateStateAttr]]


class RegistryAgentsMiddleware(AgentMiddleware):
    """Keep worker implementations lazy and selections private to each turn."""

    state_schema = RegistryAgentState

    def __init__(self, registry, *, llm, options=None, interrupt_on=None,
                 attached_workers=(), initial_agents=(), recorder=None, tool_middleware=()):
        if not isinstance(registry, AgentRegistry):
            raise TypeError("agent_registry must be an AgentRegistry.")
        self.registry = registry
        self.llm = llm
        self.options = {}
        for name, settings in (options or {}).items():
            canonical = registry.resolve_name(name)
            if canonical in self.options:
                raise ValueError("Worker options repeat a name or alias.")
            if not isinstance(settings, dict):
                raise TypeError("Worker options must be mappings.")
            self.options[canonical] = dict(settings)
        self.interrupt_on = interrupt_on
        self.attached = {spec["name"]: spec for spec in attached_workers}
        if self.attached.keys() & set(registry.names()):
            raise ValueError("Attached workers conflict with registry worker names.")
        self.initial = tuple(registry.resolve_name(name) for name in initial_agents)
        self.recorder = recorder
        self.tool_middleware = tuple(tool_middleware)
        self._workers = {}
        self._lock = RLock()

        @tool
        def search_agents(query: str, limit: int = 5) -> dict:
            """Find specialist workers by name, description, or tag; return metadata only."""
            if not 1 <= limit <= 20:
                return {"error": "limit must be between 1 and 20"}
            terms = re.findall(r"\w+", query.lower())
            matches = []
            for spec in registry.specs():
                description = f"{spec.name} {' '.join(spec.aliases)} {spec.description} {' '.join(sorted(spec.tags))}".lower()
                score = sum(term in description for term in terms)
                if score or not terms:
                    status = registry.availability(spec.name, constructor_kwargs=self.options.get(spec.name))
                    matches.append((score, spec.name, {
                        "name": spec.name, "description": spec.description,
                        "available": status.available, "issues": list(status.issues),
                    }))
            matches.sort(key=lambda item: (-item[0], item[1]))
            return {"agents": [item[2] for item in matches[:limit]]}

        @tool
        def load_agents(names: list[str], runtime: ToolRuntime) -> Command | ToolMessage:
            """Replace this turn's active workers. Load before task; [] clears selection."""
            calls = getattr(runtime.state["messages"][-1], "tool_calls", [])
            if sum(call["name"] == "load_agents" for call in calls) > 1:
                return ToolMessage(content="Call load_agents once with all needed names.",
                                   tool_call_id=runtime.tool_call_id, status="error")
            try:
                selected = list(dict.fromkeys(registry.resolve_name(name) for name in names))
                for name in selected:
                    self._get(name)
            except Exception as exc:
                return ToolMessage(content=str(exc), tool_call_id=runtime.tool_call_id, status="error")
            return Command(update={
                "active_registry_agents": selected,
                "messages": [ToolMessage(content=json.dumps({"loaded": selected}),
                                         tool_call_id=runtime.tool_call_id)],
            })

        self.tools = [search_agents, load_agents] if registry.names() else []

    def _get(self, name):
        with self._lock:
            if name not in self._workers:
                settings = dict(self.options.get(name, {}))
                worker_policy = settings.pop("interrupt_on", None)
                if worker_policy is not None and not isinstance(worker_policy, dict):
                    raise TypeError("Worker interrupt_on must be a policy mapping or None.")
                settings["interrupt_on"] = {
                    **(worker_policy or {}), **(self.interrupt_on or {}),
                } or None
                self._workers[name] = self.registry.as_subagent(name, llm=self.llm, **settings)["runnable"]
            return self._workers[name]

    def proxies(self):
        """Register metadata-only runnables with the factory's task dispatcher."""
        proxies = []
        for spec in self.registry.specs():
            def invoke(state, config, name=spec.name):
                return self._get(name).invoke(state, config=config)

            async def ainvoke(state, config, name=spec.name):
                return await self._get(name).ainvoke(state, config=config)

            proxies.append(_adapt_subagent({
                "name": spec.name, "description": spec.description,
                "runnable": RunnableLambda(invoke, afunc=ainvoke),
            }, self.recorder))
        return proxies

    def before_agent(self, state, runtime):
        if state.get("messages") and state["messages"][-1].type == "human":
            return {"active_registry_agents": list(self.initial)}
        return None

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    def after_agent(self, state, runtime):
        # A mixed direct-return batch can jump back to the model at this hook.
        if any(getattr(item, "_batch_returns_direct", lambda state: None)(state) is False
               for item in self.tool_middleware):
            return None
        return {"active_registry_agents": []}

    async def aafter_agent(self, state, runtime):
        return self.after_agent(state, runtime)

    def _model_request(self, request):
        selected = request.state.get("active_registry_agents", [])
        if set(selected) - set(self.registry.names()):
            raise ValueError("Checkpoint contains workers outside the configured catalog.")
        available = {**self.attached, **{
            name: {"description": self.registry.get_spec(name).description} for name in selected
        }}
        tools = []
        for entry in request.tools:
            name = entry.get("function", entry).get("name") if isinstance(entry, dict) else entry.name
            if name == "task":
                if not available:
                    continue
                description = "Delegate self-contained substantial work to a loaded specialist. Available agents:\n"
                description += "\n".join(f"- {name}: {spec['description']}" for name, spec in available.items())
                entry = entry.model_copy(update={"description": description})
            tools.append(entry)
        message = request.system_message
        guidance = ("Use search_agents and load_agents to discover and activate specialists before task. "
                    "Loading replaces this turn's selection. " if self.registry.names() else "")
        guidance += ("Direct work needs no delegation. Workers retain separate private tool and skill "
                     "state and honor mandatory reviews.")
        return request.override(tools=tools, system_message=SystemMessage(content=[
            *(message.content_blocks if message else []), {"type": "text", "text": guidance},
        ]))

    def wrap_model_call(self, request, handler):
        return handler(self._model_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._model_request(request))

    def _task_request(self, request):
        if request.tool_call["name"] != "task":
            return request
        name = request.tool_call["args"].get("subagent_type")
        if not isinstance(name, str):
            return ToolMessage(content="subagent_type must be a worker name string.",
                               tool_call_id=request.tool_call["id"], status="error")
        if name in self.attached:
            return request
        try:
            canonical = self.registry.resolve_name(name)
        except RegistryError:
            canonical = None
        if canonical not in request.state.get("active_registry_agents", []):
            return ToolMessage(content=f"Load configured worker {name!r} before delegating.",
                               tool_call_id=request.tool_call["id"], status="error")
        return request.override(tool_call={**request.tool_call, "args": {
            **request.tool_call["args"], "subagent_type": canonical,
        }})

    def wrap_tool_call(self, request, handler):
        selected = self._task_request(request)
        return selected if isinstance(selected, ToolMessage) else handler(selected)

    async def awrap_tool_call(self, request, handler):
        selected = self._task_request(request)
        return selected if isinstance(selected, ToolMessage) else await handler(selected)
