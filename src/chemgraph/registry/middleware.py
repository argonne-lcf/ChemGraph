"""Expose a configured local tool catalog only when an agent requests it."""

import json
import re
from typing import Annotated, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command

from chemgraph.registry.tools import RegistryError, ToolRegistry


class RegistryToolState(AgentState):
    active_registry_tools: NotRequired[Annotated[list[str], PrivateStateAttr]]


class RegistryToolsMiddleware(AgentMiddleware):
    """Discover metadata, then bind and execute selected native tools.

    Only names live in checkpoints. Implementations are cached by ToolRegistry,
    while the active selection belongs to an individual agent turn.
    """

    state_schema = RegistryToolState

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.specs = {spec.name: spec for spec in registry.specs()}

        @tool
        def search_tools(query: str, limit: int = 5) -> dict:
            """Find local tools by name, description, or tag; return metadata only."""
            if not 1 <= limit <= 20:
                return {"error": "limit must be between 1 and 20"}
            terms = re.findall(r"\w+", query.lower())
            matches = []
            for spec in self.specs.values():
                text = f"{spec.name} {spec.description} {' '.join(sorted(spec.tags))}".lower()
                score = sum(term in text for term in terms)
                if score or not terms:
                    matches.append((score, spec.name, spec.description))
            matches.sort(key=lambda match: (-match[0], match[1]))
            return {
                "tools": [
                    {"name": name, "description": description}
                    for _, name, description in matches[:limit]
                ]
            }

        @tool
        def load_tools(names: list[str], runtime: ToolRuntime) -> Command | ToolMessage:
            """Replace active local tools with these names for this turn.

            Call once with all needed names, then use their native schemas on the
            next step. An empty list clears the selection. Does not execute tools.
            """
            calls = getattr(runtime.state["messages"][-1], "tool_calls", [])
            if sum(call["name"] == "load_tools" for call in calls) > 1:
                return ToolMessage(
                    content="Call load_tools once with all needed names.",
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            selected = list(dict.fromkeys(names))
            try:
                self._resolve(selected)
            except (RegistryError, ImportError, ValueError) as exc:
                return ToolMessage(
                    content=str(exc),
                    tool_call_id=runtime.tool_call_id,
                    status="error",
                )
            return Command(
                update={
                    "active_registry_tools": selected,
                    "messages": [
                        ToolMessage(
                            content=json.dumps({"loaded": selected}),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        self.tools = [search_tools, load_tools]
        self.validate_names(
            {
                "search_tools",
                "load_tools",
                "ls",
                "read_file",
                "write_file",
                "edit_file",
                "delete",
                "glob",
                "grep",
                "execute",
                "task",
                "write_todos",
            }
        )

    def validate_names(self, existing):
        collisions = self.specs.keys() & set(existing)
        if collisions:
            raise ValueError(
                f"Registry tool names conflict with attached tools: {sorted(collisions)}"
            )

    def _resolve(self, names):
        unknown = set(names) - self.specs.keys()
        if unknown:
            raise ValueError(
                f"Tools are not in the configured registry: {sorted(unknown)}"
            )
        return self.registry.resolve(names, require_available=True)

    def _model_request(self, request):
        existing = {
            entry.get("function", entry).get("name", entry.get("type"))
            if isinstance(entry, dict)
            else entry.name
            for entry in request.tools
        }
        self.validate_names(existing)
        selected = self._resolve(request.state.get("active_registry_tools", []))
        message = request.system_message
        guidance = (
            "Local ChemGraph tools are available through search_tools and load_tools. "
            "For supported local operations, load and call native tools before "
            "considering Python scripts or reading tool implementation source. "
            "Search for unfamiliar capabilities; skills can name tools to load directly. "
            "Use scripts when a capability is unavailable or a separate batch script "
            "is required, preserving the user's execution method. "
            "Load only the tools needed now; "
            "loading replaces the selection, which clears after this turn. "
            "Local tools run in the agent process on this host, independently of "
            "the file/shell backend. Use absolute host paths for their artifacts. "
            "Keep this tool workflow in this agent; delegated workers have their own tools."
        )
        return request.override(
            tools=[*request.tools, *selected],
            system_message=SystemMessage(
                content=[
                    *(message.content_blocks if message else []),
                    {"type": "text", "text": guidance},
                ]
            ),
        )

    def wrap_model_call(self, request, handler):
        return handler(self._model_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._model_request(request))

    def _tool_request(self, request):
        name = request.tool_call["name"]
        if name not in self.specs:
            return request
        if name not in request.state.get("active_registry_tools", []):
            return ToolMessage(
                content=f"Load {name!r} before calling it.",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return request.override(tool=self._resolve([name])[0])

    def wrap_tool_call(self, request, handler):
        selected = self._tool_request(request)
        return selected if isinstance(selected, ToolMessage) else handler(selected)

    async def awrap_tool_call(self, request, handler):
        selected = self._tool_request(request)
        return (
            selected if isinstance(selected, ToolMessage) else await handler(selected)
        )

    def before_agent(self, state, runtime):
        # New user input also resets a selection left by a failed turn. Resuming
        # an interrupted tool continues its checkpoint instead of this entry node.
        if state.get("messages") and state["messages"][-1].type == "human":
            return {"active_registry_tools": []}
        return None

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    def after_agent(self, state, runtime):
        return {"active_registry_tools": []}

    async def aafter_agent(self, state, runtime):
        return self.after_agent(state, runtime)
