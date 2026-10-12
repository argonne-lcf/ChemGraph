"""Shared scoped execution for workspace graphs and native worker tool nodes."""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from chemgraph.execution.scoped import (
    ARTIFACT_KEY,
    WORKER_ARTIFACT_KEY,
    ScopedExecutionError,
    current_execution,
    execute_scoped_tool,
)
from chemgraph.registry.tools import BUILTIN_TOOL_SPECS, ToolSpec


class ScopedToolsMiddleware(AgentMiddleware):
    def __init__(self, *, registry=None, tools=()):
        builtins = {spec.name: spec for spec in BUILTIN_TOOL_SPECS}
        self.specs = {}
        for tool in tools or ():
            name = getattr(tool, "name", None)
            # submit_hpc has its own durable submission/interrupt controller.
            if name and name != "submit_hpc":
                self.specs[name] = builtins.get(name, ToolSpec(name, "", None))
        if registry is not None:
            self.specs.update({spec.name: spec for spec in registry.specs()})
        for name in ("write_file", "edit_file", "delete"):
            self.specs[name] = ToolSpec(name, "Checkpoint workspace mutation", None)

    def wrap_tool_call(self, request, handler):
        if current_execution() is not None:
            raise ScopedExecutionError(
                "Scoped tool execution requires the async graph API"
            )
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        if current_execution() is None:
            return await handler(request)
        name = request.tool_call["name"]
        if name == "execute":
            raise ScopedExecutionError(
                "Host shell execution is unavailable in scoped catalogs"
            )
        if name == "task":
            # The child locks individual native calls. Locking task would deadlock.
            result = await handler(request)
            if isinstance(result, Command) and isinstance(result.update, dict):
                update = dict(result.update)
                artifacts = update.pop(WORKER_ARTIFACT_KEY, [])
                if artifacts:
                    update["messages"] = [
                        message.model_copy(
                            update={
                                "additional_kwargs": {
                                    **message.additional_kwargs,
                                    ARTIFACT_KEY: artifacts,
                                }
                            }
                        )
                        if isinstance(message, ToolMessage)
                        else message
                        for message in update.get("messages", [])
                    ]
                result = Command(
                    graph=result.graph,
                    update=update,
                    resume=result.resume,
                    goto=result.goto,
                )
            return result
        spec = self.specs.get(name)
        if spec is None:
            return await handler(request)
        return await execute_scoped_tool(spec, request, handler)
