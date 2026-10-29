"""Ask the model why it calls a reviewed tool, then hide the reason from the tool."""

import json
from copy import deepcopy

from langchain.agents.middleware import AgentMiddleware
from langchain_core.callbacks import adispatch_custom_event, dispatch_custom_event
from pydantic import BaseModel, Field, create_model

TOOL_REASON_EVENT = "chemgraph_tool_reason"
_REASON_DESCRIPTION = "One short sentence telling the user why this call is needed."


def _reason_schema(tool):
    """Return a JSON or Pydantic v2 schema to extend; other tools pass through."""
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        declares = "reason" in schema.get("properties", {})
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        declares = "reason" in schema.model_fields
    else:
        return None  # Inferred, single-input and Pydantic v1 schemas are left untouched.
    return None if declares else schema


class ToolReasonMiddleware(AgentMiddleware):
    """Require a ``reason`` argument on reviewed tools and publish it for display.

    The model sees an extended schema; execution receives a copy of the call
    without ``reason``, so original tool schemas never validate the extra field.
    """

    def __init__(self, tool_names, *, subagent_name: str | None = None):
        super().__init__()
        self.tool_names = frozenset(tool_names)
        self.subagent_name = subagent_name
        self._schemas = {}

    def _extended_schema(self, schema):
        key = json.dumps(schema, sort_keys=True, default=str) if isinstance(schema, dict) else schema
        if key not in self._schemas:
            if isinstance(schema, dict):
                extended = deepcopy(schema)
                extended.setdefault("properties", {})["reason"] = {
                    "type": "string", "description": _REASON_DESCRIPTION,
                }
                extended["required"] = [*extended.get("required", []), "reason"]
            else:
                extended = create_model(
                    f"{schema.__name__}WithReason", __base__=schema,
                    reason=(str, Field(description=_REASON_DESCRIPTION)),
                )
            self._schemas[key] = extended
        return self._schemas[key]

    def _extend(self, tool):
        if isinstance(tool, dict) or tool.name not in self.tool_names:
            return tool
        schema = _reason_schema(tool)
        if schema is None:
            return tool
        return tool.model_copy(update={"args_schema": self._extended_schema(schema)})

    def _model_request(self, request):
        return request.override(tools=[self._extend(tool) for tool in request.tools])

    def wrap_model_call(self, request, handler):
        return handler(self._model_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._model_request(request))

    def _split(self, request):
        """Return the execution request and the event payload, if any."""
        call = request.tool_call
        args = call.get("args")
        if (call["name"] not in self.tool_names or not isinstance(args, dict)
                or "reason" not in args or _reason_schema(request.tool) is None):
            return request, None
        reason = args["reason"]
        request = request.override(tool_call={
            **call, "args": {key: value for key, value in args.items() if key != "reason"},
        })
        if not isinstance(reason, str) or not reason.strip():
            return request, None
        data = {"tool_call_id": call.get("id"), "tool_name": call["name"], "reason": reason}
        if self.subagent_name:
            data["subagent_name"] = self.subagent_name
        return request, data

    @staticmethod
    def _config(request):
        return getattr(getattr(request, "runtime", None), "config", None)

    def wrap_tool_call(self, request, handler):
        request, data = self._split(request)
        if data is not None:
            try:
                dispatch_custom_event(TOOL_REASON_EVENT, data, config=self._config(request))
            except RuntimeError:
                pass  # No parent run: display is best effort; never block the call.
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        request, data = self._split(request)
        if data is not None:
            try:
                await adispatch_custom_event(TOOL_REASON_EVENT, data, config=self._config(request))
            except RuntimeError:
                pass  # No parent run: display is best effort; never block the call.
        return await handler(request)


__all__ = ["TOOL_REASON_EVENT", "ToolReasonMiddleware"]
