"""Review, confine and journal every tool call of a scoped main-agent graph."""

from __future__ import annotations

import asyncio
from copy import copy

from langchain.agents.middleware import AgentMiddleware
from langgraph.types import interrupt

from chemgraph.runtime.artifacts import output_manifest
from chemgraph.runtime.context import (
    ScopedExecutionError, current_context, invocation, run_context,
)
from chemgraph.runtime.journal import (
    APPROVED, BLOCKED, PROPOSED, REJECTED, error_message,
)


class ScopedToolsMiddleware(AgentMiddleware):
    """The single approval and execution gate of graphs built with ``approval_mode="scoped"``.

    Framework helpers run directly. Tools outside the operator policy are
    answered with an error message, since they provably did not run. Every
    permitted call is prepared, paused for an exact decision recorded in the
    journal, and executed with its prepared arguments. The resume value is
    never treated as authorization.
    """

    def wrap_tool_call(self, request, handler):
        raise ScopedExecutionError("Scoped graphs require the asynchronous graph API.")

    async def awrap_tool_call(self, request, handler):
        context = current_context()
        if context is None:
            raise ScopedExecutionError(
                "Scoped graphs must run inside chemgraph.runtime.run_context()."
            )
        call = request.tool_call
        policy, journal = context.policy, context.journal
        if call["name"] in policy.helpers:
            return await handler(request)
        if call["name"] not in policy.tools:
            return error_message(call, f"Tool {call['name']!r} is not available in this "
                                       "scoped run. Use only the permitted tools.")
        record = await journal.propose(context, call)
        while record["state"] in {PROPOSED, APPROVED} and not journal.authorized(record):
            # Re-executed on resume: decisions come from the journal, not the resume value.
            interrupt({
                "kind": "approval", "operation_id": record["id"], "tool_call_id": call["id"],
                "tool": call["name"], "digest": record["prepared"]["digest"],
                "prepared": record["prepared"],
            })
            record = await journal.get(record["id"])
        if record["state"] == BLOCKED:
            return error_message(call, f"Not executed: {record['error']}")
        if record["state"] == REJECTED:
            reason = (record["decision"] or {}).get("message") or "The operator rejected it."
            return error_message(call, f"Not executed: {reason} Do not retry unless asked.")

        async def invoke(action):
            outputs = set()
            with invocation(action, outputs):
                message = await handler(_with_arguments(request, {**call, "args": action.arguments}))
            return message, await asyncio.to_thread(output_manifest, context.workspace, outputs)

        async with context.lock:
            with run_context(context):
                return await journal.execute(context, call, invoke)


def _with_arguments(request, call):
    if hasattr(request, "override"):
        return request.override(tool_call=call)
    resolved = copy(request)
    resolved.tool_call = call
    return resolved
