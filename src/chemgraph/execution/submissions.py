"""Native fixed-template submission tool and ChemGraph MCP job protocol.

Embedding applications supply durable authorization and tracking, not tool
schemas or scientific implementations. Optional MCP imports remain lazy.
"""

import json
from typing import Protocol


class SubmissionController(Protocol):
    async def prepare_submission(self, call_id: str, calculation: str) -> dict: ...

    async def execute_submission(self, identifier: str) -> dict: ...


def submission_tool(controller: SubmissionController):
    from langchain.tools import ToolRuntime
    from langchain_core.tools import tool
    from langgraph.types import interrupt

    @tool
    async def submit_hpc(calculation: str, runtime: ToolRuntime) -> dict:
        """Submit an operator-configured calculation template after exact human approval."""
        operation = await controller.prepare_submission(
            runtime.tool_call_id, calculation
        )
        operation = await controller.execute_submission(operation["id"])
        while operation["state"] in {"approval", "uncertain"}:
            interrupt(
                {
                    "kind": "approval"
                    if operation["state"] == "approval"
                    else "reconciliation",
                    "operation_id": operation["id"],
                    "digest": operation["digest"],
                    "question": "Operator action required for this exact computational submission.",
                }
            )
            operation = await controller.execute_submission(operation["id"])
        return {
            "batch_id": operation["batch_id"],
            "state": operation["state"],
            "message": "The embedding service will collect results; do not resubmit this operation.",
        }

    return submit_hpc


class MCPJobClient:
    def __init__(self, server_url: str, token: str | None = None):
        self.server_url, self.token = server_url, token

    async def call(self, name, arguments):
        from fastmcp import Client

        async with Client(self.server_url, auth=self.token, timeout=60) as client:
            result = await client.call_tool(name, arguments)
        if result.is_error:
            raise RuntimeError("MCP reported an error")
        data = result.data
        if data is None:
            data = json.loads(
                "".join(p.text for p in result.content if hasattr(p, "text"))
            )
        if not isinstance(data, dict) or data.get("error"):
            raise RuntimeError("MCP returned an invalid job response")
        return data

    async def submit(self, operation):
        return await self.call(operation["tool"], operation["arguments"])

    async def status(self, batch_id):
        return await self.call("check_job_status", {"batch_id": batch_id})

    async def results(self, batch_id):
        return await self.call("get_job_results", {"batch_id": batch_id})

    async def cancel(self, batch_id):
        return await self.call("cancel_job", {"batch_id": batch_id})
