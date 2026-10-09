import pytest

from chemgraph.execution.submissions import MCPJobClient, submission_tool


@pytest.mark.asyncio
async def test_mcp_job_protocol_is_owned_by_chemgraph():
    class Client(MCPJobClient):
        async def call(self, name, arguments):
            return {"name": name, "arguments": arguments}

    client = Client("https://example.invalid/mcp")
    assert await client.submit({"tool": "configured", "arguments": {"params": {}}}) == {
        "name": "configured",
        "arguments": {"params": {}},
    }
    assert (await client.status("one"))["name"] == "check_job_status"
    assert (await client.results("one"))["name"] == "get_job_results"
    assert await client.cancel("one") == {
        "name": "cancel_job",
        "arguments": {"batch_id": "one"},
    }


def test_submission_schema_contains_only_template_selection():
    native = submission_tool(object())
    assert native.name == "submit_hpc"
    assert set(native.tool_call_schema.model_json_schema()["properties"]) == {
        "calculation"
    }
