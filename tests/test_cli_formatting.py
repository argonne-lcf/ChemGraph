"""Tests for ChemGraph CLI response formatting."""

from chemgraph.cli.formatting import _is_atomic_json, format_response


def test_is_atomic_json_handles_mcp_content_blocks():
    content = [{"type": "text", "text": '{"status": "success"}'}]

    assert _is_atomic_json(content) is False


def test_format_response_handles_mcp_tool_content_blocks():
    result = {
        "messages": [
            {
                "type": "human",
                "content": "Run a calculation.",
            },
            {
                "type": "ai",
                "content": "Calling a tool.",
            },
            {
                "type": "tool",
                "content": [{"type": "text", "text": '{"status": "success"}'}],
            },
            {
                "type": "ai",
                "content": "The calculation completed.",
            },
        ]
    }

    format_response(result)
