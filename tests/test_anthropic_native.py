from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from chemgraph.models.protocols.anthropic_native import CachingChatAnthropic


TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]


def _client() -> CachingChatAnthropic:
    return CachingChatAnthropic(
        model="claude-test",
        api_key="test-key",
        max_tokens=32,
    )


def test_plain_system_prompt_gets_ephemeral_cache_breakpoint():
    payload = _client()._get_request_payload(
        [
            SystemMessage(content="Stable system prompt."),
            HumanMessage(content="Dynamic user turn."),
        ],
        tools=TOOLS,
    )

    assert payload["system"] == [
        {
            "type": "text",
            "text": "Stable system prompt.",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    assert payload["tools"] == TOOLS


def test_block_system_prompt_is_preserved():
    system = [
        {
            "type": "text",
            "text": "Caller-owned cache breakpoint.",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]

    payload = _client()._get_request_payload(
        [
            SystemMessage(content=system),
            HumanMessage(content="Dynamic user turn."),
        ],
    )

    assert payload["system"] == system


def test_absent_system_prompt_is_preserved():
    payload = _client()._get_request_payload(
        [HumanMessage(content="Dynamic user turn.")],
    )

    assert "system" not in payload


def test_empty_system_prompt_is_not_wrapped():
    payload = _client()._get_request_payload(
        [
            SystemMessage(content=""),
            HumanMessage(content="Dynamic user turn."),
        ],
    )

    assert payload["system"] == ""
