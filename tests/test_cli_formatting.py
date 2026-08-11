import json

from langchain_core.messages import AIMessage

from chemgraph.agent.main_session import MainAgentTurnResult
from chemgraph.cli.formatting import _content_text, console, format_response


def test_content_text_normalizes_structured_blocks():
    content = [
        {"type": "reasoning", "reasoning": "internal"},
        {"type": "text", "text": "Optimization complete"},
        "!",
        {"type": "tool_call", "name": "run_ase"},
    ]

    assert _content_text(content) == "Optimization complete!"


def test_format_response_renders_structured_ai_content():
    result = {
        "messages": [
            {
                "type": "ai",
                "content": [{"type": "text", "text": "Dictionary response"}],
            },
            AIMessage(
                content=[{"type": "text", "text": "Optimization complete"}]
            ),
        ]
    }

    with console.capture() as capture:
        format_response(result)

    output = capture.get()
    assert "ChemGraph Response" in output
    assert "Optimization complete" in output


def test_format_response_detects_atomic_json_in_structured_content():
    structure = json.dumps(
        {
            "numbers": [8, 1, 1],
            "positions": [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        }
    )
    result = {
        "messages": [
            AIMessage(content=[{"type": "text", "text": structure}]),
            AIMessage(content=[{"type": "text", "text": "Water optimized"}]),
        ]
    }

    with console.capture() as capture:
        format_response(result)

    output = capture.get()
    assert "ChemGraph Response" in output
    assert "Water optimized" in output
    assert "Molecular Structure Data" in output


def test_format_response_renders_main_agent_turn_result():
    result = MainAgentTurnResult(
        thread_id="thread-1",
        status="waiting_for_user",
        assistant_response="Delegated calculation complete.",
        interrupts=(),
        state={},
    )

    with console.capture() as capture:
        format_response(result)

    output = capture.get()
    assert "ChemGraph Response" in output
    assert "Delegated calculation complete" in output
