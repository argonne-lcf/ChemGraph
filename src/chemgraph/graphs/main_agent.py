"""Long-lived ChemGraph supervisor built from LangChain agent middleware."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage, convert_to_messages
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from chemgraph.graphs.single_agent import construct_single_agent_graph


DEFAULT_MAIN_AGENT_PROMPT = """\
You are the long-lived ChemGraph supervisor. You manage the conversation,
decompose user requests, delegate specialist work, and synthesize the returned
results into a clear response.

Rules:
1. Delegate computational chemistry tasks and any work requiring specialist
   tools through `task`; do not invent scientific results.
2. Give each subagent a self-contained task containing the relevant inputs and
   constraints. You may call multiple subagents when tasks are independent.
3. Review subagent results and give the user a complete response. If required
   input is missing, ask a clear question in your response.
4. Use supervisor-level tools directly when their descriptions match the task.
"""


def latest_assistant_text(messages: list[Any]) -> str:
    """Return the latest assistant message text from a message history."""
    for message in reversed(convert_to_messages(messages)):
        if isinstance(message, AIMessage):
            return str(message.text)
    return ""


def _preserve_terminal_tool_output(result: Any) -> Any:
    """Expose trailing worker tool output to Deep Agents as a final AI message."""
    if not isinstance(result, dict) or result.get("structured_response") is not None:
        return result

    messages = convert_to_messages(result.get("messages", []))
    if not messages or not isinstance(messages[-1], ToolMessage):
        return result

    trailing_text: list[str] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        if text := str(message.text).strip():
            trailing_text.append(text)
    if not trailing_text:
        return result

    return {
        **result,
        "messages": [
            *messages,
            AIMessage(content="\n".join(reversed(trailing_text))),
        ],
    }


def _adapt_subagent(spec: CompiledSubAgent) -> CompiledSubAgent:
    runnable = spec["runnable"]

    def invoke(state: Any, config: RunnableConfig) -> Any:
        return _preserve_terminal_tool_output(runnable.invoke(state, config=config))

    async def ainvoke(state: Any, config: RunnableConfig) -> Any:
        result = await runnable.ainvoke(state, config=config)
        return _preserve_terminal_tool_output(result)

    return {
        "name": spec["name"],
        "description": spec["description"],
        "runnable": RunnableLambda(invoke, afunc=ainvoke),
    }


def _validate_subagents(
    subagents: Sequence[CompiledSubAgent],
) -> list[CompiledSubAgent]:
    if not subagents:
        raise ValueError("At least one subagent must be registered.")

    validated: list[CompiledSubAgent] = []
    names: set[str] = set()
    for spec in subagents:
        name = spec.get("name")
        description = spec.get("description")
        runnable = spec.get("runnable")
        if not isinstance(name, str):
            raise TypeError("Subagent names must be strings.")
        if not name:
            raise ValueError("Subagent names must not be empty.")
        if name != name.strip():
            raise ValueError("Subagent names must not have surrounding whitespace.")
        if name in names:
            raise ValueError(f"Duplicate subagent name: {name!r}.")
        if not isinstance(description, str):
            raise TypeError(f"Subagent {name!r} must have a string description.")
        if not description.strip():
            raise ValueError(f"Subagent {name!r} must have a description.")
        if not callable(getattr(runnable, "invoke", None)) or not callable(
            getattr(runnable, "ainvoke", None)
        ):
            raise TypeError(
                f"Subagent {name!r} must provide invoke and ainvoke methods."
            )
        names.add(name)
        validated.append(_adapt_subagent(spec))
    return validated


def _validate_main_tools(main_tools: Sequence[BaseTool]) -> None:
    names = [getattr(item, "name", "") for item in main_tools]
    if any(not name for name in names):
        raise ValueError("Every supervisor tool must have a non-empty name.")
    if "task" in names:
        raise ValueError("Supervisor tool name 'task' is reserved for subagents.")
    if len(names) != len(set(names)):
        raise ValueError("Supervisor tool names must be unique.")


def construct_main_agent_graph(
    llm: Any,
    *,
    subagents: Sequence[CompiledSubAgent] | None = None,
    main_tools: list[BaseTool] | None = None,
    subagent_tools: list[BaseTool] | None = None,
    subagent_system_prompt: str | None = None,
    subagent_formatter_prompt: str | None = None,
    subagent_report_prompt: str | None = None,
    subagent_structured_output: bool = False,
    subagent_generate_report: bool = False,
    subagent_max_retries: int = 1,
    subagent_human_supervised: bool = False,
    subagent_terminal_tool_names: Collection[str] = (),
    system_prompt: str = DEFAULT_MAIN_AGENT_PROMPT,
    checkpointer: Any | None = None,
):
    """Construct a checkpointed supervisor with Deep Agents delegation."""
    if subagents is None:
        worker_kwargs: dict[str, Any] = {
            "tools": subagent_tools,
            "structured_output": subagent_structured_output,
            "generate_report": subagent_generate_report,
            "max_retries": subagent_max_retries,
            "human_supervised": subagent_human_supervised,
            "terminal_tool_names": subagent_terminal_tool_names,
            "checkpointer": None,
        }
        if subagent_system_prompt is not None:
            worker_kwargs["system_prompt"] = subagent_system_prompt
        if subagent_formatter_prompt is not None:
            worker_kwargs["formatter_prompt"] = subagent_formatter_prompt
        if subagent_report_prompt is not None:
            worker_kwargs["report_prompt"] = subagent_report_prompt
        worker = construct_single_agent_graph(llm, **worker_kwargs)
        subagents = [
            {
                "name": "chemgraph",
                "description": (
                    "Executes computational chemistry and molecular simulation "
                    "tasks with the existing ChemGraph single-agent workflow."
                ),
                "runnable": worker,
            }
        ]

    validated_subagents = _validate_subagents(subagents)
    supervisor_tools = list(main_tools or [])
    _validate_main_tools(supervisor_tools)
    middleware = SubAgentMiddleware(
        backend=StateBackend(),
        subagents=validated_subagents,
    )
    return create_agent(
        model=llm,
        tools=supervisor_tools,
        system_prompt=system_prompt,
        middleware=[middleware],
        checkpointer=checkpointer or InMemorySaver(),
        name="main_agent",
    )


__all__ = [
    "DEFAULT_MAIN_AGENT_PROMPT",
    "construct_main_agent_graph",
    "latest_assistant_text",
]
