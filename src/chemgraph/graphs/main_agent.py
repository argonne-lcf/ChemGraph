"""Long-lived ChemGraph supervisor built from LangChain agent middleware."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware import SubAgentMiddleware
from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage, convert_to_messages
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt

from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.memory.subagent_recorder import SubagentRunRecorder


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
5. When available, use `chemgraph` for molecular construction, simulation,
   calculator use, and other computational chemistry work. Use `deepagent` for
   repository exploration, coding, testing, file analysis, and other long
   workspace tasks when that specialist is available.
6. Do not launch parallel workspace-mutating tasks. Parallel delegation is
   appropriate only for independent read-only work.
7. If deepagent is available, do not generate the code or file edits yourself;
   delegate those tasks to deepagent.
"""


DEFAULT_DEEPAGENT_PROMPT = """\
You are ChemGraph's workspace specialist. Complete repository exploration,
coding, testing, file analysis, and other multi-step workspace tasks. Use the
built-in filesystem and execution tools when needed. Do not perform molecular
simulations or invent chemistry results; return those tasks to the supervisor
for delegation to the `chemgraph` specialist.

The calling supervisor sees only your final assistant message. Return a concise,
self-contained report including important results, changed paths, commands run,
and any failures or unresolved risks.
"""


_DEEPAGENT_INTERRUPT_ON = {
    "execute": {"allowed_decisions": ["approve", "reject"]},
    "write_file": {"allowed_decisions": ["approve", "reject"]},
    "edit_file": {"allowed_decisions": ["approve", "reject"]},
    "delete": {"allowed_decisions": ["approve", "reject"]},
}


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


def _adapt_subagent(
    spec: CompiledSubAgent,
    recorder: SubagentRunRecorder | None = None,
) -> CompiledSubAgent:
    runnable = spec["runnable"]

    def invoke(state: Any, config: RunnableConfig) -> Any:
        run_id = recorder.start(spec["name"], state, config) if recorder else None
        try:
            result = runnable.invoke(state, config=config)
        except GraphInterrupt:
            if recorder and run_id:
                recorder.interrupted(run_id)
            raise
        except Exception as exc:
            if recorder and run_id:
                recorder.failed(run_id, exc)
            raise
        result = _preserve_terminal_tool_output(result)
        if recorder and run_id:
            recorder.completed(run_id, result)
        return result

    async def ainvoke(state: Any, config: RunnableConfig) -> Any:
        run_id = recorder.start(spec["name"], state, config) if recorder else None
        try:
            result = await runnable.ainvoke(state, config=config)
        except GraphInterrupt:
            if recorder and run_id:
                recorder.interrupted(run_id)
            raise
        except Exception as exc:
            if recorder and run_id:
                recorder.failed(run_id, exc)
            raise
        result = _preserve_terminal_tool_output(result)
        if recorder and run_id:
            recorder.completed(run_id, result)
        return result

    return {
        "name": spec["name"],
        "description": spec["description"],
        "runnable": RunnableLambda(invoke, afunc=ainvoke),
    }


def _validate_subagents(
    subagents: Sequence[CompiledSubAgent],
    recorder: SubagentRunRecorder | None = None,
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
        validated.append(_adapt_subagent(spec, recorder))
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
    enable_deepagent: bool = False,
    deepagent_backend: BackendProtocol | None = None,
    deepagent_recursion_limit: int = 50,
    system_prompt: str = DEFAULT_MAIN_AGENT_PROMPT,
    checkpointer: Any | None = None,
    subagent_recorder: SubagentRunRecorder | None = None,
):
    """Construct a checkpointed supervisor with Deep Agents delegation."""
    if deepagent_recursion_limit <= 0:
        raise ValueError("deepagent_recursion_limit must be positive.")
    if deepagent_backend is not None and not enable_deepagent:
        raise ValueError("deepagent_backend requires enable_deepagent=True.")

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
        registered_subagents: list[CompiledSubAgent] = [
            {
                "name": "chemgraph",
                "description": (
                    "Executes computational chemistry and molecular simulation "
                    "tasks with the existing ChemGraph single-agent workflow."
                ),
                "runnable": worker,
            }
        ]
    else:
        registered_subagents = list(subagents)

    if enable_deepagent:
        workspace_agent = create_deep_agent(
            model=llm,
            tools=[],
            system_prompt=DEFAULT_DEEPAGENT_PROMPT,
            backend=(
                deepagent_backend
                if deepagent_backend is not None
                else StateBackend()
            ),
            interrupt_on=_DEEPAGENT_INTERRUPT_ON,
            checkpointer=None,
            name="deepagent",
        ).with_config({"recursion_limit": deepagent_recursion_limit})
        registered_subagents.append(
            {
                "name": "deepagent",
                "description": (
                    "Explores repositories and workspaces, edits files, runs tests, "
                    "and completes long multi-step coding or data-analysis tasks. "
                    "Use chemgraph instead for molecular simulations."
                ),
                "runnable": workspace_agent,
            }
        )

    validated_subagents = _validate_subagents(registered_subagents, subagent_recorder)
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
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        name="main_agent",
    )


__all__ = [
    "DEFAULT_DEEPAGENT_PROMPT",
    "DEFAULT_MAIN_AGENT_PROMPT",
    "construct_main_agent_graph",
    "latest_assistant_text",
]
