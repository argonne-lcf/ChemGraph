"""Experimental Codex SDK model adapter.

This module adapts the official ``openai-codex`` Python SDK to LangChain's
chat-model interface.  Codex is used only as the model backend; ChemGraph's
existing LangGraph workflow remains responsible for executing all exposed tools.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from copy import deepcopy
from dataclasses import dataclass
import logging
from typing import Any, Callable, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from chemgraph.agent.usage import TOKEN_FIELDS, USAGE_EVENT, normalize_usage

logger = logging.getLogger(__name__)

CODEX_MODEL_PREFIX = "codex:"

_BASE_INSTRUCTIONS = """\
You are acting only as a language-model backend for ChemGraph.
Follow the supplied conversation's system instructions and return only the JSON
object required by the output schema. You may request any applicable tool listed
in available_tools, subject to the supplied tool-choice constraint. These can
include filesystem reads and edits, command execution, delegation, web access,
and chemistry operations. Request them through the JSON tool_calls array;
ChemGraph executes them using its configured backends, approvals, and checkpoints.

When the task requires information from an available tool, request that tool
before answering. In particular, read a relevant skill using the listed
read_file tool before following or summarizing its full instructions. Base
capability claims on available tool schemas and returned results or errors.
The Codex thread's temporary working directory and read-only sandbox do not
determine access through ChemGraph tools; their backends resolve the paths.

Do not invoke Codex-native tools. Return requests for ChemGraph tools instead,
and use the returned tool results to continue the conversation.
"""


class CodexAuthenticationError(RuntimeError):
    """Raised when Codex is not using a ChatGPT subscription login."""


class CodexResponseError(RuntimeError):
    """Raised when Codex returns an invalid ChemGraph model decision."""


def _load_codex_sdk():
    """Import and return the official Codex SDK public classes lazily."""
    try:
        from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
    except ImportError as exc:
        raise ImportError(
            "Codex support requires the optional official SDK. Install it with "
            "`pip install 'chemgraph[codex]'`."
        ) from exc
    return Codex, CodexConfig, Sandbox, ApprovalMode


def _strip_codex_prefix(model_name: str) -> str:
    """Return the Codex model ID from a ``codex:<id>`` model name."""
    if not model_name.startswith(CODEX_MODEL_PREFIX):
        raise ValueError(
            f"Codex model names must use the {CODEX_MODEL_PREFIX}<model-id> prefix."
        )
    model_id = model_name.removeprefix(CODEX_MODEL_PREFIX).strip()
    if not model_id:
        raise ValueError("Codex model ID cannot be empty; use codex:<model-id>.")
    return model_id


def _model_dump(value: Any) -> Any:
    """Convert a Pydantic SDK object into plain Python data when possible."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    return value


def _require_chatgpt_account(account_response: Any) -> None:
    """Require the SDK's active account to be a ChatGPT-managed login."""
    data = _model_dump(account_response)
    if not isinstance(data, Mapping):
        raise CodexAuthenticationError(
            "Could not determine Codex authentication. Run `codex login` and "
            "sign in with ChatGPT."
        )

    account = _model_dump(data.get("account"))
    if account is None:
        raise CodexAuthenticationError(
            "No Codex login is available. Run `codex login` and sign in with "
            "ChatGPT before using a codex: model."
        )
    if not isinstance(account, Mapping):
        raise CodexAuthenticationError(
            "Could not determine the active Codex login type. Run `codex login "
            "status` and verify that ChatGPT authentication is active."
        )

    account_type = account.get("type")
    if account_type == "chatgpt":
        return
    if account_type == "apiKey":
        raise CodexAuthenticationError(
            "The active Codex login uses an API key, which ChemGraph's "
            "experimental codex: provider refuses. Run `codex logout`, then "
            "`codex login` and choose ChatGPT subscription authentication."
        )
    raise CodexAuthenticationError(
        f"The active Codex login type is {account_type!r}, not ChatGPT. Run "
        "`codex login` and sign in with ChatGPT."
    )


def inspect_account() -> Any:
    """Return the SDK's active account response for the reusable Codex login.

    The lookup mirrors what a model call does: an ephemeral working
    directory and cleared API-key variables, so only the login established
    by ``codex login`` is consulted.  Raises ``ImportError`` when the SDK is
    missing; SDK/transport failures propagate unchanged.
    """
    Codex, CodexConfig, _Sandbox, _ApprovalMode = _load_codex_sdk()
    with tempfile.TemporaryDirectory(prefix="chemgraph-codex-") as temp_dir:
        config = CodexConfig(
            cwd=temp_dir,
            env={"OPENAI_API_KEY": "", "CODEX_API_KEY": ""},
            client_name="chemgraph",
            client_title="ChemGraph",
        )
        with Codex(config=config) as codex:
            return codex.account()


def account_summary(account_response: Any) -> dict[str, Any]:
    """Reduce an SDK account response to ``{"type", "identity"}``.

    ``type`` is the login kind (``"chatgpt"``, ``"apiKey"``, ``None`` when
    logged out); ``identity`` is a display/identity string (email or
    account id) when the SDK exposes one.  Never raises.
    """
    data = _model_dump(account_response)
    account = _model_dump(data.get("account")) if isinstance(data, Mapping) else None
    if not isinstance(account, Mapping):
        return {"type": None, "identity": None}
    identity = None
    for key in ("email", "accountId", "account_id", "id", "planType", "plan_type"):
        value = account.get(key)
        if isinstance(value, str) and value:
            identity = value
            break
    kind = account.get("type")
    return {"type": kind if isinstance(kind, str) else None, "identity": identity}


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str, ensure_ascii=False)


def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    role_by_type = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
    }
    serialized = []
    for message in messages:
        item: dict[str, Any] = {
            "role": role_by_type.get(message.type, message.type),
            "content": _message_content(message),
        }
        if message.name is not None:
            item["name"] = message.name
        if isinstance(message, AIMessage) and message.tool_calls:
            item["tool_calls"] = deepcopy(message.tool_calls)
        if isinstance(message, ToolMessage):
            item["tool_call_id"] = message.tool_call_id
            item["status"] = message.status
        serialized.append(item)
    return serialized


def _normalize_tool_choice(tool_choice: Any, tool_names: set[str]) -> str | None:
    if tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice is True or (
        isinstance(tool_choice, str) and tool_choice in {"any", "required"}
    ):
        return "required"
    if tool_choice is False or tool_choice == "none":
        return "none"
    if isinstance(tool_choice, str):
        if tool_choice not in tool_names:
            raise ValueError(f"Unknown required tool {tool_choice!r}.")
        return tool_choice
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or name not in tool_names:
            raise ValueError(f"Invalid Codex tool choice: {tool_choice!r}.")
        return name
    raise ValueError(f"Unsupported Codex tool choice: {tool_choice!r}.")


def _decision_schema(
    tool_names: list[str],
    tool_choice: str | None,
    parallel_tool_calls: bool,
) -> dict[str, Any]:
    if tool_names:
        item_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": tool_names},
                "arguments": {"type": "string"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
    else:
        item_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    calls_schema: dict[str, Any] = {
        "type": "array",
        "items": item_schema,
    }
    if not tool_names or tool_choice == "none":
        calls_schema["maxItems"] = 0
    elif tool_choice == "required" or tool_choice in tool_names:
        calls_schema["minItems"] = 1
    if not parallel_tool_calls:
        calls_schema["maxItems"] = 1

    return {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": calls_schema,
        },
        "required": ["content", "tool_calls"],
        "additionalProperties": False,
    }


def _decision_prompt(
    messages: list[BaseMessage],
    tools: tuple[dict[str, Any], ...],
    tool_choice: str | None,
) -> str:
    tool_instruction = (
        "No tools are available. Return the answer in content and an empty "
        "tool_calls array."
    )
    if tools:
        tool_instruction = (
            "Request the necessary ChemGraph tools when the task requires "
            "information or actions they provide. Answer in content with an "
            "empty tool_calls array when the available evidence is sufficient."
        )
    if tool_choice == "required":
        tool_instruction = "Request at least one of the available tools."
    elif tool_choice == "none":
        tool_instruction = "Do not request a tool; answer in content."
    elif tool_choice:
        tool_instruction = f"Request the {tool_choice!r} tool."

    if tools and tool_choice != "none":
        tool_instruction += (
            " Each tool call's arguments field must be a JSON-encoded object "
            "string matching that tool's parameters schema. ChemGraph will "
            "execute these requests and return their results."
        )

    payload = {
        "conversation": _serialize_messages(messages),
        "available_tools": list(tools),
    }
    return (
        f"{tool_instruction}\n"
        "Return only the schema-constrained decision for this conversation:\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
    )


def _usage_metadata(result: Any) -> dict | None:
    usage = _model_dump(getattr(result, "usage", None))
    if not isinstance(usage, Mapping) or not isinstance(usage.get("total"), Mapping):
        return None
    counts = normalize_usage(usage["total"])
    if any(counts[key] is None for key in TOKEN_FIELDS[:3]):
        return None
    metadata = {key: counts[key] for key in TOKEN_FIELDS[:3]}
    if counts["cached_input_tokens"] is not None:
        metadata["input_token_details"] = {"cache_read": counts["cached_input_tokens"]}
    if counts["reasoning_output_tokens"] is not None:
        metadata["output_token_details"] = {"reasoning": counts["reasoning_output_tokens"]}
    return metadata


@dataclass
class _CodexResult:
    final_response: str | None
    usage: dict | None


def _emit_usage(run_manager, usage, model, *, complete):
    if run_manager is None or not isinstance(usage, Mapping):
        return
    counts = normalize_usage(usage.get("total"))
    try:
        CallbackManager(handlers=run_manager.handlers).on_custom_event(
            USAGE_EVENT,
            {"call_id": str(run_manager.run_id), "counts": counts,
             "raw_usage": usage, "model": model, "provider": "codex",
             "complete": complete},
            run_id=run_manager.run_id,
        )
    except Exception:
        logger.debug("Could not emit Codex usage.", exc_info=True)


def _run_codex_turn(thread, prompt, *, run_manager, model, **kwargs):
    """Collect public SDK notifications, retaining usage before SDK/parser errors.

    Each ChemGraph model invocation owns a fresh ephemeral SDK thread, so its
    cumulative total includes all internal requests. Reusing this helper for
    correction attempts on that thread replaces, rather than adds, totals.
    Unlike SDK ``thread.run``, interrupted turns raise rather than returning a
    potentially unfinished decision; their available usage is still retained.
    """
    turn = thread.turn(prompt, **kwargs)
    stream = turn.stream()
    usage = None
    messages = []
    completed = None
    try:
        for event in stream:
            payload = _model_dump(event.payload)
            if not isinstance(payload, Mapping):
                continue
            if event.method == "thread/tokenUsage/updated" and payload.get("turnId") == turn.id:
                usage = payload.get("tokenUsage")
                _emit_usage(run_manager, usage, model, complete=False)
            elif event.method == "item/completed" and payload.get("turnId") == turn.id:
                item = payload.get("item", {})
                if item.get("type") == "agentMessage":
                    messages.append(item)
            elif event.method == "turn/completed" and payload.get("turn", {}).get("id") == turn.id:
                completed = payload["turn"]
    finally:
        stream.close()
    if completed is None:
        raise RuntimeError("Codex turn completed event not received")
    if completed.get("status") != "completed":
        error = completed.get("error") or {}
        raise RuntimeError(error.get("message") or f"Codex turn {completed.get('status')}")
    _emit_usage(run_manager, usage, model, complete=True)
    final = next((item["text"] for item in reversed(messages) if item.get("phase") == "final_answer"), None)
    if final is None:
        final = next((item["text"] for item in reversed(messages) if item.get("phase") is None), None)
    return _CodexResult(final, usage)


class CodexChatModel(BaseChatModel):
    """LangChain chat model backed by the official Codex Python SDK."""

    model_id: str
    bound_tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | None = None
    parallel_tool_calls: bool = True

    @property
    def _llm_type(self) -> str:
        return "openai-codex-sdk"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model_id, "provider": "codex-subscription"}

    def bind_tools(
        self,
        tools: Sequence[
            dict[str, Any] | type | Callable[..., Any] | BaseTool
        ],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "CodexChatModel":
        converted = tuple(convert_to_openai_tool(tool) for tool in tools)
        functions = [tool.get("function") for tool in converted]
        if not all(isinstance(function, Mapping) for function in functions):
            raise ValueError("CodexChatModel supports only function tools.")
        tool_names = {function["name"] for function in functions}
        normalized_choice = _normalize_tool_choice(tool_choice, tool_names)
        parallel = bool(kwargs.pop("parallel_tool_calls", True))
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported Codex tool options: {unsupported}.")
        return self.model_copy(
            update={
                "bound_tools": deepcopy(converted),
                "tool_choice": normalized_choice,
                "parallel_tool_calls": parallel,
            }
        )

    def validate_authentication(self) -> None:
        """Validate that the reusable Codex login is ChatGPT-managed."""
        _require_chatgpt_account(inspect_account())

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("CodexChatModel does not support stop sequences.")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported Codex invocation options: {unsupported}.")

        Codex, CodexConfig, Sandbox, ApprovalMode = _load_codex_sdk()
        tool_names = [tool["function"]["name"] for tool in self.bound_tools]
        schema = _decision_schema(
            tool_names,
            self.tool_choice,
            self.parallel_tool_calls,
        )
        prompt = _decision_prompt(messages, self.bound_tools, self.tool_choice)

        with tempfile.TemporaryDirectory(prefix="chemgraph-codex-") as temp_dir:
            config = CodexConfig(
                cwd=temp_dir,
                env={"OPENAI_API_KEY": "", "CODEX_API_KEY": ""},
                client_name="chemgraph",
                client_title="ChemGraph",
            )
            with Codex(config=config) as codex:
                _require_chatgpt_account(codex.account())
                thread = codex.thread_start(
                    approval_mode=ApprovalMode.deny_all,
                    base_instructions=_BASE_INSTRUCTIONS,
                    cwd=temp_dir,
                    ephemeral=True,
                    model=self.model_id,
                    sandbox=Sandbox.read_only,
                )
                result = _run_codex_turn(
                    thread,
                    prompt,
                    run_manager=run_manager,
                    model=self.model_id,
                    approval_mode=ApprovalMode.deny_all,
                    output_schema=schema,
                    sandbox=Sandbox.read_only,
                )

        raw_response = getattr(result, "final_response", None)
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise CodexResponseError("Codex returned no final response.")
        try:
            decision = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise CodexResponseError("Codex returned an invalid JSON decision.") from exc
        if not isinstance(decision, Mapping):
            raise CodexResponseError("Codex decision must be a JSON object.")

        content = decision.get("content")
        calls = decision.get("tool_calls")
        if not isinstance(content, str) or not isinstance(calls, list):
            raise CodexResponseError(
                "Codex decision must contain string content and a tool_calls list."
            )
        if not self.parallel_tool_calls and len(calls) > 1:
            raise CodexResponseError("Codex returned parallel tool calls when disabled.")
        if self.tool_choice in {"required", *tool_names} and not calls:
            raise CodexResponseError("Codex did not return the required tool call.")
        if self.tool_choice == "none" and calls:
            raise CodexResponseError("Codex returned a tool call when tools were disabled.")

        tool_calls = []
        for call in calls:
            if not isinstance(call, Mapping):
                raise CodexResponseError("Codex returned an invalid tool call.")
            name = call.get("name")
            arguments = call.get("arguments")
            if name not in tool_names or not isinstance(arguments, str):
                raise CodexResponseError(f"Codex returned an unknown tool call: {name!r}.")
            if self.tool_choice not in {None, "required", name}:
                raise CodexResponseError(
                    f"Codex called {name!r} instead of required tool {self.tool_choice!r}."
                )
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise CodexResponseError(
                    f"Codex returned invalid arguments for tool {name!r}."
                ) from exc
            if not isinstance(parsed_arguments, dict):
                raise CodexResponseError(
                    f"Codex tool arguments for {name!r} must be a JSON object."
                )
            tool_calls.append(
                {
                    "name": name,
                    "args": parsed_arguments,
                    "id": f"call_{uuid.uuid4().hex}",
                }
            )

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=_usage_metadata(result),
            response_metadata={
                "model": self.model_id, "provider": "codex",
                "usage": (result.usage or {}).get("total"), "codex_usage": result.usage,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def load_codex_model(
    model_name: str,
    *,
    validate_authentication: bool = True,
) -> CodexChatModel:
    """Load an experimental Codex subscription-backed chat model."""
    model = CodexChatModel(model_id=_strip_codex_prefix(model_name))
    if validate_authentication:
        model.validate_authentication()
    return model
