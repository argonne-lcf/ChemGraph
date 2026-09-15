"""Experimental Codex SDK model adapter.

This module adapts the official ``openai-codex`` Python SDK to LangChain's
chat-model interface.  Codex is used only as the model backend; ChemGraph's
existing LangGraph workflow remains responsible for executing chemistry tools.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from chemgraph.models._codex_schema import argument_schema, within_limits

CODEX_MODEL_PREFIX = "codex:"
_MAX_DECISION_ATTEMPTS = 3

_BASE_INSTRUCTIONS = """\
You are acting only as a language-model backend for ChemGraph.
Do not inspect files, run commands, browse the web, or use any Codex-provided
tools. Follow the conversation's system instructions and return only the JSON
object required by the supplied output schema. ChemGraph, not Codex, executes
all requested chemistry tools.
"""


class CodexAuthenticationError(RuntimeError):
    """Raised when Codex is not using a ChatGPT subscription login."""


class CodexResponseError(RuntimeError):
    """Raised when Codex returns an invalid ChemGraph model decision."""


class _MalformedDecision(CodexResponseError):
    """A response encoding failure that can be corrected without executing tools."""


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


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str, ensure_ascii=False)


def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    role_by_type = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
    }
    return [
        {
            "role": role_by_type.get(message.type, message.type),
            "content": _message_content(message),
        }
        for message in messages
    ]


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
    argument_schemas: dict[str, dict[str, Any] | None],
    tool_choice: str | None,
    parallel_tool_calls: bool,
) -> dict[str, Any]:
    tool_names = list(argument_schemas)
    if tool_names:
        allowed_names = [tool_choice] if tool_choice in tool_names else tool_names
        item_schema: dict[str, Any] = {
            "anyOf": [{
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [name]},
                    "arguments": argument_schemas[name] or {"type": "string"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            } for name in allowed_names],
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
    if not parallel_tool_calls and calls_schema.get("maxItems") != 0:
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


def _tool_argument_schemas(tools, tool_choice, parallel_tool_calls):
    schemas = {tool["function"]["name"]: None for tool in tools}
    if not within_limits(_decision_schema(schemas, tool_choice, parallel_tool_calls)):
        raise ValueError("Codex output schema exceeds provider limits; bind fewer tools.")
    for tool in tools:
        function = tool["function"]
        name = function["name"]
        schemas[name] = argument_schema(function.get("parameters"))
        if not within_limits(_decision_schema(schemas, tool_choice, parallel_tool_calls)):
            schemas[name] = None
    return schemas


def _decision_prompt(
    messages: list[BaseMessage],
    tools: tuple[dict[str, Any], ...],
    tool_choice: str | None,
    argument_schemas: dict[str, dict[str, Any] | None],
) -> str:
    tool_instruction = (
        "No tools are available. Return the answer in content and an empty "
        "tool_calls array."
    )
    if tools:
        tool_instruction = (
            "Either answer in content with an empty tool_calls array, or request "
            "the necessary tools."
        )
    if tool_choice == "required":
        tool_instruction = "Request at least one of the available tools."
    elif tool_choice == "none":
        tool_instruction = "Do not request a tool; answer in content."
    elif tool_choice:
        tool_instruction = f"Request the {tool_choice!r} tool."

    if tools and tool_choice != "none":
        tool_instruction += (
            " Follow argument_encodings for each tool: object means return arguments "
            "as a JSON object; json_string means a JSON-encoded object string. "
            "Match that tool's parameters schema. Omit optional arguments when "
            "using their defaults; do not substitute null for omission. "
            "ChemGraph will execute these requests and return their results."
        )

    payload = {
        "conversation": _serialize_messages(messages),
        "available_tools": list(tools),
        "argument_encodings": {
            name: "object" if schema is not None else "json_string"
            for name, schema in argument_schemas.items()
        },
    }
    return (
        f"{tool_instruction}\n"
        "Return only the schema-constrained decision for this conversation:\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
    )


def _reject_json_constant(_value):
    raise ValueError("Non-finite numbers are not JSON values.")


def _decode_json(raw, stage):
    try:
        return json.loads(raw, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise _MalformedDecision(
            f"{stage}: {exc.msg} at line {exc.lineno}, column {exc.colno}."
        ) from None
    except ValueError:
        raise _MalformedDecision(f"{stage}: non-finite number in JSON.") from None


def _parse_decision(raw_response, argument_schemas, tool_choice, parallel_tool_calls):
    """Decode the entire batch; tool frameworks validate individual parameters."""
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise _MalformedDecision("Codex returned no final response.")
    decision = _decode_json(raw_response, "Codex decision JSON")
    if not isinstance(decision, dict) or set(decision) != {"content", "tool_calls"}:
        raise _MalformedDecision("Codex decision must contain only content and tool_calls.")
    content, calls = decision["content"], decision["tool_calls"]
    if not isinstance(content, str) or not isinstance(calls, list):
        raise _MalformedDecision(
            "Codex decision must contain string content and a tool_calls list."
        )
    if not parallel_tool_calls and len(calls) > 1:
        raise CodexResponseError("Codex returned parallel tool calls when disabled.")
    if tool_choice in {"required", *argument_schemas} and not calls:
        raise CodexResponseError("Codex did not return the required tool call.")
    if tool_choice == "none" and calls:
        raise CodexResponseError("Codex returned a tool call when tools were disabled.")

    decoded = []
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise _MalformedDecision("Codex tool call must contain only name and arguments.")
        name, arguments = call["name"], call["arguments"]
        if not isinstance(name, str) or name not in argument_schemas:
            raise CodexResponseError("Codex returned an unknown tool call.")
        if tool_choice not in {None, "required", name}:
            raise CodexResponseError(
                f"Codex called {name!r} instead of required tool {tool_choice!r}."
            )
        if argument_schemas[name] is None:
            if not isinstance(arguments, str):
                raise _MalformedDecision(
                    f"Codex arguments for {name!r} must be a JSON-encoded object string."
                )
            arguments = _decode_json(arguments, f"Codex arguments JSON for {name!r}")
        if not isinstance(arguments, dict):
            raise _MalformedDecision(f"Codex arguments for {name!r} must be a JSON object.")
        decoded.append({"name": name, "args": arguments})
    return content, [{**call, "id": f"call_{uuid.uuid4().hex}"} for call in decoded]


def _usage_metadata(result: Any) -> dict[str, int] | None:
    usage = getattr(result, "usage", None)
    last = getattr(usage, "last", None)
    if last is None:
        return None
    values = _model_dump(last)
    if not isinstance(values, Mapping):
        return None

    def token_value(camel_name: str, snake_name: str) -> Any:
        if camel_name in values:
            return values[camel_name]
        return values.get(snake_name)

    try:
        return {
            "input_tokens": int(token_value("inputTokens", "input_tokens")),
            "output_tokens": int(token_value("outputTokens", "output_tokens")),
            "total_tokens": int(token_value("totalTokens", "total_tokens")),
        }
    except (KeyError, TypeError, ValueError):
        return None


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
        if len(tool_names) != len(functions):
            raise ValueError("Codex function tools must have distinct names.")
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
        Codex, CodexConfig, _Sandbox, _ApprovalMode = _load_codex_sdk()
        with tempfile.TemporaryDirectory(prefix="chemgraph-codex-") as temp_dir:
            config = CodexConfig(
                cwd=temp_dir,
                env={"OPENAI_API_KEY": "", "CODEX_API_KEY": ""},
                client_name="chemgraph",
                client_title="ChemGraph",
            )
            with Codex(config=config) as codex:
                _require_chatgpt_account(codex.account())

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        if stop:
            raise ValueError("CodexChatModel does not support stop sequences.")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported Codex invocation options: {unsupported}.")

        Codex, CodexConfig, Sandbox, ApprovalMode = _load_codex_sdk()
        argument_schemas = _tool_argument_schemas(
            self.bound_tools, self.tool_choice, self.parallel_tool_calls,
        )
        schema = _decision_schema(
            argument_schemas,
            self.tool_choice,
            self.parallel_tool_calls,
        )
        prompt = _decision_prompt(
            messages, self.bound_tools, self.tool_choice, argument_schemas,
        )
        usages = []

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
                for attempt in range(1, _MAX_DECISION_ATTEMPTS + 1):
                    result = thread.run(
                        prompt,
                        approval_mode=ApprovalMode.deny_all,
                        output_schema=schema,
                        sandbox=Sandbox.read_only,
                    )
                    usages.append(_usage_metadata(result))
                    try:
                        content, tool_calls = _parse_decision(
                            getattr(result, "final_response", None), argument_schemas,
                            self.tool_choice, self.parallel_tool_calls,
                        )
                        break
                    except _MalformedDecision as exc:
                        if attempt == _MAX_DECISION_ATTEMPTS:
                            raise CodexResponseError(
                                f"Codex response invalid after {attempt} attempts: {exc}"
                            ) from None
                        prompt = (
                            f"The previous response was unusable: {exc}\n"
                            "No tool requests from that response were executed. Return a "
                            "complete replacement decision using the same output schema, "
                            "argument_encodings, and tool-choice constraint. Do not repeat "
                            "already completed operations from the supplied conversation."
                        )

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=(
                {key: sum(usage[key] for usage in usages) for key in usages[0]}
                if all(usage is not None for usage in usages) else None
            ),
            response_metadata={
                "model": self.model_id, "provider": "codex",
                "codex_decision_attempts": attempt,
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
