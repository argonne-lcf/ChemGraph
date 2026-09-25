"""Console progress for the scaling example, including executor subgraphs."""

from datetime import datetime, timezone
from itertools import islice
import json
from threading import RLock
import time

from langchain_core.callbacks import BaseCallbackHandler


def summarize(value, depth=0):
    """Bound tool payloads while retaining status, settings, and artifact paths."""
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, str):
        if len(value) <= 32_000 and value.lstrip().startswith(("{", "[")):
            try:
                return summarize(json.loads(value), depth)
            except ValueError:
                pass
        return value if len(value) <= 2000 else value[:2000] + " ... [truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= 4:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        result = {str(key): summarize(item, depth + 1)
                  for key, item in islice(value.items(), 20)}
        if len(value) > 20:
            result["omitted_fields"] = len(value) - 20
        return result
    if isinstance(value, (list, tuple)):
        preview = [summarize(item, depth + 1) for item in value[:3]]
        return preview if len(value) <= 3 else {"count": len(value), "preview": preview}
    return summarize(str(value), depth)


class ProgressLogger(BaseCallbackHandler):
    """Observe model/tool callbacks without changing graph execution."""

    run_inline = True
    raise_error = False

    def __init__(self):
        self._runs = {}
        self._lock = RLock()

    def _start(self, run_id, parent_run_id=None, metadata=None, inputs=None, name=None):
        with self._lock:
            parent = self._runs.get(parent_run_id, {})
            metadata = metadata or {}
            worker = inputs.get("executor_id") if isinstance(inputs, dict) else None
            context = {
                "worker": worker or parent.get("worker"),
                "node": metadata.get("langgraph_node") or parent.get("node") or "agent",
                "name": name or "tool",
                "started": time.monotonic(),
            }
            self._runs[run_id] = context
            return context

    def _finish(self, run_id):
        with self._lock:
            return self._runs.pop(run_id, {})

    def _print(self, context, event, value=None):
        # A broken console must not affect the simulation workflow.
        try:
            label = context.get("worker") or context.get("node", "agent")
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            detail = ""
            if value is not None:
                value = summarize(value)
                detail = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                if len(detail) > 4000:
                    detail = detail[:4000] + " ... [truncated]"
            with self._lock:
                print(f"{stamp} [{label}] {event}" + (f": {detail}" if detail else ""), flush=True)
        except Exception:
            pass

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        self._start(run_id, parent_run_id, metadata, inputs)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._finish(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._finish(run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        context = self._start(run_id, parent_run_id, metadata)
        self._print(context, "model started")

    def on_llm_end(self, response, *, run_id, **kwargs):
        context = self._finish(run_id)
        # Parent planner/analyst replies are already printed by ChemGraph.run().
        if context.get("worker") or context.get("node") == "executor_agent":
            elapsed = time.monotonic() - context.get("started", time.monotonic())
            for group in response.generations:
                for generation in group:
                    message = getattr(generation, "message", None)
                    if message is not None and message.content:
                        self._print(context, f"reply ({elapsed:.1f}s)", message.content)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._print(self._finish(run_id), "model failed", str(error))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, metadata=None, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name")
        context = self._start(run_id, parent_run_id, metadata, name=name)
        self._print(context, f"{context['name']} started", inputs if inputs is not None else input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        context = self._finish(run_id)
        elapsed = time.monotonic() - context.get("started", time.monotonic())
        status = "failed" if getattr(output, "status", None) == "error" else "finished"
        self._print(context, f"{context.get('name', 'tool')} {status} ({elapsed:.1f}s)", output)

    def on_tool_error(self, error, *, run_id, **kwargs):
        context = self._finish(run_id)
        self._print(context, f"{context.get('name', 'tool')} failed", str(error))
