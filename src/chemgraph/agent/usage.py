"""Provider-reported usage accounting, independent of conversation snapshots."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import logging
import threading
import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)
TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "total_tokens",
    "cached_input_tokens", "reasoning_output_tokens",
)
USAGE_EVENT = "chemgraph_usage"


def _mapping(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if callable(getattr(value, "model_dump", None)):
        value = value.model_dump(mode="json")
        return value if isinstance(value, dict) else {}
    return {}


def _count(values: dict, *names: str) -> int | None:
    for name in names:
        value = values.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def normalize_usage(value: Any, *, canonical: bool = False) -> dict:
    """Normalize one provider record without inventing missing token counts."""
    usage = _mapping(value)
    inputs = _count(usage, "input_tokens", "prompt_tokens", "inputTokens")
    outputs = _count(usage, "output_tokens", "completion_tokens", "outputTokens")
    total = _count(usage, "total_tokens", "totalTokens")
    input_details = _mapping(usage.get("input_token_details"))
    prompt_details = _mapping(usage.get("input_tokens_details") or usage.get("prompt_tokens_details"))
    output_details = _mapping(usage.get("output_token_details"))
    completion_details = _mapping(usage.get("output_tokens_details") or usage.get("completion_tokens_details"))
    cached = _count(usage, "cached_input_tokens", "cache_read_input_tokens", "cachedInputTokens")
    if cached is None:
        cached = _count(input_details, "cache_read")
    if cached is None:
        cached = _count(prompt_details, "cached_tokens")
    reasoning = _count(usage, "reasoning_output_tokens", "reasoningOutputTokens")
    if reasoning is None:
        reasoning = _count(output_details, "reasoning")
    if reasoning is None:
        reasoning = _count(completion_details, "reasoning_tokens")
    # Native Anthropic input_tokens excludes cache reads/writes; LangChain's
    # usage_metadata already includes them. Do not add them to canonical input.
    if not canonical and inputs is not None and any(
        key in usage for key in ("cache_read_input_tokens", "cache_creation_input_tokens")
    ):
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if key in usage:
                count = _count(usage, key)
                if count is None:
                    inputs = None
                    break
                inputs += count
    if total is None and inputs is not None and outputs is not None:
        total = inputs + outputs
    return dict(zip(TOKEN_FIELDS, (inputs, outputs, total, cached, reasoning)))


def summarize_usage(records: list[dict]) -> dict:
    """Return known subtotals with explicit coverage, never zero for unknown."""
    result = {}
    for field in TOKEN_FIELDS:
        known = [r.get("counts", {}).get(field) for r in records]
        values = [n for n in known if n is not None]
        result[field] = sum(values) if values or not records else None
    result["call_count"] = len(records)
    result["unreported_counts"] = {
        field: sum(r.get("counts", {}).get(field) is None for r in records)
        for field in TOKEN_FIELDS
    }
    result["incomplete_calls"] = sum(
        not r.get("complete", False)
        or any(r.get("counts", {}).get(key) is None for key in TOKEN_FIELDS[:3])
        for r in records
    )
    result["partial"] = bool(result["incomplete_calls"])
    return result


def response_usage(response: Any) -> dict | None:
    """Extract per-prompt usage; alternative generations share one prompt bill."""
    try:
        records = []
        raw = []
        for group in getattr(response, "generations", None) or []:
            found = None
            for generation in group or []:
                message = getattr(generation, "message", None)
                value = _mapping(getattr(message, "usage_metadata", None))
                metadata = _mapping(getattr(message, "response_metadata", None))
                if value:
                    found = normalize_usage(value, canonical=True)
                else:
                    value = _mapping(metadata.get("usage") or metadata.get("token_usage"))
                    if value:
                        found = normalize_usage(value)
                if found is not None:
                    raw.append(value)
                    break
            if group:
                records.append({"counts": found or {}, "complete": found is not None})
        # A response-level aggregate is preferable to an incomplete batch.
        output = _mapping(getattr(response, "llm_output", None))
        fallback = _mapping(output.get("token_usage") or output.get("usage") or output)
        if fallback and (not records or any(not r["complete"] for r in records)):
            counts = normalize_usage(fallback)
            if any(counts[key] is not None for key in TOKEN_FIELDS[:3]):
                return {**counts, "raw_usage": fallback, "source": "provider"}
        if not raw:
            return None
        totals = summarize_usage(records)
        return {
            **{key: totals[key] for key in TOKEN_FIELDS},
            "partial": totals["partial"], "raw_usage": raw, "source": "provider",
        }
    except Exception:
        logger.debug("Could not extract provider usage.", exc_info=True)
        return None


def add_callbacks(config: dict, callbacks: list) -> dict:
    """Copy callback lists/managers, retaining caller-owned handlers."""
    result = dict(config)
    existing = config.get("callbacks")
    if existing is None or isinstance(existing, list):
        result["callbacks"] = [*(existing or []), *callbacks]
    else:
        manager = existing.copy()
        for callback in callbacks:
            manager.add_handler(callback, inherit=True)
        result["callbacks"] = manager
    return result


def apply_history_coverage(summary: dict, history_unaccounted: bool) -> dict:
    """Label known subtotals without inventing calls for unaccounted history."""
    result = {**summary, "history_unaccounted": history_unaccounted}
    if history_unaccounted:
        result["partial"] = True
        if not result["call_count"]:
            result.update(dict.fromkeys(TOKEN_FIELDS))
    return result


def session_usage(collectors, store, session_id, *, history_unaccounted=False) -> dict:
    """Combine restored records with in-memory calls, including failed writes."""
    records = {}
    if store is not None:
        try:
            history_unaccounted = store.usage_history_unaccounted(session_id) or history_unaccounted
        except Exception:
            history_unaccounted = True
            logger.debug("Could not read historical usage coverage.", exc_info=True)
        try:
            records = {r["call_id"]: r for r in store.usage_records(session_id)}
        except Exception:
            logger.debug("Could not read stored session usage.", exc_info=True)
    for collector in collectors:
        records.update({r["call_id"]: r for r in collector.records})
    return apply_history_coverage(summarize_usage(list(records.values())), history_unaccounted)


def combine_usage(summaries: list[dict]) -> dict:
    """Combine independent sessions for an interactive CLI lifetime."""
    result = summarize_usage([
        {"counts": s, "complete": not s["partial"]} for s in summaries
    ])
    for key in ("call_count", "incomplete_calls"):
        result[key] = sum(s[key] for s in summaries)
    result["partial"] = any(s["partial"] for s in summaries)
    result["unreported_counts"] = {
        key: sum(s["unreported_counts"][key] for s in summaries) for key in TOKEN_FIELDS
    }
    return apply_history_coverage(result, any(s.get("history_unaccounted", False) for s in summaries))


class UsageCollector(BaseCallbackHandler):
    """One user turn, including nested workers and approval/retry continuations."""

    run_inline = True

    def __init__(self, session_id: str, thread_id: str, *, store=None,
                 model: str | None = None, turn_id: str | None = None,
                 records=()):
        self.session_id = session_id
        self.thread_id = thread_id
        self.turn_id = turn_id or str(uuid.uuid4())
        self.model = model
        self.store = store
        self._records = {r["call_id"]: deepcopy(r) for r in records}
        self._started: dict[str, float] = {}
        self._lock = threading.RLock()
        self._storage_failed = False
        self._persist("create_usage_turn", session_id, self.turn_id, thread_id)

    def _persist(self, method, *args):
        if self.store is None:
            return
        try:
            getattr(self.store, method)(*args)
        except Exception:
            if not self._storage_failed:
                logger.warning("Could not persist token usage; retaining in-memory counts.", exc_info=True)
                self._storage_failed = True

    @property
    def summary(self) -> dict:
        with self._lock:
            return {
                **summarize_usage(list(self._records.values())),
                "turn_id": self.turn_id, "thread_id": self.thread_id,
            }

    @property
    def records(self) -> list[dict]:
        with self._lock:
            return deepcopy(list(self._records.values()))

    def _record(self, run_id) -> dict:
        key = str(run_id)
        return self._records.setdefault(key, {
            "call_id": key, "model": self.model, "provider": None,
            "worker": None, "status": "running", "counts": {},
            "complete": False, "duration_s": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })

    def _save(self, record):
        self._persist("save_usage_call", self.turn_id, record)

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        with self._lock:
            record = self._record(run_id)
            self._started.setdefault(str(run_id), time.monotonic())
            metadata = metadata or {}
            serialized = serialized or {}
            record["model"] = metadata.get("ls_model_name") or self.model or serialized.get("name")
            record["provider"] = metadata.get("ls_provider")
            record["worker"] = metadata.get("chemgraph_subagent")
            record["parent_run_id"] = str(kwargs.get("parent_run_id") or "")
            self._save(record)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.on_chat_model_start(serialized, prompts, **kwargs)

    def on_custom_event(self, name, data, **kwargs):
        if name != USAGE_EVENT or not isinstance(data, dict) or not data.get("call_id"):
            return
        with self._lock:
            record = self._record(data["call_id"])
            # Adapter snapshots are cumulative for this model invocation. The
            # ordinary completion callback updates this row, never adds a bill.
            record["counts"] = {key: data.get("counts", {}).get(key) for key in TOKEN_FIELDS}
            record["raw_usage"] = data.get("raw_usage")
            record["complete"] = bool(data.get("complete"))
            record["provider"] = data.get("provider", record["provider"])
            record["model"] = data.get("model", record["model"])
            record["adapter_usage"] = True
            self._save(record)

    def on_llm_end(self, response, *, run_id, **kwargs):
        counts = response_usage(response)
        with self._lock:
            record = self._record(run_id)
            if counts is not None and not record.get("adapter_usage"):
                record["counts"] = {key: counts.get(key) for key in TOKEN_FIELDS}
                record["raw_usage"] = counts.get("raw_usage")
                record["complete"] = not counts.get("partial", False)
            self._end(record, "completed")

    def on_llm_error(self, error, *, run_id, **kwargs):
        counts = response_usage(kwargs.get("response"))
        with self._lock:
            record = self._record(run_id)
            if counts is not None and not record.get("adapter_usage"):
                record["counts"] = {key: counts.get(key) for key in TOKEN_FIELDS}
                record["raw_usage"] = counts.get("raw_usage")
                # A failed stream may only contain usage for its received chunks.
                record["complete"] = False
            self._end(record, "failed")

    def _end(self, record, status):
        record["status"] = status
        started = self._started.pop(record["call_id"], None)
        if started is not None:
            record["duration_s"] = round(time.monotonic() - started, 6)
        self._save(record)

    def finish(self, status: str):
        with self._lock:
            for record in self._records.values():
                if record["status"] == "running":
                    self._end(record, "cancelled" if status == "cancelled" else "incomplete")
            self._persist("update_usage_turn", self.turn_id, status)
