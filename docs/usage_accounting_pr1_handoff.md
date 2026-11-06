# Evaluation usage accounting — PR 1 handoff

## Scope and working state

Implemented PR 1 only on `fix/evaluation-usage-collection`, based on
`ab033f66cafa4dd37050993191244cff25e17053` (`main`).
Pre-existing untracked files were left alone.
PRs 2–4 (evaluation reports, journal/recovery, controls/probes) are not implemented.

Production changes:

- `agent/usage.py`: cache-creation and service-tier details; provenance; cumulative
  adapter updates; optional evidence retention and snapshot sink.
- `models/protocols/openai_compatible.py`: thin `ChatOpenAI` subclass captures
  Chat Completions usage before response/chunk normalization.
- `models/protocols/anthropic_native.py`: thin `ChatAnthropic` subclass captures
  regular usage and merges `message_start` with later cumulative deltas.
- New `models/protocols/usage_capture.py`: invocation-scoped side channel, with
  cleanup for errors, cancellation, and interleaved generators.

Tests: expanded `test_usage.py`, new `test_endpoint_usage.py`, and the usage-event
expectation in `test_llm_agent.py`. No dependency or database-schema changes.

## Contract for the next PR

```python
collector = UsageCollector(
    session_id, thread_id,
    capture_evidence=True,
    usage_sink=save_snapshot,  # optional synchronous callable(dict) -> None
)
config = add_callbacks(config, [collector])
```

The supported adapters capture correct counts whenever the collector is attached,
including when `capture_evidence=False`. The option enables retention of the new
provider `raw_usage` and `usage_evidence` fields. Existing callback/Codex fallback
payload retention is preserved for compatibility. Evidence contains usage fields
only; it is never added to model messages, request parameters, or prompts.

`collector.records` returns defensive copies. Each record retains its callback
`call_id`; cumulative updates replace/merge that call, never create another bill.

- `counts`: existing fields plus `cache_creation_input_tokens`. Missing values
  stay `None`; explicit zero stays zero. Cache/reasoning details are subsets.
- `source`: `provider_raw`, `adapter_derived`, `adapter_snapshot` (existing Codex
  channel), or `mixed` for a mixed response batch; initially `None`.
- `raw_capture`: `captured` when a supported pre-normalization hook is observed,
  otherwise `unverified`. Capturing an empty usage object does not imply complete
  counts. A fallback can have counts while raw capture remains unverified.
- `raw_usage`: merged provider snapshot when evidence retention is enabled.
- `usage_evidence`: ordered `{event, usage}` snapshots, preserving unknown fields
  and explicit nulls in the received evidence. Nulls in later cumulative updates
  do not erase earlier known counts; zero does replace a previous count.
- `complete`: terminal provider evidence observed (or existing callback fallback
  completion). Summary coverage also requires the three primary counts. A parse
  failure after receiving final provider usage can retain complete consumption.
- `sink_failed`: sticky when a sink raises. Counts survive; summary becomes
  partial. Sink exceptions are logged once per collector and never replace the
  model exception.

The sink receives defensive snapshots on call start, each usage update, and
completion/error. It runs synchronously under the collector lock; keep it short.
For a future journal, key snapshots by evaluation/attempt identity plus `call_id`.
The sink must not mutate collector state or call back into model execution.

## Why the side channel exists

Pinned `langchain-anthropic==1.5.4` drops start-event input usage and normalizes an
output-only final delta as zero input. The HTTP/SSE regression demonstrates that
upstream message metadata still reports 7 total while the collector correctly
reports 10 input + 7 output = 17. Keeping upstream messages unchanged prevents
accounting evidence from increasing subsequent prompt tokens.

Hooks were verified against `langchain-core==1.6.1`,
`langchain-openai==1.4.1`, and `langchain-anthropic==1.5.4`.
`_generate_with_cache` scopes each invoke/generate call (including batches).
Public `stream`/`astream` do not pass a run manager to adapter hooks, so their
start callback binds the collector to a temporary context. That context is reset
before each public yield and in all cleanup paths. No call state lives on the
shared model instance or in a process-wide run registry.

## Verification

Verification uses Python 3.12.11 in `.cg_fix_ase`, whose exact dependency pins
match `pyproject.toml`. The existing `.venv` has older LangGraph packages and
reproduces a baseline deadlock; do not use it for final validation.

- Focused usage/endpoint/Codex/protocol/agent tests plus the single-worker graph
  regression: **173 passed**.
- `ruff check .`: **passed** in a clean tracked-files snapshot plus PR 1 changes.
  Running it in the original workspace also scans pre-existing untracked files
  (notably `.cg_uma_env` and `inference_auth_token.py`) with unrelated lint errors.
- `uv pip check --python .cg_fix_ase/bin/python`: **passed**, 313 packages.
- `git diff --check`: **passed**.
- Full offline suite: **2,327 passed, 19 skipped, 2 deselected** (83.61 s).

The clean verification tree is `/tmp/chemgraph-pr1-verify`; the full-suite log is
`/tmp/chemgraph-pr1-pinned-pytest.log`. A file-based test launcher redirects the
session database to `/tmp/chemgraph-pr1-verify-sessions` and calls
`pytest tests/ -k "not tblite" -q --disable-warnings`. Local test servers require
localhost sockets outside the sandbox. No live-test flags were used.

To rerun focused tests from the working tree:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .cg_fix_ase/bin/python -m pytest \
  tests/test_usage.py tests/test_endpoint_usage.py tests/test_codex_model.py \
  tests/test_protocol_construction_guard.py tests/test_llm_agent.py \
  tests/test_registry_middleware.py::test_single_worker_sync_run_does_not_deadlock -q
```

Coverage includes regular and streaming sync/async calls, missing fields, zeros,
cache details, service-tier normalization, repeated cumulative updates, batches,
concurrent/interleaved streams, parse errors, disconnection, cancellation,
snapshot-sink failure, unknown fields, and unchanged responses/HTTP prompts.
Codex regression tests remain part of focused verification.

## Boundaries and next work

- Responses API and native v2 protocol streaming remain explicitly **unverified**
  for raw evidence. Tests cover the fallback label (the latter is exposed through
  `stream_events(version="v3")` by pinned LangChain).
- Cache hits that bypass provider hooks also remain unverified. Do not treat
  adapter metadata alone as proof of newly billed consumption.
- `partial` currently describes count/completion/sink coverage; it does not
  automatically reject `raw_capture="unverified"`. PR 4 must define strict policy
  using both coverage and provenance, with appropriate Codex treatment.
- No live Argo, ALCF, OpenAI, Anthropic, or Codex endpoint probe was run. Existing
  endpoint streaming options and client defaults are unchanged.
- PR 2 should expose defensive `last_usage_records` on the agent, wire the
  evidence/sink options, label formatting/judge calls, and export stage totals
  for all evaluation workflows. Reset usage before new-query setup can fail.
- PR 3 should implement the separate evaluation journal and attempt identity;
  the sink added here is only an extension point, not crash recovery.
- PR 4 should add endpoint options, transport fingerprints, diagnostics, and
  strict reporting policy. Re-test private adapter hooks on dependency upgrades.
