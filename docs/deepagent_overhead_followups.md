# Deep Agent overhead: findings and proposed follow-ups

Review notes prepared on 2026-09-18 for issue [#245](https://github.com/argonne-lcf/ChemGraph/issues/245).
[PR #246](https://github.com/argonne-lcf/ChemGraph/pull/246) includes the approval
UX fixes and the default graph limit of 200. This follow-up adds per-call limit
precedence, duplicate logging correction, and provider usage accounting.
Model/token budgets and prompt/runtime optimizations remain future work.

## Implementation status

- Usage is collected independently of dashboard subscribers, persisted per call
  in the existing session database, and retained across approval resumes and
  durable session restoration. Calls are deduplicated by callback ID; executed
  retries still count. Missing provider usage remains unknown or partial.
- The CLI prints per-user-turn input/output/total counters after answers and
  errors, and whole-session counters when interactive mode exits. These are
  locally formatted provider counts, not model-generated summaries.
- Codex usage notifications preserve cumulative thread totals, cached input,
  reasoning output, and available counts before SDK or decision parsing errors.
  The SDK thread remains ephemeral per ChemGraph model invocation.
- Explicit per-call recursion limits override the agent default and survive
  approval resumes without mutating the caller's configuration.
- Logging stops propagation at the ChemGraph namespace while preserving child
  file handlers.

The measurements below are historical observations from the original run; they
have not been regenerated using the new accounting.

## Measurement baseline

Measured against ChemGraph commit `b2a72877c9406290caf1f0fdd5ee32020456c01c`
with the local issue-245 approval changes, Deep Agents 0.7.5, and LangChain 1.4.0.
The example used `codex:gpt-5.6-sol` to run a geometry optimization for carbonic
acid. Measurements describe this environment, not every provider or calculator
installation. No additional live model calls were made to reconstruct prompts.

The completed retry recorded seven model calls: **131,454 input tokens and
1,615 output tokens, totaling 133,069**. This is cumulative usage across calls,
not one context window. The original failed run did not preserve enough data
to recover its exact usage. The completed retry is not a measurement of that
failed attempt.

| Decision | Input tokens | Output tokens |
| --- | ---: | ---: |
| Read ChemGraph skill | 14,175 | 86 |
| Read references and search tools | 15,005 | 299 |
| Load chemistry tools | 16,851 | 166 |
| Generate coordinates | 20,925 | 237 |
| Read coordinates | 21,137 | 177 |
| Run optimization | 21,473 | 334 |
| Report results | 21,888 | 316 |

The first three decisions prepared instructions and tools before creating the
structure. No delegated subagent was called in this example.

Offline prompt reconstruction using `o200k_base` produced these approximate
component sizes. This tokenizer is an estimate, not an authoritative billing
tokenizer for the selected model.

| Component | Estimated tokens per call |
| --- | ---: |
| Serialized system instructions, skill catalog, workspace and registry guidance | 1,312 |
| Eleven base workspace, delegation, and discovery tool schemas | 2,780 |
| Three chemistry schemas after loading | Additional 3,899 |
| Serialized conversation and tool results | 20 initially, growing to 3,792 |
| Reported input beyond the reconstructed decision prompt | Approximately 10,000 |

The base ChemGraph prompt itself is only about 198 tokens. The `run_ase` schema
is about 3,501 tokens in this environment, contributing about 14,000 tokens
over the final four calls. Schema size depends on the installed calculators.

The reconstructed decision prompts total approximately 61,477 tokens, compared
with 131,454 reported input tokens. The nearly constant per-call difference
strongly suggests additional Codex runtime/context overhead. Its exact contents
are not exposed by current logs, so it must not all be labeled removable waste.

The Codex adapter starts a fresh SDK client and ephemeral thread for every
decision and sends the full conversation plus current tool schemas. This also
adds repeated client startup and account checks; their latency was not isolated.
At the measurement baseline, the adapter dropped cached-input and reasoning-token
details exposed by the SDK.
Consequently, these totals cannot establish uncached consumption or monetary
cost. Fresh threads do not by themselves prove that provider caching is absent.

## 1. Usage accounting implemented in this follow-up

Relevant code: `src/chemgraph/agent/events.py`, `models/codex.py`, and
`agent/llm_agent.py` (the latter two paths are also under `src/chemgraph/`).

- Extract `AIMessage.usage_metadata` from model generations, with a provider
  `llm_output` fallback. The original event callback only read `llm_output`.
- Persist each completed call independently of final workflow snapshots. Record
  a stable call ID, user-turn ID, model, worker identity, input/output/total,
  cached input, reasoning output, duration, and available raw usage.
- Cached input and reasoning output are subsets of their respective totals;
  do not add them again. Missing usage must remain unknown, not zero.
- Count executed retries and SDK-internal attempts using cumulative Codex thread
  snapshots rather than `usage.last`. Replace snapshots instead of adding them.
- Deduplicate repeated callbacks, checkpoints, approval resumes, and worker
  events across the whole user turn.
- On failure, print accumulated usage and attempt to save the latest checkpoint
  without masking the original exception. Preserve cancellation summaries too.
- Install accounting independently of an optional dashboard/event subscriber.

Coordinate with [PR #218](https://github.com/argonne-lcf/ChemGraph/pull/218),
which proposes Anthropic caching and token events. Its Anthropic cache support
does not resolve Codex accounting.

Acceptance: scripted responses with known usage followed by a forced graph
failure still produce the correct durable summary; replayed events do not
double-count, while executed retries count; providers without usage do not crash
the workflow.

## 2. Separate graph safety limits from model and token budgets

The previous CLI/UI default was 20 graph steps; Python APIs used 50. A local
scripted reproduction reached 20 steps after only four model calls because
skills, registry, approval middleware, and tool execution also consume steps.
This does not prove how many calls the original failed run completed.

The approved immediate change makes defaults consistently 200. Explicit user
configuration and stored session limits remain authoritative. More graph steps
allow more work; this change does not reduce token consumption.

Follow-up proposals:

- Add a separate configurable model-call budget; 20 calls is an initial value
  to evaluate, not a new setting implemented by the current PR.
- Evaluate LangChain's `ModelCallLimitMiddleware`, accounting for its run/thread
  semantics. A user turn may span multiple approval resumes and worker graphs.
- Add an optional token budget and check it before subsequent calls. A budget
  based only on completed-call usage can overshoot by one call; stricter control
  requires input estimates and output reservations.
- Centralize configuration precedence and include standalone Deep Agent and
  `main_agent` worker execution. This follow-up corrects the unconditional assignment of
  `config["recursion_limit"]` inside `ChemGraph.run`.
- Return a stopped status, partial results, usage, and a clear limit reason.
  Retain a finite graph limit as protection against middleware loops.

Acceptance: normal multi-tool workflows complete beyond the old 20-step limit;
a scripted endless tool loop stops at the model budget; approvals and delegated
work cannot reset or bypass the turn budget; explicit lower limits still work.

## 3. Reduce prompt size and preparation decisions

Relevant code: `graphs/deep_agent.py`, `registry/middleware.py`,
`skills/chemgraph/`, `tools/ase_tools.py`, and `schemas/ase_input.py`, all under
`src/chemgraph/`.

- Offer an explicit chemistry configuration with the small relevant tool set
  initially available. The current `--tool` option restricts the catalog but
  does not preload those tools. Keep discovery for unfamiliar tasks.
- Shorten the local skill path and avoid unnecessary reference reads for routine
  local operations. Retain structure validation and filesystem guidance.
- Consider a narrower optimization tool schema that forwards validated inputs
  into the existing ASE implementation, while keeping full `run_ase` available
  for advanced tasks. Preserve scientific parameters and calculator choices.
- Consider returning a compact validated structure summary from preparation
  tools, reducing separate model decisions solely to inspect newly made files.
- Keep large artifacts on disk and return concise summaries with paths. Any
  result projection must preserve errors, convergence, units, and provenance.
- Evaluate an optional reduced workspace/delegation tool set for chemistry
  tasks. Preserve the full workspace configuration for coding tasks.

Do not start with aggressive summarization: it adds a model call and can lose
scientific settings. Stable prompt/schema reduction and fewer decisions are
easier to validate first. Avoid blind tool preloading, which increases every
prompt even when those tools are unused.

Acceptance: offline prompt-size measurements improve; scripted workflows retain
approval behavior, calculator choices, validation, result semantics, and paths.
Later authorized live comparisons should report task completion, calls, tokens,
cache usage, and latency rather than token savings alone.

## 4. Investigate Codex integration overhead separately

- Instrument the boundary between the serialized ChemGraph decision prompt and
  SDK-reported usage to explain the approximately 10,000-token per-call gap.
- Reuse the SDK client where lifecycle and concurrency permit, then measure
  startup latency. Client reuse alone does not reduce context tokens.
- Evaluate thread reuse with correct history synchronization, approvals,
  branching, and recovery. Do not assume incremental transport reduces the
  model's logical input-token count.
- Benchmark a direct chat-model integration for the Deep Agent loop.
- As a separate architectural option, evaluate a Codex-native workflow that
  lets Codex manage the tool loop while exposing chemistry capabilities through
  supported tools/MCP. Preserve ChemGraph permissions, sessions, and reporting.

Coordinate with [PR #240](https://github.com/argonne-lcf/ChemGraph/pull/240),
which proposes structured Codex arguments and malformed-decision recovery.
On integration, use cumulative usage snapshots from the shared SDK thread for
all correction attempts; do not add `last` or the cumulative totals twice.
Schema changes should be measured alongside correctness improvements.

## 5. Duplicate error logging corrected in this follow-up

`src/chemgraph/utils/logging_config.py::configure_logging` previously installed a handler
on the `chemgraph` namespace while leaving propagation enabled. Setting that
logger's `propagate=False` prevents a root handler from printing the same record
again. Child/file logging is preserved and repeated configuration is idempotent.
The CLI's user-facing error is a separate presentation decision.

Acceptance: with a root handler already installed, one error produces one
ChemGraph log line and intended file logging continues. This has no token effect.

## Suggested delivery order

1. Implemented here, on top of #246: usage persistence/CLI counters, recursion
   precedence, and logging correction.
2. Model-call/token budgets and their approval/resume semantics.
3. Focused skill and schema reductions, each measured against this baseline.
4. Codex lifecycle and execution-architecture experiments.

References:

- [LangGraph recursion limits](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)
- [LangChain model-call limits](https://docs.langchain.com/oss/python/langchain/middleware/built-in#model-call-limit)
- [Deep Agents customization](https://docs.langchain.com/oss/python/deepagents/customization)
