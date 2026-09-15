# Experimental Codex subscription support

ChemGraph can experimentally use the Codex Python SDK with a ChatGPT-backed
login already established by Codex CLI or an IDE integration. This route does
not use `OPENAI_API_KEY` and is distinct from OpenAI Platform API billing.

## Install

Install Codex CLI using the
[official Codex CLI guide](https://developers.openai.com/codex/cli/), then check
that it is on `PATH`:

```bash
codex --version
```

Install ChemGraph's pinned SDK integration from a source checkout:

```bash
git clone https://github.com/argonne-lcf/ChemGraph.git
cd ChemGraph
python -m pip install -e ".[codex]"
```

## Authenticate

```bash
codex login
codex login status
```

Use a ChatGPT login. ChemGraph rejects an API-key-authenticated Codex session
instead of silently moving this route to usage-based Platform billing. Review
the [official authentication guide](https://developers.openai.com/codex/auth/)
for current account behavior.

## Run

Prefix a model available to the signed-in Codex account with `codex:`:

```bash
chemgraph run \
  --model "codex:<codex-model-id>" \
  --workflow single_agent \
  --query "What is the SMILES string for aspirin?"
```

The long-lived supervisor is interactive:

```bash
chemgraph run --interactive \
  --model "codex:<codex-model-id>" \
  --workflow main_agent
```

Python uses the normal ChemGraph import:

```python
from chemgraph.agent.llm_agent import ChemGraph

agent = ChemGraph(
    model_name="codex:<codex-model-id>",
    workflow_type="single_agent",
)
```

The same model adapter can drive the workspace harness:

```bash
chemgraph run --interactive \
  --model "codex:<codex-model-id>" \
  --workflow deep_agent \
  --deepagent-workspace /path/to/disposable-checkout
```

This measures the model inside ChemGraph's Deep Agent prompt, tools, approval
policy, and checkpoint loop. It is not a native Codex runtime comparison. For
comparisons with Codex or Claude Code, use identical starting checkouts and
tasks, record the runtime and safety mode, and score resulting patches and
tests independently.

## Tool arguments and response recovery

Common tools such as `execute`, `write_file`, `read_file`, and `task` use JSON
objects for arguments in Codex's output schema. Commands and file contents need
only one layer of JSON encoding. Optional arguments stay omitted when the tool's
default should apply; an explicit `null` remains distinct from omission.

The adapter supports flat, named scalar arguments, enums, nullable scalars, and
supported scalar constraints. Named objects with unspecified
`additionalProperties` are closed for generation. Explicitly open, nested,
referenced, or otherwise unsupported schemas retain JSON-encoded argument
strings. The same fallback applies when optional arguments would require more
than 32 schema alternatives or the combined output schema would exceed provider
limits. The prompt identifies the selected encoding for each tool.

Codex's runtime enforces the output schema. ChemGraph checks response structure,
argument encoding, tool names, and tool-choice constraints before returning the
complete batch to the graph. LangChain/Pydantic retain parameter validation at
tool execution; those errors follow the graph's existing tool-error handling.
ChemGraph continues to execute tools through its configured approval flow.

An empty or malformed response receives up to two correction attempts in the
same ephemeral Codex thread (three attempts total). Corrections replace only the
pending decision; they do not replay completed tool operations. Authentication,
SDK/transport errors, and tool-choice violations fail immediately.

If correction fails, the error includes the stage, known tool name, attempt
count, and parser location when available. For example:

```text
Codex response invalid after 3 attempts: Codex arguments JSON for 'my_tool': Expecting value at line 1, column 12.
```

Argument values, commands, and file contents are excluded from these adapter
error messages. A decoding failure means that no tool calls from that rejected
response were returned for execution. It does not indicate a PBS or ASE failure;
inspect earlier tool results separately when checking already submitted jobs.

Successful `AIMessage.response_metadata` includes `codex_decision_attempts`.
Token usage sums all attempts when each attempt reports usage; otherwise the
aggregate is omitted. No new CLI options or dependencies are required. After
updating the checkout, reinstall with `python -m pip install -e ".[codex]"` and
use the same `chemgraph run` command.

## Limitations

- Only `single_agent`, `main_agent`, and `deep_agent` are supported.
- `main_agent` must be interactive and can restore its supervisor checkpoint;
  individual Codex calls still start fresh read-only threads.
- The integration pins `openai-codex==0.144.4`; check the installed ChemGraph
  release before changing that dependency.
- ChemGraph starts ephemeral, read-only Codex threads. ChemGraph's graph executes
  chemistry tools; Codex supplies model decisions.
- ChemGraph does not initiate login. Authenticate before constructing a
  `codex:` model.

Because this integration is experimental, validate model availability and
account behavior against the current official documentation and your installed
Codex CLI.
