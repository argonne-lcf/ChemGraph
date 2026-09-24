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

### From the Streamlit interface

The Streamlit app reuses the same stored login. Its **Codex (ChatGPT)**
provider card (first-run setup and *Configuration → Providers*) shows the
login state and offers *Sign in with ChatGPT*, which starts a device-code
login through the pinned Codex SDK on the machine hosting the UI and displays
the verification URL and one-time code in the page, plus *Log out*. The SDK
runs its bundled Codex runtime, so the UI does not need `codex` on `PATH`;
running `codex login` in a terminal remains an alternative. The model picker then lists the
models reported by Codex for the signed-in account (`codex:<model-id>`), with
the account default preselected and a free-text entry for unlisted ids; the
agent is rebuilt automatically when the login changes.

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

## Skills and tool access

The Codex adapter can request every tool exposed by the selected ChemGraph
workflow, including file reads and edits, execution, delegation, and attached
chemistry tools. It returns structured tool requests; ChemGraph executes them
through its configured backends and applies the usual approvals. Codex's own
native tools remain unused, and its temporary read-only thread does not limit
access through ChemGraph's tools.

Deep Agent skills are discovered by Python code. Their names, descriptions, and
paths are added to the model's system context, and the model requests `read_file`
to inspect full instructions. For an additional collection outside the workspace:

```bash
chemgraph run --interactive --workflow deep_agent \
  --model "codex:<codex-model-id>" --deepagent-workspace . \
  --deepagent-skill ../external/AtomisticSkills/.agents/skills/
```

The source directory must already exist. See [skills](skills.md) for directory
layout, discovery, and the distinction between file-tool and shell paths.
Loading a skill does not install its dependencies or attach its MCP tools.

When diagnosing an access refusal, inspect `skills_metadata`,
`skills_load_errors`, and the tool-call trace in the saved graph state. A skill
listed without loading errors was discovered successfully. A response claiming
it cannot inspect that skill without attempting `read_file` is a model decision;
an attempted read with an error provides evidence about the backend, path, or
request. The adapter preserves tool-call identities, arguments, and results
across model calls so the model can reason from the actual operations performed.

## Limitations

- Only `single_agent`, `main_agent`, and `deep_agent` are supported.
- `main_agent` must be interactive and can restore its supervisor checkpoint;
  individual Codex calls still start fresh read-only threads.
- The integration pins `openai-codex==0.144.4`; check the installed ChemGraph
  release before changing that dependency.
- ChemGraph starts ephemeral, read-only Codex threads. ChemGraph's graph executes
  all exposed tools; Codex supplies model decisions.
- ChemGraph does not initiate login. Authenticate before constructing a
  `codex:` model.

Because this integration is experimental, validate model availability and
account behavior against the current official documentation and your installed
Codex CLI.
