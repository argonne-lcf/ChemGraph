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

## Limitations

- Only `single_agent` and `main_agent` are supported.
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
