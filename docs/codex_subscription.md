# Experimental Codex subscription support

ChemGraph can experimentally use the official Codex Python SDK with the
ChatGPT-backed login already used by Codex CLI or the Codex IDE extension. This
path does not use the OpenAI Platform API or `OPENAI_API_KEY`.

## Install and authenticate

### 1. Install the Codex CLI

The `openai-codex` Python package provides the SDK and its internal runtime,
but it does not install a `codex` command on your shell `PATH`. Follow the
[official Codex CLI installation guide](https://learn.chatgpt.com/docs/codex/cli),
then verify that the command is available:

```bash
codex --version
```

### 2. Install the ChemGraph Codex extra

For a package installation:

```bash
pip install "chemgraph[codex]"
```

For an editable source checkout, install the extra into the active environment:

```bash
pip install -e ".[codex]"
```

### 3. Authenticate with ChatGPT

Use the separately installed CLI to authenticate Codex with ChatGPT:

```bash
codex login
codex login status
```

The active login must be a ChatGPT login. ChemGraph deliberately rejects a
Codex session authenticated with an API key instead of silently falling back to
usage-based OpenAI Platform billing.

## Run ChemGraph

Prefix a model available to your Codex account with `codex:`:

```bash
chemgraph \
  --model "codex:<codex-model-id>" \
  --workflow single_agent \
  --query "What is the SMILES string for aspirin?"
```

The same model syntax works through the Python API:

```python
from chemgraph import ChemGraph

agent = ChemGraph(
    model_name="codex:<codex-model-id>",
    workflow_type="single_agent",
)
```

## Current limitations

- Only the `single_agent` workflow is supported.
- The integration is experimental and pins `openai-codex==0.144.4` together
  with that SDK's bundled Codex runtime.
- ChemGraph starts ephemeral, read-only Codex threads. ChemGraph's LangGraph
  workflow executes chemistry tools; Codex is used only for model decisions.
- ChemGraph does not start a login flow. Run `codex login` before initializing
  a `codex:` model.

See the official [Codex SDK documentation](https://learn.chatgpt.com/docs/codex-sdk),
[Codex CLI installation guide](https://learn.chatgpt.com/docs/codex/cli), and
[authentication guide](https://learn.chatgpt.com/docs/auth) for supported
installation and account behavior.
