# Local models

Ollama is the simplest local model route currently wired into ChemGraph. Local
inference avoids sending prompts to a hosted provider, but chemistry tools may
still call network services or external programs.

## Ollama quickstart

Install and start Ollama using its
[official instructions](https://ollama.com/download), then pull a supported
model:

```bash
ollama pull llama3.2
ollama list
chemgraph models
chemgraph run --model llama3.2 \
  -q "Build water from SMILES O and optimize it with EMT."
```

The normal endpoint is `http://localhost:11434`. Override it when Ollama is
hosted elsewhere:

```toml
[api.local]
base_url = "http://localhost:11434"
```

## Selection and reliability

ChemGraph recognizes a curated set of local identifiers. Use
`chemgraph models` rather than assuming every Ollama tag is mapped. A model must
also support tool calling well enough to emit the schemas expected by chemistry
tools.

Smaller models may choose tools, arguments, or units less reliably. Start with
a short, explicit EMT request and inspect every tool call/result. Model memory,
accelerator needs, and downloads are determined by the selected Ollama model.

## Advanced OpenAI-compatible endpoints

The repository includes `scripts/run_vllm_server.sh` as an environment-specific
vLLM helper. Custom endpoints are advanced: they must expose a compatible API,
route through a compatible ChemGraph provider/model ID, and implement reliable
tool calling. Review the script's hardware/model assumptions and test a
non-destructive query first. Ollama remains the documented first local route.

## Aurora on-node inference (`aurora:` models)

ChemGraph has a first-class provider for on-node LLM servers on ALCF Aurora
(llama.cpp SYCL `llama-server` or vLLM-XPU) that expose an OpenAI-compatible
`/v1` endpoint. Use an `aurora:<served-model-id>` model, where the id matches
the server's advertised name (`llama-server --alias` / vLLM
`--served-model-name`). Any served id works; `chemgraph models` lists a few for
discovery.

```bash
# On the Aurora node serving the model (OpenAI-compatible /v1 on :8000):
export AURORA_BASE_URL="http://127.0.0.1:8000/v1"   # co-located, or an SSH tunnel
chemgraph run --model aurora:gpt-oss-120b \
  -q "Build water from SMILES O, optimize it with EMT, and report the energy."
```

Or configure it in `config.toml`:

```toml
[api.aurora]
base_url = "http://127.0.0.1:8000/v1"
```

The endpoint address changes per job and compute nodes have no public IP, so run
ChemGraph on the same node (`127.0.0.1`) or tunnel from a login node. The chosen
model must support tool calling. Authentication is usually disabled, so the API
key defaults to `"dummy"`.

## Privacy boundary

Local inference alone does not make a workflow offline. PubChem lookup, remote
MCP servers, ALCF execution, hosted embeddings, and model/calculator downloads
can still leave the machine. Choose local tools and pre-stage all required data
for an air-gapped run.
