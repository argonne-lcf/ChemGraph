# Models and authentication

ChemGraph selects a provider from the model identifier. Run `chemgraph models`
to see the identifiers registered by your installed version.

## Provider routes

| Provider | Model form | Credential or setup |
| --- | --- | --- |
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-...` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini-...` | `GEMINI_API_KEY` |
| Groq | `groq:<model-id>` | `GROQ_API_KEY` |
| OpenRouter | `openrouter:<model-id>` | `OPENROUTER_API_KEY` |
| Argo | `argo:<model-id>` | `ARGO_USER` or `argo_user` in config |
| ALCF endpoints | Listed by `chemgraph models` | `ALCF_ACCESS_TOKEN` |
| Ollama | Curated local ID such as `llama3.2` | Running Ollama server; no key |
| vLLM/custom | Custom ID plus a configured base URL | `VLLM_API_KEY` or placeholder |
| Codex | `codex:<model-id>` | Experimental subscription setup |

Only set credentials for providers you use. Do not commit secrets to
`config.toml`, shell scripts, notebooks, or Git history.

## Select and test a model

The default is `gpt-4o-mini`. Override it per command:

```bash
chemgraph run --model gpt-4o-mini -q "What is the formula of caffeine?"
```

Inspect the registry and validate configured credentials without starting a
workflow:

```bash
chemgraph models
chemgraph run --model gpt-4o-mini --check-keys
```

## Reasoning effort

ChemGraph exposes provider reasoning controls for a manually verified subset of
Argo models:

| Models | Accepted values | Default |
| --- | --- | --- |
| `argo:gpt-5.6-luna`, `argo:gpt-5.6-sol`, `argo:gpt-5.6-terra` | `none`, `low`, `medium`, `high`, `xhigh`, `max` | `medium` |
| `argo:claude-opus-4.8`, `argo:claude-opus-5` | `low`, `medium`, `high`, `xhigh`, `max` | `medium` |

Select the effort for a run with `--reasoning-effort`:

```bash
chemgraph run --model argo:gpt-5.6-sol --reasoning-effort high \
  -q "Compare two reaction pathways."
```

For Claude, ChemGraph forwards the value as Anthropic's overall response
effort. It does not explicitly enable adaptive thinking. Other model IDs reject
an explicit effort until their endpoint behavior has been verified.

## Public API providers

Set the provider's environment variable before launching ChemGraph:

```bash
export OPENAI_API_KEY="..."
# or ANTHROPIC_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY
```

Use the prefix shown by `chemgraph models` where one is required. Provider model
catalogs change more frequently than ChemGraph releases; the CLI registry is
the source of truth for identifiers supported by the installed version.

## Argonne routes

Argo routes use the `argo:` prefix and an Argonne username:

```bash
export ARGO_USER="<anl-username>"
chemgraph run --model argo:gpt-4o -q "Summarize the water molecule."
```

For every `argo:` model, ChemGraph resolves the identity from an explicit or
configured `argo_user`, then `ARGO_USER`, and finally the internal `chemgraph`
placeholder required by the client libraries. Argo routes do not read an
explicit `api_key` or `OPENAI_API_KEY`, so OpenAI credentials are never sent to
an Argo endpoint.

ALCF-hosted inference routes use `ALCF_ACCESS_TOKEN`. Available endpoints and
access policies are facility-managed, so use `chemgraph models` and the current
ALCF service instructions rather than copying an old model name.

## Local Ollama models

Start Ollama, pull a model supported by ChemGraph, then select it:

```bash
ollama pull llama3.2
chemgraph run --model llama3.2 -q "What is the formula of water?"
```

Local models vary substantially in tool-calling reliability. See
[Local models](running_local_models.md) for limitations and advanced endpoints.

## Configuration file

Non-secret endpoint settings and `[api.argo].argo_user` can be placed in
`config.toml`. Use `[api.vllm].base_url` for an otherwise unknown custom model;
leaving that canonical value empty prevents the direct OpenAI URL from becoming
an accidental custom-model fallback.
Choose the CLI model with `--model`; Streamlit reads its model default from the
configuration file. Environment variables remain the recommended place for tokens.
See [Configuration](configuration_with_toml.md) for supported sections and
interface-specific behavior.

## Experimental Codex route

The Codex subscription integration has separate dependencies and authentication
and is not an OpenAI API-key replacement for every workflow. See
[Codex subscription](codex_subscription.md) before using a `codex:` model.
