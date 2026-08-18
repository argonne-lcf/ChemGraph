# Configuration

A TOML file can hold repeatable non-secret endpoint, MCP, logging, evaluation,
and execution settings. For a first run, CLI flags and provider environment
variables are usually simpler.

```bash
chemgraph run --config config.toml -q "What is the SMILES string for water?"
```

## General settings and interface behavior

```toml
[general]
model = "gpt-4o-mini"
workflow = "single_agent"
output = "last_message"
structured = false
report = false
recursion_limit = 20
human_supervised = false

[logging]
level = "WARNING"
```

Streamlit consumes the `[general]` model, workflow, output, structured, report,
and supervision defaults. On the CLI, use explicit flags for those settings:

```bash
chemgraph run --model gpt-4o-mini --workflow single_agent \
  --output last_message -q "What is the SMILES string for water?"
```

The CLI currently honors selected general/config values such as
`recursion_limit`, `enable_deepagent`, and `checkpoint_db`, but its parser has
concrete defaults for several other fields. Therefore a historical `[general]`
value may not override a CLI default. The CLI flag is the reliable source for
model/workflow/output behavior.

## Provider endpoints

Environment variables are the recommended place for API keys and access tokens.
TOML provider sections configure endpoints and an optional Argo username:

```toml
[api.openai]
base_url = "https://api.openai.com/v1"
argo_user = ""

[api.anthropic]
base_url = "https://api.anthropic.com"

[api.google]
base_url = "https://generativelanguage.googleapis.com/v1beta"

[api.alcf]
base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
timeout = 30
```

`base_url` is the Sophia endpoint. Models hosted on Minerva
(`nemotron-3-ultra`, `inkling-bf16`) resolve to their own endpoint
automatically and ignore this setting; pass `--base-url` to override either.

2. Authenticate via Globus OAuth and set the access token:

```bash
pip install globus_sdk
wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py
python inference_auth_token.py authenticate
export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)
```

3. Use an ALCF model (no prefix needed):

```bash
chemgraph --config config.toml -m meta-llama/Meta-Llama-3.1-70B-Instruct \
  -q "Calculate the energy of water using MACE"
```

Access tokens are valid for ~48 hours. See the
[ALCF docs](https://docs.alcf.anl.gov/services/inference-endpoints/#available-models) for available models.

#### Groq

ChemGraph supports [Groq](https://groq.com/) for fast LLM inference. Use the `groq:` prefix to route any model through Groq.

1. Set your API key:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

2. Use any Groq model with the `groq:` prefix:

```bash
chemgraph -q "What is the SMILES for water?" -m groq:llama-3.3-70b-versatile
chemgraph -q "Optimize methane" -m groq:openai/gpt-oss-120b
```

No curated model list is maintained -- any model available on the
[Groq console](https://console.groq.com/docs/models) can be used by prefixing
it with `groq:`. The prefix is stripped before sending to the Groq API.

#### LLM Provider Prefixes

For third-party providers that share model names with other services, ChemGraph
uses a prefix convention to route models unambiguously:

| Prefix | Provider | Auth Env Var | Example |
|--------|----------|--------------|---------|
| `argo:` | Argo API (Argonne internal) | `OPENAI_API_KEY` | `argo:gpt-4o` |
| `groq:` | Groq Cloud | `GROQ_API_KEY` | `groq:llama-3.3-70b-versatile` |

Direct model names (no prefix) are used for OpenAI, Anthropic, Google, ALCF, and Ollama.

### Configuration Sections

| Section          | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `[general]`      | Basic settings like model, workflow, and output format  |
| `[llm]`          | Reserved/legacy LLM parameter documentation             |
| `[api]`          | API endpoints and timeouts for different providers      |
| `[chemistry]`    | Chemistry-specific calculation settings                 |
| `[output]`       | Output file formats and visualization settings          |
| `[logging]`      | Logging configuration and verbosity levels              |
| `[features]`     | Feature flags and experimental settings                 |
| `[security]`     | Security settings and rate limiting                     |

### Command Line Interface

ChemGraph includes a powerful command-line interface (CLI) that provides all the functionality of the web interface through the terminal. The CLI features rich formatting, interactive mode, and comprehensive configuration options.

#### Installation & Setup

The CLI is included by default when you install ChemGraph:

```bash
pip install -e .
```

#### Basic Usage

##### Quick Start

```bash
# Basic query
chemgraph -q "What is the SMILES string for water?"

# With model selection
chemgraph -q "Optimize methane geometry" -m gpt-4o

# With report generation
chemgraph -q "Calculate CO2 vibrational frequencies" -r

# Using configuration file
chemgraph --config config.toml -q "Your query here"
```

##### Command Syntax

```bash
chemgraph [OPTIONS] -q "YOUR_QUERY"
```

#### Command Line Options

**Core Arguments:**

| Option              | Short | Description                                           | Default        |
| ------------------- | ----- | ----------------------------------------------------- | -------------- |
| `--query`           | `-q`  | The computational chemistry query to execute          | Required       |
| `--model`           | `-m`  | LLM model to use                                     | `gpt-4o-mini`  |
| `--workflow`        | `-w`  | Workflow type                                        | `single_agent` |
| `--output`          | `-o`  | Output format (`state`, `last_message`)              | `state`        |
| `--structured`      | `-s`  | Use structured output format                         | `False`        |
| `--report`          | `-r`  | Generate detailed report                             | `False`        |
| `--deepagent`       |       | Enable the experimental `main_agent` workspace worker | `False`       |
| `--deepagent-workspace` |   | Root used by the experimental workspace worker       | Current directory |
| `--checkpoint-db`    |       | SQLite checkpoints for durable `main_agent` threads | `~/.chemgraph/checkpoints.db` |
| `--resume`          |       | Resume from a previous session ID (prefix supported) |                |
| `--list-sessions`   |       | List recent sessions from the memory database        |                |
| `--show-session`    |       | Show conversation for a session (prefix supported)   |                |
| `--delete-session`  |       | Delete a session from the memory database            |                |

**Model Selection:**

```bash
# OpenAI models
chemgraph -q "Your query" -m gpt-4o
chemgraph -q "Your query" -m gpt-4o-mini

# Anthropic models
chemgraph -q "Your query" -m claude-3-5-sonnet-20241022

# Google models
chemgraph -q "Your query" -m gemini-2.5-pro

# Argo models (Argonne internal, argo: prefix)
chemgraph -q "Your query" -m argo:gpt-4o
chemgraph -q "Your query" -m argo:claude-sonnet-4

# ALCF models (Globus auth required, no prefix)
chemgraph -q "Your query" -m meta-llama/Meta-Llama-3.1-70B-Instruct

# Groq models (groq: prefix, any Groq model)
chemgraph -q "Your query" -m groq:llama-3.3-70b-versatile

# Local models (Ollama)
chemgraph -q "Your query" -m llama3.2
```

**Workflow Types:**

```bash
# Single agent (default) - best for most tasks
chemgraph -q "Optimize water molecule" -w single_agent

# Long-lived supervisor - delegates chemistry tasks to the existing agent
chemgraph --interactive -w main_agent

[api.local]
base_url = "http://localhost:11434"
```

The selected model determines which section is consulted. See
[Models and authentication](models.md).

## MCP connection

Configure either streamable HTTP:

```toml
[mcp]
url = "http://localhost:9003/mcp/"
server_name = "ChemGraph General Tools"
```

or a stdio launch command:

```toml
[mcp]
command = "python -m chemgraph.mcp.mcp_tools"
server_name = "ChemGraph General Tools"
```

Do not set both unless the consuming interface explicitly supports multiple
definitions. See [MCP servers](mcp_servers.md).

## Durable main-agent state

```toml
[general]
workflow = "main_agent"
checkpoint_db = "~/.chemgraph/checkpoints.db"
enable_deepagent = false
```

`main_agent` still requires interactive CLI mode. Deep Agent is a
development-only capability with broad local access; leave it disabled unless
you understand the security boundary.

## Evaluation profiles

```toml
[eval]
default_profile = "standard"

[eval.profiles.standard]
dataset = "./evaluation/questions.json"
workflow_types = ["single_agent"]
judge_type = "structured"
structured_output = true
recursion_limit = 50
max_queries = 0
```

Profile fields may be overridden by `chemgraph eval` flags. See
[Evaluation](evaluation.md) for dataset formats and judge modes.

## Execution backend

Distributed execution reads the `[execution]` hierarchy and may also accept
backend-specific environment variables. A minimal local choice is:

```toml
[execution]
backend = "local"
```

Parsl, Ensemble Launcher, Globus Compute, and transfer settings are deployment
specific. Start from the runnable examples linked in
[HPC and Academy](hpc_and_academy.md) rather than copying credentials or endpoint
IDs into documentation.

## Which interface reads what?

| Setting area | CLI run | Streamlit | Evaluation | Execution layer |
| --- | --- | --- | --- | --- |
| `[general]` | Partial; prefer explicit run flags | Yes | No | No |
| `[api.*]` | Yes | Yes | Yes | No |
| `[mcp]` | Yes | Not the primary UI control | No | No |
| `[logging]` | Yes | Application-dependent | Yes | Yes |
| `[eval]`, `[eval.profiles.*]` | No | No | Yes | No |
| `[execution]` | Through backend tools | Through backend tools | No | Yes |

## Security

Never commit API keys, bearer tokens, endpoint secrets, or private paths. Pass
secrets through environment variables or an approved secret manager. Sanitize
configuration files before attaching them to bug reports.
