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
