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
recursion_limit = 200
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

[api.argo]
base_url = "https://apps.inside.anl.gov/argoapi/v1"
argo_user = ""

[api.vllm]
# Set this for custom OpenAI-compatible model IDs. An explicit empty value
# disables the one-release [api.openai] custom-endpoint fallback.
base_url = ""

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

Base URLs resolve in this order: an explicit CLI/Python argument, the selected
endpoint's canonical section, a supported legacy section, its environment
variable, and finally its built-in default. For one release, `argo:` and custom
model routes can read a legacy `[api.openai].base_url` when their canonical
section is absent; ChemGraph logs migration guidance whenever it does so.

`[api.argo].argo_user` is the canonical Argo identity setting. The historical
`[api.openai].argo_user` spelling remains supported for one release with a
warning. Keep API keys and access tokens in endpoint-specific environment
variables rather than TOML.

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
# workspace = "."
# skills = ["../shared-skills"]
discover_skills = true
# subagents = ["single_agent", "deep_agent"]
# tools = ["calculator", "run_ase"]
enable_deepagent = false
deepagent_discover_skills = true
# deepagent_skills = ["../external/AtomisticSkills/.agents/skills/"]
```

Main-agent `workspace`, `skills`, `discover_skills`, `subagents`, and `tools`
correspond to `--workspace`, repeatable `--skill`, `--[no-]discover-skills`,
repeatable `--subagent`, and repeatable `--tool`. CLI lists replace TOML lists.
`skills` contains host directories resolved against the invocation directory;
Python `skills` instead contains backend-relative sources. No workspace means
checkpoint files and no shell, while bundled skills and host registry tools
remain available. Shell access is not confined to the workspace.

Omitting `subagents` exposes the non-test built-in worker catalog with no active
workers. A list restricts discovery; `subagents = []` disables it. The agent loads
workers on demand for the current turn, retaining the selection during approval
pauses and restart and clearing it at completion. `tools = []` disables discovery. Inactive main-agent
settings are retained for interactive workflow switching and ignored for other
headless workflows. Explicit incompatible CLI flags are rejected.

Start a new session after the graph upgrade. Old transcripts remain readable.
Both startup `--resume` and `/resume` restore saved supported configuration and
pending approvals, overriding current main-agent settings.
The CLI displays the saved settings before asking for host-workspace access;
current CLI flags and TOML graph settings do not modify a resumed session.
Caller-owned Python configurations require Python reconstruction and, for
opaque components, their original non-secret `configuration_id`.

`main_agent` still requires interactive CLI mode. `enable_deepagent` controls
only its legacy `deep_agent` worker. To call the graph directly, select
`workflow = "deep_agent"`; `deepagent_workspace` applies to either entry point.
Deep Agent is a development-only capability with broad local access, so leave
it disabled unless you understand the security boundary.

The headless-only `--deepagent-dangerously-skip-approvals` switch is
intentionally not configurable through TOML. It must be typed explicitly for
each run together with `--deepagent-workspace`.

`deepagent_discover_skills` defaults to true and controls personal/project
skill discovery for local workspaces. Bundled skills are always available.
The matching CLI boolean flag overrides TOML. See [skills](skills.md).

`deepagent_skills` is an ordered list of additional host skill directories.
Relative paths resolve against the CLI invocation directory, not the workspace
or TOML file directory. Absolute paths, `~`, and `..` are supported.
It applies to a direct `deep_agent` or to an enabled `main_agent` worker. Later
sources override earlier sources with the same skill name. A repeated
`--deepagent-skill` CLI option replaces the TOML list for that run; explicitly
disabling the worker with `--no-deepagent` also clears its configured skills.
Omitting the list still loads bundled skills and any enabled automatic sources.

## Evaluation profiles

```toml
[eval]
default_profile = "standard"

[eval.profiles.standard]
dataset = "./evaluation/questions.json"
workflow_types = ["single_agent"]
judge_type = "structured"
structured_output = true
recursion_limit = 200
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
