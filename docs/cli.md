# Command-line interface

The `chemgraph` command provides query execution, evaluation, session
inspection, model discovery, dashboards, and Academy orchestration.

```text
chemgraph run        Run an agent query or interactive session
chemgraph eval       Evaluate a dataset
chemgraph session    List, inspect, export, or delete saved sessions
chemgraph models     List registered model identifiers
chemgraph dashboard  Launch the trace dashboard
chemgraph academy    Run Academy campaigns and workers
```

Use `chemgraph <command> --help` for the options supported by your installed
version.

## Run a query

```bash
chemgraph run \
  --model gpt-4o-mini \
  --workflow single_agent \
  --query "Build methane and optimize it with EMT."
```

Common options:

| Option | Meaning |
| --- | --- |
| `-q`, `--query` | Natural-language request |
| `-m`, `--model` | Provider/model identifier |
| `-w`, `--workflow` | Agent workflow; defaults to `single_agent` |
| `-o`, `--output` | Full `state` or `last_message` |
| `-s`, `--structured` | Request a structured final response |
| `-r`, `--report` | Enable HTML report generation |
| `--human-supervised` | Allow supported tools to pause for confirmation |
| `--recursion-limit` | Maximum graph steps; defaults to 20 |
| `--output-file` | Save the printed result to a file |
| `-v`, `-vv` | INFO or DEBUG diagnostics |

The legacy no-subcommand form, such as `chemgraph -q "..."`, remains supported,
but new scripts should use `chemgraph run`.

## Validate credentials

```bash
chemgraph models
chemgraph run --model gpt-4o-mini --check-keys
```

The check validates configuration; it does not prove that every external
calculator, database, or facility endpoint is reachable.

## Interactive mode

```bash
chemgraph run --interactive
```

Use `/help` inside the shell to see the commands available in your release.
Interactive sessions preserve conversation context and can be resumed later.

The `main_agent` workflow is a durable, supervisor-style agent and is only
available interactively:

```bash
chemgraph run --interactive --workflow main_agent
```

While a delegated subagent is working, the CLI prints each subagent tool call
and its arguments as it starts. Supervisor-only delegation and file-read calls
and tool results are not printed.

The supervisor's `read_file` tool reads only files stored in the durable graph
state by a subagent. It does not grant access to the host filesystem or to
files written under `CHEMGRAPH_LOG_DIR`.

The development workspace Deep Agent can execute broad filesystem and shell
actions. Call it directly with action reviews:

```bash
chemgraph run --interactive --workflow deep_agent \
  --deepagent-workspace /path/to/disposable-checkout
```

Or add the same graph to the supervisor as the `deepagent` subagent:

```bash
chemgraph run --interactive --workflow main_agent --deepagent \
  --deepagent-workspace /path/to/disposable-checkout \
  --deepagent-skill /workspace/.agents/skills/
```

The direct interactive workflow keeps one process-local thread until the model
or workflow changes. It is not restored across CLI processes. Shell and file
mutations use structured approve/reject prompts.

The selected directory is mounted for file tools at `/workspace`. Thus,
`--deepagent-workspace test/` makes `/workspace/example.py` refer to
`test/example.py`, not `test/workspace/example.py`. Shell execution uses the
absolute host-path mapping supplied to the model. Existing files under a
previously created `test/workspace/` directory are not migrated.

Skills are opt-in. Repeat `--deepagent-skill PATH` to provide ordered,
backend-relative source directories; the later source wins when two sources
contain the same skill name. For the virtual CLI workspace, a conventional
project source can be selected explicitly as
`/workspace/.agents/skills/`. ChemGraph does not scan project or user
directories automatically.

Each source must contain one directory per skill, and each skill directory
must contain a `SKILL.md` with `name` and `description` YAML frontmatter.
Metadata is cached in the checkpoint for the life of the thread, so restart or
reinitialize the workflow after changing the available skill set. Reading a
skill does not require an action review, while executing a bundled script or
mutating its files continues to use the normal Deep Agent approval policy.

Deep Agent run logs use the normal ChemGraph locations rather than the selected
workspace. The default is `cg_logs/session_*` for state JSON plus the configured
session database. Pending approval state and the final resumed state are both
recorded.

For automation, headless execution must opt out of those prompts explicitly:

```bash
chemgraph run --workflow deep_agent \
  --deepagent-workspace /path/to/disposable-checkout \
  --deepagent-skill /workspace/.agents/skills/ \
  --deepagent-dangerously-skip-approvals \
  --query "Run the repository tests and summarize failures."
```

The unsafe flag is accepted only for non-interactive `deep_agent`, is not read
from TOML, and requires an explicit workspace. The backend's shell is not
confined to the workspace, so use a disposable, isolated environment.

## Saved sessions

CLI sessions are stored in `~/.chemgraph/sessions.db`.

```bash
chemgraph session list
chemgraph session show <session-id>
chemgraph run --resume <session-id> -q "Continue with frequency analysis."
```

Run `chemgraph session --help` before deletion or other state-changing session
operations. Resume a session with a compatible workflow.

## Use MCP tools

Connect to one or more streamable-HTTP servers:

```bash
chemgraph run \
  --mcp-url http://localhost:9003/mcp/ \
  -q "Build a 3D structure for methane."
```

Use repeated/configured server definitions for larger deployments. See
[MCP servers](mcp_servers.md) for transports and client configuration.

## Trace and dashboard

Use `--trace-dir <directory>` with `single_agent` to record trace information,
then inspect the dashboard options available in your version:

```bash
chemgraph dashboard --help
```

## Evaluation and Academy

Evaluation needs an explicit dataset or a configured profile:

```bash
chemgraph eval --help
```

Academy commands require the `academy` extra and, depending on the backend,
additional execution extras:

```bash
chemgraph academy --help
```

See [Evaluation](evaluation.md) and [HPC and Academy](hpc_and_academy.md).
