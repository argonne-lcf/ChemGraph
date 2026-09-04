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
| `--reasoning-effort` | Reasoning effort for a supported model |
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

### Reasoning effort

Verified Argo GPT-5.6 and Claude Opus models accept an explicit reasoning
effort:

```bash
chemgraph run \
  --model argo:claude-opus-4.8 \
  --reasoning-effort xhigh \
  --query "Analyze the reaction mechanism."
```

The GPT-5.6 routes accept `none`, `low`, `medium`, `high`, `xhigh`, and `max`.
The Claude Opus 4.8 and Opus 5 routes accept `low`, `medium`, `high`, `xhigh`,
and `max`. Both model families default to `medium`. Claude effort controls
overall response work through Anthropic's `output_config.effort`; it does not
make ChemGraph explicitly enable adaptive thinking.

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
actions. Enable it only in a disposable, trusted workspace after reviewing the
CLI warning.

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
