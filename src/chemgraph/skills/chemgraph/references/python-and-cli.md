# Python and CLI workflows

Use the installed CLI's `chemgraph run --help` for available options. Workflow
`deep_agent` is standalone; `main_agent` is a supervisor with a chemistry worker
and an optional workspace worker enabled by `--deepagent`.

```sh
chemgraph run --interactive --workflow deep_agent --deepagent-workspace .
chemgraph run --interactive --workflow main_agent --deepagent --deepagent-workspace .
```

Attach an externally managed MCP server with `--mcp-url URL`. Starting a server
and attaching to one are separate actions. Do not assume that an HTTP MCP server
shares the CLI machine's filesystem.

In Python, use `ChemGraph` from `chemgraph.agent.llm_agent`, supply the intended
`workflow_type`, and call `await agent.run(query)`. For a local Deep Agent
workspace, supply a `deepagents.backends.LocalShellBackend` as
`deepagent_backend`. The default state backend has no shell. Shell access follows
the existing approval policy; preserve structured interrupts when resuming.

Bundled skills load automatically. Personal `~/.chemgraph/skills/` and workspace
`.agents/skills/` directories are discovered for supported local backends.
`deepagent_skills` / repeated `--deepagent-skill` values are additional,
backend-relative sources, with later sources overriding earlier ones.

For chemistry tool artifacts, relative writes use `CHEMGRAPH_LOG_DIR` through
`chemgraph.tools.ase_core._resolve_path`; readers use `_resolve_existing_path`.
Prefer absolute paths when crossing process boundaries. A local virtual
`/workspace/file.xyz` maps to the configured host workspace for shell commands,
but that mapping does not establish visibility on a remote MCP server.

Select HPC execution using the existing `[execution]` configuration and installed
extras. Do not treat a ChemGraph Parsl/Globus execution backend as a Deep Agents
filesystem backend. They implement different interfaces.

## Direct PBS ASE jobs

Use this route only when the user explicitly requests direct PBS execution.
Missing MCP tools do not select this route automatically. Read `pbs-hpc` for
site resources, environment setup, staging, submission, and monitoring.

Use the maintained `scripts/run_ase_pbs.py` helper instead of generating a Python
runner. Stage its exact bytes from the installed ChemGraph package into the real
execution workspace with the configured Python environment:

```sh
python - <<'PY'
from importlib import resources
from pathlib import Path

runner = resources.files("chemgraph.skills").joinpath(
    "chemgraph", "scripts", "run_ase_pbs.py"
)
Path("run_ase_pbs.py").write_bytes(runner.read_bytes())
PY
```

The virtual `/chemgraph-skills/` route is not a shell path. If staging on another
host, transfer the copied helper and inputs to the execution workspace using the
available staging tools. Keep the installed ChemGraph version consistent.

Create a separate `input.json` matching `ASEInputSchema`, preserving the requested
calculator, driver, and scientific parameters. In the PBS script, after changing
to the execution workspace and setting up the environment, run:

```sh
python run_ase_pbs.py --input /absolute/execution/path/input.json > summary.json
```

Invoke the runner once per calculation, not once per MPI rank. It requires
`PBS_JOBID`, a readable nonempty `PBS_NODEFILE`, and a hostname in that allocation
before importing the chemistry stack. It does not submit jobs or configure
resources. `pbs-hpc` owns the scheduler script and job lifecycle.

The runner preserves the working directory and ChemGraph path conventions;
paths inside the input are not relative to the JSON file's directory. Prefer
absolute paths for inputs and outputs. It validates the artifact at the exact
`results_file` returned by `run_ase_core`, without searching for another file.
Python diagnostics go to stderr; stdout contains one JSON summary with the core
result metadata, or a failure summary. Exit codes are:

- `0`: successful calculation and any required optimization converged.
- `1`: argument, allocation, input, execution, or output-artifact failure.
- `2`: completed calculation with unconverged optimization (`opt`, `vib`,
  `thermo`, or `ir`). The core result may still say `status: success`; report the
  unconverged state. `energy` and `dipole` do not require optimization convergence.
