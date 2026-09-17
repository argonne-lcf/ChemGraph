# PBS jobs with Deep Agent skills

Run Deep Agent on a PBS submission host. It reads the bundled `chemgraph` and
`pbs-hpc` skills, writes Python and PBS scripts, submits with `qsub`, and inspects
the saved job ID and results. Calculations use the existing ASE Python engine;
an MCP server and Parsl are not required.

## Prepare and submit

Use a fresh shared run directory visible from login and compute nodes. The login
environment needs ChemGraph, its configured LLM provider, and PBS commands. The
compute environment needs ChemGraph and the requested calculator dependencies.
Prepare its initialization script and Python executable. Generate or stage the
structure and stage any model weights before submission. Local structure
preparation uses the existing ChemGraph tools and requires RDKit on the agent host.
For MACE-Polar, install the matching optional dependencies described in
[calculators](calculators.md).

On Polaris, consult the current [site job guide](https://docs.alcf.anl.gov/polaris/running-jobs/)
for queue limits, filesystem declarations, and environment settings. Supply your
own project, paths, and scientific choices in this example:

```bash
chemgraph run --interactive --workflow deep_agent \
  --deepagent-workspace /absolute/shared/run --model "$LLM_MODEL"
```

> Read the chemgraph and pbs-hpc skills, the local structure preparation guide,
> and the ASE batch example. Load the preparation tools, generate water from
> SMILES O into this workspace, and verify its structure. Use the returned
> absolute path in input.json. Write calculate.py and job.pbs. Optimize the water
> and calculate frequencies in one vib job using MACE-Polar, local model
> /absolute/path/to/polar-1-m.model, CUDA, float64, charge 0, multiplicity 1,
> BFGS, fmax 0.01 eV/Å, and 200 steps. Use Polaris project YOUR_PROJECT,
> queue debug, one node, walltime 00:30:00, and filesystems home:eagle.
> Source /absolute/path/to/environment.sh and use its Python at
> /absolute/path/to/environment/bin/python. Set up one process using one GPU,
> preserve the site's proxy settings, and set TMPDIR=/tmp after activation.
> Use result.json in this directory. Validate the scripts without running the
> calculation on the login node, submit once, and save the PBS job ID.

The built-in tool catalog is searchable by default; implementations and schemas
load only when requested. Optional `--tool` flags restrict the catalog. The skill
names which tools to load. The agent can replace its selection for result inspection
with `load_tools(["extract_output_json"])`. See [tool loading](skills.md#on-demand-local-tools).

The agent reads the Python example at
`/chemgraph-skills/chemgraph/references/ase-batch.md` and adapts the existing
`/chemgraph-skills/pbs-hpc/assets/job.pbs.template`. These are virtual file-tool
paths; shell commands use the real workspace path. Existing file-write and
execution approvals still apply. Inspect the generated files before submission.

## Inspect later

Restart the same CLI command with the existing workspace and ask:

> Read job.id and input.json, inspect that PBS job's status or retained history,
> then check stdout/stderr and result.json. Report convergence, energy, frequencies,
> and artifact paths. Do not submit another job.

An accepted PBS job runs independently of the agent session. Preserve
`submission.started`, `job.id`, and `qsub.stderr`, including on uncertain or failed
submissions. Scheduler completion alone does not establish scientific success.

The documented example is tested with real local water generation, EMT, and fake
PBS commands, including rejected or failed preparation. Real agent-driven
Polaris execution remains to be validated; record the transcript, generated
scripts, PBS job ID, compute hostname, and artifacts during that smoke test.
