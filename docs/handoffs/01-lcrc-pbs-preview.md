# Milestone 1: LCRC/Bebop profile and PBS preview

> **Superseded by [milestone 3](03-pbs-bebop-skills.md).** This is a historical
> record, not current usage guidance. The `lcrc-pbs` skill was replaced by `pbs`
> and `bebop`; the LCRC profile/preview module, CLI, TOML example, and dedicated
> tests were removed. Commands, API descriptions, and test results below describe
> the earlier milestone only.

Subsequent decision: PBS management is skill-only; see the
[revised milestone 2 handoff](02-local-pbs-tools.md). This preview helper remains
optional. The next-step description below records the original milestone plan.

## Delivered

A small, dependency-free Python module loads a `[lcrc]` TOML profile and renders
PBS script text. It validates account/queue identifiers, resource requests,
walltime, module names, job names, working directories, and command arguments.

The renderer does not write files or contact PBS. Command arguments and working
directories are shell-quoted. Generated scripts require `PBS_JOBID`, change to
the explicit working directory, optionally load configured modules, and preserve
the command's exit code. `memory_gb`, `ncpus`, and `mpiprocs` apply to each PBS
select chunk; they are requested resources, not a hardware inventory. The
36-CPU/120-GB/72-hour defaults reproduce the existing project's allocation
request and are configurable.

Files:

- `src/chemgraph/hpc_configs/lcrc.py`: profile loader, renderer, preview CLI.
- `examples/lcrc/bebop.toml`: editable site-profile example.
- `tests/test_lcrc_profile.py`: hermetic tests.

## Try the preview

From the ChemGraph repository root:

```bash
PYTHONPATH=src .venv/bin/python -m chemgraph.hpc_configs.lcrc \
  --profile examples/lcrc/bebop.toml \
  --workdir "$PWD" \
  --job-name cg_preview \
  -- /bin/echo 'Hello from a future PBS job'
```

This prints a script to stdout. `YOUR_PROJECT` is deliberately a placeholder:
replace it with your allocation in a personal profile before any later live
submission. The default profile requests no explicit queue or software modules.
Account authorization, resource availability, and queue limits are not checked
by this offline preview.

To save and check the preview explicitly:

```bash
PYTHONPATH=src .venv/bin/python -m chemgraph.hpc_configs.lcrc \
  --profile examples/lcrc/bebop.toml \
  --workdir "$PWD" \
  -- /bin/echo 'Hello' > /tmp/chemgraph-preview.pbs
bash -n /tmp/chemgraph-preview.pbs
```

Python callers can use `load_lcrc_profile(path)` and
`render_pbs_script(profile, command, workdir, job_name="chemgraph")`.
`command` is a list of literal arguments, not a shell command string.

## Test

```bash
PYTHONPATH=src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest tests/test_lcrc_profile.py -p no:cacheprovider -q
```

The focused suite passed: **21 tests**. Tests use temporary directories and
local Bash processes, with a fixture `PBS_JOBID` where necessary. No scheduler,
Gaussian, Multiwfn, network, or LLM is used by these tests.

Repository-wide Ruff checks passed using `uv tool run ruff check .`.
The full core pytest command was also attempted with `PYTHONPATH=src`; collection
is blocked by the pre-existing missing `deepagents` dependency (12 collection
errors). The environment was not upgraded in this milestone. The focused tests
above exercise the new code directly from the checkout and pass independently.

## Handoff and next milestone

This change is uncommitted in the existing `feature/multiwfn-tool` checkout.
Pre-existing Multiwfn edits and untracked files were preserved. No production
calculation files were modified, and no jobs were submitted.

Stop here for review. The next change is direct local PBS submission and status
inspection: reuse this profile/renderer, invoke `qsub`/`qstat` on the login node,
and retain submission receipts to avoid duplicate jobs. Then add Gaussian as a
separate milestone. Deep Agent registration and software execution are not yet
implemented. Gaussian scratch settings belong to that later software layer.

Corrected linear response, ExternalIteration, remote submission, Globus setup,
and Parsl allocation provisioning remain deferred.
