# Milestone 2 revision: PBS management as a skill only

> **Superseded by [milestone 3](03-pbs-bebop-skills.md).** This is a historical
> record, not current usage guidance. The `lcrc-pbs` skill was replaced by `pbs`
> and `bebop`; the LCRC profile/preview module, CLI, TOML example, and dedicated
> tests were removed. Commands, API descriptions, and test results below describe
> the earlier milestone only.

The user chose flexible native PBS commands through the agent's existing shell
access. This revision replaces the earlier PBS Python wrappers with instructions.
The filename is retained so existing handoff links still resolve.

## Current files and behavior

- `src/chemgraph/skills/lcrc-pbs/SKILL.md`: operational skill for preparation,
  qsub, qstat, qdel, qalter, qhold, qrls, arrays/dependencies, and recovery.
- `src/chemgraph/skills/__init__.py`: `skill_directory()` locates packaged skills.
- `pyproject.toml`: includes `*/SKILL.md` in the `chemgraph.skills` package.
- `docs/hpc_and_academy.md`: points to this handoff.

Removed the new `pbs_core.py`, `pbs_tools.py`, and `test_pbs_tools.py`. Their CLI,
PBS_TOOLS registry, typed wrappers, receipt persistence, locking and automatic
submission deduplication are no longer provided. Old test counts for those
wrappers are historical and do not describe this revision.

The earlier LCRC profile/preview helper and its tests remain as optional script
preparation utilities. The skill does not require them. Existing ChemGraph
backend job tools and pre-existing Multiwfn work are unchanged.

## Existing skill coverage

`.opencode/skills/chemgraph/SKILL.md` covers ChemGraph development, not operational
PBS use. It was inspected and left unchanged. The new operational skill lives
inside the package so it can be supplied to ChemGraph's Deep Agent.

## How to use

Read `src/chemgraph/skills/lcrc-pbs/SKILL.md`, or locate the installed skill root:

```bash
PYTHONPATH=src .venv/bin/python -c \
  'from chemgraph.skills import skill_directory; print(skill_directory())'
```

The agent must have shell access on the LCRC login node and access to this skill
through its filesystem backend. Its skill sources must explicitly include the
backend-visible skill directory. StateBackend does not automatically read host
files. The current main_agent/Deep Agent construction has not yet been wired to
load the skill; automatic registration is the next separate milestone.

Once loaded, ask the agent, for example:

> Use the lcrc-pbs skill to inspect my jobs and queues. Do not submit anything.

Or, for an actual authorized submission:

> Use the lcrc-pbs skill to submit /absolute/path/job.pbs from
> /absolute/calculation/directory and report the PBS job ID.

The agent uses native commands such as `qsub job.pbs`, `qstat -f JOB_ID`, and
`qdel JOB_ID` through its existing execute tool. It may adapt scripts, resources,
arrays and dependencies to the task. Submission/cancellation commands modify
real jobs. No live commands were run during this revision.

## Verification and limitations

Verification passed: repository-wide `ruff check .` and `git diff --check`.
A fresh wheel was built and inspected: it contains the updated skill and excludes
`pbs_core.py` and `pbs_tools.py`. Source/test searches found no remaining references
to the removed wrapper APIs. No new behavioral tests are needed for this instruction-only revision.
The previous 21 LCRC profile tests remain available. The full core-suite collection
was previously blocked by missing `deepagents`; the environment was not changed.

The skill instructs the agent to record job IDs, reconcile ambiguous submissions,
and check scientific output, but instructions do not enforce deduplication or
scientific validation. No automatic receipts or background monitor are installed.

## Next milestone

Stop here for review. Next, make this skill discoverable by the existing Deep
Agent while retaining its shell access. Software integrations (Gaussian first)
remain separate changes. cLR, ExternalIteration and remote submission are deferred.
