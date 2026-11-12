# Milestone 3: portable PBS operations and Bebop guidance

## PR integration with current main

The initial implementation and verification below were performed in the older
local checkout. The PR is based on current `main`, which already includes
bundled-skill discovery, broader skill package-data rules, and the `pbs-hpc`
skill with ALCF references. Those features are preserved. The new `pbs` and
`bebop` skills use existing discovery without runtime changes; core license and
attribution metadata and catalog expectations follow current repository rules.

The preview module, TOML example, dedicated tests, and `lcrc-pbs` skill were
uncommitted local prototypes, so they produce no deletion diff against `main`.
The earlier handoffs are included as explicitly superseded historical records.

## Delivered

The public skills are now `pbs` and `bebop`, composed as:

```text
PBS operations + selected machine guidance + software guidance
```

- [pbs](../../src/chemgraph/skills/pbs/SKILL.md) covers native shell preparation,
  submission, status, cancellation, changes, holds, dependencies, arrays,
  ambiguous submissions, recovery, and application-result checks. It works
  independently of machine skills and ChemGraph imports, including when copied
  into another compatible agent's skill directory.
- [bebop](../../src/chemgraph/skills/bebop/SKILL.md) supplies machine identity,
  resource conventions, queues/accounts/limits, filesystems/quota, module
  discovery, example directives, and official sources verified on 2026-11-12.
  Documented capacities and limits are distinguished from runtime choices.
- Removed `src/chemgraph/skills/lcrc-pbs/`,
  `src/chemgraph/hpc_configs/lcrc.py`, `examples/lcrc/bebop.toml`, and
  `tests/test_lcrc_profile.py`. The preview CLI and Python APIs are removed;
  no compatibility wrapper or second store of machine settings is maintained.
- Retained the lightweight `chemgraph.skills.skill_directory()` locator from
  the prototype and the existing package-data rules on `main`.
- Updated [HPC documentation](../hpc_and_academy.md) and marked
  [milestone 1](01-lcrc-pbs-preview.md) and
  [milestone 2](02-local-pbs-tools.md) as superseded historical records.

No fixed template, receipt format, Python scheduler tool, or backend was added.
No personal account, email address, or software installation path is embedded.
The existing Aurora/Polaris/Crux execution configurations, Academy profiles,
backend code, and Multiwfn work are outside this change.

## Usage and discovery

Read either skill directly, or locate the installed root:

```bash
python -c 'from chemgraph.skills import skill_directory; print(skill_directory())'
```

Expose that root to a compatible agent's filesystem and skill loader. The two
subdirectories have independent frontmatter names and descriptions. The locator
does not register skills, execute commands, or import an execution backend.

Example prompts once the skills are available:

- “Use the PBS skill to inspect this job on the current machine.”
- “Use the PBS and Bebop skills to prepare this submission.”
- “Use the PBS skill with the supplied Polaris settings.”

Preparation alone does not submit a job. For an authorized run, specify the
script and submission directory along with the required site settings. Native
PBS job IDs are not ChemGraph backend batch IDs and must not be passed to the
existing Future-based job-control interfaces.

## PR verification on current main

- Core skill lint (`chemgraph skills lint src/chemgraph/skills --core`), Ruff,
  changed Markdown links, and whitespace checks passed.
- The four skill suites (`test_skill_backend.py`, `test_skill_distribution.py`,
  `test_skill_lint.py`, and `test_skills.py`) passed: **97 tests**. These include
  clean wheel/sdist builds, reading both skills from extracted and zip-imported
  distributions, the locator, and catalog discovery with both new names.
- Validation used temporary dependency overlays under `/tmp`; the original
  checkout and its environment were preserved. Current `main` package-data,
  existing skill runtime, ALCF skill/references, HPC configurations, Academy
  profiles, and execution modules have no diff in this PR.

## Initial local verification

Completed on 2026-11-12 in the older checkout before PR integration:

- Both skills passed the skill-creator `quick_validate.py` checks.
- Repository-wide `uv tool run ruff check .` passed. `git diff --check` and
  explicit whitespace checks for the untracked Markdown files passed.
- All 12 relative Markdown links in the affected documentation/skills resolved;
  linked official facility pages were read. Searches found removed names only
  in superseded handoffs or explicit removal notes, with no active source/test
  callers. The portable skill was checked independently for site/software
  coupling and personal example values.
- A clean source copy excluding cached build and egg-info artifacts produced
  `chemgraph-0.6.0-py3-none-any.whl` using the configured
  `setuptools.build_meta.build_wheel` backend. Both new skill files matched the
  source bytes; the old skill, preview module, and earlier PBS wrappers were
  absent. The existing `python -m build` frontend was incomplete, so verification
  called the configured backend directly without changing project dependencies.
- Installed the wheel with `uv pip install --no-deps --offline` into a fresh
  temporary venv. Isolated Python imported `skill_directory()`, found exactly
  `pbs` and `bebop`, and required no Deep Agent, HPC config, or execution imports.
- Hash checks confirmed all 29 sampled boundary files were unchanged: the
  locator, packaging configuration, existing HPC configurations, Academy runtime
  profiles, execution modules, and pre-existing Multiwfn work.

The required core command was attempted from the checkout:

```bash
PYTHONPATH=src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest tests/ -k 'not tblite' -p no:cacheprovider
```

Collection stopped with **12 errors**, all `ModuleNotFoundError: deepagents`
(the existing environment limitation recorded in earlier milestones); 9 tests
were skipped and 2 deselected. This is not a passing core-suite result. No live
LLM/HPC test flags were enabled and the project environment was not upgraded.
No new scheduler implementation or behavioral test suite was introduced.
Build/install artifacts and the core-test log were kept under `/tmp`, outside
the checkout.

Offline workflow review (instruction walkthroughs, not scheduler executions):

| Scenario | Expected behavior checked in the instructions |
| --- | --- |
| Inspect a queued job | Use the full ID with `qstat -f JOB_ID`; inspect comments, holds, dependencies and resources; report Q without treating it as failure |
| Submit with supplied site settings | Resolve project/queue/resources and directory, check script syntax and directives, submit only within authorization, retain the returned ID and effective settings |
| Cancel an identified job | Confirm identity and scope, distinguish array/subjob targets, use `qdel JOB_ID`, capture response and inspect the resulting state |
| Submission response is lost | Search current jobs and supported history using several identity fields; do not resubmit or cancel a guessed candidate while ambiguity remains |

Optional array syntax/index variables, JSON output, history support/retention,
and allowed resource changes require capability checks. The portable skill has
no facility names, resource defaults, software-specific setup, or imports.
Scheduler completion is explicitly distinct from application correctness.

## Remaining integration work

Existing bundled-skill discovery on current `main` makes these instructions
available. This change adds no discovery runtime or scheduler execution layer.
Live agent-driven PBS/Bebop operation remains to be validated separately.

Future standalone machine skills can follow the Bebop section structure
without changing the PBS skill. Existing ALCF references in `pbs-hpc` remain
unchanged; no additional machine skills are added here. Supplied site
settings already work with the portable instructions. Software guidance,
including Gaussian setup and restart validation, remains separate. cLR,
ExternalIteration, remote submission, and additional backend integrations are
deferred. No live scheduler submissions or control commands were run.
