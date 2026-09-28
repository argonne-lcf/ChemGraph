---
name: chemgraph
description: Run ChemGraph calculations, prepare ASE batch scripts, stage structures, and inspect chemistry results through local Python or attached MCP tools.
license: Apache-2.0
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# Use ChemGraph

Preserve the requested calculator, scientific parameters, and execution method.
Ask for missing scientific choices; report missing dependencies. Skills do not
grant tools or override approvals.

## Choose the workflow

- **Prepare batch files:** read [the ASE example](references/ase-batch.md) and
  `pbs-hpc`. Use the documented Python API directly; do not load `run_ase` merely
  to obtain its schema or inspect its implementation. For user-selected IRI
  submission, follow `hpc-batch` instead. Preparation alone does not submit a job.
- **Run locally:** use native tools. Load known names directly with `load_tools`,
  grouping tools needed for the next operations in one request; use `search_tools`
  for unfamiliar capabilities. `run_ase` executes ASE; `extract_output_json`
  inspects results. Read the loaded schemas before supplying arguments.
  For structure generation, read [local preparation](references/structure-preparation.md):
  `smiles_to_coordinate_file` generates coordinates and `file_to_atomsdata`
  validates them. User-supplied explicit coordinates can be written as XYZ and
  validated without SMILES generation. Keep validation before calculation.
- **Use attached MCP tools:** read [MCP workflows](references/mcp-workflows.md)
  and the attached schemas; local paths may not be visible to the server.
- **Configure Python or CLI:** read [interface guidance](references/python-and-cli.md)
  only when setup is part of the task.

Open skills at their catalog paths and follow direct reference links; directory
listing is unnecessary when paths are known. Batch independent reads and load
only task-relevant references, including any the user explicitly requests.
Use targeted schema/source inspection only for a concrete gap or validation
error; avoid printing whole implementations to rediscover documented behavior.

Submit requested calculations once and retain job IDs; poll pending jobs without
resubmitting. Report actual failures, units, calculator/model and artifact paths.
Prefer `potential_energy` over legacy `single_point_energy`; distinguish
convergence from job completion. Keep full structures and logs in files and read
only the portions needed to validate or diagnose results.

## Resource paths

Resolve references against this skill's directory. `/chemgraph-skills/` is a
virtual, read-only route; copy needed assets into the execution workspace before
using them in shell commands. Establish host/virtual path mappings and stage
files when agent, shell, MCP server or compute worker filesystems differ.
