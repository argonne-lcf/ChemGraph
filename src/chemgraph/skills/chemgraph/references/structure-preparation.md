# Prepare structures with local tools

Use existing ChemGraph tools for supported preparation operations before writing
calculation inputs. For water, use SMILES `O`; no name lookup is needed. For other
names, `molecule_name_to_smiles` uses PubChem and requires network access. Preserve
stereochemistry when it matters.

Standalone Deep Agent exposes the built-in tool catalog by default. If
`load_tools` is available, request the names needed for the current step, for
example `load_tools(["smiles_to_coordinate_file", "file_to_atomsdata"])`. This
replaces the active selection; use the returned native schemas on the next step.
Use `search_tools` for unfamiliar capabilities. Loading does not execute tools or
grant permissions, and the selection clears when the turn completes. A missing
tool may have been excluded by a configured filter; searching cannot enable it.
Call the loaded native tools instead of invoking their implementation through
`execute` or inspecting source to rediscover their arguments.

For water preparation:

1. Choose a fresh run directory on the host where the preparation tool executes.
   On a submission host, use a shared directory visible on compute nodes. On a
   laptop, use the selected local workspace; stage its run directory later.
2. Call `smiles_to_coordinate_file` with `smiles="O"` and `output_file` set to
   the absolute host path of `water.xyz` in that directory. This is lightweight
   RDKit coordinate generation, not the requested ASE calculation.
3. Inspect the tool's `ok`, `path`, and `natoms` fields. Read the returned path
   with `file_to_atomsdata` and verify three atoms: one O and two H, with finite
   coordinates. Stop preparation on an error or rejected write; do not submit
   an input that has not been generated and checked.
4. For direct PBS, put the verified compute-visible path in
   `ASEInputSchema.input_structure_file`. For IRI, stage the structure with the
   run directory and use a run-relative path such as `water.xyz` in `input.json`.
   A laptop-generated absolute `path` belongs to the laptop, not the compute node.
   Follow [the ASE batch example](ase-batch.md) and the selected
   [PBS](../../pbs-hpc/SKILL.md) or [IRI](../../iri-hpc/SKILL.md) workflow.
   Keep the requested calculator on compute nodes.

Local registry tools execute in the agent process, on the agent host. Their
filesystem is independent of a virtual file backend, remote shell, or MCP server.
`/workspace/...` is a file-tool mount, not a host path to pass to a local chemistry
tool. Relative writes use `CHEMGRAPH_LOG_DIR`; carry the returned absolute path
forward instead of guessing where a relative filename landed. For separate
hosts, stage the generated artifact and use the verified compute-visible path.

If native tools are not configured, run this Python fallback from the selected
fresh run directory in an environment with ChemGraph, RDKit, and ASE installed:

```python
from pathlib import Path

import numpy as np
from ase.io import read
from chemgraph.tools.cheminformatics_core import smiles_to_coordinate_file_core

artifact = smiles_to_coordinate_file_core(
    smiles="O", output_file=str(Path("water.xyz").resolve())
)
if not artifact["ok"] or artifact["natoms"] != 3:
    raise ValueError(f"Water preparation failed: {artifact}")
atoms = read(artifact["path"])
if sorted(atoms.get_chemical_symbols()) != ["H", "H", "O"]:
    raise ValueError("Expected one O and two H atoms")
if atoms.positions.shape != (3, 3) or not np.isfinite(atoms.positions).all():
    raise ValueError("Invalid water coordinates")
print({"path": artifact["path"], "natoms": len(atoms)})
```

Reuse the core implementation rather than writing coordinates or recreating
chemistry logic. Report missing dependencies and stop on preparation errors.
The checked local artifact still needs staging when execution is on another host.

Keep full structures, trajectories, and logs in files. Return concise status,
atom counts, relevant results, and artifact paths; load more data only when needed.
