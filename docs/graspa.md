# gRASPA-SYCL: H2O adsorption

ChemGraph supports the SYCL gRASPA engine with its bundled H2O model. CUDA,
CO2, and N2 simulations are not supported by this integration. The force-field
parameters and cycle definitions are unchanged by the runtime cleanup.

Install ChemGraph, then configure the executable **on each execution worker**:

```bash
export CHEMGRAPH_GRASPA_EXECUTABLE=/path/to/graspa-sycl/bin/sycl.out
export CHEMGRAPH_LOG_DIR=/path/to/writable/results
```

Without the executable override, ChemGraph searches for `sycl.out` on `PATH`.
An invalid explicit override fails instead of selecting another binary. Load
the SYCL runtime/modules required by your installation before starting workers.
`OMP_NUM_THREADS=1` and `ZE_FLAT_DEVICE_HIERARCHY=FLAT` are defaults; existing
environment values are respected. Installing ChemGraph does not install gRASPA.

## Direct calculation

```python
from chemgraph.schemas.graspa_schema import graspa_input_schema
from chemgraph.tools.graspa_core import run_graspa_core

result = run_graspa_core(graspa_input_schema(
    input_structure_file="/path/to/framework.cif",
    adsorbate="H2O",
    temperature=298.15,  # K
    pressure=1000,       # Pa, not relative humidity
    n_cycles=10000,      # each initialization and production phase
    output_directory="water-screening",
    timeout_seconds=None,  # optional positive wall-time limit
))
print(result)
```

The LangChain `run_graspa` tool accepts the same schema under `graspa_input`
and retains its successful float return (mol/kg). It raises an actionable error
on failure. The core function returns a result dictionary; invalid input paths
or schemas fail before execution. A prepared run retains its diagnostics even
when process startup, execution, or parsing fails.

`output_directory` and `timeout_seconds` currently apply only to single
calculations. Ensemble requests reject these fields and
`discovery_timeout_seconds`, including explicit `null` values. Ensemble support
for these controls is deferred to a separate change; remote directory discovery
in the backend-agnostic MCP server still uses a fixed 30-second timeout.

## Artifacts and migration

Every invocation creates a new run directory. Relative output roots resolve
under `CHEMGRAPH_LOG_DIR`, or the worker's current directory when unset. The
default root is `graspa_runs`. The input CIF is never modified.

`output_result_file` is the stdout filename within that unique directory,
defaulting to `raspa.log`. Legacy directory-qualified values still select the
parent output root, with a warning; for single calculations, migrate to
`output_directory` plus a bare filename. Do not combine both root specifications.
Names reserved for the CIF, templates, stderr, or JSON metadata are rejected.

Use the returned `run_dir`, `stdout_path`, `stderr_path`, and `results_path`;
do not reconstruct paths from temperature or pressure. `input_structure_file`
identifies the original source, while `cif_path` identifies its run-local copy.
`results.json` includes run ID, conditions, exit code, elapsed seconds, and
uptake in mol/kg. Failures have `status="failure"`, null uptake, and
`error_type`/`message`. A nonzero exit can never be accepted as a successful
calculation, even when partial stdout contains an uptake value.

## Validation status

Hermetic tests validate preparation, parsing, isolation, and failure handling
without downloading models or requiring a GPU. Before using this integration
for scientific results, run a small H2O calculation on your SYCL installation,
compare the parsed uptake with stdout, and record the executable version and
sanitized output. Real-engine validation is a collaborator handoff requirement.
