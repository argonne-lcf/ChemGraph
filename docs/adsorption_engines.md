# Adsorption engines

ChemGraph separates the scientific adsorption request from the external engine
that renders and executes it. The initial drivers are gRASPA CUDA and the
legacy gRASPA SYCL implementation. ChemGraph does not build or distribute
gRASPA itself.

## Runtime configuration

Configure one active engine in `config.toml`. The executable and environment
must be valid on the worker, not only on the machine running the MCP server.

```toml
[adsorption]
engine = "graspa_cuda"
executable = "/path/on/polaris/to/nvc_main.x"
timeout_seconds = 7200

[adsorption.environment]
OMP_NUM_THREADS = "1"
```

For Aurora, use `engine = "graspa_sycl"`, the SYCL executable path, and add
deployment-specific variables such as `ZE_FLAT_DEVICE_HIERARCHY = "FLAT"`.
The old `[graspa]` section remains readable temporarily but is deprecated.

## Capabilities

| Engine | Components per simulation | Bundled adsorbates | Accelerator |
| --- | ---: | --- | --- |
| `graspa_sycl` | 1 | CO2, N2, H2O | Intel GPU |
| `graspa_cuda` | 1–3 | CO2, N2, H2O | NVIDIA GPU |

The selected driver validates these capabilities before input files are
written. A mixture submitted to the SYCL driver fails immediately rather than
being approximated as separate pure-gas calculations. CUDA renders one
component block per gas. Canonical H2O requests map to the CUDA `TIP4P`
molecule definition.

For mixtures, provide a mole fraction for every component and make the values
sum to one. Results contain uptake and uncertainty for each component and
pairwise adsorption selectivity.

```python
from chemgraph.schemas.adsorption_schema import AdsorptionRequest
from chemgraph.tools.adsorption_core import run_adsorption_core

request = AdsorptionRequest(
    input_structure_file="charged-framework.cif",
    temperature=298.15,
    pressure=100_000,
    components=[
        {"name": "CO2", "mole_fraction": 0.15},
        {"name": "N2", "mole_fraction": 0.85},
    ],
)
result = run_adsorption_core(request)
```

The bundled profiles use Ewald electrostatics and therefore require a CIF with
an `_atom_site_charge` column.

## HPC execution

`run_adsorption_ensemble` expands structures and explicit temperature/pressure
conditions through the configured Parsl, Ensemble Launcher, or Globus Compute
backend. Runtime configuration is resolved before submission and sent with each
job. Use `remote_structure_files` for Globus Compute because its worker usually
does not share the MCP server filesystem. A remote directory may be used only
with a shared-filesystem backend.

RASPA2 and RASPA3 can be added later as drivers behind the same request/result
API. RASPA2's text input and RASPA3's JSON input/output remain isolated inside
those future drivers.
