# Calculators

ChemGraph uses ASE calculators for energies, geometry optimization, vibrations,
thermochemistry, and related tasks. Available calculators are detected at
runtime; optional engines that cannot be imported or located are omitted.

## Calculator overview

| Calculator | Setup | Best use in onboarding |
| --- | --- | --- |
| EMT | Included with ASE | Fast, offline smoke tests; limited elements/accuracy |
| MACE | MACE-MP included in core; Polar requires an add-on | Polar medium is preferred when installed; otherwise MACE-MP supplies energies and forces |
| TBLite | `pip install "chemgraph[calculators]"` | Semiempirical calculations |
| UMA / fairchem | `pip install "chemgraph[uma]"` in a separate environment | Advanced universal ML potential |
| AIMNet2 | Install its package/model dependencies separately | Supported molecular ML route when importable |
| NWChem | Install/configure NWChem for ASE | External quantum chemistry |
| ORCA | Install/configure ORCA for ASE | External quantum chemistry |

The agent may infer a calculator from a request, but explicitly naming one makes
runs more reproducible:

```bash
chemgraph run -q "Optimize water with EMT and report the final energy."
```

## What tools can run

Calculator-backed tools cover operations such as:

- single-point energy and force calculations;
- geometry optimization;
- vibrational frequencies and normal modes;
- infrared and thermochemistry workflows where supported;
- calculator-specific properties such as dipoles.

Support depends on the selected calculator. A valid property for one engine may
not exist for another.

## EMT for setup checks

EMT is lightweight and requires no download, making it a useful plumbing test.
It is not a general-purpose high-accuracy molecular method. Do not treat an EMT
result as scientifically appropriate merely because the workflow completed.

## MACE downloads

MACE is installed with the core package. When no calculator is specified,
ChemGraph uses MACE-Polar (`mace_polar`, `polar-1-m`) if the `graph-longrange`
add-on is installed; otherwise it uses MACE-MP (`mace_mp`, reported as
`medium-mpa-0`). Explicit calculator selections are preserved.

Starting with v0.7.0, install Polar from the matching source checkout or extracted
source distribution with:

```bash
python -m pip install . -r requirements/mace-polar.txt
python -m pip check
```

For a published release, pin both ChemGraph and its requirements to that release:

```bash
python -m pip install 'chemgraph==0.7.0' -r https://raw.githubusercontent.com/argonne-lcf/ChemGraph/v0.7.0/requirements/mace-polar.txt
```

Use the matching tag or commit when installing another version; the add-on files
are introduced in v0.7.0. The conda environment and Docker images explicitly
install Polar. Run `conda env create -f environment.yml` from the checkout root
so its supplemental requirements path resolves correctly.

Pretrained weights may be fetched on first use. In restricted or
offline environments, pre-stage the required model cache or choose EMT for the
initial test. MACE-Polar checkpoints are distributed under the Academic
Software License (ASL); review its terms before use.

MACE-Polar can calculate molecular dipole moments with `driver="dipole"`.
ChemGraph reports these dipole vectors in Debye.
MACE-MP does not supply Polar's dipole or IR capabilities. An explicit Polar
request without the add-on reports installation instructions before loading weights.

## UMA dependency isolation

The UMA/fairchem stack can require an `e3nn` version that conflicts with MACE.
Use a separate virtual environment for UMA rather than forcing incompatible
versions into the core environment.

## External executables

Installing ChemGraph's Python dependencies does not install ORCA, NWChem,
FDMNES, Vina, or site-specific simulation programs. Confirm licenses,
executables, environment variables, pseudopotentials/basis data, and scheduler
access independently.

## Artifacts and paths

Tool writers resolve relative artifact paths under the current session log
directory. By default it is a unique directory below `cg_logs/`. Readers search
using the same session-aware path handling. Choose a different parent before
launching ChemGraph:

```bash
export CHEMGRAPH_LOG_DIR="/absolute/path/to/calculations"
```

Typical outputs include XYZ files, trajectories, JSON/CSV data, spectra, normal
modes, and HTML reports.

## Scientific validation

Always record the calculator and model version, numerical settings, charge and
spin state, boundary conditions, units, and convergence criteria. Check whether
the method covers the system's elements and chemistry. Agent-generated prose is
not a substitute for inspecting calculation outputs.
