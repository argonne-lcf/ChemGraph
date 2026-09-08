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

Ideal-gas thermochemistry uses the requested temperature and pressure (defaults:
298.15 K and 101325 Pa). ASE single and ensemble inputs also interpret a null
temperature as 298.15 K; supplied temperatures must be finite and positive.
Enthalpy and Gibbs energy are reported in eV; entropy is reported in eV/K with a
separate `entropy_unit` field. HTML reports accept eV/K entropy, including legacy
results without this field, and reject other declared entropy units. The shared
**Units** selector converts energy and entropy together; entropy labels change
to kJ/(mol K) or kcal/(mol K) for the corresponding molar energy selection.

ChemGraph requires ASE >= 3.29.0. For `thermo`, the complete complex spectrum is
passed to `IdealGasThermo(vib_selection="highest", ignore_imag_modes=True)`.
ASE selects the expected number of modes by squared energy, then removes any
remaining imaginary or zero-energy modes. ChemGraph does not apply an additional
frequency cutoff or convert imaginary frequencies to real ones.

Reported thermochemistry frequencies, CSV entries, and trajectories match the
energies ASE actually used. `vibrational_frequencies.mode_indices` contains their
original zero-based ASE indices; `all_modes` preserves every input mode as
`mode_index`, `energy` (meV), and `frequency` (cm-1), with an `i` suffix for
imaginary values. HTML displays mode numbers starting at 1 and includes the full
spectrum with used/excluded labels. Standalone `vib` and `ir` output is unchanged.

Thermochemistry metadata records `ase_version`, `vib_selection`,
`ignore_imag_modes`, `n_imag`, `raw_imaginary_mode_count`, and `warnings`.
`n_imag` is ASE's cleanup count **after selection**, including zero-energy modes;
it is not the number of imaginary modes in the complete input. Selection may
already have excluded imaginary modes even when `n_imag` is zero. Successful
thermochemistry with excluded modes does not establish structural stability.
Warnings identify imaginary input modes and calculations with no vibrational
contribution. If ASE raises or returns non-finite thermodynamic values, ChemGraph
returns a failure with `results_file` pointing to the completed structure,
potential energy, convergence state, and full spectrum; the JSON records
`success=false` and the error, with no thermochemistry values.

Single atoms skip finite-difference vibrations for `thermo`, `vib`, and `ir`.
Atomic thermochemistry includes translation and uses the calculator's reported
multiplicity for the electronic-spin contribution. If no multiplicity is
reported, ChemGraph logs a warning and assumes a singlet, omitting the
electronic-spin entropy of open-shell species. This does not infer ground-state
multiplicities or add spin dependence to a calculator's potential energy.
Rotational symmetry analysis expects an isolated, unwrapped molecule; periodic
images are not reconstructed.

## EMT for setup checks

EMT is lightweight and requires no download, making it a useful plumbing test.
It is not a general-purpose high-accuracy molecular method. Do not treat an EMT
result as scientifically appropriate merely because the workflow completed.

## MACE downloads

MACE is installed with the core package. When no calculator is specified,
ChemGraph uses MACE-Polar (`mace_polar`, `polar-1-m`) if the `graph-longrange`
add-on is installed; otherwise it uses MACE-MP (`mace_mp`, reported as
`medium-mpa-0`). Explicit calculator selections are preserved.

Starting with v0.7.0, install Polar from the root of the matching source checkout
or extracted source distribution (the directory containing `pyproject.toml`):

```bash
python -m pip install . -r requirements/mace-polar.txt
python -m pip check
```

After v0.7.0 is published on PyPI and its Git tag exists, a wheel installation
can use the command below from any directory. Before publication, use the
development checkout instructions above. Pin both ChemGraph and its requirements
to the same release:

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
Unsupported dipole and IR requests return a failure with an explanation; IR checks
dipole support before starting optimization or vibrational analysis.

In the UI, **Automatic** leaves `chemistry.calculators.default` absent from the
saved TOML, so unrelated settings changes preserve detection. Selecting a named
calculator stores an explicit choice. Restart ChemGraph after installing an add-on
so its initialization-time calculator descriptions reflect the new environment.

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
