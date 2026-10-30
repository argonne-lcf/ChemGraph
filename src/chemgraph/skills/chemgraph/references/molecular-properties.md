# Experimental molecular properties with ORCA

Use this workflow for local, unscoped gas-phase closed-shell singlet research.
Supported atoms are H/C/N/O/F/P/S/Cl/Br, with MMFF94s parameters and explicitly
specified tetrahedral/E/Z stereo (including stereogenic protonated nitrogen).
Disconnected fragments, radicals, metals/ECPs, unspecified stereo and axial or
atropisomer workflows are outside this release. Specifying labile nitrogen stereo
does not establish configurational persistence. Conformer coverage is limited.

ORCA 6.x must be installed and licensed separately. Before importing ChemGraph,
the operator sets `ASE_ORCA_COMMAND` to one **raw executable path** (spaces are
allowed; no shell quotes, arguments, pipes or launch commands). Otherwise lookup
uses PATH, `/opt/orca/orca`, `/usr/local/orca/orca`, then `~/orca/orca`.
Schema discovery performs no execution; structured calls check the engine identity.
ASE-config-only discovery is not used. Existing NWChem/raw ORCA APIs remain
available. Python-only legacy `OrcaProfile` values are omitted from serialization;
replay resolves operator configuration. Raw ORCA stored defaults are now null,
with unchanged effective B3LYP/def2-TZVP and one-process legacy input.

Use registry discovery first:

```text
search_tools(query="conformers molecular electronic reactivity")
load_tools(names=["prepare_conformers", "run_ase", "calculate_reactivity_descriptors", "extract_output_json"])
prepare_conformers(smiles="CN", candidates=20, retained=5, mmff_max_iters=2000)
```

These artifact writers/readers and calculations wait for the configured review
policy before execution. Rejected calls create no artifacts. Explicit existing
approval bypass settings retain their meaning. These tools and ORCA/electronic
are unavailable in scoped execution; do not fall back to unscoped execution.

Preserve the preparation manifest's charge, singlet multiplicity and atom order.
Use explicit scientific settings for each `run_ase` call, for example:

```json
{
  "params": {
    "input_structure_file": "<manifest conformer XYZ>",
    "output_results_file": "<fresh optimization JSON>",
    "driver": "opt",
    "optimizer": "bfgs",
    "fmax": 0.05,
    "steps": 200,
    "calculator": {
      "calculator_type": "orca",
      "charge": 0,
      "multiplicity": 1,
      "dft_settings": {"functional": "PBE", "basis": "def2-SVP"}
    }
  }
}
```

Functionals: PBE/PBE0/B3LYP/BP86. Bases: STO-3G/def2-SVP/def2-TZVP/def2-TZVPD.
STO-3G is a smoke/regression choice, not a pharma accuracy protocol. Anions without
def2-TZVPD carry a no-diffuse-basis warning; a diffuse basis does not prove binding.
Structured calls forbid raw keywords, blocks, profiles and non-default directories.
Electronic mode generates RKS, TightSCF, ConvForced, NoAutoStart and SP, requesting
the full orbital table. Optimization uses ASE BFGS with EnGrad, not native ORCA Opt.
Native SCF, auxiliary basis and grid sections are preserved because ORCA defaults
can differ by version. Every evaluation has an isolated directory and evidence.

For the complete identity-checked sequence and JSON/CSV exports, run from a source
checkout (or import `run_workflow` from this example):

```bash
python examples/molecular_properties/run.py --smiles CN --functional PBE --basis def2-SVP --output-dir molecular_properties
```

The example prepares → optimizes → checks mapped connectivity/stereo → writes the
exact optimized XYZ → computes electronic results → derives descriptors. It
requires optimization status=success AND converged=true. Only successful,
identity-preserving conformers enter ascending electronic-energy ordering; failed
conformers retain stage/reason/artifacts. MMFF energies are preparation scores,
not final rankings. Exit codes: 0 all retained conformers succeeded, 3 partial,
1 none/preparation failed; argparse usage errors are 2. No automatic retries or
scientific setting changes occur. Each run gets a fresh directory and summary.

Defaults: 3600 seconds per evaluation and 7200 seconds per core call, shared by
probe and all optimization steps. Operator-only `CHEMGRAPH_ORCA_TIMEOUT_S` and
`CHEMGRAPH_ORCA_CALL_TIMEOUT_S` accept positive integers up to 86400. These defaults
are for small/screening molecules; larger jobs may require a larger call budget
or an explicitly chosen smaller basis. This has not been benchmarked. Timeout
kills the process group and preserves diagnostics. Failed structured calls replace
any prior result JSON atomically with `result_type="ase_failure"`; publication
errors are reported explicitly. Never interpret an old result as a new success.

`calculate_reactivity_descriptors(result_file="<electronic JSON>")` reads saved
versioned data without ORCA installed. With energies in eV it reports the orbital
approximation I=-HOMO, A=-LUMO, mu=-(I+A)/2, eta=I-A,
omega=(I+A)^2/[8(I-A)]. It rejects incomplete/unconverged data, nonpositive gaps,
and HOMO >= 0. These are global orbital estimates, not experimental IP/EA,
atom-specific reactivity, solution pKa, free-energy populations or stability proofs.
The 0.05 eV/angstrom optimization threshold is screening-level; no frequency/minimum
or SCF-stability verification is implied. `electronic_structure.scf_converged`
records SCF evidence; top-level `converged` in an energy/electronic result indicates
a successful evaluation, not geometry optimization.

Offline tests use unchanged BSD-licensed cclib ORCA fixtures and explicitly
synthetic execution output. ORCA 6.0 full MO headers are covered; the genuine 6.1
NMR fixture is deliberately rejected. Full-table live 6.1 compatibility is not
established. Live execution is experimental/unverified until opt-in tests pass:

```bash
CHEMGRAPH_TEST_ORCA=1 python -m pytest tests/test_orca_electronic.py -q
```

No executable is installed by this workflow. Remote transport and the two new
molecular tools through MCP are not promised in this release; existing ASE MCP
wrappers continue to delegate the shared core.
