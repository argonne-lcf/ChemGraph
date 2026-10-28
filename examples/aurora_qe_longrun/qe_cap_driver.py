#!/usr/bin/env python
"""Drive ChemGraph's run_ase_core with a real Quantum ESPRESSO calculator to
validate the wall-clock cap -> partial-geometry seam on Aurora.

This is the CALC-LAYER route (Route 1): it calls run_ase_core directly, bypassing
the LLM agent / tool wrapper / resume CLI. Use it to prove the cap+resume seam on
real DFT in isolation before running the full agent stack (qe_agent_e2e.py).

Stages, selected by argv[1]:

  Bulk Si (fully periodic -> keeps the configured Monkhorst-Pack mesh):
  smoke   -- single-point SCF on bulk Si (no cap). Proves pw.x + pseudo + MPI
             actually produce an energy through the ChemGraph Espresso seam.
  cap     -- geometry opt on a rattled Si cell with an allocation deadline set
             a few SCF steps out. Proves the soft cap fires at an optimizer-step
             boundary and leaves a resumable partial (.traj + restart + xyz).
  resume  -- re-run opt from the partial geometry written by the cap stage,
             uncapped, and confirm it makes progress / converges.

  H2O molecule (non-periodic -> exercises the NEW DFT code path added in this
  branch: is_nonperiodic -> atoms.center(vacuum) -> K_POINTS gamma). Bulk Si
  never touches that path, so these stages are what actually validate the fix
  on real pw.x:
  mol_smoke  -- single-point SCF on an isolated H2O. Proves the molecule is
                centered into a finite box and pw.x runs at the Gamma point
                (grep the generated espresso.pwi for "K_POINTS gamma").
  mol_cap    -- geometry opt on a rattled H2O with a short allocation deadline;
                proves the cap fires at an optimizer-step boundary on the
                molecule path and leaves a resumable partial.
  mol_resume -- resume the H2O opt uncapped and confirm it relaxes into the
                cross-code plane-wave-PBE water band (0.96-1.00 Ang, 102-106 deg).

All output goes under $RUN_DIR (default: cwd). Structures are built in-process
with ASE so no external structure file is needed for the first run.
"""
import json
import math
import os
import sys
import time

import numpy as np
from ase.build import bulk, molecule
from ase.io import write

STAGE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
RUN_DIR = os.environ.get("RUN_DIR", os.getcwd())
os.chdir(RUN_DIR)

# run_ase_core resolves relative paths against CHEMGRAPH_LOG_DIR when set; keep
# every artifact for this run together so the manifest/resume logic has one home.
os.environ.setdefault("CHEMGRAPH_LOG_DIR", RUN_DIR)

from chemgraph.tools.ase_core import run_ase_core  # noqa: E402
from chemgraph.schemas.ase_input import ASEInputSchema  # noqa: E402
from chemgraph.schemas.ase_input import get_available_calculator_names  # noqa: E402


def _banner(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


_banner(f"STAGE={STAGE}  RUN_DIR={RUN_DIR}")
print("available calculators:", get_available_calculator_names(), flush=True)
assert "EspressoCalc" in get_available_calculator_names(), (
    "EspressoCalc not available -- check pw.x on PATH / ASE_ESPRESSO_COMMAND and "
    "ESPRESSO_PSEUDO are exported BEFORE importing chemgraph."
)

# A small, cheap cell: 2-atom Si diamond primitive. Low cutoff + coarse k-mesh
# keep one SCF to a few seconds so the cap has clean step boundaries to fire on.
CALC = {
    "calculator_type": "espresso",
    "pseudopotentials": {"Si": "Si.UPF"},
    "pseudo_dir": os.environ["ESPRESSO_PSEUDO"],
    "ecutwfc": 25.0,
    "kpts": [2, 2, 2],
    "xc": "PBE",
    "input_data": {
        # keep each SCF short and robust for a validation run
        "conv_thr": 1e-6,
        "mixing_beta": 0.3,
        "electron_maxstep": 80,
    },
}


def build_si(rattle=0.0, path="si.xyz"):
    atoms = bulk("Si", "diamond", a=5.43)
    if rattle:
        atoms.rattle(stdev=rattle, seed=1)
    write(path, atoms)
    return os.path.abspath(path)


# H2O molecule config. Non-periodic (pbc=[F,F,F], cell.rank==0), so this is what
# drives the new is_nonperiodic -> center(vacuum) -> K_POINTS gamma path. kpts is
# left at the schema default on purpose: the molecule branch discards it and emits
# a single Gamma point, so keeping a stray mesh here exercises the fix
# (the branch must drop it). Cutoffs suit the pslibrary USPP PBE H/O pseudos (RRKJUS is soft, but
# O needs a healthy density cutoff; ecutrho=8*ecutwfc for ultrasoft).
MOL_CALC = {
    "calculator_type": "espresso",
    "pseudopotentials": {
        "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
        "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF",
    },
    "pseudo_dir": os.environ["ESPRESSO_PSEUDO"],
    "ecutwfc": 50.0,
    "ecutrho": 400.0,  # 8*ecutwfc -- ultrasoft O needs a dense charge grid
    "xc": "PBE",
    "vacuum": 6.0,  # padding added around the molecule by ase_core before the run
    "input_data": {
        "conv_thr": 1e-7,
        "mixing_beta": 0.3,
        "electron_maxstep": 120,
        # An isolated neutral molecule at Gamma: no smearing (fixed occupations).
    },
}


def build_h2o(rattle=0.0, path="h2o.xyz"):
    """ASE's built-in H2O: O,H,H, cell-less, pbc all False -> the molecule path.

    Writing it to .xyz and reading it back through run_ase_core is what the real
    agent does, and it is where the cell-less structure originates -- ase_core
    then centers it into a vacuum box before handing it to pw.x.
    """
    atoms = molecule("H2O")
    if rattle:
        atoms.rattle(stdev=rattle, seed=1)
    write(path, atoms)
    return os.path.abspath(path)


def water_geometry(numbers, positions):
    """Return (r1, r2, angle_deg) for an O,H,H structure (any atom order)."""
    numbers = list(numbers)
    o_idx = numbers.index(8)
    h_idx = [i for i, z in enumerate(numbers) if z == 1]
    pos = np.asarray(positions, dtype=float)
    o = pos[o_idx]
    v1 = pos[h_idx[0]] - o
    v2 = pos[h_idx[1]] - o
    r1 = float(np.linalg.norm(v1))
    r2 = float(np.linalg.norm(v2))
    angle = math.degrees(math.acos(np.dot(v1, v2) / (r1 * r2)))
    return r1, r2, angle


def _assert_gamma_in_pwi(run_dir):
    """Grep the pw.x input ase_core wrote for proof of the molecule path.

    The new code path is only truly validated if the *written* input pw.x
    consumed carried K_POINTS gamma and a non-zero cell. A parameter-only check
    can pass while the writer still emits a spurious mesh or a zero cell.
    """
    hits = []
    for root, _dirs, files in os.walk(run_dir):
        for fn in files:
            if fn.endswith(".pwi"):
                hits.append(os.path.join(root, fn))
    assert hits, f"no .pwi input found under {run_dir} to verify the gamma path"
    newest = max(hits, key=os.path.getmtime)
    txt = open(newest).read()
    print(f"verifying generated input: {newest}", flush=True)
    assert "K_POINTS gamma" in txt, (
        f"molecule path did NOT emit 'K_POINTS gamma' in {newest}:\n{txt}"
    )
    assert "CELL_PARAMETERS" in txt, (
        f"no CELL_PARAMETERS in {newest} -- centering did not reach the writer"
    )
    # A zero/near-zero cell means centering never happened; guard against it by
    # parsing the CELL_PARAMETERS block itself (the three lattice vectors on the
    # lines right after the header). Scanning for the first 3-float
    # line could match an atomic position. A finite cell has a non-zero
    # volume, so require |det| above a small threshold.
    lines = txt.splitlines()
    header = next(
        i for i, ln in enumerate(lines) if ln.split()[:1] == ["CELL_PARAMETERS"]
    )
    vectors = []
    for ln in lines[header + 1 : header + 4]:
        parts = ln.split()
        assert len(parts) == 3, (
            f"malformed CELL_PARAMETERS row in {newest}: {ln!r}"
        )
        vectors.append([float(p) for p in parts])
    volume = abs(float(np.linalg.det(np.asarray(vectors, dtype=float))))
    assert volume > 1.0, (
        f"cell in {newest} looks degenerate (volume={volume:.3g} Ang^3)"
    )
    print(f"OK: {os.path.basename(newest)} has K_POINTS gamma + a finite cell "
          f"(volume={volume:.1f} Ang^3).", flush=True)


if STAGE == "smoke":
    struct = build_si(rattle=0.0, path="si_smoke.xyz")
    params = ASEInputSchema(
        input_structure_file=struct,
        output_results_file="si_smoke_out.json",
        driver="energy",  # single-point: one SCF, no optimizer, no cap
        calculator=CALC,
    )
    t0 = time.time()
    result = run_ase_core(params)
    dt = time.time() - t0
    _banner("SMOKE RESULT")
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"\none single-point SCF wall_time ~= {dt:.1f}s", flush=True)
    assert result.get("status") == "success", "QE single-point failed"
    assert result.get("single_point_energy") is not None
    print("SMOKE OK: QE produced an energy through the ChemGraph seam.", flush=True)

elif STAGE == "cap":
    rattle = float(os.environ.get("QE_CAP_RATTLE", "0.05"))
    struct = build_si(rattle=rattle, path="si_cap.xyz")
    # Deadline: give it enough for >=1 step but force a cap before convergence.
    # SECONDS is measured from process start; the cap fires at the first
    # step-boundary past the budget. Tune via QE_CAP_SECONDS (default 60s).
    budget = float(os.environ.get("QE_CAP_SECONDS", "60"))
    os.environ["CHEMGRAPH_ALLOCATION_SECONDS"] = str(budget)
    # Margin must exceed one SCF step so the cap never breaches walltime mid-SCF.
    os.environ.setdefault("CHEMGRAPH_ALLOCATION_MARGIN", "30")
    print(
        f"cap budget={budget}s  margin={os.environ['CHEMGRAPH_ALLOCATION_MARGIN']}s",
        flush=True,
    )
    params = ASEInputSchema(
        input_structure_file=struct,
        output_results_file="si_cap_out.json",
        driver="opt",
        optimizer="bfgs",
        calculator=CALC,
        fmax=0.001,  # tight target the short budget can't reach -> guaranteed cap
        steps=100,
    )
    t0 = time.time()
    result = run_ase_core(params)
    dt = time.time() - t0
    _banner("CAP RESULT")
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"\ntotal wall_time ~= {dt:.1f}s", flush=True)
    assert result.get("wall_time_capped") is True, "expected wall_time_capped=True"
    assert result.get("resume_input_file"), "expected a partial geometry for resume"
    assert os.path.isfile(result["resume_input_file"]), "partial xyz not on disk"
    # Hand the partial path to the resume stage.
    with open(os.path.join(RUN_DIR, "partial_path.txt"), "w") as fh:
        fh.write(result["resume_input_file"] + "\n")
    print("CAP OK: soft cap fired at a step boundary; partial saved.", flush=True)

elif STAGE == "resume":
    partial = os.environ.get("QE_RESUME_INPUT")
    if not partial:
        # Fall back to the path the cap stage recorded in this RUN_DIR.
        rec = os.path.join(RUN_DIR, "partial_path.txt")
        if os.path.isfile(rec):
            partial = open(rec).read().strip()
    assert partial and os.path.isfile(partial), (
        "set QE_RESUME_INPUT to the partial xyz written by the cap stage "
        f"(looked for {os.path.join(RUN_DIR, 'partial_path.txt')})"
    )
    # Uncapped this time: let it run to convergence (or the step cap).
    for k in ("CHEMGRAPH_ALLOCATION_SECONDS", "CHEMGRAPH_ALLOCATION_DEADLINE"):
        os.environ.pop(k, None)
    params = ASEInputSchema(
        input_structure_file=partial,
        output_results_file="si_resume_out.json",
        driver="opt",
        optimizer="bfgs",
        calculator=CALC,
        fmax=0.05,
        steps=100,
    )
    t0 = time.time()
    result = run_ase_core(params)
    dt = time.time() - t0
    _banner("RESUME RESULT")
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"\nresume wall_time ~= {dt:.1f}s", flush=True)
    assert result.get("status") == "success"
    assert result.get("wall_time_capped") is not True, "resume unexpectedly capped"
    print("RESUME OK: continued from partial to completion.", flush=True)

elif STAGE == "mol_smoke":
    # Single-point on an isolated H2O -- the minimal proof that the molecule path
    # (center into a box + Gamma point) runs on real pw.x.
    struct = build_h2o(rattle=0.0, path="h2o_smoke.xyz")
    params = ASEInputSchema(
        input_structure_file=struct,
        output_results_file="h2o_smoke_out.json",
        driver="energy",  # single-point: one SCF, no optimizer, no cap
        calculator=MOL_CALC,
    )
    t0 = time.time()
    result = run_ase_core(params)
    dt = time.time() - t0
    _banner("MOL_SMOKE RESULT")
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"\none single-point SCF wall_time ~= {dt:.1f}s", flush=True)
    assert result.get("status") == "success", "QE H2O single-point failed"
    assert result.get("single_point_energy") is not None
    _assert_gamma_in_pwi(RUN_DIR)
    print("MOL_SMOKE OK: isolated H2O ran at Gamma through the molecule path.",
          flush=True)

elif STAGE == "mol_cap":
    # Rattle the water so the optimizer has real work, then cap it a few SCF
    # steps in. Proves the soft cap fires at an optimizer-step boundary on the
    # non-periodic molecule path and leaves a resumable partial.
    rattle = float(os.environ.get("QE_CAP_RATTLE", "0.08"))
    struct = build_h2o(rattle=rattle, path="h2o_cap.xyz")
    budget = float(os.environ.get("QE_CAP_SECONDS", "60"))
    os.environ["CHEMGRAPH_ALLOCATION_SECONDS"] = str(budget)
    os.environ.setdefault("CHEMGRAPH_ALLOCATION_MARGIN", "30")
    print(
        f"cap budget={budget}s  margin={os.environ['CHEMGRAPH_ALLOCATION_MARGIN']}s",
        flush=True,
    )
    params = ASEInputSchema(
        input_structure_file=struct,
        output_results_file="h2o_cap_out.json",
        driver="opt",
        optimizer="bfgs",
        calculator=MOL_CALC,
        fmax=0.01,  # tight enough that the short budget caps before convergence
        steps=100,
    )
    t0 = time.time()
    result = run_ase_core(params)
    dt = time.time() - t0
    _banner("MOL_CAP RESULT")
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"\ntotal wall_time ~= {dt:.1f}s", flush=True)
    _assert_gamma_in_pwi(RUN_DIR)
    assert result.get("wall_time_capped") is True, "expected wall_time_capped=True"
    assert result.get("resume_input_file"), "expected a partial geometry for resume"
    assert os.path.isfile(result["resume_input_file"]), "partial xyz not on disk"
    with open(os.path.join(RUN_DIR, "mol_partial_path.txt"), "w") as fh:
        fh.write(result["resume_input_file"] + "\n")
    print("MOL_CAP OK: cap fired on the molecule path; partial saved.", flush=True)

elif STAGE == "mol_resume":
    partial = os.environ.get("QE_RESUME_INPUT")
    if not partial:
        rec = os.path.join(RUN_DIR, "mol_partial_path.txt")
        if os.path.isfile(rec):
            partial = open(rec).read().strip()
    assert partial and os.path.isfile(partial), (
        "set QE_RESUME_INPUT to the partial xyz written by the mol_cap stage "
        f"(looked for {os.path.join(RUN_DIR, 'mol_partial_path.txt')})"
    )
    for k in ("CHEMGRAPH_ALLOCATION_SECONDS", "CHEMGRAPH_ALLOCATION_DEADLINE"):
        os.environ.pop(k, None)
    params = ASEInputSchema(
        input_structure_file=partial,
        output_results_file="h2o_resume_out.json",
        driver="opt",
        optimizer="bfgs",
        calculator=MOL_CALC,
        fmax=0.03,
        steps=200,
    )
    t0 = time.time()
    result = run_ase_core(params)
    dt = time.time() - t0
    _banner("MOL_RESUME RESULT")
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"\nresume wall_time ~= {dt:.1f}s", flush=True)
    assert result.get("status") == "success"
    assert result.get("wall_time_capped") is not True, "resume unexpectedly capped"
    # Physical class-C check: a correct plane-wave-PBE water optimization lands in
    # this cross-code band. A broken run (non-gamma mesh / zero cell) blows the
    # geometry up and falls outside it.
    out = json.load(open(os.path.join(RUN_DIR, "h2o_resume_out.json")))
    fs = out["final_structure"]
    r1, r2, angle = water_geometry(fs["numbers"], fs["positions"])
    _banner("MOL_RESUME GEOMETRY")
    print(f"r(O-H) = {r1:.4f}, {r2:.4f} Ang", flush=True)
    print(f"angle(H-O-H) = {angle:.2f} deg", flush=True)
    print("reference band: 0.96-1.00 Ang, 102-106 deg "
          "(JDFTx PBE ~0.98 Ang/103.7 deg; expt 0.9572 Ang/104.52 deg)",
          flush=True)
    assert 0.96 <= r1 <= 1.00 and 0.96 <= r2 <= 1.00, (
        f"O-H bond {r1:.4f}/{r2:.4f} Ang outside plane-wave-PBE band [0.96,1.00]"
    )
    assert 102.0 <= angle <= 106.0, (
        f"H-O-H angle {angle:.2f} deg outside plane-wave-PBE band [102,106]"
    )
    print("MOL_RESUME OK: resumed to a water geometry inside the cross-code "
          "plane-wave-PBE band.",
          flush=True)

else:
    sys.exit(f"unknown stage: {STAGE}")
