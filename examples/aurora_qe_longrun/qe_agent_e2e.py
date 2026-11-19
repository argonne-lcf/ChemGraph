#!/usr/bin/env python
"""Route 2: full ChemGraph AGENT end-to-end validation of the cap->manifest->
resume seam on real Quantum ESPRESSO DFT.

Unlike qe_cap_driver.py (which calls run_ase_core directly), this drives the
WHOLE stack: real LLM -> LangGraph single_agent -> run_ase tool (ToolNode,
JSON-serialized return) -> run_ase_core -> QE, plus the manifest hook that
records the capped step and the `chemgraph resume` log_dir adoption. It exercises
exactly the layers the cap commits touched (C2 arg-unwrap, Seam1 JSON parse,
C3 id-correlation, M2 clear-on-success, M3 log_dir adoption) which the calc-layer
route (run_ase_core direct) bypasses.

The physical system is selected by the QE_SYSTEM env var:

  si   (default) -- a rattled 2-atom Si diamond cell, fully periodic (pbc all
                    True). The original agent-seam check; the configured k-mesh is
                    used verbatim.
  h2o            -- an ASE molecule("H2O"), non-periodic (pbc all False, cell-less).
                    This routes the WHOLE agent stack through the branch's molecule
                    code path (is_nonperiodic -> atoms.center(vacuum) -> K_POINTS
                    gamma), so it proves that path drives real pw.x end to end
                    through the LLM/tool/manifest/resume layers as well, extending
                    the calc-layer coverage in qe_cap_driver.py. For h2o, both
                    stages also grep the generated espresso.pwi for K_POINTS
                    gamma plus a CELL_PARAMETERS block (a presence check; the
                    finite-volume cell check is done by the Route 1 driver).

Two stages, selected by argv[1]:

  run1    -- LLM optimizes the rattled system with QE under a wall-clock cap set
             so the opt is capped mid-flight. Verify the manifest (in the agent's
             OWN auto-generated log_dir) records status='capped', a step with
             status='capped', and a PENDING NEXT STEP whose input_structure_file
             is the readable *_opt.partial.xyz. The session_id + its log_dir are
             printed and the session is persisted to ~/.chemgraph/sessions.db.
  resume  -- `chemgraph resume`-equivalent: a FRESH agent (new uuid, new default
             log_dir) with resume_from=<sid> and NO cap. Verify M3 adopts the
             prior session's log_dir from the DB, the manifest is re-pointed at
             the prior file, the LLM continues from the partial to convergence,
             and M2 clears pending / resets status.

CAP TIMING (the subtle part): the env cap is measured from module import, but the
agent path adds a variable LLM round-trip before the tool runs. A fixed small
SECONDS budget would be spent by the LLM latency and cap at step 0 (no partial).
So run1 sets an ABSOLUTE CHEMGRAPH_ALLOCATION_DEADLINE *right before* cg.run(),
after the LLM+agent are built, and uses an unreachable fmax so the opt can never
converge inside the window -- the cap then fires at a step boundary regardless of
how long the LLM took. NOTE: this makes CHEMGRAPH_ALLOCATION_* effective per the
env read at tool-exec time; run_ase_core reads them live, so setting them after
import is fine (they are not cached at import).

M3 (log_dir adoption) is only genuinely tested if the resume stage does NOT
pre-set CHEMGRAPH_LOG_DIR to the run1 dir -- otherwise the fresh agent would land
in the same dir for the trivial reason. So the resume stage explicitly UNSETS it
and relies on the DB lookup to bring the agent home.
"""
import asyncio
import json
import os
import sys
import time

STAGE = sys.argv[1] if len(sys.argv) > 1 else "run1"
WORK = os.environ["WORK_DIR"]  # parent dir that holds session_id.txt across stages
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

# LLM auth: load the ALCF token into the env var ChemGraph reads. Never echo it.
os.environ["ALCF_ACCESS_TOKEN"] = open(os.path.expanduser("~/.alcf_token")).read().strip()

MODEL = os.environ.get("QE_LLM_MODEL", "openai/gpt-oss-120b")
SESSION_FILE = os.path.join(WORK, "session_id.txt")

# Which physical system the agent drives. "si" keeps the original periodic seam
# check; "h2o" routes the same agent stack through the non-periodic Gamma path.
SYSTEM = os.environ.get("QE_SYSTEM", "si").lower()
STRUCT = os.path.join(WORK, f"{SYSTEM}_agent.xyz")

# Effective-window controls (seconds). Budget must comfortably exceed the LLM
# round-trip + a couple SCF steps; the unreachable fmax guarantees the cap, not
# convergence, ends run1. Margin must exceed one SCF step (~2s here).
# window is deliberately generous: effective budget = window - margin - T_llm,
# and T_llm (LLM round-trip, incl. any self-correct retries) is variable. An
# unreachable fmax means a larger window only runs a few more steps before the
# cap (the opt can never converge before the cap fires), so we size the
# window so even a slow LLM leaves >= 1 completed SCF step (a real partial).
CAP_WINDOW = float(os.environ.get("QE_CAP_WINDOW", "40"))   # deadline = now + this
CAP_MARGIN = float(os.environ.get("QE_CAP_MARGIN", "8"))    # > one SCF step

# The calculator config the LLM must marshal into run_ase(params=...). Given
# verbatim in the prompt so the model drives the calc and does not invent
# chemistry. One config per system; the h2o config carries no kpts, so the
# non-periodic branch emits K_POINTS gamma, and a vacuum for the plane-wave box.
CALC_BY_SYSTEM = {
    "si": {
        "calculator_type": "espresso",
        "pseudopotentials": {"Si": "Si.UPF"},
        "pseudo_dir": os.environ["ESPRESSO_PSEUDO"],
        "ecutwfc": 25.0,
        "kpts": [2, 2, 2],
        "xc": "PBE",
        "input_data": {
            "conv_thr": 1e-6, "mixing_beta": 0.3, "electron_maxstep": 80,
        },
    },
    "h2o": {
        "calculator_type": "espresso",
        "pseudopotentials": {
            "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
            "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF",
        },
        "pseudo_dir": os.environ["ESPRESSO_PSEUDO"],
        "ecutwfc": 50.0,
        "ecutrho": 400.0,  # 8*ecutwfc; ultrasoft O needs a dense charge grid
        "xc": "PBE",
        "vacuum": 6.0,
        "input_data": {
            "conv_thr": 1e-7, "mixing_beta": 0.3, "electron_maxstep": 120,
        },
    },
}
if SYSTEM not in CALC_BY_SYSTEM:
    sys.exit(f"unknown QE_SYSTEM={SYSTEM!r}; expected one of {list(CALC_BY_SYSTEM)}")
CALC = CALC_BY_SYSTEM[SYSTEM]
CALC_JSON = json.dumps(CALC)

# Per-system nouns for the prompts: Si is a periodic "crystal structure", H2O is
# an isolated "molecule". Keeps each prompt physically honest.
NOUN = "molecule" if SYSTEM == "h2o" else "crystal structure"


def _banner(m):
    print(f"\n{'=' * 70}\n{m}\n{'=' * 70}", flush=True)


def build_struct():
    """Write the starting geometry for the selected system.

    Si: rattled diamond cell (periodic). H2O: an ASE molecule, left cell-less and
    non-periodic so the agent stack exercises the center(vacuum) -> Gamma path.
    """
    from ase.build import bulk, molecule
    from ase.io import write
    if SYSTEM == "h2o":
        atoms = molecule("H2O")
        atoms.rattle(stdev=0.08, seed=1)  # perturb so the opt has real work to do
    else:
        atoms = bulk("Si", "diamond", a=5.43)
        atoms.rattle(stdev=0.05, seed=1)
    write(STRUCT, atoms)
    return STRUCT


def water_geometry(numbers, positions):
    """Return (r1, r2, angle_deg) for the O,H,H atoms, for the h2o band check."""
    import numpy as np
    numbers = list(numbers)
    positions = np.asarray(positions, dtype=float)
    o = numbers.index(8)
    hs = [i for i, z in enumerate(numbers) if z == 1]
    v1, v2 = positions[hs[0]] - positions[o], positions[hs[1]] - positions[o]
    r1, r2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    cos = float(np.dot(v1, v2) / (r1 * r2))
    ang = float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))
    return r1, r2, ang


def _newest_matching(pattern, *roots):
    """Newest file matching pattern under any of roots (recursive), or None.

    The agent path splits its outputs across two dirs: the BFGS .traj and manifest
    follow CHEMGRAPH_LOG_DIR (the adopted log_dir), while EspressoCalc writes its
    .pwi into its own `directory` field, which defaults to '.' (the process cwd =
    WORK). Searching both roots finds the file wherever it landed.
    """
    import glob
    hits = []
    for root in roots:
        if root and os.path.isdir(root):
            hits += glob.glob(os.path.join(root, "**", pattern), recursive=True)
    hits = sorted(set(hits), key=os.path.getmtime)
    return hits[-1] if hits else None


def assert_gamma_pwi(log_dir, tag):
    """For h2o only: the newest espresso.pwi must carry K_POINTS gamma plus a
    CELL_PARAMETERS block (a substring presence check). This confirms the agent
    path drove the non-periodic centering/Gamma code and ruled out a stray
    k-mesh. The rigorous non-degenerate/finite-volume cell check is done by the
    Route 1 driver (qe_cap_driver.py)."""
    if SYSTEM != "h2o":
        return
    pwi = _newest_matching("*.pwi", log_dir, WORK)
    assert pwi, f"{tag}: no .pwi written under {log_dir} or {WORK}"
    low = open(pwi).read().lower()
    assert "k_points gamma" in low, (
        f"{tag}: espresso.pwi missing 'K_POINTS gamma' (non-periodic path did not "
        f"drive the writer): {pwi}"
    )
    assert "cell_parameters" in low, f"{tag}: .pwi has no CELL_PARAMETERS: {pwi}"
    print(f"{tag}: gamma+cell-block OK in {os.path.basename(pwi)}", flush=True)


def show_manifest(log_dir, tag):
    path = os.path.join(log_dir, "run_manifest.json")
    _banner(f"MANIFEST ({tag}) @ {path}")
    if not os.path.isfile(path):
        print("  <no manifest file>")
        return None
    data = json.load(open(path))
    print(json.dumps(data, indent=2, default=str)[:3000], flush=True)
    return data


# Import chemgraph AFTER env setup (token, pseudo). Allocation env is set later,
# per-stage, and read live by run_ase_core at tool-exec time.
from chemgraph.agent.llm_agent import ChemGraph  # noqa: E402


if STAGE == "run1":
    build_struct()
    # Hardening (see the resume stage and README): handing the LLM a long ABSOLUTE
    # path invites two failures we actually observed here: the model drops the
    # required single `params` wrapper AND abbreviates the path (e.g. to
    # "/home/rez.../h2o_agent.xyz"), which then fails to resolve. Give it just the
    # basename: build_struct writes the file into WORK, the driver's cwd, so
    # run_ase's `_resolve_existing_path` finds a bare name against cwd. Also spell
    # out the single-`params`-argument contract explicitly.
    struct_name = os.path.basename(STRUCT)
    prompt = (
        f"Optimize the geometry of the {NOUN} saved in the file named "
        f"'{struct_name}' using Quantum ESPRESSO. Call the run_ase tool exactly "
        f"ONCE. The run_ase tool takes a SINGLE argument named 'params' whose "
        f"value is an object; put ALL of the following inside params: "
        f"driver='opt', optimizer='bfgs', fmax=0.0001, steps=200, "
        f"input_structure_file='{struct_name}' (use exactly that filename, do not "
        f"add any directory path and do not alter a single character), and this "
        f"exact calculator configuration: {CALC_JSON}. Then report the final "
        f"energy."
    )
    cg = ChemGraph(
        model_name=MODEL, workflow_type="single_agent",
        return_option="last_message", recursion_limit=25,
    )
    sid = cg.session_id
    log_dir = cg.log_dir
    with open(SESSION_FILE, "w") as fh:
        fh.write(sid + "\n")
    print(f"session_id={sid}\nlog_dir={log_dir}", flush=True)

    # Set the ABSOLUTE deadline now, after the agent is built and just before the
    # run, so the LLM round-trip does not eat the budget (see module docstring).
    os.environ["CHEMGRAPH_ALLOCATION_DEADLINE"] = str(time.time() + CAP_WINDOW)
    os.environ["CHEMGRAPH_ALLOCATION_MARGIN"] = str(CAP_MARGIN)
    os.environ.pop("CHEMGRAPH_ALLOCATION_SECONDS", None)
    print(f"cap: deadline=now+{CAP_WINDOW}s  margin={CAP_MARGIN}s", flush=True)

    t0 = time.time()
    out = asyncio.run(cg.run(prompt))
    _banner(f"RUN1 FINAL MESSAGE (agent wall={time.time() - t0:.1f}s)")
    print(str(getattr(out, "content", out))[:1500], flush=True)

    data = show_manifest(log_dir, "after run1")
    assert data is not None, "no manifest written -- did the LLM call run_ase?"
    status = data.get("status")
    pending = data.get("pending_next_step")
    steps = data.get("steps", [])
    capped = [s for s in steps if s.get("status") == "capped"]
    print(
        f"\nstatus={status}  pending={bool(pending)}  n_steps={len(steps)}  "
        f"capped_steps={len(capped)}",
        flush=True,
    )
    assert status == "capped", f"expected manifest status 'capped', got {status!r}"
    assert capped, "expected >=1 step with status='capped'"
    assert pending, "expected a PENDING NEXT STEP block"
    pin = (pending.get("args") or {}).get("input_structure_file", "")
    print(f"pending input_structure_file={pin}", flush=True)
    assert pin.endswith("_opt.partial.xyz"), (
        f"pending input should be the partial geometry, got {pin!r}"
    )
    assert os.path.isfile(pin), f"partial geometry not on disk: {pin}"
    # C2 check: the recorded step args must be UNWRAPPED (driver/calc visible,
    # not buried under a 'params' key).
    s0 = capped[0].get("args", {})
    assert s0.get("driver") == "opt", f"C2 unwrap failed: step args={s0}"
    # For h2o: the agent path must have driven the non-periodic Gamma/centering
    # code, so the .pwi it generated must carry K_POINTS gamma + a finite cell.
    assert_gamma_pwi(log_dir, "run1")
    print("RUN1 OK: agent recorded a capped step + pending partial-geometry.",
          flush=True)

elif STAGE == "resume":
    sid = open(SESSION_FILE).read().strip()
    print(f"resuming session_id={sid}", flush=True)
    # M3 is only tested if we do NOT pre-point the fresh agent at the run1 dir.
    os.environ.pop("CHEMGRAPH_LOG_DIR", None)
    # No cap this time -> let it converge. Clear any leftover deadline too.
    for k in ("CHEMGRAPH_ALLOCATION_SECONDS", "CHEMGRAPH_ALLOCATION_DEADLINE"):
        os.environ.pop(k, None)

    # Look up run1's log_dir from the DB (sid -> log_dir), read its manifest, and
    # pull the partial-geometry path the cap left behind. This both pre-checks
    # that run1 really left a resumable pending step and lets us build a concrete,
    # non-flaky resume prompt (render_for_context injects driver+input but NOT the
    # calculator config, so a bare "continue" would leave the LLM without the QE
    # settings and the resume tool call would fail -- never exercising M2).
    from chemgraph.memory.store import SessionStore  # noqa: E402
    prior_sess = SessionStore().get_session(sid)
    prior_log_dir = getattr(prior_sess, "log_dir", None)
    assert prior_log_dir and os.path.isdir(prior_log_dir), (
        f"cannot find run1 log_dir for session {sid}: {prior_log_dir!r}"
    )
    prior_manifest = json.load(
        open(os.path.join(prior_log_dir, "run_manifest.json"))
    )
    assert prior_manifest.get("status") == "capped", (
        "run1 manifest is not 'capped' -- run the run1 stage first"
    )
    partial = (prior_manifest.get("pending_next_step") or {}).get("args", {}).get(
        "input_structure_file", ""
    )
    assert partial.endswith("_opt.partial.xyz") and os.path.isfile(partial), (
        f"run1 left no readable partial geometry: {partial!r}"
    )
    # Hardening (see the failure documented in README): asking the LLM to echo the
    # full absolute partial path made it (a) drop the required `params` wrapper and
    # (b) mistype the path. Both are avoidable:
    #   - The agent adopts run1's log_dir on resume (M3), and run_ase's input goes
    #     through `_resolve_existing_path`, which resolves a BARE filename against
    #     CHEMGRAPH_LOG_DIR. So we hand the model just the basename -- there is no
    #     long path left to mistype, and it still resolves to the partial.
    #   - Spell out the single-`params`-argument tool contract explicitly.
    partial_name = os.path.basename(partial)
    print(f"run1 log_dir={prior_log_dir}\nrun1 partial={partial}"
          f"\nresume input (bare name)={partial_name}", flush=True)

    # Loose fmax: from the near-minimum partial this converges in a couple steps,
    # so the resume completes (uncapped) and M2's clear-on-success fires.
    resume_prompt = (
        f"The previous run was capped mid-optimization; a partial {NOUN} geometry "
        f"was saved to the file named '{partial_name}'. "
        f"Continue the geometry optimization from that partial geometry using "
        f"Quantum ESPRESSO. Call the run_ase tool exactly ONCE. The run_ase tool "
        f"takes a SINGLE argument named 'params' whose value is an object; put ALL "
        f"of the following inside params: driver='opt', optimizer='bfgs', "
        f"fmax=0.05, steps=200, input_structure_file='{partial_name}' (use exactly "
        f"that filename, do not add any directory path and do not alter a single "
        f"character), and this exact calculator configuration: {CALC_JSON}. "
        f"Then report the final energy."
    )

    cg = ChemGraph(
        model_name=MODEL, workflow_type="single_agent",
        return_option="last_message", recursion_limit=25,
    )
    fresh_default = cg.log_dir  # the auto-generated dir before adoption
    print(f"fresh agent default log_dir={fresh_default}", flush=True)

    t0 = time.time()
    out = asyncio.run(cg.run(resume_prompt, resume_from=sid))
    _banner(f"RESUME FINAL MESSAGE (agent wall={time.time() - t0:.1f}s)")
    print(str(getattr(out, "content", out))[:1500], flush=True)

    adopted = cg.log_dir
    print(f"adopted log_dir={adopted}", flush=True)
    assert adopted != fresh_default, (
        "M3 FAILED: agent did not adopt the prior log_dir (still on the fresh "
        f"default {fresh_default})"
    )
    assert os.path.realpath(adopted) == os.path.realpath(prior_log_dir), (
        f"M3 FAILED: adopted {adopted!r} is not run1's log_dir {prior_log_dir!r}"
    )
    # The manifest the agent now writes to must be the prior session's file.
    assert os.path.dirname(str(cg.run_manifest._path)) == adopted, \
        "M3 FAILED: run_manifest not re-pointed at the adopted log_dir"

    data = show_manifest(adopted, "after resume")
    assert data is not None
    status = data.get("status")
    pending = data.get("pending_next_step")
    print(f"\nstatus={status}  pending={bool(pending)}", flush=True)
    assert status in ("running", "done", "completed"), \
        f"M2 FAILED: expected pending cleared / running-or-done, got {status!r}"
    assert not pending, "M2 FAILED: PENDING NEXT STEP not cleared after resume"

    # For h2o: the resumed opt must land in the coarse cross-code plane-wave-PBE
    # water band. Read this as a coarse pass/fail guard against a broken run; a converged
    # accuracy claim is a separate goal (see the README scope note). Also re-confirm the resume
    # tool call itself ran through the Gamma/centering path. run_ase writes the
    # converged geometry as `final_structure` into its output JSON (default
    # output.json, resolved into the adopted log_dir), which is the reliable
    # source: an uncapped opt writes no trajectory, so read the JSON, matching how
    # qe_cap_driver.py's mol_resume stage reads it.
    if SYSTEM == "h2o":
        assert_gamma_pwi(adopted, "resume")
        out_json = _newest_matching("output.json", adopted, WORK)
        assert out_json, f"resume: no output.json written under {adopted} or {WORK}"
        fs = json.load(open(out_json)).get("final_structure")
        assert fs and fs.get("numbers") and fs.get("positions"), (
            f"resume: output.json has no usable final_structure: {out_json}"
        )
        r1, r2, ang = water_geometry(fs["numbers"], fs["positions"])
        print(f"resume geometry: r(O-H)={r1:.4f}/{r2:.4f} A  angle={ang:.2f} deg",
              flush=True)
        assert 0.96 <= r1 <= 1.00 and 0.96 <= r2 <= 1.00, (
            f"O-H length out of band: {r1:.4f}/{r2:.4f} A (expected 0.96-1.00)"
        )
        assert 102.0 <= ang <= 106.0, (
            f"H-O-H angle out of band: {ang:.2f} deg (expected 102-106)"
        )
        print("resume geometry in cross-code plane-wave-PBE water band.", flush=True)

    print("RESUME OK: log_dir adopted (M3), pending cleared (M2), run continued.",
          flush=True)

else:
    sys.exit(f"unknown stage: {STAGE}")
