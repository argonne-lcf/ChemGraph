"""Run ChemGraph's plane-wave DFT tools (run_qe / run_vasp) via an LLM.

This drives the calculator-pinned tools ``run_qe`` (Quantum ESPRESSO / pw.x)
and ``run_vasp`` (VASP) end to end: an Argo-hosted LLM reads a natural-language
prompt, picks the plane-wave tool, fills in a physically sensible input, and the
tool launches the real DFT engine through ASE.

Only the engines detected on this host are offered to the model (same gate as
ChemGraph's calculator union), so on a machine with pw.x + pseudopotentials but
no VASP binary, the QE prompt runs for real and the VASP prompt is skipped with
a clear message. See README.md for the full end-to-end setup (DFT binaries,
pseudopotentials, and the Argo tunnel).

Run:
    python examples/plane_wave_tools/run_plane_wave.py          # QE prompt
    python examples/plane_wave_tools/run_plane_wave.py --engine vasp

Override defaults via env vars (same names as connecting_to_argo/test_run.py):
    ARGO_USER   -- your CELS login (required), e.g. jane.doe
    ARGO_MODEL  -- argo model name (default: argo:gpt-4.1-mini)
    ARGO_BASE   -- shim URL (default: http://127.0.0.1:18085/argoapi/v1)
    QE_PROMPT   -- override the Quantum ESPRESSO prompt
    VASP_PROMPT -- override the VASP prompt
    MP_API_KEY  -- Materials Project key; when set, the bulk Si cell is the
                   DFT-relaxed mp-149 structure in place of an idealized
                   ase.build lattice
    PLANE_WAVE_WORKDIR -- where the structure file is written
                   (default: <cwd>/plane_wave_example)
    QE_SI_PSEUDO -- Si UPF filename inside $ESPRESSO_PSEUDO
                   (default: Si.pbe-n-rrkjus_psl.1.0.0.UPF)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# argo-shim rejects the stripped lowercase-hyphenated model name (e.g.
# "gpt-4.1-mini") with 400 Invalid model. Tell ChemGraph's normalizer to send
# the wire form (e.g. "gpt41mini"), which the shim accepts. Set BEFORE importing
# chemgraph so the normalizer picks it up. (Same shim quirk as the argo example.)
os.environ.setdefault("CHEMGRAPH_ARGO_MODEL_FORMAT", "wire")


# Bulk Si is the canonical periodic plane-wave smoke test: 2 atoms, a real
# k-mesh, converges in seconds.
#
# The prompt has to name a structure FILE rather than just say "bulk silicon".
# No tool in the single_agent set can build a crystal. The only structure
# source is smiles_to_coordinate_file, which is RDKit-backed and returns an
# isolated molecule with no cell. Asked for "bulk silicon" with no file, the
# model reaches for it, gets one lone Si atom at pbc=[F,F,F], and the run is a
# vacuum-box atom rather than a crystal. VASP rejects that outright; pw.x
# accepts it and quietly reports the energy of the wrong system. So the example
# supplies the cell itself. The LLM still chooses the tool and fills in every
# plane-wave parameter.
# 325 eV is 1.3 * the Si PAW POTCAR's ENMAX (245.3 eV), rounded up to the
# nearest 25. The 1.3 margin is VASP's guidance for variable-cell relaxation;
# applying it to these fixed-cell runs too keeps one stated rule across the
# example and leaves every element well above its own ENMAX. A different
# element needs its own value: oxygen's ENMAX alone is 400 eV.
_DEFAULT_PROMPTS = {
    "qe": (
        "Relax the geometry of the periodic crystal in the file '{structure}' "
        "with Quantum ESPRESSO. Call run_qe with "
        "input_structure_file='{structure}', a 4x4x4 k-point mesh, a 40 Ry "
        "wavefunction cutoff and pseudopotentials={{'Si': '{si_pseudo}'}}, "
        "then report the final total energy."
    ),
    "vasp": (
        "Compute the total energy of the periodic crystal in the file "
        "'{structure}' with VASP. Call run_vasp with "
        "input_structure_file='{structure}', a 4x4x4 k-point mesh and a "
        "325 eV plane-wave cutoff."
    ),
}


# pw.x needs an explicit per-element UPF filename (ASE raises KeyError on an
# empty pseudopotentials map) and the name depends on which library is installed
# under $ESPRESSO_PSEUDO. Default to the PSLibrary PBE name; override if yours
# differs. VASP needs no equivalent: ASE derives the POTCAR path from
# $VASP_PP_PATH and the element symbol.
_DEFAULT_SI_PSEUDO = "Si.pbe-n-rrkjus_psl.1.0.0.UPF"

# Materials Project entry for diamond silicon.
_MP_SILICON = "mp-149"


def _bulk_si_from_mp(api_key: str):
    """Fetch the mp-149 silicon cell from Materials Project, or None on failure.

    Prefers a DFT-relaxed structure over an idealized textbook lattice, so the
    example exercises the same kind of input a user would really calculate on.
    Returns None on any failure (missing mp-api package, bad key, no network,
    empty result) in place of raising, so an unreachable structure source
    leaves the demo runnable.

    Parameters
    ----------
    api_key : str
        Materials Project API key.

    Returns
    -------
    ase.Atoms or None
        The relaxed conventional cell, or None if it could not be fetched.
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.io.ase import AseAtomsAdaptor

        with MPRester(api_key) as mpr:
            doc = mpr.materials.summary.search(
                material_ids=[_MP_SILICON], fields=["material_id", "structure"]
            )
            if not doc:
                return None
            atoms = AseAtomsAdaptor.get_atoms(doc[0].structure)
            atoms.info["MP-id"] = str(doc[0].material_id)
            return atoms
    except Exception as exc:  # noqa: BLE001 - any failure falls back
        print(f"Materials Project fetch failed ({exc}); using ase.build.bulk.")
        return None


def _write_bulk_si(directory: str) -> tuple[str, str]:
    """Write a bulk-Si cell and return ``(path, provenance)``.

    Uses the Materials Project structure when ``MP_API_KEY`` is set and
    reachable, otherwise an idealized diamond lattice from ``ase.build``. Either
    way the result is a genuine periodic rank-3 cell, which is the property the
    plane-wave run depends on. Overwritten on every run so a stale file cannot
    silently change the result.

    Parameters
    ----------
    directory : str
        Directory to write ``si_bulk.xyz`` into; created if missing.

    Returns
    -------
    tuple of (str, str)
        Absolute path to the structure file, and a human-readable description
        of where the cell came from.
    """
    from ase.build import bulk
    from ase.io import write

    atoms = None
    provenance = ""
    api_key = os.environ.get("MP_API_KEY")
    if api_key:
        atoms = _bulk_si_from_mp(api_key)
        if atoms is not None:
            provenance = f"Materials Project {_MP_SILICON}, {len(atoms)} atoms"
    if atoms is None:
        atoms = bulk("Si", "diamond", a=5.43)
        provenance = f"ase.build.bulk diamond a=5.43 A, {len(atoms)} atoms"
        if not api_key:
            provenance += " (set MP_API_KEY for the Materials Project cell)"

    os.makedirs(directory, exist_ok=True)
    path = os.path.abspath(os.path.join(directory, "si_bulk.xyz"))
    # extxyz keeps Lattice= and pbc="T T T", so ASE reads it back as a periodic
    # rank-3 cell, which is what makes this a bulk plane-wave run.
    write(path, atoms)
    return path, provenance


def _engine_available(engine: str) -> bool:
    """Return whether the pinned calculator for ``engine`` is registered here."""
    from chemgraph.schemas.ase_input import get_available_calculator_names

    names = get_available_calculator_names()
    return {"qe": "EspressoCalc", "vasp": "VaspCalc"}[engine] in names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=["qe", "vasp"],
        default="qe",
        help="Which plane-wave engine to prompt for (default: qe).",
    )
    args = parser.parse_args()
    engine = args.engine

    argo_user = os.environ.get("ARGO_USER")
    if not argo_user:
        print(
            "ERROR: set ARGO_USER to your CELS login "
            "(e.g. `export ARGO_USER=jane.doe`).",
            file=sys.stderr,
        )
        return 1

    if not _engine_available(engine):
        need = {
            "qe": "pw.x on PATH (or ASE_ESPRESSO_COMMAND) AND ESPRESSO_PSEUDO",
            "vasp": "a VASP binary (or VASP_COMMAND) AND VASP_PP_PATH",
        }[engine]
        print(
            f"ERROR: the {engine.upper()} engine is not registered on this host.\n"
            f"       run_{engine} is only offered when {need} is set.\n"
            "       See README.md for the setup. QE is runnable on Aurora; VASP\n"
            "       needs a licensed binary this machine may not have.",
            file=sys.stderr,
        )
        return 2

    model = os.environ.get("ARGO_MODEL", "argo:gpt-4.1-mini")
    base_url = os.environ.get("ARGO_BASE", "http://127.0.0.1:18085/argoapi/v1")

    # Written after the engine gate so an unsupported host leaves no stray dir.
    workdir = os.environ.get(
        "PLANE_WAVE_WORKDIR", os.path.join(os.getcwd(), "plane_wave_example")
    )
    structure, provenance = _write_bulk_si(workdir)
    # `or` rather than a two-arg get: .format() must not run on a user-supplied
    # prompt, whose literal braces would blow up or be silently substituted.
    prompt = os.environ.get(f"{engine.upper()}_PROMPT") or _DEFAULT_PROMPTS[
        engine
    ].format(
        structure=structure,
        si_pseudo=os.environ.get("QE_SI_PSEUDO", _DEFAULT_SI_PSEUDO),
    )

    print(f"Engine:    {engine}")
    print(f"Structure: {structure}")
    print(f"           ({provenance})")
    print(f"Model:     {model}")
    print(f"Base URL:  {base_url}")
    print(f"Argo user: {argo_user}")
    print(f"Prompt:    {prompt}")
    print()

    from chemgraph.agent.llm_agent import ChemGraph

    # run_qe / run_vasp are in the single_agent default tool set, gated on the
    # engines detected above, so a plain single_agent already exposes them.
    cg = ChemGraph(
        model_name=model,
        workflow_type="single_agent",
        base_url=base_url,
        api_key="dummy",  # argo-shim doesn't check
        argo_user=argo_user,
    )
    result = asyncio.run(cg.run(prompt))
    print()
    print("=" * 60)
    print("ChemGraph response:")
    print("=" * 60)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
