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


# A realistic default prompt per engine. Bulk Si is the canonical periodic
# plane-wave smoke test: a few atoms, a real k-mesh, converges fast.
_DEFAULT_PROMPTS = {
    "qe": (
        "Relax the geometry of bulk silicon with Quantum ESPRESSO using a "
        "4x4x4 k-point mesh and a 40 Ry wavefunction cutoff, then report the "
        "final total energy."
    ),
    "vasp": (
        "Compute the total energy of bulk silicon with VASP using a 4x4x4 "
        "k-point mesh and a 400 eV plane-wave cutoff."
    ),
}


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
    prompt = os.environ.get(
        f"{engine.upper()}_PROMPT", _DEFAULT_PROMPTS[engine]
    )

    print(f"Engine:    {engine}")
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
