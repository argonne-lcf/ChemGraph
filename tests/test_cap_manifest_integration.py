"""Cross-seam integration test: ase_core's cap result <-> the manifest parser.

The wall-clock-cap feature spans two files that communicate through the tool
result. In production LangGraph's ``ToolNode`` serializes ``run_ase_core``'s
return dict with ``json.dumps``, so the ToolMessage content the manifest hook
sees is JSON:

  * ``chemgraph.tools.ase_core.run_ase_core`` returns a dict carrying the resume
    contract (``wall_time_capped`` / ``result_file`` / ``restart_file`` /
    ``resume_input_file``).
  * ``chemgraph.agent.llm_agent.ChemGraph._manifest_observe`` parses that JSON and
    reads those structured fields to decide whether a step is "done" or "capped"
    (pending), and what a resume should continue from.

``test_real_capped_json_content_via_toolnode_is_recorded_pending`` pins the
production path: it runs the REAL ``run_ase_core``, serializes the return with
the REAL ``ToolNode`` serializer (``msg_content_output``) -- the exact JSON string
production sees -- and feeds it to the REAL ``_manifest_observe``.
``test_real_capped_message_is_recorded_as_pending`` covers the plain-text
fallback (a ToolMessage whose content is just the message string).
"""

import time

import ase.calculators.emt as emt_mod
import pytest
from ase.cluster import Icosahedron
from ase.io import write
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import msg_content_output

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.tools.ase_core import run_ase_core

# --- fixtures below mirror tests/test_cap.py verbatim (kept local so this cross
# --- seam test is self-contained; see test_cap.py for the rationale comments) --


@pytest.fixture
def cu_cluster(tmp_path, monkeypatch):
    """A rattled 13-atom Cu cluster that needs many EMT opt steps."""
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    atoms = Icosahedron("Cu", noshells=2)  # 13 atoms
    atoms.rattle(0.2, seed=1)  # perturb so optimization has real work to do
    p = tmp_path / "cu.xyz"
    write(str(p), atoms)
    return p


@pytest.fixture
def slow_emt(monkeypatch):
    """Slow every EMT force evaluation so the wall-clock cap trips deterministically.

    Bare EMT steps are sub-millisecond, so a fixed 50 ms budget races the first
    optimizer step and loses under CPU load (0 steps -> no restart file). Making
    each step ~0.1 s means step *timing* dominates the budget, so a fixed number
    of steps run before the cap regardless of machine load.
    """
    orig = emt_mod.EMT.calculate

    def _slow(self, *args, **kwargs):
        time.sleep(0.1)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(emt_mod.EMT, "calculate", _slow)
    return _slow


def _fake_agent(tmp_path):
    """A minimal object exposing the manifest + the real _manifest_observe.

    Mirrors the helper in tests/test_manifest.py: ``_pending_steps`` is the
    tool_call_id-keyed dict of open steps ``_manifest_observe`` reads on the
    ToolMessage branch, standing in for the real ChemGraph __init__ state.
    """
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.memory.manifest import RunManifest

    class _FakeAgent:
        # own manifest at a DIFFERENT path than the ase output so the two files
        # cannot collide.
        run_manifest = RunManifest(tmp_path / "run_manifest.json")
        _pending_steps = {}
        _COST_TOOLS = ChemGraph._COST_TOOLS
        _manifest_observe = ChemGraph._manifest_observe

        def __init__(self):
            self._pending_steps = {}

    return _FakeAgent()


def test_real_capped_json_content_via_toolnode_is_recorded_pending(
    cu_cluster, tmp_path, slow_emt
):
    """The production path: REAL cap result -> REAL ToolNode JSON -> REAL hook.

    run_ase_core caps a slow EMT opt; its return dict is serialized with the exact
    ToolNode serializer production uses (``msg_content_output``), yielding the JSON
    string the manifest hook parses. Asserts the step is recorded 'capped', queued
    as pending, and -- the C1<->Seam1 seam -- that the pending step's input is
    swapped to the standalone partial geometry so a resume continues from it.
    """
    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,  # effectively unreachable -> would run many steps
        steps=100000,
        max_wall_seconds=1.0,  # slow_emt ~0.1 s/step -> a few steps then cap WITH a restart file
    )
    result = run_ase_core(params)

    # sanity: a capped-with-partial run carrying the structured resume contract.
    assert result["wall_time_capped"] is True
    assert result["resume_input_file"].endswith("_opt.partial.xyz")

    # This is the exact string production sees: ToolNode json.dumps of the return.
    content = msg_content_output(result)
    assert isinstance(content, str)

    fa = _fake_agent(tmp_path)

    # 1) the AI message issuing the cost-bearing tool call -> records a step start.
    fa._manifest_observe(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_ase",
                    "id": "c1",
                    "args": {
                        "driver": "opt",
                        "calculator": {"calculator_type": "emt"},
                        "input_structure_file": str(cu_cluster),
                    },
                }
            ],
        )
    )

    # 2) the ToolMessage carries the REAL serialized JSON (not a hand-written
    #    string): this is the seam that was silently broken before the JSON parse.
    fa._manifest_observe(ToolMessage(content=content, tool_call_id="c1"))

    # --- the coupling assertions ------------------------------------------------
    assert fa.run_manifest.status == "capped"

    step = fa.run_manifest._data["steps"][0]
    assert step["status"] == "capped"

    pending = fa.run_manifest._data["pending_next_step"]
    assert pending is not None
    # the C1<->Seam1 seam: resume continues from the partial geometry.
    assert pending["args"]["input_structure_file"].endswith("_opt.partial.xyz")
    assert "partial geometry" in pending["reason"]

    assert fa._pending_steps == {}

    txt = fa.run_manifest.render_for_context()
    assert "PENDING NEXT STEP" in txt
    # a capped step is not in the "do NOT recompute" completed list (render only
    # lists status=="done"): its only surface is PENDING, so the input filename
    # must not appear in the completed portion.
    completed = txt.split("=== PENDING NEXT STEP", 1)[0]
    assert str(cu_cluster) not in completed


def test_real_capped_message_is_recorded_as_pending(cu_cluster, tmp_path, slow_emt):
    """Plain-text fallback: a message-only ToolMessage still records 'capped'.

    Older tools (or a tool whose return is a bare string) yield a ToolMessage
    whose content is NOT JSON. The hook then falls back to scraping the
    ``"saved to <path>"`` substring for the result-JSON path and reading the
    capped flags out of that file on disk. This drives that path with a real
    capped run's on-disk result JSON.
    """
    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,
        steps=100000,
        max_wall_seconds=1.0,  # slow_emt ~0.1 s/step -> a few steps then cap WITH a restart file
    )
    result = run_ase_core(params)
    assert result["wall_time_capped"] is True

    fa = _fake_agent(tmp_path)
    fa._manifest_observe(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_ase",
                    "id": "c1",
                    "args": {
                        "driver": "opt",
                        "calculator": {"calculator_type": "emt"},
                        "input_structure_file": str(cu_cluster),
                    },
                }
            ],
        )
    )

    # A plain-text (non-JSON) message pointing at the real on-disk result JSON,
    # exercising the fallback scraper along the non-structured-JSON path.
    plain = (
        "Optimization CAPPED at the wall-clock limit (not converged). "
        f"Results saved to {result['result_file']}. Resume to continue."
    )
    fa._manifest_observe(ToolMessage(content=plain, tool_call_id="c1"))

    assert fa.run_manifest.status == "capped"
    step = fa.run_manifest._data["steps"][0]
    assert step["status"] == "capped"
    pending = fa.run_manifest._data["pending_next_step"]
    assert pending is not None
    # the on-disk result JSON records a restart file -> restart-aware reason.
    assert "restart_file" in pending["reason"]
    assert fa._pending_steps == {}
