"""End-to-end proof that RESUME injects the run manifest's PENDING marker.

``ChemGraph.run(query, resume_from=<sid>)`` loads the session's manifest and
prepends its rendered content (completed steps + PENDING NEXT STEP) into the
query the graph sees, so a resumed agent continues a capped step and skips
recomputing it. Unlike ``tests/test_manifest.py`` (which exercises
``render_for_context()`` in isolation), this drives the whole ``run()`` layer and
asserts the PENDING block reaches the string handed to the graph.

Hermetic: ``load_openai_model`` patched to a ``Mock`` (as in
``tests/test_llm_agent.py``), a real SQLite session store in ``tmp_path``, and
``workflow.astream`` replaced by a capturing stand-in so no graph/LLM runs.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.memory.manifest import RunManifest
from chemgraph.memory.store import SessionStore


class _CapturingWorkflow:
    """Stand-in for the compiled graph that records what ``run()`` streams into it.

    ``ChemGraph.run`` calls ``self.workflow.astream(stream_input, ...)``; we
    capture ``stream_input`` (the ``{"messages": query}`` dict) and yield a single
    terminal state so ``run()`` completes without invoking any LLM.
    """

    def __init__(self):
        self.captured_input = None
        self._final_state = {
            "messages": [AIMessage(content="ok -- resuming the capped step")]
        }

    async def astream(self, stream_input, *, stream_mode, config):
        self.captured_input = stream_input
        yield self._final_state

    def get_state(self, config):
        # No pending interrupts; keeps _stream_until_interrupt's post-stream and
        # write_state's checkpoint reads from raising.
        return SimpleNamespace(values=self._final_state, tasks=[])


def _write_capped_manifest(manifest_dir):
    """Write a run_manifest.json exactly as production does for a capped step.

    Drives the real ``RunManifest`` write API (record a completed step, then a
    capped step whose same args become the pending next step with a
    restart-aware reason, then flip status to "capped") so the on-disk shape is
    identical to what ``_manifest_observe`` writes on a wall-clock cap.
    """
    m = RunManifest(manifest_dir / "run_manifest.json")

    # A first step that finished normally -> surfaces under "do NOT recompute".
    done_idx = m.record_step_start(
        "run_ase",
        {
            "driver": "opt",
            "calculator": {"calculator_type": "mace_mp"},
            "input_structure_file": "water.xyz",
        },
    )
    m.record_step_end(
        done_idx, result_file="/flare/run/water_opt.json", wall_time=42.1
    )

    # A second step capped at the wall-clock limit -> recorded "capped" (not
    # "done") and queued as the pending next step so a resume continues it.
    capped_args = {
        "driver": "vib",
        "calculator": {"calculator_type": "mace_mp"},
        "input_structure_file": "water_opt.json",
    }
    capped_idx = m.record_step_start("run_ase", capped_args)
    m.record_step_end(
        capped_idx,
        result_file="/flare/run/water_vib_partial.json",
        wall_time=5.0,
        status="capped",
    )
    m.set_pending(
        "run_ase",
        capped_args,
        reason="wall-clock cap; resume with restart_file=/flare/run/restart_vib.json",
    )
    m.set_status("capped")
    return m


def test_resume_injects_pending_marker_into_graph_query(tmp_path):
    """The capped manifest's PENDING block must reach the query passed to astream."""
    # 1. A durable capped manifest on the shared filesystem, in a session's log_dir.
    resume_log_dir = tmp_path / "resume_session_logs"
    resume_log_dir.mkdir()
    _write_capped_manifest(resume_log_dir)

    # 2. A real session store whose prior session points get_session().log_dir at
    #    the manifest dir, so RunManifest.for_session can discover it.
    store = SessionStore(db_path=str(tmp_path / "sessions.db"))
    resume_sid = "capped00"
    store.create_session(
        session_id=resume_sid,
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        title="prior capped run",
        log_dir=str(resume_log_dir),
    )

    # 3. Construct ChemGraph fully offline: mock the model loader (no network),
    #    reuse the same store so both prior and resumed sessions live in tmp.
    with patch(
        "chemgraph.agent.llm_agent.load_openai_model", return_value=Mock()
    ):
        agent = ChemGraph(
            model_name="gpt-4o-mini",
            workflow_type="single_agent",
            session_store=store,
            log_dir=str(tmp_path / "agent_logs"),
        )

    # 4. Replace the compiled graph with a capturing stand-in so no LLM runs and
    #    we can inspect exactly what run() streams into the graph.
    capturing = _CapturingWorkflow()
    agent.workflow = capturing

    # 5. Resume.
    original_query = "run the pending vibrational-frequency step"
    result = asyncio.run(agent.run(original_query, resume_from=resume_sid))

    # run() returned the terminal message (nothing hit an LLM).
    assert isinstance(result, AIMessage)

    # 6. Assert the manifest's PENDING block reached the graph's input query.
    stream_input = capturing.captured_input
    assert isinstance(stream_input, dict)
    injected_query = stream_input["messages"]
    assert isinstance(injected_query, str)

    # The original query is still present (context is *prepended*, not replaced).
    assert original_query in injected_query

    # The PENDING NEXT STEP block and the capped step's identity are injected.
    assert "PENDING NEXT STEP" in injected_query
    assert "run_ase" in injected_query
    assert "driver=vib" in injected_query
    assert "input=water_opt.json" in injected_query
    # The restart-aware reason (how to continue the capped work) is carried through.
    assert "restart_file=/flare/run/restart_vib.json" in injected_query

    # The completed step is injected under the "do NOT recompute" heading, and the
    # capped step (driver=vib) is NOT listed there -- it only surfaces as PENDING,
    # so the resumed agent continues it as unfinished work.
    completed_section, _, pending_section = injected_query.partition(
        "=== PENDING NEXT STEP"
    )
    assert "do NOT recompute" in completed_section
    assert "driver=opt" in completed_section  # the finished step is here
    assert "water_opt.json" in completed_section  # its result file
    assert "driver=vib" not in completed_section  # the capped step is NOT here
    assert "driver=vib" in pending_section  # ... only under PENDING


def test_resume_adopts_prior_log_dir(tmp_path):
    """Resume repoints log_dir / CHEMGRAPH_LOG_DIR / run_manifest at the prior session.

    The durable partials a resume must find ({stem}_opt.partial.xyz, vib cache)
    resolve under CHEMGRAPH_LOG_DIR. A resumed agent is built with a fresh
    auto-generated log_dir, so run() must adopt the prior session's log_dir (and
    re-point self.run_manifest at that session's manifest) before anything reads
    it -- otherwise the partials are invisible and the step recomputes.
    """
    from pathlib import Path

    resume_log_dir = tmp_path / "resume_session_logs"
    resume_log_dir.mkdir()
    _write_capped_manifest(resume_log_dir)

    store = SessionStore(db_path=str(tmp_path / "sessions.db"))
    resume_sid = "capped00"
    store.create_session(
        session_id=resume_sid,
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        title="prior capped run",
        log_dir=str(resume_log_dir),
    )

    with patch(
        "chemgraph.agent.llm_agent.load_openai_model", return_value=Mock()
    ):
        agent = ChemGraph(
            model_name="gpt-4o-mini",
            workflow_type="single_agent",
            session_store=store,
            log_dir=str(tmp_path / "agent_logs"),
        )

    # sanity: before resume the agent is on its OWN fresh log dir + manifest.
    assert agent.log_dir == str(tmp_path / "agent_logs")
    assert Path(agent.run_manifest._path).parent != resume_log_dir

    agent.workflow = _CapturingWorkflow()
    asyncio.run(agent.run("continue", resume_from=resume_sid))

    # after resume the agent adopts the prior session's log dir everywhere the
    # durable partials are resolved.
    assert agent.log_dir == str(resume_log_dir)
    import os

    assert os.environ["CHEMGRAPH_LOG_DIR"] == str(resume_log_dir)
    # and new steps append to the prior manifest, not a fresh empty one.
    assert Path(agent.run_manifest._path).parent == resume_log_dir
