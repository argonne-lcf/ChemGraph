"""Tests for the durable run manifest (cross-allocation resume worklist).

Hermetic: pure filesystem, no LLM, no network.
"""

import json

from chemgraph.memory.manifest import RunManifest


def test_records_and_renders(tmp_path):
    m = RunManifest(tmp_path / "run_manifest.json")
    idx = m.record_step_start(
        "run_ase",
        {
            "driver": "opt",
            "calculator": {"calculator_type": "mace_mp"},
            "input_structure_file": "water.xyz",
        },
    )
    m.record_step_end(idx, result_file="/flare/x/water_opt.json", wall_time=42.1)

    # flushed atomically on every call
    data = json.loads((tmp_path / "run_manifest.json").read_text())
    assert data["steps"][0]["status"] == "done"
    assert data["steps"][0]["result_file"].endswith("water_opt.json")

    txt = m.render_for_context()
    assert "do NOT recompute" in txt
    assert "water_opt.json" in txt
    assert "mace_mp" in txt


def test_pending_survives_reload(tmp_path):
    p = tmp_path / "run_manifest.json"
    m = RunManifest(p)
    m.set_pending(
        "run_ase",
        {"driver": "vib", "input_structure_file": "water_opt.json"},
        reason="walltime cap",
    )
    # reload from disk (simulates a fresh process / new allocation)
    m2 = RunManifest(p)
    txt = m2.render_for_context()
    assert "PENDING NEXT STEP" in txt
    assert "vib" in txt
    assert "walltime cap" in txt


def test_for_session_uses_log_dir(tmp_path):
    class FakeSess:
        log_dir = str(tmp_path)

    class FakeStore:
        def get_session(self, sid):
            return FakeSess()

    # no manifest yet -> None
    assert RunManifest.for_session(FakeStore(), "abc") is None

    # once a manifest exists, it's found via the session's log_dir
    RunManifest(tmp_path / "run_manifest.json").record_step_start("run_ase", {})
    m = RunManifest.for_session(FakeStore(), "abc")
    assert m is not None
    assert len(m._data["steps"]) == 1


def test_corrupt_file_does_not_crash(tmp_path):
    p = tmp_path / "run_manifest.json"
    p.write_text("{ not valid json")
    m = RunManifest(p)  # should warn + start fresh, not raise
    assert m._data["steps"] == []


# ---------------------------------------------------------------------------
# _manifest_observe hook (llm_agent) tested with mock messages -- no LLM.
# Reproduces the e2e finding that a FAILED tool call must not be recorded "done".
# ---------------------------------------------------------------------------


def _fake_agent(tmp_path):
    """A minimal object exposing the manifest + the real _manifest_observe."""
    from chemgraph.agent.llm_agent import ChemGraph

    class _FakeAgent:
        run_manifest = RunManifest(tmp_path / "run_manifest.json")
        _pending_steps = {}
        _COST_TOOLS = ChemGraph._COST_TOOLS
        _manifest_observe = ChemGraph._manifest_observe

        def __init__(self):
            # per-instance so parallel-call tests don't share the class dict
            self._pending_steps = {}

    return _FakeAgent()


def test_hook_records_successful_step(tmp_path):
    from langchain_core.messages import AIMessage, ToolMessage

    result_path = tmp_path / "water_opt.json"
    result_path.write_text(json.dumps({"wall_time": 12.3, "converged": True}))

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
                        "calculator": {"calculator_type": "mace_mp"},
                        "input_structure_file": "water.xyz",
                    },
                }
            ],
        )
    )
    assert fa._pending_steps["c1"]["idx"] == 1
    fa._manifest_observe(
        ToolMessage(
            content=f"Simulation completed. Results saved to {result_path}",
            tool_call_id="c1",
        )
    )
    assert fa._pending_steps == {}
    step = fa.run_manifest._data["steps"][0]
    assert step["status"] == "done"
    assert step["result_file"] == str(result_path)
    assert step["wall_time"] == 12.3


def test_hook_marks_failed_tool_call(tmp_path):
    """A validation/tool error must be recorded status='failed', not 'done'."""
    from langchain_core.messages import AIMessage, ToolMessage

    fa = _fake_agent(tmp_path)
    fa._manifest_observe(
        AIMessage(
            content="",
            tool_calls=[{"name": "run_ase", "id": "c1", "args": {"driver": "opt"}}],
        )
    )
    fa._manifest_observe(
        ToolMessage(
            content="Error: 1 validation error for run_ase\nparams Field required",
            tool_call_id="c1",
            status="error",
        )
    )
    step = fa.run_manifest._data["steps"][0]
    assert step["status"] == "failed"
    assert step["result_file"] is None


def test_hook_records_capped_step_as_pending(tmp_path):
    """A wall-clock-capped step is recorded 'capped' and queued as pending."""
    from langchain_core.messages import AIMessage, ToolMessage

    result_path = tmp_path / "water_opt.json"
    result_path.write_text(
        json.dumps(
            {
                "wall_time": 5.0,
                "wall_time_capped": True,
                "restart_file": "/some/restart_opt.json",
                "converged": False,
            }
        )
    )

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
                        "calculator": {"calculator_type": "mace_mp"},
                        "input_structure_file": "water.xyz",
                    },
                }
            ],
        )
    )
    fa._manifest_observe(
        ToolMessage(
            content=(
                "Optimization CAPPED at the wall-clock limit (not converged). "
                f"Partial geometry + restart saved to {result_path}. "
                "Resume with restart_file to continue."
            ),
            tool_call_id="c1",
        )
    )

    step = fa.run_manifest._data["steps"][0]
    assert step["status"] == "capped"
    assert fa.run_manifest.status == "capped"
    assert fa._pending_steps == {}

    txt = fa.run_manifest.render_for_context()
    assert "PENDING NEXT STEP" in txt
    assert "restart_file=/some/restart_opt.json" in txt
    assert "opt" in txt
    # A capped step must NOT be listed in the "do NOT recompute" completed list
    # (render only lists status=="done"): its only surface is PENDING.
    completed = txt.split("=== PENDING NEXT STEP", 1)[0]
    assert "water.xyz" not in completed


def test_hook_capped_no_restart_sets_pending_rerun(tmp_path):
    """A cap before any step (no restart) queues a rerun-with-more-budget pending."""
    from langchain_core.messages import AIMessage, ToolMessage

    result_path = tmp_path / "water_opt.json"
    result_path.write_text(
        json.dumps(
            {
                "wall_time": 0.1,
                "wall_time_capped": True,
                "restart_file": None,
                "converged": False,
            }
        )
    )

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
                        "calculator": {"calculator_type": "mace_mp"},
                        "input_structure_file": "water.xyz",
                    },
                }
            ],
        )
    )
    fa._manifest_observe(
        ToolMessage(
            content=(
                "Optimization CAPPED at the wall-clock limit before any step "
                "completed; no restart file was written. Rerun with more "
                f"wall-clock budget. Results saved to {result_path}."
            ),
            tool_call_id="c1",
        )
    )

    step = fa.run_manifest._data["steps"][0]
    assert step["status"] == "capped"
    assert fa.run_manifest.status == "capped"
    txt = fa.run_manifest.render_for_context()
    assert "rerun with more wall-clock budget" in txt


def test_hook_successful_step_leaves_status_running(tmp_path):
    """A normal (uncapped) step stays 'done' and does NOT flip status to capped."""
    from langchain_core.messages import AIMessage, ToolMessage

    result_path = tmp_path / "water_opt.json"
    result_path.write_text(
        json.dumps(
            {"wall_time": 12.3, "wall_time_capped": False, "converged": True}
        )
    )

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
                        "calculator": {"calculator_type": "mace_mp"},
                        "input_structure_file": "water.xyz",
                    },
                }
            ],
        )
    )
    fa._manifest_observe(
        ToolMessage(
            content=f"Simulation completed. Results saved to {result_path}",
            tool_call_id="c1",
        )
    )

    step = fa.run_manifest._data["steps"][0]
    assert step["status"] == "done"
    assert fa.run_manifest.status == "running"
    # A successful step leaves no PENDING block for a resume to re-render.
    assert "PENDING NEXT STEP" not in fa.run_manifest.render_for_context()


def test_success_after_cap_clears_pending(tmp_path):
    """A cap sets PENDING+status='capped'; a later uncapped step clears both.

    Models the resume happy-path: step 1 caps and queues a pending marker; the
    resumed run completes that step, which must clear the pending block and reset
    status to 'running' so the manifest no longer advertises stale work.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    capped_path = tmp_path / "water_opt.json"
    capped_path.write_text(
        json.dumps(
            {
                "wall_time": 5.0,
                "wall_time_capped": True,
                "restart_file": "/some/restart_opt.json",
                "converged": False,
            }
        )
    )
    done_path = tmp_path / "water_opt_done.json"
    done_path.write_text(
        json.dumps({"wall_time": 8.0, "wall_time_capped": False, "converged": True})
    )

    fa = _fake_agent(tmp_path)

    # 1) first attempt caps -> PENDING set, status 'capped'.
    fa._manifest_observe(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_ase",
                    "id": "c1",
                    "args": {
                        "driver": "opt",
                        "calculator": {"calculator_type": "mace_mp"},
                        "input_structure_file": "water.xyz",
                    },
                }
            ],
        )
    )
    fa._manifest_observe(
        ToolMessage(
            content=f"Optimization CAPPED. Results saved to {capped_path}.",
            tool_call_id="c1",
        )
    )
    assert fa.run_manifest.status == "capped"
    assert fa.run_manifest._data["pending_next_step"] is not None

    # 2) resume completes the step under a NEW tool_call_id -> pending cleared.
    fa._manifest_observe(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_ase",
                    "id": "c2",
                    "args": {
                        "driver": "opt",
                        "calculator": {"calculator_type": "mace_mp"},
                        "input_structure_file": "water_opt.partial.xyz",
                    },
                }
            ],
        )
    )
    fa._manifest_observe(
        ToolMessage(
            content=f"Optimization complete. Results saved to {done_path}",
            tool_call_id="c2",
        )
    )

    assert fa.run_manifest.status == "running"
    assert fa.run_manifest._data["pending_next_step"] is None
    assert "PENDING NEXT STEP" not in fa.run_manifest.render_for_context()


# ---------------------------------------------------------------------------
# C2: cost tools take one pydantic param, so the LLM's args nest under a wrapper
# key. The hook must unwrap it before recording, else the manifest reads every
# field as '?'. These feed the REAL nested shape production sees.
# ---------------------------------------------------------------------------


def test_hook_unwraps_params_wrapper(tmp_path):
    """run_ase args nested under ``params`` render with real driver/calc, not '?'."""
    from langchain_core.messages import AIMessage, ToolMessage

    result_path = tmp_path / "water_opt.json"
    result_path.write_text(json.dumps({"wall_time": 1.0, "converged": True}))

    fa = _fake_agent(tmp_path)
    fa._manifest_observe(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_ase",
                    "id": "c1",
                    # the production shape: real args nested one level down.
                    "args": {
                        "params": {
                            "driver": "opt",
                            "calculator": {"calculator_type": "mace_mp"},
                            "input_structure_file": "water.xyz",
                        }
                    },
                }
            ],
        )
    )

    step = fa.run_manifest._data["steps"][0]
    assert step["args"].get("driver") == "opt"

    # close the step so it appears in the "completed work" render.
    fa._manifest_observe(
        ToolMessage(
            content=f"Simulation completed. Results saved to {result_path}",
            tool_call_id="c1",
        )
    )
    txt = fa.run_manifest.render_for_context()
    # unwrapped -> concrete fields; the pre-fix bug rendered "driver=? calc=?".
    assert "driver=opt calc=mace_mp" in txt


def test_hook_unwraps_graspa_wrapper_and_alias(tmp_path):
    """run_graspa nests under ``graspa_input``; ``cif_path`` aliases the input file."""
    from langchain_core.messages import AIMessage

    fa = _fake_agent(tmp_path)
    fa._manifest_observe(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_graspa",
                    "id": "c1",
                    "args": {"graspa_input": {"cif_path": "mof.cif"}},
                }
            ],
        )
    )

    step = fa.run_manifest._data["steps"][0]
    # cif_path is normalized onto the canonical input_structure_file a resume reads.
    assert step["args"].get("input_structure_file") == "mof.cif"


def test_hook_records_run_mace_single(tmp_path):
    """The real MACE tool name is recorded (pre-fix _COST_TOOLS had only 'run_mace')."""
    from langchain_core.messages import AIMessage

    fa = _fake_agent(tmp_path)
    fa._manifest_observe(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_mace_single",
                    "id": "c1",
                    "args": {
                        "params": {
                            "driver": "opt",
                            "input_structure_file": "water.xyz",
                        }
                    },
                }
            ],
        )
    )

    assert len(fa.run_manifest._data["steps"]) == 1
    assert fa.run_manifest._data["steps"][0]["tool"] == "run_mace_single"


# ---------------------------------------------------------------------------
# C3: two cost tools issued in one AI message open two steps at once. The hook
# must close each by tool_call_id -- a single-slot tracker would close the wrong
# one when the ToolMessages arrive out of order.
# ---------------------------------------------------------------------------


def test_hook_two_parallel_calls_close_correct_steps(tmp_path):
    """Two parallel calls, results returned reversed -> each closes its own step."""
    from langchain_core.messages import AIMessage, ToolMessage

    done_path = tmp_path / "water_opt.json"
    done_path.write_text(
        json.dumps({"wall_time": 3.0, "wall_time_capped": False, "converged": True})
    )
    capped_path = tmp_path / "water_vib.json"
    capped_path.write_text(
        json.dumps(
            {
                "wall_time": 9.0,
                "wall_time_capped": True,
                "restart_file": "/some/vib_restart.json",
                "converged": False,
            }
        )
    )

    fa = _fake_agent(tmp_path)
    # one AI message issuing TWO cost-bearing calls: c1 opt, c2 vib.
    fa._manifest_observe(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_ase",
                    "id": "c1",
                    "args": {
                        "params": {
                            "driver": "opt",
                            "calculator": {"calculator_type": "mace_mp"},
                            "input_structure_file": "water.xyz",
                        }
                    },
                },
                {
                    "name": "run_ase",
                    "id": "c2",
                    "args": {
                        "params": {
                            "driver": "vib",
                            "calculator": {"calculator_type": "mace_mp"},
                            "input_structure_file": "water_opt.json",
                        }
                    },
                },
            ],
        )
    )
    assert set(fa._pending_steps) == {"c1", "c2"}

    # ToolMessages arrive REVERSED: c2 (vib, capped) first, then c1 (opt, done).
    fa._manifest_observe(
        ToolMessage(
            content=f"Vibrations CAPPED. Results saved to {capped_path}.",
            tool_call_id="c2",
        )
    )
    fa._manifest_observe(
        ToolMessage(
            content=f"Optimization complete. Results saved to {done_path}",
            tool_call_id="c1",
        )
    )

    assert fa._pending_steps == {}
    steps = {s["index"]: s for s in fa.run_manifest._data["steps"]}
    # step 1 was the opt (c1) -> done despite arriving second.
    assert steps[1]["args"]["driver"] == "opt"
    assert steps[1]["status"] == "done"
    # step 2 was the vib (c2) -> capped despite arriving first.
    assert steps[2]["args"]["driver"] == "vib"
    assert steps[2]["status"] == "capped"


def test_model_typed_args_round_trip_across_reload(tmp_path):
    """A Pydantic-model calculator arg must survive the reload boundary.

    Regression: json.dumps(default=str) stringified models to an unparseable
    repr, so a resumed process reloaded ``calc=?`` and lost the args.
    """

    class _Calc:
        # minimal stand-in for a pydantic v2 model (has model_dump)
        calculator_type = "mace_mp"

        def model_dump(self):
            return {"calculator_type": self.calculator_type, "model": "small"}

    p = tmp_path / "run_manifest.json"
    m = RunManifest(p)
    idx = m.record_step_start(
        "run_ase",
        {"driver": "opt", "calculator": _Calc(), "input_structure_file": "w.xyz"},
    )
    m.record_step_end(idx, result_file="/x/w_opt.json", wall_time=1.0)

    # on disk the model became a plain dict (not a str repr)
    data = json.loads(p.read_text())
    assert data["steps"][0]["args"]["calculator"] == {
        "calculator_type": "mace_mp",
        "model": "small",
    }

    # and a fresh process still resolves the calculator label
    reloaded = RunManifest(p)
    assert "calc=mace_mp" in reloaded.render_for_context()


def test_wrong_shape_json_starts_fresh(tmp_path):
    """Valid JSON of the wrong shape is ignored, not fed to readers that crash."""
    for payload in ('{"unexpected": true}', "null", "[]", '{"schema_version": 999}'):
        p = tmp_path / "run_manifest.json"
        p.write_text(payload)
        m = RunManifest(p)  # must not raise
        assert m._data["steps"] == []
        # readers stay safe on the fallback data
        assert "do NOT recompute" in m.render_for_context()


def test_valid_top_shape_with_malformed_step_does_not_crash_resume(tmp_path):
    """A right-shaped manifest whose steps hold junk must still render, not crash.

    ``_is_valid_shape`` accepts the top level (schema_version + steps-is-a-list),
    so a null step or a step missing ``status``/``index`` slips past the load
    guard. ``render_for_context`` runs unguarded on the resume path, so it must
    tolerate such elements and never raise (a TypeError/KeyError here would kill
    the very resume the manifest exists to enable).
    """
    good = {
        "index": 1,
        "tool": "run_ase",
        "args": {"driver": "opt"},
        "status": "done",
        "result_file": "/x/opt.json",
        "wall_time": 1.0,
    }
    payload = {
        "schema_version": 1,
        "steps": [None, {"tool": "run_ase"}, good],  # null, status-less, then good
        "pending_next_step": None,
        "status": "capped",
    }
    p = tmp_path / "run_manifest.json"
    p.write_text(json.dumps(payload))

    m = RunManifest(p)  # load must not raise
    rendered = m.render_for_context()  # unguarded resume-path call must not raise
    # the one well-formed done step still renders; the junk is skipped
    assert "[1] run_ase" in rendered
    # a malformed pending block is also tolerated (rendered as a safe placeholder)
    p.write_text(json.dumps({**payload, "pending_next_step": {"args": None}}))
    RunManifest(p).render_for_context()  # must not raise

    # a truthy-but-non-dict ``args`` (a list or string, not just a falsy null)
    # must also be tolerated: ``a.get(...)`` would raise AttributeError on it.
    # This covers both a done step and the pending block.
    bad_args_step = {**good, "args": ["driver", "opt"]}
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "steps": [bad_args_step],
                "pending_next_step": {"tool": "run_ase", "args": "oops"},
                "status": "capped",
            }
        )
    )
    RunManifest(p).render_for_context()  # must not raise on non-dict args

    # record_step_end also tolerates malformed elements in the list
    RunManifest(p).record_step_end(1, result_file="/x/opt.json", wall_time=1.0)


def test_non_serializable_arg_does_not_crash_the_run(tmp_path):
    """A non-JSON-serializable arg must be tolerated, not propagated."""
    p = tmp_path / "run_manifest.json"
    m = RunManifest(p)
    # a tuple dict-key survives _jsonable (keys aren't coerced) but breaks
    # json.dumps; _flush must swallow it and never crash the caller.
    m.record_step_start("run_ase", {"bad": {("a", "b"): 1}})  # should not raise
    # the tmp file must not be left behind on a failed flush
    assert not p.with_suffix(".tmp").exists()
