from chemgraph.schemas.multi_agent_response import PlannerResponse


def test_planner_response_accepts_full_payload():
    payload = {
        "thought_process": "Decomposing into per-molecule calculations.",
        "next_step": "executor_subgraph",
        "tasks": [
            {"task_index": 1, "prompt": "Calculate methane enthalpy."},
            {"task_index": 2, "prompt": "Calculate oxygen enthalpy."},
        ],
    }
    parsed = PlannerResponse.model_validate(payload)
    assert len(parsed.tasks) == 2
    assert parsed.tasks[0].task_index == 1
    assert parsed.next_step == "executor_subgraph"
    assert parsed.thought_process == "Decomposing into per-molecule calculations."


def test_planner_response_accepts_legacy_worker_tasks_key():
    """Backward compat: accept ``worker_tasks`` key and coerce to ``tasks``."""
    payload = {
        "worker_tasks": [
            {"task_index": 1, "prompt": "Calculate methane enthalpy."},
            {"task_index": 2, "prompt": "Calculate oxygen enthalpy."},
        ]
    }
    parsed = PlannerResponse.model_validate(payload)
    assert len(parsed.tasks) == 2
    assert parsed.next_step == "executor_subgraph"


def test_planner_response_accepts_bare_task_list():
    payload = [
        {"task_index": 1, "prompt": "Calculate methane enthalpy."},
        {"task_index": 2, "prompt": "Calculate oxygen enthalpy."},
    ]
    parsed = PlannerResponse.model_validate(payload)
    assert len(parsed.tasks) == 2
    assert parsed.tasks[1].prompt == "Calculate oxygen enthalpy."
    assert parsed.next_step == "executor_subgraph"


def test_planner_response_finish():
    payload = {
        "thought_process": "All tasks complete. Final answer: 42 eV.",
        "next_step": "FINISH",
    }
    parsed = PlannerResponse.model_validate(payload)
    assert parsed.next_step == "FINISH"
    assert parsed.tasks is None
