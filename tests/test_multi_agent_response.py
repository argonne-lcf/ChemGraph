from chemgraph.schemas.multi_agent_response import PlannerResponse


def test_planner_response_accepts_worker_tasks_object():
    payload = {
        "worker_tasks": [
            {"task_index": 1, "prompt": "Calculate methane enthalpy."},
            {"task_index": 2, "prompt": "Calculate oxygen enthalpy."},
        ]
    }
    parsed = PlannerResponse.model_validate(payload)
    assert len(parsed.worker_tasks) == 2
    assert parsed.worker_tasks[0].task_index == 1


def test_planner_response_accepts_bare_task_list():
    payload = [
        {"task_index": 1, "prompt": "Calculate methane enthalpy."},
        {"task_index": 2, "prompt": "Calculate oxygen enthalpy."},
    ]
    parsed = PlannerResponse.model_validate(payload)
    assert len(parsed.worker_tasks) == 2
    assert parsed.worker_tasks[1].prompt == "Calculate oxygen enthalpy."
