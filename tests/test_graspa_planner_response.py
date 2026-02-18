from chemgraph.state.graspa_state import PlannerResponse


def test_graspa_planner_response_accepts_bare_task_list():
    payload = [{"task_index": 1, "prompt": "Run worker batch 1."}]
    parsed = PlannerResponse.model_validate(payload)
    assert parsed.next_step == "executor_subgraph"
    assert parsed.tasks[0].task_index == 1


def test_graspa_planner_response_maps_worker_tasks_to_tasks():
    payload = {
        "worker_tasks": [{"task_index": 1, "prompt": "Run worker batch 1."}],
        "next_step": "executor_subgraph",
    }
    parsed = PlannerResponse.model_validate(payload)
    assert parsed.tasks[0].prompt == "Run worker batch 1."
