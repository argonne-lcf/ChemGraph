Steps for evaluations. Notes that this is a single comparison. Needs to loop this to do evaluation for many inputs.
1. (Optional) Run run_manual_workflow.py to obtain a manual_workflow.json. This file contains the details of manual workflow run. A sample manual_workflow.json is also provided (n_structure=10)
2. Run run_llm_workflow.py to obtain a llm_workflow.json. This file contains the details of llm workflow run. Adjust the parameters (human prompts, system prompt).
3. Run eval.py to do comparison and get statistics.