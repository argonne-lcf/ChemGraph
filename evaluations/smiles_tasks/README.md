Steps for evaluations. Notes that this is a single comparison. Needs to loop this to do evaluation for many inputs.
1. Run run.py to generate manual_result.json and llm_result.json sequentially
2. Run eval.py to evaluate. Currently using deepdiff for comparison. However, this probably doesn't work well for atomic coordinates.