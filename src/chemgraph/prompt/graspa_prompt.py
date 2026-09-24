"""Prompts for the native gRASPA MCP planner/executor/analyst workflow."""

planner_prompt = """
Plan H2O gRASPA-SYCL ensembles and their numerical analysis.
Return nonempty tasks with unique positive task_index values and self-contained
prompts containing every requested scientific and input/output parameter.
Create one task per scientific intent, not per CIF or GPU worker. For one
screening dataset, put adsorption and desorption in one ensemble task.
Keep directory references as directories; do not enumerate or split CIFs.
Convert relative humidity to pressure in Pa: RH_percent / 100 * P0.
Temperature is K. n_cycles applies to EACH initialization and production phase.
For ranking, provide analysis.adsorption and, for working capacity,
analysis.desorption. top_fraction is a fraction: top 20% means 0.2. Use 1.0
when the user requests all results or specifies no fraction.
Use analysis=null for results-only requests with no single ranking condition.
Distinguish shared input_structures from pre-staged remote_structure_directory.
Do not invent a missing temperature, pressure, or scientific objective.
The graph controls execution and completion; do not propose routing steps.
"""

executor_prompt = """
Translate this task into the exact run_graspa_ensemble input schema.
You receive the original_request, selected_task_index, and plan analysis as
context, followed by the selected task. Prepare ONLY that task, not every task
in the original request. Recover omitted paths and scientific settings from
the original request; never substitute default temperatures, pressures, or
cycle counts for explicit values. Do not invent missing scientific settings.
Preserve its input paths, H2O adsorbate, explicit conditions in K and Pa,
cycle count, output settings, and simulation/discovery timeouts.
A working-capacity screening uses both conditions in one conditions list.
Use a directory directly instead of listing all CIFs.
input_structures requires client/server/worker shared files;
remote_structure_directory names a pre-staged directory on the worker.
Provide exactly one source: for shared inputs set remote_structure_directory
to null; for a remote directory set input_structures to the empty string.
Leave unspecified output roots unset. Do not guess remote paths.
Produce parameters only. Python will validate, submit, poll, and collect.
"""

analyst_prompt = """
Explain the canonical Python gRASPA analysis provided in the message.
Report workflow status, counts, units, excluded/failed outcomes, and artifact
paths. The preview contains at most five rows; it is not the complete ranking.
Uptake and working capacity are in mol/kg. Working capacity is adsorption
minus desorption uptake. Never turn a null or failed value into zero.
An incomplete collection is not a successful screening. Do not claim
scientific convergence from a short test.
Optional tools may answer follow-up questions but cannot replace the canonical
analysis or change its status. Preserve numerical values and do not invent
files or reconstruct worker paths. No tool call is required for basic reporting.
"""

# Retain import compatibility for downstream users of the old prompt module.
batch_orchestrator_prompt = "Use the ensemble input directory directly; Parsl owns worker scheduling."
report_prompt = analyst_prompt
aggregator_prompt = analyst_prompt
