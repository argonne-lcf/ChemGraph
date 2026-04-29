import json
from chemgraph.utils.tool_call_eval import (
    multi_function_checker_without_order,
)
from chemgraph.tools.cheminformatics_tools import molecule_name_to_smiles, smiles_to_atomsdata
from langchain_core.utils.function_calling import convert_to_openai_function
from chemgraph.tools.ase_tools import run_ase, file_to_atomsdata, save_atomsdata_to_file

toolsets = [
    molecule_name_to_smiles,
    run_ase,
    smiles_to_atomsdata,
    file_to_atomsdata,
    save_atomsdata_to_file,
]

func_descriptions = [convert_to_openai_function(tool) for tool in toolsets]

with open("llm_workflow_2025-05-19_14-09-36.json", "r") as rf:
    model_outputs = json.load(rf)

with open(
    ("ground_truth.json"),
    "r",
) as rf:
    answers = json.load(rf)

model_output = model_outputs["Water Gas Shift Reaction"]["llm_workflow"].get("tool_calls", {})
answer = answers["Water Gas Shift Reaction"]["manual_workflow"].get("tool_calls", {})

print(
    multi_function_checker_without_order(
        func_descriptions=func_descriptions,
        model_outputs=model_output,
        answers=answer,
    )
)
