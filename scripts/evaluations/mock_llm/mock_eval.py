"""Module to evaluate LLM performance on tool-calling workflows."""

import pprint
import json
import datetime
import argparse
from langchain_core.utils.function_calling import convert_to_openai_function
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.utils.get_workflow_from_llm import get_workflow_from_state
from chemgraph.utils.tool_call_eval import multi_function_checker_with_order
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_atomsdata,
)
from chemgraph.tools.ase_tools import (
    run_ase,
    file_to_atomsdata,
    save_atomsdata_to_file,
)


def evaluate_model(
    model_name: str,
    input_file: str = "ground_truth_sample.json",
):
    """
    Evaluate the tool-calling behavior of an LLM given a list of queries.

    Parameters
    ----------
    model_name : str
        Name of the LLM model to use in ChemGraph.
    input_file : str
        Path to the ground truth sample JSON file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        list_of_queries = json.load(f)

    workflow_type = "mock_agent"
    cg = ChemGraph(
        model_name=model_name,
        workflow_type=workflow_type,
        structured_output=True,
        return_option="state",
    )

    llm_tool_calls = []
    for idx, item in enumerate(list_of_queries):
        query = item["query"]
        state = cg.run(query, {"configurable": {"thread_id": str(idx)}})
        llm_tool_call = get_workflow_from_state(state)
        llm_tool_calls.append(llm_tool_call)

    # Save tool call results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"{model_name}_{timestamp}_tool_call.json"
    with open(output_file, "w", encoding="utf-8") as wf:
        json.dump(llm_tool_calls, wf, indent=4)
    print(f"Saved tool calls to {output_file}")

    # Evaluation
    toolsets = [
        molecule_name_to_smiles,
        run_ase,
        smiles_to_atomsdata,
        file_to_atomsdata,
        save_atomsdata_to_file,
    ]

    func_descriptions = [convert_to_openai_function(tool) for tool in toolsets]
    accurate_tool_call = 0
    eval_details = {}
    for idx, toolcall in enumerate(llm_tool_calls):
        model_outputs = llm_tool_calls[idx].get("tool_calls", {})
        answers = list_of_queries[idx].get("answer", {}).get("tool_calls", {})
        eval_result = multi_function_checker_with_order(
            func_descriptions=func_descriptions,
            model_outputs=model_outputs,
            answers=answers,
        )
        if eval_result["acc_n_toolcalls"] == eval_result["n_toolcalls"]:
            accurate_tool_call += 1
        eval_details[list_of_queries[idx]["query"]] = eval_result
        print(eval_result)
    accuracy = accurate_tool_call / len(llm_tool_calls) * 100

    print(f"Accuracy of {model_name}: {accuracy}% ({accurate_tool_call}/10 accurate tool calls)")

    output_eval_file = f"{model_name}_{timestamp}_eval.txt"

    # Cannot do json.dump() due to DeepDiff output not serializable.
    with open(output_eval_file, "w", encoding="utf-8") as wf:
        pprint.pprint(eval_details, stream=wf, width=120)
    print(f"Saved evaluation results to {output_eval_file}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ChemGraph tool-calling performance for different LLMs."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the LLM model to evaluate (e.g., gpt-4o, claude-3.5-haiku)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="ground_truth_sample.json",
        help="Path to input JSON file of queries",
    )

    args = parser.parse_args()
    evaluate_model(args.model_name, args.input_file)


if __name__ == "__main__":
    main()
