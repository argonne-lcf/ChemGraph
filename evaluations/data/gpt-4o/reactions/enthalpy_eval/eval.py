import json
import math
# Right now for the results, it compares direct matching of dictionary - so this can be an issue. Need to develop better metrics.


def compare_llm_and_manual_workflow(manual_workflow_fp: str, llm_workflow_fp: str):
    """Compare a manual workflow and an LLM workflow. Criteria:
    (1) workflow's result, (2) tool names and (3) tool counts.

    Args:
        manual_workflow_fp (str): Path to log file for manual workflow.
        llm_workflow_fp (str): Path to log file for LLM workflow.
    """
    combined_eval_score = []

    with open(manual_workflow_fp, "r") as f1:
        mdata = json.load(f1)
    with open(llm_workflow_fp, "r") as f2:
        ldata = json.load(f2)

    if len(mdata) != len(ldata):
        return f"Error. Different number of logs in {manual_workflow_fp} and {llm_workflow_fp}"
    for manual_data, llm_data in zip(mdata, ldata):
        man_react_name = manual_data["name"]
        llm_react_name = llm_data["name"]

        if man_react_name != llm_react_name:
            print("Error. Different smiles found for the comparison")
            print(f"Manual reaction name: {man_react_name}")
            print(f"LLM reaction name: {llm_react_name}")
            continue
        eval_score = {
            "workflow_success": True,
            "matched_result": False,
            "matched_tool_name": False,
            "matched_tool_count": False,
        }
        if isinstance(manual_data["manual_workflow"]["result"], str):
            if manual_data["manual_workflow"]["result"].startswith("ERROR"):
                eval_score["workflow_success"] = False
                if isinstance(llm_data["llm_workflow"]["result"], str):
                    llm_result_lower = llm_data["llm_workflow"]["result"].lower()
                    if "error" in llm_result_lower or "fail" in llm_result_lower:
                        eval_score["result"] = 1
            elif isinstance(llm_data["llm_workflow"]["result"], str):
                eval_score["workflow_success"] = False
        else:
            if isinstance(manual_data["manual_workflow"]["result"]["value"], float) and isinstance(
                llm_data["llm_workflow"]["result"]["value"], float
            ):
                result_equal = math.isclose(
                    manual_data["manual_workflow"]["result"]["value"],
                    llm_data["llm_workflow"]["result"]["value"],
                    abs_tol=1e-1,
                )
            else:
                result_equal = False
        eval_score["matched_result"] = result_equal
        if not result_equal:
            print(manual_data["name"])
            print(f"Manual workflow value: {manual_data['manual_workflow']['result']['value']}")
            print(f"LLM workflow value: {llm_data['llm_workflow']['result']}")

        manual_set = set(
            list(call)[0] for call in list(manual_data["manual_workflow"]["tool_calls"])
        )
        llm_set = set(list(call)[0] for call in list(llm_data["llm_workflow"]["tool_calls"]))

        if manual_set == llm_set:
            eval_score["matched_tool_name"] = True
        else:
            print(manual_set, llm_set)
        if len(manual_data["manual_workflow"]['tool_calls']) == len(
            llm_data["llm_workflow"]['tool_calls']
        ):
            eval_score["matched_tool_count"] = True
        added_data = {}
        added_data["name"] = manual_data["name"]
        added_data["eval_score"] = eval_score
        combined_eval_score.append(added_data)
    return combined_eval_score


def get_statistics(combined_eval_score: list):
    """Give some statistics based on evaluation dictionary"""

    number_of_sims = len(combined_eval_score)
    correct_results = 0
    correct_tool_names = 0
    correct_tool_counts = 0
    failed_names = []
    for sim in combined_eval_score:
        if sim['eval_score']["matched_result"]:
            correct_results += 1
        else:
            failed_names.append(sim["name"])
        if sim['eval_score']["matched_tool_name"]:
            correct_tool_names += 1
        if sim['eval_score']["matched_tool_count"]:
            correct_tool_counts += 1
    success_rate = {}
    success_rate["correct_result"] = correct_results / number_of_sims
    success_rate["correct_tool_names"] = correct_tool_names / number_of_sims
    success_rate["correct_tool_counts"] = correct_tool_counts / number_of_sims

    print(
        f"Number of correct results: {correct_results}/{number_of_sims} ({success_rate['correct_result'] * 100}%)"
    )
    print(
        f"Number of correct tool names: {correct_tool_names}/{number_of_sims} ({success_rate['correct_tool_names'] * 100}%)"
    )
    print(
        f"Number of correct tool counts: {correct_tool_counts}/{number_of_sims} ({success_rate['correct_tool_counts'] * 100}%)"
    )

    if len(failed_names) > 0:
        print("The final results are different for the following reaction(s): ")
        for n in failed_names:
            print(f"- Name: {n}")
    return success_rate


def main():
    result = compare_llm_and_manual_workflow(
        manual_workflow_fp="manual_workflow.json", llm_workflow_fp="llm_workflow.json"
    )
    get_statistics(result)


if __name__ == "__main__":
    main()
