import json
import os
from deepdiff import DeepDiff
import argparse
import math


def compare_llm_and_manual_workflow(
    manual_workflow_fp: str,
    llm_workflow_fp: str,
    print_diff=True,
    vib_task=False,
    save_task=False,
    reaction_task=False,
):
    """Compare a manual workflow and an LLM workflow. Criteria:
    (1) workflow's result, (2) tool names and (3) tool counts.

    Args:
        manual_workflow_fp (str): Path to log file for manual workflow.
        llm_workflow_fp (str): Path to log file for LLM workflow.
    """
    combined_eval_score = {}

    with open(manual_workflow_fp, "r") as f1:
        mdata = json.load(f1)
    with open(llm_workflow_fp, "r") as f2:
        ldata = json.load(f2)

    if len(mdata) != len(ldata):
        return f"Error. Different number of logs in {manual_workflow_fp} and {llm_workflow_fp}"

    for key in mdata:
        if key not in ldata:
            print(f"{key} does not exist in {llm_workflow_fp}")
        else:
            eval_score = {
                "workflow_success": True,
                "matched_result": False,
                "matched_tool_name": False,
                "matched_tool_count": False,
            }
            manual_data = mdata[key]
            llm_data = ldata[key]

            try:
                # Handle tasks that do not save file.
                if not save_task:
                    # Handle tasks that expect string return
                    if isinstance(manual_data["manual_workflow"]["result"], str):
                        if manual_data["manual_workflow"]["result"].startswith("ERROR"):
                            eval_score["workflow_success"] = False
                            if isinstance(llm_data["llm_workflow"]["result"], str):
                                llm_result_lower = llm_data["llm_workflow"]["result"].lower()
                                if (
                                    "error" in llm_result_lower
                                    or "fail" in llm_result_lower
                                    or "issue" in llm_result_lower
                                ):
                                    eval_score["matched_result"] = True
                        else:
                            if (
                                manual_data["manual_workflow"]["result"]
                                == llm_data["llm_workflow"]["result"]
                            ):
                                eval_score["matched_result"] = True
                    else:
                        if type(manual_data["manual_workflow"]["result"]) is not type(
                            llm_data["llm_workflow"]["result"]
                        ):
                            eval_score["matched_result"] = False
                        else:
                            if not vib_task:
                                if reaction_task:
                                    if isinstance(
                                        manual_data["manual_workflow"]["result"]["value"], float
                                    ) and isinstance(
                                        llm_data["llm_workflow"]["result"]["value"], float
                                    ):
                                        eval_score["matched_result"] = math.isclose(
                                            manual_data["manual_workflow"]["result"]["value"],
                                            llm_data["llm_workflow"]["result"]["value"],
                                            abs_tol=1e-1,
                                        )
                                        if not eval_score["matched_result"]:
                                            if print_diff:
                                                print(
                                                    f"Difference between manual and LLM workflow for {key}: "
                                                )
                                                print(
                                                    f"Manual workflow result: {manual_data['manual_workflow']['result']['value']}"
                                                )
                                                print(
                                                    f"LLM workflow result: {llm_data['llm_workflow']['result']['value']}"
                                                )

                                else:
                                    diff_final = DeepDiff(
                                        manual_data["manual_workflow"]['result'],
                                        llm_data["llm_workflow"]['result'],
                                        ignore_order=True,
                                        significant_digits=3,
                                        exclude_paths={"root['cell']", "root['pbc']"},
                                    )
                                    if diff_final == {}:
                                        eval_score["matched_result"] = True
                                    else:
                                        if print_diff:
                                            print(
                                                f"Difference between manual and LLM workflow for {key}: "
                                            )
                                            print(diff_final)
                            # Handling vibrational result calculation
                            else:
                                converted_manual_result = [
                                    complex(val.replace("i", "j")) if "i" in val else float(val)
                                    for val in manual_data["manual_workflow"]["result"][
                                        "frequency_cm1"
                                    ]
                                ]
                                converted_llm_result = [
                                    complex(val.replace("i", "j")) if "i" in val else float(val)
                                    for val in llm_data["llm_workflow"]["result"]["frequency_cm1"]
                                ]
                                diff_final = DeepDiff(
                                    converted_manual_result,
                                    converted_llm_result,
                                    significant_digits=2,
                                    number_format_notation="f",
                                )
                                if diff_final == {}:
                                    eval_score["matched_result"] = True
                                else:
                                    if print_diff:
                                        print(
                                            f"Difference between manual and LLM workflow for {key}: "
                                        )
                                        print(diff_final)
                else:
                    manual_file = os.path.join("manual_files", f"{key}.xyz")
                    llm_file = os.path.join("llm_files", f"{key}.xyz")
                    if os.path.isfile(os.path.join(manual_file)):
                        if not os.path.isfile(llm_file):
                            eval_score["matched_result"] = False
                        else:
                            with open(manual_file, "r") as f1, open(llm_file, "r") as f2:
                                if f1.read() == f2.read():
                                    eval_score["matched_result"] = True
                                else:
                                    eval_score["matched_result"] = False

            except Exception as e:
                continue
            manual_set = set(
                list(call)[0] for call in list(manual_data["manual_workflow"]["tool_calls"])
            )
            llm_set = set(list(call)[0] for call in list(llm_data["llm_workflow"]["tool_calls"]))

            if manual_set == llm_set:
                eval_score["matched_tool_name"] = True
            else:
                if print_diff:
                    print(f"Difference in tool calls between manual and LLM workflow for {key}")
                    print(f"Manual tool calls: {manual_set}")
                    print(f"LLM tool calls: {llm_set}")
            if len(manual_data["manual_workflow"]['tool_calls']) == len(
                llm_data["llm_workflow"]['tool_calls']
            ):
                eval_score["matched_tool_count"] = True
            combined_eval_score[key] = {"eval_score": eval_score}
    return combined_eval_score


def get_statistics(combined_eval_score: list):
    """Give some statistics based on evaluation dictionary"""
    number_of_sims = len(combined_eval_score)
    correct_results = 0
    correct_tool_names = 0
    correct_tool_counts = 0
    failed = []
    for sim in combined_eval_score:
        if combined_eval_score[sim]['eval_score']["matched_result"]:
            correct_results += 1
        else:
            failed.append(sim)
        if combined_eval_score[sim]['eval_score']["matched_tool_name"]:
            correct_tool_names += 1
        if combined_eval_score[sim]['eval_score']["matched_tool_count"]:
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

    if len(failed) > 0:
        print("The final results are different for the following molecules/SMILES/reactions: ")
        for fd in failed:
            print(f"- {fd}")
    return success_rate


def main(manual_workflow, llm_workflow, print_diff, vib_task, save_task, reaction_task):
    result = compare_llm_and_manual_workflow(
        manual_workflow, llm_workflow, print_diff, vib_task, save_task, reaction_task
    )
    get_statistics(result)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Get statistics for comparing an LLM workflow and manual workflow."
    )
    parser.add_argument(
        "--manual_workflow",
        type=str,
        default="manual_workflow.json",
    )
    parser.add_argument(
        "--llm_workflow",
        type=str,
    )
    parser.add_argument(
        "--vib_task",
        action="store_true",
        help="Tell the eval whether the eval is for vibrational frequency calculation.",
    )
    parser.add_argument(
        "--save_task",
        action="store_true",
        help="Tell the eval whether the eval is for save file task.",
    )
    parser.add_argument(
        "--reaction_task",
        action="store_true",
        help="Tell the eval whether the eval is for save file task.",
    )

    parser.add_argument("--print_diff", type=str, default=True)
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        args.manual_workflow,
        args.llm_workflow,
        args.print_diff,
        args.vib_task,
        args.save_task,
        args.reaction_task,
    )
