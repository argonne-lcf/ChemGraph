import json
from deepdiff import DeepDiff


def compare_llm_and_manual_workflow(manual_filepath, llm_filepath):
    with open(manual_filepath, 'r') as f1:
        manual_data = json.load(f1)
    with open(llm_filepath, 'r') as f2:
        llm_data = json.load(f2)
    print("1. Comparing tool_call function names...")
    funcs1 = [list(call.keys())[0] for call in manual_data["tool_calls"]]
    funcs2 = [list(call.keys())[0] for call in llm_data["tool_calls"]]
    print(f" - Dict 1: {funcs1}")
    print(f" - Dict 2: {funcs2}")
    print(f" - Equal? {funcs1 == funcs2}\n")

    print("2. Comparing tool_call arguments...")
    args1 = [call[f] for call, f in zip(manual_data["tool_calls"], funcs1)]
    args2 = [call[f] for call, f in zip(llm_data["tool_calls"], funcs2)]
    for i, (a1, a2) in enumerate(zip(args1, args2)):
        print(f" - Step {i + 1} diff:")
        diff = DeepDiff(a1, a2, ignore_order=True, significant_digits=3)
        # print(json.dumps(diff.to_dict(), indent=2))
        print(diff)
    print()

    print("3. Comparing number of tool_calls...")
    print(f" - Dict 1: {len(manual_data['tool_calls'])} steps")
    print(f" - Dict 2: {len(llm_data['tool_calls'])} steps")
    print(f" - Equal? {len(manual_data['tool_calls']) == len(llm_data['tool_calls'])}\n")

    print("4. Comparing final results...")
    diff_final = DeepDiff(
        manual_data['result'], llm_data['result'], ignore_order=True, significant_digits=3
    )
    print(json.dumps(diff_final, indent=2))


manual_filepath = "manual_result.json"
llm_filepath = "llm_result.json"
compare_llm_and_manual_workflow(manual_filepath, llm_filepath)
