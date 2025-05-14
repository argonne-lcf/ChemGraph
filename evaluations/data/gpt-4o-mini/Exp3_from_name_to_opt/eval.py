import json
import numpy as np
import pandas as pd
import sys

# === Load data ===
with open(sys.argv[1]) as f:
    llm_data = json.load(f)
with open("manual_workflow.json") as f:
    manual_data = json.load(f)

# === Helper functions ===
def safe_parse_result(result):
    if isinstance(result, str):
        trimmed = result.strip()
        if trimmed.startswith("{") or trimmed.startswith("["):
            try:
                return json.loads(trimmed)
            except Exception:
                return None
        else:
            return trimmed
    return result

def remove_ignored_fields(obj, ignored_keys=("cell", "pbc", "optimizer")):
    if isinstance(obj, dict):
        return {k: remove_ignored_fields(v, ignored_keys) for k, v in obj.items() if k not in ignored_keys}
    elif isinstance(obj, list):
        return [remove_ignored_fields(v, ignored_keys) for v in obj]
    return obj

def simplify_args(args):
    return remove_ignored_fields(args)

def extract_tool_calls(workflow):
    return [(list(call.keys())[0], call[list(call.keys())[0]]) for call in workflow.get("tool_calls", [])]

def parse_complex_or_real(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip().lower().replace(" ", "")
        if val.endswith("i"):
            try:
                return complex(0, float(val[:-1]))
            except Exception:
                return val
        try:
            return float(val)
        except Exception:
            return val
    return val

def compare_structs_with_tolerance(a, b, tol=1e-3):
    if type(a) != type(b):
        try:
            a_val = parse_complex_or_real(a)
            b_val = parse_complex_or_real(b)
            return abs(a_val - b_val) < tol
        except:
            return False

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(compare_structs_with_tolerance(a[k], b[k], tol) for k in a)

    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(compare_structs_with_tolerance(x, y, tol) for x, y in zip(a, b))

    if isinstance(a, (float, int, complex)):
        try:
            return abs(a - b) < tol
        except:
            return False

    return a == b

# === Final result comparison ===
def compare_results(llm_result_raw, manual_result_raw):
    llm_result = safe_parse_result(llm_result_raw)
    manual_result = manual_result_raw

    if isinstance(llm_result, str) and isinstance(manual_result, str):
        return llm_result.strip() == manual_result.strip(), None

    if isinstance(llm_result, dict) and isinstance(manual_result, dict):
        # Geometry
        if "numbers" in llm_result and "positions" in llm_result:
            llm_s = {k: v for k, v in llm_result.items() if k not in ["cell", "pbc"]}
            manual_s = {k: v for k, v in manual_result.items() if k not in ["cell", "pbc"]}
            numbers_match = llm_s.get("numbers", []) == manual_s.get("numbers", [])
            try:
                pos_diff = np.linalg.norm(
                    np.array(llm_s.get("positions", [])) - np.array(manual_s.get("positions", []))
                )
            except:
                pos_diff = float("inf")
            return numbers_match, pos_diff

        # Frequencies
        if "frequency_cm1" in llm_result and "frequency_cm1" in manual_result:
            try:
                llm_freq = [parse_complex_or_real(x) for x in llm_result["frequency_cm1"]]
                manual_freq = [parse_complex_or_real(x) for x in manual_result["frequency_cm1"]]
                if len(llm_freq) != len(manual_freq):
                    return False, None
                diff = np.linalg.norm(np.array(llm_freq) - np.array(manual_freq))
                return diff < 1e-3, diff
            except:
                return False, None

        # Property-value-unit (ignore 'property' mismatch)
        if all(k in llm_result for k in ["value", "unit"]) and all(k in manual_result for k in ["value", "unit"]):
            try:
                val1 = parse_complex_or_real(llm_result["value"])
                val2 = parse_complex_or_real(manual_result["value"])
                value_match = abs(val1 - val2) < 1e-2
                unit_match = llm_result["unit"] == manual_result["unit"]

                # Optional warning for property label mismatch
                if "property" in llm_result and "property" in manual_result:
                    prop_llm = llm_result["property"].lower().replace(" ", "").replace("_", "")
                    prop_manual = manual_result["property"].lower().replace(" ", "").replace("_", "")
                    if prop_llm != prop_manual:
                        print(f"⚠ Property label mismatch in molecule (not used in comparison): {prop_llm} vs {prop_manual}")

                return value_match and unit_match, abs(val1 - val2)
            except:
                return False, None

    # Fallback: nested tolerant structure
    if isinstance(llm_result, (dict, list)) and isinstance(manual_result, (dict, list)):
        is_match = compare_structs_with_tolerance(llm_result, manual_result)
        return is_match, None

    return False, None

# === Main loop ===
summary_rows = []
detailed_mismatches = []
final_result_mismatches = []
malformed_entries = {}

total_tool_calls = 0
matching_tool_calls = 0
mismatched_tool_calls = 0

for mol in manual_data:
    try:
        llm_result_raw = llm_data[mol]["llm_workflow"]["result"]
        manual_result = manual_data[mol]["manual_workflow"]["result"]
        final_result_match, pos_diff = compare_results(llm_result_raw, manual_result)
        if not final_result_match:
            final_result_mismatches.append({
                "molecule": mol,
                "llm_result": safe_parse_result(llm_result_raw),
                "manual_result": manual_result,
                "diff": pos_diff
            })
    except Exception as e:
        malformed_entries[mol] = str(e)
        final_result_match = False
        pos_diff = None

    try:
        llm_calls = extract_tool_calls(llm_data[mol]["llm_workflow"])
        manual_calls = extract_tool_calls(manual_data[mol]["manual_workflow"])
    except Exception as e:
        malformed_entries[mol] = "Failed to extract tool calls"
        llm_calls = []
        manual_calls = []

    tool_names_match = [a[0] == b[0] for a, b in zip(llm_calls, manual_calls)]
    tool_args_match = []

    for i, (llm_call, manual_call) in enumerate(zip(llm_calls, manual_calls)):
        tool_llm, args_llm = llm_call
        _, args_manual = manual_call
        is_match = compare_structs_with_tolerance(
            simplify_args(args_llm), simplify_args(args_manual)
        )
        tool_args_match.append(is_match)

        total_tool_calls += 1
        if is_match:
            matching_tool_calls += 1
        else:
            mismatched_tool_calls += 1
            detailed_mismatches.append({
                "molecule": mol,
                "tool_index": i,
                "tool_name": tool_llm,
                "llm_args": args_llm,
                "manual_args": args_manual
            })

    summary_rows.append({
        "molecule": mol,
        "final_result_match": final_result_match,
        "positions_diff_norm": round(pos_diff, 4) if isinstance(pos_diff, float) and np.isfinite(pos_diff) else None,
        "tool_names_match": all(tool_names_match) if tool_names_match else False,
        "tool_args_match": all(tool_args_match) if tool_args_match else False,
        "num_calls_match": len(llm_calls) == len(manual_calls),
        "llm_tool_calls": len(llm_calls),
        "manual_tool_calls": len(manual_calls)
    })

# === Output ===

summary_df = pd.DataFrame(summary_rows)
mismatch_df = pd.DataFrame(detailed_mismatches)

print("\n=== COMPARISON SUMMARY ===")
print(summary_df.to_string(index=False))

if not mismatch_df.empty:
    """
    print("\n=== TOOL ARGUMENT MISMATCHES ===")
    for _, row in mismatch_df.iterrows():
        print(f"\nMolecule: {row['molecule']} | Tool: {row['tool_name']} | Index: {row['tool_index']}")
        print("LLM Args:")
        print(json.dumps(remove_ignored_fields(row['llm_args']), indent=2))
        print("Manual Args:")
        print(json.dumps(remove_ignored_fields(row['manual_args']), indent=2))
    """
if final_result_mismatches:
    print("\n=== FINAL RESULT MISMATCHES ===")
    for entry in final_result_mismatches:
        print(f"\nMolecule: {entry['molecule']}")
        print("LLM result:")
        print(json.dumps(entry['llm_result'], indent=2))
        print("Manual result:")
        print(json.dumps(entry['manual_result'], indent=2))
        if entry["diff"] is not None:
            print(f"Difference (norm or abs): {round(entry['diff'], 6)}")

# === Stats ===
# === JSON STATISTICS OUTPUT ===
total_mols = len(summary_df)
stats = {
    "total_molecules": total_mols,
    "final_results_match": int(summary_df['final_result_match'].sum()),
    "perfect_geometry_match": int((summary_df['positions_diff_norm'] == 0.0).sum()),
    "tool_names_match": int(summary_df['tool_names_match'].sum()),
    "tool_call_count_match": int(summary_df['num_calls_match'].sum()),
    "avg_llm_tool_calls": round(summary_df['llm_tool_calls'].mean(), 2),
    "std_llm_tool_calls": round(summary_df['llm_tool_calls'].std(), 2),
    "avg_manual_tool_calls": round(summary_df['manual_tool_calls'].mean(), 2),
    "tool_argument_match": matching_tool_calls,
    "tool_argument_total": total_tool_calls,
    "tool_argument_mismatch": mismatched_tool_calls,
    "malformed_entries": len(malformed_entries)
}

print("\n=== JSON STATISTICS ===")
print(json.dumps(stats, indent=2))

summary_df.to_csv("test.csv", index=False)
