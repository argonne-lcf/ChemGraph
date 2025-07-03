"""Module for quick LLM evaluations"""

from deepdiff import DeepDiff
from chemgraph.models.ase_input import ASEInputSchema


def remove_ignored_fields(obj, ignored_keys=("cell", "pbc")):
    """Remove ignored fields from object

    Args:
        obj (_type_): _description_
        ignored_keys (tuple, optional): _description_. Defaults to ("cell", "pbc").

    Returns:
        _type_: _description_
    """
    if isinstance(obj, dict):
        return {
            k: remove_ignored_fields(v, ignored_keys)
            for k, v in obj.items()
            if k not in ignored_keys
        }
    elif isinstance(obj, list):
        return [remove_ignored_fields(item, ignored_keys) for item in obj]
    else:
        return obj


def apply_defaults(args: dict, schema: dict) -> dict:
    """
    Recursively fills missing fields with default values based on a JSON-like schema.
    Handles nested objects and anyOf/default combinations.
    """
    if not isinstance(args, dict):
        return args  # Only process dicts

    args_with_defaults = dict(args)  # shallow copy

    properties = schema.get("properties", {})
    for key, prop_schema in properties.items():
        # Skip if already set
        if key in args_with_defaults:
            # Recurse into nested object
            if isinstance(args_with_defaults[key], dict) and prop_schema.get("type") == "object":
                args_with_defaults[key] = apply_defaults(args_with_defaults[key], prop_schema)
            continue

        # Handle default at top level
        if "default" in prop_schema:
            args_with_defaults[key] = prop_schema["default"]
            continue

        # Handle nested default inside anyOf (take first subschema with default)
        if "anyOf" in prop_schema:
            for option in prop_schema["anyOf"]:
                if isinstance(option, dict) and "default" in option:
                    args_with_defaults[key] = option["default"]
                    break

        # Handle nested object with defaults even if not explicitly present
        if prop_schema.get("type") == "object" and "properties" in prop_schema:
            args_with_defaults[key] = apply_defaults({}, prop_schema)

    return args_with_defaults


def lowercase_dict(obj):
    """Recursively lowercases string keys and string values in a dict/list structure."""
    if isinstance(obj, dict):
        return {(k.lower() if isinstance(k, str) else k): lowercase_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [lowercase_dict(i) for i in obj]
    elif isinstance(obj, str):
        return obj.lower()
    else:
        return obj


def single_function_checker(
    func_description: dict,
    model_output: dict,
    answer: dict,
    ignore_fields=None,
) -> dict:
    """
    Compare model tool call output with expected answer, validating with function schema.

    Args:
        func_description (dict): Tool function schema (JSON schema style)
        model_output (dict): LLM's tool call output (with "arguments" and optionally "result")
        answer (dict): Reference tool call output

    Returns:
    """

    if ignore_fields is None:
        ignore_fields = ["cell", "pbc"]

    # Extract schema and values
    schema = func_description.get("parameters", {})

    result = {"valid": False, "error": ""}

    tool_name_model, model_args_raw = next(iter(model_output.items()))
    tool_name_answer, answer_args_raw = next(iter(answer.items()))

    if tool_name_model != tool_name_answer:
        error = "different tool_name"
        result = {"valid": False, "error": error}
        return result

    # Have a special case for run_ase due to complex input schema
    if tool_name_model == "run_ase":
        try:
            model_args = ASEInputSchema(**model_args_raw["params"]).model_dump()
            answer_args = ASEInputSchema(**answer_args_raw["params"]).model_dump()
        except Exception as e:
            result = {"valid": False, "error": e}
            return result
        # Apply lower case to both sides
        model_args = lowercase_dict(model_args)
        answer_args = lowercase_dict(answer_args)

        # Remove ignored fields
        model_args = remove_ignored_fields(model_args, ignore_fields)
        answer_args = remove_ignored_fields(answer_args, ignore_fields)

    else:
        # Apply lower case to both sides
        model_args_lower = lowercase_dict(model_args_raw)
        answer_args_lower = lowercase_dict(answer_args_raw)

        # Apply defaults to both sides
        model_args_full = apply_defaults(model_args_lower, schema)
        answer_args_full = apply_defaults(answer_args_lower, schema)

        # Remove ignored fields
        model_args = remove_ignored_fields(model_args_full, ignore_fields)
        answer_args = remove_ignored_fields(answer_args_full, ignore_fields)

    diff = DeepDiff(
        model_args,
        answer_args,
        significant_digits=3,  # Controls float tolerance
        ignore_order=True,  # Ignores order in dicts and lists
    )

    if not diff:
        result = {"valid": True, "error": ""}
    else:
        result = {"valid": False, "error": diff.to_dict()}

    return result


def find_description(func_descriptions: list, func_name: str) -> dict:
    """Find the function description by name

    Args:
        func_descriptions (list): list of function descriptions
        func_name (str): function name

    Returns:
        dict: dictionary of the
    """
    if isinstance(func_descriptions, list):
        for func_description in func_descriptions:
            if func_description["name"] == func_name:
                return func_description
        return None
    else:
        return func_descriptions


def multi_function_checker_with_order(
    func_descriptions: dict,
    model_outputs: list,
    answers: list,
    ignore_fields=None,
) -> dict:
    """Evaluate multiple function calls.

    Args:
        func_description (dict): _description_
        model_output (list): _description_
        answer (list): _description_
        ignore_fields (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """
    if ignore_fields is None:
        ignore_fields = ["cell", "pbc"]

    # Initialize result
    result = {
        "valid": True,
        "error": "",
        "n_true_toolcalls": len(answers),
        "n_llm_tool_calls": len(model_outputs),
        "acc_n_toolcalls": 0,
        "args_differences": {},
    }

    if len(model_outputs) != len(answers):
        result['error'] = "Different number of tool calls"
        result['valid'] = False
        return result

    for model_output, answer in zip(model_outputs, answers):
        tool_name_model, model_args_raw = next(iter(model_output.items()))

        # Get function description
        func_description = find_description(
            func_descriptions=func_descriptions,
            func_name=tool_name_model,
        )
        if func_description is None:
            result["error"] += f"Function {tool_name_model} is not in the given functions.\n"
            continue
        else:
            result_single = single_function_checker(
                func_description=func_description,
                model_output=model_output,
                answer=answer,
            )

            if result_single["valid"] is True:
                result["acc_n_toolcalls"] += 1
            else:
                result["args_differences"][tool_name_model] = result_single["error"]

    return result


def multi_function_checker_without_order(
    func_descriptions: dict,
    model_outputs: list,
    answers: list,
    ignore_fields=None,
) -> dict:
    """Evaluate multiple function calls.

    Args:
        func_description (dict): _description_
        model_output (list): _description_
        answer (list): _description_
        ignore_fields (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """
    if ignore_fields is None:
        ignore_fields = ["cell", "pbc"]

    # Initialize result
    result = {
        "valid": True,
        "error": "",
        "n_true_toolcalls": len(answers),
        "n_llm_tool_calls": len(model_outputs),
        "acc_n_toolcalls": 0,
        "answers_without_match": [],
    }

    for model_id, model_output in enumerate(model_outputs):
        for answer_id, answer in enumerate(answers):
            tool_name_model, model_args_raw = next(iter(model_output.items()))

            # Get function description
            func_description = find_description(
                func_descriptions=func_descriptions,
                func_name=tool_name_model,
            )
            if func_description is None:
                result["error"] += f"Function {tool_name_model} is not in the given functions.\n"
                continue
            else:
                result_single = single_function_checker(
                    func_description=func_description,
                    model_output=model_output,
                    answer=answer,
                )

                if result_single["valid"] is True:
                    result["acc_n_toolcalls"] += 1

                    # Remove accurate answer from future comparison after a match.
                    answers.remove(answer)
                else:
                    continue
    if len(answers) != 0:
        for answer in answers:
            result["answers_without_match"].append(answer)
    return result
