"""LLM-as-judge evaluator for ChemGraph answer and tool-call correctness.

Compares the agent's tool-call sequence and final answer against the
ground-truth tool calls and final result using a binary scoring scheme
(1 = correct, 0 = wrong).  The judge receives the original query,
expected and actual tool calls, and expected and actual final results.
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from chemgraph.models.loader import load_chat_model
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema for judge response
# ---------------------------------------------------------------------------


class JudgeScore(BaseModel):
    """Binary grading output from the LLM judge.

    The judge decides whether the agent's answer is correct (1) or
    wrong (0) and provides a brief rationale.
    """

    score: int = Field(
        ..., ge=0, le=1, description="1 if the agent's answer is correct, 0 if wrong."
    )
    rationale: str = Field(default="", description="Brief justification for the score.")


# ---------------------------------------------------------------------------
# Rubric prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a computational chemistry AI agent called ChemGraph.

ChemGraph is an agentic framework that automates molecular simulations. Given a \
natural-language query it selects and calls chemistry tools (molecule lookup, \
structure generation, ASE simulations, calculators) to produce a result.

Your job is to decide whether the agent's answer is **correct** or **wrong** \
by comparing it to the expected answer (ground truth). You evaluate BOTH the \
tool-call sequence the agent used AND its final result.

## Rules

### Final Result
- The agent's answer is **correct (1)** if it contains the key results from \
the expected answer (numerical values, units, chemical properties, SMILES, etc.).
- Numeric values must match within **5% relative tolerance**.
- Minor formatting differences, extra explanation, rounding to fewer decimal \
places, or different phrasing are acceptable as long as the core result is present.
- Minor errors in file path and file name are acceptable as long as the expected output file is produced.
- Additional information reported is acceptable as long as the key expected results are present and correct.
- Missing tool calls is acceptable as long as the final answer is correct and the logical dependency chain is preserved.

### Tool Calls
- The agent should have called the **correct tools** (e.g. molecule_name_to_smiles, \
smiles_to_coordinate_file, run_ase, calculator, extract_output_json).
- **Key arguments** must match: calculator type (e.g. mace_mp vs TBLite), driver \
type (energy, opt, vib, thermo, dipole), SMILES strings, molecule names, \
temperature, and method (e.g. GFN2-xTB).
- Minor differences in tool call **order** are acceptable as long as the logical \
dependency chain is preserved (e.g. lookup before structure generation before simulation).
- Differences in **optional or default parameters** (fmax, steps, device, etc.) are acceptable.
- Missing or extra tool calls that do not affect the correctness of the final \
result are acceptable (e.g. an extra informational lookup).

### Overall Verdict
- The agent's answer is **correct (1)** only if BOTH the tool calls are \
substantially correct AND the final result matches the expected answer.
- The agent's answer is **wrong (0)** if it is missing key results, contains \
incorrect values, used the wrong tools or wrong key arguments, or failed to \
produce a meaningful answer.

You MUST respond with ONLY a valid JSON object matching the schema below. \
Do not include any text outside the JSON object.

```json
{{"score": 0, "rationale": "<brief justification>"}}
```\
"""


JUDGE_USER_TEMPLATE = """\
## Query
{query}

## Expected Tool Calls (Ground Truth)
```json
{expected_tool_calls}
```

## Expected Answer (Ground Truth)
```json
{expected_answer}
```

## Agent's Tool Calls
```json
{agent_tool_calls}
```

## Agent's Answer
{agent_answer}

---

Is the agent's answer correct? Consider both the tool calls and the final \
result. Respond with ONLY a JSON object: \
{{"score": <0 or 1>, "rationale": "<brief justification>"}}. \
Do not wrap in markdown fences or add any text outside the JSON.\
"""


# ---------------------------------------------------------------------------
# Judge model loader
# ---------------------------------------------------------------------------


def load_judge_model(
    model_name: str,
    base_url: Optional[str] = None,
    argo_user: Optional[str] = None,
):
    """Load an LLM for use as a judge.

    Delegates to the shared :func:`chemgraph.models.loader.load_chat_model`
    utility with ``temperature=0`` for deterministic grading.

    Parameters
    ----------
    model_name : str
        Model name from any supported provider.
    base_url : str, optional
        Provider base URL (resolved from config.toml when available).
    argo_user : str, optional
        Argo user identifier (resolved from config.toml when available).

    Returns
    -------
    BaseChatModel
        A LangChain chat model instance.
    """
    return load_chat_model(
        model_name=model_name,
        temperature=0.0,
        base_url=base_url,
        argo_user=argo_user,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_final_result(expected_result: Any) -> Any:
    """Extract the final tool-call output from a ground-truth result.

    The ground-truth ``result`` field is typically a list of step dicts,
    each shaped ``{"tool": ..., "input": ..., "output": ...}``.  This
    function returns just the ``output`` of the **last** step, which
    represents the final answer the agent should produce.

    Parameters
    ----------
    expected_result : Any
        The ground-truth result -- usually a list of step dicts, but may
        also be a plain string or other value.

    Returns
    -------
    Any
        The final result value suitable for comparison.
    """
    if isinstance(expected_result, list):
        if len(expected_result) == 0:
            return ""
        last_step = expected_result[-1]
        if isinstance(last_step, dict) and "output" in last_step:
            return last_step["output"]
        return last_step
    if expected_result is None:
        return ""
    return expected_result


def _format_expected_answer(expected_result: Any) -> str:
    """Format the ground-truth final result as a JSON string for the judge.

    Parameters
    ----------
    expected_result : Any
        The full ground-truth result (list of step dicts, string, etc.).

    Returns
    -------
    str
        JSON-formatted string of the final result only.
    """
    final = _extract_final_result(expected_result)
    if isinstance(final, str):
        return final
    return json.dumps(final, indent=2, default=str)


def _format_tool_calls(tool_calls: Any) -> str:
    """Format a list of tool-call dicts as a JSON string for the judge.

    Parameters
    ----------
    tool_calls : Any
        List of tool-call dicts (e.g. ``[{"run_ase": {"params": ...}}]``),
        or ``None`` / empty list.

    Returns
    -------
    str
        JSON-formatted string, or ``"(no tool calls)"`` when absent.
    """
    if not tool_calls:
        return "(no tool calls)"
    return json.dumps(tool_calls, indent=2, default=str)


def _parse_judge_response(content: str) -> JudgeScore:
    """Parse the judge LLM's response into a JudgeScore.

    Handles both direct JSON and markdown-fenced JSON responses.

    Parameters
    ----------
    content : str
        Raw response content from the judge LLM.

    Returns
    -------
    JudgeScore
        Validated score object.

    Raises
    ------
    ValueError
        If the response cannot be parsed into a valid JudgeScore.
    """
    text = content.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge response is not valid JSON: {e}\nRaw: {content[:500]}")

    return JudgeScore(**data)


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------


async def judge_single_query(
    judge_llm,
    query: str,
    expected_result: Any,
    model_result: Any,
    expected_tool_calls: Optional[List] = None,
    model_tool_calls: Optional[List] = None,
) -> Dict[str, Any]:
    """Have the judge LLM evaluate a single query's answer correctness.

    Compares the agent's tool-call sequence and final answer against the
    ground-truth tool calls and final result using binary scoring
    (1 = correct, 0 = wrong).

    Parameters
    ----------
    judge_llm : BaseChatModel
        The judge LLM instance.
    query : str
        The original natural-language query.
    expected_result : Any
        Ground-truth expected result (list of step dicts or string).
        The final tool-call output is extracted automatically.
    model_result : Any
        Final answer produced by the model under test.
    expected_tool_calls : list, optional
        Ground-truth expected tool-call sequence from the dataset.
    model_tool_calls : list, optional
        Actual tool calls made by the agent during execution.

    Returns
    -------
    dict
        A dict with keys:
        - ``"score"``: int (1 = correct, 0 = wrong)
        - ``"rationale"``: str
        - ``"parse_error"``: str or None if parsing failed
    """
    expected_answer_str = _format_expected_answer(expected_result)
    agent_answer_str = str(model_result) if model_result else "(no answer produced)"
    expected_tool_calls_str = _format_tool_calls(expected_tool_calls)
    agent_tool_calls_str = _format_tool_calls(model_tool_calls)

    user_message = JUDGE_USER_TEMPLATE.format(
        query=query,
        expected_tool_calls=expected_tool_calls_str,
        expected_answer=expected_answer_str,
        agent_tool_calls=agent_tool_calls_str,
        agent_answer=agent_answer_str,
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        response = await judge_llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        score = _parse_judge_response(content)

        return {
            "score": score.score,
            "rationale": score.rationale,
            "parse_error": None,
        }

    except Exception as e:
        logger.warning(f"Judge evaluation failed: {e}")
        return {
            "score": 0,
            "rationale": f"Judge evaluation failed: {e}",
            "parse_error": str(e),
        }


# ---------------------------------------------------------------------------
# Aggregate judge results
# ---------------------------------------------------------------------------


def aggregate_judge_results(per_query_judge_results: List[dict]) -> dict:
    """Compute aggregate statistics over binary judge scores.

    Parameters
    ----------
    per_query_judge_results : list[dict]
        Output of :func:`judge_single_query` for each query.

    Returns
    -------
    dict
        Aggregate judge metrics:
        - ``n_queries``: total queries evaluated
        - ``n_correct``: number scored as correct (1)
        - ``accuracy``: fraction correct
        - ``n_parse_errors``: number of judge failures
    """
    n = len(per_query_judge_results)
    if n == 0:
        return {
            "n_queries": 0,
            "n_correct": 0,
            "accuracy": 0.0,
            "n_parse_errors": 0,
        }

    valid = [r for r in per_query_judge_results if r.get("parse_error") is None]
    n_valid = len(valid)
    n_errors = n - n_valid

    if n_valid == 0:
        return {
            "n_queries": n,
            "n_correct": 0,
            "accuracy": 0.0,
            "n_parse_errors": n_errors,
        }

    n_correct = sum(1 for r in valid if r.get("score", 0) == 1)

    return {
        "n_queries": n,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_valid, 4),
        "n_parse_errors": n_errors,
    }
