"""Ground-truth dataset loading and validation for ChemGraph evaluation."""

import json
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, Field

from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Path to the bundled default ground-truth dataset.
_DEFAULT_DATASET = Path(__file__).parent / "data" / "ground_truth.json"


def default_dataset_path() -> str:
    """Return the absolute path to the bundled default ground-truth dataset.

    The dataset ships with the ``chemgraph`` package under
    ``chemgraph/eval/data/ground_truth.json`` and contains 14
    evaluation queries covering single-tool, multi-step, and
    reaction-energy calculations.

    Returns
    -------
    str
        Absolute path to the default ``ground_truth.json``.
    """
    return str(_DEFAULT_DATASET.resolve())


class GroundTruthItem(BaseModel):
    """A single evaluation query with its expected tool-call sequence"""

    id: str = Field(description="Unique identifier for the query.")
    query: str = Field(description="The natural-language query to send to the agent.")
    expected_tool_calls: list = Field(
        description="Ordered list of expected tool-call dicts."
    )
    expected_result: Any = Field(
        default="",
        description="Optional expected final result (string or list of step dicts).",
    )
    category: str = Field(
        default="",
        description="Optional category / experiment tag.",
    )


def load_dataset(path: str) -> List[GroundTruthItem]:
    """Load a ground-truth dataset from a JSON file.

    Automatically detects the two formats used in ChemGraph:

    1. **List format** -- a JSON array of ``{id, query, answer}`` objects
       (used by the bundled ``data/ground_truth.json``).
    2. **Dict format** -- a JSON object keyed by query/name, each
       containing ``manual_workflow`` with ``tool_calls`` and ``result``
       (used by legacy ``run_manual/`` baselines).

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    list[GroundTruthItem]
        Validated list of ground-truth items.

    Raises
    ------
    ValueError
        If the file cannot be parsed into either known format.
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: List[GroundTruthItem] = []

    if isinstance(raw, list):
        # List format: [{id, query, answer: {tool_calls, result}}, ...]
        for idx, entry in enumerate(raw):
            answer = entry.get("answer", {})
            items.append(
                GroundTruthItem(
                    id=str(entry.get("id", idx)),
                    query=entry["query"],
                    expected_tool_calls=answer.get("tool_calls", []),
                    expected_result=answer.get("result", ""),
                )
            )
    elif isinstance(raw, dict):
        # Dict format: {name: {manual_workflow: {tool_calls, result}, ...}, ...}
        for idx, (name, data) in enumerate(raw.items()):
            workflow = data.get("manual_workflow", data.get("llm_workflow", {}))
            tool_calls = workflow.get("tool_calls", [])
            result = workflow.get("result", "")

            # For dict format, the key is typically the molecule/reaction
            # name which also serves as the query.  If a "query" field
            # exists at the top level, prefer it.
            query = data.get("query", name)

            items.append(
                GroundTruthItem(
                    id=str(idx),
                    query=query,
                    expected_tool_calls=tool_calls,
                    expected_result=result if result else "",
                    category=name,
                )
            )
    else:
        raise ValueError(
            f"Unrecognised dataset format in {path}. Expected a JSON list or dict."
        )

    logger.info(f"Loaded {len(items)} ground-truth items from {path}")
    return items
