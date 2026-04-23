"""Shared parsing utilities for extracting structured JSON from LLM output.

Both ``single_agent.py`` and ``multi_agent.py`` need to parse free-form
LLM text into Pydantic models (``ResponseFormatter``, ``PlannerResponse``).
This module centralises the common helpers so they stay in sync.
"""

import re

from chemgraph.schemas.agent_response import ResponseFormatter
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def extract_json_block(text: str) -> str | None:
    """Try to extract a JSON object from *text*.

    Handles markdown-fenced blocks (```json ... ```) and bare JSON objects.
    Returns the extracted string or *None* if nothing looks like JSON.
    """
    # Try markdown-fenced JSON first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Try bare top-level JSON object
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1)
    return None


def parse_response_formatter(
    raw_text: str,
) -> tuple[ResponseFormatter, str | None]:
    """Parse LLM output into a :class:`ResponseFormatter`.

    Attempts direct validation first, then tries to extract a JSON block
    from the text.  Falls back to an empty ``ResponseFormatter`` (all
    fields ``None``) so the pipeline never breaks -- the raw text is
    still available in the agent's message history.

    Returns
    -------
    tuple[ResponseFormatter, str | None]
        A tuple of ``(parsed_formatter, parse_error)``.  ``parse_error``
        is ``None`` on success, or a descriptive string when parsing
        failed and the empty fallback was used.
    """
    # 1. Direct validation
    try:
        return ResponseFormatter.model_validate_json(raw_text.strip()), None
    except Exception:
        pass

    # 2. Extract JSON block and retry
    extracted = extract_json_block(raw_text)
    if extracted:
        try:
            return ResponseFormatter.model_validate_json(extracted), None
        except Exception:
            pass

    # 3. Fallback: return empty ResponseFormatter (all fields None).
    error_msg = (
        "ResponseAgent: could not parse structured output; "
        "returning empty ResponseFormatter."
    )
    logger.warning(error_msg)
    return ResponseFormatter(), error_msg
