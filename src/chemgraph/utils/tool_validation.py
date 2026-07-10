"""Deterministic guardrails for single-agent tool calls.

This module intentionally covers a small, high-impact case: a follow-up query
for one molecule should not execute ``run_ase`` on a coordinate file produced
for a different molecule earlier in the conversation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import ToolMessage


_PROPERTY_PATTERNS = (
    "dipolemoment",
    "dipole moment",
    "dipole",
    "ir spectrum",
    "infrared spectrum",
    "vibrational frequencies",
    "vibrational frequency",
    "vibration",
    "energy",
    "enthalpy",
    "gibbs free energy",
)

_TARGET_STOP_WORDS = (
    " using ",
    " with ",
    " from ",
    " at ",
    " in ",
    " for ",
    "?",
    ".",
)


def normalize_tool_call(call: Any) -> dict[str, Any] | None:
    """Normalize LangChain/OpenAI tool-call payloads."""
    if not isinstance(call, dict):
        return None

    if "function" in call:
        function = call.get("function") or {}
        raw_args = function.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args
        return {
            "id": call.get("id"),
            "name": function.get("name"),
            "args": args if isinstance(args, dict) else {},
        }

    return {
        "id": call.get("id"),
        "name": call.get("name"),
        "args": call.get("args") if isinstance(call.get("args"), dict) else {},
    }


def message_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Return normalized tool calls from a message-like object."""
    if isinstance(message, dict):
        raw_calls = message.get("tool_calls")
        if not raw_calls:
            raw_calls = (message.get("additional_kwargs") or {}).get("tool_calls")
    else:
        raw_calls = getattr(message, "tool_calls", None)
        if not raw_calls:
            raw_calls = (
                getattr(message, "additional_kwargs", {}) or {}
            ).get("tool_calls")

    calls = [normalize_tool_call(call) for call in raw_calls or []]
    return [call for call in calls if call and call.get("name")]


def latest_user_query(messages: Iterable[Any]) -> str:
    """Extract the latest human/user query from a message sequence."""
    for message in reversed(list(messages or [])):
        role = _message_role(message)
        if role in {"human", "user"}:
            return _message_content(message)
    return ""


def validate_target_scoped_tool_calls(
    query: str,
    prior_messages: Iterable[Any],
    tool_calls: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Validate that molecule-property tool calls use current-target files."""
    calls = [normalize_tool_call(call) for call in tool_calls or []]
    calls = [call for call in calls if call and call.get("name")]
    target = infer_query_target(query)
    if not target:
        return _allowed()

    coordinate_species = _collect_coordinate_species(prior_messages)
    for call in calls:
        if call.get("name") != "run_ase":
            continue

        input_file = _run_ase_input_file(call.get("args") or {})
        if not input_file:
            continue

        species = _species_for_file(input_file, coordinate_species)
        if species and not _species_matches(species, target):
            return {
                "allowed": False,
                "blocked_tool": "run_ase",
                "blocked_tool_call_ids": [call.get("id")],
                "target": target,
                "observed_species": species,
                "input_file": str(input_file),
                "reason": (
                    f"The current query targets {target!r}, but run_ase was "
                    f"called with coordinate file {str(input_file)!r} for "
                    f"{species!r}."
                ),
                "repair_instruction": (
                    f"Resolve {target!r}, generate or reuse a coordinate file "
                    "for that current target, then call run_ase on that file. "
                    "Do not reuse coordinate artifacts from a previous molecule."
                ),
            }

    return _allowed()


def build_tool_validation_messages(
    tool_calls: Iterable[dict[str, Any]],
    validation: dict[str, Any],
) -> list[ToolMessage]:
    """Build ToolMessage feedback for blocked tool calls."""
    if validation.get("allowed", True):
        return []

    blocked_ids = {
        str(call_id)
        for call_id in validation.get("blocked_tool_call_ids", [])
        if call_id
    }
    messages: list[ToolMessage] = []
    content = (
        "Tool call blocked by deterministic validation.\n"
        f"Reason: {validation.get('reason', 'not specified')}\n"
        f"Repair: {validation.get('repair_instruction', 'Retry with valid inputs.')}"
    )

    for call in tool_calls or []:
        normalized = normalize_tool_call(call)
        if not normalized:
            continue
        call_id = normalized.get("id")
        if blocked_ids and str(call_id) not in blocked_ids:
            continue
        if call_id:
            messages.append(ToolMessage(content=content, tool_call_id=str(call_id)))

    return messages


def infer_query_target(query: str) -> str | None:
    """Infer a single molecule target from common property-query phrasing."""
    text = " ".join(str(query or "").strip().split())
    if not text:
        return None
    lowered = text.lower()
    if "reaction" in lowered or "combustion" in lowered:
        return None

    for prop in _PROPERTY_PATTERNS:
        match = re.search(rf"\b{re.escape(prop)}\b", lowered)
        if not match:
            continue

        before = text[: match.start()].strip(" :,-")
        after = text[match.end() :].strip(" :,-")

        of_match = re.search(r"\bof\s+(.+)$", before, flags=re.IGNORECASE)
        if of_match:
            target = _clean_target(of_match.group(1))
            if target:
                return target

        after_of = re.match(r"^(?:of|for)\s+(.+)$", after, flags=re.IGNORECASE)
        if after_of:
            target = _clean_target(after_of.group(1))
            if target:
                return target

        target = _clean_target(before)
        if target:
            return target

    return None


def _allowed() -> dict[str, Any]:
    return {
        "allowed": True,
        "blocked_tool": None,
        "blocked_tool_call_ids": [],
        "reason": "",
        "repair_instruction": "",
    }


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").lower()
    role = getattr(message, "role", None) or getattr(message, "type", None)
    if role:
        return str(role).lower()
    class_name = message.__class__.__name__.lower()
    if "human" in class_name:
        return "human"
    if "ai" in class_name:
        return "ai"
    if "tool" in class_name:
        return "tool"
    return ""


def _message_name(message: Any) -> str | None:
    if isinstance(message, dict):
        name = message.get("name")
    else:
        name = getattr(message, "name", None)
    return str(name) if name else None


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return "" if content is None else str(content)


def _json_content(message: Any) -> dict[str, Any]:
    content = _message_content(message).strip()
    if not content:
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _collect_coordinate_species(messages: Iterable[Any]) -> dict[str, str]:
    coordinate_species: dict[str, str] = {}
    last_species: str | None = None

    for message in messages or []:
        name = _message_name(message)
        payload = _json_content(message)

        if name == "molecule_name_to_smiles":
            last_species = _payload_species(payload) or last_species
            continue

        if name != "smiles_to_coordinate_file":
            continue

        file_path = _payload_coordinate_file(payload)
        if not file_path:
            continue

        species = _payload_species(payload) or _species_from_file(file_path)
        if not species or _is_generic_species(species):
            species = last_species or species
        for key in _file_keys(file_path):
            coordinate_species[key] = species

    return coordinate_species


def _payload_species(payload: dict[str, Any]) -> str | None:
    for key in (
        "name",
        "molecule",
        "molecule_name",
        "input_name",
        "resolved_name",
        "representative_of",
    ):
        value = payload.get(key)
        if value:
            return str(value)
    return None


def _payload_coordinate_file(payload: dict[str, Any]) -> str | None:
    for key in (
        "path",
        "file_path",
        "input_file",
        "input_structure_file",
        "xyz_file",
        "filename",
        "output_file",
    ):
        value = payload.get(key)
        if value:
            return str(value)
    return None


def _run_ase_input_file(args: dict[str, Any]) -> str | None:
    params = args.get("params") if isinstance(args.get("params"), dict) else args
    for key in ("input_structure_file", "input_file", "file_path", "xyz_file"):
        value = params.get(key)
        if value:
            return str(value)
    return None


def _species_for_file(
    file_path: str,
    coordinate_species: dict[str, str],
) -> str | None:
    for key in _file_keys(file_path):
        if key in coordinate_species:
            return coordinate_species[key]
    return _species_from_file(file_path)


def _file_keys(file_path: str) -> list[str]:
    raw = str(file_path)
    path = Path(raw)
    keys = {raw.lower(), path.name.lower(), path.stem.lower()}
    return [key for key in keys if key]


def _species_from_file(file_path: str) -> str | None:
    stem = Path(str(file_path)).stem
    stem = re.sub(r"[_-](coord|coords|coordinate|coordinates|opt|optimized)$", "", stem)
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem or None


def _clean_target(value: str) -> str | None:
    target = str(value or "").strip(" :,-")
    target = re.sub(
        r"^(what is|report|calculate|compute|the|a|an)\s+",
        "",
        target,
        flags=re.IGNORECASE,
    )
    for stop in _TARGET_STOP_WORDS:
        index = target.lower().find(stop.strip().lower()) if stop.strip() else -1
        if index > 0:
            target = target[:index]
    target = target.strip(" :,-")
    return target or None


def _normalize_species(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _species_matches(species: str, target: str) -> bool:
    species_key = _normalize_species(species)
    target_key = _normalize_species(target)
    return bool(species_key and target_key and species_key == target_key)


def _is_generic_species(species: str) -> bool:
    return _normalize_species(species) in {
        "molecule",
        "structure",
        "geometry",
        "coords",
    }
