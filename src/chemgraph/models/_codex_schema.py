"""Small output-schema adapter for tools with flat, named arguments."""

from copy import deepcopy
from itertools import combinations


_ANNOTATIONS = {"title", "description", "default", "examples", "$schema", "$comment"}
_SCALAR_KEYWORDS = {
    "string": {"pattern"},
    "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
    "number": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
    "boolean": set(),
    "null": set(),
}


def _scalar_schema(schema):
    if not isinstance(schema, dict):
        return None
    result = {k: deepcopy(v) for k, v in schema.items() if k not in _ANNOTATIONS}
    if set(result) == {"anyOf"}:
        branches = [_scalar_schema(branch) for branch in result["anyOf"]]
        if len(branches) == 2 and {"type": "null"} in branches and None not in branches:
            return {"anyOf": branches}
        return None
    kind = result.get("type")
    if not isinstance(kind, str) or kind not in _SCALAR_KEYWORDS:
        return None
    if set(result) - {"type", "enum"} - _SCALAR_KEYWORDS[kind]:
        return None
    return result


def argument_schema(parameters):
    """Return a strict flat schema, or None to retain JSON-string arguments.

    Named tool schemas commonly omit additionalProperties. Close those objects
    for generation; explicitly open objects keep the compatibility encoding.
    Optional fields use alternatives so omitted values never become null/defaults.
    """
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        return None
    if set(parameters) - _ANNOTATIONS - {"type", "properties", "required", "additionalProperties"}:
        return None
    if parameters.get("additionalProperties", False) is not False:
        return None
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return None
    required = set(parameters.get("required", []))
    if required - properties.keys():
        return None
    optional = [name for name in properties if name not in required]
    if len(optional) > 5:  # At most 32 alternatives per tool.
        return None
    compiled = {name: _scalar_schema(value) for name, value in properties.items()}
    if None in compiled.values():
        return None
    choices = []
    for size in range(len(optional) + 1):
        for subset in combinations(optional, size):
            names = required | set(subset)
            fields = {name: value for name, value in compiled.items() if name in names}
            choices.append({
                "type": "object", "properties": fields,
                "required": list(fields), "additionalProperties": False,
            })
    return choices[0] if len(choices) == 1 else {"anyOf": choices}


def within_limits(schema):
    """Check provider size limits on the complete, generated output schema."""
    properties = strings = enums = 0
    pending = [(schema, 0)]
    while pending:
        node, depth = pending.pop()
        depth += node.get("type") in ("object", "array")
        fields = node.get("properties", {})
        properties += len(fields)
        strings += sum(len(name) for name in fields)
        values = node.get("enum", [])
        enums += len(values)
        enum_strings = sum(len(value) for value in values if isinstance(value, str))
        strings += enum_strings
        if (depth > 10 or properties > 5000 or strings > 120000 or enums > 1000
                or (len(values) > 250 and enum_strings > 15000)):
            return False
        children = list(fields.values()) + node.get("anyOf", [])
        if "items" in node:
            children.append(node["items"])
        pending.extend((child, depth) for child in children)
    return True
