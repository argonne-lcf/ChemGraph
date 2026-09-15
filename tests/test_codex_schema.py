from copy import deepcopy

import pytest

from chemgraph.models._codex_schema import argument_schema
from chemgraph.models.codex import _decision_schema, _tool_argument_schemas


def _parameters(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required)}


def test_optional_shapes_preserve_omission_null_defaults_and_constraints():
    parameters = _parameters({
        "path": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "default": 100},
        "offset": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}], "default": None},
    }, ["path"])
    original = deepcopy(parameters)
    variants = argument_schema(parameters)["anyOf"]
    assert {tuple(branch["properties"]) for branch in variants} == {
        ("path",), ("path", "limit"), ("path", "offset"), ("path", "limit", "offset"),
    }
    for branch in variants:
        assert branch["required"] == list(branch["properties"])
        assert branch["additionalProperties"] is False
        if "limit" in branch["properties"]:
            assert branch["properties"]["limit"] == {"type": "integer", "minimum": 1}
        if "offset" in branch["properties"]:
            assert branch["properties"]["offset"] == {
                "anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}],
            }
    assert parameters == original


@pytest.mark.parametrize("field", [
    {"type": "object", "additionalProperties": True},
    {"type": "array", "items": {"type": "string"}},
    {"$ref": "#/$defs/Input"},
    {"anyOf": [{"type": "number"}, {"type": "string"}]},
    {"type": "string", "minLength": 3},
])
def test_complex_fields_use_compatibility_encoding(field):
    assert argument_schema(_parameters({"value": field}, ["value"])) is None


@pytest.mark.parametrize("extras", [
    {"additionalProperties": True}, {"additionalProperties": {"type": "string"}},
    {"allOf": [{"required": ["name"]}]}, {"$defs": {}},
])
def test_complex_objects_use_compatibility_encoding(extras):
    parameters = {**_parameters({"name": {"type": "string"}}), **extras}
    assert argument_schema(parameters) is None


def test_scalar_constraints_and_zero_argument_tools():
    properties = {
        "mode": {"type": "string", "enum": ["optimize", "vibrations"]},
        "name": {"type": "string", "pattern": "^[a-z]+$"},
        "steps": {"type": "integer", "exclusiveMinimum": 0},
        "threshold": {"type": "number", "maximum": 1, "multipleOf": 0.1},
        "enabled": {"type": "boolean"},
    }
    assert argument_schema(_parameters(properties, properties))["properties"] == properties
    assert argument_schema(_parameters({})) == {
        "type": "object", "properties": {}, "required": [], "additionalProperties": False,
    }


def _tool(name, parameters):
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


def test_optional_expansion_and_total_schema_size_are_bounded():
    five_optional = _parameters({f"field{i}": {"type": "string"} for i in range(5)})
    assert len(argument_schema(five_optional)["anyOf"]) == 32
    six_optional = _parameters({f"field{i}": {"type": "string"} for i in range(6)})
    assert argument_schema(six_optional) is None
    tools = [_tool(f"tool{i}", five_optional) for i in range(70)]
    schemas = _tool_argument_schemas(tools, None, True)
    assert schemas["tool0"] is not None
    assert schemas["tool69"] is None
    assert schemas == _tool_argument_schemas(tools, None, True)


@pytest.mark.parametrize("field", [
    {"type": "string", "enum": [str(i) for i in range(1001)]},
    {"type": "string", "enum": [f"{i}:" + "x" * 61 for i in range(251)]},
])
def test_large_enums_use_compatibility_encoding(field):
    tools = [_tool("large", _parameters({"value": field}, ["value"]))]
    assert _tool_argument_schemas(tools, None, True) == {"large": None}


def test_oversized_tool_names_fail_before_sdk_request():
    with pytest.raises(ValueError, match="bind fewer tools"):
        _tool_argument_schemas([_tool("x" * 120001, _parameters({}))], None, True)


def test_output_schema_enforces_forced_and_disabled_tools():
    schemas = {"first": argument_schema(_parameters({})), "second": None}
    forced = _decision_schema(schemas, "first", False)["properties"]["tool_calls"]
    assert forced["minItems"] == forced["maxItems"] == 1
    assert [branch["properties"]["name"]["enum"] for branch in forced["items"]["anyOf"]] == [["first"]]
    for choice, tools in [("none", schemas), (None, {})]:
        calls = _decision_schema(tools, choice, False)["properties"]["tool_calls"]
        assert calls["maxItems"] == 0
