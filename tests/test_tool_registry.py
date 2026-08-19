from types import SimpleNamespace

import pytest
from langchain_core.tools import BaseTool, tool

from chemgraph.registry import (
    DuplicateRegistryEntryError,
    RegistryUnavailableError,
    RuntimeRequirement,
    ToolRegistry,
    ToolSpec,
    UnknownRegistryEntryError,
)


EXPECTED_TOOLS = (
    "extract_output_json",
    "file_to_atomsdata",
    "save_atomsdata_to_file",
    "get_symmetry_number",
    "is_linear_molecule",
    "run_ase",
    "molecule_name_to_smiles",
    "smiles_to_atomsdata",
    "smiles_to_coordinate_file",
    "calculator",
    "ask_human",
    "python_repl",
    "run_docking",
    "run_graspa",
    "load_document",
    "query_knowledge_base",
    "generate_html",
    "run_xanes",
    "fetch_xanes_data",
    "plot_xanes_data",
)


@tool
def custom_tool(value: str) -> str:
    """Return a value unchanged."""
    return value


def test_builtin_registry_contains_only_llm_tools():
    registry = ToolRegistry()

    assert registry.names() == EXPECTED_TOOLS
    assert "main_agent" not in registry.names()
    assert all("mcp" not in spec.tags for spec in registry.specs())


def test_every_builtin_resolves_to_matching_base_tool():
    registry = ToolRegistry()

    resolved = registry.resolve()

    assert len(resolved) == len(EXPECTED_TOOLS)
    assert all(isinstance(item, BaseTool) for item in resolved)
    assert tuple(item.name for item in resolved) == EXPECTED_TOOLS


def test_registry_filters_by_all_requested_tags():
    registry = ToolRegistry()

    assert registry.names(tags={"ase", "analysis"}) == (
        "get_symmetry_number",
        "is_linear_molecule",
    )
    assert [tool.name for tool in registry.resolve(tags={"docking"})] == [
        "run_docking"
    ]


def test_lazy_spec_is_not_imported_until_get(monkeypatch):
    calls = []

    def fake_import(name):
        calls.append(name)
        return SimpleNamespace(candidate=custom_tool)

    monkeypatch.setattr("chemgraph.registry.tools.importlib.import_module", fake_import)
    registry = ToolRegistry(
        specs=(
            ToolSpec(
                name="custom_tool",
                description="Custom tool",
                import_path="example.lazy:candidate",
            ),
        )
    )

    assert registry.names() == ("custom_tool",)
    assert calls == []
    assert registry.get("custom_tool") is custom_tool
    assert calls == ["example.lazy"]
    assert registry.get("custom_tool") is custom_tool
    assert calls == ["example.lazy"]


def test_register_eager_tool_and_reject_duplicate():
    registry = ToolRegistry(specs=())
    registry.register(custom_tool, tags={"custom"})

    assert registry.get("custom_tool") is custom_tool
    assert registry.names(tags={"custom"}) == ("custom_tool",)
    with pytest.raises(DuplicateRegistryEntryError, match="already registered"):
        registry.register(custom_tool)


def test_unknown_tool_is_clear():
    with pytest.raises(UnknownRegistryEntryError, match="Unknown tool"):
        ToolRegistry().get("main_agent")


def test_require_available_aggregates_runtime_issue(monkeypatch):
    monkeypatch.delenv("CHEMGRAPH_TEST_EXECUTABLE", raising=False)
    registry = ToolRegistry(
        specs=(
            ToolSpec(
                name="custom_tool",
                description="Custom tool",
                import_path="example.lazy:candidate",
                requirements=(
                    RuntimeRequirement(
                        "environment",
                        "CHEMGRAPH_TEST_EXECUTABLE",
                        "set it for this test tool",
                    ),
                ),
            ),
        )
    )

    status = registry.availability("custom_tool")
    assert status.available is False
    assert status.issues == (
        "missing environment variable 'CHEMGRAPH_TEST_EXECUTABLE' "
        "(set it for this test tool)",
    )
    with pytest.raises(RegistryUnavailableError, match="CHEMGRAPH_TEST_EXECUTABLE"):
        registry.get("custom_tool", require_available=True)
