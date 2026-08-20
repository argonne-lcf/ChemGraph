"""Registry for ChemGraph's in-process LangChain tools."""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Literal

from langchain_core.tools import BaseTool


class RegistryError(Exception):
    """Base class for registry errors."""


class UnknownRegistryEntryError(RegistryError, KeyError):
    """Raised when a registry name is unknown."""


class DuplicateRegistryEntryError(RegistryError, ValueError):
    """Raised when a name or alias is registered more than once."""


class RegistryLoadError(RegistryError, RuntimeError):
    """Raised when a lazy registry entry cannot be loaded."""


class RegistryUnavailableError(RegistryError, RuntimeError):
    """Raised when an entry's runtime requirements are unavailable."""

    def __init__(self, name: str, issues: Iterable[str]):
        self.name = name
        self.issues = tuple(issues)
        details = "; ".join(self.issues)
        super().__init__(f"Registry entry {name!r} is unavailable: {details}")


RequirementKind = Literal["module", "environment", "executable", "path"]


@dataclass(frozen=True, slots=True)
class RuntimeRequirement:
    """One import, environment, or executable needed at runtime."""

    kind: RequirementKind
    value: str
    hint: str = ""

    def issue(self) -> str | None:
        """Return an actionable issue when this requirement is not met."""
        available = False
        if self.kind == "module":
            try:
                available = importlib.util.find_spec(self.value) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                available = False
        elif self.kind == "environment":
            available = bool(os.environ.get(self.value))
        elif self.kind == "executable":
            available = shutil.which(self.value) is not None
        elif self.kind == "path":
            path = Path(self.value)
            available = path.is_file() and os.access(path, os.X_OK)

        if available:
            return None
        label = {
            "module": "Python module",
            "environment": "environment variable",
            "executable": "executable",
            "path": "executable path",
        }[self.kind]
        message = f"missing {label} {self.value!r}"
        return f"{message} ({self.hint})" if self.hint else message


@dataclass(frozen=True, slots=True)
class RegistryAvailability:
    """Availability result for a registry entry."""

    name: str
    issues: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """Whether all checked requirements are available."""
        return not self.issues


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Lazy metadata for one ChemGraph tool."""

    name: str
    description: str
    import_path: str | None
    tags: frozenset[str] = field(default_factory=frozenset)
    requirements: tuple[RuntimeRequirement, ...] = ()
    interactive: bool = False
    executes_code: bool = False


_GRASPA_EXECUTABLE = (
    "/lus/flare/projects/IQC/thang/soft/gRASPA/graspa-sycl/bin/sycl.out"
)


BUILTIN_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "extract_output_json",
        "Load a JSON result produced by an ASE calculation.",
        "chemgraph.tools.ase_tools:extract_output_json",
        frozenset({"ase", "files"}),
    ),
    ToolSpec(
        "file_to_atomsdata",
        "Read a structure file into ChemGraph's AtomsData schema.",
        "chemgraph.tools.ase_tools:file_to_atomsdata",
        frozenset({"ase", "files"}),
    ),
    ToolSpec(
        "save_atomsdata_to_file",
        "Write ChemGraph AtomsData to a structure file.",
        "chemgraph.tools.ase_tools:save_atomsdata_to_file",
        frozenset({"ase", "files"}),
    ),
    ToolSpec(
        "get_symmetry_number",
        "Determine a molecule's rotational symmetry number.",
        "chemgraph.tools.ase_tools:get_symmetry_number",
        frozenset({"ase", "analysis"}),
    ),
    ToolSpec(
        "is_linear_molecule",
        "Determine whether a molecule is linear.",
        "chemgraph.tools.ase_tools:is_linear_molecule",
        frozenset({"ase", "analysis"}),
    ),
    ToolSpec(
        "run_ase",
        "Run an ASE energy, optimization, vibration, or thermochemistry task.",
        "chemgraph.tools.ase_tools:run_ase",
        frozenset({"ase", "simulation"}),
    ),
    ToolSpec(
        "molecule_name_to_smiles",
        "Resolve a molecule name to a SMILES string.",
        "chemgraph.tools.cheminformatics_tools:molecule_name_to_smiles",
        frozenset({"cheminformatics", "structure"}),
    ),
    ToolSpec(
        "smiles_to_atomsdata",
        "Generate AtomsData coordinates from a SMILES string.",
        "chemgraph.tools.cheminformatics_tools:smiles_to_atomsdata",
        frozenset({"cheminformatics", "structure"}),
    ),
    ToolSpec(
        "smiles_to_coordinate_file",
        "Generate a coordinate file from a SMILES string.",
        "chemgraph.tools.cheminformatics_tools:smiles_to_coordinate_file",
        frozenset({"cheminformatics", "structure", "files"}),
    ),
    ToolSpec(
        "calculator",
        "Evaluate a mathematical expression safely.",
        "chemgraph.tools.generic_tools:calculator",
        frozenset({"generic"}),
    ),
    ToolSpec(
        "ask_human",
        "Pause graph execution to request human input.",
        "chemgraph.tools.generic_tools:ask_human",
        frozenset({"generic", "interactive"}),
        interactive=True,
    ),
    ToolSpec(
        "python_repl",
        "Execute Python code in a persistent in-process REPL.",
        "chemgraph.tools.generic_tools:repl_tool",
        frozenset({"generic", "python"}),
        executes_code=True,
    ),
    ToolSpec(
        "run_docking",
        "Dock a small molecule with AutoDock Vina.",
        "chemgraph.tools.docking_tools:run_docking",
        frozenset({"docking", "simulation"}),
        (
            RuntimeRequirement(
                "module", "vina", "install AutoDock Vina from conda-forge"
            ),
            RuntimeRequirement("module", "meeko", "install chemgraph[docking]"),
        ),
    ),
    ToolSpec(
        "run_graspa",
        "Run a gRASPA adsorption calculation.",
        "chemgraph.tools.graspa_tools:run_graspa",
        frozenset({"graspa", "simulation"}),
        (
            RuntimeRequirement(
                "path", _GRASPA_EXECUTABLE, "configure the gRASPA-SYCL runtime"
            ),
        ),
    ),
    ToolSpec(
        "load_document",
        "Load a document into the in-process RAG vector store.",
        "chemgraph.tools.rag_tools:load_document",
        frozenset({"rag", "files"}),
        (
            RuntimeRequirement("module", "faiss", "install chemgraph[rag]"),
            RuntimeRequirement(
                "module", "langchain_text_splitters", "install chemgraph[rag]"
            ),
        ),
    ),
    ToolSpec(
        "query_knowledge_base",
        "Query documents loaded into the in-process RAG vector store.",
        "chemgraph.tools.rag_tools:query_knowledge_base",
        frozenset({"rag", "analysis"}),
        (RuntimeRequirement("module", "faiss", "install chemgraph[rag]"),),
    ),
    ToolSpec(
        "generate_html",
        "Generate an HTML report for an ASE result.",
        "chemgraph.tools.report_tools:generate_html",
        frozenset({"report", "files"}),
    ),
    ToolSpec(
        "run_xanes",
        "Run an FDMNES XANES calculation.",
        "chemgraph.tools.xanes_tools:run_xanes",
        frozenset({"xanes", "simulation"}),
        (
            RuntimeRequirement(
                "environment", "FDMNES_EXE", "set it to the FDMNES executable"
            ),
        ),
    ),
    ToolSpec(
        "fetch_xanes_data",
        "Fetch structures for XANES calculations from Materials Project.",
        "chemgraph.tools.xanes_tools:fetch_xanes_data",
        frozenset({"xanes", "data"}),
        (RuntimeRequirement("module", "mp_api", "install chemgraph[xanes]"),),
    ),
    ToolSpec(
        "plot_xanes_data",
        "Plot XANES spectra from FDMNES results.",
        "chemgraph.tools.xanes_tools:plot_xanes_data",
        frozenset({"xanes", "analysis"}),
    ),
)


class ToolRegistry:
    """Discover and lazily resolve ChemGraph's in-process LLM tools."""

    def __init__(self, specs: Iterable[ToolSpec] | None = None):
        self._specs: dict[str, ToolSpec] = {}
        self._tools: dict[str, BaseTool] = {}
        for spec in BUILTIN_TOOL_SPECS if specs is None else specs:
            self.register(spec)

    def register(
        self,
        entry: ToolSpec | BaseTool,
        *,
        tags: Iterable[str] = (),
        replace: bool = False,
    ) -> None:
        """Register a lazy specification or an already-created tool."""
        if isinstance(entry, BaseTool):
            spec = ToolSpec(
                name=entry.name,
                description=entry.description,
                import_path=None,
                tags=frozenset(tags),
            )
            tool = entry
        elif isinstance(entry, ToolSpec):
            spec = entry
            tool = None
        else:
            raise TypeError("ToolRegistry entries must be ToolSpec or BaseTool objects.")

        if not spec.name:
            raise ValueError("Registry names must not be empty.")
        if spec.name in self._specs and not replace:
            raise DuplicateRegistryEntryError(
                f"Tool {spec.name!r} is already registered."
            )
        self._specs[spec.name] = spec
        self._tools.pop(spec.name, None)
        if tool is not None:
            self._tools[spec.name] = tool

    def names(self, *, tags: Iterable[str] = ()) -> tuple[str, ...]:
        """Return canonical names in deterministic registration order."""
        requested = frozenset(tags)
        return tuple(
            name
            for name, spec in self._specs.items()
            if not requested or requested.issubset(spec.tags)
        )

    def specs(self, *, tags: Iterable[str] = ()) -> tuple[ToolSpec, ...]:
        """Return specifications, optionally filtered by tags."""
        return tuple(self._specs[name] for name in self.names(tags=tags))

    def get_spec(self, name: str) -> ToolSpec:
        """Return metadata for one tool without importing it."""
        try:
            return self._specs[name]
        except KeyError as exc:
            raise UnknownRegistryEntryError(f"Unknown tool: {name!r}.") from exc

    def availability(self, name: str) -> RegistryAvailability:
        """Check declared runtime requirements without loading the tool."""
        spec = self.get_spec(name)
        issues = tuple(
            issue for requirement in spec.requirements if (issue := requirement.issue())
        )
        return RegistryAvailability(name=name, issues=issues)

    def get(self, name: str, *, require_available: bool = False) -> BaseTool:
        """Resolve and cache one tool."""
        spec = self.get_spec(name)
        if require_available and not (status := self.availability(name)).available:
            raise RegistryUnavailableError(name, status.issues)
        if name in self._tools:
            return self._tools[name]
        if spec.import_path is None:
            raise RegistryLoadError(f"Tool {name!r} has no import path or instance.")

        module_name, separator, attribute = spec.import_path.partition(":")
        if not separator or not module_name or not attribute:
            raise RegistryLoadError(
                f"Tool {name!r} has invalid import path {spec.import_path!r}."
            )
        try:
            module = importlib.import_module(module_name)
            tool = getattr(module, attribute)
        except (ImportError, AttributeError) as exc:
            raise RegistryLoadError(
                f"Could not load tool {name!r} from {spec.import_path!r}: {exc}"
            ) from exc
        if not isinstance(tool, BaseTool):
            raise RegistryLoadError(
                f"{spec.import_path!r} resolved to {type(tool).__name__}, not BaseTool."
            )
        if tool.name != name:
            raise RegistryLoadError(
                f"Tool {spec.import_path!r} is named {tool.name!r}, expected {name!r}."
            )
        self._tools[name] = tool
        return tool

    def resolve(
        self,
        names: Iterable[str] | None = None,
        *,
        tags: Iterable[str] = (),
        require_available: bool = False,
    ) -> list[BaseTool]:
        """Resolve tools in requested or registry order."""
        selected = tuple(names) if names is not None else self.names(tags=tags)
        requested_tags = frozenset(tags)
        if requested_tags:
            selected = tuple(
                name
                for name in selected
                if requested_tags.issubset(self.get_spec(name).tags)
            )
        return [
            self.get(name, require_available=require_available) for name in selected
        ]


__all__ = [
    "BUILTIN_TOOL_SPECS",
    "DuplicateRegistryEntryError",
    "RegistryAvailability",
    "RegistryError",
    "RegistryLoadError",
    "RegistryUnavailableError",
    "RuntimeRequirement",
    "ToolRegistry",
    "ToolSpec",
    "UnknownRegistryEntryError",
]
