"""Registry for ChemGraph worker graph constructors."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from deepagents.middleware.subagents import CompiledSubAgent

from chemgraph.registry.tools import (
    DuplicateRegistryEntryError,
    RegistryAvailability,
    RegistryLoadError,
    RegistryUnavailableError,
    RuntimeRequirement,
    UnknownRegistryEntryError,
)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Lazy metadata for one worker graph."""

    name: str
    description: str
    import_path: str
    aliases: tuple[str, ...] = ()
    default_tool_names: tuple[str, ...] = ()
    requirements: tuple[RuntimeRequirement, ...] = ()
    required_arguments: tuple[str, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)
    test_only: bool = False


_GRASPA_EXECUTABLE = (
    "/lus/flare/projects/IQC/thang/soft/gRASPA/graspa-sycl/bin/sycl.out"
)
_CORE_TOOLS = (
    "smiles_to_coordinate_file",
    "molecule_name_to_smiles",
    "run_ase",
    "extract_output_json",
    "calculator",
)


BUILTIN_AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        "single_agent",
        "General ChemGraph worker for molecular construction and ASE simulations.",
        "chemgraph.graphs.single_agent:construct_single_agent_graph",
        default_tool_names=_CORE_TOOLS,
        tags=frozenset({"general", "chemistry"}),
    ),
    AgentSpec(
        "multi_agent",
        "Planner-executor ChemGraph worker for decomposable chemistry tasks.",
        "chemgraph.graphs.multi_agent:construct_multi_agent_graph",
        default_tool_names=_CORE_TOOLS,
        tags=frozenset({"planning", "chemistry"}),
    ),
    AgentSpec(
        "python_relp",
        "Python REPL worker for calculations and programmatic analysis.",
        "chemgraph.graphs.python_relp_agent:construct_relp_graph",
        aliases=("python_repl",),
        default_tool_names=("python_repl", "calculator"),
        tags=frozenset({"python", "analysis"}),
    ),
    AgentSpec(
        "graspa",
        "Specialist worker for gRASPA adsorption calculations.",
        "chemgraph.graphs.graspa_agent:construct_graspa_graph",
        aliases=("graspa_agent",),
        default_tool_names=("run_graspa",),
        requirements=(
            RuntimeRequirement(
                "path", _GRASPA_EXECUTABLE, "configure the gRASPA-SYCL runtime"
            ),
        ),
        tags=frozenset({"graspa", "adsorption"}),
    ),
    AgentSpec(
        "mock_agent",
        "Non-executing workflow used to test tool-call generation.",
        "chemgraph.graphs.mock_agent:construct_mock_agent_graph",
        default_tool_names=(
            "file_to_atomsdata",
            "smiles_to_atomsdata",
            "run_ase",
            "molecule_name_to_smiles",
            "save_atomsdata_to_file",
            "calculator",
        ),
        tags=frozenset({"testing"}),
        test_only=True,
    ),
    AgentSpec(
        "graspa_mcp",
        "Planner-executor worker for externally supplied gRASPA MCP tools.",
        "chemgraph.graphs.graspa_mcp:construct_graspa_mcp_graph",
        required_arguments=("executor_tools", "analysis_tools"),
        tags=frozenset({"graspa", "mcp", "planning"}),
    ),
    AgentSpec(
        "rag_agent",
        "RAG worker for document-grounded chemistry questions.",
        "chemgraph.graphs.rag_agent:construct_rag_agent_graph",
        default_tool_names=(
            "load_document",
            "query_knowledge_base",
            "file_to_atomsdata",
            "smiles_to_coordinate_file",
            "run_ase",
            "molecule_name_to_smiles",
            "save_atomsdata_to_file",
            "calculator",
        ),
        requirements=(
            RuntimeRequirement("module", "faiss", "install chemgraph[rag]"),
            RuntimeRequirement(
                "module", "langchain_text_splitters", "install chemgraph[rag]"
            ),
        ),
        tags=frozenset({"rag", "chemistry"}),
    ),
    AgentSpec(
        "single_agent_xanes",
        "Specialist worker for FDMNES XANES simulations and analysis.",
        "chemgraph.graphs.single_agent_xanes:construct_single_agent_xanes_graph",
        default_tool_names=(
            "molecule_name_to_smiles",
            "smiles_to_coordinate_file",
            "run_ase",
            "run_xanes",
            "fetch_xanes_data",
            "plot_xanes_data",
        ),
        requirements=(
            RuntimeRequirement(
                "environment", "FDMNES_EXE", "set it to the FDMNES executable"
            ),
        ),
        tags=frozenset({"xanes", "spectroscopy"}),
    ),
    AgentSpec(
        "molecular_docking",
        "Specialist worker for AutoDock Vina molecular docking.",
        "chemgraph.graphs.molecular_docking:construct_molecular_docking_graph",
        default_tool_names=("run_docking", "molecule_name_to_smiles"),
        requirements=(
            RuntimeRequirement(
                "module", "vina", "install AutoDock Vina from conda-forge"
            ),
            RuntimeRequirement("module", "meeko", "install chemgraph[docking]"),
        ),
        tags=frozenset({"docking", "drug-discovery"}),
    ),
    AgentSpec(
        "single_agent_iri",
        "ALCF IRI Facility API worker for facility status and HPC operations.",
        "chemgraph.graphs.single_agent_iri:construct_iri_graph",
        aliases=("iri",),
        tags=frozenset({"alcf", "iri", "hpc"}),
    ),
)


class AgentRegistry:
    """Discover, construct, and adapt ChemGraph worker graphs."""

    def __init__(self, specs: Iterable[AgentSpec] | None = None):
        self._specs: dict[str, AgentSpec] = {}
        self._aliases: dict[str, str] = {}
        self._constructors: dict[str, Callable[..., Any]] = {}
        for spec in BUILTIN_AGENT_SPECS if specs is None else specs:
            self.register(spec)

    def register(self, spec: AgentSpec, *, replace: bool = False) -> None:
        """Register or replace one worker specification."""
        if not isinstance(spec, AgentSpec):
            raise TypeError("AgentRegistry entries must be AgentSpec objects.")
        if not spec.name:
            raise ValueError("Registry names must not be empty.")

        previous = self._specs.get(spec.name)
        claimed = (spec.name, *spec.aliases)
        collisions = []
        for name in claimed:
            owner = name if name in self._specs else self._aliases.get(name)
            if owner is None:
                continue
            replacing_same_agent = (
                replace and previous is not None and owner == spec.name
            )
            if not replacing_same_agent:
                collisions.append(name)
        if collisions:
            raise DuplicateRegistryEntryError(
                f"Agent name or alias already registered: {collisions!r}."
            )
        if previous is not None:
            for alias in previous.aliases:
                self._aliases.pop(alias, None)

        self._specs[spec.name] = spec
        self._constructors.pop(spec.name, None)
        for alias in spec.aliases:
            self._aliases[alias] = spec.name

    def resolve_name(self, name: str) -> str:
        """Resolve a canonical name or registered alias."""
        canonical = self._aliases.get(name, name)
        if canonical not in self._specs:
            raise UnknownRegistryEntryError(f"Unknown worker agent: {name!r}.")
        return canonical

    def names(self, *, tags: Iterable[str] = ()) -> tuple[str, ...]:
        """Return canonical worker names in registration order."""
        requested = frozenset(tags)
        return tuple(
            name
            for name, spec in self._specs.items()
            if not requested or requested.issubset(spec.tags)
        )

    def specs(self, *, tags: Iterable[str] = ()) -> tuple[AgentSpec, ...]:
        """Return worker specifications, optionally filtered by tags."""
        return tuple(self._specs[name] for name in self.names(tags=tags))

    def get_spec(self, name: str) -> AgentSpec:
        """Return metadata for one worker without importing its graph module."""
        return self._specs[self.resolve_name(name)]

    def availability(
        self,
        name: str,
        *,
        constructor_kwargs: Mapping[str, Any] | None = None,
    ) -> RegistryAvailability:
        """Check runtime and required-constructor inputs for a worker."""
        spec = self.get_spec(name)
        kwargs = constructor_kwargs or {}
        issues = [
            issue
            for requirement in spec.requirements
            if (issue := requirement.issue())
        ]
        issues.extend(
            f"missing non-empty constructor argument {argument!r}"
            for argument in spec.required_arguments
            if not kwargs.get(argument)
        )
        return RegistryAvailability(name=spec.name, issues=tuple(issues))

    def _get_constructor(self, spec: AgentSpec) -> Callable[..., Any]:
        if spec.name in self._constructors:
            return self._constructors[spec.name]
        module_name, separator, attribute = spec.import_path.partition(":")
        if not separator or not module_name or not attribute:
            raise RegistryLoadError(
                f"Agent {spec.name!r} has invalid import path {spec.import_path!r}."
            )
        try:
            module = importlib.import_module(module_name)
            constructor = getattr(module, attribute)
        except (ImportError, AttributeError) as exc:
            raise RegistryLoadError(
                f"Could not load agent {spec.name!r} from {spec.import_path!r}: {exc}"
            ) from exc
        if not callable(constructor):
            raise RegistryLoadError(
                f"Agent constructor {spec.import_path!r} is not callable."
            )
        self._constructors[spec.name] = constructor
        return constructor

    def _prepare_constructor(
        self,
        spec: AgentSpec,
        llm: Any,
        constructor_kwargs: Mapping[str, Any],
    ) -> Callable[..., Any]:
        """Load a constructor and validate its arguments without invoking it."""
        constructor = self._get_constructor(spec)
        try:
            inspect.signature(constructor).bind(llm, **constructor_kwargs)
        except TypeError as exc:
            raise TypeError(f"Invalid options for agent {spec.name!r}: {exc}") from exc
        return constructor

    def build(
        self,
        name: str,
        *,
        llm: Any,
        require_available: bool = True,
        **constructor_kwargs: Any,
    ) -> Any:
        """Build one registered worker through its existing constructor."""
        spec = self.get_spec(name)
        status = self.availability(
            spec.name,
            constructor_kwargs=constructor_kwargs,
        )
        if require_available and not status.available:
            raise RegistryUnavailableError(spec.name, status.issues)

        constructor = self._prepare_constructor(spec, llm, constructor_kwargs)
        return constructor(llm, **constructor_kwargs)

    def as_subagent(
        self,
        name: str,
        *,
        llm: Any,
        require_available: bool = True,
        **constructor_kwargs: Any,
    ) -> CompiledSubAgent:
        """Build one worker as a parent-checkpointed CompiledSubAgent."""
        if constructor_kwargs.get("checkpointer") is not None and (
            "checkpointer" in constructor_kwargs
        ):
            raise ValueError("Registry subagents must inherit the parent checkpointer.")
        constructor_kwargs["checkpointer"] = None
        spec = self.get_spec(name)
        runnable = self.build(
            spec.name,
            llm=llm,
            require_available=require_available,
            **constructor_kwargs,
        )
        return {
            "name": spec.name,
            "description": spec.description,
            "runnable": runnable,
        }

    def as_subagents(
        self,
        names: Iterable[str],
        *,
        llm: Any,
        options: Mapping[str, Mapping[str, Any]] | None = None,
        require_available: bool = True,
    ) -> list[CompiledSubAgent]:
        """Prevalidate and build an explicit worker list."""
        requested = tuple(names)
        canonical = tuple(self.resolve_name(name) for name in requested)
        if len(canonical) != len(set(canonical)):
            raise DuplicateRegistryEntryError(
                "A worker was requested more than once through a name or alias."
            )
        by_name = options or {}
        prepared: list[tuple[AgentSpec, dict[str, Any]]] = []
        unavailable: list[str] = []
        for requested_name, canonical_name in zip(requested, canonical, strict=True):
            spec = self.get_spec(canonical_name)
            kwargs = dict(by_name.get(requested_name, by_name.get(canonical_name, {})))
            if kwargs.get("checkpointer") is not None:
                raise ValueError(
                    "Registry subagents must inherit the parent checkpointer."
                )
            kwargs["checkpointer"] = None
            status = self.availability(
                canonical_name,
                constructor_kwargs=kwargs,
            )
            if require_available and not status.available:
                unavailable.extend(f"{canonical_name}: {issue}" for issue in status.issues)
            prepared.append((spec, kwargs))
        if unavailable:
            raise RegistryUnavailableError("worker agents", unavailable)

        constructors = [
            self._prepare_constructor(spec, llm, kwargs) for spec, kwargs in prepared
        ]
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "runnable": constructor(llm, **kwargs),
            }
            for (spec, kwargs), constructor in zip(
                prepared, constructors, strict=True
            )
        ]


__all__ = ["AgentRegistry", "AgentSpec", "BUILTIN_AGENT_SPECS"]
