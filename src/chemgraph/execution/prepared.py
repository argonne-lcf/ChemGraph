"""Exact, JSON-serializable preparations for the existing ASE tool surface.

Resources are provisioned by the operator, never downloaded during preparation.
The prepared record is an approval document, not an executable Python object.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path


PREPARATION_VERSION = 1
ASE_TOOLS = frozenset({
    "run_ase", "file_to_atomsdata", "save_atomsdata_to_file",
    "smiles_to_atomsdata", "smiles_to_coordinate_file", "extract_output_json",
    "get_symmetry_number", "is_linear_molecule",
})
CALCULATORS = frozenset({
    "emt", "mace_polar", "mace_mp", "mace_off", "mace_anicc", "fairchem",
    "aimnet2", "tblite", "nwchem", "orca",
})
FRAMEWORK_HELPERS = frozenset({
    "search_tools", "load_tools", "ask_human", "write_todos", "ls", "read_file",
    "glob", "grep",
})


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class ExecutionPolicy:
    """Trusted operator policy. Empty calculator/tool sets allow no science.

    model_resources maps ``calculator:model-name`` (or ``calculator:default``)
    to an installed weights file. executable_bindings maps NWChem/ORCA to a
    command plus every executable/resource file whose identity must be checked.
    """

    tools: frozenset[str] = ASE_TOOLS
    calculators: frozenset[str] = CALCULATORS
    helpers: frozenset[str] = FRAMEWORK_HELPERS
    configuration_id: str = ""
    default_calculator: dict | None = None
    model_resources: dict = field(default_factory=dict)
    executable_bindings: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)

    def __post_init__(self):
        if set(self.limits) - {"max_steps", "max_atoms"}:
            raise ValueError("Supported preparation limits are max_steps and max_atoms")
        if any(not isinstance(value, int) or value <= 0 for value in self.limits.values()):
            raise ValueError("Execution limits must be positive integers")

    @property
    def identity(self):
        data = asdict(self)
        for key in ("tools", "calculators", "helpers"):
            data[key] = sorted(data[key])
        return digest(data)

    def check_tool(self, name):
        if name not in self.tools or name not in ASE_TOOLS:
            raise PermissionError(f"No permitted preparation for tool {name!r}")


@dataclass(frozen=True)
class PreparedAction:
    version: int
    operation_id: str
    owner: str
    thread_id: str
    task_id: str
    tool: str
    requested_arguments: dict
    arguments: dict
    inputs: list[dict]
    resources: list[dict]
    calculator: dict | None
    configuration_id: str
    policy_identity: str
    limits: dict
    digest: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        action = cls(**value)
        action.verify_digest()
        return action

    def verify_digest(self):
        value = self.to_dict()
        saved = value.pop("digest")
        if self.version != PREPARATION_VERSION or digest(value) != saved:
            raise ValueError("Prepared action version or digest is invalid")


_active: ContextVar[PreparedAction | None] = ContextVar("prepared_action", default=None)


def active_preparation():
    return _active.get()


@contextmanager
def using_preparation(action):
    token = _active.set(action)
    try:
        yield
    finally:
        _active.reset(token)


def _resource(context, path, role):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or not any(
        path.is_relative_to(root) for root in context.model_roots
    ):
        raise PermissionError("Resource must be a file in operator-configured model roots")
    return {"path": str(path), "sha256": file_digest(path), "role": role}


def _snapshot(context, path):
    from chemgraph.tools.ase_core import _resolve_existing_path

    source = Path(_resolve_existing_path(path))
    content = source.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    destination = context.workspace / ".chemgraph-inputs" / checksum / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_digest(destination) != checksum:
            raise ValueError("Input snapshot was tampered with")
    else:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
    return {"source": str(source), "path": str(destination), "sha256": checksum}


def _calculator(context, payload):
    from chemgraph.schemas.ase_input import ASEInputSchema, _engine_available
    from chemgraph.execution.scoped import scoped_calculator

    policy = context.policy
    if payload.get("calculator") is None:
        if policy.default_calculator is None:
            raise ValueError("Clarification required: select a calculator explicitly")
        payload["calculator"] = dict(policy.default_calculator)
    scoped_calculator(payload["calculator"])
    # Preserve the established schema coercion, aliases and availability checks.
    params = ASEInputSchema.model_validate(payload)
    calc = params.calculator.model_dump(mode="json")
    kind = calc["calculator_type"].lower()
    if kind not in policy.calculators:
        raise PermissionError(f"Calculator {kind!r} is not permitted")
    module = {"fairchem": "fairchem.core", "aimnet2": "aimnet2calc", "tblite": "tblite"}.get(kind)
    if kind.startswith("mace_"):
        module = "mace"
    if module and not _engine_available(module):
        raise ImportError(f"Provision the {module} calculator dependency before approval")
    if kind == "mace_polar":
        from chemgraph.utils.calculator_defaults import mace_polar_available

        if not mace_polar_available():
            raise ImportError("Provision the MACE-Polar graph-longrange add-on before approval")
    resources = []
    identity = {"type": kind}
    package = {
        "emt": "ase", "tblite": "tblite", "fairchem": "fairchem-core",
        "aimnet2": "aimnet2calc", "nwchem": "ase", "orca": "ase",
    }.get(kind, "mace-torch")
    try:
        identity["package"] = package
        identity["package_version"] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        identity["package_version"] = "uninstalled"
    if kind.startswith("mace_") or kind in {"fairchem", "aimnet2"}:
        model_key = "model_name" if kind == "fairchem" else "model"
        selection = calc.get(model_key)
        resource = policy.model_resources.get(f"{kind}:{selection or 'default'}")
        if resource is None and selection and Path(selection).is_file():
            resource = selection
        if resource is None:
            raise ValueError(f"Provision exact model resources for {kind}:{selection or 'default'}")
        binding = resource if isinstance(resource, dict) else {"weights": resource}
        descriptor = _resource(context, binding["weights"], "weights")
        resources.append(descriptor)
        identity.update(model=selection, weights=descriptor)
        if kind == "fairchem":
            if not binding.get("atom_refs"):
                raise ValueError("Provision FAIRChem atomic-reference data before approval")
            refs = _resource(context, binding["atom_refs"], "atom_refs")
            identity["atom_refs"] = refs
            resources.append(refs)
        if kind != "fairchem":
            calc[model_key] = descriptor["path"]
    if kind in {"orca", "nwchem"}:
        binding = policy.executable_bindings.get(kind)
        if not binding or not binding.get("command") or not binding.get("files"):
            raise ValueError(f"Configure an executable binding with files for {kind}")
        identity["executable"] = json.loads(canonical(binding))
        resources.extend(_resource(context, path, "executable") for path in binding["files"])
        calc["directory"] = scoped_calculator({"directory": calc["directory"]})["directory"]
    resolved = params.model_dump(mode="json")
    resolved["calculator"] = calc
    return resolved, identity, resources


def prepare_action(context, call) -> PreparedAction:
    """Resolve defaults and save referenced bytes before presenting a review."""
    from chemgraph.execution.scoped import execution_context
    from chemgraph.tools.ase_core import _resolve_path

    if context.policy is None:
        raise ValueError("Exact preparation requires an explicit execution policy")
    policy = context.policy
    name = call["name"]
    policy.check_tool(name)
    requested = json.loads(canonical(call.get("args", {})))
    arguments = json.loads(canonical(requested))
    inputs, resources, calculator = [], [], None
    with execution_context(context):
        if name == "run_ase":
            params, calculator, resources = _calculator(context, arguments.get("params", {}))
            if params["driver"] not in {"energy", "dipole", "opt", "vib", "ir", "thermo"}:
                raise ValueError("Clarification required: select an existing ASE driver")
            if not params["output_results_file"].endswith(".json"):
                raise ValueError("ASE result output must end with .json")
            if "max_steps" in policy.limits and params["steps"] > policy.limits["max_steps"]:
                raise PermissionError("Calculation exceeds the configured step limit")
            snapshot = _snapshot(context, params["input_structure_file"])
            inputs.append(snapshot)
            if "max_atoms" in policy.limits:
                from ase.io import read

                if len(read(snapshot["path"])) > policy.limits["max_atoms"]:
                    raise PermissionError("Structure exceeds the configured atom limit")
            params["input_structure_file"] = snapshot["path"]
            params["output_results_file"] = _resolve_path(params["output_results_file"])
            arguments = {"params": params}
        else:
            from chemgraph.tools import ase_tools, cheminformatics_tools

            tool = getattr(ase_tools, name, None) or getattr(cheminformatics_tools, name)
            arguments = tool.args_schema.model_validate(arguments).model_dump(mode="json")
            if name in {"file_to_atomsdata", "extract_output_json"}:
                key = "fname" if name == "file_to_atomsdata" else "json_file"
                snapshot = _snapshot(context, arguments[key])
                inputs.append(snapshot)
                arguments[key] = snapshot["path"]
            if name in {"save_atomsdata_to_file", "smiles_to_coordinate_file"}:
                key = "fname" if name == "save_atomsdata_to_file" else "output_file"
                arguments[key] = _resolve_path(arguments[key])
    action = PreparedAction(
        version=PREPARATION_VERSION,
        operation_id=digest([context.owner, context.thread_id, context.task_id, call["id"]]),
        owner=context.owner, thread_id=context.thread_id, task_id=context.task_id,
        tool=name, requested_arguments=requested, arguments=arguments,
        inputs=inputs, resources=resources, calculator=calculator,
        configuration_id=policy.configuration_id, policy_identity=policy.identity,
        limits=dict(policy.limits),
    )
    value = action.to_dict()
    value.pop("digest")
    return PreparedAction(**value, digest=digest(value))


def validate_preparation(context, call, action):
    """Revalidate the approved snapshot, policy and resources before invocation."""
    action.verify_digest()
    if (
        action.operation_id != digest([context.owner, context.thread_id, context.task_id, call["id"]])
        or action.tool != call["name"]
        or canonical(action.requested_arguments) != canonical(call.get("args", {}))
        or action.policy_identity != context.policy.identity
    ):
        raise PermissionError("Preparation does not match this operation or policy")
    context.policy.check_tool(action.tool)
    if action.calculator is not None:
        try:
            version = importlib.metadata.version(action.calculator["package"])
        except importlib.metadata.PackageNotFoundError:
            version = "uninstalled"
        if version != action.calculator["package_version"]:
            raise ValueError("Calculator installation changed; a new review is required")
    for resource in action.inputs + action.resources:
        path = Path(resource["path"])
        if path.is_symlink() or file_digest(path) != resource["sha256"]:
            raise ValueError("Prepared input or resource changed; a new review is required")
    # Re-resolve destinations in case a workspace symlink changed after review.
    from chemgraph.execution.scoped import scoped_path

    params = action.arguments.get("params", {})
    paths = [params.get("output_results_file"), params.get("calculator", {}).get("directory")]
    if action.tool in {"save_atomsdata_to_file", "smiles_to_coordinate_file"}:
        key = "fname" if action.tool == "save_atomsdata_to_file" else "output_file"
        paths.append(action.arguments[key])
    for path in paths:
        if path is not None:
            scoped_path(path)
