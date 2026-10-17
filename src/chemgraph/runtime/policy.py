"""Operator policy and exact, JSON-serializable preparations of ASE tool calls.

A preparation resolves defaults, snapshots referenced input bytes and binds
installed model files before a human reviews the action. Its digest is what an
approval refers to; execution consumes the prepared arguments, never the raw
model-proposed ones. Resources are provisioned by the operator, never
downloaded during preparation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from chemgraph.runtime.artifacts import (
    INPUTS_DIR, file_identity, internal_path, publish_file,
)

PREPARATION_VERSION = 1

# Tools with an exact preparation. Others are unavailable in scoped runs.
SCOPED_TOOLS = frozenset({
    "run_ase", "file_to_atomsdata", "save_atomsdata_to_file",
    "smiles_to_atomsdata", "smiles_to_coordinate_file", "extract_output_json",
    "get_symmetry_number", "is_linear_molecule",
})
# Explicit opt-in for services that permit external PubChem requests. Existing
# embedded policies retain their local-only default catalog.
PREPARABLE_TOOLS = SCOPED_TOOLS | {"molecule_name_to_smiles"}
# Calculators whose resources a preparation can bind exactly.
SCOPED_CALCULATORS = frozenset({"emt", "mace_mp", "mace_off", "mace_anicc"})
_MODEL_CALCULATORS = SCOPED_CALCULATORS - {"emt"}
# Framework tools that read checkpoint state or ask the user; never journaled.
FRAMEWORK_HELPERS = frozenset({
    "ask_human", "write_todos", "ls", "read_file", "glob", "grep",
    "search_tools", "load_tools",
})
_DRIVERS = frozenset({"energy", "dipole", "opt", "vib", "ir", "thermo"})
_LIMITS = frozenset({"max_steps", "max_atoms"})


class PolicyDenied(PermissionError):
    """The operator policy does not permit this action; it was not executed."""


class PreparationError(ValueError):
    """The action cannot be prepared now (e.g. its input does not exist yet)."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


_digests: dict[tuple, str] = {}


def cached_digest(path) -> str:
    """SHA-256 of a file, cached by inode, size and modification time."""
    stat = os.stat(path)
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _digests:
        _digests[key] = file_identity(path)["sha256"]
    return _digests[key]


@dataclass(frozen=True)
class ExecutionPolicy:
    """Trusted operator configuration for scoped runs.

    ``model_resources`` maps ``"<calculator>:<model>"`` (or
    ``"<calculator>:default"``) to an installed weights file under one of
    ``model_roots``. Empty ``tools`` or ``calculators`` permit no science.
    """

    tools: frozenset[str] = SCOPED_TOOLS
    calculators: frozenset[str] = frozenset({"emt"})
    helpers: frozenset[str] = FRAMEWORK_HELPERS
    default_calculator: dict | None = None
    model_resources: dict = field(default_factory=dict)
    model_roots: tuple = ()
    limits: dict = field(default_factory=dict)
    configuration_id: str = ""

    def __post_init__(self):
        if unknown := set(self.tools) - PREPARABLE_TOOLS:
            raise ValueError(f"No exact preparation exists for tools: {sorted(unknown)}")
        if unknown := set(self.calculators) - SCOPED_CALCULATORS:
            raise ValueError(f"Scoped runs do not yet support calculators: {sorted(unknown)}")
        if set(self.limits) - _LIMITS:
            raise ValueError("Supported limits are max_steps and max_atoms.")
        if any(not isinstance(value, int) or value <= 0 for value in self.limits.values()):
            raise ValueError("Execution limits must be positive integers.")
        for name in ("tools", "calculators", "helpers"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        object.__setattr__(self, "model_roots", tuple(
            str(Path(root).expanduser().resolve()) for root in self.model_roots
        ))
        object.__setattr__(self, "model_resources", {
            str(key): str(value) for key, value in self.model_resources.items()
        })

    @property
    def identity(self) -> str:
        data = asdict(self)
        for name in ("tools", "calculators", "helpers"):
            data[name] = sorted(data[name])
        data["model_roots"] = list(data["model_roots"])
        return digest(data)


@dataclass(frozen=True)
class PreparedAction:
    """An approval document: exact tool, arguments, inputs and resources."""

    version: int
    operation_id: str
    tool: str
    requested_arguments: dict
    arguments: dict
    inputs: list[dict]
    resources: list[dict]
    calculator: dict | None
    policy_identity: str
    digest: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> PreparedAction:
        action = cls(**value)
        content = action.to_dict()
        saved = content.pop("digest")
        if action.version != PREPARATION_VERSION or digest(content) != saved:
            raise ValueError("Prepared action version or digest is invalid.")
        return action


def operation_id(context, tool_call_id: str) -> str:
    return digest([context.owner, context.thread_id, context.task_id, tool_call_id])


def call_digest(call: dict) -> str:
    return digest({"id": call["id"], "name": call["name"], "args": call.get("args", {})})


def _relative(context, path) -> str:
    from chemgraph.runtime.context import scoped_path

    return Path(scoped_path(path)).relative_to(context.workspace).as_posix()


def _snapshot(context, path) -> dict:
    """Preserve the referenced input bytes under a content-addressed name."""
    from chemgraph.tools.ase_core import _resolve_existing_path

    source = Path(_resolve_existing_path(path))
    if not source.is_file():
        raise PreparationError(
            f"Input {path!r} does not exist yet; propose this action after the "
            "step that creates it has completed."
        )
    identity = file_identity(source)
    destination = internal_path(context.workspace, INPUTS_DIR, identity["sha256"], source.name)
    publish_file(context.workspace, source, destination, identity)
    return {
        "source": source.relative_to(context.workspace).as_posix(),
        "path": destination.relative_to(context.workspace).as_posix(),
        **identity,
    }


def _resource(policy, path, role) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or not any(path.is_relative_to(root) for root in policy.model_roots):
        raise PolicyDenied(f"Resource {path} is not a file under the operator's model roots.")
    return {"path": str(path), "sha256": cached_digest(path), "role": role}


def _calculator(policy, payload: dict) -> tuple[dict, dict, list[dict]]:
    from chemgraph.schemas.ase_input import ASEInputSchema

    calculator = payload.get("calculator")
    if calculator is not None and not isinstance(calculator, Mapping):
        raise PreparationError("'calculator' must be an object with a calculator_type.")
    calculator = calculator or policy.default_calculator
    if calculator is None:
        raise PreparationError("Select a calculator explicitly; scoped runs have no implicit default.")
    calculator = dict(calculator)
    kind = str(calculator.get("calculator_type", "")).lower()
    if kind not in policy.calculators:
        raise PolicyDenied(f"Calculator {kind!r} is not permitted by the operator policy.")
    if any(key in calculator for key in ("command", "profile", "directory")):
        raise PolicyDenied("Calculator commands and directories are operator-controlled.")
    resources = []
    if kind in _MODEL_CALCULATORS:
        selection = calculator.get("model")
        binding = policy.model_resources.get(f"{kind}:{selection or 'default'}")
        if binding is None:
            raise PolicyDenied(
                f"No installed model is configured for {kind}:{selection or 'default'}."
            )
        resources.append(_resource(policy, binding, "weights"))
        calculator["model"] = resources[0]["path"]
    try:
        params = ASEInputSchema.model_validate({**payload, "calculator": calculator})
    except ValueError as exc:
        raise PreparationError(str(exc)) from exc
    package = "ase" if kind == "emt" else "mace-torch"
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreparationError(f"Install {package} before proposing {kind} calculations.") from exc
    identity = {"type": kind, "package": package, "package_version": version,
                "model": selection if kind in _MODEL_CALCULATORS else None}
    return params.model_dump(mode="json"), identity, resources


def _prepare_run_ase(context, arguments) -> tuple[dict, list, list, dict]:
    policy = context.policy
    params = arguments.get("params")
    if params is not None and not isinstance(params, Mapping):
        raise PreparationError("run_ase 'params' must be an object.")
    params, calculator, resources = _calculator(policy, dict(params or {}))
    if params["driver"] not in _DRIVERS:
        raise PreparationError(f"Unsupported ASE driver {params['driver']!r}.")
    if not params["output_results_file"].endswith(".json"):
        raise PreparationError("The ASE result file must end with .json.")
    if params["steps"] > policy.limits.get("max_steps", params["steps"]):
        raise PolicyDenied("The calculation exceeds the configured step limit.")
    snapshot = _snapshot(context, params["input_structure_file"])
    if "max_atoms" in policy.limits:
        from ase.io import read
        from ase.io.formats import UnknownFileTypeError

        try:
            atoms = read(context.workspace / snapshot["path"])
        except UnknownFileTypeError as exc:
            raise PreparationError(
                f"Cannot read input structure {params['input_structure_file']!r}: {exc}"
            ) from exc

        if len(atoms) > policy.limits["max_atoms"]:
            raise PolicyDenied("The structure exceeds the configured atom limit.")
    params["input_structure_file"] = snapshot["path"]
    params["output_results_file"] = _relative(context, params["output_results_file"])
    return {"params": params}, [snapshot], resources, calculator


_READS = {"file_to_atomsdata": "fname", "extract_output_json": "json_file"}
_WRITES = {"save_atomsdata_to_file": "fname", "smiles_to_coordinate_file": "output_file"}


def prepare(context, call: dict) -> PreparedAction:
    """Resolve and snapshot a tool call for review. Raises before any side effect."""
    from chemgraph.runtime.context import run_context

    policy = context.policy
    name = call["name"]
    if name not in policy.tools:
        raise PolicyDenied(f"Tool {name!r} is not permitted by the operator policy.")
    requested = json.loads(canonical(call.get("args", {})))
    inputs, resources, calculator = [], [], None
    with run_context(context):
        try:
            if name == "run_ase":
                arguments, inputs, resources, calculator = _prepare_run_ase(context, requested)
            else:
                from chemgraph.tools import ase_tools, cheminformatics_tools

                tool = getattr(ase_tools, name, None) or getattr(cheminformatics_tools, name)
                arguments = tool.args_schema.model_validate(requested).model_dump(mode="json")
                if name in _READS:
                    snapshot = _snapshot(context, arguments[_READS[name]])
                    inputs.append(snapshot)
                    arguments[_READS[name]] = snapshot["path"]
                if name in _WRITES:
                    arguments[_WRITES[name]] = _relative(context, arguments[_WRITES[name]])
        except PermissionError as exc:
            raise PolicyDenied(str(exc)) from exc
        except (ValueError, OSError) as exc:
            if isinstance(exc, PreparationError):
                raise
            raise PreparationError(str(exc)) from exc
    content = PreparedAction(
        version=PREPARATION_VERSION, operation_id=operation_id(context, call["id"]),
        tool=name, requested_arguments=requested, arguments=arguments,
        inputs=inputs, resources=resources, calculator=calculator,
        policy_identity=policy.identity,
    ).to_dict()
    content.pop("digest")
    return PreparedAction(**content, digest=digest(content))


def validate(context, call: dict, action: PreparedAction) -> None:
    """Re-check the approved preparation immediately before invocation."""
    from chemgraph.runtime.context import run_context, scoped_path

    if (action.operation_id != operation_id(context, call["id"]) or action.tool != call["name"]
            or canonical(action.requested_arguments) != canonical(call.get("args", {}))
            or action.policy_identity != context.policy.identity):
        raise PreparationError("The preparation does not match this operation or policy.")
    if action.calculator is not None:
        try:
            version = importlib.metadata.version(action.calculator["package"])
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != action.calculator["package_version"]:
            raise PreparationError("The calculator installation changed; a new review is required.")
    for item in action.inputs:
        path = internal_path(context.workspace, INPUTS_DIR, Path(item["path"]).relative_to(INPUTS_DIR))
        if cached_digest(path) != item["sha256"]:
            raise PreparationError("A prepared input changed; a new review is required.")
    for item in action.resources:
        if Path(item["path"]).is_symlink() or cached_digest(item["path"]) != item["sha256"]:
            raise PreparationError("A model resource changed; a new review is required.")
    # Re-resolve destinations in case a workspace symlink changed after review.
    outputs = [action.arguments.get("params", {}).get("output_results_file")]
    if action.tool in _WRITES:
        outputs.append(action.arguments[_WRITES[action.tool]])
    with run_context(context):
        for path in filter(None, outputs):
            scoped_path(path)
