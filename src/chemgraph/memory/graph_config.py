"""Stable, non-secret identities for main-agent reconstruction."""

import hashlib
import json

from deepagents.backends import LocalShellBackend, StateBackend

from chemgraph.graphs.workspace import cli_backend_descriptor


GRAPH_SCHEMA_VERSION = 3
NEW_SESSION_GUIDANCE = "Start a new session; the old transcript remains readable."


def validate_configuration_id(value):
    """Validate a caller-owned identity without interpreting its contents."""
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("configuration_id must be a non-empty, non-secret string.")
    return value


def fingerprint(payload):
    """Hash canonical JSON; never fall back to arbitrary object representations."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def describe_backend(backend):
    """Return a public descriptor, CLI eligibility, and caller-identity requirement."""
    if backend is None or type(backend) is StateBackend:
        return {"type": "state"}, True, False
    if descriptor := cli_backend_descriptor(backend):
        return descriptor, True, False
    descriptor = {"type": f"{type(backend).__module__}.{type(backend).__qualname__}"}
    if isinstance(backend, LocalShellBackend):
        descriptor.update(
            workspace=str(backend.cwd.resolve()), virtual_mode=backend.virtual_mode,
            timeout=backend._default_timeout, max_output_bytes=backend._max_output_bytes,
        )
    # Environment values and opaque internals are deliberately caller-owned.
    return descriptor, False, True


def describe_tool_registry(registry):
    """Version built-ins without hashing prose; preserve full custom specifications."""
    from chemgraph.registry.tools import BUILTIN_TOOL_CATALOG_VERSION, ToolRegistry

    builtins = {spec.name: spec for spec in ToolRegistry().specs()}
    described = []
    custom = False
    for spec in registry.specs():
        if builtins.get(spec.name) == spec:
            described.append((spec.name, spec.import_path, BUILTIN_TOOL_CATALOG_VERSION))
        else:
            custom = True
            described.append((
                spec.name, spec.import_path, spec.description, sorted(spec.tags),
                [(req.kind, req.value, req.hint, req.env_var) for req in spec.requirements],
                spec.interactive, spec.executes_code,
            ))
    return described, custom


def describe_agent_registry(registry):
    """Version the entire discoverable catalog without importing worker graphs."""
    from importlib.metadata import version
    from chemgraph.registry.agents import AgentRegistry, BUILTIN_AGENT_CATALOG_VERSION
    from chemgraph.registry.tools import BUILTIN_TOOL_CATALOG_VERSION

    builtins = {spec.name: spec for spec in AgentRegistry().specs()}
    entries = []
    custom = False
    for spec in registry.specs():
        builtin = builtins.get(spec.name) == spec
        custom |= not builtin
        entries.append({
            "name": spec.name, "import_path": spec.import_path,
            "compatibility_version": spec.compatibility_version,
            "default_tools": spec.default_tool_names,
            "aliases": spec.aliases, "required_arguments": spec.required_arguments,
            "requirements": [(req.kind, req.value, req.env_var) for req in spec.requirements],
            **({} if builtin else {"description": spec.description, "tags": sorted(spec.tags)}),
        })
    return {
        "runtime": ("deepagents", version("deepagents")),
        "catalog_version": BUILTIN_AGENT_CATALOG_VERSION,
        "tool_catalog_version": BUILTIN_TOOL_CATALOG_VERSION,
        "workers": entries,
    }, custom


def describe_worker_options(options):
    """Describe supported worker settings; opaque settings need a caller ID."""
    result = {}
    opaque = False
    scalar_names = {
        "system_prompt", "formatter_prompt", "report_prompt", "recursion_limit",
        "structured_output", "generate_report", "max_retries", "human_supervised",
        "skills", "skill_dirs", "discover_skills", "user_skills_dir",
        "terminal_tool_names", "interrupt_on", "checkpointer",
    }
    for name, settings in options.items():
        described = {}
        for key, value in settings.items():
            if key == "backend":
                described[key], _, needs_id = describe_backend(value)
                opaque |= needs_id
            elif key in scalar_names and _is_json(value):
                described[key] = value
            else:
                described[key] = {"caller_owned": True}
                opaque = True
        result[name] = described
    return result, opaque


def _is_json(value):
    if value is None or type(value) in (str, bool, int, float):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json(item) for key, item in value.items())
    return False
