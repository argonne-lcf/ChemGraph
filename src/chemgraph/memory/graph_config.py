"""Stable, non-secret identities for main-agent reconstruction."""

import hashlib
import json

from deepagents.backends import LocalShellBackend, StateBackend

from chemgraph.graphs.workspace import cli_backend_descriptor


GRAPH_SCHEMA_VERSION = 2


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
