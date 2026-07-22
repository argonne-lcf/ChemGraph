"""Copy-on-edit campaign editing.

Shipped campaigns under ``chemgraph/academy/campaigns/*/campaign.jsonc``
are treated as READ-ONLY examples. The first save copies the campaign
to ``~/.chemgraph-academy/user-campaigns/<name>/campaign.jsonc`` as
plain JSON (no comments) and edits it there. Subsequent saves mutate
the user copy.

New campaigns (created from the canvas) skip the shipped step
entirely -- ``create_blank_campaign`` writes a minimal skeleton
directly under the user path.

Single entry point ``apply_action`` dispatches on the action name so
the HTTP layer has one endpoint that grows by adding cases here, not
by adding routes.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chemgraph.academy.campaigns import resolve_campaign
from chemgraph.academy.core.campaign import _load_jsonc

USER_CAMPAIGNS_DIRNAME = ".chemgraph-academy"
USER_CAMPAIGNS_SUBDIR = "user-campaigns"

# Fields on an agent that the UI is allowed to edit as a whole value.
# Everything gets stored as-is in the JSON; validation is per-field.
EDITABLE_AGENT_FIELDS = frozenset({
    "role",
    "mission",
    "allowed_peers",
    "mcp_servers",
    "allowed_tools",
    "resources",
    "engine",
})

# Campaign-level fields the UI can edit.
EDITABLE_CAMPAIGN_FIELDS = frozenset({
    "user_task",
    "initial_agent",
})

# launch_defaults fields the UI can edit as a whole value. Free-form
# strings; the launcher validates on use. Kept small so the editor
# doesn't accidentally invalidate the JSON schema.
EDITABLE_LAUNCH_FIELDS = frozenset({
    "extra_launcher_argv",   # string; shlex.split at launch time
    "lm_config_template",
    "agent_count",
    "agents_per_node",
    "max_decisions",
})

# Simple identifier check for agent names + campaign names -- alnum,
# dash, underscore. Prevents shell/URL/path shenanigans.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def user_campaigns_root() -> Path:
    """Root directory for user-edited campaign copies on the LAPTOP."""
    return Path.home() / USER_CAMPAIGNS_DIRNAME / USER_CAMPAIGNS_SUBDIR


@dataclass(frozen=True)
class ResolvedCampaign:
    """Where a campaign lives on disk today."""

    name: str
    path: Path
    is_user_copy: bool


def resolve_editable(campaign_name: str) -> ResolvedCampaign:
    """Return the path we should edit for this campaign."""
    user_path = user_campaigns_root() / campaign_name / "campaign.jsonc"
    if user_path.exists():
        return ResolvedCampaign(name=campaign_name, path=user_path, is_user_copy=True)
    shipped = resolve_campaign(campaign_name)
    return ResolvedCampaign(name=campaign_name, path=shipped, is_user_copy=False)


def _load_any(path: Path) -> dict[str, Any]:
    return _load_jsonc(path)  # comment-strip is a no-op on plain JSON


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _user_write_path(campaign_name: str) -> Path:
    return user_campaigns_root() / campaign_name / "campaign.jsonc"


def _validate_id(kind: str, value: Any) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(
            f"{kind} must match [A-Za-z0-9][A-Za-z0-9_-]* (got {value!r})",
        )
    return value


def _validate_str_list(kind: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{kind} must be a list of strings (got {value!r})")
    return list(value)


def _find_agent(agents: list[dict[str, Any]], name: str) -> int:
    for i, spec in enumerate(agents):
        if isinstance(spec, dict) and spec.get("name") == name:
            return i
    raise ValueError(f"agent {name!r} not found")


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


def apply_action(
    *,
    campaign_name: str,
    action: str,
    params: dict[str, Any],
) -> ResolvedCampaign:
    """Apply one action to a campaign. Copy-on-edit if needed.

    Actions (all mutate the JSON doc in memory, then write atomically):

    - ``edit_agent_field``   {agent, field, value}
    - ``edit_campaign_field`` {field, value}
    - ``add_agent``          {agent, role?, mission?, allowed_peers?, mcp_servers?, allowed_tools?}
    - ``remove_agent``       {agent}
    - ``rename_agent``       {agent, new_name}   (also updates allowed_peers + initial_agent refs)
    - ``set_edge``           {source, target, add}  (add=True/False on source.allowed_peers)
    - ``add_mcp_server``     {name, command}    (adds to campaign.mcp_servers)
    - ``remove_mcp_server``  {name}             (also prunes agent.mcp_servers refs)
    - ``edit_launch_field``  {field, value}     (fields in EDITABLE_LAUNCH_FIELDS)
    - ``set_pbs_script``     {site, script}     (per-site custom PBS bash; empty=clear)

    New actions add a case below and update the frontend; no new HTTP
    routes needed.
    """
    _validate_id("campaign_name", campaign_name)
    resolved = resolve_editable(campaign_name)
    data = _load_any(resolved.path)
    if "agents" not in data or not isinstance(data["agents"], list):
        raise ValueError(f"campaign {campaign_name!r} has no agents list")

    if action == "edit_agent_field":
        _do_edit_agent_field(data, params)
    elif action == "edit_campaign_field":
        _do_edit_campaign_field(data, params)
    elif action == "add_agent":
        _do_add_agent(data, params)
    elif action == "remove_agent":
        _do_remove_agent(data, params)
    elif action == "rename_agent":
        _do_rename_agent(data, params)
    elif action == "set_edge":
        _do_set_edge(data, params)
    elif action == "add_mcp_server":
        _do_add_mcp_server(data, params)
    elif action == "remove_mcp_server":
        _do_remove_mcp_server(data, params)
    elif action == "edit_launch_field":
        _do_edit_launch_field(data, params)
    elif action == "set_pbs_script":
        _do_set_pbs_script(data, params)
    else:
        raise ValueError(f"unknown action {action!r}")

    out_path = resolved.path if resolved.is_user_copy else _user_write_path(campaign_name)
    _write_atomic(out_path, data)
    return ResolvedCampaign(name=campaign_name, path=out_path, is_user_copy=True)


def create_blank_campaign(campaign_name: str) -> ResolvedCampaign:
    """Create a new campaign from scratch as an empty user copy.

    Raises FileExistsError if a user copy already exists (frontend
    should prompt for a different name rather than overwrite).
    """
    _validate_id("campaign_name", campaign_name)
    out_path = _user_write_path(campaign_name)
    if out_path.exists():
        raise FileExistsError(f"campaign {campaign_name!r} already exists at {out_path}")
    skeleton = {
        "run_id": campaign_name,
        "user_task": "",
        "prompt_profile": "prompt_profiles/default.json",
        "initial_agent": "",
        # launch_defaults holds runtime knobs (agent_count, lm_config
        # template, etc.). Empty lm_config_template forces the operator
        # to pick one before launch; sensible defaults on the numeric
        # fields so a blank campaign is still runnable.
        "launch_defaults": {
            "lm_config_template": "",
            "agent_count": 0,
            "agents_per_node": 1,
            "max_decisions": 12,
        },
        "resources": {},
        "mcp_servers": [],
        "agents": [],
    }
    _write_atomic(out_path, skeleton)
    return ResolvedCampaign(name=campaign_name, path=out_path, is_user_copy=True)


def clone_campaign(*, source_name: str, new_name: str) -> ResolvedCampaign:
    """Create a new user-copy campaign seeded from an existing one.

    Reads whichever copy of ``source_name`` is current (user copy if
    present, else shipped). Writes to
    ``~/.chemgraph-academy/user-campaigns/<new_name>/campaign.jsonc``.
    Rewrites the top-level ``run_id`` to match ``new_name`` -- otherwise
    two campaigns would compete for the same on-Eagle run scratch dir.

    Raises FileExistsError if a user copy at ``new_name`` already
    exists (frontend prompts for a different name).
    """
    _validate_id("campaign_name", new_name)
    src = resolve_editable(source_name)
    out_path = _user_write_path(new_name)
    if out_path.exists():
        raise FileExistsError(f"campaign {new_name!r} already exists at {out_path}")
    data = _load_any(src.path)
    data["run_id"] = new_name
    _write_atomic(out_path, data)
    return ResolvedCampaign(name=new_name, path=out_path, is_user_copy=True)


# ---------------------------------------------------------------------------
# Per-action implementations
# ---------------------------------------------------------------------------


def _do_edit_agent_field(data: dict[str, Any], params: dict[str, Any]) -> None:
    agent = _validate_id("agent", params.get("agent"))
    field = params.get("field")
    value = params.get("value")
    if field not in EDITABLE_AGENT_FIELDS:
        raise ValueError(
            f"field {field!r} is not editable; allowed: "
            f"{sorted(EDITABLE_AGENT_FIELDS)}",
        )
    idx = _find_agent(data["agents"], agent)
    if field in {"allowed_peers", "mcp_servers", "allowed_tools", "resources"}:
        value = _validate_str_list(field, value)
    elif not isinstance(value, str):
        raise ValueError(f"{field!r} must be a string")
    data["agents"][idx] = {**data["agents"][idx], field: value}


def _do_edit_campaign_field(data: dict[str, Any], params: dict[str, Any]) -> None:
    field = params.get("field")
    value = params.get("value")
    if field not in EDITABLE_CAMPAIGN_FIELDS:
        raise ValueError(
            f"field {field!r} is not editable; allowed: "
            f"{sorted(EDITABLE_CAMPAIGN_FIELDS)}",
        )
    if field == "initial_agent":
        _validate_id("initial_agent", value)
        # Must reference a real agent.
        names = {a.get("name") for a in data["agents"] if isinstance(a, dict)}
        if value not in names:
            raise ValueError(f"initial_agent {value!r} is not a declared agent")
    elif not isinstance(value, str):
        raise ValueError(f"{field!r} must be a string")
    data[field] = value


def _do_add_agent(data: dict[str, Any], params: dict[str, Any]) -> None:
    name = _validate_id("agent", params.get("agent"))
    existing = {a.get("name") for a in data["agents"] if isinstance(a, dict)}
    if name in existing:
        raise ValueError(f"agent {name!r} already exists")
    new_agent = {
        "name": name,
        "role": params.get("role") or "Agent",
        "mission": params.get("mission") or "",
        "allowed_peers": _validate_str_list("allowed_peers", params.get("allowed_peers") or []),
        "mcp_servers": _validate_str_list("mcp_servers", params.get("mcp_servers") or []),
        "allowed_tools": _validate_str_list("allowed_tools", params.get("allowed_tools") or []),
        "resources": _validate_str_list("resources", params.get("resources") or []),
    }
    data["agents"].append(new_agent)
    # If this is the first agent and no initial_agent set, adopt it.
    if not data.get("initial_agent"):
        data["initial_agent"] = name


def _do_remove_agent(data: dict[str, Any], params: dict[str, Any]) -> None:
    name = _validate_id("agent", params.get("agent"))
    idx = _find_agent(data["agents"], name)
    del data["agents"][idx]
    # Prune peer references so the campaign stays consistent.
    for spec in data["agents"]:
        if isinstance(spec, dict):
            spec["allowed_peers"] = [p for p in spec.get("allowed_peers", []) if p != name]
    # Repoint initial_agent if we just removed it.
    if data.get("initial_agent") == name:
        data["initial_agent"] = data["agents"][0]["name"] if data["agents"] else ""


def _do_rename_agent(data: dict[str, Any], params: dict[str, Any]) -> None:
    old = _validate_id("agent", params.get("agent"))
    new = _validate_id("new_name", params.get("new_name"))
    if old == new:
        return
    existing = {a.get("name") for a in data["agents"] if isinstance(a, dict)}
    if new in existing:
        raise ValueError(f"agent {new!r} already exists")
    idx = _find_agent(data["agents"], old)
    data["agents"][idx] = {**data["agents"][idx], "name": new}
    for spec in data["agents"]:
        if isinstance(spec, dict):
            spec["allowed_peers"] = [
                new if p == old else p for p in spec.get("allowed_peers", [])
            ]
    if data.get("initial_agent") == old:
        data["initial_agent"] = new


def _do_add_mcp_server(data: dict[str, Any], params: dict[str, Any]) -> None:
    name = _validate_id("mcp_server", params.get("name"))
    command = params.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("mcp_server command must be a non-empty string")
    servers = data.setdefault("mcp_servers", [])
    if not isinstance(servers, list):
        raise ValueError("campaign mcp_servers must be a list")
    for s in servers:
        if isinstance(s, dict) and s.get("name") == name:
            raise ValueError(f"mcp_server {name!r} already exists")
    servers.append({"name": name, "command": command})


def _do_remove_mcp_server(data: dict[str, Any], params: dict[str, Any]) -> None:
    name = _validate_id("mcp_server", params.get("name"))
    servers = data.get("mcp_servers") or []
    remaining = [s for s in servers if not (isinstance(s, dict) and s.get("name") == name)]
    if len(remaining) == len(servers):
        raise ValueError(f"mcp_server {name!r} not found")
    data["mcp_servers"] = remaining
    # Prune references from agent.mcp_servers.
    for spec in data.get("agents", []):
        if isinstance(spec, dict):
            spec["mcp_servers"] = [m for m in spec.get("mcp_servers", []) if m != name]


def _do_set_edge(data: dict[str, Any], params: dict[str, Any]) -> None:
    source = _validate_id("source", params.get("source"))
    target = _validate_id("target", params.get("target"))
    add = bool(params.get("add", True))
    if source == target:
        raise ValueError("source and target must differ")
    idx = _find_agent(data["agents"], source)
    target_idx = _find_agent(data["agents"], target)  # ensures target exists
    del target_idx  # only used for validation
    peers = list(data["agents"][idx].get("allowed_peers", []))
    if add:
        if target not in peers:
            peers.append(target)
    else:
        peers = [p for p in peers if p != target]
    data["agents"][idx] = {**data["agents"][idx], "allowed_peers": peers}


def _do_edit_launch_field(data: dict[str, Any], params: dict[str, Any]) -> None:
    field = params.get("field")
    value = params.get("value")
    if field not in EDITABLE_LAUNCH_FIELDS:
        raise ValueError(
            f"launch_defaults field {field!r} is not editable; allowed: "
            f"{sorted(EDITABLE_LAUNCH_FIELDS)}",
        )
    if field in {"agent_count", "agents_per_node", "max_decisions"}:
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field!r} must be an integer (got {value!r})")
        if value < 0:
            raise ValueError(f"{field!r} must be >= 0")
    elif not isinstance(value, str):
        raise ValueError(f"{field!r} must be a string")
    block = data.setdefault("launch_defaults", {})
    if not isinstance(block, dict):
        raise ValueError("launch_defaults must be an object")
    block[field] = value


def _do_set_pbs_script(data: dict[str, Any], params: dict[str, Any]) -> None:
    """Set (or clear) the per-site PBS script override.

    Empty string clears the override; the launcher then falls back to
    the built-in template. The script is stored verbatim -- the
    launcher does ``${VAR}`` substitution at qsub time.
    """
    site = params.get("site")
    if not isinstance(site, str) or not _ID_RE.match(site):
        raise ValueError(
            f"site must match [A-Za-z0-9][A-Za-z0-9_-]* (got {site!r})",
        )
    script = params.get("script")
    if script is not None and not isinstance(script, str):
        raise ValueError("script must be a string or null")
    block = data.setdefault("launch_defaults", {})
    if not isinstance(block, dict):
        raise ValueError("launch_defaults must be an object")
    overrides = block.setdefault("per_site_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("launch_defaults.per_site_overrides must be an object")
    site_block = overrides.setdefault(site, {})
    if not isinstance(site_block, dict):
        raise ValueError(f"per_site_overrides.{site} must be an object")
    if not script:
        site_block.pop("pbs_script", None)
        # Prune empty leaves so the JSON stays tidy.
        if not site_block:
            overrides.pop(site, None)
        if not overrides:
            block.pop("per_site_overrides", None)
    else:
        site_block["pbs_script"] = script
