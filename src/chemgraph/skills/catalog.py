"""Inspectable skill inventories using the same discovery as agent turns."""

import html

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from chemgraph.skills.runtime import (
    BUNDLED_SKILLS_PATH,
    EXTERNAL_SKILLS_PATH,
    USER_SKILLS_PATH,
    ChemGraphSkillsMiddleware,
    prepare_skill_backend,
    resolve_skill_dirs,
)


class _InventoryMiddleware(ChemGraphSkillsMiddleware):
    """Retain candidates before the normal merge selects each name's winner."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.candidates = []

    def _merge_update(self, source, update, skills, errors):
        super()._merge_update(source, update, skills, errors)
        self.candidates.extend((source, skill) for skill in update.get("skills_metadata", []))


def list_skills(*, workspace=".", skill_dirs=None, discover=True, include_shadowed=False):
    """Inspect local sources without constructing a graph or execution backend.

    Missing optional sources and source failures follow runtime policy. Invalid
    individual files retain upstream log diagnostics. ``skills`` contains the
    complete upstream metadata, its source, host collection path, and active flag.
    """
    root = resolve_skill_dirs([str(workspace)])[0]
    files = FilesystemBackend(root_dir=root, virtual_mode=True)
    backend, sources, optional = prepare_skill_backend(
        CompositeBackend(default=StateBackend(), routes={"/workspace/": files}),
        (), discover_skills=discover, skill_dirs=skill_dirs,
    )
    loader = _InventoryMiddleware(backend=backend, sources=sources, optional=optional)
    state = loader.before_agent({}, None, {})
    effective = {skill["name"]: skill for skill in state["skills_metadata"]}
    records = []
    for source, skill in loader.candidates:
        active = effective[skill["name"]] is skill
        if not active and not include_shadowed:
            continue
        if source == BUNDLED_SKILLS_PATH:
            kind, host = "bundled", None
        elif source == USER_SKILLS_PATH:
            kind, host = "personal", str(backend.routes[source].cwd)
        elif source.startswith(EXTERNAL_SKILLS_PATH):
            kind, host = "explicit", str(backend.routes[source].cwd)
        else:
            kind, host = "project", str(files.cwd / ".agents/skills")
        records.append({
            **skill, "source": source, "source_kind": kind,
            "host_path": host, "active": active,
        })
    return {"skills": records, "warnings": state["skills_load_errors"]}


def render_catalog(records):
    """Render plain metadata as a deterministic Markdown table."""
    def cell(value):
        value = html.escape(str(value or "—"), quote=True)
        for char in "\\|[]*_`":
            value = value.replace(char, f"&#{ord(char)};")
        return " ".join(value.splitlines())

    columns = ("Name", "Description", "Authors", "Maintainers", "License", "Version", "Status")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for skill in sorted(records, key=lambda skill: skill["name"]):
        metadata = skill["metadata"]
        values = (skill["name"], skill["description"], metadata.get("authors"),
                  metadata.get("maintainers"), skill.get("license"),
                  metadata.get("version"), metadata.get("status"))
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    return "\n".join(lines) + "\n"
