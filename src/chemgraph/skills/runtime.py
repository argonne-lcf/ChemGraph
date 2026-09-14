"""Skill routing and per-turn discovery shared by ChemGraph Deep Agents."""

import hashlib
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.backends.protocol import LsResult
from deepagents.middleware.skills import SkillsMiddleware

from chemgraph.skills.backend import BundledSkillsBackend


BUNDLED_SKILLS_PATH = "/chemgraph-skills/"
USER_SKILLS_PATH = "/chemgraph-user-skills/"
EXTERNAL_SKILLS_PATH = "/chemgraph-external-skills/"
logger = logging.getLogger(__name__)


def resolve_skill_dirs(skill_dirs: Sequence[str] | None) -> tuple[str, ...]:
    """Freeze explicit host collections against the caller's working directory."""
    if skill_dirs is None:
        return ()
    if isinstance(skill_dirs, (str, bytes)):
        raise TypeError("skill_dirs must be a sequence of path strings, not a string.")
    resolved_dirs = []
    for source in skill_dirs:
        if not isinstance(source, str):
            raise TypeError("Every skill directory must be a string.")
        if not source.strip():
            raise ValueError("Skill directory paths must not be empty.")
        resolved = source
        try:
            resolved = Path(source).expanduser().absolute()
            resolved = resolved.resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError("Expected a skill collection directory.")
            # Opening the listing checks actual access, including ACL failures.
            with os.scandir(resolved):
                pass
        except (OSError, ValueError, RuntimeError) as exc:
            hint = (
                " CLI skill paths are host paths; replace virtual /workspace/ "
                "with the actual workspace directory."
                if source == "/workspace" or source.startswith("/workspace/")
                else ""
            )
            raise ValueError(
                f"Cannot access skill directory {source!r} "
                f"(resolved to {str(resolved)!r}): {exc}.{hint}"
            ) from exc
        canonical = str(resolved)
        resolved_dirs = [path for path in resolved_dirs if path != canonical]
        resolved_dirs.append(canonical)
    return tuple(resolved_dirs)


def local_skill_workspace(backend):
    """Return (host root, backend root) only for identifiable local workspaces."""
    if isinstance(backend, CompositeBackend):
        workspace = backend.routes.get("/workspace/", backend.routes.get("/workspace"))
        if isinstance(workspace, FilesystemBackend) and workspace.virtual_mode:
            return workspace.cwd, "/workspace/"
        return None
    if isinstance(backend, FilesystemBackend):
        return (
            backend.cwd,
            "/" if backend.virtual_mode else backend.cwd.as_posix() + "/",
        )
    return None


def resolve_user_skills_dir(backend, discover_skills, user_skills_dir=None):
    """Freeze the personal root so a saved session does not follow a new home."""
    if not discover_skills or local_skill_workspace(backend) is None:
        return None
    return str(
        Path(user_skills_dir or Path.home() / ".chemgraph/skills")
        .expanduser()
        .resolve()
    )


def prepare_skill_backend(
    backend, skills, *, discover_skills=True, user_skills_dir=None, skill_dirs=None
):
    """Mount skill collections while preserving the workspace and executor."""
    if not isinstance(discover_skills, bool):
        raise TypeError("discover_skills must be a boolean.")
    routes = dict(backend.routes) if isinstance(backend, CompositeBackend) else {}
    # Upstream accepts slashless routes for reads, but listing path remapping
    # assumes the final character is a slash. Normalize only our copied map.
    if "/workspace" in routes and "/workspace/" not in routes:
        routes["/workspace/"] = routes.pop("/workspace")
    for route in routes:
        for reserved in (BUNDLED_SKILLS_PATH, USER_SKILLS_PATH, EXTERNAL_SKILLS_PATH):
            prefix = route.rstrip("/") + "/"
            if prefix.startswith(reserved) or reserved.startswith(prefix):
                raise ValueError(f"Skill route conflicts with reserved path: {route}")
    routes[BUNDLED_SKILLS_PATH] = BundledSkillsBackend()
    optional = {}
    sources = [BUNDLED_SKILLS_PATH]
    workspace = local_skill_workspace(backend)
    user_root = resolve_user_skills_dir(backend, discover_skills, user_skills_dir)
    if user_root is not None:
        routes[USER_SKILLS_PATH] = FilesystemBackend(
            root_dir=user_root, virtual_mode=True
        )
        project_source = workspace[1] + ".agents/skills/"
        optional = {
            USER_SKILLS_PATH: Path(user_root),
            project_source: workspace[0] / ".agents/skills",
        }
        sources.extend(optional)
    for root in resolve_skill_dirs(skill_dirs):
        digest = hashlib.sha256(root.encode("utf-8")).hexdigest()
        route = f"{EXTERNAL_SKILLS_PATH}{digest}/"
        routes[route] = FilesystemBackend(root_dir=root, virtual_mode=True)
        sources.append(route)
    # Explicit duplicates belong at their final, highest-priority position.
    for source in skills:
        sources = [
            existing
            for existing in sources
            if existing.rstrip("/") != source.rstrip("/")
        ]
        sources.append(source)
        optional.pop(source, None)
        optional.pop(source.rstrip("/") + "/", None)
    composite = CompositeBackend(
        default=backend.default if isinstance(backend, CompositeBackend) else backend,
        routes=routes,
        artifacts_root=backend.artifacts_root
        if isinstance(backend, CompositeBackend)
        else "/",
    )
    return composite, sources, optional


class ChemGraphSkillsMiddleware(SkillsMiddleware):
    """Refresh the catalog between turns, including after checkpoint restoration.

    Keep the upstream middleware name so Deep Agents replaces its default slot
    and passes this implementation to the general-purpose child. Each call uses
    a separate loader; concurrent threads never mutate shared source lists.
    """

    @property
    def name(self):
        return "SkillsMiddleware"

    def __init__(self, *, backend, sources, optional):
        super().__init__(backend=backend, sources=sources)
        self.optional = optional

    def _loaders(self, errors):
        for source in self.sources:
            if source in self.optional:
                try:
                    self.optional[source].stat()
                except FileNotFoundError:
                    continue
                except (OSError, ValueError, RuntimeError) as exc:
                    error = f"Cannot inspect optional skills at {source}: {exc}"
                    logger.warning("%s", error)
                    errors.append(error)
                    continue
            yield source, SkillsMiddleware(backend=self._backend, sources=[source])

    def _empty_source_backend(self, source, update):
        if (
            source in self.optional
            or update["skills_metadata"]
            or update.get("skills_load_errors")
        ):
            return None
        backend, path = self._backend, source
        # Use the same routing rules as file tools, including nested composites.
        while isinstance(backend, CompositeBackend):
            backend, path = backend._get_backend_and_key(path)
        if isinstance(backend, (StateBackend, StoreBackend)):
            return backend, path
        return None

    @staticmethod
    def _require_files(listing):
        if isinstance(listing, LsResult):
            if listing.error:
                raise ValueError(listing.error)
            listing = listing.entries
        if not listing:
            raise ValueError(
                "Explicit state/store skill sources must contain files before "
                "the turn starts; the source is missing or empty."
            )

    def _merge_update(self, source, update, skills, errors):
        source_errors = update.get("skills_load_errors", [])
        if source_errors:
            if source not in self.optional:
                raise ValueError("; ".join(source_errors))
            for error in source_errors:
                logger.warning("%s", error)
            errors.extend(source_errors)
            return
        for skill in update["skills_metadata"]:
            skills[skill["name"]] = skill

    def before_agent(self, state, runtime, config):
        skills, errors = {}, []
        for source, loader in self._loaders(errors):
            try:
                update = loader.before_agent({}, runtime, config)
                if target := self._empty_source_backend(source, update):
                    backend, path = target
                    self._require_files(backend.ls(path))
            except (OSError, ValueError, RuntimeError) as exc:
                update = {"skills_load_errors": [
                    f"Cannot load skills from '{source}': {exc}"
                ]}
            self._merge_update(source, update, skills, errors)
        return {"skills_metadata": list(skills.values()), "skills_load_errors": errors}

    async def abefore_agent(self, state, runtime, config):
        skills, errors = {}, []
        for source, loader in self._loaders(errors):
            try:
                update = await loader.abefore_agent({}, runtime, config)
                if target := self._empty_source_backend(source, update):
                    backend, path = target
                    self._require_files(await backend.als(path))
            except (OSError, ValueError, RuntimeError) as exc:
                update = {"skills_load_errors": [
                    f"Cannot load skills from '{source}': {exc}"
                ]}
            self._merge_update(source, update, skills, errors)
        return {"skills_metadata": list(skills.values()), "skills_load_errors": errors}
