"""Skill routing and per-turn discovery shared by ChemGraph Deep Agents."""

import logging
from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware

from chemgraph.skills.backend import BundledSkillsBackend


BUNDLED_SKILLS_PATH = "/chemgraph-skills/"
USER_SKILLS_PATH = "/chemgraph-user-skills/"
logger = logging.getLogger(__name__)


def local_skill_workspace(backend):
    """Return (host root, backend root) only for identifiable local workspaces."""
    if isinstance(backend, CompositeBackend):
        workspace = backend.routes.get("/workspace/")
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
    backend, skills, *, discover_skills=True, user_skills_dir=None
):
    """Add package/personal routes without obscuring shell-to-workspace mappings."""
    if not isinstance(discover_skills, bool):
        raise TypeError("discover_skills must be a boolean.")
    routes = dict(backend.routes) if isinstance(backend, CompositeBackend) else {}
    for route in routes:
        for reserved in (BUNDLED_SKILLS_PATH, USER_SKILLS_PATH):
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

    def _loader(self):
        sources = []
        for source in self.sources:
            if source in self.optional:
                try:
                    self.optional[source].stat()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    logger.warning(
                        "Cannot inspect optional skills at %s: %s", source, exc
                    )
                    continue
            sources.append(source)
        return SkillsMiddleware(backend=self._backend, sources=sources)

    def _checked_update(self, update):
        errors = update.get("skills_load_errors", [])
        # Missing optional roots were filtered out. Fail clearly for explicitly
        # requested sources; state-backed paths are checked inside the graph.
        required_errors = [
            error
            for error in errors
            if not any(
                error.startswith(f"Cannot load skills from '{source}':")
                for source in self.optional
            )
        ]
        if required_errors:
            raise ValueError("; ".join(required_errors))
        update["skills_load_errors"] = errors
        return update

    def before_agent(self, state, runtime, config):
        update = self._loader().before_agent({}, runtime, config)
        return self._checked_update(update)

    async def abefore_agent(self, state, runtime, config):
        update = await self._loader().abefore_agent({}, runtime, config)
        return self._checked_update(update)
