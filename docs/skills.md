# Deep Agent skills

ChemGraph ships two Agent Skills: **chemgraph** for its Python, CLI, and chemistry
MCP workflows, and **pbs-hpc** for PBS job preparation, monitoring, and facility
guidance. They are available to standalone Deep Agents, registry-created Deep
Agent workers, and the optional `main_agent` workspace worker.

For direct submission from a login-node shell, see [PBS jobs with skills](pbs_jobs_with_skills.md).
Deep Agent writes calculation and batch scripts using the skills and existing
Python APIs; it does not need attached chemistry MCP tools for this route.

## Where skills live

Maintained skills live in `src/chemgraph/skills/` and ship in the Python wheel
and source distribution. Each skill is an immediate child directory containing
`SKILL.md`, with YAML `name` and `description` fields:

```text
skills/
└── example-workflow/
    ├── SKILL.md
    ├── references/          # optional documentation
    ├── assets/              # optional templates/data
    └── scripts/             # optional executable helpers
```

Use the [Agent Skills format](https://agentskills.io/specification). Keep skill
directories flat at the source root; place longer supporting material under the
skill. Add future bundled skills as siblings and include their resources in
package data. Keep facility accounts, private paths, and endpoints in deployment
configuration, not in bundled instructions.

For local workspaces, sources load in this order; later sources override earlier
ones with the same skill name:

| Source | Storage | Agent-visible path |
| --- | --- | --- |
| Bundled | Installed ChemGraph package | `/chemgraph-skills/` |
| Personal | `~/.chemgraph/skills/` | `/chemgraph-user-skills/` |
| Project | `<workspace>/.agents/skills/` | `/workspace/.agents/skills/` with the CLI backend |
| Explicit host | CLI directories or Python `skill_dirs` | `/chemgraph-external-skills/<stable-id>/` |
| Explicit backend | Python `skills` | Paths supplied by the caller |

The personal and project directories are optional and are not created on startup.
Shared `~/.agents/skills/` collections can be configured explicitly. Do not keep
another maintained copy of the bundled skills in project directories.

```sh
# Bundled, personal, and project skills are available automatically.
chemgraph run --interactive -w deep_agent --deepagent-workspace .

# Add a site collection, overriding earlier sources with matching names.
chemgraph run --interactive -w deep_agent --deepagent-workspace . \
  --deepagent-skill ../shared-skills/

# Load only bundled and explicitly configured skills.
chemgraph run --interactive -w deep_agent --deepagent-workspace . \
  --no-deepagent-discover-skills
```

In `[general]` TOML, use `deepagent_discover_skills = false` to disable automatic
local discovery. `deepagent_skills` remains an ordered list of additional paths.
CLI skill paths replace that TOML list, while the discovery CLI flag overrides
the TOML boolean. An empty explicit list leaves bundled/discovered skills enabled.

CLI skill paths are host directories independent of `--deepagent-workspace`.
Relative paths (including `../`) resolve against the directory where you launch
ChemGraph; absolute paths, `~`, spaces, and existing symlinks are supported.
TOML paths follow the same rule. ChemGraph resolves and validates each directory,
then exposes it through a stable backend route without copying files or creating
symlinks. Supported saved sessions retain canonical host paths and rebuild these
routes when resumed from a different directory. Missing or unreadable explicit
directories fail with a path-specific error when Deep Agent is selected. Other
interactive workflows retain saved directories without checking their access.
Switching into Deep Agent validates them; if validation fails, the current
workflow stays active. Restore the directory and retry the switch. Relative
paths keep their invocation-directory meaning throughout the session.

```sh
chemgraph run --interactive --workflow deep_agent --deepagent-workspace . \
  --deepagent-skill ../external/AtomisticSkills/.agents/skills/ \
  --model argo:gpt-5.6-luna
```

Pass the collection directory containing `<skill-name>/SKILL.md`; this does not
recursively search a repository or install its environments and tools. For
AtomisticSkills, the Widom insertion skill is `chem-sorption-widom`.

**CLI migration:** `/workspace/...` is now a literal host path when passed to
`--deepagent-skill`. Replace older virtual-path examples with the actual host
path or a path relative to your invocation directory. Agent file tools still
use `/workspace/...` for project files.

## Python and other filesystems

```python
from deepagents.backends import LocalShellBackend
from chemgraph.graphs.deep_agent import construct_deep_agent_graph

graph = construct_deep_agent_graph(
    model,
    backend=LocalShellBackend(root_dir="/path/to/project", env={}),
    discover_skills=True,
    skill_dirs=["../external/AtomisticSkills/.agents/skills/"],
    skills=["/workspace/site-skills/"],
)
```

The corresponding `ChemGraph` and `construct_main_agent_graph` options are
`deepagent_discover_skills`, `deepagent_skill_dirs`, and `deepagent_skills`.
`skill_dirs` explicitly mounts host collections with any backend, even when
automatic discovery is disabled. `skills` retains its backend-relative meaning
and has precedence over host collections. `user_skills_dir` on the
constructor (`deepagent_user_skills_dir` on the higher-level APIs) fixes or
overrides the personal directory for a supported local workspace; saved sessions
use this to retain the original resolved root.

Automatic host directory discovery applies to `LocalShellBackend`,
`FilesystemBackend`, and composites with an identifiable virtual local
`/workspace/` route. Pure virtual filesystem backends use `/.agents/skills/` for
their project source; non-virtual local backends use the absolute project path.
For state, store, and arbitrary remote backends, bundled resources are still
available. Use `skill_dirs` to explicitly mount host collections, or `skills`
for sources already mapped into the backend.
There is no implicit inspection of the agent host's home for remote backends.

| Backend | File operations | `execute` |
| --- | --- | --- |
| `StateBackend` | Conversation state/checkpointer | No shell |
| `StoreBackend` | Configured LangGraph store | No shell |
| `FilesystemBackend` | Host filesystem | No shell |
| `LocalShellBackend` | Host filesystem | Host shell |
| Sandbox/custom executor | Provider's filesystem | Provider's execution environment |
| `CompositeBackend` | Routed by file path | Default backend's executor |

Custom file backends implement `BackendProtocol`. Execution additionally needs
`SandboxBackendProtocol`. ChemGraph's own Parsl/Globus execution backends implement
a separate `ExecutionBackend` interface; selecting one does not create a remote
Deep Agent shell. See the [upstream backend guide](https://docs.langchain.com/oss/python/deepagents/backends).

Bundled resources are loaded using `importlib.resources` and exposed through a
read-only route. They require neither a checkout nor files seeded in graph state.
File operations and downloads work with local, state, store, and remote defaults;
write/edit/delete/upload operations on this route fail. The bundled backend does
not expose host paths or execute commands. Reserved skill routes must not overlap
caller-supplied composite routes. Existing routes, the default executor, and the
artifact root are retained without an extra composite layer.

## Reading skills and executing helpers

Deep Agents 0.7.5 advertises names, descriptions, and paths in the prompt, then
uses **`read_file`** to load `SKILL.md` on demand. There is no separate
`read_skill` tool. Relative references in a skill resolve against its directory
in the file backend. See [upstream skills documentation](https://docs.langchain.com/oss/python/deepagents/skills).

External filesystem collections also appear in the prompt's "Shell paths vs.
virtual paths" mapping when using a local shell. Use these host paths for helper
scripts; a command in an external skill may assume its own repository root, so
resolve script paths against that collection rather than the workspace. With a
remote executor, transfer required files to its filesystem first.

The initial bundles contain text instructions and a PBS template. To execute a
future helper stored in the catalog, first copy it into the executor's filesystem.
For example, read the PBS template at
`/chemgraph-skills/pbs-hpc/assets/job.pbs.template`, fill its placeholders, and
write it to `/workspace/job.pbs`. With the CLI's local backend, execute commands
using the host workspace path supplied in the system prompt. With a sandbox,
write/upload to a path that actually exists in that sandbox before execution.
Virtual skill/state/store paths are not shell mounts, and MCP servers can have
different filesystems again.

Personal, project, and explicit host skill routes are writable: file-tool edits
modify the original host collection, including shared checkouts. Existing
write/edit/delete and execute approvals apply; disabling approvals also permits
these skill edits without review. Updated instructions can affect future turns.
Bundled resources remain read-only through their route, which does not confine
a local shell. Skill frontmatter does not grant additional tools or permissions.
Standalone and main-agent workspace Deep Agents share
`DEFAULT_DEEPAGENT_PROMPT`, which permits using attached chemistry tools. Tools
are configured separately; the built-in main-agent workspace worker has no
chemistry tools attached. A custom `PromptConfig.deepagent` is preserved
verbatim; its default `None` selects the shared prompt.

## On-demand local tools

Skills describe workflows; tools implement their operations. With standalone
`deep_agent`, the built-in local catalog is searchable by default without sending
every tool schema to the model:

```sh
chemgraph run --interactive -w deep_agent --deepagent-workspace .
```

To restrict discovery, use repeated `--tool NAME` flags or `[general]` TOML
`tools = ["smiles_to_coordinate_file", "file_to_atomsdata"]`. CLI names replace
the TOML list loaded with `--config`, and duplicates are collapsed. Omitting the
setting selects the built-ins except interactive tools such as `ask_human`, which
require `--human-supervised`. Explicitly listing `ask_human` also opts in.
`tools = []` disables discovery. This is independent of
`--no-deepagent-discover-skills`, which controls personal/project skills only.
The CLI displays the catalog size or disabled status at initialization.
The option applies to standalone Deep Agent; configured names are ignored for
other noninteractive workflows. Interactive sessions validate and retain configured
names even when starting in another workflow, so startup selection, model changes,
and workflow switches preserve the restriction.

`ChemGraph(workflow_type="deep_agent")` uses the same default catalog, enabling
interactive tools when `human_supervised=True`. Explicit catalogs and attached
tools count as deliberate opt-in, independently of that flag.
Pass `deepagent_tool_registry=preparation` to replace it, or
`deepagent_tool_registry=ToolRegistry([])` to disable it. Explicit `None` selects
the default. The lower-level shared constructor remains opt-in so existing
delegated workers do not gain tools; supply a catalog explicitly:

```python
from chemgraph.registry import ToolRegistry

catalog = ToolRegistry()
preparation = ToolRegistry(catalog.get_spec(name) for name in (
    "smiles_to_coordinate_file", "file_to_atomsdata", "extract_output_json",
))
graph = construct_deep_agent_graph(model, backend=backend, tool_registry=preparation)
```

Existing `tools=` objects, including MCP tools, remain attached as before. Their
names are excluded from the automatic catalog. Names in an explicitly supplied
catalog must not collide with attached tools or discovery tools. Custom tools can
be registered using the existing `ToolSpec`/`BaseTool` interfaces. Registering a
`BaseTool` uses an already-created object; `ToolSpec` defers importing its module.

The agent initially sees workspace tools plus `search_tools` and `load_tools`.
Search returns bounded name/description matches without imports. Skills can name
tools directly, skipping search. Use native tools for supported local operations
before considering scripts or reading their implementation source. Scripts remain
appropriate for unavailable capabilities and separate batch jobs.
`load_tools(names)` replaces the active selection and exposes native schemas on
the next model call; `load_tools([])` clears it.
It returns names rather than duplicating schemas in chat. Missing dependencies or
unknown names leave the previous selection intact. Active names are checkpointed
per conversation, survive approval pauses, and clear at the end of a completed
turn. Restore pending checkpoints with the same catalog. The built-in delegated
worker has its own tools; keep this preparation workflow in the standalone agent.

Registry tools execute on the agent host, independently of the file/shell backend.
Use absolute host paths and returned artifact paths. The default approval policy
also covers `smiles_to_coordinate_file` and `save_atomsdata_to_file`; custom tools
need appropriate `interrupt_on` entries when constructing the graph. `python_repl`
requires the same execution review as `execute`. Explicit approval overrides
retain their existing meaning.

On-demand loading reduces repeated schema input for larger catalogs, but adds a
round trip. A few always-attached tools can be cheaper for a short task. Keep
results compact and large artifacts in files; do not equate schema bytes with
billed tokens or assume a fixed saving. This route needs neither an extra LLM
selector nor provider-specific tool search.

## Sessions and diagnostics

Skill metadata is rediscovered before each new agent turn, including after graph
reconstruction; an old checkpoint catalog cannot hide newly added skills. The
default general-purpose child receives the same discovery middleware. Refreshing
does not restart an interrupted file mutation or bypass its pending approval.
Independently compiled external subagents retain their own configuration; use
the shared ChemGraph constructor for a Deep Agent worker that needs this catalog.

Durable main-agent metadata retains the discovery setting and resolved personal
root alongside existing workspace/source settings. Legacy session records default
to local discovery disabled. Bundled skills remain available. Adding/editing
skills is reflected on the next new turn, not midway through a pending approval.

The resolved personal root is part of the topology fingerprint. Moving a session
database to another account or container retains the original account's path;
it does not adopt the new home directory. To use a different personal root, start
a new session with that configuration. Changing the root while restoring the
existing session produces an incompatible-topology error.

Missing optional directories are skipped. With Deep Agents 0.7.5, invalid YAML or
missing required metadata causes a skill to be skipped with log warnings; valid
siblings remain available. Name/directory mismatches are accepted with warnings.
These file-level warnings do not populate `skills_load_errors`. If a skill is
absent from the catalog, check the logs and inspect its `SKILL.md` frontmatter
using `read_file` at the collection's backend path.

Optional sources rejected by the backend (for example, a project skill directory
symlinked outside the virtual workspace) are skipped
with a warning; other sources remain available. These failures do not relax the
backend's filesystem boundaries. Warnings clear after the source recovers on a
new turn.
If an optional backend returns partial results with an error, valid skills are
retained alongside the warning and still override earlier sources.

Unreadable explicitly configured sources raise a clear configuration error before
the model call. Explicit state/store sources must contain files before each turn
starts: seed state files in the graph input or populate the store first. These
backends cannot distinguish missing directories from empty ones, so both are
rejected. Existing empty filesystem directories remain valid; custom backends
retain their reported-error semantics. Populated sources with invalid skill
metadata retain upstream diagnostics. Missing or invalid bundled skills fail
graph construction, so a broken installation cannot silently lose defaults.

Availability does not guarantee that an LLM follows every instruction. Enforce
mandatory scientific or operational requirements in tools or middleware.
