# Deep Agent skills

ChemGraph ships two Agent Skills: **chemgraph** for its Python, CLI, and chemistry
MCP workflows, and **pbs-hpc** for PBS job preparation, monitoring, and facility
guidance. They are available to standalone Deep Agents, registry-created Deep
Agent workers, and the optional `main_agent` workspace worker.

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
| Explicit | Additional backend-relative directories | Paths supplied by the caller |

The personal and project directories are optional and are not created on startup.
Shared `~/.agents/skills/` collections can be configured explicitly. Do not keep
another maintained copy of the bundled skills in project directories.

```sh
# Bundled, personal, and project skills are available automatically.
chemgraph run --interactive -w deep_agent --deepagent-workspace .

# Add a site collection, overriding earlier sources with matching names.
chemgraph run --interactive -w deep_agent --deepagent-workspace . \
  --deepagent-skill /workspace/site-skills/

# Load only bundled and explicitly configured skills.
chemgraph run --interactive -w deep_agent --deepagent-workspace . \
  --no-deepagent-discover-skills
```

In `[general]` TOML, use `deepagent_discover_skills = false` to disable automatic
local discovery. `deepagent_skills` remains an ordered list of additional paths.
CLI skill paths replace that TOML list, while the discovery CLI flag overrides
the TOML boolean. An empty explicit list leaves bundled/discovered skills enabled.

## Python and other filesystems

```python
from deepagents.backends import LocalShellBackend
from chemgraph.graphs.deep_agent import construct_deep_agent_graph

graph = construct_deep_agent_graph(
    model,
    backend=LocalShellBackend(root_dir="/path/to/project", env={}),
    discover_skills=True,
    skills=["/workspace/site-skills/"],
)
```

The corresponding `ChemGraph` and `construct_main_agent_graph` options are
`deepagent_discover_skills` and `deepagent_skills`. `user_skills_dir` on the
constructor (`deepagent_user_skills_dir` on the higher-level APIs) fixes or
overrides the personal directory for a supported local workspace; saved sessions
use this to retain the original resolved root.

Automatic host directory discovery applies to `LocalShellBackend`,
`FilesystemBackend`, and composites with an identifiable virtual local
`/workspace/` route. Pure virtual filesystem backends use `/.agents/skills/` for
their project source; non-virtual local backends use the absolute project path.
For state, store, and arbitrary remote backends, bundled resources are still
available, but additional sources must be explicitly mapped into the backend.
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

The initial bundles contain text instructions and a PBS template. To execute a
future helper stored in the catalog, first copy it into the executor's filesystem.
For example, read the PBS template at
`/chemgraph-skills/pbs-hpc/assets/job.pbs.template`, fill its placeholders, and
write it to `/workspace/job.pbs`. With the CLI's local backend, execute commands
using the host workspace path supplied in the system prompt. With a sandbox,
write/upload to a path that actually exists in that sandbox before execution.
Virtual skill/state/store paths are not shell mounts, and MCP servers can have
different filesystems again.

The read-only route does not confine a local shell. Existing write/execute
approvals remain in force. Skill frontmatter does not grant additional tools or
permissions. The standalone default prompt permits attached chemistry tools;
the main-agent workspace worker continues to delegate simulations to the
chemistry specialist. A custom `PromptConfig.deepagent` is preserved verbatim;
its default `None` selects the appropriate prompt for the workflow.

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

Missing optional directories are skipped. Invalid optional skill files produce
upstream diagnostics. Unreadable explicitly configured sources raise a clear
configuration error before the model call. Missing or invalid bundled skills
fail graph construction, so a broken installation cannot silently lose defaults.

Availability does not guarantee that an LLM follows every instruction. Enforce
mandatory scientific or operational requirements in tools or middleware.
