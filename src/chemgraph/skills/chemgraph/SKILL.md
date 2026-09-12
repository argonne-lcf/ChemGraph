---
name: chemgraph
description: Use ChemGraph Python and CLI workflows and attached chemistry MCP tools. Use for molecular simulations, ASE calculations, staging structures, tracking calculation batches, and reporting computed results.
---

# Use ChemGraph

1. Identify the requested calculation, input structures, calculator/model,
   execution system, and output location. Inspect attached tool schemas before
   selecting arguments. Ask for missing scientific choices rather than inventing
   them; do not replace the requested calculator with another one silently.
2. Follow your assigned role. A standalone Deep Agent can use attached chemistry
   tools. A workspace worker under `main_agent` prepares files and delegates
   simulations back to the `chemgraph` specialist through its supervisor.
   This skill does not grant tools or override approvals.
3. For Python or CLI setup, read [interface guidance](references/python-and-cli.md).
   For calculations through attached tools, read
   [MCP workflow guidance](references/mcp-workflows.md).
4. Establish where every file lives: the agent's virtual filesystem, its shell,
   the MCP server, or the compute worker. A path readable with `read_file` need
   not exist on another host. Use existing staging tools when files must move.
5. Submit the requested calculation once. Save returned job or batch identifiers.
   When a calculation is pending, poll the available status tools or return the
   identifier and pending state; do not resubmit merely because it is unfinished.
6. Report actual tool results, including failures, units, calculator/model, and
   output paths. For ASE, prefer `potential_energy`; `single_point_energy` is a
   legacy compatibility field. Distinguish optimization convergence from job
   completion. Never invent numerical results or label pending work successful.

For scheduler scripts or allocations, also read the `pbs-hpc` skill from the
available skill catalog. Facility-specific values belong in user configuration.

## Resource paths

Resolve these relative references against this SKILL.md's directory using file
tools. `/chemgraph-skills/` is a virtual, read-only resource route. To use a
template or future helper with `execute`, first write/copy it into the execution
backend's workspace and use that environment's actual path. State/store files
are not automatically visible to a shell or an MCP server.
