---
name: chemgraph
description: Use ChemGraph Python and CLI workflows, agent-written batch scripts, and attached chemistry MCP tools. Use for molecular simulations, ASE calculations, staging structures, tracking calculations, and reporting computed results.
license: Apache-2.0
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# Use ChemGraph

1. Identify the requested calculation, input structures, calculator/model,
   execution system, and output location. Preserve explicit scientific choices.
   For omitted choices, use configured or schema defaults for the selected
   execution environment; ask only for required information that remains
   unresolved. Inspect the selected interface's schemas before choosing arguments.
2. Use native chemistry tools for supported local operations. When discovery is
   available, search/load the needed tools before considering scripts or reading
   their implementation source. Scripts can reuse ChemGraph's Python APIs when
   a capability is unavailable or a separate batch script is needed.
   Report missing dependencies. This skill does not grant tools or override
   approvals.
3. Preserve the user's selected execution method. Otherwise, choose a route based
   on the available tools and execution host:
   - Local chemistry tools: use [MCP workflow guidance](references/mcp-workflows.md)
     for attached tools, or [Python and CLI guidance](references/python-and-cli.md)
     when using those interfaces.
   - Direct PBS on an authorized submission host: use
     [the ASE batch example](references/ase-batch.md) and
     [pbs-hpc](../pbs-hpc/SKILL.md); no chemistry MCP server or Parsl is required.
   - ALCF submission through the IRI API, including from a host without `qsub`:
     use [iri-hpc](../iri-hpc/SKILL.md).
   When inputs need construction, follow
   [local structure preparation](references/structure-preparation.md).
4. Establish where every file lives: the agent's virtual filesystem, its shell,
   the MCP server, or the compute worker. A path readable with `read_file` need
   not exist on another host. Use existing staging tools when files must move.
5. Submit the requested calculation once. Save returned job or batch identifiers.
   For “submit a job,” report the accepted ID and observed status. For “calculate
   and return results,” monitor and collect the outputs; if still pending when
   reporting back, say results remain outstanding and retain the ID for resuming.
   Do not resubmit merely because a calculation is unfinished.
6. Report actual tool results, including failures, units, calculator/model, and
   output paths. For ASE, prefer `potential_energy`; `single_point_energy` is a
   legacy compatibility field. Distinguish optimization convergence from job
   completion. Never invent numerical results or label pending work successful.

Facility-specific values belong in user configuration.

## Resource paths

Resolve these relative references against this SKILL.md's directory using file
tools. `/chemgraph-skills/` is a virtual, read-only resource route. To use a
template or future helper with `execute`, first write/copy it into the execution
backend's workspace and use that environment's actual path. State/store files
are not automatically visible to a shell or an MCP server.
