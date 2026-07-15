"""Academy-side MCP tool wrappers used by the demo.

Each submodule is an independently-launchable FastMCP server:

    python -m chemgraph.academy.tools.pacmof2_mcp --transport streamable_http \
        --host 127.0.0.1 --port <PORT>

Follows the same CLI convention as mofforge.mcp.server, chosen to match
chemgraph.academy.runtime.mcp_supervisor's spawn contract. Kept small
and heavy-dependency-isolated so agents that don't need a given tool
don't pay its import cost.

Replacement status (2026-07-14):
  - ``pacmof2_mcp``      demo scaffolding; drop when ChemGraph ships
                         an official PACMOF2 MCP.
  - ``graspa_dummy_mcp`` demo scaffolding; drop when the real gRASPA
                         MCP (``chemgraph.mcp.graspa_mcp_hpc`` on
                         dev-graspa) is stable and packaged. Campaign
                         swaps the ``command`` string to move.
  - ``globus_flex_mcp``  novel, keep. Per-call flexible source/dest
                         alternative to chemgraph's manager-bound
                         upstream ``chemgraph.mcp.transfer_tools``.
"""
