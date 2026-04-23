"""Multi-agent MCP workflow — thin wrapper around multi_agent.py.

With the Send()-based architecture, LangGraph's ``ToolNode`` handles
both sync LangChain tools and async MCP tools natively.  There is no
longer a need for a separate ``AsyncBasicToolNode`` or a distinct graph
structure.  This module re-exports ``construct_multi_agent_graph`` under
the legacy ``construct_multi_agent_mcp_graph`` name so that existing
references in ``llm_agent.py`` and configuration continue to work.
"""

from chemgraph.graphs.multi_agent import construct_multi_agent_graph


def construct_multi_agent_mcp_graph(**kwargs):
    """Construct the multi-agent MCP graph.

    Delegates entirely to :func:`construct_multi_agent_graph`.
    All keyword arguments are forwarded unchanged.
    """
    return construct_multi_agent_graph(**kwargs)
