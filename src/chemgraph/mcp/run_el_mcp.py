from .el_mcp_tools import create_mcp, register_tools, start_el

if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    try:
        el = start_el()
        mcp = create_mcp()
        register_tools(mcp)
        run_mcp_server(mcp, default_port=9003)
    finally:
        el.stop()
