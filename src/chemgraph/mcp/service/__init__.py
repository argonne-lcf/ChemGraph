"""Opt-in authenticated MCP service. Importing this package starts nothing."""


def create_app(config):
    from chemgraph.mcp.service.app import create_app as factory

    return factory(config)
