"""Protocol builders: the sole construction sites for LangChain chat clients.

Each protocol module owns exactly one client class. Endpoints prepare fully
formed keyword arguments and hand them here; these builders add no
endpoint-specific logic.
"""
