"""Optional Academy inbox glue; applications own their ChemGraph workflows."""

from chemgraph.academy.message import PeerMessage
from chemgraph.academy.tools import make_peer_message_tool

__all__ = ["ChemGraphPeer", "PeerMessage", "make_peer_message_tool"]


def __getattr__(name):
    if name == "ChemGraphPeer":
        from chemgraph.academy.peer import ChemGraphPeer

        return ChemGraphPeer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
