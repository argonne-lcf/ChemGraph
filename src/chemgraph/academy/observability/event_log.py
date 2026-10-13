"""Compatibility imports for the legacy campaign runtime."""
from chemgraph.observability.event_log import EventLog, EventKind, RunEvent, read_events

CampaignEvent = RunEvent
__all__ = ["CampaignEvent", "EventLog", "EventKind", "read_events"]
