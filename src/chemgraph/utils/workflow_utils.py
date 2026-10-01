"""Migration guidance for removed workflow names."""

REMOVED_WORKFLOWS = {
    "python_relp": "deep_agent",
    "python_repl": "deep_agent",
}


def get_removed_workflow_message(name: str) -> str | None:
    """Return migration guidance for a removed workflow, if any."""
    replacement = REMOVED_WORKFLOWS.get(name)
    if replacement is None:
        return None
    return (
        f"Workflow '{name}' has been removed. Use '{replacement}' for Python "
        "execution and start a new session. See "
        "https://argonne-lcf.github.io/ChemGraph/workflows/#migrating-from-python-repl"
    )
