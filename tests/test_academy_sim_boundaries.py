from __future__ import annotations

from pathlib import Path


def test_academy_sim_does_not_import_chemgraph_academy():
    root = Path("src/chemgraph/academy_sim")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "chemgraph.academy." in text or "from chemgraph.academy import" in text:
            offenders.append(str(path))

    assert offenders == []


def test_chemgraph_runtime_owns_mcp_loading_boundary():
    runtime_text = Path("src/chemgraph/agent/graph_runtime.py").read_text(
        encoding="utf-8"
    )
    launcher_text = Path("src/chemgraph/academy_sim/launcher.py").read_text(
        encoding="utf-8"
    )

    assert "load_mcp_tools" in runtime_text
    assert "load_mcp_tools" not in launcher_text
    assert "run_turn" not in runtime_text


def test_peer_send_tools_end_current_graph_turn():
    launcher_text = Path("src/chemgraph/academy_sim/launcher.py").read_text(
        encoding="utf-8"
    )

    assert "terminal_tool_names=tuple(tool.name for tool in peer_tools)" in launcher_text
    assert "mark_graph_done" not in launcher_text
    assert "completion_tools" not in launcher_text
