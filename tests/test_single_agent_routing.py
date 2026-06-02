from langchain_core.messages import AIMessage

from chemgraph.graphs.single_agent import route_report_tools, route_tools


def test_route_report_tools_routes_to_tool_before_report_exists():
    state = {
        "messages": [
            AIMessage(
                content="calling report tool",
                tool_calls=[
                    {
                        "name": "generate_html",
                        "args": {"output_path": "/tmp/report.html", "ase_output": {}},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }

    assert route_report_tools(state) == "tools"


def test_route_report_tools_stops_after_successful_report_generation():
    state = {
        "messages": [
            {"name": "generate_html", "content": "/app/cg_logs/report.html"},
            AIMessage(
                content="calling report tool again",
                tool_calls=[
                    {
                        "name": "generate_html",
                        "args": {"output_path": "/app/cg_logs/report.html", "ase_output": {}},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }

    assert route_report_tools(state) == "done"


def test_route_tools_stops_on_repeated_identical_tool_cycle():
    state = {
        "messages": [
            AIMessage(
                content="first call",
                tool_calls=[
                    {
                        "name": "molecule_name_to_smiles",
                        "args": {"name": "H2O"},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "smiles_to_coordinate_file",
                        "args": {"smiles": "O"},
                        "id": "call_2",
                        "type": "tool_call",
                    },
                ],
            ),
            {"name": "molecule_name_to_smiles", "content": '{"name":"H2O","smiles":"O"}'},
            {"name": "smiles_to_coordinate_file", "content": '{"ok":true}'},
            AIMessage(
                content="same calls again",
                tool_calls=[
                    {
                        "name": "molecule_name_to_smiles",
                        "args": {"name": "H2O"},
                        "id": "call_3",
                        "type": "tool_call",
                    },
                    {
                        "name": "smiles_to_coordinate_file",
                        "args": {"smiles": "O"},
                        "id": "call_4",
                        "type": "tool_call",
                    },
                ],
            ),
        ]
    }

    assert route_tools(state) == "done"


def test_route_tools_continues_on_new_tool_args():
    state = {
        "messages": [
            AIMessage(
                content="first call",
                tool_calls=[
                    {
                        "name": "molecule_name_to_smiles",
                        "args": {"name": "H2O"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            {"name": "molecule_name_to_smiles", "content": '{"name":"H2O","smiles":"O"}'},
            AIMessage(
                content="new args",
                tool_calls=[
                    {
                        "name": "molecule_name_to_smiles",
                        "args": {"name": "caffeine"},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }

    assert route_tools(state) == "tools"
