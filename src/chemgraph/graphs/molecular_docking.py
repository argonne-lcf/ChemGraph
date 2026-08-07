"""Molecular docking graph.

Reuses the single-agent graph setup, changing only the default tool list and the
default system prompt so the agent is focused on molecular docking. Keeping docking
tools in their own graph (rather than in the single-agent defaults) means the general
single-agent workflow and its evaluations are unaffected.

The graph is intentionally easy to grow: add docking-related tools (receptor
preparation, pocket/box detection, redocking validation, pose visualization, ...) to
the default tool list below as they become available.
"""

from collections.abc import Collection

from langchain_openai import ChatOpenAI

from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.prompt.molecular_docking_prompt import molecular_docking_prompt
from chemgraph.prompt.single_agent_prompt import formatter_prompt, report_prompt
from chemgraph.tools.cheminformatics_tools import molecule_name_to_smiles
from chemgraph.tools.docking_tools import run_docking

# Default tools for the molecular docking workflow. Only tools helpful when studying
# ligand-receptor binding are included; general gas-phase/DFT tools are omitted.
DEFAULT_DOCKING_TOOLS = [run_docking, molecule_name_to_smiles]


def construct_molecular_docking_graph(
    llm: ChatOpenAI,
    system_prompt: str = molecular_docking_prompt,
    structured_output: bool = False,
    formatter_prompt: str = formatter_prompt,
    generate_report: bool = False,
    report_prompt: str = report_prompt,
    tools: list | None = None,
    max_retries: int = 1,
    human_supervised: bool = False,
    terminal_tool_names: Collection[str] = (),
):
    """Construct the molecular docking graph.

    Identical to :func:`construct_single_agent_graph` except the default tool list is
    the docking tool set and the default system prompt is the docking prompt.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use for the graph.
    system_prompt : str, optional
        System prompt, by default :data:`molecular_docking_prompt`.
    tools : list, optional
        Tools for the agent. Defaults to :data:`DEFAULT_DOCKING_TOOLS` when ``None``.
    (remaining parameters match :func:`construct_single_agent_graph`.)

    Returns
    -------
    StateGraph
        The constructed molecular docking graph.
    """
    if tools is None:
        tools = list(DEFAULT_DOCKING_TOOLS)
    return construct_single_agent_graph(
        llm,
        system_prompt=system_prompt,
        structured_output=structured_output,
        formatter_prompt=formatter_prompt,
        generate_report=generate_report,
        report_prompt=report_prompt,
        tools=tools,
        max_retries=max_retries,
        human_supervised=human_supervised,
        terminal_tool_names=terminal_tool_names,
    )
