"""OCSR graph: read a molecule's structure diagram and return its SMILES.

Reuses the single-agent graph setup, changing only the default tool list and the
default system prompt so the agent is focused on reading structure images. Keeping
the OCSR tools in their own graph, instead of adding them to the single-agent
defaults, leaves the general single-agent workflow and its evaluations untouched.
That is the same reasoning :mod:`chemgraph.graphs.molecular_docking` gives.

The tools are built per graph, not imported as a fixed list, because
``image_to_smiles`` closes over the agent's own ``llm`` to serve as the vision
fallback. ``validate_smiles`` and ``list_ocsr_models`` have no such dependency and
are module-level.

The tool list is deliberately small. Reading the image is the whole job here: the
agent returns a SMILES and the user decides what to do with it. Adding
``smiles_to_coordinate_file`` and ``run_ase`` would make this a general chemistry
workflow that happens to read images, which is what ``single_agent`` already is once
a caller passes these tools to it::

    ChemGraph(workflow_type="single_agent",
              tools=[*make_ocsr_tools(llm), smiles_to_coordinate_file, run_ase])
"""

from collections.abc import Collection

from langchain_openai import ChatOpenAI

from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.prompt.ocsr_prompt import ocsr_agent_prompt
from chemgraph.prompt.single_agent_prompt import formatter_prompt, report_prompt
from chemgraph.tools.ocsr_tools import (
    list_ocsr_models,
    make_ocsr_tools,
    validate_smiles,
)


def default_ocsr_tools(llm) -> list:
    """The OCSR tool set, with ``llm`` bound as the vision fallback.

    ``image_to_smiles`` reads the picture. ``validate_smiles`` checks a SMILES the
    agent proposes itself, the case ``image_to_smiles`` cannot cover since its own
    return already carries validity. ``list_ocsr_models`` lets the agent answer "what
    can this machine run" from the machine instead of from memory.
    """
    return [*make_ocsr_tools(llm), validate_smiles, list_ocsr_models]


def construct_ocsr_graph(
    llm: ChatOpenAI,
    system_prompt: str = ocsr_agent_prompt,
    structured_output: bool = False,
    formatter_prompt: str = formatter_prompt,
    generate_report: bool = False,
    report_prompt: str = report_prompt,
    tools: list | None = None,
    max_retries: int = 1,
    human_supervised: bool = False,
    terminal_tool_names: Collection[str] = (),
):
    """Construct the OCSR graph.

    Identical to :func:`construct_single_agent_graph` except the default tool list is
    the OCSR tool set and the default system prompt is the OCSR prompt.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use for the graph, and the vision fallback for
        ``image_to_smiles``.
    system_prompt : str, optional
        System prompt, by default :data:`ocsr_agent_prompt`.
    tools : list, optional
        Tools for the agent. Defaults to :func:`default_ocsr_tools` when ``None``.
    (remaining parameters match :func:`construct_single_agent_graph`.)

    Returns
    -------
    StateGraph
        The constructed OCSR graph.
    """
    if tools is None:
        tools = default_ocsr_tools(llm)
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
