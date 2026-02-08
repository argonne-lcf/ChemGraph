import pytest
from chemgraph.agent.llm_agent import ChemGraph

@pytest.mark.llm
@pytest.mark.asyncio
async def test_single_agent_graph():
    """
    Test that the agent can take a molecule name, 
    convert it to SMILES, and update the state.
    """
    cg = ChemGraph(model_name="gpt-4o-mini", workflow_type="single_agent", return_option="state")
    
    # We use a real chemistry query that requires molecule_name_to_smiles
    query= "What is the SMILES for Aspirin?"
    config = {"configurable": {"thread_id": "test_single_agent"}}

    # Execute the full graph until it hits END
    result = await cg.run(query, config=config)

    # Check that the state contains the tool output
    # The message history should have: User, AI (tool call), Tool (result), AI (final)
    #content = result.content
    content = result["messages"][-1]["content"]
    assert len(result["messages"]) >= 2
    assert "CC(=O)OC1=CC=CC=C1C(=O)O" in content
    assert "Aspirin" in content

@pytest.mark.llm
@pytest.mark.asyncio
async def test_multi_agent_graph():
    """
    Test that the agent can take a molecule name, 
    convert it to SMILES, and update the state.
    """
    cg = ChemGraph(model_name="gpt-4o-mini", workflow_type="multi_agent", return_option="state")
    
    # We use a real chemistry query that requires molecule_name_to_smiles
    query= "What are the SMILES strings for carbon dioxide, nitrogen dioxide and methanol?"
    config = {"configurable": {"thread_id": "test_multi_agent"}}

    # Execute the full graph until it hits END
    result = await cg.run(query, config=config)
    content = result["messages"][-1]["content"]
    assert "C(=O)=O*" in content
    assert "CO" in content
