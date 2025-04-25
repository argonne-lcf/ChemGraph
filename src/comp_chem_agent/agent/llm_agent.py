from langgraph.prebuilt import create_react_agent
from comp_chem_agent.tools.local_model_loader import load_ollama_model
from comp_chem_agent.tools.ASE_tools import (
    run_ase,
    molecule_name_to_smiles,
    file_to_atomsdata,
    smiles_to_atomsdata,
)
from comp_chem_agent.tools.alcf_loader import load_alcf_model
from comp_chem_agent.tools.openai_loader import load_openai_model
from comp_chem_agent.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class CompChemAgent:
    def __init__(
        self,
        model_name="gpt-4o-mini",
        tools=None,
        prompt=None,
        base_url=None,
        api_key=None,
        temperature=0,
    ):
        try:
            if model_name in ["gpt-3.5-turbo", "gpt-4o-mini"]:
                llm = load_openai_model(
                    model_name=model_name, api_key=api_key, temperature=temperature
                )
                logger.info(f"Loaded {model_name}")
            elif model_name in ["llama3.2", "llama3.1"]:
                llm = load_ollama_model(model_name=model_name, temperature=temperature)
                logger.info(f"Loaded {model_name}")
            else:
                llm = load_alcf_model(
                    model_name=model_name, base_url=base_url, api_key=api_key
                )
                logger.info(f"Loaded {model_name}")

        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            raise

        if prompt is None:
            system_message = "You are a helpful assistant."

        tools = [
            smiles_to_atomsdata,
            run_ase,
            molecule_name_to_smiles,
            file_to_atomsdata,
        ]

        self.llm = llm
        self.graph = create_react_agent(llm, tools, state_modifier=system_message)

    def run(self, query):
        try:
            inputs = {"messages": [("user", query)]}
            for s in self.graph.stream(inputs, stream_mode="values"):
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    logger.info(message)
                else:
                    logger.info(message.content)
        except Exception as e:
            logger.error(f"Error running query: {str(e)}")
            raise

    def runq(self, query):
        try:
            messages = self.llm.invoke(query)
            logger.info(messages)
            return messages
        except Exception as e:
            logger.error(f"Error in runq: {str(e)}")
            raise

    def return_input(self, query, simulation_class):
        structured_llm = self.llm.with_structured_output(simulation_class)
        messages = structured_llm.invoke(query)
        return messages
