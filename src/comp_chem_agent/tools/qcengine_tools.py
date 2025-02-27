# from comp_chem_agent.state.state import MultiAgentState
from comp_chem_agent.state.opt_vib_state import MultiAgentState
from langchain_core.messages import HumanMessage
import json
from langchain.tools import tool
import qcengine
import numpy as np
from comp_chem_agent.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@tool
def run_qcengine(state: MultiAgentState, program="psi4"):
    atomic_numbers = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
    }

    params = state["parameter_response"][-1]
    input = json.loads(params.content)
    program = input["program"]

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert NumPy array to list
            if isinstance(obj, np.generic):
                return obj.item()  # Convert NumPy scalar to Python scalar
            return super().default(obj)

    def parse_atomsdata_to_mol(input):
        numbers = input["atomsdata"]["numbers"]
        positions = input["atomsdata"]["positions"]

        # Convert atomic numbers to element symbols
        symbols = [atomic_numbers[num] for num in numbers]

        # Flatten positions list for QCEngine format
        geometry = [coord for position in positions for coord in position]
        return {"symbols": symbols, "geometry": geometry}

    input["molecule"] = parse_atomsdata_to_mol(input)
    del input["atomsdata"]
    del input["program"]
    try:
        logger.info("Starting QCEngine calculation")
        result = qcengine.compute(input, program).dict()
        del result["stdout"]
        output = []
        output.append(
            HumanMessage(role="system", content=json.dumps(result, cls=NumpyEncoder))
        )
        logger.info("QCEngine calculation completed successfully")
        return {"opt_response": output}
    except Exception as e:
        logger.error(f"Error in QCEngine calculation: {str(e)}")
        raise
