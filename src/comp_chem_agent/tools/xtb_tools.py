from langchain_core.tools import tool
from comp_chem_agent.models.xtb import XTBSimulationInput
import subprocess
import shlex

@tool
def setup_xtb_input() -> XTBSimulationInput:
    return True

@tool
def run_xtb_calculation(input: XTBSimulationInput, mode: str = "local") -> bool:
    """
    Run geometry optimization using XTB.

    Args:
        input (XTBSimulationInput): Input parameters for the XTB simulation.
        mode (str): Execution mode, e.g., 'local'. Defaults to 'local'.

    Returns:
        bool: True if the simulation performs correctly, False otherwise.
    """
    try:
        # Build the XTB command
        command = (
            f"xtb test.xyz --o {input.opt} "
            f"--gfn {input.gfn} --chrg {input.chrg} --acc {input.acc}"
        )
        print(command)
        # Use shlex to safely split the command into parts
        command_args = shlex.split(command)

        # Execute the command and capture the output
        result = subprocess.run(command_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print("XTB calculation output:")
        print(result.stdout)

        return True
    except subprocess.CalledProcessError as e:
        print("Error during XTB calculation:")
        print(e.stderr)
        return False
    except ValueError as ve:
        print(f"Input validation error: {ve}")
        return False
    except Exception as ex:
        print(f"Unexpected error: {ex}")
        return False

