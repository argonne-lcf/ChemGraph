import chainlit.cli
import os
import sys

if __name__ == "__main__":
    # Add src to the Python path to allow for absolute imports
    sys.path.insert(0, os.path.abspath("src"))

    # Set the target app
    target = os.path.join("src", "cg_ui", "app.py")

    # Run chainlit programmatically
    sys.argv = ["chainlit", "run", target, "--watch"]
    chainlit.cli.cli()
