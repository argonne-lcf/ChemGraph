!!! note
      Before exploring example usage in the `notebooks/` directory, ensure you have specified the necessary API tokens in your environment. 

=== "OpenAI API Key"
      1. Log in to your OpenAI account at the OpenAI Platform website. If you don't have an account, you'll need to create one first.

      2. Navigate to the API keys section. You can find this by clicking on your profile icon in the top-right corner and selecting "API keys."

      3. Click the + Create new secret key button.

      4. Give your key a descriptive name (e.g., "ChemGraph").

      5. Click Create secret key. A new key will be generated.

      6. Copy the key and save it in a secure location. You will not be able to see it again after this step.

      7. Set the key in your environment using the command provided in the instructions:
         ```bash
         export OPENAI_API_KEY="your_api_key_here"  # On Unix or macOS
         setx OPENAI_API_KEY "your_api_key_here"  # On Windows
         ```
      8. Restart your terminal or IDE to ensure the environment variable is loaded.

=== "Anthropic API Key"
      1. Sign up or log in to your Anthropic account at the [Anthropic console](https://console.anthropic.com/).

      2. In the left-hand navigation menu, select API Keys.

      3. Click on the option to create a new API key.

      4. Provide a name for your API key (e.g., "ChemGraph").

      5. Click Create Key again.

      6. Copy the generated key and store it securely, as you may not be able to view it again.

      7. Set the key in your environment using the command provided in the instructions:
         ```bash
         export ANTHROPIC_API_KEY="your_api_key_here"  # On Unix or macOS
         setx ANTHROPIC_API_KEY "your_api_key_here"  # On Windows
         ```
      8. Restart your terminal or IDE to ensure the environment variable is loaded.

=== "Google AI Studio (Gemini) API Key"
      1. Go to the Google AI Studio website at [Google AI Studio](https://ai.google.com/studio) and sign in with your Google account.

      2. In the left-hand menu, select Get API key.

      3. Click the Create API key in new project button. A new key will be instantly generated.

      4. Copy the API key by clicking the copy icon next to it.

      5. Set the key as an environment variable:
         ```bash
         export GEMINI_API_KEY="your_api_key_here"  # On Unix or macOS
         setx GEMINI_API_KEY "your_api_key_here"  # On Windows
         ```
      6. Restart your terminal or IDE to ensure the environment variable is loaded.

???+ info "**Explore Example Notebooks**"
      Navigate to the `notebooks/` directory to explore various example notebooks demonstrating different capabilities of ChemGraph.

      - **[Single-Agent System with MACE](https://github.com/argonne-lcf/ChemGraph/blob/main/notebooks/1_Demo_single_agent.ipynb)**: This notebook demonstrates how a single agent can utilize multiple tools with MACE/xTB support.

      - **[Single-Agent System with UMA](https://github.com/argonne-lcf/ChemGraph/blob/main/notebooks/Demo_single_agent_UMA.ipynb)**: This notebook demonstrates how a single agent can utilize multiple tools with UMA support.

      - **[Multi-Agent System](https://github.com/argonne-lcf/ChemGraph/blob/main/notebooks/Demo-multi_agent.ipynb)**: This notebook demonstrates a multi-agent setup where different agents (Planner, Executor and Aggregator) handle various tasks exemplifying the collaborative potential of ChemGraph.

      - **[Model Context Protocol (MCP) Server](https://github.com/argonne-lcf/ChemGraph/blob/main/notebooks/3_MCP_server.ipynb)**: This notebook shows how to run and connect to ChemGraph MCP tooling.

      - **[Single-Agent System with gRASPA](https://github.com/argonne-lcf/ChemGraph/blob/main/notebooks/Demo_graspa_agent.ipynb)**: This notebook provides a sample guide on executing a gRASPA simulation using a single agent. For gRASPA-related installation instructions, visit the [gRASPA GitHub repository](https://github.com/snurr-group/gRASPA). The notebook's functionality has been validated on a single compute node at ALCF Polaris.
