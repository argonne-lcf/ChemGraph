"""
Lists of supported models for different LLM providers.
"""

# OpenAI models that are supported
supported_openai_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "o3-mini",
    "o1-mini",
]

# Ollama models that are supported
supported_ollama_models = ["llama3.2", "llama3.1"]

# ALCF models that are supported (these would be models available through ALCF's infrastructure)
supported_alcf_models = [
    "AuroraGPT-IT-v4-0125_2",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]

# Anthropic models
supported_anthropic_models = [
    "claude-3-5-haiku-20241022",
]
