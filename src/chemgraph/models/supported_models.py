"""
Lists of supported models for different LLM providers.
"""

# OpenAI models that are supported
supported_openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-3.5-turbo-0125"]

# Ollama models that are supported
supported_ollama_models = ["llama3.2", "llama3.1"]

# ALCF models that are supported (these would be models available through ALCF's infrastructure)
supported_alcf_models = [
    "AuroraGPT-IT-v4-0125_2",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/QwQ-32B-Preview",
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
]

# Anthropic models
supported_anthropic_models = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

supported_argo_models = [
    "argo:gpt-3.5-turbo",
    "argo:gpt-3.5-turbo-16k",
    "argo:gpt-4",
    "argo:gpt-4-32k",
    "argo:gpt-4-turbo",
    "argo:gpt-4o",
    "argo:gpt-4o-latest",
    "argo:gpt-o1-preview",
    "argo:o1-preview",
    "argo:gpt-o1-mini",
    "argo:o1-mini",
    "argo:gpt-o3-mini",
    "argo:o3-mini",
    "argo:gpt-o1",
    "argo:o1",
    "argo:gpt-o3",
    "argo:o3",
    "argo:gpt-o4-mini",
    "argo:o4-mini",
    "argo:gpt-4.1",
    "argo:gpt-4.1-mini",
    "argo:gpt-4.1-nano",
]
