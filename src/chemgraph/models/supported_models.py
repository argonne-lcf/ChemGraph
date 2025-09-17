"""
Lists of supported models for different LLM providers.
"""
# OpenAI models that are supported
supported_openai_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-3.5-turbo-0125",
]
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
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b"
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
# Gemini models. gemini-2.0 doesn't work with toolcall in our last test.
supported_gemini_models = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

# GROQ models
supported_groq_models = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "groq/compound",
    "groq/compound-mini",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "moonshotai/kimi-k2-instruct-0905",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
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

all_supported_models = (
    supported_openai_models
    + supported_ollama_models
    + supported_alcf_models
    + supported_anthropic_models
    + supported_argo_models
    + supported_gemini_models
    + supported_groq_models
)
