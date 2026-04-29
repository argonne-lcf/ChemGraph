"""
Lists of supported models for different LLM providers.
"""

# OpenAI models that are supported
supported_openai_models = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "gpt-5",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-3.5-turbo-0125",
]
# Ollama models that are supported
supported_ollama_models = ["llama3.2", "llama3.1"]
# Default ALCF inference API base URL (Sophia cluster, vLLM).
ALCF_DEFAULT_BASE_URL = (
    "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)

# ALCF models available through the ALCF inference endpoints.
# See https://docs.alcf.anl.gov/services/inference-endpoints/#available-models
supported_alcf_models = [
    # Meta Llama Family
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # Mistral Family
    "mistralai/Mistral-Large-Instruct-2407",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Devstral-2-123B-Instruct-2512",
    # OpenAI Family
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    # Aurora GPT Family
    "argonne/AuroraGPT-IT-v4-0125",
    "argonne/AuroraGPT-Tulu3-SFT-0125",
    "argonne/AuroraGPT-DPO-UFB-0225",
    "argonne/AuroraGPT-KTO-UFB-0325",
    # Google Family
    "google/gemma-3-27b-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
    "google/gemma-4-E4B-it",
    # Other Models
    "allenai/Llama-3.1-Tulu-3-405B",
    "arcee-ai/Trinity-Large-Thinking-W4A16",
    "nvidia/nemotron-3-super-120b",
    "mgoin/Nemotron-4-340B-Instruct-hf",
    "AstroMLab/AstroSage-70B-20251009",
    # Vision Language Models
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
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
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

# GROQ models -- use the "groq:" prefix (e.g. "groq:llama-3.3-70b-versatile").
# The prefix is stripped before sending to the Groq API.
# No curated list is maintained; any model available on Groq can be used.
# See https://console.groq.com/docs/models for current models.
supported_groq_models: list[str] = []

# Default Argo API base URL (used when no --base-url is provided).
ARGO_DEFAULT_BASE_URL = "https://apps.inside.anl.gov/argoapi/v1"

# Argo models -- all use the "argo:" prefix.
# Which endpoint they hit depends on --base-url / config.
# Default: ARGO_DEFAULT_BASE_URL (Argo API).
supported_argo_models = [
    # GPT family
    "argo:gpt-3.5-turbo",
    "argo:gpt-3.5-turbo-16k",
    "argo:gpt-4",
    "argo:gpt-4-32k",
    "argo:gpt-4-turbo",
    "argo:gpt-4o",
    "argo:gpt-4o-latest",
    "argo:gpt-4o-mini",
    "argo:gpt-4.1",
    "argo:gpt-4.1-mini",
    "argo:gpt-4.1-nano",
    "argo:gpt-5",
    "argo:gpt-5-mini",
    "argo:gpt-5-nano",
    "argo:gpt-5.1",
    "argo:gpt-5.2",
    "argo:gpt-5.4",
    # Reasoning / o-series
    "argo:o1-preview",
    "argo:o1-mini",
    "argo:o1",
    "argo:o3-mini",
    "argo:o3",
    "argo:o4-mini",
    # Gemini via Argo
    "argo:gemini-2.5-pro",
    "argo:gemini-2.5-flash",
    # Claude via Argo
    "argo:claude-opus-4.6",
    "argo:claude-opus-4.5",
    "argo:claude-opus-4.1",
    "argo:claude-opus-4",
    "argo:claude-haiku-4.5",
    "argo:claude-sonnet-4.5",
    "argo:claude-sonnet-4",
    "argo:claude-sonnet-3.5-v2",
    "argo:claude-haiku-3.5",
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
