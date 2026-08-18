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
# ALCF inference API base URLs. Each cluster is a separate endpoint.
# Default: Sophia (NVIDIA A100, vLLM).
ALCF_DEFAULT_BASE_URL = (
    "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)
# Minerva (NVIDIA B200).
ALCF_MINERVA_BASE_URL = (
    "https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1"
)
# Metis (SambaNova SN40L). Chat completions only -- ALCF does not offer tool
# calling on this cluster, so ChemGraph's tool-driven workflows cannot use it.
ALCF_METIS_BASE_URL = (
    "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
)

# ALCF models -- all use the "alcf:" prefix (e.g. "alcf:inkling-bf16").
# The prefix routes the request inside ChemGraph and is stripped before the
# model name is sent to the endpoint. The cluster serving a model decides its
# base URL; anything not listed under another cluster below uses Sophia.
# See https://docs.alcf.anl.gov/services/inference-endpoints/#available-models
supported_alcf_models = [
    # -- Sophia --------------------------------------------------------------
    # Meta Llama Family
    "alcf:meta-llama/Meta-Llama-3.1-8B-Instruct",
    "alcf:meta-llama/Meta-Llama-3.1-70B-Instruct",
    "alcf:meta-llama/Meta-Llama-3.1-405B-Instruct",
    "alcf:meta-llama/Llama-3.3-70B-Instruct",
    "alcf:meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "alcf:meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # Mistral Family
    "alcf:mistralai/Mistral-Large-Instruct-2407",
    "alcf:mistralai/Mixtral-8x22B-Instruct-v0.1",
    "alcf:mistralai/Devstral-2-123B-Instruct-2512",
    # OpenAI Family
    "alcf:openai/gpt-oss-20b",
    "alcf:openai/gpt-oss-120b",
    # Aurora GPT Family
    "alcf:argonne/AuroraGPT-IT-v4-0125",
    "alcf:argonne/AuroraGPT-Tulu3-SFT-0125",
    "alcf:argonne/AuroraGPT-DPO-UFB-0225",
    "alcf:argonne/AuroraGPT-KTO-UFB-0325",
    # Google Family
    "alcf:google/gemma-3-27b-it",
    "alcf:google/gemma-4-26B-A4B-it",
    "alcf:google/gemma-4-31B-it",
    "alcf:google/gemma-4-E4B-it",
    # Other Models
    "alcf:allenai/Llama-3.1-Tulu-3-405B",
    "alcf:arcee-ai/Trinity-Large-Thinking-W4A16",
    "alcf:nvidia/nemotron-3-super-120b",
    "alcf:mgoin/Nemotron-4-340B-Instruct-hf",
    "alcf:AstroMLab/AstroSage-70B-20251009",
    # Vision Language Models
    "alcf:meta-llama/Llama-3.2-90B-Vision-Instruct",
    # -- Minerva -------------------------------------------------------------
    "alcf:nemotron-3-ultra",
    "alcf:inkling-bf16",
    # -- Metis (no tool calling) ---------------------------------------------
    "alcf:gpt-oss-120b",
    "alcf:Mistral-Large-3-675B-Instruct-2512",
    "alcf:gemma-4-31B-it",
]

# Subsets of supported_alcf_models served by a cluster other than the Sophia
# default, so they resolve to that cluster's base URL. Check which cluster
# serves a model with:
#   curl -H "Authorization: Bearer $ALCF_ACCESS_TOKEN" \
#     https://inference-api.alcf.anl.gov/resource_server/list-endpoints
supported_alcf_minerva_models = [
    "alcf:nemotron-3-ultra",
    "alcf:inkling-bf16",
]
# Metis reuses upstream names that Sophia carries with an org prefix
# ("alcf:gpt-oss-120b" here vs "alcf:openai/gpt-oss-120b" on Sophia), so the
# two are distinct entries and route to different clusters.
supported_alcf_metis_models = [
    "alcf:gpt-oss-120b",
    "alcf:Mistral-Large-3-675B-Instruct-2512",
    "alcf:gemma-4-31B-it",
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
    "argo:gpt-4o",
    "argo:gpt-4.1",
    "argo:gpt-4.1-mini",
    "argo:gpt-4.1-nano",
    "argo:gpt-5",
    "argo:gpt-5-mini",
    "argo:gpt-5-nano",
    "argo:gpt-5.1",
    "argo:gpt-5.2",
    "argo:gpt-5.4",
    "argo:gpt-5.4-mini",
    "argo:gpt-5.4-nano",  
    "argo:gpt-5.5",
    "argo:gpt-5.6-sol",
    "argo:gpt-5.6-terra",
    "argo:gpt-5.6-luna",
    # Reasoning / o-series
    "argo:o1",
    "argo:o3-mini",
    "argo:o3",
    "argo:o4-mini",
    # Gemini via Argo
    "argo:gemini-2.5-pro",
    "argo:gemini-2.5-flash",
    "argo:gemini-3.1-flash-lite",
    "argo:gemini-3.5-flash",
    # Claude via Argo
    "argo:claude-opus-4.8",
    "argo:claude-opus-4.7",
    "argo:claude-opus-4.6",
    "argo:claude-opus-4.5",
    "argo:claude-opus-4.1",
    "argo:claude-haiku-4.5",
    "argo:claude-sonnet-5",
    "argo:claude-sonnet-4.6",
    "argo:claude-sonnet-4.5",
]

# Exact Argo model routes that require minimal ChatOpenAI construction.
# Optional sampling parameters are omitted for every entry in this set. Add
# entries only after validating them against the deployed endpoint.
MODELS_WITHOUT_TEMPERATURE = frozenset(
    {
        "argo:claude-sonnet-5",
        "argo:gpt-5.6-luna",
        "argo:gpt-5.6-sol",
        "argo:gpt-5.6-terra",
        "argo:gpt-5.5",
        "argo:gpt-5",
        "argo:gpt-5-mini",
        "argo:gpt-5-nano",

    }
)
MODELS_WITH_REASONING_EFFORT = frozenset(
    {
        "argo:gpt-5.6-luna",
        "argo:gpt-5.6-sol",
        "argo:gpt-5.6-terra",
    }
)
SUPPORTED_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)

all_supported_models = (
    supported_openai_models
    + supported_ollama_models
    + supported_alcf_models
    + supported_anthropic_models
    + supported_argo_models
    + supported_gemini_models
    + supported_groq_models
)
