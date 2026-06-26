"""
LLM Factory — returns the correct LLM instance based on the chosen provider.

Usage:
    from generation.llm_factory import get_llm
    llm = get_llm()                          # uses Config.LLM_PROVIDER
    llm = get_llm(provider="gemini")         # force Gemini
    llm = get_llm(provider="groq")           # force Groq
    llm = get_llm(provider="gemini", model_name="gemini-1.5-pro")  # custom model
"""

import logging
from utils.config import Config

logger = logging.getLogger(__name__)

# Catalogue of all supported providers and their available models
PROVIDERS = {
    "auto": {
        "label": "Auto (Best Available)",
        "models": ["auto"],
        "default_model": "auto",
    },
    "groq": {
        "label": "Groq",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "label": "Google Gemini",
        "models": [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
        ],
        "default_model": "gemini-1.5-flash",
    },
    "openrouter": {
        "label": "OpenRouter",
        "models": [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-haiku",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-7b-instruct",
            "google/gemini-flash-1.5",
        ],
        "default_model": "openai/gpt-4o-mini",
    },
}

# Priority order for auto-selection (first match with a valid key wins)
_AUTO_PRIORITY = ["openrouter", "gemini", "groq"]


def resolve_auto_provider() -> str:
    """
    Detect which API keys are configured and return the best available
    provider key.  Priority: OpenRouter → Gemini → Groq.

    Returns:
        Provider key string (e.g. "openrouter").

    Raises:
        ValueError: If no provider has a usable API key.
    """
    key_map = {
        "openrouter": Config.OPENROUTER_API_KEY,
        "gemini":     Config.GEMINI_API_KEY,
        "groq":       Config.GROQ_API_KEY,
    }
    placeholder_values = {"", "your_groq_api_key_here",
                          "your_google_ai_api_key_here",
                          "your_openrouter_api_key_here"}

    for provider in _AUTO_PRIORITY:
        key = (key_map.get(provider) or "").strip()
        if key and key not in placeholder_values:
            logger.info(f"Auto-select resolved to provider: {provider}")
            return provider

    raise ValueError(
        "Auto-select failed: no API keys are configured. "
        "Please set at least one of GROQ_API_KEY, GEMINI_API_KEY, "
        "or OPENROUTER_API_KEY in your .env file."
    )


def get_llm(
    provider: str = None,
    model_name: str = None,
    temperature: float = None,
    max_tokens: int = None,
):
    """
    Return an LLM instance for the requested provider.

    Args:
        provider:    "groq", "gemini", or "openrouter". Falls back to Config.LLM_PROVIDER.
        model_name:  Override the model within that provider.
        temperature: Sampling temperature (0-1).
        max_tokens:  Maximum response tokens.

    Returns:
        A GroqLLM, GeminiLLM, or OpenRouterLLM instance with a compatible interface.

    Raises:
        ValueError: If an unsupported provider name is requested.
    """
    chosen = (provider or Config.LLM_PROVIDER).lower().strip()

    # Resolve "auto" to a concrete provider
    if chosen == "auto":
        chosen = resolve_auto_provider()
        model_name = None  # let the resolved provider use its own default

    if chosen == "groq":
        from generation.llm import GroqLLM
        logger.info(f"LLM factory: creating GroqLLM (model={model_name or Config.LLM_MODEL_NAME})")
        return GroqLLM(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif chosen == "gemini":
        from generation.gemini_llm import GeminiLLM
        logger.info(f"LLM factory: creating GeminiLLM (model={model_name or Config.GEMINI_MODEL_NAME})")
        return GeminiLLM(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif chosen == "openrouter":
        from generation.openrouter_llm import OpenRouterLLM
        logger.info(f"LLM factory: creating OpenRouterLLM (model={model_name or Config.OPENROUTER_MODEL_NAME})")
        return OpenRouterLLM(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{chosen}'. "
            f"Supported providers: {list(PROVIDERS.keys())}"
        )


def list_providers() -> dict:
    """
    Return the catalogue of all supported providers and their models.

    Returns:
        Dict mapping provider keys to metadata (label, models, default_model).
    """
    return PROVIDERS
