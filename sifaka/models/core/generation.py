"""
Text generation functionality for the ModelProviderCore class.

This module provides functions for generating text using a ModelProviderCore
instance, including input validation, configuration handling, and cache management.
"""

from typing import Any, Dict, TYPE_CHECKING
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from .provider import ModelProviderCore
logger = get_logger(__name__)


def process_input(provider: "ModelProviderCore", prompt: str, **kwargs) -> Any:
    """
    Process the input prompt and generate text.

    This function processes an input prompt and generates text using
    a ModelProviderCore instance. It handles input validation, configuration
    overrides, and cache management.

    Args:
        provider: The model provider instance
        prompt: The prompt to generate from
        **kwargs: Optional overrides for model configuration
            - temperature: Control randomness (0-1)
            - max_tokens: Maximum tokens to generate
            - api_key: Override API key
            - trace_enabled: Override tracing setting

    Returns:
        The generated text

    Raises:
        TypeError: If prompt is not a string
        ValueError: If prompt is empty or API key is missing
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt or not prompt.strip():
        raise ValueError("prompt cannot be empty")
    cache = provider._state_manager.get("cache", {})
    cache_key = f"{prompt}_{str(kwargs)}"
    if cache_key in cache:
        provider._state_manager.set_metadata("cache_hit", True)
        return cache[cache_key]
    config = ModelConfig(
        temperature=kwargs.pop("temperature", provider.config.temperature),
        max_tokens=kwargs.pop("max_tokens", provider.config.max_tokens),
        api_key=kwargs.pop("api_key", provider.config.api_key),
        trace_enabled=kwargs.pop("trace_enabled", provider.config.trace_enabled),
    )
    if not config.api_key:
        model_specific_env = (
            f"{provider.__class__.__name__.replace('Provider', '').upper()}_API_KEY"
        )
        raise ValueError(
            f"API key is missing. Please provide an API key either by setting the {model_specific_env} environment variable or by passing it explicitly via the api_key parameter or config."
        )
    generation_service = provider._state_manager.get("generation_service")
    result = generation_service.generate(prompt, config)
    if len(cache) < 100:
        cache[cache_key] = result
        provider._state_manager.update("cache", cache)
    return result


def get_generation_service(provider: "ModelProviderCore") -> Any:
    """
    Get the generation service from the provider's state.

    This function retrieves the generation service from a ModelProviderCore
    instance's state, ensuring that the provider is initialized first.

    Args:
        provider: The model provider instance

    Returns:
        The generation service

    Raises:
        RuntimeError: If the generation service is not found
    """
    if not provider._state_manager.get("initialized", False):
        provider.warm_up()
    generation_service = provider._state_manager.get("generation_service")
    if not generation_service:
        raise RuntimeError("Generation service not found")
    return generation_service


def clear_cache(provider: "ModelProviderCore") -> None:
    """
    Clear the generation cache.

    This function clears the generation cache for a ModelProviderCore instance.

    Args:
        provider: The model provider instance
    """
    provider._state_manager.update("cache", {})


def get_cache_stats(provider: "ModelProviderCore") -> Any:
    """
    Get cache statistics.

    This function retrieves cache statistics for a ModelProviderCore instance,
    including cache size and hit rate.

    Args:
        provider: The model provider instance

    Returns:
        A dictionary containing cache statistics
    """
    cache = provider._state_manager.get("cache", {})
    cache_hits = provider._state_manager.get_metadata("cache_hit_count", 0)
    cache_misses = provider._state_manager.get_metadata("cache_miss_count", 0)
    total_requests = cache_hits + cache_misses
    return {
        "cache_size": len(cache),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "hit_rate": cache_hits / total_requests if total_requests > 0 else 0,
    }
