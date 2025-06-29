"""Sifaka's main API for text improvement through iterative critique.

This module provides the primary interfaces for using Sifaka:
- improve(): Async text improvement with sensible defaults
- improve_sync(): Synchronous wrapper for improve()
- improve_advanced(): Full control over all parameters

The API is designed to be simple for basic use cases while allowing
full customization for advanced users."""

from typing import List, Optional, Union, Dict, Any
import asyncio

from .core.models import SifakaResult
from .core.config import Config
from .core.interfaces import Validator
from .core.engine import SifakaEngine
from .core.constants import DEFAULT_CRITIC, DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL
from .core.retry import RetryConfig
from .core.middleware import MiddlewarePipeline
from .storage import StorageBackend


async def improve(
    text: str,
    *,
    critics: Optional[List[str]] = None,
    max_iterations: int = 3,
    validators: Optional[List[Validator]] = None,
    config: Optional[Config] = None,
    storage: Optional[StorageBackend] = None,
    middleware: Optional[MiddlewarePipeline] = None,
) -> SifakaResult:
    """Improve text through iterative critique and refinement.

    This is the main entry point for Sifaka. It applies selected critics
    to iteratively improve text until it meets quality standards or reaches
    the maximum iterations.

    Args:
        text: The text to improve. Can be any length, from a sentence
            to multiple paragraphs.
        critics: List of critic names to apply. Available critics include
            'reflexion', 'self_refine', 'constitutional', 'meta_rewarding',
            'self_consistency', 'n_critics', 'self_rag', and 'style'.
            Defaults to ['reflexion'] if not specified.
        max_iterations: Maximum number of improvement iterations. Each
            iteration applies all critics and generates an improved version.
            Valid range is 1-10. Defaults to 3.
        validators: Optional list of quality validators to ensure the text
            meets specific criteria (e.g., minimum length, no profanity).
            If not provided, a basic length validator (50 chars) is used.
        config: Configuration object for advanced settings including model
            selection, temperature, timeout, style parameters, and more.
            If not provided, uses default configuration.
        storage: Optional storage backend for persisting results. Supports
            FileStorage, RedisStorage, PostgresStorage, or custom backends.
            Defaults to FileStorage if not specified.
        middleware: Optional middleware pipeline for adding custom behavior
            like logging, monitoring, or transformation.

    Returns:
        SifakaResult containing:
        - original_text: The input text
        - final_text: The improved text after all iterations
        - critiques: List of all critique feedback
        - generations: List of all intermediate versions
        - metadata: Processing details and metrics

    Raises:
        ValueError: If invalid critic names or parameters are provided
        TimeoutError: If processing exceeds the configured timeout
        ModelProviderError: If the LLM API fails

    Example:
        >>> # Basic usage with default settings
        >>> result = await improve("Write about AI benefits")
        >>> print(result.final_text)
        
        >>> # Advanced usage with multiple critics
        >>> result = await improve(
        ...     "Explain quantum computing",
        ...     critics=["reflexion", "self_refine", "constitutional"],
        ...     max_iterations=5,
        ...     config=Config(model="gpt-4", temperature=0.8)
        ... )
        
        >>> # Style transformation
        >>> config = Config(
        ...     style_description="Casual blog style",
        ...     style_examples=["Hey there!", "Let me tell you..."]
        ... )
        >>> result = await improve(
        ...     formal_text,
        ...     critics=["style"],
        ...     config=config
        ... )

    Note:
        The improve() function is stateless and can be called concurrently
        for processing multiple texts in parallel.
    """
    # Set defaults
    if critics is None:
        critics = [DEFAULT_CRITIC]

    if config is None:
        config = Config()

    # Create engine config - copy all fields from user config
    engine_config = config.model_copy(
        update={
            "max_iterations": max_iterations,
            "critics": critics,
        }
    )

    # Run improvement
    engine = SifakaEngine(engine_config, storage)

    # If middleware is provided, use it
    if middleware:
        context = {
            "critics": critics,
            "validators": validators,
            "model": config.model,
            "temperature": config.temperature,
            "config": engine_config,
        }

        async def final_handler(text: str) -> SifakaResult:
            return await engine.improve(text, validators)

        return await middleware.execute(text, final_handler, context)
    else:
        return await engine.improve(text, validators)


def improve_sync(
    text: str,
    *,
    critics: Optional[List[str]] = None,
    max_iterations: int = 3,
    validators: Optional[List[Validator]] = None,
    config: Optional[Config] = None,
    storage: Optional[StorageBackend] = None,
) -> SifakaResult:
    """Synchronous wrapper for the improve() function.

    Provides a blocking interface for environments that don't support
    async/await. Uses asyncio.run() internally to execute the async
    improve() function.

    Args:
        text: The text to improve
        critics: List of critic names to apply (default: ['reflexion'])
        max_iterations: Maximum improvement iterations (default: 3)
        validators: Optional quality validators
        config: Configuration object for advanced settings
        storage: Optional storage backend

    Returns:
        SifakaResult with improved text and full audit trail

    Raises:
        Same exceptions as improve()

    Example:
        >>> # Simple synchronous usage
        >>> result = improve_sync("Explain machine learning")
        >>> print(f"Improved: {result.final_text}")
        
        >>> # With configuration
        >>> config = Config(model="claude-3-sonnet", temperature=0.9)
        >>> result = improve_sync(
        ...     "Write a creative story intro",
        ...     critics=["reflexion", "meta_rewarding"],
        ...     config=config
        ... )

    Note:
        This function creates a new event loop for each call. For better
        performance with multiple calls, use the async improve() function
        with asyncio.gather() or similar patterns.
    """
    return asyncio.run(
        improve(
            text,
            critics=critics,
            max_iterations=max_iterations,
            validators=validators,
            config=config,
            storage=storage,
        )
    )


# Advanced API for power users
async def improve_advanced(
    text: str,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    model: str = DEFAULT_MODEL,
    critics: Optional[List[str]] = None,
    validators: Optional[List[Validator]] = None,
    temperature: float = 0.7,
    timeout_seconds: int = 300,
    storage: Optional[StorageBackend] = None,
    force_improvements: bool = False,
    show_improvement_prompt: bool = False,
    critic_model: Optional[str] = None,
    critic_temperature: Optional[float] = None,
    retry_config: Optional[Union[Dict[str, Any], "RetryConfig"]] = None,
) -> SifakaResult:
    """Advanced API with granular control over all parameters.

    This function exposes all configuration options as individual parameters
    for backwards compatibility and cases where creating a Config object
    is inconvenient. For new code, prefer using improve() with a Config object.

    Args:
        text: The text to improve
        max_iterations: Maximum improvement iterations (1-10)
        model: LLM model to use for text generation. Options include
            'gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', etc.
        critics: List of critic names to apply
        validators: Optional quality validators
        temperature: Generation temperature (0.0-2.0). Higher values
            produce more creative output.
        timeout_seconds: Maximum time in seconds before timeout
        storage: Optional storage backend for persistence
        force_improvements: If True, always generate improvements even
            when critics are satisfied
        show_improvement_prompt: If True, include the improvement prompt
            in the result metadata for debugging
        critic_model: Override model specifically for critics. Useful
            for using a cheaper model for critique.
        critic_temperature: Override temperature for critics
        retry_config: Retry configuration for handling transient failures.
            Can be a dict or RetryConfig object.

    Returns:
        SifakaResult with improved text and full audit trail

    Raises:
        ValueError: If parameters are invalid
        TimeoutError: If timeout_seconds is exceeded
        ModelProviderError: If LLM API fails

    Example:
        >>> # Fine-tuned control over all aspects
        >>> result = await improve_advanced(
        ...     "Technical documentation draft",
        ...     max_iterations=5,
        ...     model="gpt-4",
        ...     critics=["reflexion", "self_refine"],
        ...     temperature=0.3,  # Lower for technical content
        ...     timeout_seconds=600,  # 10 minutes for long docs
        ...     critic_model="gpt-3.5-turbo",  # Cheaper model for critics
        ...     critic_temperature=0.7,
        ...     force_improvements=True  # Always try to improve
        ... )

    Warning:
        This function exists primarily for backwards compatibility.
        The improve() function with a Config object provides the same
        functionality with a cleaner interface.
    """
    # RetryConfig removed - handled in Config

    # Create config
    config = Config(
        model=model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        critic_model=critic_model,
        critic_temperature=critic_temperature,
        force_improvements=force_improvements,
        show_improvement_prompt=show_improvement_prompt,
        # retry config embedded in Config
    )

    return await improve(
        text,
        critics=critics,
        max_iterations=max_iterations,
        validators=validators,
        config=config,
        storage=storage,
    )
