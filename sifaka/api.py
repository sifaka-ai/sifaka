"""Sifaka's main API for text improvement through iterative critique.

This module provides the primary interface for using Sifaka:
- improve(): Main async text improvement function
- improve_sync(): Synchronous wrapper for convenience

The API is designed to be simple for basic use cases while allowing
full customization through the Config object."""

import asyncio
from typing import List, Optional, Union

from .core.config import Config
from .core.constants import DEFAULT_MAX_ITERATIONS
from .core.engine import SifakaEngine
from .core.interfaces import Validator
from .core.middleware import MiddlewarePipeline
from .core.models import SifakaResult
from .core.monitoring import monitor
from .core.type_defs import MiddlewareContext
from .core.types import CriticType
from .core.validation import validate_config_params, validate_improve_params
from .storage.base import StorageBackend


async def improve(
    text: str,
    *,
    critics: Optional[Union[CriticType, List[CriticType]]] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
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
        text: The text to improve. Can be any length.
        critics: Single CriticType enum or list of CriticType enums. Available:
            CriticType.REFLEXION, CriticType.SELF_REFINE, CriticType.CONSTITUTIONAL,
            CriticType.META_REWARDING, CriticType.SELF_CONSISTENCY, CriticType.N_CRITICS,
            CriticType.SELF_RAG, CriticType.STYLE. Defaults to CriticType.REFLEXION.
        max_iterations: Maximum improvement iterations (1-10). Default: 3.
        validators: Optional quality validators to ensure text meets criteria.
        config: Configuration for models, temperature, style, etc.
            Use Config.fast(), Config.quality(), or Config.creative().
        storage: Optional storage backend for persistence.
        middleware: Optional middleware pipeline for custom behavior.

    Returns:
        SifakaResult with original text, final text, critiques, and metadata.

    Raises:
        ValueError: If invalid parameters are provided
        TimeoutError: If processing exceeds timeout
        ModelProviderError: If LLM API fails

    Examples:
        >>> # Basic usage
        >>> result = await improve("Write about AI benefits")

        >>> # With multiple critics
        >>> result = await improve(
        ...     "Explain quantum computing",
        ...     critics=[CriticType.REFLEXION, CriticType.SELF_REFINE],
        ...     max_iterations=5
        ... )

        >>> # Fast mode with specific model
        >>> result = await improve(
        ...     text,
        ...     config=Config.fast()
        ... )

        >>> # Style transformation
        >>> result = await improve(
        ...     formal_text,
        ...     critics=CriticType.STYLE,
        ...     config=Config(
        ...         style_guide="Casual blog post",
        ...         style_examples=["Hey there!", "Let me tell you..."]
        ...     )
        ... )
    """
    # Validate input parameters
    validated = validate_improve_params(text, critics, max_iterations)
    text = validated.text
    max_iterations = validated.max_iterations

    # Handle critics - validation always returns a list or None
    critics_list: List[CriticType]
    if validated.critics is not None:
        critics_list = validated.critics
    else:
        critics_list = [CriticType.REFLEXION]  # Default critic

    # Use default config if none provided
    if config is None:
        config = Config()

    # Validate config parameters with enhanced validation
    try:
        validate_config_params(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_iterations=max_iterations,
            timeout_seconds=config.llm.timeout_seconds,
        )
    except ValueError as e:
        raise ValueError(f"Configuration validation failed: {e}") from e

    # Create engine config
    engine_config = config.model_copy(deep=True)
    engine_config.engine.max_iterations = max_iterations
    engine_config.critic.critics = critics_list

    # Create and run engine
    engine = SifakaEngine(engine_config, storage)

    # Execute with monitoring
    async with monitor() as m:
        # Start monitoring for this operation
        m.start_monitoring(max_iterations=max_iterations)

        # Execute with middleware if provided
        if middleware:
            critics_context: List[Union[str, CriticType]] = [c for c in critics_list]
            context: MiddlewareContext = {
                "critics": critics_context,
                "validators": validators,
                "config": engine_config,
            }

            async def final_handler(text: str) -> SifakaResult:
                return await engine.improve(text, validators)

            result = await middleware.execute(text, final_handler, context)
        else:
            result = await engine.improve(text, validators)

        # Record final metrics
        m.update_from_result(result)
        return result


def improve_sync(
    text: str,
    *,
    critics: Optional[Union[CriticType, List[CriticType]]] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    validators: Optional[List[Validator]] = None,
    config: Optional[Config] = None,
    storage: Optional[StorageBackend] = None,
) -> SifakaResult:
    """Synchronous wrapper for improve().

    For environments that don't support async/await.

    Args:
        Same as improve() minus middleware (not supported in sync mode).

    Returns:
        SifakaResult with improved text.

    Examples:
        >>> result = improve_sync("Explain machine learning")
        >>> print(result.final_text)
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
