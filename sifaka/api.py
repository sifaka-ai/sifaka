"""Sifaka's main API for text improvement through iterative critique.

This module provides the primary interface for using Sifaka:
- improve(): Main async text improvement function
- improve_sync(): Synchronous wrapper for convenience

The API is designed to be simple for basic use cases while allowing
full customization through the Config object."""

from typing import Any, Dict, List, Optional, Union
import asyncio

from .core.models import SifakaResult
from .core.config import Config
from .core.interfaces import Validator
from .core.engine import SifakaEngine
from .core.constants import DEFAULT_CRITIC, DEFAULT_MAX_ITERATIONS
from .core.middleware import MiddlewarePipeline
from .storage.base import StorageBackend


async def improve(
    text: str,
    *,
    critics: Optional[Union[str, List[str]]] = None,
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
        critics: Single critic name or list of critics to apply. Available:
            'reflexion', 'self_refine', 'constitutional', 'meta_rewarding',
            'self_consistency', 'n_critics', 'self_rag', 'style'.
            Defaults to 'reflexion'.
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
        ...     critics=["reflexion", "self_refine"],
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
        ...     critics="style",
        ...     config=Config(
        ...         style_guide="Casual blog post",
        ...         style_examples=["Hey there!", "Let me tell you..."]
        ...     )
        ... )
    """
    # Handle single critic string
    if isinstance(critics, str):
        critics = [critics]
    elif critics is None:
        critics = [DEFAULT_CRITIC]

    # Use default config if none provided
    if config is None:
        config = Config()

    # Create engine config
    engine_config = config.model_copy(
        update={
            "max_iterations": max_iterations,
            "critics": critics,
        }
    )

    # Create and run engine
    engine = SifakaEngine(engine_config, storage)

    # Execute with middleware if provided
    if middleware:
        context: Dict[str, Any] = {
            "critics": critics,
            "validators": validators,
            "config": engine_config,
        }

        async def final_handler(text: str) -> SifakaResult:
            return await engine.improve(text, validators)

        return await middleware.execute(text, final_handler, context)

    return await engine.improve(text, validators)


def improve_sync(
    text: str,
    *,
    critics: Optional[Union[str, List[str]]] = None,
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
