"""Simplified API for Sifaka."""

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
    """Improve text through iterative critique - simplified API.

    Args:
        text: The text to improve
        critics: List of critics to use (default: ["reflexion"])
        max_iterations: Maximum improvement iterations (1-10)
        validators: Optional quality validators
        config: Advanced configuration options
        storage: Optional storage backend

    Returns:
        SifakaResult with improved text and audit trail

    Example:
        result = await improve(
            "Write about AI",
            critics=["reflexion", "constitutional"],
            max_iterations=3
        )
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
    """Synchronous wrapper for improve().

    Same arguments as improve() but runs synchronously.

    Example:
        result = improve_sync(
            "Write about AI",
            critics=["reflexion"],
        )
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
    """Advanced API with all parameters exposed.

    This is the original API with all parameters for backwards compatibility
    and advanced use cases.
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
