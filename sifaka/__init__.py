"""Sifaka: Simple AI text improvement through research-backed critique.

This is the main API for Sifaka - one simple function that does everything you need.
"""

from typing import List, Optional

from .core.models import SifakaResult, Config
from .core.interfaces import Validator
from .core.engine import SifakaEngine
from .core.retry import RetryConfig
from .core.exceptions import (
    SifakaError,
    ConfigurationError,
    ModelProviderError,
    CriticError,
    ValidationError,
    StorageError,
    PluginError,
    TimeoutError,
)
from .storage import StorageBackend, MemoryStorage, FileStorage
from .plugins import (
    register_storage_backend,
    get_storage_backend,
    list_storage_backends,
    create_storage_backend,
)


async def improve(
    text: str,
    *,
    max_iterations: int = 3,
    model: str = "gpt-4o-mini",
    critics: Optional[List[str]] = None,
    validators: Optional[List[Validator]] = None,
    temperature: float = 0.7,
    timeout_seconds: int = 300,
    storage: Optional[StorageBackend] = None,
    force_improvements: bool = False,
    show_improvement_prompt: bool = False,
    critic_model: Optional[str] = None,
    critic_temperature: Optional[float] = None,
    retry_config: Optional[RetryConfig] = None,
) -> SifakaResult:
    """Improve text through iterative critique and refinement.

    This is the main function for Sifaka. It takes your text and improves it
    through research-backed critique techniques with complete observability.

    Args:
        text: The text to improve
        max_iterations: Maximum number of improvement iterations (1-10)
        model: OpenAI model to use (gpt-4o-mini, gpt-4, etc.)
        critics: List of critics to use ["reflexion", "constitutional", "self_refine"]
        validators: List of validator instances to check quality
        temperature: Model temperature (0.0-2.0)
        timeout_seconds: Maximum processing time in seconds
        storage: Storage backend for persisting results (default: MemoryStorage)
        force_improvements: Always run critics and try to improve text, even if validation passes
        show_improvement_prompt: Print the prompt used for text improvements
        critic_model: Model to use for critics (default: same as model)
        critic_temperature: Temperature for critic model (default: same as temperature)
        retry_config: Retry configuration for handling transient failures

    Returns:
        SifakaResult with improved text and complete audit trail

    Example:
        ```python
        import asyncio
        from sifaka import improve

        async def main():
            result = await improve(
                "Write about renewable energy benefits",
                max_iterations=3,
                critics=["reflexion", "constitutional"]
            )
            print(f"Final: {result.final_text}")
            print(f"Iterations: {result.iteration}")

        asyncio.run(main())
        ```
    """
    # Set default critics
    if critics is None:
        critics = ["reflexion"]

    # Create configuration
    config = Config(
        model=model,
        temperature=temperature,
        max_iterations=max_iterations,
        critics=critics,
        timeout_seconds=timeout_seconds,
        force_improvements=force_improvements,
        show_improvement_prompt=show_improvement_prompt,
        critic_model=critic_model,
        critic_temperature=critic_temperature,
        retry_config=retry_config,
    )

    # Create engine and run improvement
    engine = SifakaEngine(config, storage)
    return await engine.improve(text, validators)



# Expose key classes for advanced usage
__all__ = [
    "improve",
    # Core classes
    "SifakaResult",
    "Config",
    "SifakaEngine",
    "Validator",
    "StorageBackend",
    "MemoryStorage",
    "FileStorage",
    # Retry configuration
    "RetryConfig",
    # Plugin system
    "register_storage_backend",
    "get_storage_backend",
    "list_storage_backends",
    "create_storage_backend",
    # Exceptions
    "SifakaError",
    "ConfigurationError",
    "ModelProviderError",
    "CriticError",
    "ValidationError",
    "StorageError",
    "PluginError",
    "TimeoutError",
]
