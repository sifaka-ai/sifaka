"""Sifaka - PydanticAI-native AI validation, improvement, and evaluation framework.

This is a complete rewrite of Sifaka built on PydanticAI's graph capabilities.
The new architecture provides:

- Graph-based workflow orchestration using pydantic_graph
- Pure async implementation throughout
- State persistence for resumable workflows
- Parallel execution of validators and critics
- Rich observability and analytics
- Type-safe operations with Pydantic models

Key Components:
- SifakaEngine: Main orchestration engine
- SifakaThought: Core state container with full audit trail
- Graph Nodes: Generate, Validate, and Critique operations
- Critics: Research-based improvement agents (Reflexion, Constitutional, Self-Refine)
- Validators: Content validation (length, coherence, factual accuracy)
- Storage: Pluggable storage backends (memory, file, Redis)

**PRIMARY API - Configuration Presets (Recommended):**
    ```python
    import sifaka

    # Ready-to-use presets for common scenarios (covers 90% of use cases)
    result = await sifaka.academic_writing("Explain quantum computing")
    result = await sifaka.creative_writing("Write a short story")
    result = await sifaka.technical_docs("Document the API")
    result = await sifaka.business_writing("Write a project proposal")
    result = await sifaka.quick_draft("Brainstorm ideas")
    result = await sifaka.high_quality("Write a research summary")

    print(result.final_text)
    ```

**SECONDARY API - Simple Customization (When presets aren't enough):**
    ```python
    import sifaka

    # Simple customization with sensible defaults
    result = await sifaka.improve(
        "Write about renewable energy",
        max_rounds=5,
        model="openai:gpt-4",
        min_length=200
    )
    ```

**ADVANCED API - Full Control (For complex use cases):**
    ```python
    from sifaka.advanced import SifakaEngine, SifakaConfig

    # Full configuration control for advanced users
    config = SifakaConfig(
        model="openai:gpt-4",
        max_iterations=5,
        critics=["reflexion", "constitutional"],
        validators=[...],
    )
    engine = SifakaEngine(config=config)
    result = await engine.think("Explain renewable energy")
    ```

For more advanced usage, see the documentation and examples.
"""

from typing import Optional, List

# Core components (only what's needed for the main API)
from sifaka.core.thought import SifakaThought  # Return type, users need this
from sifaka.utils import (
    SifakaError,
    ValidationError,
    CritiqueError,
    GraphExecutionError,
    ConfigurationError,
)
from sifaka import presets

__version__ = "0.5.0-alpha"

# ============================================================================
# PRIMARY API - Configuration Presets (Recommended for 90% of use cases)
# ============================================================================

# Import preset functions to module level for direct access
from sifaka.presets import (
    academic_writing,
    creative_writing,
    technical_docs,
    business_writing,
    quick_draft,
    high_quality,
    # Aliases for convenience
    academic,
    creative,
    technical,
    business,
    draft,
    premium,
)


async def improve(
    prompt: str,
    *,
    max_rounds: Optional[int] = None,
    model: Optional[str] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    required_sentiment: Optional[str] = None,
    critics: Optional[List[str]] = None,
    enable_logging: bool = False,
    log_level: str = "INFO",
    log_content: bool = False,
    enable_timing: bool = False,
    enable_caching: bool = False,
    cache_size: int = 1000,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> SifakaThought:
    """Simple one-liner API for improving text with Sifaka.

    This is the main entry point for common use cases. It provides a simple
    interface that handles 80% of user needs with minimal configuration.

    Args:
        prompt: The text prompt to improve
        max_rounds: Maximum number of improvement iterations (default: 3)
        model: Model to use for generation (default: "openai:gpt-4")
        min_length: Minimum length validation (optional)
        max_length: Maximum length validation (optional)
        required_sentiment: Required sentiment validation (optional)
        critics: List of critics to enable (default: ["reflexion"])
        enable_logging: Enable workflow logging (default: False)
        log_level: Logging level when enabled (default: "INFO")
        log_content: Include content in logs (default: False)
        enable_timing: Enable performance timing (default: False)
        enable_caching: Enable result caching (default: False)
        cache_size: Maximum cache size when enabled (default: 1000)
        user_id: User ID for tracking (optional)
        session_id: Session ID for tracking (optional)

    Returns:
        SifakaThought with the improved text and full audit trail

    Raises:
        ConfigurationError: If parameters are invalid
        CritiqueError: If model/API issues occur
        ValidationError: If validation fails

    Example:
        ```python
        import sifaka

        # Simple usage
        result = await sifaka.improve("Write about AI", max_rounds=3)
        print(result.final_text)

        # With configuration
        result = await sifaka.improve(
            "Write about renewable energy",
            max_rounds=5,
            model="openai:gpt-4",
            min_length=200,
            critics=["reflexion", "constitutional"]
        )
        ```
    """
    from sifaka.utils.errors import ConfigurationError

    # Validate inputs with helpful error messages
    if not prompt or not prompt.strip():
        raise ConfigurationError(
            "Prompt cannot be empty",
            config_key="prompt",
            config_value=prompt,
            suggestions=[
                "Provide a non-empty text prompt",
                "Example: 'Write about artificial intelligence'",
                "The prompt should describe what you want to generate",
            ],
        )

    # Set defaults
    if max_rounds is None:
        max_rounds = 3
    if model is None:
        model = "openai:gpt-4"
    if critics is None:
        critics = ["reflexion"]

    # Validate max_rounds
    if max_rounds is not None and (
        not isinstance(max_rounds, int) or max_rounds < 1 or max_rounds > 20
    ):
        raise ConfigurationError(
            f"max_rounds must be an integer between 1 and 20, got: {max_rounds}",
            config_key="max_rounds",
            config_value=max_rounds,
            suggestions=[
                "Use a positive integer between 1 and 20",
                "Example: max_rounds=3",
                "Higher values allow more improvement but take longer",
            ],
        )

    # Validate length parameters
    if min_length is not None and (not isinstance(min_length, int) or min_length < 0):
        raise ConfigurationError(
            f"min_length must be a non-negative integer, got: {min_length}",
            config_key="min_length",
            config_value=min_length,
            suggestions=[
                "Use a non-negative integer",
                "Example: min_length=50",
                "Set to None to disable minimum length validation",
            ],
        )

    if max_length is not None and (not isinstance(max_length, int) or max_length < 0):
        raise ConfigurationError(
            f"max_length must be a non-negative integer, got: {max_length}",
            config_key="max_length",
            config_value=max_length,
            suggestions=[
                "Use a non-negative integer",
                "Example: max_length=500",
                "Set to None to disable maximum length validation",
            ],
        )

    if min_length is not None and max_length is not None and min_length > max_length:
        raise ConfigurationError(
            f"min_length ({min_length}) cannot be greater than max_length ({max_length})",
            config_key="length_validation",
            config_value=f"min={min_length}, max={max_length}",
            suggestions=[
                "Ensure min_length <= max_length",
                f"Try: min_length={min(min_length, max_length)}, max_length={max(min_length, max_length)}",
                "Or set one of them to None to disable that constraint",
            ],
        )

    try:
        # Import advanced components locally to keep main API clean
        from sifaka.advanced import SifakaEngine, SifakaConfig

        # Create configuration
        config = SifakaConfig(
            model=model,
            max_iterations=max_rounds,
            min_length=min_length,
            max_length=max_length,
            required_sentiment=required_sentiment,
            critics=critics,
            enable_logging=enable_logging,
            log_level=log_level,
            log_content=log_content,
            enable_timing=enable_timing,
            enable_caching=enable_caching,
            cache_size=cache_size,
        )

        # Create and run engine
        engine = SifakaEngine(config=config)
        return await engine.think(
            prompt, max_iterations=max_rounds, user_id=user_id, session_id=session_id
        )

    except Exception as e:
        # Re-raise our custom errors as-is
        if isinstance(e, (ConfigurationError, ValidationError, CritiqueError, GraphExecutionError)):
            raise

        # Wrap other exceptions with helpful context
        error_str = str(e).lower()
        if "api" in error_str and "key" in error_str:
            # API key error
            provider = "OpenAI" if "openai" in model else "Model Provider"
            key_name = f"{provider.upper()}_API_KEY"
            raise ConfigurationError(
                f"API key error: {str(e)}",
                config_key="api_key",
                suggestions=[
                    f"Set your {provider} API key: export {key_name}='your-key-here'",
                    f"Get an API key from {provider.lower()}.com",
                    "Verify your API key is valid and has necessary permissions",
                    "Check that your API key environment variable is set correctly",
                ],
            ) from e
        else:
            # Generic error with helpful context
            raise ConfigurationError(
                f"Failed to improve text: {str(e)}",
                suggestions=[
                    "Check your internet connection",
                    "Verify your API keys are set correctly",
                    "Try using a different model",
                    "Simplify your configuration parameters",
                ],
            ) from e


__all__ = [
    # ============================================================================
    # PRIMARY API - Configuration Presets (Recommended)
    # ============================================================================
    "academic_writing",
    "creative_writing",
    "technical_docs",
    "business_writing",
    "quick_draft",
    "high_quality",
    # Aliases
    "academic",
    "creative",
    "technical",
    "business",
    "draft",
    "premium",
    # ============================================================================
    # SECONDARY API - Simple Customization
    # ============================================================================
    "improve",
    # ============================================================================
    # ADVANCED API - Full Control (moved to sifaka.advanced)
    # ============================================================================
    # Note: Advanced components are available via sifaka.advanced import
    # This keeps the main API clean and focused
    # ============================================================================
    # LEGACY/COMPATIBILITY (backwards compatibility only)
    # ============================================================================
    "presets",  # For backwards compatibility: sifaka.presets.academic_writing()
    "SifakaThought",  # Return type, users need this
    # ============================================================================
    # DEPRECATED (moved to sifaka.advanced - will be removed in future versions)
    # ============================================================================
    # "SifakaEngine",  # Use sifaka.advanced.SifakaEngine instead
    # "SifakaDependencies",  # Use sifaka.advanced.SifakaDependencies instead
    # "Sifaka",  # Use sifaka.advanced.Sifaka instead
    # "SifakaConfig",  # Use sifaka.advanced.SifakaConfig instead
    # Error types (always needed)
    "SifakaError",
    "ValidationError",
    "CritiqueError",
    "GraphExecutionError",
    "ConfigurationError",
]
