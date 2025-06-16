"""Configuration presets for common Sifaka use cases.

This module provides ready-to-use configurations for typical scenarios,
making it easy for users to get started without understanding all the options.

Example:
    ```python
    import sifaka.presets as presets

    # Academic writing with high quality standards
    result = await presets.academic_writing("Explain quantum computing")

    # Creative writing with style focus
    result = await presets.creative_writing("Write a short story about AI")

    # Technical documentation with clarity focus
    result = await presets.technical_docs("Document the API endpoints")
    ```
"""

from typing import Optional, List, TYPE_CHECKING
from sifaka import improve

if TYPE_CHECKING:
    from sifaka.core.thought import SifakaThought


def _validate_preset_params(
    prompt: str,
    max_rounds: int,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    model: str = "openai:gpt-4",
) -> None:
    """Validate common preset parameters.

    Args:
        prompt: The writing prompt
        max_rounds: Maximum improvement rounds
        min_length: Minimum length requirement
        max_length: Maximum length requirement
        model: Model to use

    Raises:
        ValueError: If parameters are invalid
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    if max_rounds > 10:
        raise ValueError("max_rounds should not exceed 10 for performance reasons")

    if min_length is not None and min_length < 1:
        raise ValueError("min_length must be positive")

    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be positive")

    if min_length is not None and max_length is not None and min_length > max_length:
        raise ValueError("min_length cannot be greater than max_length")

    if not model or not model.strip():
        raise ValueError("Model cannot be empty")

    # Validate model format (provider:model)
    if ":" not in model:
        raise ValueError(f"Model '{model}' should be in format 'provider:model'")

    provider, model_name = model.split(":", 1)
    if not provider or not model_name:
        raise ValueError(f"Invalid model format: '{model}'")

    # Validate known providers
    known_providers = ["openai", "anthropic", "ollama", "gemini"]
    if provider not in known_providers:
        import warnings

        warnings.warn(
            f"Unknown model provider '{provider}'. Known providers: {known_providers}", UserWarning
        )


async def academic_writing(
    prompt: str,
    *,
    min_length: int = 300,
    max_rounds: int = 5,
    model: str = "openai:gpt-4",
    **kwargs,
) -> "SifakaThought":
    """Preset for academic writing with high quality standards.

    Optimized for:
    - Formal tone and structure
    - Factual accuracy
    - Comprehensive coverage
    - Proper citations and evidence

    Args:
        prompt: The writing prompt
        min_length: Minimum length (default: 300 words)
        max_rounds: Maximum improvement rounds (default: 5)
        **kwargs: Additional arguments passed to improve()

    Returns:
        SifakaThought with improved academic text
    """
    _validate_preset_params(prompt, max_rounds, min_length=min_length, model=model)

    return await improve(
        prompt,
        max_rounds=max_rounds,
        model=model,
        min_length=min_length,
        critics=["reflexion", "constitutional", "self_refine"],
        enable_logging=True,
        **kwargs,
    )


async def creative_writing(
    prompt: str,
    *,
    max_length: int = 800,
    max_rounds: int = 4,
    model: str = "anthropic:claude-3-5-sonnet-20241022",
    **kwargs,
) -> "SifakaThought":
    """Preset for creative writing with style and engagement focus.

    Optimized for:
    - Engaging narrative
    - Creative language use
    - Emotional impact
    - Style consistency

    Args:
        prompt: The writing prompt
        max_length: Maximum length (default: 800 words)
        max_rounds: Maximum improvement rounds (default: 4)
        **kwargs: Additional arguments passed to improve()

    Returns:
        SifakaThought with improved creative text
    """
    _validate_preset_params(prompt, max_rounds, max_length=max_length, model=model)

    return await improve(
        prompt,
        max_rounds=max_rounds,
        model=model,
        max_length=max_length,
        critics=["constitutional", "self_consistency"],
        enable_timing=True,
        **kwargs,
    )


async def technical_docs(
    prompt: str,
    *,
    min_length: int = 200,
    max_rounds: int = 4,
    model: str = "openai:gpt-4",
    **kwargs,
) -> "SifakaThought":
    """Preset for technical documentation with clarity focus.

    Optimized for:
    - Clear explanations
    - Logical structure
    - Practical examples
    - User-friendly language

    Args:
        prompt: The documentation prompt
        min_length: Minimum length (default: 200 words)
        max_rounds: Maximum improvement rounds (default: 4)
        **kwargs: Additional arguments passed to improve()

    Returns:
        SifakaThought with improved technical documentation
    """
    _validate_preset_params(prompt, max_rounds, min_length=min_length, model=model)
    
    return await improve(
        prompt,
        max_rounds=max_rounds,
        model=model,
        min_length=min_length,
        critics=["reflexion", "self_refine"],
        enable_logging=True,
        enable_timing=True,
        **kwargs,
    )


async def business_writing(
    prompt: str,
    *,
    max_length: int = 500,
    max_rounds: int = 3,
    model: str = "openai:gpt-4o-mini",
    **kwargs,
) -> "SifakaThought":
    """Preset for business writing with professional tone.

    Optimized for:
    - Professional tone
    - Concise communication
    - Action-oriented language
    - Clear recommendations

    Args:
        prompt: The business writing prompt
        max_length: Maximum length (default: 500 words)
        max_rounds: Maximum improvement rounds (default: 3)
        **kwargs: Additional arguments passed to improve()

    Returns:
        SifakaThought with improved business text
    """
    _validate_preset_params(prompt, max_rounds, min_length=min_length, model=model)
    
    return await improve(
        prompt,
        max_rounds=max_rounds,
        model=model,
        max_length=max_length,
        critics=["constitutional"],
        enable_caching=True,
        **kwargs,
    )


async def quick_draft(
    prompt: str, *, max_rounds: int = 2, model: str = "openai:gpt-4o-mini", **kwargs
) -> "SifakaThought":
    """Preset for quick drafting with minimal processing.

    Optimized for:
    - Speed over perfection
    - Basic quality checks
    - Rapid iteration
    - Cost efficiency

    Args:
        prompt: The writing prompt
        max_rounds: Maximum improvement rounds (default: 2)
        **kwargs: Additional arguments passed to improve()

    Returns:
        SifakaThought with quickly improved text
    """
    _validate_preset_params(prompt, max_rounds, min_length=min_length, model=model)
    
    return await improve(
        prompt,
        max_rounds=max_rounds,
        model=model,
        critics=["reflexion"],
        enable_caching=True,
        **kwargs,
    )


async def high_quality(
    prompt: str,
    *,
    min_length: int = 400,
    max_rounds: int = 7,
    model: str = "openai:gpt-4",
    **kwargs,
) -> "SifakaThought":
    """Preset for highest quality output with comprehensive processing.

    Optimized for:
    - Maximum quality
    - Multiple perspectives
    - Thorough validation
    - Comprehensive improvement

    Args:
        prompt: The writing prompt
        min_length: Minimum length (default: 400 words)
        max_rounds: Maximum improvement rounds (default: 7)
        **kwargs: Additional arguments passed to improve()

    Returns:
        SifakaThought with highest quality improved text
    """
    _validate_preset_params(prompt, max_rounds, min_length=min_length, model=model)
    
    return await improve(
        prompt,
        max_rounds=max_rounds,
        model=model,
        min_length=min_length,
        critics=["reflexion", "constitutional", "self_refine", "self_consistency"],
        enable_logging=True,
        enable_timing=True,
        enable_caching=True,
        **kwargs,
    )


# Convenience aliases for common use cases
academic = academic_writing
creative = creative_writing
technical = technical_docs
business = business_writing
draft = quick_draft
premium = high_quality
