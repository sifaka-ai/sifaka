"""
Utility functions for models in Sifaka.

This module provides utility functions for models in the Sifaka framework,
including error handling, result creation, and generation helpers.
"""

from typing import Any, Dict, Optional, TypeVar, Callable

from .result import GenerationResult
from sifaka.utils.errors.handling import try_operation
from sifaka.core.results import ErrorResult


# Define create_model_error_result function
def create_model_error_result(
    error: Exception,
    component_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """
    Create a standardized error result for a model operation.

    Args:
        error: The exception that occurred
        component_name: Name of the model where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        Standardized error result
    """
    # Get error type and message
    error_type = type(error).__name__
    error_message = str(error)

    # Create metadata
    metadata = {
        "component": component_name,
        "error_type": error_type,
        **(additional_metadata or {}),
    }

    # Include traceback if requested
    if include_traceback:
        import traceback

        metadata["traceback"] = traceback.format_exc()

    # Log the error
    import logging

    logger = logging.getLogger(__name__)
    log_func = getattr(logger, log_level, logger.error)
    log_func(f"Model error in {component_name}: {error_message}")

    # Create error result
    return ErrorResult(
        error_type=error_type,
        error_message=error_message,
        passed=False,
        message=error_message,
        metadata=metadata,
        score=0.0,
    )


# Type variable for return type
T = TypeVar("T")


def create_generation_result(
    text: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> GenerationResult:
    """
    Create a generation result with standardized structure.

    Args:
        text: Generated text
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        metadata: Additional metadata

    Returns:
        Standardized generation result
    """
    return GenerationResult(
        output=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        metadata=metadata or {},
    )


def create_error_result(
    error: Exception,
    model_name: str,
    log_level: str = "error",
) -> GenerationResult:
    """
    Create a result for a model error.

    This function creates a standardized result for a model error,
    including error type, message, and metadata.

    Args:
        error: The exception that occurred
        model_name: Name of the model where the error occurred
        log_level: Log level to use (default: "error")

    Returns:
        Generation result with error information
    """
    # Handle the error
    error_result = create_model_error_result(
        error=error,
        component_name=model_name,
        log_level=log_level,
    )

    # Create a generation result with error information
    return GenerationResult(
        output="",
        prompt_tokens=0,
        completion_tokens=0,
        metadata={
            "error": True,
            "error_type": error_result.error_type,
            "error_message": error_result.error_message,
            "component": model_name,
            "model_name": model_name,
            **error_result.metadata,
        },
    )


def try_generate(
    generation_func: Callable[[], T],
    model_name: str,
    log_level: str = "error",
    default_result: Optional[T] = None,
) -> T:
    """
    Execute a generation function with standardized error handling.

    This function executes a generation function and handles any errors
    that occur, providing standardized error handling and logging.

    Args:
        generation_func: The generation function to execute
        model_name: Name of the model executing the generation
        log_level: Log level to use for errors
        default_result: Default result to return if generation fails

    Returns:
        Result of the generation or default result if it fails

    Examples:
        ```python
        from sifaka.models.utils import try_generate, create_generation_result

        def generate_text(prompt: str) -> GenerationResult:
            # Generation logic
            text = model.generate(prompt)
            return create_generation_result(
                text=text,
                prompt_tokens=len(prompt.split()) if prompt else 0,
                completion_tokens=len(text.split()) if text else 0,
            )

        # Use try_generate to handle errors
        result = try_generate(
            lambda: generate_text(prompt),
            model_name="gpt-4",
        )
        ```
    """
    # Use try_operation with default result
    return try_operation(
        operation=generation_func,
        component_name=f"Model:{model_name}",
        default_value=default_result,
        log_level=log_level,
    )


__all__ = [
    "create_generation_result",
    "create_error_result",
    "create_model_error_result",
    "try_generate",
]
