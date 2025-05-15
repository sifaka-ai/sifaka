"""
Utility functions for critics in Sifaka.

This module provides utility functions for critics in the Sifaka framework,
including error handling, result creation, and critique helpers.
"""

from typing import Any, Dict, List, Optional, TypeVar, Callable, cast

from pydantic import BaseModel

from sifaka.utils.errors.handling import try_operation
from sifaka.utils.errors.results import create_critic_error_result

# Type variable for return type
T = TypeVar("T")


class CriticMetadata(BaseModel):
    """
    Metadata for critic results.

    This class provides a standardized structure for critic metadata,
    including scores, feedback, issues, and suggestions.

    Attributes:
        score: Score for the critique (0.0 to 1.0)
        feedback: Human-readable feedback
        issues: List of identified issues
        suggestions: List of improvement suggestions
        metadata: Additional metadata
    """

    score: float
    feedback: str
    issues: List[str] = []
    suggestions: List[str] = []
    metadata: Dict[str, Any] = {}


def create_critic_metadata(
    score: float,
    feedback: str,
    issues: Optional[Optional[List[str]]] = None,
    suggestions: Optional[Optional[List[str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> CriticMetadata:
    """
    Create critic metadata with standardized structure.

    Args:
        score: Score for the critique (0.0 to 1.0)
        feedback: Human-readable feedback
        issues: List of identified issues
        suggestions: List of improvement suggestions
        metadata: Additional metadata

    Returns:
        Standardized critic metadata
    """
    return CriticMetadata(
        score=score,
        feedback=feedback,
        issues=issues or [],
        suggestions=suggestions or [],
        metadata=metadata or {},
    )


def create_error_metadata(
    error: Exception,
    critic_name: str,
    log_level: str = "error",
) -> CriticMetadata:
    """
    Create critic metadata for an error.

    This function creates standardized critic metadata for an error,
    including error type, message, and metadata.

    Args:
        error: The exception that occurred
        critic_name: Name of the critic where the error occurred
        log_level: Log level to use (default: "error")

    Returns:
        Critic metadata representing the error
    """
    # Handle the error
    error_result = create_critic_error_result(error, critic_name, log_level, True, None)

    # Create critic metadata
    return create_critic_metadata(
        score=0.0,
        feedback=f"Error during critique: {error_result.error_message}",
        issues=[f"Critique process failed: {error_result.error_message}"],
        metadata={
            "error_type": error_result.error_type,
            "error_message": error_result.error_message,
            "component": critic_name,
            **error_result.metadata,
        },
    )


def try_critique(
    critique_func: Callable[[], T],
    critic_name: str,
    log_level: str = "error",
    default_result: Optional[Optional[T]] = None,
) -> T:
    """
    Execute a critique function with standardized error handling.

    This function executes a critique function and handles any errors
    that occur, providing standardized error handling and logging.

    Args:
        critique_func: The critique function to execute
        critic_name: Name of the critic executing the critique
        log_level: Log level to use for errors
        default_result: Default result to return if critique fails

    Returns:
        Result of the critique or default result if it fails

    Examples:
        ```python
        from sifaka.critics.utils import try_critique, create_critic_metadata

        def critique_text(text: str) -> CriticMetadata:
            # Critique logic
            score = calculate_score(text)
            feedback = generate_feedback(text)
            return create_critic_metadata(
                score=score,
                feedback=feedback,
                issues=["Issue 1", "Issue 2"],
                suggestions=["Suggestion 1", "Suggestion 2"],
            )

        # Use try_critique to handle errors
        result = try_critique(
            lambda: critique_text(input_text),
            critic_name="my_critic",
        )
        ```
    """
    # Use try_operation with default result
    return cast(
        T,
        try_operation(
            operation=critique_func,
            component_name=f"Critic:{critic_name}",
            default_value=default_result,
            log_level=log_level,
        ),
    )


__all__ = [
    "CriticMetadata",
    "create_critic_metadata",
    "create_error_metadata",
    "try_critique",
]
