"""
Text processing utilities for Sifaka.

This module provides common text processing functions used throughout the Sifaka framework,
including standardized handling for empty text validation.
"""

from typing import Dict, Optional, Any, Literal, Union, TypeVar, Generic, overload

from sifaka.rules.base import RuleResult
from sifaka.classifiers.base import ClassificationResult

R = TypeVar("R")


def is_empty_text(text: str) -> bool:
    """
    Check if text is empty or contains only whitespace.

    Args:
        text: The text to check

    Returns:
        True if the text is empty or contains only whitespace, False otherwise
    """
    return not text or not text.strip()


def handle_empty_text(
    text: str,
    passed: bool = True,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    component_type: Literal["rule", "adapter", "classifier"] = "rule",
) -> Optional[RuleResult]:
    """
    Standardized handling for empty text validation.

    This function provides consistent handling of empty text across different
    components in the Sifaka framework. It should be used by all validators,
    rules, and adapters to ensure consistent behavior.

    Args:
        text: The text to check
        passed: Whether empty text should pass validation (default: True)
        message: Custom message for the result (default: based on passed value)
        metadata: Additional metadata to include in the result
        component_type: The type of component calling this function

    Returns:
        RuleResult if text is empty, None otherwise

    Examples:
        ```python
        # In a validator's validate method
        def validate(self, text: str, **kwargs) -> RuleResult:
            # Handle empty text first
            from sifaka.utils.text import handle_empty_text
            empty_result = handle_empty_text(text)
            if empty_result:
                return empty_result

            # Proceed with normal validation
            # ...
        ```
    """
    if not is_empty_text(text):
        return None

    # Set default message based on passed value
    if message is None:
        message = "Empty text validation skipped" if passed else "Empty text provided"

    # Set default metadata
    final_metadata = {"reason": "empty_input"}
    if metadata:
        final_metadata.update(metadata)

    # Add input length to metadata
    if "input_length" not in final_metadata:
        final_metadata["input_length"] = len(text)

    return RuleResult(
        passed=passed,
        message=message,
        metadata=final_metadata,
    )


def handle_empty_text_for_classifier(
    text: str,
    label: str = "unknown",
    confidence: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[ClassificationResult[R]]:
    """
    Standardized handling for empty text in classifiers.

    This function provides consistent handling of empty text for classifiers
    in the Sifaka framework.

    Args:
        text: The text to check
        label: The label to return for empty text (default: "unknown")
        confidence: The confidence score for empty text (default: 0.0)
        metadata: Additional metadata to include in the result

    Returns:
        ClassificationResult if text is empty, None otherwise

    Examples:
        ```python
        # In a classifier's classify method
        def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
            # Handle empty text first
            from sifaka.utils.text import handle_empty_text_for_classifier
            empty_result = handle_empty_text_for_classifier(text)
            if empty_result:
                return empty_result

            # Proceed with normal classification
            # ...
        ```
    """
    if not is_empty_text(text):
        return None

    # Set default metadata
    final_metadata = {"reason": "empty_input"}
    if metadata:
        final_metadata.update(metadata)

    # Add input length to metadata
    if "input_length" not in final_metadata:
        final_metadata["input_length"] = len(text)

    return ClassificationResult[R](
        label=label,
        confidence=confidence,
        metadata=final_metadata,
    )
