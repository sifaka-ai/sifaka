"""
Text processing utilities for Sifaka.

This module provides common text processing functions used throughout the Sifaka framework,
including standardized handling for empty text validation.

## Empty Text Handling

The module provides standardized empty text handling:

1. **is_empty_text**: Check if text is empty or contains only whitespace
2. **handle_empty_text**: Standardized handling for empty text in all components
3. **handle_empty_text_for_classifier**: Standardized handling for empty text in classifiers

## Standardized Usage Guidelines

For consistent behavior across the codebase, follow these guidelines:

- For rules: use `handle_empty_text(text, passed=False)`
- For adapters: use `handle_empty_text(text, passed=True)`
- For core components: use `handle_empty_text(text, passed=False)`
- For chain components: use `handle_empty_text(text, passed=False)`
- For critics components: use `handle_empty_text(text, passed=False)`
- For retrieval components: use `handle_empty_text(text, passed=False)`
- For classifiers: use `handle_empty_text_for_classifier(text)`

## Usage Examples

```python
from sifaka.utils.text import (
    is_empty_text, handle_empty_text, handle_empty_text_for_classifier
)

# Check if text is empty
if is_empty_text(input_text):
    print("Input text is empty")

# Handle empty text in a rule or adapter
def validate(text: str) -> RuleResult:
    # Handle empty text first
    empty_result = handle_empty_text(text, passed=True)
    if empty_result:
        return empty_result

    # Proceed with normal validation
    # ...

# Handle empty text in a classifier
def classify(text: str) -> ClassificationResult[str]:
    # Handle empty text first
    empty_result = handle_empty_text_for_classifier(text)
    if empty_result:
        return empty_result

    # Proceed with normal classification
    # ...
```
"""

from typing import Dict, Optional, Any

from sifaka.utils.result_types import BaseResult, ClassificationResult


def is_empty_text(text: str) -> bool:
    """
    Check if text is empty or contains only whitespace.

    Args:
        text: The text to check

    Returns:
        True if the text is empty or contains only whitespace, False otherwise

    Examples:
        ```python
        from sifaka.utils.text import is_empty_text

        # Check empty string
        is_empty_text("")  # Returns True

        # Check whitespace-only string
        is_empty_text("   \t\n")  # Returns True

        # Check non-empty string
        is_empty_text("Hello, world!")  # Returns False

        # Use in a conditional
        text = get_user_input()
        if is_empty_text(text):
            print("Please enter some text")
        else:
            process_text(text)
        ```
    """
    return not text or not text.strip()


def handle_empty_text(
    text: str,
    passed: bool = True,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    component_type: str = "component",
) -> Optional[BaseResult]:
    """
    Standardized handling for empty text validation.

    This function provides consistent handling of empty text across different
    components in the Sifaka framework. It should be used by all validators,
    rules, adapters, chain components, critics, and retrieval components to ensure
    consistent behavior.

    Usage guidelines:
    - For rules: use with passed=False
    - For adapters: use with passed=True
    - For core components: use with passed=False
    - For chain components: use with passed=False
    - For critics components: use with passed=False
    - For retrieval components: use with passed=False

    Args:
        text: The text to check
        passed: Whether empty text should pass validation (default: True)
        message: Custom message for the result (default: based on passed value)
        metadata: Additional metadata to include in the result
        component_type: Type of component calling this function (for logging)

    Returns:
        BaseResult if text is empty, None otherwise

    Examples:
        ```python
        # In a validator's validate method
        def validate(self, text: str, **kwargs) -> BaseResult:
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
    final_metadata = {"reason": "empty_input", "component_type": component_type}
    if metadata:
        final_metadata.update(metadata)

    # Add input length to metadata
    if "input_length" not in final_metadata:
        final_metadata["input_length"] = len(text)

    return BaseResult(
        passed=passed,
        message=message,
        metadata=final_metadata,
    )


def handle_empty_text_for_classifier(
    text: str,
    label: Any = "unknown",
    confidence: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[ClassificationResult]:
    """
    Standardized handling for empty text in classifiers.

    This function provides consistent handling of empty text for classifiers
    in the Sifaka framework. All classifier components should use this function
    to handle empty text validation consistently.

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
        def _classify_impl_uncached(self, text: str) -> ClassificationResult:
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

    return ClassificationResult(
        passed=False,
        message="Empty text",
        label=label,
        confidence=confidence,
        metadata=final_metadata,
    )
