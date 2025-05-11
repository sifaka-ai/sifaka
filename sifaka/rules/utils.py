"""
Utility functions for rules in Sifaka.

This module provides utility functions for rules in the Sifaka framework,
including error handling and validation helpers. It uses the standardized
result creation utilities from sifaka.utils.results.

Usage Example:
    ```python
    from sifaka.rules.utils import try_validate
    from sifaka.utils.results import create_rule_result

    # Create a rule result
    result = create_rule_result(
        passed=True,
        message="Text length is within acceptable range",
        component_name="length_rule",
        metadata={"length": 100, "min_length": 10, "max_length": 1000}
    )

    # Use try_validate to handle errors
    def validate_text(text):
        if len(text) < 10:
            return create_rule_result(
                passed=False,
                message="Text is too short",
                component_name="length_rule"
            )
        return create_rule_result(
            passed=True,
            message="Text length is valid",
            component_name="length_rule"
        )

    result = try_validate(
        lambda: validate_text(input_text),
        rule_name="length_rule"
    )
    ```
"""

import time
from typing import Callable, TypeVar, Dict, Optional, Any, Literal

from ..core.results import RuleResult, create_error_result
from sifaka.utils.errors import try_operation
from sifaka.utils.logging import get_logger

# Get logger
logger = get_logger(__name__)

# Type variable for return type
T = TypeVar("T")


# The create_rule_result and create_error_result functions have been moved to sifaka.utils.results
# Import them from there instead:
# from sifaka.utils.results import create_rule_result, create_error_result


def is_empty_text(text: str) -> bool:
    """
    Check if text is empty or contains only whitespace.

    Args:
        text: The text to check

    Returns:
        True if the text is empty or contains only whitespace, False otherwise

    Examples:
        ```python
        from sifaka.rules.utils import is_empty_text

        # Check empty string
        is_empty_text("")  # Returns True

        # Check whitespace-only string
        is_empty_text("   \t\n")  # Returns True

        # Check non-empty string
        is_empty_text("Hello, world!")  # Returns False
        ```
    """
    return not text or not text.strip()


def handle_empty_text(
    text: str,
    passed: bool = True,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    component_type: Literal[
        "rule", "adapter", "classifier", "chain", "critic", "retrieval", "component"
    ] = "rule",
) -> Optional[RuleResult]:
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
        component_type: The type of component calling this function

    Returns:
        RuleResult if text is empty, None otherwise

    Examples:
        ```python
        # In a validator's validate method
        def validate(self, text: str, **kwargs) -> RuleResult:
            # Handle empty text first
            from sifaka.rules.utils import handle_empty_text
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


def try_validate(
    validation_func: Callable[[], RuleResult],
    rule_name: str,
    log_level: str = "error",
) -> RuleResult:
    """
    Execute a validation function with standardized error handling.

    This function executes a validation function and handles any errors
    that occur, providing standardized error handling and logging.

    Args:
        validation_func: The validation function to execute
        rule_name: Name of the rule executing the validation
        log_level: Log level to use for errors

    Returns:
        Result of the validation or error result if it fails

    Examples:
        ```python
        from sifaka.rules.utils import try_validate
        from sifaka.utils.results import create_rule_result

        def validate_text(text: str) -> RuleResult:
            # Validation logic
            if len(text) < 10:
                return create_rule_result(
                    passed=False,
                    message="Text is too short",
                    component_name="length_rule",
                    issues=["Text is too short"],
                    suggestions=["Add more content to meet minimum length"]
                )
            return create_rule_result(
                passed=True,
                message="Text length is valid",
                component_name="length_rule"
            )

        # Use try_validate to handle errors
        result = try_validate(
            lambda: validate_text(input_text),
            rule_name="length_rule"
        )
        ```
    """
    start_time = time.time()

    try:
        # Define error handler
        def error_handler(e: Exception) -> RuleResult:
            return create_error_result(
                message=f"Error during validation: {str(e)}",
                component_name=rule_name,
                error_type=type(e).__name__,
                severity=log_level,
            )

        # Use try_operation with the error handler
        result = try_operation(
            operation=validation_func,
            component_name=f"Rule:{rule_name}",
            error_handler=error_handler,
        )

        # Add processing time if not already present
        if hasattr(result, "metadata") and result.metadata.get("processing_time_ms") is None:
            elapsed_ms = (time.time() - start_time) * 1000
            result = result.with_metadata(processing_time_ms=elapsed_ms)

        return result

    except Exception as e:
        # This should only happen if try_operation itself fails
        logger.error(f"Unexpected error in try_validate: {e}")
        return create_error_result(
            message=f"Unexpected error in try_validate: {str(e)}",
            component_name=rule_name,
            error_type=type(e).__name__,
            severity=log_level,
        )


__all__ = [
    "try_validate",
    "is_empty_text",
    "handle_empty_text",
]
