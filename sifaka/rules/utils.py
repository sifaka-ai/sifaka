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
from typing import Callable, TypeVar

from .result import RuleResult
from sifaka.utils.errors import try_operation
from sifaka.utils.logging import get_logger
from sifaka.utils.results import create_error_result

# Get logger
logger = get_logger(__name__)

# Type variable for return type
T = TypeVar("T")


# The create_rule_result and create_error_result functions have been moved to sifaka.utils.results
# Import them from there instead:
# from sifaka.utils.results import create_rule_result, create_error_result


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
]
