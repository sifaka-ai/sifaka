"""
Utility functions for rules in Sifaka.

This module provides utility functions for rules in the Sifaka framework,
including error handling, result creation, and validation helpers.
"""

from typing import Any, Dict, Optional, TypeVar, Callable, cast

from pydantic import BaseModel

from .result import RuleResult
from sifaka.utils.errors import RuleError, try_operation
from sifaka.utils.error_patterns import handle_rule_error, ErrorResult

# Type variable for return type
T = TypeVar("T")


def create_rule_result(
    passed: bool,
    rule_name: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    score: Optional[float] = None,
) -> RuleResult:
    """
    Create a rule result with standardized structure.

    Args:
        passed: Whether the validation passed
        rule_name: Name of the rule
        message: Human-readable message describing the result
        metadata: Additional metadata about the validation
        score: Optional score for the validation (0.0 to 1.0)

    Returns:
        Standardized rule result
    """
    return RuleResult(
        passed=passed,
        rule_name=rule_name,
        message=message,
        metadata=metadata or {},
        score=score,
    )


def create_error_result(
    error: Exception,
    rule_name: str,
    log_level: str = "error",
) -> RuleResult:
    """
    Create a rule result for an error.

    This function creates a standardized rule result for an error,
    including error type, message, and metadata.

    Args:
        error: The exception that occurred
        rule_name: Name of the rule where the error occurred
        log_level: Log level to use (default: "error")

    Returns:
        Rule result representing the error
    """
    # Handle the error
    error_result = handle_rule_error(
        error=error,
        rule_name=rule_name,
        log_level=log_level,
    )

    # Create a rule result
    return create_rule_result(
        passed=False,
        rule_name=rule_name,
        message=f"Error during validation: {error_result.error_message}",
        metadata={
            "error_type": error_result.error_type,
            "error_message": error_result.error_message,
            "component": rule_name,
            **error_result.metadata,
        },
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

        def validate_text(text: str) -> RuleResult:
            # Validation logic
            if len(text) < 10:
                return create_rule_result(
                    passed=False,
                    rule_name="length_rule",
                    message="Text is too short",
                )
            return create_rule_result(
                passed=True,
                rule_name="length_rule",
                message="Text length is valid",
            )

        # Use try_validate to handle errors
        result = try_validate(
            lambda: validate_text(input_text),
            rule_name="length_rule",
        )
        ```
    """
    # Define error handler
    def error_handler(e: Exception) -> RuleResult:
        return create_error_result(e, rule_name, log_level)

    # Use try_operation with the error handler
    return try_operation(
        operation=validation_func,
        component_name=f"Rule:{rule_name}",
        error_handler=error_handler,
    )


__all__ = [
    "create_rule_result",
    "create_error_result",
    "try_validate",
]
