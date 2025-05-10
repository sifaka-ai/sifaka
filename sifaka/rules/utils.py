"""
Utility functions for rules in Sifaka.

This module provides utility functions for rules in the Sifaka framework,
including error handling, result creation, and validation helpers.

Usage Example:
    ```python
    from sifaka.rules.utils import create_rule_result, try_validate

    # Create a rule result
    result = create_rule_result(
        passed=True,
        rule_name="length_rule",
        message="Text length is within acceptable range",
        metadata={"length": 100, "min_length": 10, "max_length": 1000},
        score=0.8
    )

    # Use try_validate to handle errors
    def validate_text(text):
        if len(text) < 10:
            return create_rule_result(
                passed=False,
                rule_name="length_rule",
                message="Text is too short"
            )
        return create_rule_result(
            passed=True,
            rule_name="length_rule",
            message="Text length is valid"
        )

    result = try_validate(
        lambda: validate_text(input_text),
        rule_name="length_rule"
    )
    ```
"""

import time
from typing import Any, Dict, List, Optional, TypeVar, Callable, cast, Union

from pydantic import BaseModel

from .result import RuleResult
from sifaka.utils.errors import RuleError, try_operation
from sifaka.utils.error_patterns import create_rule_error_result, ErrorResult
from sifaka.utils.logging import get_logger

# Get logger
logger = get_logger(__name__)

# Type variable for return type
T = TypeVar("T")


def create_rule_result(
    passed: bool,
    rule_name: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    score: Optional[float] = None,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    processing_time_ms: Optional[float] = None,
) -> RuleResult:
    """
    Create a rule result with standardized structure.

    Args:
        passed: Whether the validation passed
        rule_name: Name of the rule
        message: Human-readable message describing the result
        metadata: Additional metadata about the validation
        score: Optional score for the validation (0.0 to 1.0)
        issues: List of issues identified during validation
        suggestions: List of suggestions for fixing issues
        processing_time_ms: Time taken to perform the validation in milliseconds

    Returns:
        Standardized rule result

    Examples:
        ```python
        from sifaka.rules.utils import create_rule_result

        # Create a basic result
        result = create_rule_result(
            passed=True,
            rule_name="length_rule",
            message="Text length is within acceptable range",
            metadata={"length": 100, "min_length": 10, "max_length": 1000},
            score=0.8
        )

        # Create a result with issues and suggestions
        result = create_rule_result(
            passed=False,
            rule_name="content_rule",
            message="Content contains prohibited terms",
            metadata={"found_terms": ["bad_word1", "bad_word2"]},
            score=0.2,
            issues=["Content contains prohibited terms"],
            suggestions=["Remove prohibited terms", "Replace with appropriate alternatives"]
        )
        ```
    """
    start_time = time.time() if processing_time_ms is None else None

    try:
        # Create the result
        result = RuleResult(
            passed=passed,
            rule_name=rule_name,
            message=message,
            metadata=metadata or {},
            score=score,
            issues=issues or [],
            suggestions=suggestions or [],
            processing_time_ms=processing_time_ms,
        )

        # Add processing time if not provided
        if start_time is not None:
            elapsed_ms = (time.time() - start_time) * 1000
            result = result.with_metadata(processing_time_ms=elapsed_ms)

        return result

    except Exception as e:
        logger.error(f"Error creating rule result: {e}")
        # Fallback to minimal result
        return RuleResult(
            passed=False,
            rule_name=rule_name,
            message=f"Error creating rule result: {str(e)}",
            metadata={"error": str(e), "error_type": type(e).__name__},
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

    Examples:
        ```python
        from sifaka.rules.utils import create_error_result

        try:
            # Some validation logic
            result = validate_text(text)
            return result
        except Exception as e:
            # Handle the error
            return create_error_result(
                error=e,
                rule_name="length_rule"
            )
        ```
    """
    start_time = time.time()

    try:
        # Handle the error
        error_result = create_rule_error_result(
            error=error,
            component_name=rule_name,
            log_level=log_level,
        )

        # Create error message
        error_message = f"Error during validation: {error_result.error_message}"

        # Create suggestions based on error type
        suggestions = ["Check input format and try again"]
        if "value_error" in error_result.error_type.lower():
            suggestions.append("Ensure input values are of the correct type")
        elif "key_error" in error_result.error_type.lower():
            suggestions.append("Check that all required keys are present")
        elif "index_error" in error_result.error_type.lower():
            suggestions.append("Check array indices are within bounds")

        # Create a rule result
        return create_rule_result(
            passed=False,
            rule_name=rule_name,
            message=error_message,
            metadata={
                "error_type": error_result.error_type,
                "error_message": error_result.error_message,
                "component": rule_name,
                **error_result.metadata,
            },
            issues=[error_message],
            suggestions=suggestions,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        # If error handling itself fails, create a minimal error result
        logger.error(f"Error creating error result: {e}")
        return RuleResult(
            passed=False,
            rule_name=rule_name,
            message=f"Error during validation: {str(error)}",
            metadata={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "component": rule_name,
                "meta_error": str(e),
            },
            issues=[f"Error during validation: {str(error)}"],
            suggestions=["Check input format and try again"],
            processing_time_ms=(time.time() - start_time) * 1000,
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
        from sifaka.rules.utils import try_validate, create_rule_result

        def validate_text(text: str) -> RuleResult:
            # Validation logic
            if len(text) < 10:
                return create_rule_result(
                    passed=False,
                    rule_name="length_rule",
                    message="Text is too short",
                    issues=["Text is too short"],
                    suggestions=["Add more content to meet minimum length"]
                )
            return create_rule_result(
                passed=True,
                rule_name="length_rule",
                message="Text length is valid"
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
            return create_error_result(e, rule_name, log_level)

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
        return create_error_result(error=e, rule_name=rule_name, log_level=log_level)


__all__ = [
    "create_rule_result",
    "create_error_result",
    "try_validate",
]
