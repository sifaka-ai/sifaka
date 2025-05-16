"""
Custom validator for user-defined validation functions.

This module provides a validator that uses a user-defined function for validation,
allowing for custom validation logic beyond the built-in validators.
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from ..types import ValidationResult


class CustomValidator:
    """
    Validator that uses a user-defined function for validation.

    This validator allows users to provide their own validation function,
    making it easy to implement custom validation logic without creating
    a new validator class from scratch.
    """

    def __init__(
        self,
        validation_func: Callable[[str], Union[bool, Tuple[bool, Dict[str, Any]]]],
        name: str = "Custom Validator",
        description: str = "User-defined validation",
        failure_message: str = "Custom validation failed",
        success_message: str = "Custom validation passed",
        failure_score: float = 0.0,
        success_score: float = 1.0,
        default_issues: Optional[List[str]] = None,
        default_suggestions: Optional[List[str]] = None,
    ):
        """
        Initialize the custom validator.

        Args:
            validation_func: User-defined function that takes a string and returns either:
                             1. A boolean (True if validation passes, False otherwise)
                             2. A tuple of (bool, dict) where the dict can contain extra validation details
            name: Name of the validator
            description: Description of what the validator checks
            failure_message: Message to use when validation fails
            success_message: Message to use when validation passes
            failure_score: Score to assign when validation fails (0.0 to 1.0)
            success_score: Score to assign when validation passes (0.0 to 1.0)
            default_issues: Default issues to report when validation fails
            default_suggestions: Default suggestions to provide when validation fails
        """
        self.validation_func = validation_func
        self.name = name
        self.description = description
        self.failure_message = failure_message
        self.success_message = success_message
        self.failure_score = failure_score
        self.success_score = success_score
        self.default_issues = default_issues or [failure_message]
        self.default_suggestions = default_suggestions or ["Check input and try again"]

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text using the user-defined function.

        The validation function may return either:
        1. A boolean (True if validation passes, False otherwise)
        2. A tuple of (bool, dict) where the dict can contain 'message', 'score', 'issues',
           'suggestions', and any other metadata

        Args:
            text: The text to validate

        Returns:
            A ValidationResult based on the validation function's output
        """
        if not text.strip():
            return ValidationResult(
                passed=False,
                message="Empty text fails validation",
                score=0.0,
                issues=["Text is empty"],
                suggestions=["Provide non-empty content"],
                metadata={"validator_name": self.name},
            )

        try:
            # Call the validation function
            result = self.validation_func(text)

            # Process the result
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
                # If result is a tuple of (bool, dict)
                passed, details = result
                details = details or {}

                # Extract information from details
                message = details.get(
                    "message", self.success_message if passed else self.failure_message
                )
                score = details.get("score", self.success_score if passed else self.failure_score)
                issues = details.get("issues", [] if passed else self.default_issues)
                suggestions = details.get("suggestions", [] if passed else self.default_suggestions)

                # Get any additional metadata
                metadata = {
                    k: v
                    for k, v in details.items()
                    if k not in ["message", "score", "issues", "suggestions"]
                }
                metadata["validator_name"] = self.name

            elif isinstance(result, bool):
                # If result is just a boolean
                passed = result
                message = self.success_message if passed else self.failure_message
                score = self.success_score if passed else self.failure_score
                issues = [] if passed else self.default_issues
                suggestions = [] if passed else self.default_suggestions
                metadata = {"validator_name": self.name}

            else:
                # Invalid result type
                raise ValueError(
                    f"Invalid validation function result: {result}. "
                    "Expected bool or tuple(bool, dict)"
                )

            return ValidationResult(
                passed=passed,
                message=message,
                score=score,
                issues=issues,
                suggestions=suggestions,
                metadata=metadata,
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Error in custom validation: {str(e)}",
                score=0.0,
                issues=[f"Validation function error: {str(e)}"],
                suggestions=["Check validation function implementation"],
                metadata={
                    "validator_name": self.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )


def create_regex_validator(
    pattern: str,
    mode: str = "match",
    case_sensitive: bool = False,
    name: str = "Regex Validator",
    failure_message: str = "Text does not match the required pattern",
    success_message: str = "Text matches the required pattern",
) -> CustomValidator:
    """
    Create a custom validator that uses regex for validation.

    This is a convenience function for creating a common type of custom validator
    that checks if text matches (or doesn't match) a regular expression.

    Args:
        pattern: Regular expression pattern to check
        mode: 'match' to require a match, 'not_match' to forbid a match
        case_sensitive: Whether regex matching should be case sensitive
        name: Name for the validator
        failure_message: Message to use when validation fails
        success_message: Message to use when validation passes

    Returns:
        A CustomValidator configured for regex validation
    """
    import re

    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_pattern = re.compile(pattern, flags)

    def validate_regex(text: str) -> Tuple[bool, Dict[str, Any]]:
        match = bool(compiled_pattern.search(text))
        passed = match if mode == "match" else not match

        details = {
            "pattern": pattern,
            "mode": mode,
            "case_sensitive": case_sensitive,
        }

        if passed:
            return True, {
                "message": success_message,
                "score": 1.0,
                "issues": [],
                "suggestions": [],
                **details,
            }
        else:
            message = (
                "Text doesn't match the required pattern"
                if mode == "match"
                else "Text matches forbidden pattern"
            )
            suggestion = (
                "Modify text to match the pattern"
                if mode == "match"
                else "Remove content matching the pattern"
            )

            return False, {
                "message": failure_message,
                "score": 0.0,
                "issues": [message],
                "suggestions": [suggestion],
                **details,
            }

    return CustomValidator(
        validation_func=validate_regex,
        name=name,
        description=f"Regex validation ({'requiring' if mode == 'match' else 'forbidding'} pattern: {pattern})",
        failure_message=failure_message,
        success_message=success_message,
    )


def create_threshold_validator(
    extraction_func: Callable[[str], float],
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
    name: str = "Threshold Validator",
    description: str = "Validates text against numeric thresholds",
    value_name: str = "value",
) -> CustomValidator:
    """
    Create a custom validator that checks if an extracted value falls within thresholds.

    This is a convenience function for creating a validator that extracts a numeric
    value from text and checks if it's within specified thresholds.

    Args:
        extraction_func: Function that extracts a numeric value from text
        min_threshold: Minimum acceptable value (inclusive, None for no minimum)
        max_threshold: Maximum acceptable value (inclusive, None for no maximum)
        name: Name for the validator
        description: Description of what the validator checks
        value_name: Name of the value being checked (for messages)

    Returns:
        A CustomValidator configured for threshold validation
    """
    if min_threshold is None and max_threshold is None:
        raise ValueError("At least one of min_threshold or max_threshold must be specified")

    def validate_threshold(text: str) -> Tuple[bool, Dict[str, Any]]:
        try:
            value = extraction_func(text)

            # Check thresholds
            min_passed = True if min_threshold is None else value >= min_threshold
            max_passed = True if max_threshold is None else value <= max_threshold
            passed = min_passed and max_passed

            details = {
                "value": value,
                "min_threshold": min_threshold,
                "max_threshold": max_threshold,
            }

            if passed:
                return True, {
                    "message": f"{value_name.capitalize()} ({value}) is within acceptable range",
                    "score": 1.0,
                    "issues": [],
                    "suggestions": [],
                    **details,
                }
            else:
                issues = []
                suggestions = []

                if not min_passed:
                    issues.append(
                        f"{value_name.capitalize()} ({value}) is below minimum threshold ({min_threshold})"
                    )
                    suggestions.append(f"Increase {value_name}")

                if not max_passed:
                    issues.append(
                        f"{value_name.capitalize()} ({value}) exceeds maximum threshold ({max_threshold})"
                    )
                    suggestions.append(f"Decrease {value_name}")

                return False, {
                    "message": f"{value_name.capitalize()} ({value}) is outside acceptable range",
                    "score": 0.0,
                    "issues": issues,
                    "suggestions": suggestions,
                    **details,
                }

        except Exception as e:
            return False, {
                "message": f"Error extracting {value_name}: {str(e)}",
                "score": 0.0,
                "issues": [f"Could not extract {value_name} from text: {str(e)}"],
                "suggestions": ["Check text format or extraction function"],
                "error": str(e),
            }

    return CustomValidator(
        validation_func=validate_threshold,
        name=name,
        description=description,
    )
