"""Base validator implementations for Sifaka.

This module provides base validator implementations that can be used to validate
text against specific criteria. Validators check if text meets certain requirements
and return a validation result with information about whether the validation passed,
any issues found, and suggestions for improvement.

Validators are used in the Sifaka chain to ensure that generated text meets
specified criteria before it is returned to the user or passed to the next stage
of the chain.
"""

import re
from typing import Any, Dict, List, Optional, Pattern

from sifaka.core.interfaces import Validator
from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.error_handling import ValidationError, validation_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class LengthValidator:
    """Validator that checks if text meets length requirements.

    This validator checks if text meets minimum and maximum length requirements
    in terms of character count.

    Attributes:
        min_length: Minimum required length in characters.
        max_length: Maximum allowed length in characters.
    """

    def __init__(self, min_length: int = 0, max_length: int = 10000):
        """Initialize the validator.

        Args:
            min_length: Minimum required length in characters.
            max_length: Maximum allowed length in characters.
        """
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, thought: Thought) -> ValidationResult:
        """Validate text against length requirements.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.

        Raises:
            ValidationError: If the validation fails due to an error.
        """
        with validation_context(
            validator_name="LengthValidator",
            operation="length validation",
            message_prefix="Failed to validate text length",
        ):
            # Check if text is available
            if not thought.text:
                return ValidationResult(
                    passed=False,
                    message="No text available for validation",
                    issues=["Text is empty or None"],
                    suggestions=["Provide text to validate"],
                )

            # Get text length
            text_length = len(thought.text)
            logger.debug(f"Text length: {text_length} characters")

            # Check if text meets length requirements
            issues = []
            suggestions = []

            if text_length < self.min_length:
                issues.append(f"Text is too short ({text_length} < {self.min_length} characters)")
                suggestions.append(f"Expand the text to at least {self.min_length} characters")

            if text_length > self.max_length:
                issues.append(f"Text is too long ({text_length} > {self.max_length} characters)")
                suggestions.append(f"Shorten the text to at most {self.max_length} characters")

            # Return validation result
            passed = len(issues) == 0
            message = (
                "Text meets length requirements"
                if passed
                else "Text does not meet length requirements"
            )

            return ValidationResult(
                passed=passed,
                message=message,
                score=1.0 if passed else 0.0,
                issues=issues if issues else None,
                suggestions=suggestions if suggestions else None,
            )


class RegexValidator:
    """Validator that checks if text matches or doesn't match regex patterns.

    This validator checks if text matches required patterns and doesn't match
    forbidden patterns.

    Attributes:
        required_patterns: List of regex patterns that text must match.
        forbidden_patterns: List of regex patterns that text must not match.
    """

    def __init__(
        self,
        required_patterns: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
    ):
        """Initialize the validator.

        Args:
            required_patterns: List of regex patterns that text must match.
            forbidden_patterns: List of regex patterns that text must not match.
        """
        self.required_patterns = required_patterns or []
        self.forbidden_patterns = forbidden_patterns or []

        # Compile patterns for efficiency
        self.required_compiled: List[Pattern] = [
            re.compile(pattern) for pattern in self.required_patterns
        ]
        self.forbidden_compiled: List[Pattern] = [
            re.compile(pattern) for pattern in self.forbidden_patterns
        ]

    def validate(self, thought: Thought) -> ValidationResult:
        """Validate text against regex patterns.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.

        Raises:
            ValidationError: If the validation fails due to an error.
        """
        with validation_context(
            validator_name="RegexValidator",
            operation="regex validation",
            message_prefix="Failed to validate text against regex patterns",
        ):
            # Check if text is available
            if not thought.text:
                return ValidationResult(
                    passed=False,
                    message="No text available for validation",
                    issues=["Text is empty or None"],
                    suggestions=["Provide text to validate"],
                )

            # Check if text matches required patterns
            issues = []
            suggestions = []

            # Check required patterns
            for i, pattern in enumerate(self.required_compiled):
                if not pattern.search(thought.text):
                    pattern_str = self.required_patterns[i]
                    issues.append(f"Text does not match required pattern: {pattern_str}")
                    suggestions.append(f"Modify the text to include content matching: {pattern_str}")

            # Check forbidden patterns
            for i, pattern in enumerate(self.forbidden_compiled):
                if pattern.search(thought.text):
                    pattern_str = self.forbidden_patterns[i]
                    issues.append(f"Text matches forbidden pattern: {pattern_str}")
                    suggestions.append(f"Remove content matching: {pattern_str}")

            # Return validation result
            passed = len(issues) == 0
            message = (
                "Text matches all required patterns and no forbidden patterns"
                if passed
                else "Text does not meet pattern requirements"
            )

            return ValidationResult(
                passed=passed,
                message=message,
                score=1.0 if passed else 0.0,
                issues=issues if issues else None,
                suggestions=suggestions if suggestions else None,
            )
