"""
Length validator for Sifaka.

This module provides a validator that checks if text meets length requirements.
"""

from typing import Optional, Dict, Any

from sifaka.results import ValidationResult
from sifaka.errors import ValidationError
from sifaka.registry import register_validator


class LengthValidator:
    """Validator that checks if text meets length requirements.

    This validator checks if text meets minimum and maximum length requirements
    in terms of words or characters.

    Attributes:
        min_words: Minimum number of words required.
        max_words: Maximum number of words allowed.
        min_chars: Minimum number of characters required.
        max_chars: Maximum number of characters allowed.
    """

    def __init__(
        self,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
    ):
        """Initialize the length validator.

        Args:
            min_words: Minimum number of words required.
            max_words: Maximum number of words allowed.
            min_chars: Minimum number of characters required.
            max_chars: Maximum number of characters allowed.

        Raises:
            ValidationError: If no constraints are provided or if min > max.
        """
        self.min_words = min_words
        self.max_words = max_words
        self.min_chars = min_chars
        self.max_chars = max_chars

        # Validate that at least one constraint is provided
        if min_words is None and max_words is None and min_chars is None and max_chars is None:
            raise ValidationError("At least one length constraint must be provided")

        # Validate that min <= max for words
        if min_words is not None and max_words is not None and min_words > max_words:
            raise ValidationError(f"min_words ({min_words}) must be <= max_words ({max_words})")

        # Validate that min <= max for characters
        if min_chars is not None and max_chars is not None and min_chars > max_chars:
            raise ValidationError(f"min_chars ({min_chars}) must be <= max_chars ({max_chars})")

    def validate(self, text: str) -> ValidationResult:
        """Validate text against length requirements.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text meets the length requirements.
        """
        # Count words and characters
        word_count = len(text.split())
        char_count = len(text)

        # Check word count constraints
        if self.min_words is not None and word_count < self.min_words:
            return ValidationResult(
                passed=False,
                message=f"Text is too short: {word_count} words, minimum {self.min_words} words required",
                details={
                    "word_count": word_count,
                    "min_words": self.min_words,
                    "constraint_violated": "min_words",
                },
            )

        if self.max_words is not None and word_count > self.max_words:
            return ValidationResult(
                passed=False,
                message=f"Text is too long: {word_count} words, maximum {self.max_words} words allowed",
                details={
                    "word_count": word_count,
                    "max_words": self.max_words,
                    "constraint_violated": "max_words",
                },
            )

        # Check character count constraints
        if self.min_chars is not None and char_count < self.min_chars:
            return ValidationResult(
                passed=False,
                message=f"Text is too short: {char_count} characters, minimum {self.min_chars} characters required",
                details={
                    "char_count": char_count,
                    "min_chars": self.min_chars,
                    "constraint_violated": "min_chars",
                },
            )

        if self.max_chars is not None and char_count > self.max_chars:
            return ValidationResult(
                passed=False,
                message=f"Text is too long: {char_count} characters, maximum {self.max_chars} characters allowed",
                details={
                    "char_count": char_count,
                    "max_chars": self.max_chars,
                    "constraint_violated": "max_chars",
                },
            )

        # All constraints satisfied
        return ValidationResult(
            passed=True,
            message="Text meets length requirements",
            details={
                "word_count": word_count,
                "char_count": char_count,
                "constraints": {
                    "min_words": self.min_words,
                    "max_words": self.max_words,
                    "min_chars": self.min_chars,
                    "max_chars": self.max_chars,
                },
            },
        )


@register_validator("length")
def create_length_validator(
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    **options: Any,
) -> LengthValidator:
    """Create a length validator.

    This factory function creates a LengthValidator with the specified constraints.
    It is registered with the registry system for dependency injection.

    Args:
        min_words: Minimum number of words required.
        max_words: Maximum number of words allowed.
        min_chars: Minimum number of characters required.
        max_chars: Maximum number of characters allowed.
        **options: Additional options (ignored).

    Returns:
        A LengthValidator instance.

    Raises:
        ValidationError: If no constraints are provided or if min > max.
    """
    return LengthValidator(
        min_words=min_words,
        max_words=max_words,
        min_chars=min_chars,
        max_chars=max_chars,
    )


def length(
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> LengthValidator:
    """Create a length validator.

    This is a convenience function for creating a LengthValidator.

    Args:
        min_words: Minimum number of words required.
        max_words: Maximum number of words allowed.
        min_chars: Minimum number of characters required.
        max_chars: Maximum number of characters allowed.

    Returns:
        A LengthValidator instance.

    Raises:
        ValidationError: If no constraints are provided or if min > max.
    """
    return LengthValidator(
        min_words=min_words,
        max_words=max_words,
        min_chars=min_chars,
        max_chars=max_chars,
    )
