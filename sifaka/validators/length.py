"""
Length validator for Sifaka.

This module provides a validator that checks if text meets length requirements.
"""

import logging
from typing import Any, Optional

from sifaka.errors import ValidationError
from sifaka.registry import register_validator
from sifaka.results import ValidationResult as SifakaValidationResult
from sifaka.utils.error_handling import log_error, validation_context
from sifaka.validators.base import BaseValidator

# Configure logger
logger = logging.getLogger(__name__)


class LengthValidator(BaseValidator):
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
        name: Optional[str] = None,
    ):
        """Initialize the length validator.

        Args:
            min_words: Minimum number of words required.
            max_words: Maximum number of words allowed.
            min_chars: Minimum number of characters required.
            max_chars: Maximum number of characters allowed.
            name: Optional name for the validator.

        Raises:
            ValidationError: If no constraints are provided or if min > max.
        """
        # Initialize the base validator with a name
        super().__init__(name=name or "LengthValidator")

        # Store configuration
        self.min_words = min_words
        self.max_words = max_words
        self.min_chars = min_chars
        self.max_chars = max_chars

        # Log initialization
        logger.debug(
            f"Initialized {self.name} with constraints: "
            f"min_words={min_words}, max_words={max_words}, "
            f"min_chars={min_chars}, max_chars={max_chars}"
        )

        # Validate constraints with improved error handling
        with validation_context(
            validator_name=self.name,
            operation="initialization",
            message_prefix="Failed to initialize length validator",
            suggestions=["Check the constraint values"],
            metadata={
                "min_words": min_words,
                "max_words": max_words,
                "min_chars": min_chars,
                "max_chars": max_chars,
            },
        ):
            # Validate that at least one constraint is provided
            if min_words is None and max_words is None and min_chars is None and max_chars is None:
                logger.error(f"{self.name}: At least one length constraint must be provided")
                raise ValidationError(
                    message="At least one length constraint must be provided",
                    component="LengthValidator",
                    operation="initialization",
                    suggestions=[
                        "Provide at least one of: min_words, max_words, min_chars, max_chars"
                    ],
                    metadata={},
                )

            # Validate that min <= max for words
            if min_words is not None and max_words is not None and min_words > max_words:
                logger.error(
                    f"{self.name}: min_words ({min_words}) must be <= max_words ({max_words})"
                )
                raise ValidationError(
                    message=f"min_words ({min_words}) must be <= max_words ({max_words})",
                    component="LengthValidator",
                    operation="initialization",
                    suggestions=[
                        f"Ensure min_words ({min_words}) is less than or equal to max_words ({max_words})",
                        "Consider using only one of min_words or max_words if appropriate",
                    ],
                    metadata={"min_words": min_words, "max_words": max_words},
                )

            # Validate that min <= max for characters
            if min_chars is not None and max_chars is not None and min_chars > max_chars:
                logger.error(
                    f"{self.name}: min_chars ({min_chars}) must be <= max_chars ({max_chars})"
                )
                raise ValidationError(
                    message=f"min_chars ({min_chars}) must be <= max_chars ({max_chars})",
                    component="LengthValidator",
                    operation="initialization",
                    suggestions=[
                        f"Ensure min_chars ({min_chars}) is less than or equal to max_chars ({max_chars})",
                        "Consider using only one of min_chars or max_chars if appropriate",
                    ],
                    metadata={"min_chars": min_chars, "max_chars": max_chars},
                )

    def _validate(self, text: str) -> SifakaValidationResult:
        """Validate text against length requirements.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text meets the length requirements.
        """
        import time

        start_time = time.time()

        # Count words and characters
        word_count = len(text.split())
        char_count = len(text)

        logger.debug(
            f"{self.name}: Validating text with {word_count} words and {char_count} characters"
        )

        # Check constraints with improved error handling
        with validation_context(
            validator_name=self.name,
            operation="validation",
            message_prefix="Failed to validate text length",
            suggestions=["Check the text length against the constraints"],
            metadata={
                "word_count": word_count,
                "char_count": char_count,
                "min_words": self.min_words,
                "max_words": self.max_words,
                "min_chars": self.min_chars,
                "max_chars": self.max_chars,
            },
        ):
            # Check word count constraints
            if self.min_words is not None and word_count < self.min_words:
                logger.debug(
                    f"{self.name}: Text is too short: {word_count} words, minimum {self.min_words} words required"
                )

                # Calculate score based on how close to min_words
                score = max(0.0, word_count / self.min_words)

                # Create issues and suggestions
                issues = [
                    f"Text is too short: {word_count} words, minimum {self.min_words} words required"
                ]
                suggestions = [
                    f"Add at least {self.min_words - word_count} more words to meet the minimum requirement",
                    "Expand on your ideas to increase the word count",
                ]

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                return SifakaValidationResult(
                    passed=False,
                    message=f"Text is too short: {word_count} words, minimum {self.min_words} words required",
                    _details={
                        "word_count": word_count,
                        "min_words": self.min_words,
                        "constraint_violated": "min_words",
                        "validator_name": self.name,
                        "processing_time_ms": processing_time,
                    },
                    score=score,
                    issues=issues,
                    suggestions=suggestions,
                )

            if self.max_words is not None and word_count > self.max_words:
                logger.debug(
                    f"{self.name}: Text is too long: {word_count} words, maximum {self.max_words} words allowed"
                )

                # Calculate score based on how close to max_words
                score = max(0.0, self.max_words / word_count)

                # Create issues and suggestions
                issues = [
                    f"Text is too long: {word_count} words, maximum {self.max_words} words allowed"
                ]
                suggestions = [
                    f"Remove at least {word_count - self.max_words} words to meet the maximum limit",
                    "Be more concise to reduce the word count",
                ]

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                return SifakaValidationResult(
                    passed=False,
                    message=f"Text is too long: {word_count} words, maximum {self.max_words} words allowed",
                    _details={
                        "word_count": word_count,
                        "max_words": self.max_words,
                        "constraint_violated": "max_words",
                        "validator_name": self.name,
                        "processing_time_ms": processing_time,
                    },
                    score=score,
                    issues=issues,
                    suggestions=suggestions,
                )

            # Check character count constraints
            if self.min_chars is not None and char_count < self.min_chars:
                logger.debug(
                    f"{self.name}: Text is too short: {char_count} characters, minimum {self.min_chars} characters required"
                )

                # Calculate score based on how close to min_chars
                score = max(0.0, char_count / self.min_chars)

                # Create issues and suggestions
                issues = [
                    f"Text is too short: {char_count} characters, minimum {self.min_chars} characters required"
                ]
                suggestions = [
                    f"Add at least {self.min_chars - char_count} more characters to meet the minimum requirement",
                    "Provide more details to increase the character count",
                ]

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                return SifakaValidationResult(
                    passed=False,
                    message=f"Text is too short: {char_count} characters, minimum {self.min_chars} characters required",
                    _details={
                        "char_count": char_count,
                        "min_chars": self.min_chars,
                        "constraint_violated": "min_chars",
                        "validator_name": self.name,
                        "processing_time_ms": processing_time,
                    },
                    score=score,
                    issues=issues,
                    suggestions=suggestions,
                )

            if self.max_chars is not None and char_count > self.max_chars:
                logger.debug(
                    f"{self.name}: Text is too long: {char_count} characters, maximum {self.max_chars} characters allowed"
                )

                # Calculate score based on how close to max_chars
                score = max(0.0, self.max_chars / char_count)

                # Create issues and suggestions
                issues = [
                    f"Text is too long: {char_count} characters, maximum {self.max_chars} characters allowed"
                ]
                suggestions = [
                    f"Remove at least {char_count - self.max_chars} characters to meet the maximum limit",
                    "Be more concise to reduce the character count",
                ]

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                return SifakaValidationResult(
                    passed=False,
                    message=f"Text is too long: {char_count} characters, maximum {self.max_chars} characters allowed",
                    _details={
                        "char_count": char_count,
                        "max_chars": self.max_chars,
                        "constraint_violated": "max_chars",
                        "validator_name": self.name,
                        "processing_time_ms": processing_time,
                    },
                    score=score,
                    issues=issues,
                    suggestions=suggestions,
                )

            # All constraints satisfied
            logger.debug(f"{self.name}: Text meets all length requirements")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Calculate score based on how close to ideal length
            score = 1.0  # Perfect score for passing

            return SifakaValidationResult(
                passed=True,
                message="Text meets length requirements",
                _details={
                    "word_count": word_count,
                    "char_count": char_count,
                    "constraints": {
                        "min_words": self.min_words,
                        "max_words": self.max_words,
                        "min_chars": self.min_chars,
                        "max_chars": self.max_chars,
                    },
                    "validator_name": self.name,
                    "processing_time_ms": processing_time,
                },
                score=score,
                issues=[],
                suggestions=[],
            )


@register_validator("length")
def create_length_validator(
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    name: Optional[str] = None,
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
        name: Optional name for the validator.
        **options: Additional options (ignored).

    Returns:
        A LengthValidator instance.

    Raises:
        ValidationError: If no constraints are provided or if min > max.
    """
    try:
        # Log factory function call
        logger.debug(
            f"Creating length validator with constraints: "
            f"min_words={min_words}, max_words={max_words}, "
            f"min_chars={min_chars}, max_chars={max_chars}"
        )

        # Create the validator
        validator = LengthValidator(
            min_words=min_words,
            max_words=max_words,
            min_chars=min_chars,
            max_chars=max_chars,
            name=name or options.get("name"),
        )

        # Log successful creation
        logger.debug(f"Successfully created length validator: {validator.name}")

        return validator

    except Exception as e:
        # Log the error
        log_error(e, logger, component="LengthValidatorFactory", operation="create_validator")

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create length validator: {str(e)}",
                component="LengthValidatorFactory",
                operation="create_validator",
                suggestions=[
                    "Check that at least one constraint is provided",
                    "Verify that min values are less than or equal to max values",
                ],
                metadata={
                    "min_words": min_words,
                    "max_words": max_words,
                    "min_chars": min_chars,
                    "max_chars": max_chars,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
        raise


def length(
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    name: Optional[str] = None,
) -> LengthValidator:
    """Create a length validator.

    This is a convenience function for creating a LengthValidator.

    Args:
        min_words: Minimum number of words required.
        max_words: Maximum number of words allowed.
        min_chars: Minimum number of characters required.
        max_chars: Maximum number of characters allowed.
        name: Optional name for the validator.

    Returns:
        A LengthValidator instance.

    Raises:
        ValidationError: If no constraints are provided or if min > max.
    """
    try:
        # Create the validator
        return LengthValidator(
            min_words=min_words,
            max_words=max_words,
            min_chars=min_chars,
            max_chars=max_chars,
            name=name,
        )

    except Exception as e:
        # Log the error
        log_error(e, logger, component="LengthValidatorFactory", operation="length")

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create length validator: {str(e)}",
                component="LengthValidatorFactory",
                operation="length",
                suggestions=[
                    "Check that at least one constraint is provided",
                    "Verify that min values are less than or equal to max values",
                ],
                metadata={
                    "min_words": min_words,
                    "max_words": max_words,
                    "min_chars": min_chars,
                    "max_chars": max_chars,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
        raise
