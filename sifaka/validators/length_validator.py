"""
Length validator for Sifaka.

This module provides a validator that checks if text meets length requirements.
"""

import logging
import time
from typing import Optional

from sifaka.core.interfaces import Validator
from sifaka.core.thought import Thought

# Configure logger
logger = logging.getLogger(__name__)


class LengthValidator(Validator):
    """
    Validator that checks if text meets length requirements.

    This validator checks if text meets minimum and maximum length requirements
    in terms of words or characters.
    """

    def __init__(
        self,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the length validator.

        Args:
            min_words: Minimum number of words required.
            max_words: Maximum number of words allowed.
            min_chars: Minimum number of characters required.
            max_chars: Maximum number of characters allowed.
            name: Optional name for the validator.

        Raises:
            ValueError: If no constraints are provided or if min > max.
        """
        self._name = name or "LengthValidator"

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

        # Validate constraints
        if min_words is None and max_words is None and min_chars is None and max_chars is None:
            logger.error(f"{self.name}: At least one length constraint must be provided")
            raise ValueError("At least one length constraint must be provided")

        # Validate that min <= max for words
        if min_words is not None and max_words is not None and min_words > max_words:
            logger.error(f"{self.name}: min_words ({min_words}) must be <= max_words ({max_words})")
            raise ValueError(f"min_words ({min_words}) must be <= max_words ({max_words})")

        # Validate that min <= max for characters
        if min_chars is not None and max_chars is not None and min_chars > max_chars:
            logger.error(f"{self.name}: min_chars ({min_chars}) must be <= max_chars ({max_chars})")
            raise ValueError(f"min_chars ({min_chars}) must be <= max_chars ({max_chars})")

    @property
    def name(self) -> str:
        """Return the name of the validator."""
        return self._name

    def validate(self, thought: Thought) -> bool:
        """
        Validate the text in the thought.

        Args:
            thought: The thought containing the text to validate.

        Returns:
            True if the text passes validation, False otherwise.
        """
        start_time = time.time()
        text = thought.text

        try:
            # Handle empty text
            if not text or not text.strip():
                thought.add_validation_result(
                    validator_name=self.name,
                    passed=False,
                    score=0.0,
                    details={
                        "validator_name": self.name,
                        "error_type": "EmptyText",
                    },
                    message="Empty text is not valid",
                )
                return False

            # Count words and characters
            word_count = len(text.split())
            char_count = len(text)

            logger.debug(
                f"{self.name}: Validating text with {word_count} words and {char_count} characters"
            )

            # Check word count constraints
            if self.min_words is not None and word_count < self.min_words:
                logger.debug(
                    f"{self.name}: Text is too short: {word_count} words, minimum {self.min_words} words required"
                )

                # Calculate score based on how close to min_words
                score = max(0.0, word_count / self.min_words)

                # Add validation result to thought
                thought.add_validation_result(
                    validator_name=self.name,
                    passed=False,
                    score=score,
                    details={
                        "word_count": word_count,
                        "min_words": self.min_words,
                        "constraint_violated": "min_words",
                        "validator_name": self.name,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    },
                    message=f"Text is too short: {word_count} words, minimum {self.min_words} words required",
                )
                return False

            if self.max_words is not None and word_count > self.max_words:
                logger.debug(
                    f"{self.name}: Text is too long: {word_count} words, maximum {self.max_words} words allowed"
                )

                # Calculate score based on how close to max_words
                score = max(0.0, self.max_words / word_count)

                # Add validation result to thought
                thought.add_validation_result(
                    validator_name=self.name,
                    passed=False,
                    score=score,
                    details={
                        "word_count": word_count,
                        "max_words": self.max_words,
                        "constraint_violated": "max_words",
                        "validator_name": self.name,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    },
                    message=f"Text is too long: {word_count} words, maximum {self.max_words} words allowed",
                )
                return False

            # Check character count constraints
            if self.min_chars is not None and char_count < self.min_chars:
                logger.debug(
                    f"{self.name}: Text is too short: {char_count} characters, minimum {self.min_chars} characters required"
                )

                # Calculate score based on how close to min_chars
                score = max(0.0, char_count / self.min_chars)

                # Add validation result to thought
                thought.add_validation_result(
                    validator_name=self.name,
                    passed=False,
                    score=score,
                    details={
                        "char_count": char_count,
                        "min_chars": self.min_chars,
                        "constraint_violated": "min_chars",
                        "validator_name": self.name,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    },
                    message=f"Text is too short: {char_count} characters, minimum {self.min_chars} characters required",
                )
                return False

            if self.max_chars is not None and char_count > self.max_chars:
                logger.debug(
                    f"{self.name}: Text is too long: {char_count} characters, maximum {self.max_chars} characters allowed"
                )

                # Calculate score based on how close to max_chars
                score = max(0.0, self.max_chars / char_count)

                # Add validation result to thought
                thought.add_validation_result(
                    validator_name=self.name,
                    passed=False,
                    score=score,
                    details={
                        "char_count": char_count,
                        "max_chars": self.max_chars,
                        "constraint_violated": "max_chars",
                        "validator_name": self.name,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    },
                    message=f"Text is too long: {char_count} characters, maximum {self.max_chars} characters allowed",
                )
                return False

            # All constraints satisfied
            logger.debug(f"{self.name}: Text meets all length requirements")

            # Add validation result to thought
            thought.add_validation_result(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details={
                    "word_count": word_count,
                    "char_count": char_count,
                    "constraints": {
                        "min_words": self.min_words,
                        "max_words": self.max_words,
                        "min_chars": self.min_chars,
                        "max_chars": self.max_chars,
                    },
                    "validator_name": self.name,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                },
                message="Text meets length requirements",
            )
            return True

        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")

            # Add error validation result to thought
            thought.add_validation_result(
                validator_name=self.name,
                passed=False,
                score=0.0,
                details={
                    "validator_name": self.name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                },
                message=f"Validation error: {str(e)}",
            )
            return False


def create_length_validator(
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    name: Optional[str] = None,
) -> LengthValidator:
    """
    Create a length validator.

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
        ValueError: If no constraints are provided or if min > max.
    """
    return LengthValidator(
        min_words=min_words,
        max_words=max_words,
        min_chars=min_chars,
        max_chars=max_chars,
        name=name,
    )
