"""
Content validator for Sifaka.

This module provides a validator that checks if text contains prohibited content.
"""

import re
import logging
from typing import List, Optional, Any
from re import Pattern

from sifaka.results import ValidationResult as SifakaValidationResult
from sifaka.errors import ValidationError
from sifaka.registry import register_validator
from sifaka.validators.base import BaseValidator
from sifaka.utils.error_handling import validation_context, log_error

# Configure logger
logger = logging.getLogger(__name__)


class ContentValidator(BaseValidator):
    """Validator that checks if text contains prohibited content.

    This validator checks if text contains prohibited words, phrases, or patterns.

    Attributes:
        prohibited: List of prohibited words, phrases, or patterns.
        case_sensitive: Whether the matching should be case-sensitive.
        whole_word: Whether to match whole words only.
        regex: Whether the prohibited items are regular expressions.
    """

    def __init__(
        self,
        prohibited: List[str],
        case_sensitive: bool = False,
        whole_word: bool = False,
        regex: bool = False,
        name: Optional[str] = None,
    ):
        """Initialize the content validator.

        Args:
            prohibited: List of prohibited words, phrases, or patterns.
            case_sensitive: Whether the matching should be case-sensitive.
            whole_word: Whether to match whole words only.
            regex: Whether the prohibited items are regular expressions.
            name: Optional name for the validator.

        Raises:
            ValidationError: If the prohibited list is empty or contains invalid items.
        """
        # Initialize the base validator with a name
        super().__init__(name=name or "ContentValidator")

        # Validate prohibited list
        if not prohibited:
            logger.error("Prohibited list cannot be empty")
            raise ValidationError(
                message="Prohibited list cannot be empty",
                component="ContentValidator",
                operation="initialization",
                suggestions=["Provide at least one prohibited word, phrase, or pattern"],
                metadata={},
            )

        # Store configuration
        self.prohibited = prohibited
        self.case_sensitive = case_sensitive
        self.whole_word = whole_word
        self.regex = regex

        # Log initialization
        logger.debug(
            f"Initialized {self.name} with {len(prohibited)} prohibited items, "
            f"case_sensitive={case_sensitive}, whole_word={whole_word}, regex={regex}"
        )

        # Compile patterns for efficient matching
        try:
            self._patterns = self._compile_patterns()
            logger.debug(f"{self.name}: Successfully compiled {len(self._patterns)} patterns")
        except Exception as e:
            log_error(e, logger, component="ContentValidator", operation="compile_patterns")
            raise ValidationError(
                message=f"Failed to compile patterns: {str(e)}",
                component="ContentValidator",
                operation="initialization",
                suggestions=["Check that all prohibited items are valid patterns"],
                metadata={
                    "prohibited": prohibited,
                    "regex": regex,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )

    def _compile_patterns(self) -> List[Pattern[str]]:
        """Compile the prohibited items into regular expression patterns.

        Returns:
            A list of compiled regular expression patterns.

        Raises:
            ValidationError: If a prohibited item is an invalid regular expression.
        """
        patterns = []

        with validation_context(
            validator_name=self.name,
            operation="compile_patterns",
            message_prefix="Failed to compile patterns",
            suggestions=["Check that all prohibited items are valid patterns"],
            metadata={
                "prohibited_count": len(self.prohibited),
                "regex": self.regex,
                "whole_word": self.whole_word,
                "case_sensitive": self.case_sensitive,
            },
        ):
            for i, item in enumerate(self.prohibited):
                try:
                    if self.regex:
                        # Item is already a regular expression
                        pattern_str = item
                    else:
                        # Escape special characters in the item
                        pattern_str = re.escape(item)

                        # Add word boundaries if whole_word is True
                        if self.whole_word:
                            pattern_str = r"\b" + pattern_str + r"\b"

                    # Compile the pattern with the appropriate flags
                    flags = 0 if self.case_sensitive else re.IGNORECASE

                    pattern = re.compile(pattern_str, flags)
                    patterns.append(pattern)

                except re.error as e:
                    # Log the error
                    logger.error(f"{self.name}: Invalid regular expression '{item}': {str(e)}")

                    # Raise as ValidationError with more context
                    raise ValidationError(
                        message=f"Invalid regular expression '{item}': {str(e)}",
                        component="ContentValidator",
                        operation="compile_patterns",
                        suggestions=[
                            "Check the syntax of the regular expression",
                            "If not using regex mode, consider escaping special characters",
                        ],
                        metadata={
                            "item": item,
                            "item_index": i,
                            "regex_mode": self.regex,
                            "error_message": str(e),
                        },
                    )

        return patterns

    def _validate(self, text: str) -> SifakaValidationResult:
        """Validate text against prohibited content.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text contains prohibited content.
        """
        import time

        start_time = time.time()

        # Handle empty text
        if not text:
            logger.debug(f"{self.name}: Empty text provided, returning pass result")
            return SifakaValidationResult(
                passed=True,
                message="Empty text contains no prohibited content",
                details={"matches": []},
                score=1.0,
                issues=[],
                suggestions=[],
            )

        # Check for matches
        matches = []

        with validation_context(
            validator_name=self.name,
            operation="pattern_matching",
            message_prefix="Failed to match patterns",
            suggestions=["Check the text format"],
            metadata={"text_length": len(text), "pattern_count": len(self._patterns)},
        ):
            for i, pattern in enumerate(self._patterns):
                for match in pattern.finditer(text):
                    matches.append(
                        {
                            "prohibited_item": self.prohibited[i],
                            "match": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )

            logger.debug(f"{self.name}: Found {len(matches)} matches in text of length {len(text)}")

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        if matches:
            # Found prohibited content
            match_items = [m["match"] for m in matches]
            prohibited_items = [m["prohibited_item"] for m in matches]

            # Create issues and suggestions
            match_items_str = [str(item) for item in match_items]
            prohibited_items_str = [str(item) for item in prohibited_items]
            issues = [f"Text contains prohibited content: {', '.join(match_items_str)}"]
            suggestions = [
                f"Remove or rephrase the following prohibited content: {', '.join(match_items_str)}",
                f"Avoid using terms like: {', '.join(prohibited_items_str)}",
            ]

            # Calculate score based on number of matches
            score = max(0.0, 1.0 - (len(matches) / len(self.prohibited)))

            logger.debug(
                f"{self.name}: Validation failed with {len(matches)} matches in {processing_time:.2f}ms"
            )

            return SifakaValidationResult(
                passed=False,
                message=f"Text contains prohibited content: {', '.join(match_items_str)}",
                details={
                    "matches": matches,
                    "match_count": len(matches),
                    "validator_name": self.name,
                    "processing_time_ms": processing_time,
                },
                score=score,
                issues=issues,
                suggestions=suggestions,
            )

        # No prohibited content found
        logger.debug(f"{self.name}: Validation passed with no matches in {processing_time:.2f}ms")

        return SifakaValidationResult(
            passed=True,
            message="Text contains no prohibited content",
            details={
                "matches": [],
                "match_count": 0,
                "validator_name": self.name,
                "processing_time_ms": processing_time,
            },
            score=1.0,
            issues=[],
            suggestions=[],
        )


@register_validator("content")
def create_content_validator(
    prohibited: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
    name: Optional[str] = None,
    **options: Any,
) -> ContentValidator:
    """Create a content validator.

    This factory function creates a ContentValidator with the specified parameters.
    It is registered with the registry system for dependency injection.

    Args:
        prohibited: List of prohibited words, phrases, or patterns.
        case_sensitive: Whether the matching should be case-sensitive.
        whole_word: Whether to match whole words only.
        regex: Whether the prohibited items are regular expressions.
        name: Optional name for the validator.
        **options: Additional options (ignored).

    Returns:
        A ContentValidator instance.

    Raises:
        ValidationError: If the prohibited list is empty or contains invalid items.
    """
    import time

    start_time = time.time()

    try:
        # Log factory function call
        logger.debug(
            f"Creating content validator with {len(prohibited) if prohibited else 0} prohibited items, "
            f"case_sensitive={case_sensitive}, whole_word={whole_word}, regex={regex}"
        )

        # Validate parameters
        if not prohibited:
            logger.error("Prohibited list cannot be empty")
            raise ValidationError(
                message="Prohibited list cannot be empty",
                component="ContentValidatorFactory",
                operation="create_validator",
                suggestions=["Provide at least one prohibited word, phrase, or pattern"],
                metadata={
                    "case_sensitive": case_sensitive,
                    "whole_word": whole_word,
                    "regex": regex,
                },
            )

        # Create the validator
        validator = ContentValidator(
            prohibited=prohibited,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            regex=regex,
            name=name or options.get("name"),
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log successful creation
        logger.debug(
            f"Successfully created content validator: {validator.name} in {processing_time:.2f}ms"
        )

        return validator

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(e, logger, component="ContentValidatorFactory", operation="create_validator")

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create content validator: {str(e)}",
                component="ContentValidatorFactory",
                operation="create_validator",
                suggestions=[
                    "Check that the prohibited list is not empty",
                    "Verify that all prohibited items are valid patterns if using regex mode",
                ],
                metadata={
                    "prohibited_count": len(prohibited) if prohibited else 0,
                    "case_sensitive": case_sensitive,
                    "whole_word": whole_word,
                    "regex": regex,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise


def prohibited_content(
    prohibited: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
    name: Optional[str] = None,
) -> ContentValidator:
    """Create a content validator.

    This is a convenience function for creating a ContentValidator.

    Args:
        prohibited: List of prohibited words, phrases, or patterns.
        case_sensitive: Whether the matching should be case-sensitive.
        whole_word: Whether to match whole words only.
        regex: Whether the prohibited items are regular expressions.
        name: Optional name for the validator.

    Returns:
        A ContentValidator instance.

    Raises:
        ValidationError: If the prohibited list is empty or contains invalid items.
    """
    import time

    start_time = time.time()

    try:
        # Log function call
        logger.debug(
            f"Creating content validator with {len(prohibited) if prohibited else 0} prohibited items, "
            f"case_sensitive={case_sensitive}, whole_word={whole_word}, regex={regex}"
        )

        # Validate parameters
        if not prohibited:
            logger.error("Prohibited list cannot be empty")
            raise ValidationError(
                message="Prohibited list cannot be empty",
                component="ContentValidatorFunction",
                operation="prohibited_content",
                suggestions=["Provide at least one prohibited word, phrase, or pattern"],
                metadata={
                    "case_sensitive": case_sensitive,
                    "whole_word": whole_word,
                    "regex": regex,
                },
            )

        # Create the validator
        validator = ContentValidator(
            prohibited=prohibited,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            regex=regex,
            name=name,
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log successful creation
        logger.debug(
            f"Successfully created content validator: {validator.name} in {processing_time:.2f}ms"
        )

        return validator

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(e, logger, component="ContentValidatorFunction", operation="prohibited_content")

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create content validator: {str(e)}",
                component="ContentValidatorFunction",
                operation="prohibited_content",
                suggestions=[
                    "Check that the prohibited list is not empty",
                    "Verify that all prohibited items are valid patterns if using regex mode",
                ],
                metadata={
                    "prohibited_count": len(prohibited) if prohibited else 0,
                    "case_sensitive": case_sensitive,
                    "whole_word": whole_word,
                    "regex": regex,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
