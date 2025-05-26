"""Content validator for Sifaka.

This module provides a ContentValidator that checks if text contains prohibited
content, words, phrases, or patterns. It supports case-sensitive/insensitive
matching, whole word matching, and regular expression patterns.

The ContentValidator is designed to prevent generation of harmful, inappropriate,
or unwanted content by checking against a list of prohibited items.
"""

import re
import time
from typing import List, Pattern

from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.error_handling import ValidationError
from sifaka.utils.logging import get_logger
from sifaka.validators.shared import BaseValidator

# Configure logger
logger = get_logger(__name__)


class ContentValidator(BaseValidator):
    """Validator that checks if text contains prohibited content.

    This validator checks if text contains prohibited words, phrases, or patterns.
    It supports various matching modes including case sensitivity, whole word matching,
    and regular expression patterns.

    Attributes:
        prohibited: List of prohibited words, phrases, or patterns.
        case_sensitive: Whether the matching should be case-sensitive.
        whole_word: Whether to match whole words only.
        regex: Whether the prohibited items are regular expressions.
        name: The name of the validator.
    """

    def __init__(
        self,
        prohibited: List[str],
        case_sensitive: bool = False,
        whole_word: bool = False,
        regex: bool = False,
        name: str = "ContentValidator",
    ):
        """Initialize the validator.

        Args:
            prohibited: List of prohibited words, phrases, or patterns.
            case_sensitive: Whether the matching should be case-sensitive.
            whole_word: Whether to match whole words only.
            regex: Whether the prohibited items are regular expressions.
            name: The name of the validator.

        Raises:
            ValidationError: If the prohibited list is empty or invalid.
        """
        if not prohibited:
            raise ValidationError(
                message="Prohibited list cannot be empty",
                component="ContentValidator",
                operation="initialization",
                suggestions=["Provide at least one prohibited word, phrase, or pattern"],
            )

        # Initialize the base class
        super().__init__(name=name)

        self.prohibited = prohibited
        self.case_sensitive = case_sensitive
        self.whole_word = whole_word
        self.regex = regex

        # Compile patterns for efficiency
        self._compiled_patterns: List[Pattern[str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile prohibited items into regex patterns for efficient matching."""
        self._compiled_patterns = []

        for item in self.prohibited:
            try:
                if self.regex:
                    # Item is already a regex pattern
                    pattern = item
                else:
                    # Escape special regex characters
                    escaped_item = re.escape(item)

                    if self.whole_word:
                        # Add word boundaries
                        pattern = rf"\b{escaped_item}\b"
                    else:
                        pattern = escaped_item

                # Compile with appropriate flags
                flags = 0 if self.case_sensitive else re.IGNORECASE
                compiled_pattern = re.compile(pattern, flags)
                self._compiled_patterns.append(compiled_pattern)

            except re.error as e:
                logger.warning(f"Invalid regex pattern '{item}': {e}")
                # Skip invalid patterns but continue with others

    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Validate text against prohibited content.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        start_time = time.time()

        # Find prohibited content matches
        matches = []
        for i, pattern in enumerate(self._compiled_patterns):
            for match in pattern.finditer(thought.text):
                matches.append(
                    {
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "prohibited_item": self.prohibited[i],
                        "pattern_index": i,
                    }
                )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        if not matches:
            # No prohibited content found
            logger.debug(f"{self.name}: No prohibited content found in {processing_time:.2f}ms")
            return self.create_validation_result(
                passed=True,
                message="Text does not contain prohibited content",
                score=1.0,
                metadata={
                    "validator": self.name,
                    "processing_time_ms": processing_time,
                    "patterns_checked": len(self._compiled_patterns),
                },
            )

        # Found prohibited content
        match_items = [str(m["match"]) for m in matches]
        prohibited_items = [str(m["prohibited_item"]) for m in matches]

        # Create issues and suggestions
        unique_matches = set(match_items)
        unique_prohibited = set(prohibited_items)
        issues = [f"Text contains prohibited content: {', '.join(unique_matches)}"]
        suggestions = [
            f"Remove or rephrase the following prohibited content: {', '.join(unique_matches)}",
            f"Avoid using terms like: {', '.join(unique_prohibited)}",
        ]

        # Calculate score based on number of matches
        score = max(0.0, 1.0 - (len(matches) / len(self.prohibited)))

        logger.debug(
            f"{self.name}: Validation failed with {len(matches)} matches in {processing_time:.2f}ms"
        )

        return self.create_validation_result(
            passed=False,
            message=f"Text contains {len(matches)} prohibited content match(es)",
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "validator": self.name,
                "processing_time_ms": processing_time,
                "matches_found": len(matches),
                "unique_matches": list(unique_matches),
                "patterns_checked": len(self._compiled_patterns),
            },
        )

    async def _validate_async(self, thought: Thought) -> ValidationResult:
        """Validate text against prohibited content asynchronously.

        This is the internal async implementation that provides the same functionality
        as the sync validate method but can be called concurrently with other validators.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        # Content validation is CPU-bound and fast, so we can just call the sync version
        # In a real implementation, you might want to run this in a thread pool for consistency
        return self.validate(thought)


def create_content_validator(
    prohibited: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
    name: str = "ContentValidator",
) -> ContentValidator:
    """Create a content validator.

    This factory function creates a ContentValidator with the specified parameters.

    Args:
        prohibited: List of prohibited words, phrases, or patterns.
        case_sensitive: Whether the matching should be case-sensitive.
        whole_word: Whether to match whole words only.
        regex: Whether the prohibited items are regular expressions.
        name: The name of the validator.

    Returns:
        A ContentValidator instance.

    Raises:
        ValidationError: If the prohibited list is empty or invalid.
    """
    return ContentValidator(
        prohibited=prohibited,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        regex=regex,
        name=name,
    )


def prohibited_content(
    prohibited: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
) -> ContentValidator:
    """Create a content validator for prohibited content.

    This is a convenience function for creating a ContentValidator.

    Args:
        prohibited: List of prohibited words, phrases, or patterns.
        case_sensitive: Whether the matching should be case-sensitive.
        whole_word: Whether to match whole words only.
        regex: Whether the prohibited items are regular expressions.

    Returns:
        A ContentValidator instance.
    """
    return create_content_validator(
        prohibited=prohibited,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        regex=regex,
        name="ProhibitedContentValidator",
    )
