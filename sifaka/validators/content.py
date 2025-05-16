"""
Content validator for Sifaka.

This module provides a validator that checks if text contains prohibited content.
"""

import re
from typing import List, Optional, Dict, Any, Union, Pattern

from sifaka.results import ValidationResult
from sifaka.errors import ValidationError
from sifaka.registry import register_validator


class ContentValidator:
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
    ):
        """Initialize the content validator.

        Args:
            prohibited: List of prohibited words, phrases, or patterns.
            case_sensitive: Whether the matching should be case-sensitive.
            whole_word: Whether to match whole words only.
            regex: Whether the prohibited items are regular expressions.

        Raises:
            ValidationError: If the prohibited list is empty or contains invalid items.
        """
        if not prohibited:
            raise ValidationError("Prohibited list cannot be empty")

        self.prohibited = prohibited
        self.case_sensitive = case_sensitive
        self.whole_word = whole_word
        self.regex = regex

        # Compile patterns for efficient matching
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[Pattern]:
        """Compile the prohibited items into regular expression patterns.

        Returns:
            A list of compiled regular expression patterns.

        Raises:
            ValidationError: If a prohibited item is an invalid regular expression.
        """
        patterns = []

        for item in self.prohibited:
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

            try:
                pattern = re.compile(pattern_str, flags)
                patterns.append(pattern)
            except re.error as e:
                raise ValidationError(f"Invalid regular expression '{item}': {str(e)}")

        return patterns

    def validate(self, text: str) -> ValidationResult:
        """Validate text against prohibited content.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text contains prohibited content.
        """
        if not text:
            return ValidationResult(
                passed=True,
                message="Empty text contains no prohibited content",
                details={"matches": []},
            )

        # Check for matches
        matches = []

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

        if matches:
            # Found prohibited content
            return ValidationResult(
                passed=False,
                message=f"Text contains prohibited content: {', '.join(m['match'] for m in matches)}",
                details={
                    "matches": matches,
                    "match_count": len(matches),
                },
            )

        # No prohibited content found
        return ValidationResult(
            passed=True,
            message="Text contains no prohibited content",
            details={
                "matches": [],
                "match_count": 0,
            },
        )


@register_validator("content")
def create_content_validator(
    prohibited: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
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
        **options: Additional options (ignored).

    Returns:
        A ContentValidator instance.

    Raises:
        ValidationError: If the prohibited list is empty or contains invalid items.
    """
    return ContentValidator(
        prohibited=prohibited,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        regex=regex,
    )


def prohibited_content(
    prohibited: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
) -> ContentValidator:
    """Create a content validator.

    This is a convenience function for creating a ContentValidator.

    Args:
        prohibited: List of prohibited words, phrases, or patterns.
        case_sensitive: Whether the matching should be case-sensitive.
        whole_word: Whether to match whole words only.
        regex: Whether the prohibited items are regular expressions.

    Returns:
        A ContentValidator instance.

    Raises:
        ValidationError: If the prohibited list is empty or contains invalid items.
    """
    return ContentValidator(
        prohibited=prohibited,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        regex=regex,
    )
