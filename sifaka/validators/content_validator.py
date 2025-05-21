"""
Content validator for Sifaka.

This module provides a validator that checks if text contains prohibited content.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Pattern

from sifaka.validators.base_validator import BaseValidator

# Configure logger
logger = logging.getLogger(__name__)


class ContentValidator(BaseValidator):
    """
    Validator that checks if text contains prohibited content.

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
        **options: Any,
    ):
        """
        Initialize the content validator.

        Args:
            prohibited: List of prohibited words, phrases, or patterns.
            case_sensitive: Whether the matching should be case-sensitive.
            whole_word: Whether to match whole words only.
            regex: Whether the prohibited items are regular expressions.
            name: Optional name for the validator.
            **options: Additional options for the validator.

        Raises:
            ValueError: If the prohibited list is empty or contains invalid items.
        """
        super().__init__(name=name or "ContentValidator", **options)

        # Validate parameters
        if not prohibited:
            logger.error(f"{self.name}: Prohibited list cannot be empty")
            raise ValueError("Prohibited list cannot be empty")

        # Store configuration
        self.prohibited = prohibited
        self.case_sensitive = case_sensitive
        self.whole_word = whole_word
        self.regex = regex

        # Compile patterns
        self._patterns: List[Pattern] = []
        self._compile_patterns()

        # Log initialization
        logger.debug(
            f"Initialized {self.name} with {len(prohibited)} prohibited items, "
            f"case_sensitive={case_sensitive}, whole_word={whole_word}, regex={regex}"
        )

    def _compile_patterns(self) -> None:
        """
        Compile the prohibited items into regular expression patterns.

        Raises:
            ValueError: If a prohibited item is an invalid regular expression.
        """
        for item in self.prohibited:
            try:
                if self.regex:
                    # Use the item as a regular expression
                    pattern_str = item
                else:
                    # Escape special characters if not using regex mode
                    pattern_str = re.escape(item)

                # Add word boundaries if whole_word is True
                if self.whole_word and not self.regex:
                    pattern_str = r"\b" + pattern_str + r"\b"

                # Compile the pattern with the appropriate flags
                flags = 0 if self.case_sensitive else re.IGNORECASE
                pattern = re.compile(pattern_str, flags)

                self._patterns.append(pattern)

            except re.error as e:
                logger.error(f"{self.name}: Invalid pattern '{item}': {str(e)}")
                raise ValueError(f"Invalid pattern '{item}': {str(e)}")

    def _validate(self, text: str) -> Dict[str, Any]:
        """
        Validate text against prohibited content.

        Args:
            text: The text to validate.

        Returns:
            A dictionary with validation results.
        """
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

        logger.debug(f"{self.name}: Found {len(matches)} matches in text of length {len(text)}")

        if matches:
            # Found prohibited content
            match_items = [m["match"] for m in matches]
            prohibited_items = [m["prohibited_item"] for m in matches]

            # Create issues and suggestions
            issues = [f"Text contains prohibited content: {', '.join(match_items)}"]
            suggestions = [
                f"Remove or rephrase the following prohibited content: {', '.join(match_items)}",
                f"Avoid using terms like: {', '.join(prohibited_items)}",
            ]

            # Calculate score based on number of matches
            score = max(0.0, 1.0 - (len(matches) / len(self.prohibited)))

            return {
                "passed": False,
                "message": f"Text contains prohibited content: {', '.join(match_items)}",
                "details": {
                    "matches": matches,
                    "match_count": len(matches),
                    "validator_name": self.name,
                },
                "score": score,
                "issues": issues,
                "suggestions": suggestions,
            }

        # No prohibited content found
        return {
            "passed": True,
            "message": "Text contains no prohibited content",
            "details": {
                "matches": [],
                "match_count": 0,
                "validator_name": self.name,
            },
            "score": 1.0,
            "issues": [],
            "suggestions": [],
        }


def create_content_validator(
    prohibited: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
    name: Optional[str] = None,
    **options: Any,
) -> ContentValidator:
    """
    Create a content validator.

    This is a convenience function for creating a ContentValidator.

    Args:
        prohibited: List of prohibited words, phrases, or patterns.
        case_sensitive: Whether the matching should be case-sensitive.
        whole_word: Whether to match whole words only.
        regex: Whether the prohibited items are regular expressions.
        name: Optional name for the validator.
        **options: Additional options for the validator.

    Returns:
        A ContentValidator instance.

    Raises:
        ValueError: If the prohibited list is empty or contains invalid items.
    """
    return ContentValidator(
        prohibited=prohibited,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        regex=regex,
        name=name,
        **options,
    )


def prohibited_content(
    prohibited: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
    name: Optional[str] = None,
) -> ContentValidator:
    """
    Create a content validator.

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
        ValueError: If the prohibited list is empty or contains invalid items.
    """
    return create_content_validator(
        prohibited=prohibited,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        regex=regex,
        name=name,
    )
