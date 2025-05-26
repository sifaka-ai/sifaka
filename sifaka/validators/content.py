"""Content validator for Sifaka.

This module provides a ContentValidator that checks if text contains prohibited
content, words, phrases, or patterns. It supports case-sensitive/insensitive
matching, whole word matching, and regular expression patterns.

The ContentValidator is designed to prevent generation of harmful, inappropriate,
or unwanted content by checking against a list of prohibited items.
"""

import re
import time
from typing import List, Optional, Pattern

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
        prohibited: Optional[List[str]] = None,
        required: Optional[List[str]] = None,
        case_sensitive: bool = False,
        whole_word: bool = False,
        regex: bool = False,
        name: str = "ContentValidator",
    ):
        """Initialize the validator.

        Args:
            prohibited: List of prohibited words, phrases, or patterns.
            required: List of required words, phrases, or patterns.
            case_sensitive: Whether the matching should be case-sensitive.
            whole_word: Whether to match whole words only.
            regex: Whether the prohibited/required items are regular expressions.
            name: The name of the validator.

        Raises:
            ValidationError: If both prohibited and required lists are empty or invalid.
        """
        if not prohibited and not required:
            raise ValidationError(
                message="Either prohibited or required list must be provided",
                component="ContentValidator",
                operation="initialization",
                suggestions=[
                    "Provide at least one prohibited or required word, phrase, or pattern"
                ],
            )

        # Initialize the base class
        super().__init__(name=name)

        self.prohibited = prohibited or []
        self.required = required or []
        self.case_sensitive = case_sensitive
        self.whole_word = whole_word
        self.regex = regex

        # Compile patterns for efficiency
        self._compiled_prohibited_patterns: List[Pattern[str]] = []
        self._compiled_required_patterns: List[Pattern[str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile prohibited and required items into regex patterns for efficient matching."""
        self._compiled_prohibited_patterns = []
        self._compiled_required_patterns = []

        # Compile prohibited patterns
        for item in self.prohibited:
            try:
                pattern = self._create_pattern(item)
                flags = 0 if self.case_sensitive else re.IGNORECASE
                compiled_pattern = re.compile(pattern, flags)
                self._compiled_prohibited_patterns.append(compiled_pattern)
            except re.error as e:
                logger.warning(f"Invalid prohibited regex pattern '{item}': {e}")

        # Compile required patterns
        for item in self.required:
            try:
                pattern = self._create_pattern(item)
                flags = 0 if self.case_sensitive else re.IGNORECASE
                compiled_pattern = re.compile(pattern, flags)
                self._compiled_required_patterns.append(compiled_pattern)
            except re.error as e:
                logger.warning(f"Invalid required regex pattern '{item}': {e}")

    def _create_pattern(self, item: str) -> str:
        """Create a regex pattern from an item."""
        if self.regex:
            # Item is already a regex pattern
            return item
        else:
            # Escape special regex characters
            escaped_item = re.escape(item)
            if self.whole_word:
                # Add word boundaries
                return rf"\b{escaped_item}\b"
            else:
                return escaped_item

    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Validate text against prohibited and required content.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        # Check for None text
        if thought.text is None:
            return self.create_empty_text_result(self.name)

        start_time = time.time()
        issues = []
        suggestions = []

        # Check prohibited content
        prohibited_matches = []
        for i, pattern in enumerate(self._compiled_prohibited_patterns):
            for match in pattern.finditer(thought.text):
                prohibited_matches.append(
                    {
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "prohibited_item": self.prohibited[i],
                        "pattern_index": i,
                    }
                )

        # Check required content
        missing_required = []
        for i, pattern in enumerate(self._compiled_required_patterns):
            if not pattern.search(thought.text):
                missing_required.append(self.required[i])

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Determine if validation passed
        passed = len(prohibited_matches) == 0 and len(missing_required) == 0

        if passed:
            # All validations passed
            logger.debug(f"{self.name}: Content validation passed in {processing_time:.2f}ms")
            return self.create_validation_result(
                passed=True,
                message="Text meets all content requirements",
                score=1.0,
            )

        # Handle prohibited content violations
        if prohibited_matches:
            match_items = [str(m["match"]) for m in prohibited_matches]
            prohibited_items = [str(m["prohibited_item"]) for m in prohibited_matches]
            unique_matches = set(match_items)
            unique_prohibited = set(prohibited_items)
            issues.append(f"Text contains prohibited content: {', '.join(unique_matches)}")
            suggestions.extend(
                [
                    f"Remove or rephrase the following prohibited content: {', '.join(unique_matches)}",
                    f"Avoid using terms like: {', '.join(unique_prohibited)}",
                ]
            )

        # Handle missing required content
        if missing_required:
            issues.append(f"Text is missing required content: {', '.join(missing_required)}")
            suggestions.append(
                f"Include the following required content: {', '.join(missing_required)}"
            )

        # Calculate score based on violations
        total_violations = len(prohibited_matches) + len(missing_required)
        total_checks = len(self.prohibited) + len(self.required)
        score = max(0.0, 1.0 - (total_violations / max(1, total_checks)))

        logger.debug(
            f"{self.name}: Validation failed with {total_violations} violations in {processing_time:.2f}ms"
        )

        return self.create_validation_result(
            passed=False,
            message=f"Text has {total_violations} content violation(s)",
            score=score,
            issues=issues,
            suggestions=suggestions,
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
