"""Base validator implementations for Sifaka.

This module provides base validator implementations that can be used to validate
text against specific criteria. Validators check if text meets certain requirements
and return a validation result with information about whether the validation passed,
any issues found, and suggestions for improvement.

Validators are used in the Sifaka chain to ensure that generated text meets
specified criteria before it is returned to the user or passed to the next stage
of the chain.

The validators support both sync and async implementations internally, with sync
methods wrapping async implementations using asyncio.run() for backward compatibility.
"""

from typing import List, Optional

from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.logging import get_logger
from sifaka.validators.shared import LengthValidatorBase, RegexValidatorBase

# Configure logger
logger = get_logger(__name__)


class LengthValidator(LengthValidatorBase):
    """Validator that checks if text meets length requirements.

    This validator checks if text meets minimum and maximum length requirements
    in terms of character count. It extends the shared LengthValidatorBase to
    reduce code duplication.

    Attributes:
        min_length: Minimum required length in characters.
        max_length: Maximum allowed length in characters.
    """

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 10000,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        unit: str = "characters",
        name: Optional[str] = None,
    ):
        """Initialize the validator.

        Args:
            min_length: Minimum required length in characters.
            max_length: Maximum allowed length in characters.
            min_words: Minimum required length in words (overrides min_length if set).
            max_words: Maximum allowed length in words (overrides max_length if set).
            unit: Unit of measurement ("characters" or "words").
        """
        # If word-based parameters are provided, use them
        if min_words is not None or max_words is not None:
            unit = "words"
            min_length = min_words if min_words is not None else 0
            max_length = max_words if max_words is not None else 10000

        super().__init__(
            min_length=min_length if min_length > 0 else None,
            max_length=max_length if max_length < 10000 else None,
            unit=unit,
            name=name or "length",
        )

    async def _validate_async(self, thought: Thought) -> ValidationResult:
        """Validate text against length requirements asynchronously.

        This is the internal async implementation that provides the same functionality
        as the sync validate method but can be called concurrently with other validators.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        # Length validation is CPU-bound and fast, so we can just call the sync version
        # In a real implementation, you might want to run this in a thread pool for consistency
        return self.validate(thought)


class RegexValidator(RegexValidatorBase):
    """Validator that checks if text matches or doesn't match regex patterns.

    This validator checks if text matches required patterns and doesn't match
    forbidden patterns. It extends the shared RegexValidatorBase to reduce
    code duplication.

    Attributes:
        required_patterns: List of regex patterns that text must match.
        forbidden_patterns: List of regex patterns that text must not match.
    """

    def __init__(
        self,
        required_patterns: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
        prohibited_patterns: Optional[List[str]] = None,
        case_sensitive: bool = False,
        name: Optional[str] = None,
    ):
        """Initialize the validator.

        Args:
            required_patterns: List of regex patterns that text must match.
            forbidden_patterns: List of regex patterns that text must not match.
            prohibited_patterns: Alias for forbidden_patterns.
            case_sensitive: Whether pattern matching should be case-sensitive.
        """
        # Handle prohibited_patterns as alias for forbidden_patterns
        if prohibited_patterns is not None:
            if forbidden_patterns is not None:
                forbidden_patterns.extend(prohibited_patterns)
            else:
                forbidden_patterns = prohibited_patterns
        # Convert to the format expected by the base class
        patterns = {}

        if required_patterns:
            for i, pattern in enumerate(required_patterns):
                patterns[f"required_{i}"] = pattern

        if forbidden_patterns:
            for i, pattern in enumerate(forbidden_patterns):
                patterns[f"forbidden_{i}"] = pattern

        # Determine mode based on what patterns we have
        if required_patterns and forbidden_patterns:
            # Need to check both - use require_all for required patterns
            # We'll override _validate_content to handle forbidden patterns
            mode = "require_all"
        elif required_patterns:
            mode = "require_all"
        elif forbidden_patterns:
            mode = "forbid_all"
        else:
            mode = "require_all"  # Default, though no patterns means always pass

        super().__init__(patterns=patterns, mode=mode, name=name or "regex")

        # Store original patterns for backward compatibility
        self.required_patterns = required_patterns or []
        self.forbidden_patterns = forbidden_patterns or []
        self.case_sensitive = case_sensitive

    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Override to handle both required and forbidden patterns.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with the regex validation outcome.
        """
        text = thought.text
        if text is None:
            return self.create_empty_text_result(self.name)

        issues = []
        suggestions = []

        # Check required patterns
        for pattern_str in self.required_patterns:
            import re

            flags = 0 if self.case_sensitive else re.IGNORECASE
            pattern = re.compile(pattern_str, flags)
            if not pattern.search(text):
                issues.append(f"Text does not match required pattern: {pattern_str}")
                suggestions.append(f"Modify the text to include content matching: {pattern_str}")

        # Check forbidden patterns
        for pattern_str in self.forbidden_patterns:
            import re

            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(pattern_str, flags)
                if pattern.search(text):
                    issues.append(f"Text matches forbidden pattern: {pattern_str}")
                    suggestions.append(f"Remove content matching: {pattern_str}")
            except re.error as e:
                issues.append(f"Invalid forbidden pattern '{pattern_str}': {e}")
                suggestions.append(f"Fix the regex pattern: {pattern_str}")

        # Determine if validation passed
        passed = len(issues) == 0

        if passed:
            message = "Text matches all required patterns and no forbidden patterns"
        else:
            message = "Text does not meet pattern requirements"

        return self.create_validation_result(
            passed=passed,
            message=message,
            score=1.0 if passed else 0.0,
            issues=issues,
            suggestions=suggestions,
            validator_name=self.name,
        )

    async def _validate_async(self, thought: Thought) -> ValidationResult:
        """Validate text against regex patterns asynchronously.

        This is the internal async implementation that provides the same functionality
        as the sync validate method but can be called concurrently with other validators.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        # Regex validation is CPU-bound and fast, so we can just call the sync version
        # In a real implementation, you might want to run this in a thread pool for consistency
        return self.validate(thought)
