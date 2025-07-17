"""Base implementation for all Sifaka validators.

This module provides the abstract base class that all validators inherit from,
along with common utilities for score calculation and result formatting.

## Key Components:

- **BaseValidator**: Abstract class with common validation logic
- **ValidatorConfig**: Configuration for validator behavior
- **Utility Methods**: Score calculation and formatting helpers

## Creating Custom Validators:

    >>> from sifaka.validators.base import BaseValidator
    >>>
    >>> class MyValidator(BaseValidator):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_validator"
    ...
    ...     async def _perform_validation(self, text, result):
    ...         # Validation logic here
    ...         passed = len(text) > 100
    ...         score = min(1.0, len(text) / 100)
    ...         details = f"Text length: {len(text)}"
    ...         return passed, score, details

## Validator Design Principles:

1. **Deterministic**: Same input always produces same result
2. **Fast**: Validators run on every iteration
3. **Clear Feedback**: Provide actionable details
4. **Configurable**: Support different strictness levels
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

from pydantic import BaseModel, Field

from ..core.interfaces import Validator
from ..core.models import SifakaResult, ValidationResult


class ValidatorConfig(BaseModel):
    """Configuration options for validator behavior.

    This configuration allows fine-tuning how validators evaluate text
    and report results. It provides control over scoring thresholds,
    strictness levels, and feedback detail.

    Example:
        >>> # Strict validation that requires perfect scores
        >>> config = ValidatorConfig(
        ...     strict_mode=True,
        ...     pass_threshold=1.0,
        ...     detailed_feedback=True
        ... )
        >>> validator = MyValidator(config=config)

        >>> # Lenient validation with minimal feedback
        >>> config = ValidatorConfig(
        ...     pass_threshold=0.5,
        ...     detailed_feedback=False
        ... )

    Attributes:
        min_score: Minimum possible score (floor)
        max_score: Maximum possible score (ceiling)
        pass_threshold: Score required to pass validation
        strict_mode: If True, any imperfection fails validation
        detailed_feedback: If True, provide detailed explanations
    """

    # Score calculation
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_score: float = Field(default=1.0, ge=0.0, le=1.0)
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Behavior flags
    strict_mode: bool = Field(default=False, description="Fail on any violation")
    detailed_feedback: bool = Field(
        default=True, description="Provide detailed feedback"
    )


class BaseValidator(Validator, ABC):
    """Abstract base class providing common validator functionality.

    BaseValidator handles the standard validation workflow, score normalization,
    and error handling, allowing subclasses to focus on their specific
    validation logic.

    The base class provides:
    - Configuration management
    - Score calculation utilities
    - Result formatting helpers
    - Error handling with graceful degradation
    - Consistent validation workflow

    Subclasses only need to implement:
    - name property: Unique validator identifier
    - _perform_validation method: Core validation logic

    Example:
        >>> class WordCountValidator(BaseValidator):
        ...     def __init__(self, min_words: int, max_words: int, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.min_words = min_words
        ...         self.max_words = max_words
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return f"word_count_{self.min_words}_{self.max_words}"
        ...
        ...     async def _perform_validation(self, text, result):
        ...         word_count = len(text.split())
        ...         passed = self.min_words <= word_count <= self.max_words
        ...         score = self._calculate_score(
        ...             word_count,
        ...             (self.min_words + self.max_words) / 2
        ...         )
        ...         details = f"Word count: {word_count}"
        ...         return passed, score, details
    """

    def __init__(self, config: Optional[ValidatorConfig] = None):
        """Initialize validator with optional configuration.

        Args:
            config: Configuration for validator behavior. If not provided,
                uses default configuration with sensible values.
        """
        self.config = config or ValidatorConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifier for this validator.

        The name is used for tracking validation results and should be
        stable across runs. Include configuration in the name if it
        affects behavior.

        Returns:
            Unique string identifier, e.g., "length_100_500"
        """

    @abstractmethod
    async def _perform_validation(
        self, text: str, result: SifakaResult
    ) -> tuple[bool, float, str]:
        """Perform the core validation logic.

        This is the main method subclasses implement to define their
        validation behavior. It should analyze the text and return
        validation results.

        Args:
            text: The current text to validate
            result: Complete result object with history for context

        Returns:
            Tuple containing:
            - passed (bool): Whether validation passed
            - score (float): Quality score from 0.0 to 1.0
            - details (str): Human-readable explanation

        Note:
            This method should be deterministic and fast. Avoid
            external API calls or heavy computation.
        """

    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate text and return standardized result.

        This method implements the complete validation workflow:
        1. Call _perform_validation to get raw results
        2. Apply configuration rules (strict mode, thresholds)
        3. Normalize scores to configured range
        4. Format details based on configuration
        5. Handle any errors gracefully

        Args:
            text: Text to validate
            result: Current result with history

        Returns:
            ValidationResult with pass/fail status, score, and details

        Note:
            This method handles all error cases, always returning a valid
            ValidationResult even if validation logic fails.
        """
        try:
            # Perform validation
            passed, score, details = await self._perform_validation(text, result)

            # Apply configuration rules
            if self.config.strict_mode and score < 1.0:
                passed = False

            # Ensure score is in valid range
            score = max(self.config.min_score, min(score, self.config.max_score))

            # Apply pass threshold
            if score < self.config.pass_threshold:
                passed = False

            # Format details based on config
            if not self.config.detailed_feedback:
                details = "Validation " + ("passed" if passed else "failed")

            return ValidationResult(
                validator=self.name, passed=passed, score=score, details=details
            )

        except Exception as e:
            # Error handling with validation result
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details=f"Validation error: {e!s}",
            )

    def _calculate_score(
        self,
        value: Union[int, float],
        target: Union[int, float],
        tolerance: float = 0.1,
    ) -> float:
        """Calculate a score based on proximity to target value.

        This utility method provides a standard way to score numeric
        values against targets. It uses linear scoring within tolerance
        and exponential decay beyond.

        Args:
            value: The actual measured value
            target: The ideal target value
            tolerance: Acceptable deviation as a fraction (0.1 = ±10%).
                Within this range, scoring is linear. Beyond it,
                score drops exponentially.

        Returns:
            Score between 0.0 and 1.0, where:
            - 1.0 = perfect match
            - 0.9 = 10% deviation (with default tolerance)
            - 0.5 = 50% deviation
            - 0.0 = 100% or more deviation

        Example:
            >>> # Score text length against target of 100 words
            >>> score = self._calculate_score(
            ...     value=len(text.split()),
            ...     target=100,
            ...     tolerance=0.2  # ±20% acceptable
            ... )
        """
        if target == 0:
            return 1.0 if value == 0 else 0.0

        deviation = abs(value - target) / target
        if deviation <= tolerance:
            # Linear scoring within tolerance
            score = 1.0 - (deviation / tolerance)
        else:
            # Exponential decay beyond tolerance
            score = max(0.0, 1.0 - deviation)

        return score

    def _format_details(
        self,
        primary: str,
        secondary: Optional[str] = None,
        suggestions: Optional[list[str]] = None,
    ) -> str:
        """Format validation feedback in a consistent structure.

        This utility helps create well-formatted validation messages
        with optional additional context and suggestions.

        Args:
            primary: Main validation message (required)
            secondary: Additional context or measurements
            suggestions: List of actionable improvement suggestions

        Returns:
            Multi-line formatted string with clear structure

        Example:
            >>> details = self._format_details(
            ...     primary="Text is too short",
            ...     secondary="Current: 50 words, Required: 100 words",
            ...     suggestions=[
            ...         "Add more detail to the introduction",
            ...         "Expand on the main points",
            ...         "Include a conclusion paragraph"
            ...     ]
            ... )
            >>> print(details)
            Text is too short
            Current: 50 words, Required: 100 words
            Suggestions:
              1. Add more detail to the introduction
              2. Expand on the main points
              3. Include a conclusion paragraph
        """
        parts = [primary]

        if secondary:
            parts.append(secondary)

        if suggestions:
            parts.append("Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                parts.append(f"  {i}. {suggestion}")

        return "\n".join(parts)
