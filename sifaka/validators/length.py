"""Length validator for Sifaka.

This module provides validators for checking text length against specified criteria.
Supports both character and word-based length validation with configurable thresholds.
"""

from typing import Optional

from sifaka.core.thought import SifakaThought
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger
from sifaka.validators.base import BaseValidator, ValidationResult, TextLengthMixin, TimingMixin

logger = get_logger(__name__)


class LengthValidator(BaseValidator, TextLengthMixin, TimingMixin):
    """Validator that checks if text meets length requirements.

    This validator checks if text meets minimum and maximum length requirements
    in terms of character count or word count. It provides detailed feedback
    about length violations and suggestions for improvement.

    Attributes:
        min_length: Minimum required length (None for no minimum)
        max_length: Maximum allowed length (None for no maximum)
        unit: Unit of measurement ("characters" or "words")
        strict: Whether to fail validation on any violation (vs. scoring)
    """

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        unit: str = "characters",
        strict: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the length validator.

        Args:
            min_length: Minimum required length (None for no minimum)
            max_length: Maximum allowed length (None for no maximum)
            unit: Unit of measurement ("characters" or "words")
            strict: Whether to fail validation on any violation
            name: Custom name for the validator
            description: Custom description for the validator

        Raises:
            ValidationError: If configuration is invalid
        """
        # Validate configuration
        if min_length is not None and min_length < 0:
            raise ValidationError(
                "Minimum length cannot be negative",
                error_code="invalid_config",
                context={"min_length": min_length},
                suggestions=[
                    "Use a non-negative minimum length",
                    "Set min_length to None for no minimum",
                ],
            )

        if max_length is not None and max_length < 0:
            raise ValidationError(
                "Maximum length cannot be negative",
                error_code="invalid_config",
                context={"max_length": max_length},
                suggestions=[
                    "Use a non-negative maximum length",
                    "Set max_length to None for no maximum",
                ],
            )

        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValidationError(
                "Minimum length cannot be greater than maximum length",
                error_code="invalid_config",
                context={"min_length": min_length, "max_length": max_length},
                suggestions=["Ensure min_length <= max_length", "Check your length configuration"],
            )

        if unit not in ["characters", "words"]:
            raise ValidationError(
                f"Invalid unit: {unit}",
                error_code="invalid_config",
                context={"unit": unit},
                suggestions=["Use 'characters' or 'words' as the unit"],
            )

        if min_length is None and max_length is None:
            raise ValidationError(
                "At least one of min_length or max_length must be specified",
                error_code="invalid_config",
                suggestions=["Specify min_length, max_length, or both"],
            )

        # Set default name and description
        if name is None:
            name = f"length_{unit}"

        if description is None:
            parts = []
            if min_length is not None:
                parts.append(f"min {min_length}")
            if max_length is not None:
                parts.append(f"max {max_length}")
            description = f"Validates text length ({' and '.join(parts)} {unit})"

        super().__init__(name=name, description=description)

        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit
        self.strict = strict

        logger.debug(
            f"Created LengthValidator",
            extra={
                "validator_name": self.name,
                "min_length": self.min_length,
                "max_length": self.max_length,
                "unit": self.unit,
                "strict": self.strict,
            },
        )

    async def validate_async(self, thought: SifakaThought) -> ValidationResult:
        """Validate text length asynchronously.

        Args:
            thought: The SifakaThought to validate

        Returns:
            ValidationResult with length validation information
        """
        # Check if we have text to validate
        text = thought.current_text
        if not text:
            logger.debug(
                f"Length validation failed: no text",
                extra={"validator": self.name, "thought_id": thought.id},
            )
            return self.create_empty_text_result()

        with self.time_operation("length_validation") as timer:
            # Get text length in specified units
            try:
                length = self.get_text_length(text, self.unit)
            except ValueError as e:
                raise ValidationError(
                    f"Failed to measure text length: {str(e)}",
                    error_code="length_measurement_error",
                    context={"unit": self.unit, "text_preview": text[:100]},
                    suggestions=["Check unit configuration", "Verify text is valid"],
                ) from e

            # Check length constraints
            issues = []
            suggestions = []
            violations = 0

            # Check minimum length
            if self.min_length is not None and length < self.min_length:
                violations += 1
                deficit = self.min_length - length
                issues.append(f"Text too short: {length} {self.unit} (minimum: {self.min_length})")

                # Enhanced suggestions based on deficit size
                if deficit <= 10:
                    suggestions.append(
                        f"Add just {deficit} more {self.unit} - you're almost there!"
                    )
                elif deficit <= 50:
                    suggestions.append(
                        f"Add {deficit} more {self.unit} by expanding with examples or details"
                    )
                else:
                    suggestions.append(
                        f"Add {deficit} more {self.unit} - consider adding more sections or topics"
                    )

                # Content-specific suggestions
                if self.unit == "words":
                    suggestions.extend(
                        [
                            "Add descriptive adjectives and adverbs",
                            "Include specific examples or case studies",
                            "Expand on key points with more explanation",
                        ]
                    )
                else:  # characters
                    suggestions.extend(
                        [
                            "Add more detailed explanations",
                            "Include relevant examples",
                            "Expand abbreviations and add context",
                        ]
                    )

            # Check maximum length
            if self.max_length is not None and length > self.max_length:
                violations += 1
                excess = length - self.max_length
                issues.append(f"Text too long: {length} {self.unit} (maximum: {self.max_length})")

                # Enhanced suggestions based on excess amount
                if excess <= 10:
                    suggestions.append(f"Remove just {excess} {self.unit} - you're very close!")
                elif excess <= 50:
                    suggestions.append(
                        f"Trim {excess} {self.unit} by removing unnecessary words or phrases"
                    )
                else:
                    suggestions.append(
                        f"Reduce by {excess} {self.unit} - consider removing entire sentences or sections"
                    )

                # Content-specific suggestions
                if self.unit == "words":
                    suggestions.extend(
                        [
                            "Remove redundant or repetitive phrases",
                            "Use more concise language",
                            "Combine related sentences",
                            "Remove filler words like 'very', 'really', 'quite'",
                        ]
                    )
                else:  # characters
                    suggestions.extend(
                        [
                            "Use shorter synonyms where possible",
                            "Remove unnecessary punctuation or spacing",
                            "Abbreviate where appropriate",
                            "Combine sentences to reduce conjunctions",
                        ]
                    )

            # Determine if validation passed
            passed = violations == 0

            # Calculate score
            if passed:
                score = 1.0
            elif self.strict:
                score = 0.0
            else:
                # Calculate proportional score based on how close we are to requirements
                score = self._calculate_proportional_score(length)

            # Create result message
            if passed:
                message = self.format_length_message(
                    length, self.unit, self.min_length, self.max_length
                )
                message = f"Length validation passed: {message}"
            else:
                message = f"Length validation failed: {violations} violation(s)"

            # Get processing time from timer context
            processing_time = getattr(timer, "duration_ms", 0.0)

            result = self.create_validation_result(
                passed=passed,
                message=message,
                score=score,
                issues=issues,
                suggestions=suggestions,
                metadata={
                    "length": length,
                    "unit": self.unit,
                    "min_length": self.min_length,
                    "max_length": self.max_length,
                    "violations": violations,
                    "strict_mode": self.strict,
                },
                processing_time_ms=processing_time,
            )

            logger.debug(
                f"Length validation completed",
                extra={
                    "validator": self.name,
                    "thought_id": thought.id,
                    "passed": passed,
                    "length": length,
                    "unit": self.unit,
                    "violations": violations,
                    "score": score,
                },
            )

            return result

    def _calculate_proportional_score(self, length: int) -> float:
        """Calculate a proportional score when not in strict mode.

        Args:
            length: Current text length

        Returns:
            Score between 0.0 and 1.0 based on how close length is to requirements
        """
        if self.min_length is not None and self.max_length is not None:
            # Both min and max specified
            if length < self.min_length:
                # Too short - score based on how close to minimum
                ratio = length / self.min_length
                return max(0.1, ratio * 0.8)  # Score between 0.1 and 0.8
            elif length > self.max_length:
                # Too long - score based on how much over maximum
                excess_ratio = (length - self.max_length) / self.max_length
                return max(0.1, 0.8 - (excess_ratio * 0.7))  # Decreasing score
            else:
                return 1.0  # Within range

        elif self.min_length is not None:
            # Only minimum specified
            if length < self.min_length:
                ratio = length / self.min_length
                return max(0.1, ratio * 0.8)
            else:
                return 1.0

        elif self.max_length is not None:
            # Only maximum specified
            if length > self.max_length:
                excess_ratio = (length - self.max_length) / self.max_length
                return max(0.1, 0.8 - (excess_ratio * 0.7))
            else:
                return 1.0

        return 1.0  # Should not reach here due to constructor validation


def create_length_validator(
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    unit: str = "characters",
    strict: bool = True,
    name: Optional[str] = None,
) -> LengthValidator:
    """Create a length validator with the specified parameters.

    Args:
        min_length: Minimum required length
        max_length: Maximum allowed length
        unit: Unit of measurement ("characters" or "words")
        strict: Whether to fail validation on any violation
        name: Custom name for the validator

    Returns:
        Configured LengthValidator instance
    """
    return LengthValidator(
        min_length=min_length,
        max_length=max_length,
        unit=unit,
        strict=strict,
        name=name,
    )


def min_length_validator(
    min_length: int, unit: str = "characters", strict: bool = True
) -> LengthValidator:
    """Create a validator that checks minimum length only.

    Args:
        min_length: Minimum required length
        unit: Unit of measurement ("characters" or "words")
        strict: Whether to fail validation on violation

    Returns:
        LengthValidator configured for minimum length checking

    Example:
        ```python
        from sifaka.validators import min_length_validator
        from sifaka.advanced import SifakaEngine, SifakaConfig

        # Ensure content is at least 200 characters
        validator = min_length_validator(200)

        config = SifakaConfig(
            model="openai:gpt-4",
            validators=[validator]
        )

        engine = SifakaEngine(config=config)
        result = await engine.think("Write about AI")
        ```
    """
    return create_length_validator(
        min_length=min_length,
        unit=unit,
        strict=strict,
        name=f"min_{min_length}_{unit}",
    )


def max_length_validator(
    max_length: int, unit: str = "characters", strict: bool = True
) -> LengthValidator:
    """Create a validator that checks maximum length only.

    Args:
        max_length: Maximum allowed length
        unit: Unit of measurement ("characters" or "words")
        strict: Whether to fail validation on violation

    Returns:
        LengthValidator configured for maximum length checking

    Example:
        ```python
        from sifaka.validators import max_length_validator
        import sifaka

        # Ensure content is no more than 500 characters
        result = await sifaka.improve(
            "Write a brief summary of AI",
            max_length=500  # Built-in parameter
        )

        # Or use validator directly for advanced configuration
        from sifaka.advanced import SifakaEngine, SifakaConfig

        validator = max_length_validator(500, unit="characters")
        config = SifakaConfig(validators=[validator])
        engine = SifakaEngine(config=config)
        ```
    """
    return create_length_validator(
        max_length=max_length,
        unit=unit,
        strict=strict,
        name=f"max_{max_length}_{unit}",
    )
